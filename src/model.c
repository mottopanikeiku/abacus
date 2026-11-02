// Loading a model file and running the transformer.
#include "abacus.h"
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#define MAX_SEQ 4096
#define BATCH 128 // prompt tokens processed together

const uint8_t *tokenizer_load(Tokenizer *t, const uint8_t *p);

static void die(const char *msg) {
    fprintf(stderr, "abacus: %s\n", msg);
    exit(1);
}

static void *xcalloc(size_t n, size_t size) {
    void *p = calloc(n, size);
    if (!p) die("out of memory");
    return p;
}

// ---------------------------------------------------------------- loading

typedef struct {
    const uint8_t *base;
    size_t off;
} Reader;

// Tensors are stored as bf16, each starting on a 64-byte boundary.
static const bf16 *read_bf16(Reader *r, size_t n) {
    r->off = (r->off + 63) & ~(size_t)63;
    const bf16 *p = (const bf16 *)(r->base + r->off);
    r->off += n * sizeof(bf16);
    return p;
}

// Matrices are quantized to int8 as they are read.
static QMatrix read_matrix(Reader *r, int rows, int cols) {
    return quantize_matrix(read_bf16(r, (size_t)rows * cols), rows, cols);
}

// Small vectors (norm weights) are widened to float once.
static float *read_f32(Reader *r, size_t n) {
    const bf16 *src = read_bf16(r, n);
    float *dst = xcalloc(n, sizeof(float));
    for (size_t i = 0; i < n; i++) dst[i] = bf16_to_f32(src[i]);
    return dst;
}

Model *model_load(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) die("cannot open model file");
    struct stat st;
    fstat(fd, &st);
    const uint8_t *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) die("cannot map model file");
    close(fd);
    if (memcmp(data, "abacus\0\0", 8)) die("not an abacus model file (run tools/export.py)");

    Model *m = xcalloc(1, sizeof(Model));
    Config *c = &m->c;
    int32_t h[9];
    memcpy(h, data + 12, sizeof h);
    c->dim = h[0], c->hidden_dim = h[1], c->n_layers = h[2], c->n_heads = h[3];
    c->n_kv_heads = h[4], c->head_dim = h[5], c->vocab_size = h[6];
    c->seq_len = h[7] < MAX_SEQ ? h[7] : MAX_SEQ;
    int tied = h[8];
    memcpy(&c->rope_theta, data + 48, 4);
    memcpy(&c->norm_eps, data + 52, 4);
    memcpy(&m->tok.bos, data + 56, 4);
    memcpy(&m->tok.eos, data + 60, 4);

    Reader r = {data, tokenizer_load(&m->tok, data + 64) - data};
    int dim = c->dim, hd = c->head_dim, hidden = c->hidden_dim;
    int q_dim = c->n_heads * hd, kv_dim = c->n_kv_heads * hd;
    m->embed = read_bf16(&r, (size_t)c->vocab_size * dim);
    m->layers = xcalloc(c->n_layers, sizeof(Layer));
    for (int l = 0; l < c->n_layers; l++) {
        // q, k and v follow each other in the file with no padding (each is a
        // multiple of 64 bytes), as do gate and up, so each set loads as one
        // taller matrix: one product per set instead of three or two.
        Layer *L = &m->layers[l];
        L->attn_norm = read_f32(&r, dim);
        L->wqkv = read_matrix(&r, q_dim + 2 * kv_dim, dim);
        L->wo = read_matrix(&r, dim, q_dim);
        L->q_norm = read_f32(&r, hd);
        L->k_norm = read_f32(&r, hd);
        L->ffn_norm = read_f32(&r, dim);
        L->w_gate_up = read_matrix(&r, 2 * hidden, dim);
        L->w_down = read_matrix(&r, dim, hidden);
    }
    m->norm = read_f32(&r, dim);
    m->lm_head = tied ? quantize_matrix(m->embed, c->vocab_size, dim) : read_matrix(&r, c->vocab_size, dim);
    if (r.off > (size_t)st.st_size) die("model file is truncated");

    m->x = xcalloc((size_t)BATCH * dim, sizeof(float));
    m->xn = xcalloc((size_t)BATCH * dim, sizeof(float));
    m->qkv = xcalloc((size_t)BATCH * (q_dim + 2 * kv_dim), sizeof(float));
    m->att = xcalloc((size_t)BATCH * q_dim, sizeof(float));
    m->gate_up = xcalloc((size_t)BATCH * 2 * hidden, sizeof(float));
    m->act = xcalloc((size_t)BATCH * hidden, sizeof(float));
    m->k_cache = xcalloc((size_t)c->n_layers * c->seq_len * kv_dim, sizeof(float));
    m->v_cache = xcalloc((size_t)c->n_layers * c->seq_len * kv_dim, sizeof(float));
    return m;
}

void model_free(Model *m) {
    // The process exits right after; the OS reclaims everything.
    (void)m;
}

// ---------------------------------------------------------------- math

// Sum of a[i] * b[i]. A single running sum waits on every multiply-add; NEON
// keeps four 4-wide sums in flight instead.
static float dot(const float *a, const float *b, int n) {
    float sum = 0.0f;
    int i = 0;
#ifdef __ARM_NEON
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0, s2 = s0, s3 = s0;
    for (; i + 16 <= n; i += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
#endif
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = w[i] * (x[i] * scale);
}

// Rotary embedding on each head: rotate the pair (i, i + head_dim/2) by pos * theta^(-2i/head_dim).
static void rope(float *x, int n_heads, int head_dim, int pos, float theta) {
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
        float angle = (float)pos * freq;
        float c = cosf(angle), s = sinf(angle);
        for (int h = 0; h < n_heads; h++) {
            float *v = x + h * head_dim;
            float a = v[i], b = v[i + half];
            v[i] = a * c - b * s;
            v[i + half] = b * c + a * s;
        }
    }
}

static void softmax(float *x, int n) {
    float max = x[0];
    for (int i = 1; i < n; i++) max = x[i] > max ? x[i] : max;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// ---------------------------------------------------------------- transformer

// Query head h of the token at position pos attends over cache positions 0..pos.
static void attention_head(const Model *m, int layer, const float *q, float *out, int h, int pos) {
    const Config *c = &m->c;
    int hd = c->head_dim, kv_dim = c->n_kv_heads * hd;
    size_t base = (size_t)layer * c->seq_len * kv_dim + (size_t)(h / (c->n_heads / c->n_kv_heads)) * hd;
    const float *k_cache = m->k_cache + base, *v_cache = m->v_cache + base;
    float scores[MAX_SEQ], scale = 1.0f / sqrtf((float)hd);
    q += h * hd, out += h * hd;
    for (int t = 0; t <= pos; t++) scores[t] = dot(q, k_cache + (size_t)t * kv_dim, hd) * scale;
    softmax(scores, pos + 1);
    memset(out, 0, hd * sizeof(float));
    for (int t = 0; t <= pos; t++) {
        const float *v = v_cache + (size_t)t * kv_dim;
        for (int i = 0; i < hd; i++) out[i] += scores[t] * v[i];
    }
}

// One layer step over a batch: which layer, where the batch starts, and the
// per-token function to run.
typedef struct Step Step;
struct Step {
    Model *m;
    const Layer *L;
    int layer, pos, n;
    const float *norm; // for add_norm
    int add;           // for add_norm
    void (*row)(const Step *s, int i);
};

static void rows_task(void *arg, int thread, int n_threads) {
    const Step *s = arg;
    for (int i = s->n * thread / n_threads; i < s->n * (thread + 1) / n_threads; i++) s->row(s, i);
}

// Runs row(s, i) for every token of the batch, spread over the threads.
static void for_each_token(Step *s, void (*row)(const Step *s, int i)) {
    s->row = row;
    if (s->n == 1) row(s, 0); // not worth waking the team
    else pool_run(rows_task, s);
}

// x += the last block's output (left in xn), then xn = rmsnorm(x).
static void add_norm(const Step *s, int i) {
    int dim = s->m->c.dim;
    float *x = s->m->x + (size_t)i * dim, *xn = s->m->xn + (size_t)i * dim;
    if (s->add)
        for (int j = 0; j < dim; j++) x[j] += xn[j];
    rmsnorm(xn, x, s->norm, dim, s->m->c.norm_eps);
}

// Normalize each query and key head, rotate them, append k and v to the cache.
static void rotate_and_cache(const Step *s, int i) {
    const Config *c = &s->m->c;
    int hd = c->head_dim, q_dim = c->n_heads * hd, kv_dim = c->n_kv_heads * hd, pos = s->pos + i;
    float *q = s->m->qkv + (size_t)i * (q_dim + 2 * kv_dim), *k = q + q_dim, *v = k + kv_dim;
    for (int h = 0; h < c->n_heads; h++) rmsnorm(q + h * hd, q + h * hd, s->L->q_norm, hd, c->norm_eps);
    for (int h = 0; h < c->n_kv_heads; h++) rmsnorm(k + h * hd, k + h * hd, s->L->k_norm, hd, c->norm_eps);
    rope(q, c->n_heads, hd, pos, c->rope_theta);
    rope(k, c->n_kv_heads, hd, pos, c->rope_theta);
    size_t at = ((size_t)s->layer * c->seq_len + pos) * kv_dim;
    memcpy(s->m->k_cache + at, k, kv_dim * sizeof(float));
    memcpy(s->m->v_cache + at, v, kv_dim * sizeof(float));
}

// Every (token, head) pair in the batch is an independent job.
static void attention_jobs(void *arg, int thread, int n_threads) {
    const Step *s = arg;
    const Config *c = &s->m->c;
    int q_dim = c->n_heads * c->head_dim, qkv_dim = q_dim + 2 * c->n_kv_heads * c->head_dim;
    int jobs = s->n * c->n_heads;
    for (int j = jobs * thread / n_threads; j < jobs * (thread + 1) / n_threads; j++) {
        int i = j / c->n_heads, h = j % c->n_heads;
        attention_head(s->m, s->layer, s->m->qkv + (size_t)i * qkv_dim, s->m->att + (size_t)i * q_dim, h, s->pos + i);
    }
}

// act = silu(gate) * up, split by element so that one token is spread too.
static void swiglu(void *arg, int thread, int n_threads) {
    const Step *s = arg;
    int hidden = s->m->c.hidden_dim;
    size_t total = (size_t)s->n * hidden;
    for (size_t e = total * thread / n_threads; e < total * (thread + 1) / n_threads; e++) {
        const float *gate = s->m->gate_up + e / hidden * 2 * hidden + e % hidden, *up = gate + hidden;
        s->m->act[e] = *gate / (1.0f + expf(-*gate)) * *up;
    }
}

// Runs n <= BATCH tokens through every layer at once, so each weight is read
// once per batch instead of once per token. Leaves the final hidden states in m->xn.
static void forward_batch(Model *m, const int *tokens, int n, int pos) {
    const Config *c = &m->c;
    int dim = c->dim;
    Step s = {.m = m, .pos = pos, .n = n};
    for (int i = 0; i < n; i++)
        for (int j = 0; j < dim; j++) m->x[(size_t)i * dim + j] = bf16_to_f32(m->embed[(size_t)tokens[i] * dim + j]);
    for (int l = 0; l < c->n_layers; l++) {
        const Layer *L = s.L = &m->layers[l];
        s.layer = l;

        // attention: project, normalize each head, rotate, append to the cache
        s.norm = L->attn_norm, s.add = l > 0;
        for_each_token(&s, add_norm);
        matmul(m->qkv, m->xn, n, &L->wqkv);
        for_each_token(&s, rotate_and_cache);
        pool_run(attention_jobs, &s);
        matmul(m->xn, m->att, n, &L->wo);

        // MLP: down(silu(gate(x)) * up(x))
        s.norm = L->ffn_norm, s.add = 1;
        for_each_token(&s, add_norm);
        matmul(m->gate_up, m->xn, n, &L->w_gate_up);
        pool_run(swiglu, &s);
        matmul(m->xn, m->act, n, &L->w_down);
    }
    s.norm = m->norm, s.add = 1;
    for_each_token(&s, add_norm);
}

void forward(Model *m, const int *tokens, int n, int pos, float *logits, int n_logits) {
    const Config *c = &m->c;
    if (pos + n > c->seq_len) die("context is full");
    for (int start = 0; start < n; start += BATCH) {
        int len = n - start < BATCH ? n - start : BATCH;
        forward_batch(m, tokens + start, len, pos + start);
        // logits for the rows of this batch that are among the last n_logits
        int first = start > n - n_logits ? start : n - n_logits;
        if (first < start + len)
            matmul(logits + (size_t)(first - (n - n_logits)) * c->vocab_size, m->xn + (size_t)(first - start) * c->dim,
                   start + len - first, &m->lm_head);
    }
}
