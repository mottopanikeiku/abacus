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

static float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

// Tensors are stored as bf16, each starting on a 64-byte boundary.
static float *read_tensor(Reader *r, size_t n) {
    r->off = (r->off + 63) & ~(size_t)63;
    const uint16_t *src = (const uint16_t *)(r->base + r->off);
    r->off += n * sizeof(uint16_t);
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
    int dim = c->dim, hd = c->head_dim;
    int q_dim = c->n_heads * hd, kv_dim = c->n_kv_heads * hd;
    m->embed = read_tensor(&r, (size_t)c->vocab_size * dim);
    m->layers = xcalloc(c->n_layers, sizeof(Layer));
    for (int l = 0; l < c->n_layers; l++) {
        Layer *L = &m->layers[l];
        L->attn_norm = read_tensor(&r, dim);
        L->wq = read_tensor(&r, (size_t)q_dim * dim);
        L->wk = read_tensor(&r, (size_t)kv_dim * dim);
        L->wv = read_tensor(&r, (size_t)kv_dim * dim);
        L->wo = read_tensor(&r, (size_t)dim * q_dim);
        L->q_norm = read_tensor(&r, hd);
        L->k_norm = read_tensor(&r, hd);
        L->ffn_norm = read_tensor(&r, dim);
        L->w_gate = read_tensor(&r, (size_t)c->hidden_dim * dim);
        L->w_up = read_tensor(&r, (size_t)c->hidden_dim * dim);
        L->w_down = read_tensor(&r, (size_t)dim * c->hidden_dim);
    }
    m->norm = read_tensor(&r, dim);
    m->lm_head = tied ? m->embed : read_tensor(&r, (size_t)c->vocab_size * dim);
    if (r.off > (size_t)st.st_size) die("model file is truncated");

    m->x = xcalloc(dim, sizeof(float));
    m->xn = xcalloc(dim, sizeof(float));
    m->q = xcalloc(q_dim, sizeof(float));
    m->k = xcalloc(kv_dim, sizeof(float));
    m->v = xcalloc(kv_dim, sizeof(float));
    m->att = xcalloc(q_dim, sizeof(float));
    m->scores = xcalloc((size_t)c->n_heads * c->seq_len, sizeof(float));
    m->gate = xcalloc(c->hidden_dim, sizeof(float));
    m->up = xcalloc(c->hidden_dim, sizeof(float));
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

typedef struct {
    float *out;
    const float *x, *w;
    int n, d;
} Matmul;

static void matmul_rows(void *arg, int thread, int n_threads) {
    const Matmul *a = arg;
    int begin = (int)((long)a->d * thread / n_threads), end = (int)((long)a->d * (thread + 1) / n_threads);
    for (int i = begin; i < end; i++) a->out[i] = dot(a->w + (size_t)i * a->n, a->x, a->n);
}

// out = w[d, n] @ x[n], each thread computing a slice of the rows
static void matmul(float *out, const float *x, const float *w, int n, int d) {
    Matmul a = {out, x, w, n, d};
    pool_run(matmul_rows, &a);
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

// Query head h attends over cache positions 0..pos; writes its slice of m->att.
static void attention_head(Model *m, int layer, int h, int pos) {
    const Config *c = &m->c;
    int hd = c->head_dim, kv_dim = c->n_kv_heads * hd;
    size_t base = (size_t)layer * c->seq_len * kv_dim + (size_t)(h / (c->n_heads / c->n_kv_heads)) * hd;
    const float *q = m->q + h * hd, *k_cache = m->k_cache + base, *v_cache = m->v_cache + base;
    float *scores = m->scores + (size_t)h * c->seq_len, *out = m->att + h * hd;
    float scale = 1.0f / sqrtf((float)hd);
    for (int t = 0; t <= pos; t++) scores[t] = dot(q, k_cache + (size_t)t * kv_dim, hd) * scale;
    softmax(scores, pos + 1);
    memset(out, 0, hd * sizeof(float));
    for (int t = 0; t <= pos; t++) {
        const float *v = v_cache + (size_t)t * kv_dim;
        for (int i = 0; i < hd; i++) out[i] += scores[t] * v[i];
    }
}

typedef struct {
    Model *m;
    int layer, pos;
} Attention;

static void attention_heads(void *arg, int thread, int n_threads) {
    const Attention *a = arg;
    int heads = a->m->c.n_heads;
    for (int h = heads * thread / n_threads; h < heads * (thread + 1) / n_threads; h++)
        attention_head(a->m, a->layer, h, a->pos);
}

static void forward_token(Model *m, int token, int pos) {
    const Config *c = &m->c;
    int dim = c->dim, hd = c->head_dim;
    int q_dim = c->n_heads * hd, kv_dim = c->n_kv_heads * hd;
    float *x = m->x, *xn = m->xn;

    memcpy(x, m->embed + (size_t)token * dim, dim * sizeof(float));
    for (int l = 0; l < c->n_layers; l++) {
        const Layer *L = &m->layers[l];

        // attention: project, normalize each head, rotate, append to the cache
        rmsnorm(xn, x, L->attn_norm, dim, c->norm_eps);
        matmul(m->q, xn, L->wq, dim, q_dim);
        matmul(m->k, xn, L->wk, dim, kv_dim);
        matmul(m->v, xn, L->wv, dim, kv_dim);
        for (int h = 0; h < c->n_heads; h++) rmsnorm(m->q + h * hd, m->q + h * hd, L->q_norm, hd, c->norm_eps);
        for (int h = 0; h < c->n_kv_heads; h++) rmsnorm(m->k + h * hd, m->k + h * hd, L->k_norm, hd, c->norm_eps);
        rope(m->q, c->n_heads, hd, pos, c->rope_theta);
        rope(m->k, c->n_kv_heads, hd, pos, c->rope_theta);
        float *k_cache = m->k_cache + (size_t)l * c->seq_len * kv_dim;
        float *v_cache = m->v_cache + (size_t)l * c->seq_len * kv_dim;
        memcpy(k_cache + (size_t)pos * kv_dim, m->k, kv_dim * sizeof(float));
        memcpy(v_cache + (size_t)pos * kv_dim, m->v, kv_dim * sizeof(float));
        Attention att = {m, l, pos};
        pool_run(attention_heads, &att);
        matmul(xn, m->att, L->wo, q_dim, dim);
        for (int i = 0; i < dim; i++) x[i] += xn[i];

        // MLP: down(silu(gate(x)) * up(x))
        rmsnorm(xn, x, L->ffn_norm, dim, c->norm_eps);
        matmul(m->gate, xn, L->w_gate, dim, c->hidden_dim);
        matmul(m->up, xn, L->w_up, dim, c->hidden_dim);
        for (int i = 0; i < c->hidden_dim; i++) {
            float g = m->gate[i];
            m->gate[i] = g / (1.0f + expf(-g)) * m->up[i];
        }
        matmul(xn, m->gate, L->w_down, c->hidden_dim, dim);
        for (int i = 0; i < dim; i++) x[i] += xn[i];
    }
    rmsnorm(x, x, m->norm, dim, c->norm_eps);
}

void forward(Model *m, const int *tokens, int n, int pos, float *logits, int n_logits) {
    const Config *c = &m->c;
    if (pos + n > c->seq_len) die("context is full");
    for (int i = 0; i < n; i++) {
        forward_token(m, tokens[i], pos + i);
        int j = i - (n - n_logits);
        if (j >= 0) matmul(logits + (size_t)j * c->vocab_size, m->x, m->lm_head, c->dim, c->vocab_size);
    }
}
