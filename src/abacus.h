// abacus: a small CPU inference engine for Qwen3 models.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    int dim;        // residual stream width (1024 for Qwen3-0.6B)
    int hidden_dim; // MLP width (3072)
    int n_layers;   // 28
    int n_heads;    // query heads (16)
    int n_kv_heads; // key/value heads, shared by n_heads / n_kv_heads query heads (8)
    int head_dim;   // 128
    int vocab_size; // 151936
    int seq_len;    // KV cache length
    float rope_theta, norm_eps;
} Config;

typedef struct {
    uint64_t key; // left token << 32 | right token
    int rank, result;
} Merge;

typedef struct {
    int n_tokens;
    char **piece; // raw bytes of every token
    int *len;
    int byte_token[256];
    int n_added; // special tokens matched literally in text, like <|im_start|>
    int *added;
    int n_merges;
    Merge *merges; // sorted by key
    int bos, eos;  // <|endoftext|> and <|im_end|>
} Tokenizer;

typedef uint16_t bf16; // bfloat16: the top 16 bits of a float32
typedef _Float16 f16;

static inline float bf16_to_f32(bf16 h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

// Weights in blocks of 32 along each row, one f16 scale per block. With 8 bits
// these are the numbers of llama.cpp's Q8_0, with 4 bits those of its Q4_0,
// with values and scales in separate arrays. Rows are interleaved in groups
// of 4; matmul.c describes the layout.
typedef struct {
    int8_t *q; // [rows / 4][cols / 32][16 * bits]
    f16 *d;    // [rows / 4][cols / 32][4]
    int rows, cols, bits;
} QMatrix;

typedef struct {
    float *attn_norm, *q_norm, *k_norm, *ffn_norm; // RMSNorm weights
    QMatrix wqkv, wo;                              // attention: q, k, v stacked; output
    QMatrix w_gate_up, w_down;                     // SwiGLU MLP: gate and up stacked; down
} Layer;

typedef struct {
    Config c;
    Tokenizer tok;
    const bf16 *embed; // [vocab, dim] token embeddings
    QMatrix lm_head;   // [vocab, dim], a quantized copy of embed when tied
    float *norm;       // final RMSNorm
    Layer *layers;
    // activations for a batch of up to BATCH tokens, one row per token
    float *x, *xn, *qkv, *att, *gate_up, *act;
    float *k_cache, *v_cache; // [layer][position][n_kv_heads * head_dim]
    float *rope_cos, *rope_sin; // [position][head_dim / 2]
} Model;

Model *model_load(const char *path, int bits); // bits: 8, or 4 for the layer weights
void model_free(Model *m);

// Run tokens[0..n) at positions pos..pos+n. Writes logits for the last n_logits of them.
void forward(Model *m, const int *tokens, int n, int pos, float *logits, int n_logits);

int tokenizer_encode(const Tokenizer *t, const char *text, int *out, int max);
const char *tokenizer_piece(const Tokenizer *t, int id, int *len);
int tokenizer_is_special(const Tokenizer *t, int id);

QMatrix quantize_matrix(const bf16 *w, int rows, int cols, int bits);
void matmul(float *out, const float *x, int n, const QMatrix *w); // out[n, rows] = x[n, cols] @ w^T
const char *matmul_kernels(void);                                 // name of the kernels in use

#define BLOCK 32 // values that share one scale

// Kernels for one kind of CPU. x1 multiplies row group g of m (rows 4g to
// 4g+3) by one token: x is the token quantized, xd its block scales. x4, if
// there is one, multiplies groups g and g+1 by four tokens stored one after
// another, writing out with row stride ldo. matmul.c has plain C; cpu_kernels
// (neon.c, x86.c) swaps in faster ones when this CPU has them.
typedef struct {
    const char *name;
    void (*x1)(float *out, const QMatrix *m, int g, const int8_t *x, const float *xd);
    void (*x4)(float *out, int ldo, const QMatrix *m, int g, const int8_t *x, const float *xd);
} Kernels;
void cpu_kernels(Kernels *k);

// Thread pool: run fn(arg, i, n) on each of n threads and wait for all of them.
typedef void (*Task)(void *arg, int thread, int n_threads);
void pool_start(int n_threads);
void pool_run(Task fn, void *arg);
int pool_size(void);
int default_threads(void); // CPUs this process may run on
