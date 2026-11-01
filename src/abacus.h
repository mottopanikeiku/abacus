// abacus: a small CPU inference engine for Qwen3 models.
#pragma once
#include <stddef.h>
#include <stdint.h>

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

typedef struct {
    float *attn_norm, *q_norm, *k_norm, *ffn_norm; // RMSNorm weights
    float *wq, *wk, *wv, *wo;                      // attention projections, [out, in]
    float *w_gate, *w_up, *w_down;                 // SwiGLU MLP
} Layer;

typedef struct {
    Config c;
    Tokenizer tok;
    float *embed;   // [vocab, dim] token embeddings
    float *lm_head; // [vocab, dim], the same tensor as embed when tied
    float *norm;    // final RMSNorm
    Layer *layers;
    // activations for the token being processed
    float *x, *xn, *q, *k, *v, *att, *scores, *gate, *up;
    float *k_cache, *v_cache; // [layer][position][n_kv_heads * head_dim]
} Model;

Model *model_load(const char *path);
void model_free(Model *m);

// Run tokens[0..n) at positions pos..pos+n. Writes logits for the last n_logits of them.
void forward(Model *m, const int *tokens, int n, int pos, float *logits, int n_logits);

int tokenizer_encode(const Tokenizer *t, const char *text, int *out, int max);
const char *tokenizer_piece(const Tokenizer *t, int id, int *len);
int tokenizer_is_special(const Tokenizer *t, int id);
