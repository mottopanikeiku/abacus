// Command line: chat with the model, benchmark it, or dump logits for the checks.
#include "abacus.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *usage =
    "usage: abacus MODEL [options]\n"
    "  -p TEXT          prompt, wrapped in the chat template (default: a question)\n"
    "  -n N             tokens to generate (default 512)\n"
    "  -t N             threads (default: every CPU this process may use)\n"
    "  --temp T         sampling temperature, 0 = greedy (default 0.7)\n"
    "  --seed S         sampling seed\n"
    "  --raw            continue the prompt as plain text, no chat template\n"
    "  --q4             4-bit layer weights (half the memory traffic, less accurate)\n"
    "  --bench PP TG    time a PP-token prefill and a TG-token decode, print JSON\n"
    "  --tokenize TEXT  print token ids\n"
    "  --ids \"1 2 3\"    start from token ids; prints N greedy token ids\n"
    "  --dump-logits F  with --ids: write float32 logits for positions >= --from K - 1,\n"
    "                   feeding the ids in chunks of --chunk C (default: all at once)\n"
    "  --info           print the model shape and weight format\n";

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

// Top-20 sampling at temperature temp.
static int sample(const float *logits, int n, float temp, uint64_t *rng) {
    if (temp <= 0.0f) return argmax(logits, n);
    enum { K = 20 };
    int top[K], k = 0;
    for (int i = 0; i < n; i++) {
        if (k == K && logits[i] <= logits[top[K - 1]]) continue;
        int j = k < K ? k++ : K - 1;
        for (; j > 0 && logits[top[j - 1]] < logits[i]; j--) top[j] = top[j - 1];
        top[j] = i;
    }
    float p[K], sum = 0.0f;
    for (int j = 0; j < k; j++) sum += p[j] = expf((logits[top[j]] - logits[top[0]]) / temp);
    *rng ^= *rng << 13, *rng ^= *rng >> 7, *rng ^= *rng << 17; // xorshift64
    float r = (float)(*rng >> 40) / (float)(1 << 24) * sum;
    for (int j = 0; j < k; j++)
        if ((r -= p[j]) <= 0.0f) return top[j];
    return top[k - 1];
}

static int parse_ids(const char *s, int *out, int max) {
    int n = 0;
    char *end;
    for (long v = strtol(s, &end, 10); end != s && n < max; v = strtol(s, &end, 10)) out[n++] = (int)v, s = end;
    return n;
}

static void bench(Model *m, int pp, int tg) {
    int vocab = m->c.vocab_size;
    float *logits = malloc(vocab * sizeof(float));
    int *tokens = malloc((pp > 16 ? pp : 16) * sizeof(int));
    for (int i = 0; i < (pp > 16 ? pp : 16); i++) tokens[i] = 1000 + 37 * i % 50000;
    forward(m, tokens, 16, 0, logits, 1); // warm up
    double t0 = now();
    forward(m, tokens, pp, 0, logits, 1);
    double t1 = now();
    int token = tokens[0];
    for (int pos = 0; pos < tg; pos++) {
        forward(m, &token, 1, pos, logits, 1);
        token = argmax(logits, vocab);
    }
    double t2 = now();
    printf("{\"prefill_tok_s\": %.3f, \"decode_tok_s\": %.3f, \"pp\": %d, \"tg\": %d, \"threads\": %d}\n",
           pp / (t1 - t0), tg / (t2 - t1), pp, tg, pool_size());
}

// Teacher forcing: run the given ids and write the logits that predict ids[from..].
static void dump_logits(Model *m, const int *ids, int n, int from, int chunk, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "abacus: cannot write %s\n", path);
        exit(1);
    }
    int vocab = m->c.vocab_size;
    if (chunk <= 0) chunk = n;
    float *logits = malloc((size_t)chunk * vocab * sizeof(float));
    for (int pos = 0; pos < n; pos += chunk) {
        int len = n - pos < chunk ? n - pos : chunk;
        int first = pos > from - 1 ? pos : from - 1;
        int n_logits = pos + len - first > 0 ? pos + len - first : 0;
        forward(m, ids + pos, len, pos, logits, n_logits);
        fwrite(logits, sizeof(float), (size_t)n_logits * vocab, f);
    }
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 2 || argv[1][0] == '-') {
        fputs(usage, stderr);
        return 1;
    }
    const char *prompt = "Why is the ocean salty?", *ids_arg = NULL, *dump = NULL, *text = NULL;
    int steps = 512, raw = 0, info = 0, from = 1, chunk = 0, pp = 0, tg = 0, bits = 8, threads = default_threads();
    float temp = 0.7f;
    uint64_t seed = (uint64_t)time(NULL);
    for (int i = 2; i < argc; i++) {
        const char *a = argv[i];
        int more = i + 1 < argc;
        if (!strcmp(a, "-p") && more) prompt = argv[++i];
        else if (!strcmp(a, "-n") && more) steps = atoi(argv[++i]);
        else if (!strcmp(a, "-t") && more) threads = atoi(argv[++i]);
        else if (!strcmp(a, "--temp") && more) temp = (float)atof(argv[++i]);
        else if (!strcmp(a, "--seed") && more) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "--raw")) raw = 1;
        else if (!strcmp(a, "--q4")) bits = 4;
        else if (!strcmp(a, "--bench") && i + 2 < argc) pp = atoi(argv[++i]), tg = atoi(argv[++i]);
        else if (!strcmp(a, "--tokenize") && more) text = argv[++i];
        else if (!strcmp(a, "--ids") && more) ids_arg = argv[++i];
        else if (!strcmp(a, "--dump-logits") && more) dump = argv[++i];
        else if (!strcmp(a, "--from") && more) from = atoi(argv[++i]);
        else if (!strcmp(a, "--chunk") && more) chunk = atoi(argv[++i]);
        else if (!strcmp(a, "--info")) info = 1;
        else {
            fputs(usage, stderr);
            return 1;
        }
    }

    pool_start(threads); // loading quantizes on every thread
    Model *m = model_load(argv[1], bits);
    const Config *c = &m->c;
    int vocab = c->vocab_size;
    int *tokens = malloc(c->seq_len * sizeof(int));
    float *logits = malloc(vocab * sizeof(float));

    if (info) {
        printf("qwen3 dim=%d hidden=%d layers=%d heads=%d/%d head_dim=%d vocab=%d weights=q%d_0 kernels=%s\n", c->dim,
               c->hidden_dim, c->n_layers, c->n_heads, c->n_kv_heads, c->head_dim, vocab, bits, matmul_kernels());
    } else if (text) {
        int n = tokenizer_encode(&m->tok, text, tokens, c->seq_len);
        for (int i = 0; i < n; i++) printf(i ? " %d" : "%d", tokens[i]);
        printf("\n");
    } else if (pp > 0 || tg > 0) {
        bench(m, pp, tg);
    } else if (ids_arg) {
        int n = parse_ids(ids_arg, tokens, c->seq_len);
        if (dump) {
            dump_logits(m, tokens, n, from, chunk, dump);
        } else {
            forward(m, tokens, n, 0, logits, 1);
            for (int i = 0; i < steps && n + i < c->seq_len; i++) {
                int next = argmax(logits, vocab);
                printf(i ? " %d" : "%d", next);
                forward(m, &next, 1, n + i, logits, 1);
            }
            printf("\n");
        }
    } else {
        char *chat = malloc(strlen(prompt) + 128);
        if (raw) strcpy(chat, prompt);
        else sprintf(chat, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", prompt);
        int n = tokenizer_encode(&m->tok, chat, tokens, c->seq_len);
        double t0 = now();
        forward(m, tokens, n, 0, logits, 1);
        double t1 = now();
        int generated = 0;
        for (; generated < steps && n + generated < c->seq_len; generated++) {
            int next = sample(logits, vocab, temp, &seed);
            if (next == m->tok.eos || next == m->tok.bos) break;
            int len;
            const char *piece = tokenizer_piece(&m->tok, next, &len);
            if (!tokenizer_is_special(&m->tok, next)) fwrite(piece, 1, len, stdout);
            fflush(stdout);
            forward(m, &next, 1, n + generated, logits, 1);
        }
        double t2 = now();
        fprintf(stderr, "\n\nprompt %d tokens, %.1f tok/s | generated %d tokens, %.1f tok/s\n", n, n / (t1 - t0),
                generated, generated / (t2 - t1));
    }
    model_free(m);
    return 0;
}
