// int8 matrix products. Weights and inputs are both quantized in blocks of 32:
// 32 int8 values times one scale. A block's share of a dot product is an exact
// integer sum times two scales.
//
// With 8-bit weights, every block of 32 weights and every block of 32 inputs
// is rotated by the same 32x32 Hadamard matrix before it is quantized. The
// matrix is orthogonal and symmetric, so the rotated blocks have the same dot
// product as the originals, but a block's largest value gets spread over all
// 32 of them and no longer sets a scale that rounds its neighbours to zero.
// Some Qwen3 inputs are thousands of times larger than the rest of their
// block, and rotating is what got Qwen3-1.7B through the accuracy check.
// 4-bit weights stay unrotated because their rounding already lands the
// largest value exactly, and rotating them cost more accuracy than it saved.
// Rotating before quantizing is the idea behind QuaRot and QuIP#.
//
// Weights are kept in groups of 4 rows. Inside a group, each block of 32
// columns is 128 bytes: 8 bytes of row 0, 8 of row 1, 8 of row 2, 8 of row 3,
// then the next 8 of each row, and so on (llama.cpp's q8_0 4x8 layout).
// Decoding streams one array instead of four rows at once, SMMLA finds the
// row pairs it multiplies already side by side, and one 32-byte AVX2 load is
// 8 columns of all 4 rows.
//
// 4-bit weights (--q4) keep the same order, two to a byte: byte i of a group's
// 64-byte block holds value i in its low half and value i + 64 in its high
// half, in 4-bit two's complement. byte & 0xF0, read as an int8, is then 16
// times the high value and byte << 4 is 16 times the low one: one instruction
// unpacks a register of weights, and the kernels divide by 16 at the end.
//
// This file quantizes and splits the work over threads. The products
// themselves are kernels: plain C here, NEON in neon.c, AVX2 in x86.c, picked
// for the CPU on first use.
#include "abacus.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const bf16 *w;
    QMatrix *m;
} Quantize;

// The 32-point Walsh-Hadamard transform, scaled by 1 / sqrt(32) so that
// applying it twice gives back x. Its five butterfly stages pair values 1, 2,
// 4, 8 and 16 apart; held as eight 4-wide vectors, the first two stages are
// a lane swap and a multiply-add inside each vector, the other three add and
// subtract whole vectors.
typedef float v4 __attribute__((vector_size(16)));
#if defined(__clang__) || __GNUC__ >= 12
#define LANES(v, a, b, c, d) __builtin_shufflevector(v, v, a, b, c, d)
#else // GCC 11 and older only have their own builtin
typedef int v4i __attribute__((vector_size(16)));
#define LANES(v, a, b, c, d) __builtin_shuffle(v, (v4i){a, b, c, d})
#endif
static void rotate(float *x) {
    v4 v[BLOCK / 4];
    memcpy(v, x, sizeof v);
    for (int i = 0; i < BLOCK / 4; i++) {
        v[i] = LANES(v[i], 1, 0, 3, 2) + v[i] * (v4){1, -1, 1, -1};
        v[i] = LANES(v[i], 2, 3, 0, 1) + v[i] * (v4){1, 1, -1, -1};
    }
    for (int h = 1; h < BLOCK / 4; h *= 2)
        for (int i = 0; i < BLOCK / 4; i += 2 * h)
            for (int j = i; j < i + h; j++) {
                v4 a = v[j], b = v[j + h];
                v[j] = a + b, v[j + h] = a - b;
            }
    for (int i = 0; i < BLOCK / 4; i++) v[i] *= 0.17677669529663687f;
    memcpy(x, v, sizeof v);
}

// Weights are quantized once, at load. With 8 bits, scale = max|w| / 127 per
// (rotated) block of a row. With 4 bits, scale = -m / 8 where m is the weight
// of largest magnitude: m becomes -8 and the others land in -8..7. Each thread
// takes a share of the row groups (the 4 rows of a group share bytes at 4 bits).
static void quantize_groups(void *arg, int thread, int n_threads) {
    const Quantize *a = arg;
    QMatrix *m = a->m;
    int groups = m->rows / 4, cols = m->cols, blocks = cols / BLOCK, bits = m->bits, bytes = 16 * bits;
    for (int r = groups * thread / n_threads * 4; r < groups * (thread + 1) / n_threads * 4; r++)
        for (int b = 0; b < blocks; b++) {
            float w[BLOCK], big = 0.0f;
            for (int i = 0; i < BLOCK; i++) w[i] = bf16_to_f32(a->w[(size_t)r * cols + b * BLOCK + i]);
            if (bits == 8) rotate(w);
            for (int i = 0; i < BLOCK; i++)
                if (fabsf(w[i]) > fabsf(big)) big = w[i];
            float d = bits == 8 ? fabsf(big) / 127.0f : big / -8.0f, inv = d ? 1.0f / d : 0.0f;
            size_t block = (size_t)(r / 4) * blocks + b; // position of this block in group order
            m->d[block * 4 + r % 4] = (f16)d;
            int8_t *q = m->q + block * bytes;
            for (int i = 0; i < BLOCK; i++) {
                int k = (i / 8) * 32 + (r % 4) * 8 + i % 8, v = (int)roundf(w[i] * inv);
                if (bits == 8) q[k] = (int8_t)v;
                else q[k % 64] |= (int8_t)(((v < 7 ? v : 7) & 15) << (k < 64 ? 0 : 4));
            }
        }
}

QMatrix quantize_matrix(const bf16 *w, int rows, int cols, int bits) {
    if (rows % 8 || cols % BLOCK) { // products work on 8 rows at a time
        fprintf(stderr, "abacus: matrix shape %dx%d is not a multiple of 8x32\n", rows, cols);
        exit(1);
    }
    int blocks = cols / BLOCK;
    QMatrix m = {calloc((size_t)rows / 4 * blocks, 16 * bits), malloc((size_t)rows * blocks * sizeof(f16)), rows,
                 cols, bits};
    pool_run(quantize_groups, &(Quantize){w, &m});
    return m;
}

// Inputs are quantized the same way before every product, one row per token,
// and rotated first if the weights are.
static void quantize_vector(int8_t *q, float *d, const float *x, int n, int rotated) {
    for (int b = 0; b < n / BLOCK; b++) {
        float v[BLOCK], amax = 0.0f;
        memcpy(v, x + b * BLOCK, sizeof v);
        if (rotated) rotate(v);
        for (int i = 0; i < BLOCK; i++) amax = fmaxf(amax, fabsf(v[i]));
        d[b] = amax / 127.0f;
        float inv = d[b] ? 1.0f / d[b] : 0.0f;
        for (int i = 0; i < BLOCK; i++) q[b * BLOCK + i] = (int8_t)roundf(v[i] * inv);
    }
}

// Weight k (0..127) of a group's block, 16 times too big if it has 4 bits.
static int weight(const int8_t *q, int bits, int k) {
    if (bits == 8) return q[k];
    uint8_t byte = (uint8_t)q[k % 64];
    return (int8_t)(k < 64 ? byte << 4 : byte & 0xF0);
}

// One group (4 rows) times one token, in plain C.
static void c_x1(float *out, const QMatrix *m, int g, const int8_t *x, const float *xd) {
    int blocks = m->cols / BLOCK, bytes = 16 * m->bits;
    const int8_t *q = m->q + (size_t)g * blocks * bytes;
    const f16 *d = m->d + (size_t)g * blocks * 4;
    for (int r = 0; r < 4; r++) {
        float sum = 0.0f;
        for (int b = 0; b < blocks; b++) {
            int32_t isum = 0;
            for (int i = 0; i < BLOCK; i++)
                isum += weight(q + b * bytes, m->bits, (i / 8) * 32 + r * 8 + i % 8) * x[b * BLOCK + i];
            sum += (float)isum * ((float)d[b * 4 + r] * xd[b]);
        }
        out[r] = m->bits == 4 ? sum / 16 : sum;
    }
}

static Kernels kernels;

const char *matmul_kernels(void) {
    if (!kernels.name) {
        kernels = (Kernels){"c", c_x1, NULL};
#if defined(__aarch64__) || defined(__x86_64__)
        cpu_kernels(&kernels);
#endif
    }
    return kernels.name;
}

typedef struct {
    float *out;     // [n, rows]
    const float *x; // [n, cols]
    const QMatrix *w;
    int8_t *xq; // x quantized
    float *xd;
    int n;
} Matmul;

static void quantize_rows(void *arg, int thread, int n_threads) {
    const Matmul *a = arg;
    int cols = a->w->cols;
    for (int t = a->n * thread / n_threads; t < a->n * (thread + 1) / n_threads; t++)
        quantize_vector(a->xq + (size_t)t * cols, a->xd + (size_t)t * (cols / BLOCK), a->x + (size_t)t * cols, cols,
                        a->w->bits == 8);
}

static void matmul_rows(void *arg, int thread, int n_threads) {
    const Matmul *a = arg;
    const QMatrix *w = a->w;
    int rows = w->rows, cols = w->cols, blocks = cols / BLOCK, tiles = rows / 8;
    for (int i = tiles * thread / n_threads; i < tiles * (thread + 1) / n_threads; i++) {
        float *out = a->out + 8 * i;
        int t = 0;
        if (kernels.x4)
            for (; t + 4 <= a->n; t += 4)
                kernels.x4(out + (size_t)t * rows, rows, w, 2 * i, a->xq + (size_t)t * cols,
                           a->xd + (size_t)t * blocks);
        for (; t < a->n; t++)
            for (int g = 0; g < 2; g++)
                kernels.x1(out + (size_t)t * rows + 4 * g, w, 2 * i + g, a->xq + (size_t)t * cols,
                           a->xd + (size_t)t * blocks);
    }
}

// out[n, rows] = x[n, cols] @ w^T, each thread computing a slice of the row groups
void matmul(float *out, const float *x, int n, const QMatrix *w) {
    static int8_t *xq;
    static float *xd;
    static size_t capacity;
    if ((size_t)n * w->cols > capacity) {
        capacity = (size_t)n * w->cols;
        xq = realloc(xq, capacity);
        xd = realloc(xd, capacity / BLOCK * sizeof(float));
    }
    matmul_kernels();
    Matmul a = {out, x, w, xq, xd, n};
    if (n > 1) pool_run(quantize_rows, &a);
    else quantize_rows(&a, 0, 1);
    pool_run(matmul_rows, &a);
}
