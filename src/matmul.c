// int8 matrix products. Weights and inputs are both quantized in blocks of 32:
// 32 int8 values times one scale. A block's share of a dot product is an exact
// integer sum times two scales.
//
// Weights are kept in groups of 4 rows. Inside a group, each block of 32
// columns is 128 bytes: 8 bytes of row 0, 8 of row 1, 8 of row 2, 8 of row 3,
// then the next 8 of each row, and so on (llama.cpp's q8_0 4x8 layout).
// Decoding streams one array instead of four rows at once, SMMLA finds the
// row pairs it multiplies already side by side, and one 32-byte AVX2 load is
// 8 columns of all 4 rows.
//
// This file quantizes and splits the work over threads. The products
// themselves are kernels: plain C here, NEON in neon.c, AVX2 in x86.c, picked
// for the CPU on first use.
#include "abacus.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Weights are quantized once, at load: scale = max|w| / 127 per block of a row.
QMatrix quantize_matrix(const bf16 *w, int rows, int cols) {
    if (rows % 8 || cols % BLOCK) { // products work on 8 rows at a time
        fprintf(stderr, "abacus: matrix shape %dx%d is not a multiple of 8x32\n", rows, cols);
        exit(1);
    }
    int blocks = cols / BLOCK;
    QMatrix m = {malloc((size_t)rows * cols), malloc((size_t)rows * blocks * sizeof(f16)), rows, cols};
    for (int r = 0; r < rows; r++)
        for (int b = 0; b < blocks; b++) {
            const bf16 *src = w + (size_t)r * cols + b * BLOCK;
            float amax = 0.0f;
            for (int i = 0; i < BLOCK; i++) amax = fmaxf(amax, fabsf(bf16_to_f32(src[i])));
            float d = amax / 127.0f, inv = d ? 1.0f / d : 0.0f;
            size_t block = (size_t)(r / 4) * blocks + b; // position of this block in group order
            m.d[block * 4 + r % 4] = (f16)d;
            for (int i = 0; i < BLOCK; i++)
                m.q[block * 128 + (i / 8) * 32 + (r % 4) * 8 + i % 8] = (int8_t)roundf(bf16_to_f32(src[i]) * inv);
        }
    return m;
}

// Inputs are quantized the same way before every product, one row per token.
static void quantize_vector(int8_t *q, float *d, const float *x, int n) {
    for (int b = 0; b < n / BLOCK; b++) {
        float amax = 0.0f;
        for (int i = 0; i < BLOCK; i++) amax = fmaxf(amax, fabsf(x[b * BLOCK + i]));
        d[b] = amax / 127.0f;
        float inv = d[b] ? 1.0f / d[b] : 0.0f;
        for (int i = 0; i < BLOCK; i++) q[b * BLOCK + i] = (int8_t)roundf(x[b * BLOCK + i] * inv);
    }
}

// One group (4 rows) times one token, in plain C.
static void c_x1(float *out, const QMatrix *m, int g, const int8_t *x, const float *xd) {
    int blocks = m->cols / BLOCK;
    const int8_t *q = m->q + (size_t)g * blocks * 128;
    const f16 *d = m->d + (size_t)g * blocks * 4;
    for (int r = 0; r < 4; r++) {
        float sum = 0.0f;
        for (int b = 0; b < blocks; b++) {
            int32_t isum = 0;
            for (int i = 0; i < BLOCK; i++) isum += q[b * 128 + (i / 8) * 32 + r * 8 + i % 8] * x[b * BLOCK + i];
            sum += (float)isum * ((float)d[b * 4 + r] * xd[b]);
        }
        out[r] = sum;
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
        quantize_vector(a->xq + (size_t)t * cols, a->xd + (size_t)t * (cols / BLOCK), a->x + (size_t)t * cols, cols);
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
