// int8 matrix products. Weights and inputs are both quantized in blocks of 32:
// 32 int8 values times one scale. A block's share of a dot product is an exact
// integer sum times two scales.
//
// Weights are kept in groups of 4 rows. Inside a group, each block of 32
// columns is 128 bytes: 8 bytes of row 0, 8 of row 1, 8 of row 2, 8 of row 3,
// then the next 8 of each row, and so on (llama.cpp's q8_0 4x8 layout).
// Decoding streams one array instead of four rows at once, and SMMLA finds
// the row pairs it multiplies already side by side.
#include "abacus.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#define BLOCK 32

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

// One group (4 rows) times one token. SDOT multiplies 16 int8 pairs into four
// int32 lanes; fed 8 bytes of two rows against the same 8 inputs twice, its
// lanes hold (row 0, row 0, row 1, row 1).
static void group_x1(float *out, const int8_t *q, const f16 *d, const int8_t *x, const float *xd, int blocks) {
#ifdef __ARM_FEATURE_DOTPROD
    float32x4_t acc01 = vdupq_n_f32(0.0f), acc23 = acc01;
    for (int b = 0; b < blocks; b++, q += 128, d += 4) {
        int32x4_t s01 = vdupq_n_s32(0), s23 = s01;
        for (int c = 0; c < 4; c++) {
            int8x8_t xc = vld1_s8(x + b * BLOCK + c * 8);
            int8x16_t xx = vcombine_s8(xc, xc);
            s01 = vdotq_s32(s01, vld1q_s8(q + c * 32), xx);
            s23 = vdotq_s32(s23, vld1q_s8(q + c * 32 + 16), xx);
        }
        float32x4_t scale = vmulq_n_f32(vcvt_f32_f16(vld1_f16((const float16_t *)d)), xd[b]); // rows 0..3
        acc01 = vfmaq_f32(acc01, vcvtq_f32_s32(s01), vzip1q_f32(scale, scale));
        acc23 = vfmaq_f32(acc23, vcvtq_f32_s32(s23), vzip2q_f32(scale, scale));
    }
    vst1q_f32(out, vpaddq_f32(acc01, acc23));
#else
    for (int r = 0; r < 4; r++) {
        float sum = 0.0f;
        for (int b = 0; b < blocks; b++) {
            int32_t isum = 0;
            for (int i = 0; i < BLOCK; i++) isum += q[b * 128 + (i / 8) * 32 + r * 8 + i % 8] * x[b * BLOCK + i];
            sum += (float)isum * ((float)d[b * 4 + r] * xd[b]);
        }
        out[r] = sum;
    }
#endif
}

#ifdef __ARM_FEATURE_MATMUL_INT8
// Two groups (8 rows) times four tokens. SMMLA multiplies a 2x8 int8 matrix
// by the transpose of another 2x8 one and adds the 2x2 int32 result: two rows
// times two tokens, eight deep, twice the work of an SDOT. The tile is 4 row
// pairs by 2 token pairs, so each loaded input feeds four SMMLAs. out has row
// stride ldo.
static void tile_8x4(float *out, int ldo, const int8_t *q, const f16 *d, const int8_t *x, const float *xd, int blocks) {
    int cols = blocks * BLOCK;
    size_t next_q = (size_t)blocks * 128, next_d = (size_t)blocks * 4; // second group
    float32x4_t acc[4][2]; // [row pair][token pair], lanes (r0,t0) (r0,t1) (r1,t0) (r1,t1)
    for (int p = 0; p < 4; p++)
        for (int t = 0; t < 2; t++) acc[p][t] = vdupq_n_f32(0.0f);
    for (int b = 0; b < blocks; b++) {
        int32x4_t s[4][2];
        for (int p = 0; p < 4; p++)
            for (int t = 0; t < 2; t++) s[p][t] = vdupq_n_s32(0);
        for (int c = 0; c < 4; c += 2) {
            // 16 bytes of each token cover two chunks; pair the tokens up chunk by chunk
            const int8_t *xc = x + b * BLOCK + c * 8;
            int64x2_t x0 = vreinterpretq_s64_s8(vld1q_s8(xc)), x1 = vreinterpretq_s64_s8(vld1q_s8(xc + cols));
            int64x2_t x2 = vreinterpretq_s64_s8(vld1q_s8(xc + 2 * cols)), x3 = vreinterpretq_s64_s8(vld1q_s8(xc + 3 * cols));
            int8x16_t xt[2][2] = {{vreinterpretq_s8_s64(vzip1q_s64(x0, x1)), vreinterpretq_s8_s64(vzip1q_s64(x2, x3))},
                                  {vreinterpretq_s8_s64(vzip2q_s64(x0, x1)), vreinterpretq_s8_s64(vzip2q_s64(x2, x3))}};
            for (int h = 0; h < 2; h++)
                for (int p = 0; p < 4; p++) {
                    int8x16_t w = vld1q_s8(q + (p / 2) * next_q + (size_t)b * 128 + (c + h) * 32 + (p % 2) * 16);
                    for (int t = 0; t < 2; t++) s[p][t] = vmmlaq_s32(s[p][t], w, xt[h][t]);
                }
        }
        float32x4_t ts = {xd[b], xd[blocks + b], xd[2 * blocks + b], xd[3 * blocks + b]};
        float32x4_t tok[2] = {vcombine_f32(vget_low_f32(ts), vget_low_f32(ts)),
                              vcombine_f32(vget_high_f32(ts), vget_high_f32(ts))}; // (t0,t1,t0,t1), (t2,t3,t2,t3)
        for (int g = 0; g < 2; g++) {
            float32x4_t rs = vcvt_f32_f16(vld1_f16((const float16_t *)d + g * next_d + b * 4));
            float32x4_t row[2] = {vzip1q_f32(rs, rs), vzip2q_f32(rs, rs)}; // (r0,r0,r1,r1), (r2,r2,r3,r3)
            for (int p = 0; p < 2; p++)
                for (int t = 0; t < 2; t++)
                    acc[2 * g + p][t] = vfmaq_f32(acc[2 * g + p][t], vcvtq_f32_s32(s[2 * g + p][t]), vmulq_f32(row[p], tok[t]));
        }
    }
    for (int p = 0; p < 4; p++)
        for (int t = 0; t < 2; t++) {
            float v[4];
            vst1q_f32(v, acc[p][t]);
            out[(size_t)(2 * t) * ldo + 2 * p] = v[0];
            out[(size_t)(2 * t + 1) * ldo + 2 * p] = v[1];
            out[(size_t)(2 * t) * ldo + 2 * p + 1] = v[2];
            out[(size_t)(2 * t + 1) * ldo + 2 * p + 1] = v[3];
        }
}
#endif

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
        const int8_t *q = w->q + (size_t)2 * i * blocks * 128;
        const f16 *d = w->d + (size_t)2 * i * blocks * 4;
        float *out = a->out + 8 * i;
        int t = 0;
#ifdef __ARM_FEATURE_MATMUL_INT8
        for (; t + 4 <= a->n; t += 4)
            tile_8x4(out + (size_t)t * rows, rows, q, d, a->xq + (size_t)t * cols, a->xd + (size_t)t * blocks, blocks);
#endif
        for (; t < a->n; t++)
            for (int g = 0; g < 2; g++)
                group_x1(out + (size_t)t * rows + 4 * g, q + (size_t)g * blocks * 128, d + (size_t)g * blocks * 4,
                         a->xq + (size_t)t * cols, a->xd + (size_t)t * blocks, blocks);
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
    Matmul a = {out, x, w, xq, xd, n};
    if (n > 1) pool_run(quantize_rows, &a);
    else quantize_rows(&a, 0, 1);
    pool_run(matmul_rows, &a);
}
