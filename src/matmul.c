// int8 matrix-vector products. Weights and the input vector are both quantized
// in blocks of 32: each block is 32 int8 values times one scale. A block's
// contribution to a dot product is an exact integer sum times two scales.
#include "abacus.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#define BLOCK 32

// Weights are quantized once, at load: scale = max|w| / 127 per block.
QMatrix quantize_matrix(const bf16 *w, int rows, int cols) {
    QMatrix m = {malloc((size_t)rows * cols), malloc((size_t)rows * cols / BLOCK * sizeof(f16)), rows, cols};
    for (size_t b = 0; b < (size_t)rows * cols / BLOCK; b++) {
        float v[BLOCK], amax = 0.0f;
        for (int i = 0; i < BLOCK; i++) {
            v[i] = bf16_to_f32(w[b * BLOCK + i]);
            amax = fmaxf(amax, fabsf(v[i]));
        }
        float d = amax / 127.0f, inv = d ? 1.0f / d : 0.0f;
        m.d[b] = (f16)d;
        for (int i = 0; i < BLOCK; i++) m.q[b * BLOCK + i] = (int8_t)roundf(v[i] * inv);
    }
    return m;
}

// The input vector is quantized the same way before every product.
static void quantize_vector(int8_t *q, float *d, const float *x, int n) {
    for (int b = 0; b < n / BLOCK; b++) {
        float amax = 0.0f;
        for (int i = 0; i < BLOCK; i++) amax = fmaxf(amax, fabsf(x[b * BLOCK + i]));
        d[b] = amax / 127.0f;
        float inv = d[b] ? 1.0f / d[b] : 0.0f;
        for (int i = 0; i < BLOCK; i++) q[b * BLOCK + i] = (int8_t)roundf(x[b * BLOCK + i] * inv);
    }
}

// One row times the quantized input. SDOT multiplies 16 int8 pairs and adds
// them into four int32 lanes in one instruction.
static float dot_q8(const int8_t *w, const f16 *wd, const int8_t *x, const float *xd, int blocks) {
#ifdef __ARM_FEATURE_DOTPROD
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = acc0;
    int b = 0;
    for (; b + 2 <= blocks; b += 2) {
        const int8_t *w0 = w + b * BLOCK, *x0 = x + b * BLOCK;
        int32x4_t s0 = vdotq_s32(vdupq_n_s32(0), vld1q_s8(w0), vld1q_s8(x0));
        int32x4_t s1 = vdotq_s32(vdupq_n_s32(0), vld1q_s8(w0 + 32), vld1q_s8(x0 + 32));
        s0 = vdotq_s32(s0, vld1q_s8(w0 + 16), vld1q_s8(x0 + 16));
        s1 = vdotq_s32(s1, vld1q_s8(w0 + 48), vld1q_s8(x0 + 48));
        acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(s0), (float)wd[b] * xd[b]);
        acc1 = vfmaq_n_f32(acc1, vcvtq_f32_s32(s1), (float)wd[b + 1] * xd[b + 1]);
    }
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
#else
    float sum = 0.0f;
    int b = 0;
#endif
    for (; b < blocks; b++) {
        int32_t isum = 0;
        for (int i = 0; i < BLOCK; i++) isum += w[b * BLOCK + i] * x[b * BLOCK + i];
        sum += (float)isum * ((float)wd[b] * xd[b]);
    }
    return sum;
}

#if defined(__ARM_FEATURE_DOTPROD) && !defined(__ARM_FEATURE_MATMUL_INT8)
// Two weight rows times four tokens: eight dot products that share every
// load, so each weight byte brought in from memory is used four times.
static void tile_2x4(float *out, int ldo, const QMatrix *w, int row, const int8_t *x, const float *xd) {
    int cols = w->cols, blocks = cols / BLOCK;
    const int8_t *wq = w->q + (size_t)row * cols;
    const f16 *wd = w->d + (size_t)row * blocks;
    float32x4_t acc[2][4];
    for (int r = 0; r < 2; r++)
        for (int t = 0; t < 4; t++) acc[r][t] = vdupq_n_f32(0.0f);
    for (int b = 0; b < blocks; b++) {
        int8x16_t wl[2], wh[2], xl[4], xh[4];
        float ws[2], xs[4];
        for (int r = 0; r < 2; r++) {
            wl[r] = vld1q_s8(wq + (size_t)r * cols + b * BLOCK);
            wh[r] = vld1q_s8(wq + (size_t)r * cols + b * BLOCK + 16);
            ws[r] = (float)wd[r * blocks + b];
        }
        for (int t = 0; t < 4; t++) {
            xl[t] = vld1q_s8(x + (size_t)t * cols + b * BLOCK);
            xh[t] = vld1q_s8(x + (size_t)t * cols + b * BLOCK + 16);
            xs[t] = xd[t * blocks + b];
        }
        for (int r = 0; r < 2; r++)
            for (int t = 0; t < 4; t++) {
                int32x4_t s = vdotq_s32(vdotq_s32(vdupq_n_s32(0), wl[r], xl[t]), wh[r], xh[t]);
                acc[r][t] = vfmaq_n_f32(acc[r][t], vcvtq_f32_s32(s), ws[r] * xs[t]);
            }
    }
    for (int r = 0; r < 2; r++)
        for (int t = 0; t < 4; t++) out[(size_t)t * ldo + row + r] = vaddvq_f32(acc[r][t]);
}
#endif

#ifdef __ARM_FEATURE_MATMUL_INT8
// SMMLA multiplies a 2x8 int8 matrix by the transpose of another 2x8 one and
// adds the 2x2 int32 result: two rows times two tokens, eight deep, in one
// instruction, twice the work of an SDOT. A 4x4 tile is four of them.
static void tile_4x4(float *out, int ldo, const QMatrix *w, int row, const int8_t *x, const float *xd) {
    int cols = w->cols, blocks = cols / BLOCK;
    const int8_t *wq = w->q + (size_t)row * cols;
    const f16 *wd = w->d + (size_t)row * blocks;
    float32x4_t acc[2][2]; // [row pair][token pair], lanes (r0,t0) (r0,t1) (r1,t0) (r1,t1)
    for (int p = 0; p < 2; p++)
        for (int q = 0; q < 2; q++) acc[p][q] = vdupq_n_f32(0.0f);
    for (int b = 0; b < blocks; b++) {
        int32x4_t s[2][2];
        for (int p = 0; p < 2; p++)
            for (int q = 0; q < 2; q++) s[p][q] = vdupq_n_s32(0);
        for (int k = b * BLOCK; k < (b + 1) * BLOCK; k += 16) {
            // pair up rows: lo holds bytes 0-7 of both rows, hi bytes 8-15
            int8x16_t w_lo[2], w_hi[2], x_lo[2], x_hi[2];
            for (int p = 0; p < 2; p++) {
                int64x2_t r0 = vreinterpretq_s64_s8(vld1q_s8(wq + (size_t)(2 * p) * cols + k));
                int64x2_t r1 = vreinterpretq_s64_s8(vld1q_s8(wq + (size_t)(2 * p + 1) * cols + k));
                w_lo[p] = vreinterpretq_s8_s64(vzip1q_s64(r0, r1));
                w_hi[p] = vreinterpretq_s8_s64(vzip2q_s64(r0, r1));
                int64x2_t t0 = vreinterpretq_s64_s8(vld1q_s8(x + (size_t)(2 * p) * cols + k));
                int64x2_t t1 = vreinterpretq_s64_s8(vld1q_s8(x + (size_t)(2 * p + 1) * cols + k));
                x_lo[p] = vreinterpretq_s8_s64(vzip1q_s64(t0, t1));
                x_hi[p] = vreinterpretq_s8_s64(vzip2q_s64(t0, t1));
            }
            for (int p = 0; p < 2; p++)
                for (int q = 0; q < 2; q++) s[p][q] = vmmlaq_s32(vmmlaq_s32(s[p][q], w_lo[p], x_lo[q]), w_hi[p], x_hi[q]);
        }
        for (int p = 0; p < 2; p++) {
            float32x4_t ws = vcombine_f32(vdup_n_f32((float)wd[2 * p * blocks + b]),
                                          vdup_n_f32((float)wd[(2 * p + 1) * blocks + b]));
            for (int q = 0; q < 2; q++) {
                float32x2_t xs = {xd[2 * q * blocks + b], xd[(2 * q + 1) * blocks + b]};
                acc[p][q] = vfmaq_f32(acc[p][q], vcvtq_f32_s32(s[p][q]), vmulq_f32(ws, vcombine_f32(xs, xs)));
            }
        }
    }
    for (int p = 0; p < 2; p++)
        for (int q = 0; q < 2; q++) {
            float v[4];
            vst1q_f32(v, acc[p][q]);
            out[(size_t)(2 * q) * ldo + row + 2 * p] = v[0];
            out[(size_t)(2 * q + 1) * ldo + row + 2 * p] = v[1];
            out[(size_t)(2 * q) * ldo + row + 2 * p + 1] = v[2];
            out[(size_t)(2 * q + 1) * ldo + row + 2 * p + 1] = v[3];
        }
}
#define TILE_ROWS 4
#define tile tile_4x4
#elif defined(__ARM_FEATURE_DOTPROD)
#define TILE_ROWS 2
#define tile tile_2x4
#else
#define TILE_ROWS 1
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
    int cols = w->cols, blocks = cols / BLOCK, groups = w->rows / TILE_ROWS;
    // slices start on a tile boundary so that tiles never straddle two threads
    int begin = TILE_ROWS * (int)((long)groups * thread / n_threads);
    int end = thread == n_threads - 1 ? w->rows : TILE_ROWS * (int)((long)groups * (thread + 1) / n_threads);
    for (int r = begin; r < end; r += TILE_ROWS) {
        int t = 0;
#if TILE_ROWS > 1
        for (; r + TILE_ROWS <= end && t + 4 <= a->n; t += 4)
            tile(a->out + (size_t)t * w->rows, w->rows, w, r, a->xq + (size_t)t * cols, a->xd + (size_t)t * blocks);
#endif
        for (; t < a->n; t++)
            for (int i = r; i < r + TILE_ROWS && i < end; i++)
                a->out[(size_t)t * w->rows + i] = dot_q8(w->q + (size_t)i * cols, w->d + (size_t)i * blocks,
                                                         a->xq + (size_t)t * cols, a->xd + (size_t)t * blocks, blocks);
    }
}

// out[n, rows] = x[n, cols] @ w^T, each thread computing a slice of the rows
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
