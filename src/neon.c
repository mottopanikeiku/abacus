// NEON kernels for arm64: SDOT for one token, SMMLA (i8mm) for four. Both
// read the 4-row groups described in matmul.c.
#ifdef __aarch64__
#include "abacus.h"
#include <arm_neon.h>

#ifdef __ARM_FEATURE_DOTPROD
// One group (4 rows) times one token. SDOT multiplies 16 int8 pairs into four
// int32 lanes; fed 8 bytes of two rows against the same 8 inputs twice, its
// lanes hold (row 0, row 0, row 1, row 1).
static void x1_sdot(float *out, const QMatrix *m, int g, const int8_t *x, const float *xd) {
    int blocks = m->cols / BLOCK;
    const int8_t *q = m->q + (size_t)g * blocks * 128;
    const f16 *d = m->d + (size_t)g * blocks * 4;
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
}
#endif

#ifdef __ARM_FEATURE_MATMUL_INT8
// Two groups (8 rows) times four tokens. SMMLA multiplies a 2x8 int8 matrix
// by the transpose of another 2x8 one and adds the 2x2 int32 result: two rows
// times two tokens, eight deep, twice the work of an SDOT. The tile is 4 row
// pairs by 2 token pairs, so each loaded input feeds four SMMLAs. out has row
// stride ldo.
static void x4_smmla(float *out, int ldo, const QMatrix *m, int g, const int8_t *x, const float *xd) {
    int blocks = m->cols / BLOCK, cols = m->cols;
    const int8_t *q = m->q + (size_t)g * blocks * 128;
    const f16 *d = m->d + (size_t)g * blocks * 4;
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
        for (int h = 0; h < 2; h++) {
            float32x4_t rs = vcvt_f32_f16(vld1_f16((const float16_t *)d + h * next_d + b * 4));
            float32x4_t row[2] = {vzip1q_f32(rs, rs), vzip2q_f32(rs, rs)}; // (r0,r0,r1,r1), (r2,r2,r3,r3)
            for (int p = 0; p < 2; p++)
                for (int t = 0; t < 2; t++)
                    acc[2 * h + p][t] = vfmaq_f32(acc[2 * h + p][t], vcvtq_f32_s32(s[2 * h + p][t]), vmulq_f32(row[p], tok[t]));
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

void cpu_kernels(Kernels *k) {
#ifdef __ARM_FEATURE_DOTPROD
    *k = (Kernels){"neon dotprod", x1_sdot, NULL};
#endif
#ifdef __ARM_FEATURE_MATMUL_INT8
    *k = (Kernels){"neon i8mm", x1_sdot, x4_smmla};
#endif
    (void)k;
}
#endif
