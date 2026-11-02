// AVX2 kernels for x86-64, used when the CPU has AVX2, FMA and F16C (Intel
// since Haswell, AMD since Zen). Both read the 4-row groups described in
// matmul.c: one 32-byte load is 8 columns of 4 rows, met by 8 inputs
// repeated four times.
//
// AVX2 has no int8 x int8 dot product. VPMADDUBSW multiplies unsigned bytes
// by signed ones and adds neighbours into int16, so the weights lend their
// signs to the inputs, |w| * (x * sign w) = w * x, and VPMADDWD adds the int16
// pairs into int32. |x| <= 127 keeps the int16 sums from saturating.
//
// VNNI's VPDPBUSD does the multiply, both additions and the accumulation in
// one instruction. AVX-VNNI (Alder Lake, Zen 5) and AVX-512 VNNI (Ice Lake,
// Zen 4) encode it differently, so the kernels are compiled three times, once
// for each way of taking the dot product.
#ifdef __x86_64__
#include "abacus.h"
#include <immintrin.h>

#define AVX2 __attribute__((target("avx2,fma,f16c")))
#define AVXVNNI __attribute__((target("avx2,fma,f16c,avxvnni")))
#define AVX512VNNI __attribute__((target("avx2,fma,f16c,avx512vl,avx512vnni")))

// acc += w * x, 32 byte products summed in fours into 8 int32 lanes; aw = |w|
typedef __m256i Dot(__m256i acc, __m256i aw, __m256i w, __m256i x);

AVX2 static inline __m256i dot_avx2(__m256i acc, __m256i aw, __m256i w, __m256i x) {
    __m256i pairs = _mm256_maddubs_epi16(aw, _mm256_sign_epi8(x, w));
    return _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, _mm256_set1_epi16(1)));
}

AVXVNNI static inline __m256i dot_avxvnni(__m256i acc, __m256i aw, __m256i w, __m256i x) {
    return _mm256_dpbusd_avx_epi32(acc, aw, _mm256_sign_epi8(x, w));
}

AVX512VNNI static inline __m256i dot_avx512vnni(__m256i acc, __m256i aw, __m256i w, __m256i x) {
    return _mm256_dpbusd_epi32(acc, aw, _mm256_sign_epi8(x, w));
}

// 8 inputs, repeated for each of the 4 rows of a load
AVX2 static inline __m256i repeat8(const int8_t *x) {
    int64_t v;
    memcpy(&v, x, 8);
    return _mm256_set1_epi64x(v);
}

// Lanes 2r and 2r+1 of a sum belong to row r: give each its row's scale.
AVX2 static inline __m256 row_scales(const f16 *d) {
    __m128 s = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)d));
    return _mm256_permutevar8x32_ps(_mm256_castps128_ps256(s), _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3));
}

// out[r] = lane 2r + lane 2r+1
AVX2 static inline void store_rows(float *out, __m256 acc) {
    __m256 s = _mm256_hadd_ps(acc, acc); // r0 r1 r0 r1 | r2 r3 r2 r3
    _mm_storeu_ps(out, _mm_movelh_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1)));
}

// One group (4 rows) times one token.
AVX2 static inline __attribute__((always_inline)) void x1(float *out, const QMatrix *m, int g, const int8_t *x,
                                                          const float *xd, Dot *dot) {
    int blocks = m->cols / BLOCK;
    const int8_t *q = m->q + (size_t)g * blocks * 128;
    const f16 *d = m->d + (size_t)g * blocks * 4;
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < blocks; b++, q += 128, d += 4, x += BLOCK) {
        __m256i s = _mm256_setzero_si256();
        for (int c = 0; c < 4; c++) {
            __m256i w = _mm256_loadu_si256((const __m256i *)(q + 32 * c));
            s = dot(s, _mm256_abs_epi8(w), w, repeat8(x + 8 * c));
        }
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_mul_ps(row_scales(d), _mm256_set1_ps(xd[b])), acc);
    }
    store_rows(out, acc);
}

// Two groups (8 rows) times four tokens: each weight load feeds four tokens.
AVX2 static inline __attribute__((always_inline)) void x4(float *out, int ldo, const QMatrix *m, int g,
                                                          const int8_t *x, const float *xd, Dot *dot) {
    int blocks = m->cols / BLOCK, cols = m->cols;
    for (int h = 0; h < 2; h++, out += 4) {
        const int8_t *q = m->q + (size_t)(g + h) * blocks * 128;
        const f16 *d = m->d + (size_t)(g + h) * blocks * 4;
        __m256 acc[4];
        for (int t = 0; t < 4; t++) acc[t] = _mm256_setzero_ps();
        for (int b = 0; b < blocks; b++, q += 128, d += 4) {
            __m256i s[4];
            for (int t = 0; t < 4; t++) s[t] = _mm256_setzero_si256();
            for (int c = 0; c < 4; c++) {
                __m256i w = _mm256_loadu_si256((const __m256i *)(q + 32 * c)), aw = _mm256_abs_epi8(w);
                for (int t = 0; t < 4; t++) s[t] = dot(s[t], aw, w, repeat8(x + (size_t)t * cols + b * BLOCK + 8 * c));
            }
            for (int t = 0; t < 4; t++) {
                __m256 scale = _mm256_mul_ps(row_scales(d), _mm256_set1_ps(xd[t * blocks + b]));
                acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s[t]), scale, acc[t]);
            }
        }
        for (int t = 0; t < 4; t++) store_rows(out + (size_t)t * ldo, acc[t]);
    }
}

// x1 and x4 compiled for one instruction set, with its dot product inlined
#define KERNELS(isa, target)                                                                                         \
    target static void x1_##isa(float *out, const QMatrix *m, int g, const int8_t *x, const float *xd) {            \
        x1(out, m, g, x, xd, dot_##isa);                                                                             \
    }                                                                                                                \
    target static void x4_##isa(float *out, int ldo, const QMatrix *m, int g, const int8_t *x, const float *xd) {   \
        x4(out, ldo, m, g, x, xd, dot_##isa);                                                                        \
    }
KERNELS(avx2, AVX2)
KERNELS(avxvnni, AVXVNNI)
KERNELS(avx512vnni, AVX512VNNI)

void cpu_kernels(Kernels *k) {
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma") || !__builtin_cpu_supports("f16c")) return;
    if (__builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512vl"))
        *k = (Kernels){"avx512vnni", x1_avx512vnni, x4_avx512vnni};
    else if (__builtin_cpu_supports("avxvnni"))
        *k = (Kernels){"avxvnni", x1_avxvnni, x4_avxvnni};
    else
        *k = (Kernels){"avx2", x1_avx2, x4_avx2};
}
#endif
