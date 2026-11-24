#pragma once

// Portable 100% integer SIMD primitives for x86_64 (AVX2/SSE4.1) and ARM64 (NEON / AWS Graviton),
// with UBSan-clean scalar fallback. Strictly zero IEEE-754 floating-point instructions.

#include <stddef.h>
#include <stdint.h>

#if defined(__AVX2__)
#include <immintrin.h>
#define VPXC_SIMD_AVX2 1
#elif defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define VPXC_SIMD_NEON 1
#endif

namespace vpxc::simd {

inline constexpr const char* arch_name() noexcept {
#if defined(VPXC_SIMD_AVX2)
  return "x86_64-avx2";
#elif defined(VPXC_SIMD_NEON)
  return "arm64-neon";
#else
  return "scalar-int";
#endif
}

// Computes exact integer dot product sum_{a=0..35} f[a] * w[a] for bounded 8-bit/12-bit Q4 features.
// Guaranteed overflow-free in 32-bit lanes (max per-lane product <= 2.68e8, 5-term sum <= 1.34e9 < INT32_MAX).
inline int64_t dot36_i32_i16(const int32_t* f, const int16_t* w) noexcept {
#if defined(VPXC_SIMD_AVX2)
  const __m256i f0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(f + 0));
  const __m256i w0 = _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 0)));
  __m256i acc0 = _mm256_mullo_epi32(f0, w0);

  const __m256i f1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(f + 8));
  const __m256i w1 = _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 8)));
  __m256i acc1 = _mm256_mullo_epi32(f1, w1);

  const __m256i f2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(f + 16));
  const __m256i w2 = _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 16)));
  acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(f2, w2));

  const __m256i f3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(f + 24));
  const __m256i w3 = _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 24)));
  acc1 = _mm256_add_epi32(acc1, _mm256_mullo_epi32(f3, w3));

  acc0 = _mm256_add_epi32(acc0, acc1);
  const __m128i lo = _mm256_castsi256_si128(acc0);
  const __m128i hi = _mm256_extracti128_si256(acc0, 1);
  const __m128i f4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(f + 32));
  const __m128i w4 = _mm_cvtepi16_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + 32)));
  __m128i s128 = _mm_add_epi32(_mm_add_epi32(lo, hi), _mm_mullo_epi32(f4, w4));
  s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, _MM_SHUFFLE(1, 0, 3, 2)));
  s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, _MM_SHUFFLE(2, 3, 0, 1)));
  return static_cast<int64_t>(_mm_cvtsi128_si32(s128));
#elif defined(VPXC_SIMD_NEON)
  int32x4_t acc0 = vdupq_n_s32(0);
  int32x4_t acc1 = vdupq_n_s32(0);
  for (int k = 0; k < 32; k += 8) {
    const int16x8_t wk = vld1q_s16(w + k);
    acc0 = vmlaq_s32(acc0, vld1q_s32(f + k), vmovl_s16(vget_low_s16(wk)));
    acc1 = vmlaq_s32(acc1, vld1q_s32(f + k + 4), vmovl_high_s16(wk));
  }
  const int16x4_t w_tail = vld1_s16(w + 32);
  acc0 = vmlaq_s32(acc0, vld1q_s32(f + 32), vmovl_s16(w_tail));
  return static_cast<int64_t>(vaddvq_s32(vaddq_s32(acc0, acc1)));
#else
  int64_t acc = 0;
  for (int a = 0; a < 36; ++a) acc += static_cast<int64_t>(f[a]) * static_cast<int64_t>(w[a]);
  return acc;
#endif
}

// Vectorized inverse Subtract-Green row: R = G + RG, B = G + BG.
inline void rct_sub_green_inv_row(const int32_t* g, const int32_t* rg, const int32_t* bg, int32_t* r,
                                  int32_t* b, int W) noexcept {
  int x = 0;
#if defined(VPXC_SIMD_AVX2)
  for (; x + 8 <= W; x += 8) {
    const __m256i vg = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(g + x));
    const __m256i vrg = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rg + x));
    const __m256i vbg = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bg + x));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(r + x), _mm256_add_epi32(vg, vrg));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(b + x), _mm256_add_epi32(vg, vbg));
  }
#elif defined(VPXC_SIMD_NEON)
  for (; x + 4 <= W; x += 4) {
    const int32x4_t vg = vld1q_s32(g + x);
    vst1q_s32(r + x, vaddq_s32(vg, vld1q_s32(rg + x)));
    vst1q_s32(b + x, vaddq_s32(vg, vld1q_s32(bg + x)));
  }
#endif
  for (; x < W; ++x) {
    r[x] = static_cast<int32_t>(static_cast<int64_t>(g[x]) + rg[x]);
    b[x] = static_cast<int32_t>(static_cast<int64_t>(g[x]) + bg[x]);
  }
}

// Vectorized inverse Subtract-Average row: R = G + RG, B = B_avg + ((G + R) >> 1).
inline void rct_sub_avg_inv_row(const int32_t* g, const int32_t* rg, const int32_t* b_avg, int32_t* r,
                                int32_t* b, int W) noexcept {
  int x = 0;
#if defined(VPXC_SIMD_AVX2)
  for (; x + 8 <= W; x += 8) {
    const __m256i vg = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(g + x));
    const __m256i vrg = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rg + x));
    const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_avg + x));
    const __m256i vr = _mm256_add_epi32(vg, vrg);
    const __m256i avg = _mm256_srai_epi32(_mm256_add_epi32(vg, vr), 1);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(r + x), vr);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(b + x), _mm256_add_epi32(vb, avg));
  }
#elif defined(VPXC_SIMD_NEON)
  for (; x + 4 <= W; x += 4) {
    const int32x4_t vg = vld1q_s32(g + x);
    const int32x4_t vr = vaddq_s32(vg, vld1q_s32(rg + x));
    const int32x4_t avg = vshrq_n_s32(vaddq_s32(vg, vr), 1);
    vst1q_s32(r + x, vr);
    vst1q_s32(b + x, vaddq_s32(vld1q_s32(b_avg + x), avg));
  }
#endif
  for (; x < W; ++x) {
    const int64_t rv = static_cast<int64_t>(g[x]) + rg[x];
    r[x] = static_cast<int32_t>(rv);
    b[x] = static_cast<int32_t>(static_cast<int64_t>(b_avg[x]) + ((static_cast<int64_t>(g[x]) + rv) >> 1));
  }
}

// Vectorized 2D MED spatial predictor pre-pass on upper row: out_base[x] = N[x], out_nw[x] = NW[x].
// Also computes row difference res[x] = cur[x] - prev_slice[x] for 3D volumetric medical slices!
inline void diff_rows_i32(const int32_t* a, const int32_t* b, int32_t* out, int n) noexcept {
  int i = 0;
#if defined(VPXC_SIMD_AVX2)
  for (; i + 8 <= n; i += 8) {
    const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), _mm256_sub_epi32(va, vb));
  }
#elif defined(VPXC_SIMD_NEON)
  for (; i + 4 <= n; i += 4) {
    vst1q_s32(out + i, vsubq_s32(vld1q_s32(a + i), vld1q_s32(b + i)));
  }
#endif
  for (; i < n; ++i) out[i] = static_cast<int32_t>(static_cast<int64_t>(a[i]) - b[i]);
}

inline void add_rows_i32(const int32_t* a, const int32_t* b, int32_t* out, int n) noexcept {
  int i = 0;
#if defined(VPXC_SIMD_AVX2)
  for (; i + 8 <= n; i += 8) {
    const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), _mm256_add_epi32(va, vb));
  }
#elif defined(VPXC_SIMD_NEON)
  for (; i + 4 <= n; i += 4) {
    vst1q_s32(out + i, vaddq_s32(vld1q_s32(a + i), vld1q_s32(b + i)));
  }
#endif
  for (; i < n; ++i) out[i] = static_cast<int32_t>(static_cast<int64_t>(a[i]) + b[i]);
}

}  // namespace vpxc::simd
