// Offline cross-validated prior trainer for the Frontier lossless codec.
//
// Fits strictly out-of-sample regularized priors on a training directory of raw/PNG
// images and validates hyperparameter choices on a disjoint validation directory:
//   1. Separate Luma & Chroma 8-expert WP prior weights,
//   2. 8-fold image-grouped cross-validated Q8 sector linear predictors (with t-stat
//      sign-stability shrinkage so collinear/noisy taps shrink cleanly to 0),
//   3. Causal integer NLMS online adaptation step sizes,
//   4. Out-of-fold pruned Universal MA Context Trees (eliminating online tree search
//      at fast effort and serving as an anti-overfitting baseline at higher efforts).
//
// Usage:
//   g++ -std=c++20 -O3 -march=native -fopenmp training/train_frontier.cpp \
//       -Icpp/include cpp/build/libvpxc.a -lz -o /tmp/train_frontier
//   /tmp/train_frontier /tmp/datasets/train /tmp/datasets/val cpp/include/vpxc/frontier_priors.hpp

#include "vpxc/color.hpp"
#include "vpxc/entropy.hpp"
#include "vpxc/image.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using namespace vpxc;

namespace {

constexpr int kModeSubAvg = 5;
constexpr int kSubAvgPerms[6][3] = {
    {1, 0, 2}, {1, 2, 0}, {0, 1, 2}, {0, 2, 1}, {2, 1, 0}, {2, 0, 1},
};

constexpr uint32_t kDiv[65] = {
    16777216, 16777216, 8388608, 5592405, 4194304, 3355443, 2796202, 2396745, 2097152, 1864135, 1677721,
    1525201,  1398101,  1290555, 1198372, 1118481, 1048576, 986895,  932067,  883011,  838860,  798915,
    762600,   729444,   699050,  671088,  645277,  621378,  599186,  578524,  559240,  541200,  524288,
    508400,   493447,   479349,  466033,  453438,  441505,  430185,  419430,  409200,  399457,  390167,
    381300,   372827,   364722,  356962,  349525,  342392,  335544,  328965,  322638,  316551,  310689,
    305040,   299593,   294337,  289262,  284359,  279620,  275036,  270600,  266305,  262144,
};

struct FastTables {
  uint16_t bilat[11][96];
  uint32_t recip[6145];
  uint16_t patch_wt[20];
  uint32_t patch_recip[260];
};

constexpr FastTables make_fast_tables() {
  FastTables t{};
  for (int d2 = 1; d2 <= 10; ++d2) {
    for (int dg = 0; dg < 96; ++dg) {
      int w = 8192 / ((dg * dg + 8) * d2);
      t.bilat[d2][dg] = static_cast<uint16_t>(w < 1 ? 1 : w);
    }
  }
  t.recip[0] = 0;
  for (int d = 1; d <= 6144; ++d) {
    t.recip[d] = static_cast<uint32_t>(((1ULL << 24) + static_cast<uint64_t>(d >> 1)) / static_cast<uint64_t>(d));
  }
  for (int s = 0; s < 20; ++s) {
    t.patch_wt[s] = static_cast<uint16_t>(256 / (s * s + 4));
  }
  t.patch_recip[0] = 0;
  for (int d = 1; d < 260; ++d) {
    t.patch_recip[d] = static_cast<uint32_t>(((1ULL << 20) + static_cast<uint64_t>(d >> 1)) / static_cast<uint64_t>(d));
  }
  return t;
}

constexpr FastTables kFast = make_fast_tables();

inline int flog2_u32(uint32_t x) { return x ? (31 - __builtin_clz(x)) : 0; }
inline int flog2_u64(uint64_t x) { return x ? (63 - __builtin_clzll(x)) : 0; }
inline int32_t iabs32(int32_t v) { return v >= 0 ? v : -v; }
inline int64_t iabs64(int64_t v) { return v >= 0 ? v : -v; }

inline int64_t mul_recip24(int64_t num, int bd) {
  const uint32_t r = kFast.recip[bd];
  if (num >= 0) return static_cast<int64_t>((static_cast<uint64_t>(num) * r + (1ULL << 23)) >> 24);
  return -static_cast<int64_t>((static_cast<uint64_t>(-num) * r + (1ULL << 23)) >> 24);
}

inline int64_t mul_patch_recip20(int64_t num, int bd) {
  const uint32_t r = kFast.patch_recip[bd];
  if (num >= 0) return static_cast<int64_t>((static_cast<uint64_t>(num) * r + (1ULL << 19)) >> 20);
  return -static_cast<int64_t>((static_cast<uint64_t>(-num) * r + (1ULL << 19)) >> 20);
}

uint32_t wp_wt(uint64_t x, uint32_t maxw) {
  int shift = flog2_u64(x + 1) - 5;
  if (shift < 0) shift = 0;
  uint64_t idx = x >> shift;
  if (idx >= 64) return 4;
  return 4u + static_cast<uint32_t>((static_cast<uint64_t>(maxw) * kDiv[idx + 1]) >> shift);
}

inline int clamp_same_parity(int c, int len) {
  if (c >= 0 && c < len) return c;
  if (len <= 1) return 0;
  int p = c & 1;
  if (c < 0) return p < len ? p : 0;
  int last = len - 1;
  if ((last & 1) != p) --last;
  return last >= 0 ? last : 0;
}

std::pair<int64_t, int64_t> pixel_bounds_q4(int rct_mode, int c_idx, int maxv, const Plane* C0, const Plane* C1,
                                            int y, int x) {
  if (rct_mode == MODE_IDENTITY || c_idx == 3) return {0, static_cast<int64_t>(maxv) << 4};
  if (rct_mode == MODE_SUB) {
    if (c_idx == 0) return {0, static_cast<int64_t>(maxv) << 4};
    if (c_idx <= 2 && C0 != nullptr) {
      const int64_t a = C0->at(y, x);
      return {-a << 4, (static_cast<int64_t>(maxv) - a) << 4};
    }
  }
  if (rct_mode == kModeSubAvg) {
    if (c_idx == 0) return {0, static_cast<int64_t>(maxv) << 4};
    if (c_idx == 1 && C0 != nullptr) {
      const int64_t a = C0->at(y, x);
      return {-a << 4, (static_cast<int64_t>(maxv) - a) << 4};
    }
    if (c_idx == 2 && C0 != nullptr && C1 != nullptr) {
      const int64_t a = C0->at(y, x);
      const int64_t b = a + C1->at(y, x);
      const int64_t mid = (a + b) >> 1;
      return {-mid << 4, (static_cast<int64_t>(maxv) - mid) << 4};
    }
  }
  return {-static_cast<int64_t>(maxv * 4) << 4, static_cast<int64_t>(maxv * 4) << 4};
}

struct FastPatchPredictor {
  static constexpr int kBuckets = 8192;
  std::array<uint32_t, kBuckets * 2> table{};
  std::array<uint8_t, kBuckets> ptr{};
  int qsh = 0;
  bool enabled = true;

  FastPatchPredictor(int maxv, bool en) : qsh(maxv > 255 ? 8 : 0), enabled(en) {}

  uint32_t hash_key(int32_t d1, int32_t d2, int32_t d3, int32_t d4) const {
    const int sh = qsh + 1;
    const uint32_t h = (static_cast<uint32_t>(d1 >> sh) * 73856093u) ^
                       (static_cast<uint32_t>(d2 >> sh) * 19349663u) ^
                       (static_cast<uint32_t>(d3 >> sh) * 83492791u) ^
                       (static_cast<uint32_t>(d4 >> sh) * 2654435761u);
    return (h ^ (h >> 13)) & (kBuckets - 1);
  }

  std::pair<int64_t, int> query(const Plane& P, int y, int x, int32_t Wv, int32_t Nv, int32_t NW, int32_t NE,
                                int32_t WW, int32_t NN, int64_t fallback_q4) const {
    if (!enabled || y < 2 || x < 2 || x + 1 >= P.width) return {fallback_q4, 0};
    const int32_t d1 = Wv - NW;
    const int32_t d2 = Nv - NW;
    const int32_t d3 = NE - Nv;
    if (((iabs32(d1) + iabs32(d2) + iabs32(d3)) >> qsh) < 6) return {fallback_q4, 0};
    const int32_t d4 = WW - Wv;
    const uint32_t h = hash_key(d1, d2, d3, d4);
    int64_t best_num = 0;
    int best_den = 0;
    int best_sad = 999999;
    const int W = P.width;
    const int32_t* s = P.samples.data();
    for (int k = 0; k < 2; ++k) {
      const uint32_t packed = table[h * 2 + static_cast<uint32_t>(k)];
      if (packed == 0) continue;
      const int my = static_cast<int>(packed >> 16);
      const int mx = static_cast<int>(packed & 0xffffu);
      if (std::abs(my - y) + std::abs(mx - x) > 192) continue;
      const int row_m = my * W;
      const int row_m1 = (my - 1) * W;
      const int32_t mNW = s[row_m1 + mx - 1];
      const int32_t mW = s[row_m + mx - 1];
      const int32_t mN = s[row_m1 + mx];
      const int sad = (iabs32((mW - mNW) - d1) +
                       iabs32((mN - mNW) - d2) +
                       iabs32((s[row_m1 + mx + 1] - mN) - d3) +
                       iabs32((s[row_m + mx - 2] - mW) - d4) +
                       iabs32((s[(my - 2) * W + mx] - mN) - (NN - Nv))) >>
                      qsh;
      if (sad < 20) {
        const int wt = kFast.patch_wt[sad];
        const int64_t cand = static_cast<int64_t>(NW) + (static_cast<int64_t>(s[row_m + mx]) - mNW);
        best_num += static_cast<int64_t>(wt) * (cand << 4);
        best_den += wt;
        if (sad < best_sad) best_sad = sad;
      }
    }
    if (best_den == 0) return {fallback_q4, 0};
    const int conf = best_sad == 0 ? 15 : (best_sad <= 2 ? 10 : (best_sad <= 6 ? 5 : 2));
    return {mul_patch_recip20(best_num, best_den), conf};
  }

  void insert(const Plane& P, int y, int x, int32_t Wv, int32_t Nv, int32_t NW, int32_t NE, int32_t WW) {
    if (!enabled || y < 2 || x < 2 || x + 1 >= P.width || y > 32767 || x > 32767) return;
    const int32_t d1 = Wv - NW;
    const int32_t d2 = Nv - NW;
    const int32_t d3 = NE - Nv;
    if (((iabs32(d1) + iabs32(d2) + iabs32(d3)) >> qsh) < 6) return;
    const uint32_t h = hash_key(d1, d2, d3, WW - Wv);
    uint8_t& p = ptr[h];
    table[h * 2 + (p & 1u)] = (static_cast<uint32_t>(y) << 16) | static_cast<uint32_t>(x);
    ++p;
  }
};

struct FastRasterState {
  static constexpr int KEXP = 8;
  int H, W, c_idx, rct_mode, maxv, qsh, D, n_sec, stride;
  std::vector<uint32_t> perr[KEXP];
  std::vector<int32_t> wp_err;
  Plane true_err;
  FastPatchPredictor patcher;
  uint32_t mw[KEXP];

  FastRasterState(int h, int w, int c, int rm, int mv, bool has_c0, bool has_c1, bool use_patch,
                  const uint32_t* custom_mw = nullptr)
      : H(h),
        W(w),
        c_idx(c),
        rct_mode(rm),
        maxv(mv),
        qsh(mv > 255 ? 8 : 0),
        D(!has_c0 ? 21 : (!has_c1 ? 30 : 36)),
        n_sec(16),
        stride(w + 2),
        wp_err(static_cast<size_t>((w + 2) * 2), 0),
        true_err(h, w, 0),
        patcher(mv, use_patch) {
    const uint32_t def_mw[KEXP] = {0xd, 0xc, 0xc, 0xb, 0xc, 0xa, 0xa, 0xb};
    for (int k = 0; k < KEXP; ++k) {
      mw[k] = custom_mw ? custom_mw[k] : def_mw[k];
      perr[k].assign(static_cast<size_t>(stride * 2), 0);
    }
  }

  int64_t predict_features(const Plane& P, const Plane* C0, const Plane* C0_err, const Plane* C1,
                           const Plane* C1_err, int y, int x, int32_t* f, int& sec, int16_t* prp,
                           int64_t pr_out[KEXP]) const {
    const int cur = (y & 1) ? 0 : stride;
    const int prv = (y & 1) ? stride : 0;
    const int32_t* ps = P.samples.data();
    const int row = y * W;
    const int row1 = (y - 1) * W;
    const int row2 = (y - 2) * W;

    const int32_t Nv = y > 0 ? ps[row1 + x] : (x > 0 ? ps[row + x - 1] : 0);
    const int32_t Wv = x > 0 ? ps[row + x - 1] : Nv;
    const int32_t NE = y > 0 ? ps[row1 + (x + 1 < W ? x + 1 : W - 1)] : Nv;
    const int32_t NW = (y > 0 && x > 0) ? ps[row1 + x - 1] : Nv;
    const int32_t NN = y > 1 ? ps[row2 + x] : Nv;
    const int32_t WW = x > 1 ? ps[row + x - 2] : Wv;
    const int32_t NWW = (y > 0 && x > 1) ? ps[row1 + x - 2] : Wv;
    const int32_t NNW = (y > 1 && x > 0) ? ps[row2 + x - 1] : Nv;
    const int32_t NNE = (y > 1 && x + 1 < W) ? ps[row2 + x + 1] : NE;
    const int32_t NEE = (y > 0 && x + 2 < W) ? ps[row1 + x + 2] : NE;

    const int32_t hi = std::max(Wv, Nv);
    const int32_t lo = std::min(Wv, Nv);
    const int32_t med = (NW < lo) ? hi : ((NW > hi) ? lo : (Wv + Nv - NW));
    const int32_t dh_g = iabs32(Wv - WW) + iabs32(Nv - NW) + iabs32(Nv - NE);
    const int32_t dv_g = iabs32(Wv - NW) + iabs32(Nv - NN) + iabs32(NE - NNE);
    const int32_t gap_thr = 80 << qsh;
    const int32_t gap = (dv_g - dh_g > gap_thr)
                            ? Wv
                            : ((dh_g - dv_g > gap_thr) ? Nv : (((Wv + Nv) >> 1) + ((NE - NW) >> 2)));
    const int32_t planar = (2 * Wv + 2 * Nv - NW + NE + 2) >> 2;

    // Precompute bilateral CfL when C0 is available so Expert 7 can also leverage it!
    int64_t bilat_q4 = static_cast<int64_t>(med) << 4;
    int32_t g0 = 0, gN = 0, gS = 0, gW = 0, gE = 0, gNW = 0, gNE = 0, gSW = 0, gSE = 0;
    if (C0 != nullptr) {
      const int32_t* gs = C0->samples.data();
      const int ym1 = y > 0 ? (y - 1) * W : 0;
      const int yp1 = (y + 1 < H ? y + 1 : H - 1) * W;
      const int xm1 = x > 0 ? x - 1 : 0;
      const int xp1 = x + 1 < W ? x + 1 : W - 1;
      const int xm2 = x > 1 ? x - 2 : 0;
      const int ym2 = y > 1 ? (y - 2) * W : 0;
      g0 = gs[row + x];
      gW = gs[row + xm1];
      gE = gs[row + xp1];
      gN = gs[ym1 + x];
      gS = gs[yp1 + x];
      gNW = gs[ym1 + xm1];
      gNE = gs[ym1 + xp1];
      gSW = gs[yp1 + xm1];
      gSE = gs[yp1 + xp1];
      const int32_t nbp[6] = {Wv, Nv, NW, NE, WW, NN};
      const int32_t nbg[6] = {gW, gN, gNW, gNE, gs[row + xm2], gs[ym2 + x]};
      const int nbd[6] = {1, 1, 2, 2, 3, 3};
      int64_t bn = 0;
      int bd = 0;
      if (C1 == nullptr) {
        for (int m = 0; m < 6; ++m) {
          const int dg = std::min<int>(iabs32(g0 - nbg[m]) >> qsh, 95);
          const int wt = kFast.bilat[nbd[m]][dg];
          bn += static_cast<int64_t>(wt) * (static_cast<int64_t>(nbp[m]) << 4);
          bd += wt;
        }
      } else {
        const int32_t* us = C1->samples.data();
        const int32_t u0 = us[row + x];
        const int32_t nbu[6] = {us[row + xm1], us[ym1 + x],   us[ym1 + xm1],
                                us[ym1 + xp1], us[row + xm2], us[ym2 + x]};
        for (int m = 0; m < 6; ++m) {
          const int dg = std::min<int>((iabs32(g0 - nbg[m]) + iabs32(u0 - nbu[m])) >> qsh, 95);
          const int wt = kFast.bilat[nbd[m]][dg];
          bn += static_cast<int64_t>(wt) * (static_cast<int64_t>(nbp[m]) << 4);
          bd += wt;
        }
      }
      bilat_q4 = mul_recip24(bn, bd);
    }

    const int pN = prv + x;
    const int pNE = x < W - 1 ? pN + 1 : pN;
    const int pNW = x > 0 ? pN - 1 : pN;
    const int pW = x > 0 ? cur + x - 1 : pN;
    const int64_t teW = x == 0 ? 0 : wp_err[static_cast<size_t>(cur + x - 1)];
    const int64_t teN = wp_err[static_cast<size_t>(pN)];
    const int64_t teNW = wp_err[static_cast<size_t>(pNW)];
    const int64_t teNE = wp_err[static_cast<size_t>(pNE)];
    const int64_t n = static_cast<int64_t>(Nv) << 4;
    const int64_t wv = static_cast<int64_t>(Wv) << 4;
    const int64_t ne = static_cast<int64_t>(NE) << 4;
    const int64_t nw = static_cast<int64_t>(NW) << 4;
    const int64_t nn = static_cast<int64_t>(NN) << 4;

    const auto [patch_q4, patch_conf] = patcher.query(P, y, x, Wv, Nv, NW, NE, WW, NN, med << 4);
    const int64_t lo3 = std::min({wv, ne, n});
    const int64_t hi3 = std::max({wv, ne, n});
    const int64_t trend_q4 = std::clamp<int64_t>(
        (5 * (wv + n) - 2 * nw - (static_cast<int64_t>(WW) << 4) - nn + ne + 2) >> 2, lo3, hi3);

    pr_out[0] = wv + ne - n;
    pr_out[1] = n - (((teN + teW + teNE) * 8) >> 5);
    pr_out[2] = wv - (((teN + teW + teNW) * 8) >> 5);
    pr_out[3] = n - ((teNW * 4 + teNE * 3 + (nn - n) * 23 + (nw - wv) * 2) >> 5);
    pr_out[4] = static_cast<int64_t>(med) << 4;
    pr_out[5] = static_cast<int64_t>(gap) << 4;
    pr_out[6] = static_cast<int64_t>(planar) << 4;
    pr_out[7] = patch_conf > 0 ? patch_q4 : (C0 != nullptr ? bilat_q4 : trend_q4);

    uint32_t wts[KEXP];
    for (int k = 0; k < KEXP; ++k) {
      const uint64_t acc = uint64_t(perr[k][static_cast<size_t>(pN)]) +
                           uint64_t(perr[k][static_cast<size_t>(pNE)]) +
                           uint64_t(perr[k][static_cast<size_t>(pNW)]) +
                           (x > 0 ? uint64_t(perr[k][static_cast<size_t>(pW)]) : 0ull);
      wts[k] = wp_wt(acc >> qsh, mw[k]);
    }

    uint32_t sw = 0;
    for (int k = 0; k < KEXP; ++k) sw += wts[k];
    if (sw < 16) sw = 16;
    const int lw = flog2_u32(sw);
    sw = 0;
    for (int k = 0; k < KEXP; ++k) {
      if (lw >= 4) wts[k] >>= (lw - 4);
      sw += wts[k];
    }
    if (sw == 0) sw = 1;
    else if (sw > 64) sw = 64;
    int64_t s = static_cast<int64_t>(sw >> 1) - 1;
    for (int k = 0; k < KEXP; ++k) s += pr_out[k] * static_cast<int64_t>(wts[k]);
    int64_t pred_q4 = (s * static_cast<int64_t>(kDiv[sw])) >> 24;
    if (((teN ^ teW) | (teN ^ teNW)) <= 0 && patch_conf < 10 && C0 == nullptr) {
      pred_q4 = std::clamp(pred_q4, lo3, hi3);
    }

    const int32_t* es = true_err.samples.data();
    const int32_t reW = x > 0 ? es[row + x - 1] : 0;
    const int32_t reN = y > 0 ? es[row1 + x] : 0;
    const int32_t reNW = (y > 0 && x > 0) ? es[row1 + x - 1] : 0;
    const int32_t reNE = (y > 0 && x + 1 < W) ? es[row1 + x + 1] : 0;
    const int32_t reWW = x > 1 ? es[row + x - 2] : 0;
    const int32_t reNN = y > 1 ? es[row2 + x] : 0;

    const int32_t pq4 = static_cast<int32_t>(pred_q4);
    int k = 0;
    f[k++] = static_cast<int32_t>(wv) - pq4;
    f[k++] = static_cast<int32_t>(n) - pq4;
    f[k++] = static_cast<int32_t>(nw) - pq4;
    f[k++] = static_cast<int32_t>(ne) - pq4;
    f[k++] = (WW << 4) - pq4;
    f[k++] = static_cast<int32_t>(nn) - pq4;
    f[k++] = (NWW << 4) - pq4;
    f[k++] = (NNW << 4) - pq4;
    f[k++] = (NNE << 4) - pq4;
    f[k++] = (NEE << 4) - pq4;
    f[k++] = (med << 4) - pq4;
    f[k++] = (gap << 4) - pq4;
    f[k++] = (planar << 4) - pq4;
    f[k++] = static_cast<int32_t>(pr_out[0]) - pq4;
    f[k++] = patch_conf > 0 ? (static_cast<int32_t>(patch_q4) - pq4) : 0;
    f[k++] = reW;
    f[k++] = reN;
    f[k++] = reNW;
    f[k++] = reNE;
    f[k++] = reWW;
    f[k++] = reNN;

    int32_t bilat_diff = 0;
    if (C0 == nullptr) {
      const int32_t d45 = iabs32(Wv - Nv) + iabs32(NW - NN);
      const int32_t d135 = iabs32(Nv - NE) + iabs32(Wv - NW);
      const int dir = (dh_g * 2 < dv_g) ? 0 : ((dv_g * 2 < dh_g) ? 1 : ((d45 < d135) ? 2 : 3));
      const int32_t gsum = (dh_g + dv_g) >> qsh;
      const int con = (gsum < 16) ? 0 : ((gsum < 48) ? 1 : ((gsum < 128) ? 2 : 3));
      sec = con * 4 + dir;
    } else {
      bilat_diff = static_cast<int32_t>(bilat_q4) - pq4;
      f[k++] = bilat_diff;
      f[k++] = (g0 - gW) << 4;
      f[k++] = (g0 - gN) << 4;
      f[k++] = (2 * g0 - gW - gE) << 3;
      f[k++] = (2 * g0 - gN - gS) << 3;
      f[k++] = (4 * g0 - gNW - gNE - gSW - gSE) << 2;
      f[k++] = (gN - gS) << 3;
      f[k++] = (gW - gE) << 3;
      f[k++] = C0_err->samples[row + x];
      if (C1 != nullptr) {
        const int32_t* us = C1->samples.data();
        const int ym1 = y > 0 ? (y - 1) * W : 0;
        const int yp1 = (y + 1 < H ? y + 1 : H - 1) * W;
        const int xm1 = x > 0 ? x - 1 : 0;
        const int xp1 = x + 1 < W ? x + 1 : W - 1;
        const int32_t u0 = us[row + x];
        const int32_t uN = us[ym1 + x];
        const int32_t uS = us[yp1 + x];
        const int32_t uW = us[row + xm1];
        const int32_t uE = us[row + xp1];
        f[k++] = (u0 - uW) << 4;
        f[k++] = (u0 - uN) << 4;
        f[k++] = (2 * u0 - uW - uE) << 3;
        f[k++] = (2 * u0 - uN - uS) << 3;
        f[k++] = (uN - uS) << 3;
        f[k++] = C1_err->samples[row + x];
      }
      const int phase = ((y & 1) << 1) | (x & 1);
      const int32_t gdh = iabs32(gW - gE) >> qsh;
      const int32_t gdv = iabs32(gN - gS) >> qsh;
      const int ori = (gdh + gdv < 16) ? 0 : ((gdh > gdv * 2) ? 1 : ((gdv > gdh * 2) ? 2 : 3));
      sec = phase * 4 + ori;
    }

    if (prp != nullptr) {
      const uint32_t ew = static_cast<uint32_t>(iabs32(reW));
      const uint32_t en = static_cast<uint32_t>(iabs32(reN));
      const uint32_t enw = static_cast<uint32_t>(iabs32(reNW));
      const uint32_t ene = static_cast<uint32_t>(iabs32(reNE));
      const uint32_t eww = static_cast<uint32_t>(iabs32(reWW));
      const uint32_t enn = static_cast<uint32_t>(iabs32(reNN));
      prp[0] = static_cast<int16_t>(flog2_u32(2 * (ew + en) + enw + ene + ((eww + enn) >> 1) + 1));
      prp[1] = static_cast<int16_t>(flog2_u32(ew + 1));
      prp[2] = static_cast<int16_t>(flog2_u32(en + 1));
      prp[3] = static_cast<int16_t>(flog2_u32(enw + ene + 1));
      prp[4] = static_cast<int16_t>(flog2_u32(static_cast<uint32_t>(iabs32(f[0])) + 1));
      prp[5] = static_cast<int16_t>(flog2_u32(static_cast<uint32_t>(iabs32(f[1])) + 1));
      prp[6] = static_cast<int16_t>(flog2_u32(static_cast<uint32_t>(iabs32(f[0] - f[1])) + 1));
      prp[7] = static_cast<int16_t>(flog2_u32(static_cast<uint32_t>(iabs32(f[10])) + 1));
      prp[8] = static_cast<int16_t>(sec);
      prp[9] = static_cast<int16_t>((reW >= 0 ? 1 : -1) * flog2_u32(ew + 1));
      prp[10] = static_cast<int16_t>((reN >= 0 ? 1 : -1) * flog2_u32(en + 1));
      prp[11] = static_cast<int16_t>(patch_conf);
      prp[12] = static_cast<int16_t>(
          C0 != nullptr ? flog2_u32(static_cast<uint32_t>(iabs32(C0_err->samples[row + x])) + 1)
                        : (((y & 1) << 1) | (x & 1)));
      prp[13] = static_cast<int16_t>(
          C0 != nullptr ? flog2_u32(static_cast<uint32_t>(iabs32(bilat_diff)) + 1)
                        : flog2_u32(static_cast<uint32_t>(dh_g + dv_g) + 1));
    }
    return pred_q4;
  }

  void commit(const Plane& P, int y, int x, int32_t truth, int64_t pred_q4, int64_t mu_q4,
              const int64_t pr_out[KEXP]) {
    const int cur = (y & 1) ? 0 : stride;
    const int prv = (y & 1) ? stride : 0;
    const int64_t val_q4 = static_cast<int64_t>(truth) << 4;
    true_err.samples[y * W + x] = static_cast<int32_t>(std::clamp<int64_t>(val_q4 - mu_q4, -8388608, 8388608));
    wp_err[static_cast<size_t>(cur + x)] = static_cast<int32_t>(pred_q4 - val_q4);
    for (int m = 0; m < KEXP; ++m) {
      const uint64_t e64 = static_cast<uint64_t>((iabs64(pr_out[m] - val_q4) + 8) >> 4);
      const uint32_t e = static_cast<uint32_t>(std::min<uint64_t>(e64, 1u << 28));
      perr[m][static_cast<size_t>(cur + x)] = e;
      perr[m][static_cast<size_t>(prv + x + 1)] += e;
    }
    if (patcher.enabled && y >= 2 && x >= 2 && x + 1 < W) {
      const int32_t* ps = P.samples.data();
      const int row = y * W;
      const int row1 = (y - 1) * W;
      patcher.insert(P, y, x, ps[row + x - 1], ps[row1 + x], ps[row1 + x - 1], ps[row1 + x + 1], ps[row + x - 2]);
    }
  }
};

constexpr int kStageNbs1[12][2] = {
    {-1, -1}, {-1, 1}, {1, -1}, {1, 1}, {-3, -1}, {-3, 1}, {3, -1}, {3, 1}, {-1, -3}, {1, -3}, {-1, 3}, {1, 3}};
constexpr int kStageNbs2[10][2] = {
    {0, -1}, {0, 1}, {-1, 0}, {1, 0}, {0, -3}, {0, 3}, {-2, -1}, {-2, 1}, {2, -1}, {2, 1}};
constexpr int kStageNbs3[10][2] = {
    {-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}, {-3, 0}, {3, 0}};

int64_t fast_interp_features(const Plane& P, const Plane& C0, const Plane& C0_err, const Plane* C1,
                             const Plane* C1_err, const Plane& stage_err, int maxv, int stg, int y, int x,
                             int32_t* f, int& sec, int16_t* prp) {
  const int H = P.height;
  const int W = P.width;
  const int qsh = maxv > 255 ? 8 : 0;
  const auto* nbs = stg == 1 ? kStageNbs1 : (stg == 2 ? kStageNbs2 : kStageNbs3);
  const int Knb = stg == 1 ? 12 : 10;
  const int32_t* ps = P.samples.data();
  const int32_t* gs = C0.samples.data();

  int32_t nb_vals[12];
  int ny_arr[8], nx_arr[8];
  for (int k = 0; k < Knb; ++k) {
    const int ny = clamp_same_parity(y + nbs[k][0], H);
    const int nx = clamp_same_parity(x + nbs[k][1], W);
    if (k < 8) {
      ny_arr[k] = ny;
      nx_arr[k] = nx;
    }
    nb_vals[k] = ps[ny * W + nx];
  }
  const int32_t b_q4 = (nb_vals[0] + nb_vals[1] + nb_vals[2] + nb_vals[3]) << 2;

  const int32_t g0 = gs[y * W + x];
  int64_t bn = 0;
  int bd = 0;
  for (int k = 0; k < 8; ++k) {
    const int idx = ny_arr[k] * W + nx_arr[k];
    int dg = iabs32(g0 - gs[idx]) >> qsh;
    if (C1 != nullptr) dg += iabs32(C1->samples[y * W + x] - C1->samples[idx]) >> qsh;
    dg = std::min(dg, 95);
    const int dist2 = nbs[k][0] * nbs[k][0] + nbs[k][1] * nbs[k][1];
    const int wt = kFast.bilat[dist2][dg];
    bn += static_cast<int64_t>(wt) * (static_cast<int64_t>(nb_vals[k]) << 4);
    bd += wt;
  }
  const int32_t bilat_diff = static_cast<int32_t>(mul_recip24(bn, bd)) - b_q4;

  int m = 0;
  for (int k = 0; k < Knb; ++k) f[m++] = (nb_vals[k] << 4) - b_q4;
  f[m++] = bilat_diff;

  const int ym1 = (y > 0 ? y - 1 : 0) * W;
  const int yp1 = (y + 1 < H ? y + 1 : H - 1) * W;
  const int xm1 = x > 0 ? x - 1 : 0;
  const int xp1 = x + 1 < W ? x + 1 : W - 1;
  const int32_t gN = gs[ym1 + x];
  const int32_t gS = gs[yp1 + x];
  const int32_t gW = gs[y * W + xm1];
  const int32_t gE = gs[y * W + xp1];
  const int32_t gNW = gs[ym1 + xm1];
  const int32_t gNE = gs[ym1 + xp1];
  const int32_t gSW = gs[yp1 + xm1];
  const int32_t gSE = gs[yp1 + xp1];
  f[m++] = (2 * g0 - gW - gE) << 3;
  f[m++] = (2 * g0 - gN - gS) << 3;
  f[m++] = (4 * g0 - gNW - gNE - gSW - gSE) << 2;
  f[m++] = (g0 - ((gN + gS + gW + gE) >> 2)) << 4;
  f[m++] = (gN - gS) << 3;
  f[m++] = (gW - gE) << 3;
  f[m++] = (gNW - gSE) << 3;
  f[m++] = (gNE - gSW) << 3;
  f[m++] = C0_err.samples[y * W + x];
  if (C1 != nullptr) {
    const int32_t* us = C1->samples.data();
    const int32_t u0 = us[y * W + x];
    const int32_t uN = us[ym1 + x];
    const int32_t uS = us[yp1 + x];
    const int32_t uW = us[y * W + xm1];
    const int32_t uE = us[y * W + xp1];
    const int32_t uNW = us[ym1 + xm1];
    const int32_t uNE = us[ym1 + xp1];
    const int32_t uSW = us[yp1 + xm1];
    const int32_t uSE = us[yp1 + xp1];
    f[m++] = (2 * u0 - uW - uE) << 3;
    f[m++] = (2 * u0 - uN - uS) << 3;
    f[m++] = (4 * u0 - uNW - uNE - uSW - uSE) << 2;
    f[m++] = (u0 - ((uN + uS + uW + uE) >> 2)) << 4;
    f[m++] = (uN - uS) << 3;
    f[m++] = C1_err->samples[y * W + x];
  }
  const int32_t gdh = (iabs32(gW - gE) + iabs32(2 * g0 - gW - gE)) >> qsh;
  const int32_t gdv = (iabs32(gN - gS) + iabs32(2 * g0 - gN - gS)) >> qsh;
  sec = (gdh + gdv < 12) ? 0 : ((gdh > gdv * 2) ? 1 : ((gdv > gdh * 2) ? 2 : 3));

  if (prp != nullptr) {
    uint32_t prev_e = 0;
    for (int k = 0; k < 4; ++k) prev_e += static_cast<uint32_t>(stage_err.samples[ny_arr[k] * W + nx_arr[k]]);
    const uint32_t causal_same =
        (x >= 2 ? static_cast<uint32_t>(stage_err.samples[y * W + x - 2]) : 0u) +
        (y >= 2 ? static_cast<uint32_t>(stage_err.samples[(y - 2) * W + x]) : 0u);
    uint32_t c_span = 0;
    for (int k = 0; k < 4; ++k) c_span += static_cast<uint32_t>(iabs32(f[k]));
    prp[0] = static_cast<int16_t>(flog2_u32(causal_same + 1));
    prp[1] = static_cast<int16_t>(flog2_u32(prev_e + 1));
    prp[2] = static_cast<int16_t>(flog2_u32(c_span + 1));
    prp[3] = static_cast<int16_t>(flog2_u32(static_cast<uint32_t>(iabs32(bilat_diff)) + 1));
    prp[4] = static_cast<int16_t>(flog2_u32(static_cast<uint32_t>(iabs32(C0_err.samples[y * W + x])) + 1));
    prp[5] = static_cast<int16_t>(flog2_u32(static_cast<uint32_t>(iabs32(gW - gE) + iabs32(gN - gS)) + 1));
    prp[6] = static_cast<int16_t>(sec);
    prp[7] = static_cast<int16_t>((f[0] - f[1] >= 0 ? 1 : -1) *
                                  flog2_u32(static_cast<uint32_t>(iabs32(f[0] - f[1])) + 1));
    prp[8] = static_cast<int16_t>(
        C1 != nullptr ? flog2_u32(static_cast<uint32_t>(iabs32(C1_err->samples[y * W + x])) + 1) : 0);
    prp[9] = static_cast<int16_t>(
        flog2_u32(static_cast<uint32_t>(iabs32(2 * g0 - gW - gE) + iabs32(2 * g0 - gN - gS)) + 1));
    prp[10] = static_cast<int16_t>(flog2_u32(static_cast<uint32_t>(iabs32(f[0]) + iabs32(f[1])) + 1));
  }
  return b_q4;
}

struct RawImg {
  std::string name;
  int H, W;
  std::vector<Plane> rgb;
};

std::vector<RawImg> load_raw_dir(const std::string& dir) {
  std::vector<std::string> paths;
  for (const auto& entry : fs::directory_iterator(dir)) {
    const std::string p = entry.path().string();
    if (p.size() > 4 && p.substr(p.size() - 4) == ".rgb") paths.push_back(p);
  }
  std::sort(paths.begin(), paths.end());
  std::vector<RawImg> out;
  for (const auto& p : paths) {
    const std::string stem = fs::path(p).stem().string();
    const auto u2 = stem.rfind('_');
    const auto u1 = stem.rfind('_', u2 - 1);
    const int H = std::stoi(stem.substr(u1 + 1, u2 - u1 - 1));
    const int W = std::stoi(stem.substr(u2 + 1));
    std::ifstream in(p, std::ios::binary);
    std::vector<uint8_t> buf(static_cast<size_t>(H * W * 3));
    in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    std::vector<Plane> pl(3, Plane(H, W));
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        for (int c = 0; c < 3; ++c) pl[c].at(y, x) = buf[static_cast<size_t>((y * W + x) * 3 + c)];
      }
    }
    out.push_back(RawImg{stem.substr(0, u1), H, W, std::move(pl)});
  }
  return out;
}

bool solve_spd(std::vector<double> A, std::vector<double> b, int D, std::vector<double>& x) {
  for (int k = 0; k < D; ++k) {
    int piv = k;
    for (int i = k + 1; i < D; ++i) {
      if (std::fabs(A[i * D + k]) > std::fabs(A[piv * D + k])) piv = i;
    }
    if (!(std::fabs(A[piv * D + k]) > 1e-9) || !std::isfinite(A[piv * D + k])) return false;
    if (piv != k) {
      for (int j = k; j < D; ++j) std::swap(A[k * D + j], A[piv * D + j]);
      std::swap(b[k], b[piv]);
    }
    for (int i = k + 1; i < D; ++i) {
      const double f = A[i * D + k] / A[k * D + k];
      for (int j = k + 1; j < D; ++j) A[i * D + j] -= f * A[k * D + j];
      b[i] -= f * b[k];
    }
  }
  x.assign(static_cast<size_t>(D), 0.0);
  for (int i = D - 1; i >= 0; --i) {
    double sum = b[i];
    for (int j = i + 1; j < D; ++j) sum -= A[i * D + j] * x[j];
    x[i] = sum / A[i * D + i];
    if (!std::isfinite(x[i])) return false;
  }
  return true;
}

// Roles:
// 0: Luma Raster (16x21)
// 1: Chroma1 Raster (16x30)
// 2: Chroma2 Raster (16x36)
// 3: Chroma1 Coset S1 (4x22), 4: S2 (4x20), 5: S3 (4x20)
// 6: Chroma2 Coset S1 (4x28), 7: S2 (4x26), 8: S3 (4x26)
constexpr int kRoleSec[9] = {16, 16, 16, 4, 4, 4, 4, 4, 4};
constexpr int kRoleDim[9] = {21, 30, 36, 22, 20, 20, 28, 26, 26};

struct RoleGram {
  std::vector<double> G;
  std::vector<double> R;
  double Y2 = 0.0;
  int64_t N = 0;
  void init(int D) {
    G.assign(static_cast<size_t>(D * D), 0.0);
    R.assign(static_cast<size_t>(D), 0.0);
    Y2 = 0.0;
    N = 0;
  }
};

// Cross-validated universal MA tree builder on discretized property histograms
constexpr int kPropMin = -20;
constexpr int kPropBins = 48;  // [-20..+27]
constexpr int kErrBins = 16;   // log2(|resid*16 - (frac-8)| + 1)

struct FlatTreeNode {
  int16_t thr = 0;
  int8_t prop = -1;
  uint8_t leaf_id = 0;
  int16_t left = -1;
  int16_t right = -1;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: train_frontier <train_dir> <val_dir> <out_hpp>\n";
    return 1;
  }
  const auto train_imgs = load_raw_dir(argv[1]);
  const auto val_imgs = load_raw_dir(argv[2]);
  const std::string out_hpp = argv[3];
  std::printf("Loaded %zu train images and %zu validation images.\n", train_imgs.size(), val_imgs.size());

  // -------------------------------------------------------------------------
  // Step 1: Coordinate optimization of Luma & Chroma WP8 expert weights
  // -------------------------------------------------------------------------
  uint32_t mw_luma[8] = {12, 4, 14, 17, 20, 17, 4, 4};
  uint32_t mw_chroma[8] = {17, 4, 7, 17, 20, 4, 4, 20};

  auto eval_wp_loss = [&](const std::vector<RawImg>& corpus, bool is_chroma, const uint32_t mw[8]) -> double {
    double total_l1 = 0.0;
    #pragma omp parallel for reduction(+ : total_l1)
    for (size_t idx = 0; idx < corpus.size(); ++idx) {
      const auto& im = corpus[idx];
      const int H = im.H, W = im.W;
      Plane G = im.rgb[1];
      Plane RG(H, W);
      for (int i = 0; i < H * W; ++i) RG.samples[i] = im.rgb[0].samples[i] - G.samples[i];
      int32_t f[40];
      int sec = 0;
      int64_t pr[8];
      if (!is_chroma) {
        FastRasterState st(H, W, 0, MODE_SUB, 255, false, false, true, mw);
        for (int y = 0; y < H; y += 2) {
          for (int x = 0; x < W; ++x) {
            const int64_t pq4 = st.predict_features(G, nullptr, nullptr, nullptr, nullptr, y, x, f, sec, nullptr, pr);
            const int32_t tr = G.at(y, x);
            total_l1 += std::fabs(static_cast<double>((int64_t(tr) << 4) - pq4));
            st.commit(G, y, x, tr, pq4, pq4, pr);
          }
        }
      } else {
        Plane eG(H, W, 0);
        FastRasterState stG(H, W, 0, MODE_SUB, 255, false, false, true, mw_luma);
        for (int y = 0; y < H; ++y) {
          for (int x = 0; x < W; ++x) {
            const int64_t pq4 = stG.predict_features(G, nullptr, nullptr, nullptr, nullptr, y, x, f, sec, nullptr, pr);
            const int32_t tr = G.at(y, x);
            eG.at(y, x) = static_cast<int32_t>((int64_t(tr) << 4) - pq4);
            stG.commit(G, y, x, tr, pq4, pq4, pr);
          }
        }
        FastRasterState stC(H, W, 1, MODE_SUB, 255, true, false, true, mw);
        for (int y = 0; y < H; y += 2) {
          for (int x = 0; x < W; ++x) {
            const int64_t pq4 = stC.predict_features(RG, &G, &eG, nullptr, nullptr, y, x, f, sec, nullptr, pr);
            const int32_t tr = RG.at(y, x);
            total_l1 += std::fabs(static_cast<double>((int64_t(tr) << 4) - pq4));
            stC.commit(RG, y, x, tr, pq4, pq4, pr);
          }
        }
      }
    }
    return total_l1 / static_cast<double>(corpus.size() * 393216 / 2);
  };

  for (int is_c = 0; is_c < 2; ++is_c) {
    uint32_t* mw = is_c ? mw_chroma : mw_luma;
    const uint32_t init_def[8] = {0xd, 0xc, 0xc, 0xb, 0xc, 0xa, 0xa, 0xb};
    double base_val = eval_wp_loss(val_imgs, is_c != 0, init_def);
    double cur_tr = eval_wp_loss(train_imgs, is_c != 0, mw);
    for (int pass = 0; pass < 1; ++pass) {
      for (int k = 0; k < 8; ++k) {
        uint32_t best_v = mw[k];
        for (uint32_t cand : {4u, 7u, 10u, 12u, 14u, 17u, 20u}) {
          uint32_t tmp[8];
          std::copy(mw, mw + 8, tmp);
          tmp[k] = cand;
          const double l = eval_wp_loss(train_imgs, is_c != 0, tmp);
          if (l < cur_tr) {
            cur_tr = l;
            best_v = cand;
          }
        }
        mw[k] = best_v;
      }
    }
    double new_val = eval_wp_loss(val_imgs, is_c != 0, mw);
    std::printf("WP8 %s Val Q4-MAE: default=%.3f -> trained=%.3f  weights={%u,%u,%u,%u,%u,%u,%u,%u}\n",
                is_c ? "Chroma" : "Luma  ", base_val, new_val,
                mw[0], mw[1], mw[2], mw[3], mw[4], mw[5], mw[6], mw[7]);
    if (new_val > base_val) std::copy(init_def, init_def + 8, mw);
  }

  // -------------------------------------------------------------------------
  // Step 2: 2-Pass Cross-Validated Universal Sector Ridge Priors (9 Roles)
  // -------------------------------------------------------------------------
  std::vector<std::vector<std::array<int16_t, 36>>> w_prior(9);
  for (int r = 0; r < 9; ++r) {
    w_prior[r].assign(static_cast<size_t>(kRoleSec[r]), std::array<int16_t, 36>{});
  }

  for (int iter = 0; iter < 2; ++iter) {
    constexpr int kFolds = 8;
    std::vector<std::vector<std::vector<RoleGram>>> fold_grams(kFolds, std::vector<std::vector<RoleGram>>(9));
    for (int kf = 0; kf < kFolds; ++kf) {
      for (int r = 0; r < 9; ++r) {
        fold_grams[kf][r].resize(static_cast<size_t>(kRoleSec[r]));
        for (int s = 0; s < kRoleSec[r]; ++s) fold_grams[kf][r][s].init(kRoleDim[r]);
      }
    }

    auto accum_sample = [&](RoleGram& rg, int D, const int32_t* f, double yv) {
      rg.N += 1;
      rg.Y2 += yv * yv;
      for (int a = 0; a < D; ++a) {
        const double fa = static_cast<double>(f[a]);
        rg.R[a] += fa * yv;
        for (int b = a; b < D; ++b) rg.G[a * D + b] += fa * static_cast<double>(f[b]);
      }
    };

    #pragma omp parallel
    {
      std::vector<std::vector<std::vector<RoleGram>>> local_fg(kFolds, std::vector<std::vector<RoleGram>>(9));
      for (int kf = 0; kf < kFolds; ++kf) {
        for (int r = 0; r < 9; ++r) {
          local_fg[kf][r].resize(static_cast<size_t>(kRoleSec[r]));
          for (int s = 0; s < kRoleSec[r]; ++s) local_fg[kf][r][s].init(kRoleDim[r]);
        }
      }

      #pragma omp for schedule(dynamic)
      for (size_t idx = 0; idx < train_imgs.size(); ++idx) {
        const int kf = static_cast<int>(idx % kFolds);
        const auto& im = train_imgs[idx];
        const int H = im.H, W = im.W;
        // Per-image variance normalization so log-loss (bpsp) is optimized across images
        // rather than letting a single high-variance synthetic pattern dominate L2 sums.
        std::vector<std::vector<RoleGram>> img_fg(9);
        for (int r = 0; r < 9; ++r) {
          img_fg[r].resize(static_cast<size_t>(kRoleSec[r]));
          for (int s = 0; s < kRoleSec[r]; ++s) img_fg[r][s].init(kRoleDim[r]);
        }
        // Train across both perm=0 (G, R-G, B-(G+R)/2) and perm=1 (G, B-G, R-(G+B)/2) for color symmetry!
        const int perm = static_cast<int>(idx & 1);
        const int ia = kSubAvgPerms[perm][0];
        const int ib = kSubAvgPerms[perm][1];
        const int ic = kSubAvgPerms[perm][2];
        Plane C0 = im.rgb[ia];
        Plane C1(H, W), C2(H, W);
        for (int i = 0; i < H * W; ++i) {
          const int32_t a = im.rgb[ia].samples[i];
          const int32_t b = im.rgb[ib].samples[i];
          const int32_t c = im.rgb[ic].samples[i];
          C1.samples[i] = b - a;
          C2.samples[i] = c - ((a + b) >> 1);
        }

        int32_t f[40];
        int sec = 0;
        int64_t pr[8];
        auto run_ras = [&](const Plane& P, int role, int c_idx, const Plane* pC0, const Plane* pC0e,
                           const Plane* pC1, const Plane* pC1e, Plane& out_e) {
          out_e = Plane(H, W, 0);
          const int D = kRoleDim[role];
          FastRasterState st(H, W, c_idx, kModeSubAvg, 255, pC0 != nullptr, pC1 != nullptr, true,
                             c_idx == 0 ? mw_luma : mw_chroma);
          for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
              const int64_t pq4 = st.predict_features(P, pC0, pC0e, pC1, pC1e, y, x, f, sec, nullptr, pr);
              int64_t acc = 128;
              const auto& ws = w_prior[role][sec];
              for (int a = 0; a < D; ++a) acc += static_cast<int64_t>(f[a]) * ws[a];
              int64_t mu_q4 = pq4 + (acc >> 8);
              const auto [blo, bhi] = pixel_bounds_q4(kModeSubAvg, c_idx, 255, pC0, pC1, y, x);
              mu_q4 = std::clamp(mu_q4, blo, bhi);
              const int32_t tr = P.at(y, x);
              out_e.at(y, x) = static_cast<int32_t>((int64_t(tr) << 4) - mu_q4);
              if ((((y >> 1) ^ (x >> 1)) & 1) == 0) {
                accum_sample(img_fg[role][sec], D, f, static_cast<double>((int64_t(tr) << 4) - pq4));
              }
              st.commit(P, y, x, tr, pq4, mu_q4, pr);
            }
          }
        };

        Plane e0, e1, e2;
        run_ras(C0, 0, 0, nullptr, nullptr, nullptr, nullptr, e0);
        run_ras(C1, 1, 1, &C0, &e0, nullptr, nullptr, e1);
        run_ras(C2, 2, 2, &C0, &e0, &C1, &e1, e2);

        auto run_cos = [&](const Plane& P, int base_role, int c_idx, const Plane* pC1, const Plane* pC1e) {
          Plane stage_err(H, W, 0);
          for (int y = 0; y < H; y += 2) {
            for (int x = 0; x < W; x += 2) stage_err.at(y, x) = std::abs(c_idx == 1 ? e1.at(y, x) : e2.at(y, x));
          }
          const int py_s[4] = {0, 1, 0, 1};
          const int px_s[4] = {0, 1, 1, 0};
          for (int stg = 1; stg <= 3; ++stg) {
            const int role = base_role + (stg - 1);
            const int D = kRoleDim[role];
            for (int y = py_s[stg]; y < H; y += 2) {
              for (int x = px_s[stg]; x < W; x += 2) {
                const int64_t b_q4 =
                    fast_interp_features(P, C0, e0, pC1, pC1e, stage_err, 255, stg, y, x, f, sec, nullptr);
                int64_t acc = 128;
                const auto& ws = w_prior[role][sec];
                for (int a = 0; a < D; ++a) acc += static_cast<int64_t>(f[a]) * ws[a];
                const int32_t tr = P.at(y, x);
                stage_err.at(y, x) = static_cast<int32_t>(iabs64((int64_t(tr) << 4) - (b_q4 + (acc >> 8))));
                accum_sample(img_fg[role][sec], D, f, static_cast<double>((int64_t(tr) << 4) - b_q4));
              }
            }
          }
        };
        run_cos(C1, 3, 1, nullptr, nullptr);
        run_cos(C2, 6, 2, &C1, &e1);

        // Normalize each image+role by 1 / (role_mse + 256.0) so log-variance (bpsp) is minimized!
        for (int r = 0; r < 9; ++r) {
          const int D = kRoleDim[r];
          double role_y2 = 0.0;
          int64_t role_n = 0;
          for (int s = 0; s < kRoleSec[r]; ++s) {
            role_y2 += img_fg[r][s].Y2;
            role_n += img_fg[r][s].N;
          }
          const double mse = role_y2 / std::max<int64_t>(1, role_n);
          const double wt_im = 1024.0 / (mse + 256.0);
          for (int s = 0; s < kRoleSec[r]; ++s) {
            auto& dst = local_fg[kf][r][s];
            const auto& src = img_fg[r][s];
            dst.N += src.N;
            dst.Y2 += src.Y2 * wt_im;
            for (int i = 0; i < D; ++i) dst.R[i] += src.R[i] * wt_im;
            for (int i = 0; i < D * D; ++i) dst.G[i] += src.G[i] * wt_im;
          }
        }
      }

      #pragma omp critical
      {
        for (int kf = 0; kf < kFolds; ++kf) {
          for (int r = 0; r < 9; ++r) {
            const int D = kRoleDim[r];
            for (int s = 0; s < kRoleSec[r]; ++s) {
              auto& dst = fold_grams[kf][r][s];
              const auto& src = local_fg[kf][r][s];
              dst.N += src.N;
              dst.Y2 += src.Y2;
              for (int i = 0; i < D; ++i) dst.R[i] += src.R[i];
              for (int i = 0; i < D * D; ++i) dst.G[i] += src.G[i];
            }
          }
        }
      }
    }

    // 8-fold CV selection of relative L2 ridge lambda & stability t-threshold per role
    for (int r = 0; r < 9; ++r) {
      const int D = kRoleDim[r];
      const int NS = kRoleSec[r];
      const double cands_rel[8] = {1e-4, 5e-4, 2e-3, 1e-2, 4e-2, 1.5e-1, 5e-1, 1.5};
      double best_rel = 2e-3;
      double best_cv_mse = std::numeric_limits<double>::infinity();

      for (double rel : cands_rel) {
        double cv_se = 0.0;
        int64_t cv_n = 0;
        for (int s = 0; s < NS; ++s) {
          for (int val_f = 0; val_f < kFolds; ++val_f) {
            std::vector<double> Gtr(static_cast<size_t>(D * D), 0.0);
            std::vector<double> Rtr(static_cast<size_t>(D), 0.0);
            for (int kf = 0; kf < kFolds; ++kf) {
              if (kf == val_f) continue;
              for (int a = 0; a < D; ++a) {
                Rtr[a] += fold_grams[kf][r][s].R[a];
                for (int b = a; b < D; ++b) Gtr[a * D + b] += fold_grams[kf][r][s].G[a * D + b];
              }
            }
            for (int a = 0; a < D; ++a) {
              for (int b = 0; b < a; ++b) Gtr[a * D + b] = Gtr[b * D + a];
              Gtr[a * D + a] += std::max(Gtr[a * D + a] * rel, 4096.0);
            }
            std::vector<double> w;
            if (!solve_spd(Gtr, Rtr, D, w)) w.assign(static_cast<size_t>(D), 0.0);
            const auto& gv = fold_grams[val_f][r][s];
            double err = gv.Y2;
            for (int a = 0; a < D; ++a) {
              err -= 2.0 * w[a] * gv.R[a];
              err += w[a] * w[a] * gv.G[a * D + a];
              for (int b = a + 1; b < D; ++b) err += 2.0 * w[a] * w[b] * gv.G[a * D + b];
            }
            cv_se += err;
            cv_n += gv.N;
          }
        }
        const double mse = cv_se / std::max<int64_t>(1, cv_n);
        if (mse < best_cv_mse) {
          best_cv_mse = mse;
          best_rel = rel;
        }
      }

      // Fit per-fold weights at best_rel to compute cross-fold t-statistic stability shrinkage!
      double zero_y2 = 0.0;
      int64_t tot_n = 0;
      for (int s = 0; s < NS; ++s) {
        std::vector<std::vector<double>> fold_w(kFolds, std::vector<double>(static_cast<size_t>(D), 0.0));
        for (int val_f = 0; val_f < kFolds; ++val_f) {
          std::vector<double> Gtr(static_cast<size_t>(D * D), 0.0);
          std::vector<double> Rtr(static_cast<size_t>(D), 0.0);
          for (int kf = 0; kf < kFolds; ++kf) {
            if (kf == val_f) continue;
            for (int a = 0; a < D; ++a) {
              Rtr[a] += fold_grams[kf][r][s].R[a];
              for (int b = a; b < D; ++b) Gtr[a * D + b] += fold_grams[kf][r][s].G[a * D + b];
            }
          }
          for (int a = 0; a < D; ++a) {
            for (int b = 0; b < a; ++b) Gtr[a * D + b] = Gtr[b * D + a];
            Gtr[a * D + a] += std::max(Gtr[a * D + a] * best_rel, 4096.0);
          }
          solve_spd(Gtr, Rtr, D, fold_w[val_f]);
        }

        std::vector<double> Gall(static_cast<size_t>(D * D), 0.0);
        std::vector<double> Rall(static_cast<size_t>(D), 0.0);
        for (int kf = 0; kf < kFolds; ++kf) {
          zero_y2 += fold_grams[kf][r][s].Y2;
          tot_n += fold_grams[kf][r][s].N;
          for (int a = 0; a < D; ++a) {
            Rall[a] += fold_grams[kf][r][s].R[a];
            for (int b = a; b < D; ++b) Gall[a * D + b] += fold_grams[kf][r][s].G[a * D + b];
          }
        }
        for (int a = 0; a < D; ++a) {
          for (int b = 0; b < a; ++b) Gall[a * D + b] = Gall[b * D + a];
          Gall[a * D + a] += std::max(Gall[a * D + a] * best_rel, 4096.0);
        }
        std::vector<double> wall(static_cast<size_t>(D), 0.0);
        solve_spd(Gall, Rall, D, wall);

        for (int a = 0; a < D; ++a) {
          double mean = 0.0;
          for (int kf = 0; kf < kFolds; ++kf) mean += fold_w[kf][a];
          mean /= kFolds;
          double var = 0.0;
          for (int kf = 0; kf < kFolds; ++kf) {
            const double d = fold_w[kf][a] - mean;
            var += d * d;
          }
          // Jackknife variance of (K-1)/K leave-one-fold-out estimates is (K-1)/K * sum(d^2)
          const double se = std::sqrt(((kFolds - 1.0) / kFolds) * var + 1e-12);
          const double t_stat = std::fabs(mean) / se;
          // James-Stein / Empirical-Bayes shrinkage factor max(0, 1 - 1.5 / t_stat^2)
          const double shrink = t_stat <= 1.25 ? 0.0 : std::max(0.0, 1.0 - 1.5 / (t_stat * t_stat));
          const int q8 = std::clamp<int>(static_cast<int>(std::llround(wall[a] * shrink * 256.0)), -1024, 1024);
          w_prior[r][s][a] = static_cast<int16_t>(q8);
        }
      }
      if (iter == 1) {
        std::printf("Role %d (S=%2d, D=%2d): best_rel=%.4f  zero_RMS=%.2f -> 8fold_CV_RMS=%.2f\n",
                    r, NS, D, best_rel, std::sqrt(zero_y2 / tot_n), std::sqrt(best_cv_mse));
      }
    }
  }

  // -------------------------------------------------------------------------
  // Step 3: Validate W_prior + Cross-Validate Causal Integer NLMS on Val!
  // -------------------------------------------------------------------------
  auto eval_prior_nlms_val = [&](const std::vector<RawImg>& corpus, bool use_prior, int nlms_num_luma,
                                 int nlms_num_chroma) -> std::pair<double, double> {
    double luma_mae = 0.0, chroma_mae = 0.0;
    #pragma omp parallel for reduction(+ : luma_mae, chroma_mae)
    for (size_t idx = 0; idx < corpus.size(); ++idx) {
      const auto& im = corpus[idx];
      const int H = im.H, W = im.W;
      Plane C0 = im.rgb[1];
      Plane C1(H, W), C2(H, W);
      for (int i = 0; i < H * W; ++i) {
        const int32_t g = im.rgb[1].samples[i];
        const int32_t r = im.rgb[0].samples[i];
        const int32_t b = im.rgb[2].samples[i];
        C1.samples[i] = r - g;
        C2.samples[i] = b - ((g + r) >> 1);
      }
      int32_t f[40];
      int sec = 0;
      int64_t pr[8];

      auto run_ch = [&](const Plane& P, int role, int c_idx, int nlms_num, const Plane* pC0, const Plane* pC0e,
                        const Plane* pC1, const Plane* pC1e, Plane& out_e) -> double {
        out_e = Plane(H, W, 0);
        const int D = kRoleDim[role];
        std::array<std::array<int32_t, 36>, 16> wlms{};
        for (int s = 0; s < 16; ++s) {
          for (int a = 0; a < D; ++a) wlms[s][a] = use_prior ? (int32_t(w_prior[role][s][a]) << 8) : 0;
        }
        FastRasterState st(H, W, c_idx, kModeSubAvg, 255, pC0 != nullptr, pC1 != nullptr, true,
                           c_idx == 0 ? mw_luma : mw_chroma);
        double sum_abs = 0.0;
        for (int y = 0; y < H; ++y) {
          for (int x = 0; x < W; ++x) {
            const int64_t pq4 = st.predict_features(P, pC0, pC0e, pC1, pC1e, y, x, f, sec, nullptr, pr);
            int64_t acc = 128;
            auto& ws = wlms[sec];
            for (int a = 0; a < D; ++a) acc += static_cast<int64_t>(f[a]) * (ws[a] >> 8);
            const int64_t raw_mu = pq4 + (acc >> 8);
            const auto [blo, bhi] = pixel_bounds_q4(kModeSubAvg, c_idx, 255, pC0, pC1, y, x);
            const int64_t mu_q4 = std::clamp(raw_mu, blo, bhi);
            const int32_t tr = P.at(y, x);
            const int64_t err_q4 = (static_cast<int64_t>(tr) << 4) - mu_q4;
            out_e.at(y, x) = static_cast<int32_t>(err_q4);
            sum_abs += std::fabs(static_cast<double>(err_q4));
            if (nlms_num > 0) {
              uint64_t e2 = 4096;
              for (int a = 0; a < D; ++a) e2 += static_cast<uint64_t>(int64_t(f[a]) * f[a]);
              const int sh = std::max(0, flog2_u64(e2) - 5);
              const uint64_t idx_d = std::min<uint64_t>(63, e2 >> sh);
              // Approximate 1/e2 via kDiv[idx_d + 1] >> sh (100% division-free integer NLMS!)
              const int64_t raw_err = (static_cast<int64_t>(tr) << 4) - raw_mu;
              const int64_t inv_q24 = static_cast<int64_t>(kDiv[idx_d + 1]) >> sh;
              const int64_t step = std::clamp<int64_t>((raw_err * nlms_num * inv_q24) >> 8, -2048, 2048);
              for (int a = 0; a < D; ++a) {
                ws[a] = std::clamp<int32_t>(ws[a] + static_cast<int32_t>((int64_t(f[a]) * step) >> 8),
                                            -1536 << 8, 1536 << 8);
              }
            }
            st.commit(P, y, x, tr, pq4, mu_q4, pr);
          }
        }
        return sum_abs / (H * W);
      };

      Plane e0, e1, e2;
      luma_mae += run_ch(C0, 0, 0, nlms_num_luma, nullptr, nullptr, nullptr, nullptr, e0);
      chroma_mae += 0.5 * (run_ch(C1, 1, 1, nlms_num_chroma, &C0, &e0, nullptr, nullptr, e1) +
                           run_ch(C2, 2, 2, nlms_num_chroma, &C0, &e0, &C1, &e1, e2));
    }
    return {luma_mae / corpus.size(), chroma_mae / corpus.size()};
  };

  const auto [val_l0, val_c0] = eval_prior_nlms_val(val_imgs, false, 0, 0);
  const auto [val_lp, val_cp] = eval_prior_nlms_val(val_imgs, true, 0, 0);
  std::printf("\nOut-of-Sample VAL Single-Pass Q4-MAE:\n");
  std::printf("  Zero Prior (no Ridge):   Luma = %.3f   Chroma = %.3f\n", val_l0, val_c0);
  std::printf("  Trained W_prior (0 B):   Luma = %.3f   Chroma = %.3f\n", val_lp, val_cp);

  int best_nlms_l = 0, best_nlms_c = 0;
  double best_val_l = val_lp, best_val_c = val_cp;
  for (int cand : {2, 4, 6, 8, 12, 16, 20, 24, 32}) {
    const auto [vl, vc] = eval_prior_nlms_val(val_imgs, true, cand, cand);
    std::printf("  W_prior + NLMS(mu=%2d):   Luma = %.3f   Chroma = %.3f\n", cand, vl, vc);
    if (vl < best_val_l) {
      best_val_l = vl;
      best_nlms_l = cand;
    }
    if (vc < best_val_c) {
      best_val_c = vc;
      best_nlms_c = cand;
    }
  }
  std::printf("Selected NLMS step: Luma=%d (MAE=%.3f), Chroma=%d (MAE=%.3f)\n",
              best_nlms_l, best_val_l, best_nlms_c, best_val_c);

  // -------------------------------------------------------------------------
  // Step 4: Train Out-of-Fold Pruned Universal MA Context Trees (5 Modes)
  //   0: Luma Raster (14 props, 32 leaves)
  //   1: Chroma Raster (14 props, 32 leaves)
  //   2,3,4: Chroma Coset Stage 1,2,3 (11 props, 24 leaves)
  // -------------------------------------------------------------------------
  const ParametricTables& ptbl = get_parametric_tables();
  double sigma_eb_cost[NUM_SIGMAS][kErrBins]{};
  {
    // Expected Core66 + remainder bits for a sample in log-error bin `eb` under `sigma` (nu=1)
    std::vector<double> nlog(65536);
    for (int i = 0; i < 65536; ++i) nlog[i] = -std::log2((i + 1) / 65536.0);
    for (int eb = 0; eb < kErrBins; ++eb) {
      // Representative Q4 magnitude in bin eb: eb==0 -> 0, else (1<<(eb-1)) * 1.35 in Q2 units -> r in sample units
      const int rep_q2 = eb == 0 ? 0 : static_cast<int>(std::llround((1 << (eb - 1)) * 1.35));
      const int r_mag = (rep_q2 + 2) >> 2;
      for (int s = 0; s < NUM_SIGMAS; ++s) {
        const int sh = ptbl.shifts[s];
        const int64_t bias = sh > 0 ? (1LL << (sh - 1)) : 0;
        double c_acc = 0.0;
        for (int sign : {-1, 1}) {
          const int64_t r = sign * r_mag;
          const int64_t q = sh > 0 ? ((r + bias) >> sh) : r;
          const int64_t mq = iabs64(q);
          const int sym = mq > 32 ? 65 : static_cast<int>(q + 32);
          double c = sh + nlog[ptbl.freqs[1][s][8][sym] - 1];
          if (sym == 65) c += 2.0 * std::log2(static_cast<double>(mq - 31)) + 3.0;
          c_acc += 0.5 * c;
        }
        sigma_eb_cost[s][eb] = c_acc;
      }
    }
  }

  struct SampleRec {
    int16_t prp[14];
    uint8_t eb;
    uint8_t is_val_half;
  };
  std::vector<std::vector<SampleRec>> tree_samples(5);
  for (int m = 0; m < 5; ++m) tree_samples[m].reserve(900000);

  for (size_t idx = 0; idx < train_imgs.size(); ++idx) {
    const uint8_t half = static_cast<uint8_t>(idx & 1);
    const auto& im = train_imgs[idx];
    const int H = im.H, W = im.W;
    Plane C0 = im.rgb[1];
    Plane C1(H, W), C2(H, W);
    for (int i = 0; i < H * W; ++i) {
      const int32_t g = im.rgb[1].samples[i];
      const int32_t r = im.rgb[0].samples[i];
      const int32_t b = im.rgb[2].samples[i];
      C1.samples[i] = r - g;
      C2.samples[i] = b - ((g + r) >> 1);
    }
    int32_t f[40];
    int16_t prp[14];
    int sec = 0;
    int64_t pr[8];

    auto collect_ras = [&](const Plane& P, int role, int tree_m, int c_idx, int nlms_num, const Plane* pC0,
                           const Plane* pC0e, const Plane* pC1, const Plane* pC1e, Plane& out_e) {
      out_e = Plane(H, W, 0);
      const int D = kRoleDim[role];
      std::array<std::array<int32_t, 36>, 16> wlms{};
      for (int s = 0; s < 16; ++s) {
        for (int a = 0; a < D; ++a) wlms[s][a] = int32_t(w_prior[role][s][a]) << 8;
      }
      FastRasterState st(H, W, c_idx, kModeSubAvg, 255, pC0 != nullptr, pC1 != nullptr, true,
                         c_idx == 0 ? mw_luma : mw_chroma);
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          const int64_t pq4 = st.predict_features(P, pC0, pC0e, pC1, pC1e, y, x, f, sec, prp, pr);
          int64_t acc = 128;
          auto& ws = wlms[sec];
          for (int a = 0; a < D; ++a) acc += static_cast<int64_t>(f[a]) * (ws[a] >> 8);
          const int64_t raw_mu = pq4 + (acc >> 8);
          const auto [blo, bhi] = pixel_bounds_q4(kModeSubAvg, c_idx, 255, pC0, pC1, y, x);
          const int64_t mu_q4 = std::clamp(raw_mu, blo, bhi);
          const int32_t tr = P.at(y, x);
          const int64_t err_q4 = (static_cast<int64_t>(tr) << 4) - mu_q4;
          out_e.at(y, x) = static_cast<int32_t>(err_q4);
          if ((((y >> 1) * 37 + (x >> 1) * 13 + int(idx)) & 63) == 0) {
            SampleRec sr{};
            for (int p = 0; p < 14; ++p) sr.prp[p] = prp[p];
            sr.eb = static_cast<uint8_t>(std::min(15, flog2_u32(static_cast<uint32_t>((iabs64(err_q4) >> 2) + 1))));
            sr.is_val_half = half;
            tree_samples[tree_m].push_back(sr);
          }
          if (nlms_num > 0) {
            uint64_t e2 = 4096;
            for (int a = 0; a < D; ++a) e2 += static_cast<uint64_t>(int64_t(f[a]) * f[a]);
            const int sh = std::max(0, flog2_u64(e2) - 5);
            const uint64_t idx_d = std::min<uint64_t>(63, e2 >> sh);
            const int64_t raw_err = (static_cast<int64_t>(tr) << 4) - raw_mu;
            const int64_t inv_q24 = static_cast<int64_t>(kDiv[idx_d + 1]) >> sh;
            const int64_t step = std::clamp<int64_t>((raw_err * nlms_num * inv_q24) >> 8, -2048, 2048);
            for (int a = 0; a < D; ++a) {
              ws[a] = std::clamp<int32_t>(ws[a] + static_cast<int32_t>((int64_t(f[a]) * step) >> 8),
                                          -1536 << 8, 1536 << 8);
            }
          }
          st.commit(P, y, x, tr, pq4, mu_q4, pr);
        }
      }
    };

    Plane e0, e1, e2;
    collect_ras(C0, 0, 0, 0, best_nlms_l, nullptr, nullptr, nullptr, nullptr, e0);
    collect_ras(C1, 1, 1, 1, best_nlms_c, &C0, &e0, nullptr, nullptr, e1);
    collect_ras(C2, 2, 1, 2, best_nlms_c, &C0, &e0, &C1, &e1, e2);

    Plane stage_err(H, W, 0);
    for (int y = 0; y < H; y += 2) {
      for (int x = 0; x < W; x += 2) stage_err.at(y, x) = std::abs(e1.at(y, x));
    }
    const int py_s[4] = {0, 1, 0, 1};
    const int px_s[4] = {0, 1, 1, 0};
    for (int stg = 1; stg <= 3; ++stg) {
      const int role = 3 + (stg - 1);
      const int D = kRoleDim[role];
      int cnt = 0;
      for (int y = py_s[stg]; y < H; y += 2) {
        for (int x = px_s[stg]; x < W; x += 2, ++cnt) {
          const int64_t b_q4 =
              fast_interp_features(C1, C0, e0, nullptr, nullptr, stage_err, 255, stg, y, x, f, sec, prp);
          int64_t acc = 128;
          for (int a = 0; a < D; ++a) acc += static_cast<int64_t>(f[a]) * w_prior[role][sec][a];
          int64_t mu_q4 = b_q4 + (acc >> 8);
          const auto [blo, bhi] = pixel_bounds_q4(kModeSubAvg, 1, 255, &C0, nullptr, y, x);
          mu_q4 = std::clamp(mu_q4, blo, bhi);
          const int64_t eq4 = (static_cast<int64_t>(C1.at(y, x)) << 4) - mu_q4;
          stage_err.at(y, x) = static_cast<int32_t>(iabs64(eq4));
          if ((cnt & 15) == int(idx & 15)) {
            SampleRec sr{};
            for (int p = 0; p < 11; ++p) sr.prp[p] = prp[p];
            sr.eb = static_cast<uint8_t>(std::min(15, flog2_u32(static_cast<uint32_t>((iabs64(eq4) >> 2) + 1))));
            sr.is_val_half = half;
            tree_samples[1 + stg].push_back(sr);
          }
        }
      }
    }
  }

  auto best_hist_cost = [&](const int64_t cnt[kErrBins]) -> double {
    double best = std::numeric_limits<double>::infinity();
    for (int s = 0; s < NUM_SIGMAS; ++s) {
      double c = 0.0;
      for (int eb = 0; eb < kErrBins; ++eb) c += static_cast<double>(cnt[eb]) * sigma_eb_cost[s][eb];
      if (c < best) best = c;
    }
    return best;
  };

  std::vector<std::vector<FlatTreeNode>> univ_trees(5);
  std::vector<int> univ_leaf_counts(5, 0);
  for (int m = 0; m < 5; ++m) {
    const int n_props = m <= 1 ? 14 : 11;
    const int target_leaves = m <= 1 ? 32 : 24;
    const auto& samps = tree_samples[m];

    struct TLeaf {
      int node_id;
      std::vector<int> idx;
      double gain;
      int best_p;
      int best_t;
      int depth;
    };

    auto eval_tl = [&](TLeaf& lf) {
      lf.gain = -1.0;
      if (lf.depth >= 10 || lf.idx.size() < 512) return;
      int64_t h_fit[14][kPropBins][kErrBins]{};
      int64_t h_val[14][kPropBins][kErrBins]{};
      int64_t tot_fit[kErrBins]{}, tot_val[kErrBins]{};
      for (int id : lf.idx) {
        const auto& sr = samps[id];
        if (sr.is_val_half == 0) tot_fit[sr.eb]++;
        else tot_val[sr.eb]++;
        for (int p = 0; p < n_props; ++p) {
          const int b = std::clamp<int>(int(sr.prp[p]) - kPropMin, 0, kPropBins - 1);
          if (sr.is_val_half == 0) h_fit[p][b][sr.eb]++;
          else h_val[p][b][sr.eb]++;
        }
      }
      const double base_f = best_hist_cost(tot_fit);
      const double base_v = best_hist_cost(tot_val);
      for (int p = 0; p < n_props; ++p) {
        int64_t lf_c[kErrBins]{}, lv_c[kErrBins]{};
        int64_t nf_l = 0, nv_l = 0, nf_tot = 0, nv_tot = 0;
        for (int eb = 0; eb < kErrBins; ++eb) {
          nf_tot += tot_fit[eb];
          nv_tot += tot_val[eb];
        }
        for (int b = 0; b < kPropBins - 1; ++b) {
          for (int eb = 0; eb < kErrBins; ++eb) {
            lf_c[eb] += h_fit[p][b][eb];
            lv_c[eb] += h_val[p][b][eb];
            nf_l += h_fit[p][b][eb];
            nv_l += h_val[p][b][eb];
          }
          if (nf_l < 128 || (nf_tot - nf_l) < 128 || nv_l < 128 || (nv_tot - nv_l) < 128) continue;
          int64_t rf_c[kErrBins], rv_c[kErrBins];
          for (int eb = 0; eb < kErrBins; ++eb) {
            rf_c[eb] = tot_fit[eb] - lf_c[eb];
            rv_c[eb] = tot_val[eb] - lv_c[eb];
          }
          const double gf = base_f - best_hist_cost(lf_c) - best_hist_cost(rf_c);
          const double gv = base_v - best_hist_cost(lv_c) - best_hist_cost(rv_c);
          // Require BOTH halves to agree on positive out-of-fold gain!
          if (gf > 20.0 && gv > 20.0 && std::min(gf, gv) > lf.gain) {
            lf.gain = std::min(gf, gv);
            lf.best_p = p;
            lf.best_t = b + kPropMin;
          }
        }
      }
    };

    auto& nodes = univ_trees[m];
    nodes.push_back(FlatTreeNode{0, -1, 0, -1, -1});
    std::vector<int> all_i(samps.size());
    std::iota(all_i.begin(), all_i.end(), 0);
    std::vector<TLeaf> leaves;
    leaves.push_back(TLeaf{0, std::move(all_i), -1.0, 0, 0, 0});
    eval_tl(leaves[0]);

    while (static_cast<int>(leaves.size()) < target_leaves) {
      int bk = -1;
      double bg = 20.0;
      for (size_t k = 0; k < leaves.size(); ++k) {
        if (leaves[k].gain > bg) {
          bg = leaves[k].gain;
          bk = static_cast<int>(k);
        }
      }
      if (bk < 0) break;
      TLeaf cur = std::move(leaves[bk]);
      leaves.erase(leaves.begin() + bk);
      std::vector<int> li, ri;
      for (int id : cur.idx) {
        if (samps[id].prp[cur.best_p] <= cur.best_t) li.push_back(id);
        else ri.push_back(id);
      }
      const int lid = static_cast<int>(nodes.size());
      nodes.push_back(FlatTreeNode{0, -1, 0, -1, -1});
      const int rid = static_cast<int>(nodes.size());
      nodes.push_back(FlatTreeNode{0, -1, 0, -1, -1});
      nodes[cur.node_id].prop = static_cast<int8_t>(cur.best_p);
      nodes[cur.node_id].thr = static_cast<int16_t>(cur.best_t);
      nodes[cur.node_id].left = static_cast<int16_t>(lid);
      nodes[cur.node_id].right = static_cast<int16_t>(rid);
      TLeaf ll{lid, std::move(li), -1.0, 0, 0, cur.depth + 1};
      TLeaf rl{rid, std::move(ri), -1.0, 0, 0, cur.depth + 1};
      eval_tl(ll);
      eval_tl(rl);
      leaves.push_back(std::move(ll));
      leaves.push_back(std::move(rl));
    }

    int lid_cnt = 0;
    auto num_dfs = [&](auto&& self, int id) -> void {
      if (nodes[id].left < 0) {
        nodes[id].leaf_id = static_cast<uint8_t>(lid_cnt++);
        return;
      }
      self(self, nodes[id].left);
      self(self, nodes[id].right);
    };
    num_dfs(num_dfs, 0);
    univ_leaf_counts[m] = lid_cnt;
    std::printf("Universal MA Tree %d: %zu samples -> %d validated leaves (%zu nodes)\n",
                m, samps.size(), lid_cnt, nodes.size());
  }

  // -------------------------------------------------------------------------
  // Step 5: Emit self-contained constexpr C++20 header frontier_priors.hpp
  // -------------------------------------------------------------------------
  std::ofstream out(out_hpp);
  out << "#pragma once\n\n";
  out << "// Auto-generated by training/train_frontier.cpp on strictly disjoint CLIC+Synthetic\n";
  out << "// training data (zero Kodak / benchmark overlap), validated with 8-fold image-grouped\n";
  out << "// cross-validation and James-Stein t-statistic stability shrinkage.\n\n";
  out << "#include <cstdint>\n\n";
  out << "namespace vpxc::priors {\n\n";

  out << "inline constexpr uint32_t kWpWeightsLuma[8] = {";
  for (int i = 0; i < 8; ++i) out << mw_luma[i] << (i + 1 < 8 ? ", " : "");
  out << "};\n";
  out << "inline constexpr uint32_t kWpWeightsChroma[8] = {";
  for (int i = 0; i < 8; ++i) out << mw_chroma[i] << (i + 1 < 8 ? ", " : "");
  out << "};\n\n";

  out << "inline constexpr int kNlmsGainLuma = " << best_nlms_l << ";\n";
  out << "inline constexpr int kNlmsGainChroma = " << best_nlms_c << ";\n\n";

  const char* role_names[9] = {
      "kPriorLuma",       "kPriorChromaRas1", "kPriorChromaRas2",
      "kPriorCos1Stage1", "kPriorCos1Stage2", "kPriorCos1Stage3",
      "kPriorCos2Stage1", "kPriorCos2Stage2", "kPriorCos2Stage3",
  };
  for (int r = 0; r < 9; ++r) {
    const int NS = kRoleSec[r];
    const int D = kRoleDim[r];
    out << "inline constexpr int16_t " << role_names[r] << "[" << NS << "][" << D << "] = {\n";
    for (int s = 0; s < NS; ++s) {
      out << "    {";
      for (int a = 0; a < D; ++a) out << w_prior[r][s][a] << (a + 1 < D ? ", " : "");
      out << "},\n";
    }
    out << "};\n\n";
  }

  out << "struct UnivTreeNode {\n";
  out << "  int16_t thr;\n";
  out << "  int8_t prop;\n";
  out << "  uint8_t leaf_id;\n";
  out << "  int16_t left;\n";
  out << "  int16_t right;\n";
  out << "};\n\n";

  out << "inline constexpr int kUnivTreeLeaves[5] = {";
  for (int m = 0; m < 5; ++m) out << univ_leaf_counts[m] << (m + 1 < 5 ? ", " : "");
  out << "};\n\n";

  const char* tree_names[5] = {"kUnivTreeLuma", "kUnivTreeChromaRas", "kUnivTreeCosS1", "kUnivTreeCosS2",
                               "kUnivTreeCosS3"};
  for (int m = 0; m < 5; ++m) {
    out << "inline constexpr UnivTreeNode " << tree_names[m] << "[" << univ_trees[m].size() << "] = {\n";
    for (const auto& nd : univ_trees[m]) {
      out << "    {" << nd.thr << ", " << int(nd.prop) << ", " << int(nd.leaf_id) << ", " << nd.left << ", "
          << nd.right << "},\n";
    }
    out << "};\n\n";
  }

  out << "}  // namespace vpxc::priors\n";
  std::printf("Wrote %s successfully.\n", out_hpp.c_str());
  return 0;
}
