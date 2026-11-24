#include "vpxc/frontier.hpp"
#include "vpxc/frontier_priors.hpp"

#include "vpxc/bitio.hpp"
#include "vpxc/color.hpp"
#include "vpxc/entropy.hpp"
#include "vpxc/error.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <numeric>
#include <span>
#include <thread>
#include <utility>
#include <vector>

namespace vpxc {
namespace {

constexpr int kModeSubAvg = 5;

constexpr int kSubAvgPerms[6][3] = {
    {1, 0, 2},  // 0: G, R-G, B - ((G+R)>>1)
    {1, 2, 0},  // 1: G, B-G, R - ((G+B)>>1)
    {0, 1, 2},  // 2: R, G-R, B - ((R+G)>>1)
    {0, 2, 1},  // 3: R, B-R, G - ((R+B)>>1)
    {2, 1, 0},  // 4: B, G-B, R - ((B+G)>>1)
    {2, 0, 1},  // 5: B, R-B, G - ((B+R)>>1)
};

constexpr uint32_t kDiv[65] = {
    16777216, 16777216, 8388608, 5592405, 4194304, 3355443, 2796202, 2396745, 2097152, 1864135, 1677721,
    1525201,  1398101,  1290555, 1198372, 1118481, 1048576, 986895,  932067,  883011,  838860,  798915,
    762600,   729444,   699050,  671088,  645277,  621378,  599186,  578524,  559240,  541200,  524288,
    508400,   493447,   479349,  466033,  453438,  441505,  430185,  419430,  409200,  399457,  390167,
    381300,   372827,   364722,  356962,  349525,  342392,  335544,  328965,  322638,  316551,  310689,
    305040,   299593,   294337,  289262,  284359,  279620,  275036,  270600,  266305,  262144,
};

struct FastIntLuts {
  uint16_t bilat[11][96];
  uint32_t recip[6145];
  uint16_t patch_wt[20];
  uint32_t patch_recip[260];
};

constexpr FastIntLuts make_fast_int_luts() {
  FastIntLuts t{};
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

constexpr FastIntLuts kFast = make_fast_int_luts();

inline int flog2_u32(uint32_t x) { return x ? (31 - __builtin_clz(x)) : 0; }
inline int flog2_u64(uint64_t x) { return x ? (63 - __builtin_clzll(x)) : 0; }

inline int32_t iabs32(int32_t v) {
  if (v >= 0) return v;
  if (v == INT32_MIN) return INT32_MAX;
  return -v;
}

inline int64_t iabs64(int64_t v) {
  if (v >= 0) return v;
  if (v == INT64_MIN) return INT64_MAX;
  return -v;
}

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

__attribute__((always_inline)) inline uint32_t wp_wt(uint64_t x, uint32_t maxw) {
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

inline int16_t get_prior_weight(int role, int sec, int a) {
  if (sec < 0 || a < 0) return 0;
  switch (role) {
    case 0: return (sec < 16 && a < 21) ? priors::kPriorLuma[sec][a] : 0;
    case 1: return (sec < 16 && a < 30) ? priors::kPriorChromaRas1[sec][a] : 0;
    case 2: return (sec < 16 && a < 36) ? priors::kPriorChromaRas2[sec][a] : 0;
    case 3: return (sec < 4 && a < 22) ? priors::kPriorCos1Stage1[sec][a] : 0;
    case 4: return (sec < 4 && a < 20) ? priors::kPriorCos1Stage2[sec][a] : 0;
    case 5: return (sec < 4 && a < 20) ? priors::kPriorCos1Stage3[sec][a] : 0;
    case 6: return (sec < 4 && a < 28) ? priors::kPriorCos2Stage1[sec][a] : 0;
    case 7: return (sec < 4 && a < 26) ? priors::kPriorCos2Stage2[sec][a] : 0;
    case 8: return (sec < 4 && a < 26) ? priors::kPriorCos2Stage3[sec][a] : 0;
    default: return 0;
  }
}

// Normative division-free integer NLMS update shared identically by Encode & Decode.
__attribute__((always_inline)) inline void nlms_update(int32_t* ws_q16, int16_t* ws_q8, int D, const int32_t* f,
                                                       int64_t raw_err_q4, int nlms_gain, int qsh) {
  const int64_t err_s = raw_err_q4 >> qsh;
  if (err_s >= -1 && err_s <= 1) return;
  uint64_t e2 = 4096;
  if (qsh == 0) {
    for (int a = 0; a < D; ++a) {
      const int64_t fs = f[a];
      e2 += static_cast<uint64_t>(fs * fs);
    }
    const int sh = std::max(0, flog2_u64(e2) - 5);
    if (sh >= 24) return;
    const uint64_t idx_d = std::min<uint64_t>(63, e2 >> sh);
    const int64_t inv_q24 = static_cast<int64_t>(kDiv[idx_d + 1]) >> sh;
    const int32_t step = static_cast<int32_t>(std::clamp<int64_t>((err_s * nlms_gain * inv_q24) >> 8, -2048, 2048));
    if (step == 0) return;
    for (int a = 0; a < D; ++a) {
      const int32_t nw = std::clamp<int32_t>(
          ws_q16[a] + static_cast<int32_t>((static_cast<int64_t>(f[a]) * step) >> 8), -1536 << 8, 1536 << 8);
      ws_q16[a] = nw;
      ws_q8[a] = static_cast<int16_t>(nw >> 8);
    }
  } else {
    for (int a = 0; a < D; ++a) {
      const int64_t fs = static_cast<int64_t>(f[a] >> qsh);
      e2 += static_cast<uint64_t>(fs * fs);
    }
    const int sh = std::max(0, flog2_u64(e2) - 5);
    if (sh >= 24) return;
    const uint64_t idx_d = std::min<uint64_t>(63, e2 >> sh);
    const int64_t inv_q24 = static_cast<int64_t>(kDiv[idx_d + 1]) >> sh;
    const int64_t step = std::clamp<int64_t>((err_s * nlms_gain * inv_q24) >> 8, -2048, 2048);
    if (step == 0) return;
    for (int a = 0; a < D; ++a) {
      const int64_t fs = static_cast<int64_t>(f[a] >> qsh);
      const int32_t nw =
          std::clamp<int32_t>(ws_q16[a] + static_cast<int32_t>((fs * step) >> 8), -1536 << 8, 1536 << 8);
      ws_q16[a] = nw;
      ws_q8[a] = static_cast<int16_t>(nw >> 8);
    }
  }
}

std::vector<Plane> color_fwd_frontier(const std::vector<Plane>& ch, int mode, int perm) {
  if (ch.size() < 3 || mode == MODE_IDENTITY) return ch;
  const int H = ch[0].height;
  const int W = ch[0].width;
  if (mode == kModeSubAvg) {
    const int p = std::clamp(perm, 0, 5);
    const int ia = kSubAvgPerms[p][0];
    const int ib = kSubAvgPerms[p][1];
    const int ic = kSubAvgPerms[p][2];
    std::vector<Plane> out;
    out.reserve(ch.size());
    out.push_back(ch[static_cast<size_t>(ia)]);
    out.emplace_back(H, W);
    out.emplace_back(H, W);
    for (size_t i = 0; i < ch[0].samples.size(); ++i) {
      const int32_t a = ch[static_cast<size_t>(ia)].samples[i];
      const int32_t b = ch[static_cast<size_t>(ib)].samples[i];
      const int32_t c = ch[static_cast<size_t>(ic)].samples[i];
      out[1].samples[i] = b - a;
      out[2].samples[i] = c - ((a + b) >> 1);
    }
    for (size_t k = 3; k < ch.size(); ++k) out.push_back(ch[k]);
    return out;
  }
  ColorTransform tr;
  tr.mode = mode;
  tr.perm = perm;
  return apply_color_forward(ch, tr);
}

__attribute__((noinline)) std::vector<Plane> color_inv_frontier(const std::vector<Plane>& pl, int mode, int perm) {
  if (pl.size() < 3 || mode == MODE_IDENTITY) return pl;
  const int H = pl[0].height;
  const int W = pl[0].width;
  if (mode == kModeSubAvg) {
    const int p = std::clamp(perm, 0, 5);
    const int ia = kSubAvgPerms[p][0];
    const int ib = kSubAvgPerms[p][1];
    const int ic = kSubAvgPerms[p][2];
    std::vector<Plane> out(pl.size(), Plane(H, W));
    out[static_cast<size_t>(ia)] = pl[0];
    for (size_t i = 0; i < pl[0].samples.size(); ++i) {
      const int64_t a = pl[0].samples[i];
      const int64_t b = static_cast<int64_t>(pl[1].samples[i]) + a;
      const int64_t c = static_cast<int64_t>(pl[2].samples[i]) + ((a + b) >> 1);
      if (b < -262144 || b > 262144 || c < -262144 || c > 262144) {
        throw CodecError("decoded frontier color inverse overflow");
      }
      out[static_cast<size_t>(ib)].samples[i] = static_cast<int32_t>(b);
      out[static_cast<size_t>(ic)].samples[i] = static_cast<int32_t>(c);
    }
    for (size_t k = 3; k < pl.size(); ++k) out[k] = pl[k];
    return out;
  }
  ColorTransform tr;
  tr.mode = mode;
  tr.perm = perm;
  return apply_color_inverse(pl, tr);
}

__attribute__((always_inline)) inline std::pair<int64_t, int64_t> pixel_bounds_q4(int rct_mode, int c_idx, int maxv,
                                                                                   const Plane* C0, const Plane* C1,
                                                                                   int y, int x) {
  if (rct_mode == MODE_IDENTITY || c_idx == 3) {
    return {0, static_cast<int64_t>(maxv) << 4};
  }
  if (rct_mode == MODE_SUB) {
    if (c_idx == 0) return {0, static_cast<int64_t>(maxv) << 4};
    if (c_idx <= 2 && C0 != nullptr) {
      const int64_t a = C0->samples[y * C0->width + x];
      return {-a << 4, (static_cast<int64_t>(maxv) - a) << 4};
    }
  }
  if (rct_mode == kModeSubAvg) {
    if (c_idx == 0) return {0, static_cast<int64_t>(maxv) << 4};
    if (c_idx == 1 && C0 != nullptr) {
      const int64_t a = C0->samples[y * C0->width + x];
      return {-a << 4, (static_cast<int64_t>(maxv) - a) << 4};
    }
    if (c_idx == 2 && C0 != nullptr && C1 != nullptr) {
      const int idx = y * C0->width + x;
      const int64_t a = C0->samples[idx];
      const int64_t b = a + C1->samples[idx];
      const int64_t mid = (a + b) >> 1;
      return {-mid << 4, (static_cast<int64_t>(maxv) - mid) << 4};
    }
  }
  return {-static_cast<int64_t>(maxv * 4) << 4, static_cast<int64_t>(maxv * 4) << 4};
}

// ============================================================================
// Normative Integer Adaptive Core66 Model (L1/L2-Compact + O(1) Decode Hints)
// ============================================================================
struct AdaptiveCore66 {
  static constexpr int kFracBins = 8;

  struct Slot {
    uint16_t counts[CORE_BINS];
    int32_t cum[CORE_BINS + 1];
    uint16_t pending = 0;
    uint16_t total = 0;

    void init(const int32_t* prior_freq) {
      uint32_t tot = 0;
      int32_t run = 0;
      cum[0] = 0;
      for (int i = 0; i < CORE_BINS; ++i) {
        run += prior_freq[i];
        cum[i + 1] = run;
        int32_t c = (prior_freq[i] + 64) >> 7;
        if (c < 1) c = 1;
        counts[i] = static_cast<uint16_t>(c);
        tot += static_cast<uint32_t>(c);
      }
      total = static_cast<uint16_t>(tot);
      pending = 0;
    }

    inline void observe(int sym) {
      counts[sym] += 2;
      total += 2;
      const uint16_t step = total >= 1024 ? 128 : 32;
      if (++pending >= step) rebuild();
    }

    __attribute__((noinline)) void rebuild() {
      constexpr uint64_t kRemPool = RANS_M - CORE_BINS;
      const uint64_t inv_tot = (kRemPool << 32) / static_cast<uint64_t>(total);
      int32_t freqs[CORE_BINS];
      int32_t sum = 0;
      int best_k = 32;
      int32_t best_c = -1;
      for (int i = 0; i < CORE_BINS; ++i) {
        const int32_t ci = counts[i];
        const int32_t f = 1 + static_cast<int32_t>((static_cast<uint64_t>(ci) * inv_tot) >> 32);
        freqs[i] = f;
        sum += f;
        if (ci > best_c) {
          best_c = ci;
          best_k = i;
        }
      }
      freqs[best_k] += (RANS_M - sum);
      int32_t run = 0;
      cum[0] = 0;
      for (int i = 0; i < CORE_BINS; ++i) {
        run += freqs[i];
        cum[i + 1] = run;
      }
      if (total > 8192) {
        uint32_t t = 0;
        for (int i = 0; i < CORE_BINS; ++i) {
          counts[i] = static_cast<uint16_t>((counts[i] + 1) >> 1);
          t += counts[i];
        }
        total = static_cast<uint16_t>(t);
      }
      pending = 0;
    }
  };

  std::vector<Slot> slots;

  __attribute__((noinline)) void init(int nu, const std::vector<int32_t>& leaf_sigmas) {
    const ParametricTables& tables = get_parametric_tables();
    const size_t nl = leaf_sigmas.size();
    slots.resize(nl * kFracBins);
    for (size_t l = 0; l < nl; ++l) {
      const int sig = std::clamp(int(leaf_sigmas[l]), 0, NUM_SIGMAS - 1);
      for (int fb = 0; fb < kFracBins; ++fb) {
        slots[l * kFracBins + static_cast<size_t>(fb)].init(tables.freqs[nu][sig][fb * 2 + 1]);
      }
    }
  }

  inline Slot& at(int leaf, int frac) {
    return slots[static_cast<size_t>(leaf) * kFracBins + static_cast<size_t>((frac >> 1) & 7)];
  }
};

struct RansOp {
  uint16_t start;
  uint16_t freq_m1;  // freq - 1 fits in uint16_t for 1 <= freq <= 65536!
};

std::vector<uint8_t> pack_rans_ops(const std::vector<RansOp>& ops, BitWriter& raw_bits) {
  RansEncoder enc;
  for (size_t i = ops.size(); i-- > 0;) {
    enc.put(static_cast<int>(ops[i].start), static_cast<int>(ops[i].freq_m1) + 1);
  }
  const auto rans_b = enc.finish();
  const auto tail_b = raw_bits.finish();
  std::vector<uint8_t> out;
  out.reserve(8 + rans_b.size() + tail_b.size());
  const uint32_t rn = static_cast<uint32_t>(rans_b.size());
  const uint32_t tn = static_cast<uint32_t>(tail_b.size());
  for (int k = 0; k < 4; ++k) out.push_back(static_cast<uint8_t>(rn >> (8 * k)));
  for (int k = 0; k < 4; ++k) out.push_back(static_cast<uint8_t>(tn >> (8 * k)));
  out.insert(out.end(), rans_b.begin(), rans_b.end());
  out.insert(out.end(), tail_b.begin(), tail_b.end());
  return out;
}

std::vector<uint8_t> encode_adaptive_rans_stream(const std::vector<int32_t>& residuals, int nu,
                                                 const std::vector<int32_t>& leaf_ids,
                                                 const std::vector<int32_t>& leaf_sigmas,
                                                 const std::vector<int32_t>& frac_idx) {
  const ParametricTables& tables = get_parametric_tables();
  const size_t n = residuals.size();
  AdaptiveCore66 model;
  model.init(nu, leaf_sigmas);

  std::vector<RansOp> ops;
  ops.reserve(n * 2);
  BitWriter raw_bits;

  for (size_t i = 0; i < n; ++i) {
    const int l = leaf_ids[i];
    const int sig = leaf_sigmas[static_cast<size_t>(l)];
    const int b = tables.shifts[sig];
    const int f = frac_idx[i];
    auto& slot = model.at(l, f);

    const int64_t r = residuals[i];
    const int64_t bias = b > 0 ? (1LL << (b - 1)) : 0;
    const int64_t q = b > 0 ? ((r + bias) >> b) : r;
    const int64_t mq = q < 0 ? -q : q;
    const int sym = mq > 32 ? 65 : static_cast<int>(q + 32);

    const int32_t st = slot.cum[sym];
    const int32_t frq = slot.cum[sym + 1] - st;
    ops.push_back(RansOp{static_cast<uint16_t>(st), static_cast<uint16_t>(frq - 1)});
    slot.observe(sym);

    if (sym == 65) {
      raw_bits.write_bits(q < 0 ? 1u : 0u, 1);
      const uint64_t code = static_cast<uint64_t>(mq - 33);
      if (code > (1ull << 24)) throw CodecError("frontier escape overflow");
      raw_bits.write_exp_golomb(static_cast<uint32_t>(code));
    }
    if (b > 0) {
      const int64_t rem = r - ((q << b) - bias);
      if (mq <= TILT_QMAX) {
        const int depth = std::min(b, TILT_MAX_DEPTH);
        const int rem_low = b - depth;
        const int64_t top = rem >> rem_low;
        const int32_t* p0_row = tables.tilt_p0[nu][sig][static_cast<int>(q + TILT_QMAX)];
        int branch = 0;
        for (int lvl = 0; lvl < depth; ++lvl) {
          const int bit = static_cast<int>((top >> (depth - 1 - lvl)) & 1);
          const int node = ((1 << lvl) - 1) + branch;
          const int p0 = p0_row[node];
          if (bit == 0) ops.push_back(RansOp{0, static_cast<uint16_t>(p0 - 1)});
          else ops.push_back(RansOp{static_cast<uint16_t>(p0), static_cast<uint16_t>(RANS_M - p0 - 1)});
          branch = (branch << 1) | bit;
        }
        if (rem_low > 0) raw_bits.write_bits(static_cast<uint32_t>(rem & ((1 << rem_low) - 1)), rem_low);
      } else {
        raw_bits.write_bits(static_cast<uint32_t>(rem), b);
      }
    }
  }
  return pack_rans_ops(ops, raw_bits);
}

struct AdaptiveStreamDecoder {
  RansDecoder rans;
  BitReader raw_bits;
  const ParametricTables& tables;
  int nu = 1;
  bool adaptive = false;
  std::vector<int32_t> leaf_sigmas;
  AdaptiveCore66 model;

  AdaptiveStreamDecoder(std::span<const uint8_t> blob, int nu_in, bool adapt_in,
                        const std::vector<int32_t>& sigmas)
      : rans(split_rans(blob)),
        raw_bits(blob.data() + 8 + split_rans_len(blob), split_extra_len(blob)),
        tables(get_parametric_tables()),
        nu(nu_in),
        adaptive(adapt_in),
        leaf_sigmas(sigmas) {
    if (nu < 0 || nu >= NUM_NUS) throw CodecError("frontier nu out of range");
    if (leaf_sigmas.empty()) throw CodecError("frontier MA tree has no leaves");
    if (adaptive) model.init(nu, leaf_sigmas);
  }

  static uint32_t u32le(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
  }

  static uint32_t split_rans_len(std::span<const uint8_t> b) {
    if (b.size() < 8) throw CodecError("truncated frontier rANS header");
    const uint32_t rn = u32le(b.data());
    const uint32_t tn = u32le(b.data() + 4);
    if (rn < 8 || uint64_t(8) + rn + tn != b.size()) throw CodecError("corrupt frontier rANS payload length");
    return rn;
  }

  static uint32_t split_extra_len(std::span<const uint8_t> b) {
    (void)split_rans_len(b);
    return u32le(b.data() + 4);
  }

  static std::span<const uint8_t> split_rans(std::span<const uint8_t> b) {
    const uint32_t rn = split_rans_len(b);
    return b.subspan(8, rn);
  }

  inline int pull_sym(const int32_t* cum) {
    const int target = rans.cf();
    if (target >= cum[31] && target < cum[34]) {
      const int s = (target < cum[32]) ? 31 : ((target < cum[33]) ? 32 : 33);
      rans.advance(cum[s], cum[s + 1] - cum[s]);
      return s;
    }
    int lo = (target < cum[31]) ? 0 : 34;
    int hi = (target < cum[31]) ? 30 : (CORE_BINS - 1);
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (cum[mid] <= target) lo = mid;
      else hi = mid - 1;
    }
    rans.advance(cum[lo], cum[lo + 1] - cum[lo]);
    return lo;
  }

  inline int32_t read(int leaf, int frac) {
    if (static_cast<unsigned>(leaf) >= static_cast<unsigned>(leaf_sigmas.size())) {
      throw CodecError("frontier leaf index out of range");
    }
    const int sig = leaf_sigmas[static_cast<size_t>(leaf)];
    const int f_clamped = std::clamp(frac, 0, NUM_FRACS - 1);
    const int b = tables.shifts[sig];
    int sym = 0;
    if (adaptive) {
      auto& slot = model.at(leaf, f_clamped);
      sym = pull_sym(slot.cum);
      slot.observe(sym);
    } else {
      sym = pull_sym(tables.cums[nu][sig][f_clamped]);
    }
    int64_t q = 0;
    if (sym == 65) {
      const uint32_t sign = raw_bits.read_bits(1);
      const uint64_t code = raw_bits.read_exp_golomb();
      if (code > (1ull << 24)) throw CodecError("frontier escape quotient out of range");
      const uint64_t mag = code + 33ull;
      q = sign ? -static_cast<int64_t>(mag) : static_cast<int64_t>(mag);
    } else {
      q = static_cast<int64_t>(sym) - 32;
    }
    if (b <= 0) return static_cast<int32_t>(q);
    const int64_t bias = 1LL << (b - 1);
    const int64_t mq = q < 0 ? -q : q;
    int64_t rem = 0;
    if (mq <= TILT_QMAX) {
      const int depth = std::min(b, TILT_MAX_DEPTH);
      const int rem_low = b - depth;
      const int32_t* p0_row = tables.tilt_p0[nu][sig][static_cast<int>(q + TILT_QMAX)];
      int branch = 0;
      for (int lvl = 0; lvl < depth; ++lvl) {
        const int node = ((1 << lvl) - 1) + branch;
        const int p0 = p0_row[node];
        const int cf = rans.cf();
        const int bit = cf < p0 ? 0 : 1;
        if (bit == 0) rans.advance(0, p0);
        else rans.advance(p0, RANS_M - p0);
        branch = (branch << 1) | bit;
      }
      const int64_t low = rem_low > 0 ? static_cast<int64_t>(raw_bits.read_bits(rem_low)) : 0;
      rem = (static_cast<int64_t>(branch) << rem_low) | low;
    } else {
      rem = static_cast<int64_t>(raw_bits.read_bits(b));
    }
    const int64_t full = ((q << b) - bias) + rem;
    if (full < -524288 || full > 524288) throw CodecError("frontier residual out of range");
    return static_cast<int32_t>(full);
  }

  __attribute__((noinline)) void check_finished() const {
    rans.check_finished();
    raw_bits.check_zero_tail();
  }
};

// ============================================================================
// MA Context Tree (Offline Universal Topology + Fast Histogram Custom Tree)
// ============================================================================
struct MATree {
  struct Node {
    int16_t thr = 0;
    int8_t prop = -1;
    uint8_t sigma = 4;
    int16_t left = -1;
    int16_t right = -1;
    int16_t leaf_id = -1;
  };
  std::vector<Node> nodes;
  std::vector<int32_t> leaf_sigmas;

  __attribute__((always_inline)) inline int walk(const int16_t* props) const {
    const Node* nds = nodes.data();
    int cur = 0;
    while (nds[cur].left >= 0) {
      const Node& nd = nds[cur];
      cur = (props[nd.prop] <= nd.thr) ? nd.left : nd.right;
    }
    return nds[cur].leaf_id;
  }

  void write_custom(BitWriter& w) const {
    auto rec = [&](auto&& self, int id) -> void {
      const Node& nd = nodes[static_cast<size_t>(id)];
      if (nd.left < 0) {
        w.write_bits(1, 1);
        w.write_bits(static_cast<uint32_t>(nd.sigma & 31), 5);
        return;
      }
      w.write_bits(0, 1);
      w.write_bits(static_cast<uint32_t>(nd.prop & 15), 4);
      w.write_signed_golomb(nd.thr);
      self(self, nd.left);
      self(self, nd.right);
    };
    rec(rec, 0);
  }

  static MATree from_universal(int tree_mode) {
    const priors::UnivTreeNode* src = nullptr;
    size_t count = 0;
    switch (tree_mode) {
      case 0: src = priors::kUnivTreeLuma; count = std::size(priors::kUnivTreeLuma); break;
      case 1: src = priors::kUnivTreeChromaRas; count = std::size(priors::kUnivTreeChromaRas); break;
      case 2: src = priors::kUnivTreeCosS1; count = std::size(priors::kUnivTreeCosS1); break;
      case 3: src = priors::kUnivTreeCosS2; count = std::size(priors::kUnivTreeCosS2); break;
      default: src = priors::kUnivTreeCosS3; count = std::size(priors::kUnivTreeCosS3); break;
    }
    const int n_leaves = priors::kUnivTreeLeaves[std::clamp(tree_mode, 0, 4)];
    MATree t;
    t.nodes.resize(count);
    t.leaf_sigmas.assign(static_cast<size_t>(n_leaves), 4);
    for (size_t i = 0; i < count; ++i) {
      t.nodes[i].thr = src[i].thr;
      t.nodes[i].prop = src[i].prop;
      t.nodes[i].sigma = 4;
      t.nodes[i].left = src[i].left;
      t.nodes[i].right = src[i].right;
      t.nodes[i].leaf_id = src[i].left < 0 ? static_cast<int16_t>(src[i].leaf_id) : -1;
    }
    return t;
  }

  __attribute__((noinline)) static MATree read(BitReader& r, int n_props, int tree_mode) {
    const bool is_custom = r.read_bits(1) == 1;
    if (!is_custom) {
      MATree t = from_universal(tree_mode);
      const int nl = static_cast<int>(t.leaf_sigmas.size());
      int prev = static_cast<int>(r.read_bits(5));
      t.leaf_sigmas[0] = prev;
      for (int l = 1; l < nl; ++l) {
        const int d = r.read_signed_golomb();
        if (d < -64 || d > 64) throw CodecError("MA universal sigma delta out of range");
        prev = std::clamp(prev + d, 0, NUM_SIGMAS - 1);
        t.leaf_sigmas[static_cast<size_t>(l)] = prev;
      }
      for (auto& nd : t.nodes) {
        if (nd.left < 0 && nd.leaf_id >= 0 && static_cast<size_t>(nd.leaf_id) < t.leaf_sigmas.size()) {
          nd.sigma = static_cast<uint8_t>(t.leaf_sigmas[static_cast<size_t>(nd.leaf_id)]);
        }
      }
      return t;
    }

    MATree t;
    auto rec = [&](auto&& self, int depth) -> int {
      if (depth > 24 || t.nodes.size() >= 512) throw CodecError("MA tree limit exceeded");
      const int id = static_cast<int>(t.nodes.size());
      t.nodes.push_back(Node{});
      if (r.read_bits(1) == 1) {
        const int sig = static_cast<int>(r.read_bits(5));
        const int lid = static_cast<int>(t.leaf_sigmas.size());
        t.leaf_sigmas.push_back(sig);
        t.nodes[static_cast<size_t>(id)].sigma = static_cast<uint8_t>(sig);
        t.nodes[static_cast<size_t>(id)].leaf_id = static_cast<int16_t>(lid);
        return id;
      }
      const int p = static_cast<int>(r.read_bits(4));
      if (p < 0 || p >= n_props) throw CodecError("MA tree property out of range");
      const int thr = r.read_signed_golomb();
      if (thr < -32768 || thr > 32767) throw CodecError("MA tree threshold out of range");
      const int l = self(self, depth + 1);
      const int rg = self(self, depth + 1);
      t.nodes[static_cast<size_t>(id)].prop = static_cast<int8_t>(p);
      t.nodes[static_cast<size_t>(id)].thr = static_cast<int16_t>(thr);
      t.nodes[static_cast<size_t>(id)].left = static_cast<int16_t>(l);
      t.nodes[static_cast<size_t>(id)].right = static_cast<int16_t>(rg);
      return id;
    };
    rec(rec, 0);
    return t;
  }
};

struct OpenedMAStream {
  MATree tree;
  int nu = 1;
  bool use_adapt = false;
  std::span<const uint8_t> payload;
};

__attribute__((noinline)) OpenedMAStream open_ma_stream(const uint8_t* data, size_t size, size_t& pos, int n_props,
                                                        int tree_mode) {
  if (pos + 2 > size) throw CodecError("truncated MA stream header");
  const uint16_t hn = static_cast<uint16_t>(data[pos] | (uint16_t(data[pos + 1]) << 8));
  pos += 2;
  if (hn == 0 || pos + hn + 4 > size) throw CodecError("truncated MA stream header");
  BitReader hr(data + pos, hn);
  pos += hn;
  OpenedMAStream st;
  st.nu = static_cast<int>(hr.read_bits(2));
  st.use_adapt = hr.read_bits(1) == 1;
  st.tree = MATree::read(hr, n_props, tree_mode);
  hr.check_zero_tail();
  const uint32_t pn = uint32_t(data[pos]) | (uint32_t(data[pos + 1]) << 8) |
                      (uint32_t(data[pos + 2]) << 16) | (uint32_t(data[pos + 3]) << 24);
  pos += 4;
  if (pn > size - pos) throw CodecError("truncated MA stream payload");
  st.payload = std::span<const uint8_t>(data + pos, pn);
  pos += pn;
  return st;
}

__attribute__((noinline)) std::vector<std::array<int16_t, 36>> read_sector_ridge(BitReader& hr, int role, int n_sec,
                                                                                 int D) {
  std::vector<std::array<int16_t, 36>> out_w(static_cast<size_t>(n_sec), std::array<int16_t, 36>{});
  for (int s = 0; s < n_sec; ++s) {
    for (int a = 0; a < D; ++a) out_w[static_cast<size_t>(s)][static_cast<size_t>(a)] = get_prior_weight(role, s, a);
  }
  if (hr.read_bits(1) == 0) return out_w;

  std::array<int16_t, 36> dg{};
  for (int a = 0; a < D; ++a) {
    const int g = hr.read_signed_golomb();
    if (g < -4096 || g > 4096) throw CodecError("frontier ridge global delta out of range");
    dg[static_cast<size_t>(a)] = static_cast<int16_t>(g);
  }
  for (int s = 0; s < n_sec; ++s) {
    for (int a = 0; a < D; ++a) {
      out_w[static_cast<size_t>(s)][static_cast<size_t>(a)] =
          static_cast<int16_t>(std::clamp<int>(int(out_w[static_cast<size_t>(s)][static_cast<size_t>(a)]) + dg[a],
                                               -2047, 2047));
    }
    if (hr.read_bits(1) == 1) {
      for (int a = 0; a < D; ++a) {
        const int d = hr.read_signed_golomb();
        if (d < -4096 || d > 4096) throw CodecError("frontier ridge sector delta out of range");
        out_w[static_cast<size_t>(s)][static_cast<size_t>(a)] =
            static_cast<int16_t>(std::clamp<int>(int(out_w[static_cast<size_t>(s)][static_cast<size_t>(a)]) + d,
                                                 -2047, 2047));
      }
    }
  }
  return out_w;
}

// ============================================================================
// Causal L-Template Non-Local Patch Predictor (Fused Single-Hash LRU-2)
// ============================================================================
struct PatchPredictor {
  static constexpr int kBuckets = 8192;
  std::vector<uint32_t> table;
  int qsh = 0;
  int grad_thr = 12;
  bool enabled = true;

  __attribute__((noinline)) PatchPredictor(int maxv, bool en, bool is_chroma = false)
      : table(en ? kBuckets * 2 : 0, 0u),
        qsh(maxv > 255 ? 8 : 0),
        grad_thr(is_chroma ? 18 : 12),
        enabled(en) {}

  inline uint32_t hash_key(int32_t d1, int32_t d2, int32_t d3, int32_t d4) const {
    const int sh = qsh + 1;
    const uint32_t h = (static_cast<uint32_t>(d1 >> sh) * 73856093u) ^
                       (static_cast<uint32_t>(d2 >> sh) * 19349663u) ^
                       (static_cast<uint32_t>(d3 >> sh) * 83492791u) ^
                       (static_cast<uint32_t>(d4 >> sh) * 2654435761u);
    return (h ^ (h >> 13)) & (kBuckets - 1);
  }

  // Fused causal query + LRU-2 update in ONE hash computation and ONE cache line!
  inline std::pair<int64_t, int> query_and_update(const Plane& P, int y, int x, int32_t Wv, int32_t Nv,
                                                  int32_t NW, int32_t NE, int32_t WW, int32_t NN,
                                                  int64_t fallback_q4) {
    if (!enabled || y < 2 || x < 2 || x + 1 >= P.width || y > 32767 || x > 32767) return {fallback_q4, 0};
    const int32_t d1 = Wv - NW;
    const int32_t d2 = Nv - NW;
    const int32_t d3 = NE - Nv;
    if (((iabs32(d1) + iabs32(d2) + iabs32(d3)) >> qsh) < grad_thr) return {fallback_q4, 0};
    const int32_t d4 = WW - Wv;
    const uint32_t h = hash_key(d1, d2, d3, d4);
    uint32_t* tb = table.data() + h * 2;
    const uint32_t cands[2] = {tb[0], tb[1]};
    tb[1] = cands[0];
    tb[0] = (static_cast<uint32_t>(y) << 16) | static_cast<uint32_t>(x);

    int64_t best_num = 0;
    int best_den = 0;
    int best_sad = 999999;
    const int W = P.width;
    const int32_t* s = P.samples.data();
    const int32_t d5 = NN - Nv;
    for (int k = 0; k < 2; ++k) {
      const uint32_t packed = cands[k];
      if (packed == 0) break;
      const int my = static_cast<int>(packed >> 16);
      const int mx = static_cast<int>(packed & 0xffffu);
      if ((y - my) + iabs32(mx - x) > 192) continue;
      const int row_m = my * W;
      const int row_m1 = (my - 1) * W;
      const int32_t mNW = s[row_m1 + mx - 1];
      const int32_t mW = s[row_m + mx - 1];
      const int32_t mN = s[row_m1 + mx];
      const int64_t sad64 = (int64_t(iabs32((mW - mNW) - d1)) + int64_t(iabs32((mN - mNW) - d2)) +
                             int64_t(iabs32((s[row_m1 + mx + 1] - mN) - d3)) +
                             int64_t(iabs32((s[row_m + mx - 2] - mW) - d4)) +
                             int64_t(iabs32((s[(my - 2) * W + mx] - mN) - d5))) >>
                            qsh;
      if (sad64 < 20) {
        const int sad = static_cast<int>(sad64);
        const int wt = kFast.patch_wt[sad];
        const int64_t cand = static_cast<int64_t>(NW) + (static_cast<int64_t>(s[row_m + mx]) - mNW);
        best_num += static_cast<int64_t>(wt) * (cand << 4);
        best_den += wt;
        if (sad < best_sad) best_sad = sad;
        if (sad <= 1) break;
      }
    }
    if (best_den == 0) return {fallback_q4, 0};
    const int conf = best_sad == 0 ? 15 : (best_sad <= 2 ? 10 : (best_sad <= 6 ? 5 : 2));
    return {mul_patch_recip20(best_num, best_den), conf};
  }
};

// ============================================================================
// Shared Raster Predictor State (L1-Ring-Buffered Pure Integer Encode & Decode)
// ============================================================================
struct RasterState {
  static constexpr int KEXP = 8;
  struct alignas(32) Err8 {
    uint32_t e[KEXP];
  };
  int H;
  int W;
  int c_idx;
  int rct_mode;
  int maxv;
  int qsh;
  int D;
  int n_sec;
  int stride;
  int err_stride;
  std::vector<Err8> perr;
  std::vector<int32_t> wp_err;
  std::vector<int32_t> true_err_ring;  // 4-row power-of-2 L1 ring buffer (1-cycle & 3 indexing!)
  PatchPredictor patcher;
  uint32_t mw[KEXP];

  __attribute__((noinline)) RasterState(int h, int w, int c, int rm, int mv, bool has_c0, bool has_c1, bool use_patch)
      : H(h),
        W(w),
        c_idx(c),
        rct_mode(rm),
        maxv(mv),
        qsh(mv > 255 ? 8 : 0),
        D(!has_c0 ? 21 : (!has_c1 ? 30 : 36)),
        n_sec(16),
        stride(w + 2),
        err_stride(w + 4),
        perr(static_cast<size_t>((w + 2) * 2), Err8{}),
        wp_err(static_cast<size_t>((w + 2) * 2), 0),
        true_err_ring(static_cast<size_t>((w + 4) * 4), 0),
        patcher(mv, use_patch, has_c0) {
    const uint32_t* src_mw = has_c0 ? priors::kWpWeightsChroma : priors::kWpWeightsLuma;
    for (int k = 0; k < KEXP; ++k) mw[k] = src_mw[k];
  }

  inline void reset_row_state() {
    std::fill(perr.begin(), perr.end(), Err8{});
    std::fill(wp_err.begin(), wp_err.end(), 0);
    std::fill(true_err_ring.begin(), true_err_ring.end(), 0);
  }

  inline int64_t predict_features(const Plane& P, const Plane* C0, const Plane* C0_err,
                                  const Plane* C1, const Plane* C1_err, int y, int x,
                                  int32_t* f, int& sec, int16_t* prp,
                                  int64_t pr_out[KEXP]) {
    const int cur = (y & 1) ? 0 : stride;
    const int prv = (y & 1) ? stride : 0;
    const int32_t* ps = P.samples.data();
    const int row = y * W;
    const int row1 = (y - 1) * W;
    const int row2 = (y - 2) * W;
    const bool interior = (static_cast<unsigned>(y - 2) < static_cast<unsigned>(H - 3)) &&
                          (static_cast<unsigned>(x - 2) < static_cast<unsigned>(W - 4));

    int32_t Nv, Wv, NE, NW, NN, WW, NWW, NNW, NNE, NEE;
    if (interior) {
      const int idx = row + x;
      const int idx1 = row1 + x;
      const int idx2 = row2 + x;
      Nv = ps[idx1];
      Wv = ps[idx - 1];
      NE = ps[idx1 + 1];
      NW = ps[idx1 - 1];
      NN = ps[idx2];
      WW = ps[idx - 2];
      NWW = ps[idx1 - 2];
      NNW = ps[idx2 - 1];
      NNE = ps[idx2 + 1];
      NEE = ps[idx1 + 2];
    } else {
      Nv = y > 0 ? ps[row1 + x] : (x > 0 ? ps[row + x - 1] : 0);
      Wv = x > 0 ? ps[row + x - 1] : Nv;
      NE = y > 0 ? ps[row1 + (x + 1 < W ? x + 1 : W - 1)] : Nv;
      NW = (y > 0 && x > 0) ? ps[row1 + x - 1] : Nv;
      NN = y > 1 ? ps[row2 + x] : Nv;
      WW = x > 1 ? ps[row + x - 2] : Wv;
      NWW = (y > 0 && x > 1) ? ps[row1 + x - 2] : Wv;
      NNW = (y > 1 && x > 0) ? ps[row2 + x - 1] : Nv;
      NNE = (y > 1 && x + 1 < W) ? ps[row2 + x + 1] : NE;
      NEE = (y > 0 && x + 2 < W) ? ps[row1 + x + 2] : NE;
    }

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

    const auto [patch_q4, patch_conf] = patcher.enabled
                                            ? patcher.query_and_update(P, y, x, Wv, Nv, NW, NE, WW, NN, int64_t(med) << 4)
                                            : std::pair<int64_t, int>{int64_t(med) << 4, 0};
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
    const Err8& eN = perr[static_cast<size_t>(pN)];
    const Err8& eNE = perr[static_cast<size_t>(pNE)];
    const Err8& eNW = perr[static_cast<size_t>(pNW)];
    if (x > 0) {
      const Err8& eW = perr[static_cast<size_t>(pW)];
      for (int k = 0; k < KEXP; ++k) {
        const uint64_t acc = uint64_t(eN.e[k]) + uint64_t(eNE.e[k]) + uint64_t(eNW.e[k]) + uint64_t(eW.e[k]);
        wts[k] = wp_wt(acc >> qsh, mw[k]);
      }
    } else {
      for (int k = 0; k < KEXP; ++k) {
        const uint64_t acc = uint64_t(eN.e[k]) + uint64_t(eNE.e[k]) + uint64_t(eNW.e[k]);
        wts[k] = wp_wt(acc >> qsh, mw[k]);
      }
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

    // 4-row power-of-2 ring buffer (1-cycle & 3 indexing, zero division!)
    const int32_t* er0 = true_err_ring.data() + (y & 3) * err_stride + 2;
    const int32_t* er1 = true_err_ring.data() + ((y - 1) & 3) * err_stride + 2;
    const int32_t* er2 = true_err_ring.data() + ((y - 2) & 3) * err_stride + 2;
    const int32_t reW = x > 0 ? er0[x - 1] : 0;
    const int32_t reWW = x > 1 ? er0[x - 2] : 0;
    const int32_t reN = y > 0 ? er1[x] : 0;
    const int32_t reNW = (y > 0 && x > 0) ? er1[x - 1] : 0;
    const int32_t reNE = (y > 0 && x + 1 < W) ? er1[x + 1] : 0;
    const int32_t reNN = y > 1 ? er2[x] : 0;

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
      prp[0] = static_cast<int16_t>(flog2_u64(2ull * (ew + en) + enw + ene + ((eww + enn) >> 1) + 1ull));
      prp[1] = static_cast<int16_t>(flog2_u32(ew + 1));
      prp[2] = static_cast<int16_t>(flog2_u32(en + 1));
      prp[3] = static_cast<int16_t>(flog2_u32(enw + ene + 1));
      prp[4] = static_cast<int16_t>(flog2_u32(static_cast<uint32_t>(iabs32(f[0])) + 1));
      prp[5] = static_cast<int16_t>(flog2_u32(static_cast<uint32_t>(iabs32(f[1])) + 1));
      prp[6] = static_cast<int16_t>(flog2_u64(static_cast<uint64_t>(iabs64(int64_t(f[0]) - f[1])) + 1ull));
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
                        : flog2_u64(static_cast<uint64_t>(dh_g) + static_cast<uint64_t>(dv_g) + 1ull));
    }
    return pred_q4;
  }

  inline void commit(int y, int x, int32_t truth, int64_t pred_q4, int64_t mu_q4,
                     const int64_t pr_out[KEXP]) {
    const int cur = (y & 1) ? 0 : stride;
    const int prv = (y & 1) ? stride : 0;
    const int64_t val_q4 = static_cast<int64_t>(truth) << 4;
    true_err_ring[static_cast<size_t>((y & 3) * err_stride + 2 + x)] =
        static_cast<int32_t>(std::clamp<int64_t>(val_q4 - mu_q4, -8388608, 8388608));
    wp_err[static_cast<size_t>(cur + x)] =
        static_cast<int32_t>(std::clamp<int64_t>(pred_q4 - val_q4, -16777216, 16777216));
    Err8& ec = perr[static_cast<size_t>(cur + x)];
    Err8& ep = perr[static_cast<size_t>(prv + x + 1)];
    for (int m = 0; m < KEXP; ++m) {
      const uint64_t e64 = static_cast<uint64_t>((iabs64(pr_out[m] - val_q4) + 8) >> 4);
      const uint32_t e = static_cast<uint32_t>(std::min<uint64_t>(e64, 1u << 28));
      ec.e[m] = e;
      ep.e[m] += e;
    }
  }
};

// ============================================================================
// 2x2 Coset-Scheduled Chroma Interpolation (Strictly Causal Parity-Safe)
// ============================================================================
constexpr int kStageNbs1[12][2] = {
    {-1, -1}, {-1, 1}, {1, -1}, {1, 1}, {-3, -1}, {-3, 1}, {3, -1}, {3, 1}, {-1, -3}, {1, -3}, {-1, 3}, {1, 3}};
constexpr int kStageNbs2[10][2] = {
    {0, -1}, {0, 1}, {-1, 0}, {1, 0}, {0, -3}, {0, 3}, {-2, -1}, {-2, 1}, {2, -1}, {2, 1}};
constexpr int kStageNbs3[10][2] = {
    {-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}, {-3, 0}, {3, 0}};

inline int64_t interp_features_exact(const Plane& P, const Plane& C0, const Plane& C0_err,
                                     const Plane* C1, const Plane* C1_err, const Plane& stage_err,
                                     int maxv, int stg, int y, int x, int32_t* f, int& sec,
                                     int16_t* prp) {
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
  const int64_t sum4 = int64_t(nb_vals[0]) + nb_vals[1] + nb_vals[2] + nb_vals[3];
  const int64_t b_q4 = sum4 << 2;
  const int32_t b32 = static_cast<int32_t>(b_q4);

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
  const int32_t bilat_diff = static_cast<int32_t>(mul_recip24(bn, bd) - b_q4);

  int m = 0;
  for (int k = 0; k < Knb; ++k) f[m++] = (nb_vals[k] << 4) - b32;
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
    uint64_t prev_e = 0;
    for (int k = 0; k < 4; ++k) prev_e += static_cast<uint64_t>(stage_err.samples[ny_arr[k] * W + nx_arr[k]]);
    const uint64_t causal_same =
        (x >= 2 ? static_cast<uint64_t>(stage_err.samples[y * W + x - 2]) : 0ull) +
        (y >= 2 ? static_cast<uint64_t>(stage_err.samples[(y - 2) * W + x]) : 0ull);
    uint64_t c_span = 0;
    for (int k = 0; k < 4; ++k) c_span += static_cast<uint64_t>(iabs32(f[k]));
    prp[0] = static_cast<int16_t>(flog2_u64(causal_same + 1));
    prp[1] = static_cast<int16_t>(flog2_u64(prev_e + 1));
    prp[2] = static_cast<int16_t>(flog2_u64(c_span + 1));
    prp[3] = static_cast<int16_t>(flog2_u32(static_cast<uint32_t>(iabs32(bilat_diff)) + 1));
    prp[4] = static_cast<int16_t>(flog2_u32(static_cast<uint32_t>(iabs32(C0_err.samples[y * W + x])) + 1));
    prp[5] = static_cast<int16_t>(flog2_u64(uint64_t(iabs32(gW - gE)) + uint64_t(iabs32(gN - gS)) + 1ull));
    prp[6] = static_cast<int16_t>(sec);
    prp[7] = static_cast<int16_t>((f[0] - f[1] >= 0 ? 1 : -1) *
                                  flog2_u64(static_cast<uint64_t>(iabs64(int64_t(f[0]) - f[1])) + 1ull));
    prp[8] = static_cast<int16_t>(
        C1 != nullptr ? flog2_u32(static_cast<uint32_t>(iabs32(C1_err->samples[y * W + x])) + 1) : 0);
    prp[9] = static_cast<int16_t>(
        flog2_u64(uint64_t(iabs32(2 * g0 - gW - gE)) + uint64_t(iabs32(2 * g0 - gN - gS)) + 1ull));
    prp[10] = static_cast<int16_t>(flog2_u64(uint64_t(iabs32(f[0])) + uint64_t(iabs32(f[1])) + 1ull));
  }
  return b_q4;
}

// ============================================================================
// Encode-Only Linear Solver, Histogram MA Tree Builder & Fast Pre-Screening
// ============================================================================
bool solve_spd(std::vector<double>& A, std::vector<double>& b, int D, std::vector<double>& x) {
  for (int k = 0; k < D; ++k) {
    int piv = k;
    for (int i = k + 1; i < D; ++i) {
      if (std::fabs(A[static_cast<size_t>(i * D + k)]) > std::fabs(A[static_cast<size_t>(piv * D + k)])) piv = i;
    }
    if (!(std::fabs(A[static_cast<size_t>(piv * D + k)]) > 1e-9) ||
        !std::isfinite(A[static_cast<size_t>(piv * D + k)])) {
      return false;
    }
    if (piv != k) {
      for (int j = k; j < D; ++j) {
        std::swap(A[static_cast<size_t>(k * D + j)], A[static_cast<size_t>(piv * D + j)]);
      }
      std::swap(b[static_cast<size_t>(k)], b[static_cast<size_t>(piv)]);
    }
    for (int i = k + 1; i < D; ++i) {
      const double f = A[static_cast<size_t>(i * D + k)] / A[static_cast<size_t>(k * D + k)];
      for (int j = k + 1; j < D; ++j) {
        A[static_cast<size_t>(i * D + j)] -= f * A[static_cast<size_t>(k * D + j)];
      }
      b[static_cast<size_t>(i)] -= f * b[static_cast<size_t>(k)];
    }
  }
  x.assign(static_cast<size_t>(D), 0.0);
  for (int i = D - 1; i >= 0; --i) {
    double sum = b[static_cast<size_t>(i)];
    for (int j = i + 1; j < D; ++j) sum -= A[static_cast<size_t>(i * D + j)] * x[static_cast<size_t>(j)];
    const double diag = A[static_cast<size_t>(i * D + i)];
    if (diag == 0.0 || !std::isfinite(diag) || !std::isfinite(sum)) return false;
    x[static_cast<size_t>(i)] = sum / diag;
    if (!std::isfinite(x[static_cast<size_t>(i)])) return false;
  }
  return true;
}

struct SectorGram {
  double G[36 * 36]{};
  double R[36]{};
  double Y2 = 0.0;
  int64_t N = 0;

  void add(int D, const int32_t* f, double y_res) {
    N += 1;
    Y2 += y_res * y_res;
    for (int a = 0; a < D; ++a) {
      const double fa = static_cast<double>(f[a]);
      R[a] += fa * y_res;
      for (int b = a; b < D; ++b) G[a * D + b] += fa * static_cast<double>(f[b]);
    }
  }

  double eval_quad(int D, const double* w) const {
    double e = Y2;
    for (int a = 0; a < D; ++a) {
      e -= 2.0 * w[a] * R[a];
      e += w[a] * w[a] * G[a * D + a];
      for (int b = a + 1; b < D; ++b) e += 2.0 * w[a] * w[b] * G[a * D + b];
    }
    return std::max(0.0, e);
  }
};

int golomb_signed_bits(int v) {
  const uint64_t mag = (static_cast<uint64_t>(iabs64(v)) << 1) + 1ull;
  return 2 * (64 - __builtin_clzll(mag)) - 1;
}

// Solves Bayesian MAP Ridge shrunk toward offline W_prior in O(D^3) from subsampled SectorGrams.
// Returns true if custom MAP Ridge weights were written, false if Prior-Only (0 bytes) won MDL.
bool solve_and_write_map_ridge(BitWriter& hw, int role, int n_sec, int D, int total_N,
                               const std::vector<SectorGram>& sgrams,
                               std::vector<std::array<int16_t, 36>>& out_w) {
  out_w.assign(static_cast<size_t>(n_sec), std::array<int16_t, 36>{});
  for (int s = 0; s < n_sec; ++s) {
    for (int a = 0; a < D; ++a) out_w[static_cast<size_t>(s)][static_cast<size_t>(a)] = get_prior_weight(role, s, a);
  }
  int64_t sub_N = 0;
  for (int s = 0; s < n_sec; ++s) sub_N += sgrams[static_cast<size_t>(s)].N;
  if (sub_N < 64 || D == 0) {
    hw.write_bits(0, 1);
    return false;
  }
  const double inv_sub_ratio = static_cast<double>(total_N) / static_cast<double>(sub_N);

  std::vector<double> Gg(static_cast<size_t>(D * D), 0.0);
  std::vector<double> Rg(static_cast<size_t>(D), 0.0);
  double Y2g = 0.0;
  for (int s = 0; s < n_sec; ++s) {
    const auto& sg = sgrams[static_cast<size_t>(s)];
    Y2g += sg.Y2;
    for (int a = 0; a < D; ++a) {
      Rg[static_cast<size_t>(a)] += sg.R[a];
      for (int b = a; b < D; ++b) Gg[static_cast<size_t>(a * D + b)] += sg.G[a * D + b];
    }
  }
  std::vector<double> Gg_reg = Gg;
  for (int a = 0; a < D; ++a) {
    for (int b = 0; b < a; ++b) Gg_reg[static_cast<size_t>(a * D + b)] = Gg_reg[static_cast<size_t>(b * D + a)];
    Gg_reg[static_cast<size_t>(a * D + a)] += std::max(Gg_reg[static_cast<size_t>(a * D + a)] * 2e-4, 512.0);
  }
  std::vector<double> sol_g;
  if (!solve_spd(Gg_reg, Rg, D, sol_g)) {
    hw.write_bits(0, 1);
    return false;
  }

  std::array<int16_t, 36> dg{};
  double dg_f[36]{};
  int global_hdr_bits = 1 + n_sec;
  for (int a = 0; a < D; ++a) {
    dg[static_cast<size_t>(a)] =
        static_cast<int16_t>(std::clamp<int>(static_cast<int>(std::llround(sol_g[static_cast<size_t>(a)] * 256.0)),
                                             -1536, 1536));
    dg_f[a] = static_cast<double>(dg[static_cast<size_t>(a)]) / 256.0;
    global_hdr_bits += golomb_signed_bits(dg[static_cast<size_t>(a)]);
  }

  double e_after_g = Y2g;
  for (int a = 0; a < D; ++a) {
    e_after_g -= 2.0 * dg_f[a] * Rg[static_cast<size_t>(a)];
    e_after_g += dg_f[a] * dg_f[a] * Gg[static_cast<size_t>(a * D + a)];
    for (int b = a + 1; b < D; ++b) e_after_g += 2.0 * dg_f[a] * dg_f[b] * Gg[static_cast<size_t>(a * D + b)];
  }
  e_after_g = std::max(0.0, e_after_g);
  const double mse0 = Y2g / static_cast<double>(sub_N);
  const double mseg_raw = e_after_g / static_cast<double>(sub_N);
  const double mseg = mseg_raw * (static_cast<double>(sub_N + D) / std::max(1.0, static_cast<double>(sub_N - D)));
  const double total_gain_g =
      0.5 * static_cast<double>(total_N) * (std::log2(mse0 + 16.0) - std::log2(mseg + 16.0));

  // Try per-sector MAP deltas on top of dg (with FPE degrees-of-freedom penalty against subsample overfitting!)
  std::vector<bool> use_sec(static_cast<size_t>(n_sec), false);
  std::vector<std::array<int16_t, 36>> ds_all(static_cast<size_t>(n_sec), std::array<int16_t, 36>{});
  double total_sec_net_gain = 0.0;

  for (int s = 0; s < n_sec; ++s) {
    const auto& sg = sgrams[static_cast<size_t>(s)];
    if (sg.N < static_cast<int64_t>(8 * D)) continue;
    const double rel_reg = 8e-4 + 4.0 / static_cast<double>(sg.N);
    std::vector<double> Gs(static_cast<size_t>(D * D), 0.0);
    std::vector<double> Rs(static_cast<size_t>(D), 0.0);
    for (int a = 0; a < D; ++a) {
      double r_rem = sg.R[a];
      for (int b = 0; b < D; ++b) {
        const double gab = a <= b ? sg.G[a * D + b] : sg.G[b * D + a];
        Gs[static_cast<size_t>(a * D + b)] = gab;
        r_rem -= gab * dg_f[b];
      }
      Rs[static_cast<size_t>(a)] = r_rem;
      Gs[static_cast<size_t>(a * D + a)] += std::max(Gs[static_cast<size_t>(a * D + a)] * rel_reg, 4096.0);
    }
    std::vector<double> sol_s;
    if (!solve_spd(Gs, Rs, D, sol_s)) continue;
    double tot_f[36]{};
    int shdr = 0;
    for (int a = 0; a < D; ++a) {
      const int p0 = get_prior_weight(role, s, a);
      const int base_w = std::clamp(p0 + int(dg[static_cast<size_t>(a)]), -2047, 2047);
      const int target_w = std::clamp(
          base_w + static_cast<int>(std::llround(sol_s[static_cast<size_t>(a)] * 256.0)), -2047, 2047);
      const int d = target_w - base_w;
      ds_all[static_cast<size_t>(s)][static_cast<size_t>(a)] = static_cast<int16_t>(d);
      tot_f[a] = static_cast<double>(dg[static_cast<size_t>(a)] + d) / 256.0;
      shdr += golomb_signed_bits(d);
    }
    const double eg_s = sg.eval_quad(D, dg_f) / static_cast<double>(sg.N);
    const double es_raw = sg.eval_quad(D, tot_f) / static_cast<double>(sg.N);
    const double es_s = es_raw * (static_cast<double>(sg.N + D) / std::max(1.0, static_cast<double>(sg.N - D)));
    const double est_sec_N = static_cast<double>(sg.N) * inv_sub_ratio;
    const double s_gain = 0.5 * est_sec_N * (std::log2(eg_s + 16.0) - std::log2(es_s + 16.0));
    if (s_gain > static_cast<double>(shdr) + 12.0) {
      use_sec[static_cast<size_t>(s)] = true;
      total_sec_net_gain += (s_gain - static_cast<double>(shdr));
    }
  }

  // Strict out-of-sample MDL gate against Prior-Only (0 bytes)!
  if (total_gain_g + total_sec_net_gain <= static_cast<double>(global_hdr_bits) + 8.0) {
    hw.write_bits(0, 1);
    return false;
  }

  hw.write_bits(1, 1);
  for (int a = 0; a < D; ++a) hw.write_signed_golomb(dg[static_cast<size_t>(a)]);
  for (int s = 0; s < n_sec; ++s) {
    for (int a = 0; a < D; ++a) {
      out_w[static_cast<size_t>(s)][static_cast<size_t>(a)] =
          static_cast<int16_t>(std::clamp<int>(int(out_w[static_cast<size_t>(s)][static_cast<size_t>(a)]) + dg[a],
                                               -2047, 2047));
    }
    if (use_sec[static_cast<size_t>(s)]) {
      hw.write_bits(1, 1);
      for (int a = 0; a < D; ++a) {
        const int d = ds_all[static_cast<size_t>(s)][static_cast<size_t>(a)];
        hw.write_signed_golomb(d);
        out_w[static_cast<size_t>(s)][static_cast<size_t>(a)] =
            static_cast<int16_t>(std::clamp<int>(int(out_w[static_cast<size_t>(s)][static_cast<size_t>(a)]) + d,
                                                 -2047, 2047));
      }
    } else {
      hw.write_bits(0, 1);
    }
  }
  return true;
}

struct HistCostTables {
  double neg_log[65536];
  uint16_t neg_log_q6[65536];
  float sigma_rm[NUM_SIGMAS][32];
};

const HistCostTables& get_hist_cost_tables() {
  static const HistCostTables t = [] {
    HistCostTables h{};
    const ParametricTables& ptbl = get_parametric_tables();
    for (int i = 0; i < 65536; ++i) {
      h.neg_log[i] = -std::log2((i + 1) / 65536.0);
      h.neg_log_q6[i] = static_cast<uint16_t>(std::min(65535.0, std::rint(h.neg_log[i] * 64.0)));
    }
    for (int rm = 0; rm < 32; ++rm) {
      const int r_mag = rm < 31 ? rm : 42;
      for (int s = 0; s < NUM_SIGMAS; ++s) {
        const int sh = ptbl.shifts[s];
        const int64_t bias = sh > 0 ? (1LL << (sh - 1)) : 0;
        if (r_mag == 0) {
          const int64_t q = sh > 0 ? (bias >> sh) : 0;
          const int sym = static_cast<int>(q + 32);
          h.sigma_rm[s][rm] = static_cast<float>(sh + h.neg_log[ptbl.freqs[1][s][8][sym] - 1]);
        } else {
          double c_acc = 0.0;
          for (int sign : {-1, 1}) {
            const int64_t r = sign * r_mag;
            const int64_t q = sh > 0 ? ((r + bias) >> sh) : r;
            const int64_t mq = iabs64(q);
            const int sym = mq > 32 ? 65 : static_cast<int>(q + 32);
            double c = sh + h.neg_log[ptbl.freqs[1][s][8][sym] - 1];
            if (sym == 65) c += 2.0 * std::log2(static_cast<double>(mq - 31)) + 3.0;
            c_acc += 0.5 * c;
          }
          h.sigma_rm[s][rm] = static_cast<float>(c_acc);
        }
      }
    }
    return h;
  }();
  return t;
}

int best_sigma_strided(const ParametricTables& tbl, int nu, const std::vector<int>& idx,
                       const std::vector<int32_t>& resid, const std::vector<int32_t>& frac,
                       const double* neg_log, double& out_cost) {
  if (idx.empty()) {
    out_cost = 0.0;
    return 4;
  }
  const int step = std::max(1, static_cast<int>(idx.size()) / 1024);
  double sum_q8 = 0.0;
  int cnt = 0;
  for (size_t k = 0; k < idx.size(); k += static_cast<size_t>(step), ++cnt) {
    const int i = idx[k];
    const int64_t inner = static_cast<int64_t>(resid[static_cast<size_t>(i)]) * 16 -
                          (static_cast<int64_t>(frac[static_cast<size_t>(i)]) - 8);
    sum_q8 += static_cast<double>(iabs64(inner) << 4);
  }
  const double est = (sum_q8 / static_cast<double>(cnt)) * 0.92 + 20.0;
  int guess = 0;
  double bd = std::fabs(static_cast<double>(SIGMA_Q8[0]) - est);
  for (int s = 1; s < NUM_SIGMAS; ++s) {
    const double d = std::fabs(static_cast<double>(SIGMA_Q8[s]) - est);
    if (d < bd) {
      bd = d;
      guess = s;
    }
  }
  int best_s = guess;
  double best_c = std::numeric_limits<double>::infinity();
  for (int cand = std::max(0, guess - 2); cand < std::min(NUM_SIGMAS, guess + 3); ++cand) {
    const int sh = tbl.shifts[cand];
    const int64_t bias = sh > 0 ? (1LL << (sh - 1)) : 0;
    double cost = static_cast<double>(cnt * sh);
    for (size_t k = 0; k < idx.size(); k += static_cast<size_t>(step)) {
      const int i = idx[k];
      const int64_t r = resid[static_cast<size_t>(i)];
      const int64_t q = sh > 0 ? ((r + bias) >> sh) : r;
      const int64_t mq = iabs64(q);
      const int sym = mq > 32 ? 65 : static_cast<int>(q + 32);
      cost += neg_log[tbl.freqs[nu][cand][frac[static_cast<size_t>(i)]][sym] - 1];
      if (sym == 65) cost += 2.0 * std::log2(static_cast<double>(mq - 31)) + 3.0;
    }
    if (cost < best_c) {
      best_c = cost;
      best_s = cand;
    }
  }
  out_cost = best_c * static_cast<double>(idx.size()) / static_cast<double>(cnt);
  return best_s;
}

// Fast single-pass static vs adaptive Core66 bit-cost pre-decision, executing rANS only once!
std::pair<bool, std::vector<uint8_t>> encode_best_rans_once(const std::vector<int32_t>& resid, int best_nu,
                                                            const std::vector<int32_t>& leaf_ids,
                                                            const std::vector<int32_t>& leaf_sigmas,
                                                            const std::vector<int32_t>& frac, int effort) {
  const size_t n = resid.size();
  std::vector<int32_t> sig_idx(n);
  for (size_t i = 0; i < n; ++i) sig_idx[i] = leaf_sigmas[static_cast<size_t>(leaf_ids[i])];
  if (n < 512) {
    return {false, encode_parametric_rans(resid, best_nu, sig_idx, frac)};
  }
  if (effort >= 3) {
    const auto sp = encode_parametric_rans(resid, best_nu, sig_idx, frac);
    const auto ap = encode_adaptive_rans_stream(resid, best_nu, leaf_ids, leaf_sigmas, frac);
    if (ap.size() < sp.size()) return {true, ap};
    return {false, sp};
  }

  const ParametricTables& tables = get_parametric_tables();
  const auto& htbl = get_hist_cost_tables();
  AdaptiveCore66 model;
  model.init(best_nu, leaf_sigmas);
  std::vector<RansOp> ops;
  ops.reserve(n * 2);
  BitWriter raw_bits;
  uint64_t stat_q6 = 0;
  uint64_t adap_q6 = 0;
  for (size_t i = 0; i < n; ++i) {
    const int l = leaf_ids[i];
    const int sig = sig_idx[i];
    const int b = tables.shifts[sig];
    const int f = frac[i];
    const int64_t r = resid[i];
    const int64_t bias = b > 0 ? (1LL << (b - 1)) : 0;
    const int64_t q = b > 0 ? ((r + bias) >> b) : r;
    const int64_t mq = q < 0 ? -q : q;
    const int sym = mq > 32 ? 65 : static_cast<int>(q + 32);
    auto& slot = model.at(l, f);
    const int32_t st = slot.cum[sym];
    const int32_t frq = slot.cum[sym + 1] - st;
    stat_q6 += htbl.neg_log_q6[tables.freqs[best_nu][sig][f][sym] - 1];
    adap_q6 += htbl.neg_log_q6[frq - 1];
    ops.push_back(RansOp{static_cast<uint16_t>(st), static_cast<uint16_t>(frq - 1)});
    slot.observe(sym);

    if (sym == 65) {
      raw_bits.write_bits(q < 0 ? 1u : 0u, 1);
      const uint64_t code = static_cast<uint64_t>(mq - 33);
      if (code > (1ull << 24)) throw CodecError("frontier escape overflow");
      raw_bits.write_exp_golomb(static_cast<uint32_t>(code));
    }
    if (b > 0) {
      const int64_t rem = r - ((q << b) - bias);
      if (mq <= TILT_QMAX) {
        const int depth = std::min(b, TILT_MAX_DEPTH);
        const int rem_low = b - depth;
        const int64_t top = rem >> rem_low;
        const int32_t* p0_row = tables.tilt_p0[best_nu][sig][static_cast<int>(q + TILT_QMAX)];
        int branch = 0;
        for (int lvl = 0; lvl < depth; ++lvl) {
          const int bit = static_cast<int>((top >> (depth - 1 - lvl)) & 1);
          const int node = ((1 << lvl) - 1) + branch;
          const int p0 = p0_row[node];
          if (bit == 0) ops.push_back(RansOp{0, static_cast<uint16_t>(p0 - 1)});
          else ops.push_back(RansOp{static_cast<uint16_t>(p0), static_cast<uint16_t>(RANS_M - p0 - 1)});
          branch = (branch << 1) | bit;
        }
        if (rem_low > 0) raw_bits.write_bits(static_cast<uint32_t>(rem & ((1 << rem_low) - 1)), rem_low);
      } else {
        raw_bits.write_bits(static_cast<uint32_t>(rem), b);
      }
    }
  }
  if (adap_q6 + 128 < stat_q6) {
    return {true, pack_rans_ops(ops, raw_bits)};
  }
  return {false, encode_parametric_rans(resid, best_nu, sig_idx, frac)};
}

// Ultra-fast Universal-Tree-only MA stream encoder (used at effort=1: zero tree search, zero props buffer!).
std::vector<uint8_t> encode_ma_stream_universal_fast(const std::vector<int32_t>& resid,
                                                     const std::vector<int32_t>& frac,
                                                     const std::vector<int32_t>& leaf_ids, int tree_mode) {
  const ParametricTables& tbl = get_parametric_tables();
  const auto& htbl = get_hist_cost_tables();
  const int N = static_cast<int>(resid.size());
  const int n_leaves = priors::kUnivTreeLeaves[std::clamp(tree_mode, 0, 4)];
  std::vector<std::vector<int>> buckets(static_cast<size_t>(n_leaves));
  const int step = std::max(1, N / 16384);
  for (int i = 0; i < N; i += step) {
    buckets[static_cast<size_t>(leaf_ids[static_cast<size_t>(i)])].push_back(i);
  }

  int best_nu = 1;
  double best_nu_c = std::numeric_limits<double>::infinity();
  for (int nu : {0, 1, 2}) {
    double tc = 0.0;
    for (int l = 0; l < n_leaves; ++l) {
      double lc = 0.0;
      best_sigma_strided(tbl, nu, buckets[static_cast<size_t>(l)], resid, frac, htbl.neg_log, lc);
      tc += lc;
    }
    if (tc < best_nu_c) {
      best_nu_c = tc;
      best_nu = nu;
    }
  }

  std::vector<int32_t> leaf_sigmas(static_cast<size_t>(n_leaves), 4);
  int prev_s = 4;
  for (int l = 0; l < n_leaves; ++l) {
    if (buckets[static_cast<size_t>(l)].empty()) {
      leaf_sigmas[static_cast<size_t>(l)] = prev_s;
    } else {
      double lc = 0.0;
      prev_s = best_sigma_strided(tbl, best_nu, buckets[static_cast<size_t>(l)], resid, frac, htbl.neg_log, lc);
      leaf_sigmas[static_cast<size_t>(l)] = prev_s;
    }
  }

  const auto [use_adapt, pay] = encode_best_rans_once(resid, best_nu, leaf_ids, leaf_sigmas, frac, 1);
  BitWriter tw;
  tw.write_bits(static_cast<uint32_t>(best_nu & 3), 2);
  tw.write_bits(use_adapt ? 1u : 0u, 1);
  tw.write_bits(0u, 1);  // is_custom = 0 (Universal Tree!)
  tw.write_bits(static_cast<uint32_t>(leaf_sigmas[0] & 31), 5);
  for (int l = 1; l < n_leaves; ++l) {
    tw.write_signed_golomb(leaf_sigmas[static_cast<size_t>(l)] - leaf_sigmas[static_cast<size_t>(l - 1)]);
  }
  const auto hdr = tw.finish();

  std::vector<uint8_t> out;
  out.reserve(6 + hdr.size() + pay.size());
  const uint16_t hn = static_cast<uint16_t>(hdr.size());
  out.push_back(static_cast<uint8_t>(hn));
  out.push_back(static_cast<uint8_t>(hn >> 8));
  out.insert(out.end(), hdr.begin(), hdr.end());
  const uint32_t pn = static_cast<uint32_t>(pay.size());
  for (int k = 0; k < 4; ++k) out.push_back(static_cast<uint8_t>(pn >> (8 * k)));
  out.insert(out.end(), pay.begin(), pay.end());
  return out;
}

// O(1) L1-Histogram MA Tree Builder vs Universal MA Tree (MDL-gated against overfitting!).
std::vector<uint8_t> encode_ma_stream(const std::vector<int32_t>& resid, const std::vector<int32_t>& frac,
                                      const std::vector<int16_t>& props, int n_props, int tree_mode,
                                      int max_leaves, int effort) {
  const ParametricTables& tbl = get_parametric_tables();
  const auto& htbl = get_hist_cost_tables();
  const int N = static_cast<int>(resid.size());
  if (N == 0) {
    MATree tree = MATree::from_universal(tree_mode);
    BitWriter tw;
    tw.write_bits(1, 2);
    tw.write_bits(0, 1);
    tw.write_bits(0, 1);
    tw.write_bits(4, 5);
    for (size_t l = 1; l < tree.leaf_sigmas.size(); ++l) tw.write_signed_golomb(0);
    const auto hdr = tw.finish();
    const auto pay = encode_parametric_rans({}, 1, {}, {});
    std::vector<uint8_t> out;
    const uint16_t hn = static_cast<uint16_t>(hdr.size());
    out.push_back(static_cast<uint8_t>(hn));
    out.push_back(static_cast<uint8_t>(hn >> 8));
    out.insert(out.end(), hdr.begin(), hdr.end());
    const uint32_t pn = static_cast<uint32_t>(pay.size());
    for (int k = 0; k < 4; ++k) out.push_back(static_cast<uint8_t>(pn >> (8 * k)));
    out.insert(out.end(), pay.begin(), pay.end());
    return out;
  }

  // Precompute exact 32-bin integer residual magnitude per sample once (L2-resident).
  std::vector<uint8_t> rm_vec(static_cast<size_t>(N));
  for (int i = 0; i < N; ++i) {
    rm_vec[static_cast<size_t>(i)] = static_cast<uint8_t>(std::min<int32_t>(31, iabs32(resid[static_cast<size_t>(i)])));
  }

  constexpr int kPropMin = -20;
  constexpr int kPropBins = 48;

  auto hist_best_cost = [&](const uint32_t cnt[32]) -> float {
    float best = std::numeric_limits<float>::infinity();
    for (int s = 0; s < NUM_SIGMAS; ++s) {
      float c = 0.0f;
      for (int rm = 0; rm < 32; ++rm) c += static_cast<float>(cnt[rm]) * htbl.sigma_rm[s][rm];
      if (c < best) best = c;
    }
    return best;
  };

  struct LeafWork {
    int node_idx;
    std::vector<int> indices;
    double best_gain;
    int best_p;
    int best_t;
    int depth;
  };

  auto eval_leaf = [&](LeafWork& lw) {
    lw.best_gain = -1.0;
    const int sz = static_cast<int>(lw.indices.size());
    if (lw.depth >= 14 || sz < 128) return;
    const int step = std::max(1, sz / 4096);
    uint32_t hist[14][kPropBins][32]{};
    uint32_t tot[32]{};
    int sub_n = 0;
    for (int k = 0; k < sz; k += step, ++sub_n) {
      const int i = lw.indices[static_cast<size_t>(k)];
      const uint8_t rm = rm_vec[static_cast<size_t>(i)];
      tot[rm]++;
      const int16_t* pr = &props[static_cast<size_t>(i) * static_cast<size_t>(n_props)];
      for (int p = 0; p < n_props; ++p) {
        const int b = std::clamp<int>(int(pr[p]) - kPropMin, 0, kPropBins - 1);
        hist[p][b][rm]++;
      }
    }
    const float base_c = hist_best_cost(tot);
    const int min_side = std::max(24, 64 / step);
    for (int p = 0; p < n_props; ++p) {
      uint32_t lcnt[32]{};
      int nl = 0;
      for (int b = 0; b < kPropBins - 1; ++b) {
        int b_tot = 0;
        for (int rm = 0; rm < 32; ++rm) {
          const uint32_t v = hist[p][b][rm];
          lcnt[rm] += v;
          b_tot += static_cast<int>(v);
        }
        if (b_tot == 0) continue;
        nl += b_tot;
        if (nl < min_side || (sub_n - nl) < min_side) continue;
        uint32_t rcnt[32];
        for (int rm = 0; rm < 32; ++rm) rcnt[rm] = tot[rm] - lcnt[rm];
        const double gain = static_cast<double>(base_c - hist_best_cost(lcnt) - hist_best_cost(rcnt)) * step;
        if (gain > lw.best_gain) {
          lw.best_gain = gain;
          lw.best_p = p;
          lw.best_t = b + kPropMin;
        }
      }
    }
  };

  MATree custom_tree;
  custom_tree.nodes.push_back(MATree::Node{0, -1, 4, -1, -1, -1});
  std::vector<int> all_idx(static_cast<size_t>(N));
  std::iota(all_idx.begin(), all_idx.end(), 0);
  std::vector<LeafWork> leaves;
  leaves.push_back(LeafWork{0, std::move(all_idx), -1.0, 0, 0, 0});
  eval_leaf(leaves[0]);

  while (static_cast<int>(leaves.size()) < max_leaves) {
    int best_k = -1;
    double best_g = 18.0;
    for (size_t k = 0; k < leaves.size(); ++k) {
      if (leaves[k].best_gain > best_g) {
        best_g = leaves[k].best_gain;
        best_k = static_cast<int>(k);
      }
    }
    if (best_k < 0) break;
    LeafWork cur = std::move(leaves[static_cast<size_t>(best_k)]);
    leaves.erase(leaves.begin() + best_k);

    std::vector<int> li;
    std::vector<int> ri;
    li.reserve(cur.indices.size());
    ri.reserve(cur.indices.size());
    for (int i : cur.indices) {
      if (props[static_cast<size_t>(i) * static_cast<size_t>(n_props) + static_cast<size_t>(cur.best_p)] <= cur.best_t) {
        li.push_back(i);
      } else {
        ri.push_back(i);
      }
    }
    const int lid = static_cast<int>(custom_tree.nodes.size());
    custom_tree.nodes.push_back(MATree::Node{0, -1, 4, -1, -1, -1});
    const int rid = static_cast<int>(custom_tree.nodes.size());
    custom_tree.nodes.push_back(MATree::Node{0, -1, 4, -1, -1, -1});
    custom_tree.nodes[static_cast<size_t>(cur.node_idx)].prop = static_cast<int8_t>(cur.best_p);
    custom_tree.nodes[static_cast<size_t>(cur.node_idx)].thr = static_cast<int16_t>(cur.best_t);
    custom_tree.nodes[static_cast<size_t>(cur.node_idx)].left = static_cast<int16_t>(lid);
    custom_tree.nodes[static_cast<size_t>(cur.node_idx)].right = static_cast<int16_t>(rid);

    LeafWork lw_l{lid, std::move(li), -1.0, 0, 0, cur.depth + 1};
    LeafWork lw_r{rid, std::move(ri), -1.0, 0, 0, cur.depth + 1};
    eval_leaf(lw_l);
    eval_leaf(lw_r);
    leaves.push_back(std::move(lw_l));
    leaves.push_back(std::move(lw_r));
  }

  int best_nu = 1;
  double best_custom_data = std::numeric_limits<double>::infinity();
  for (int nu = 0; nu < NUM_NUS; ++nu) {
    double tc = 0.0;
    for (auto& lw : leaves) {
      double lc = 0.0;
      best_sigma_strided(tbl, nu, lw.indices, resid, frac, htbl.neg_log, lc);
      tc += lc;
    }
    if (tc < best_custom_data) {
      best_custom_data = tc;
      best_nu = nu;
    }
  }
  for (auto& lw : leaves) {
    double lc = 0.0;
    custom_tree.nodes[static_cast<size_t>(lw.node_idx)].sigma =
        static_cast<uint8_t>(best_sigma_strided(tbl, best_nu, lw.indices, resid, frac, htbl.neg_log, lc));
  }
  custom_tree.leaf_sigmas.clear();
  auto number_dfs = [&](auto&& self, int id) -> void {
    if (custom_tree.nodes[static_cast<size_t>(id)].left < 0) {
      custom_tree.nodes[static_cast<size_t>(id)].leaf_id = static_cast<int16_t>(custom_tree.leaf_sigmas.size());
      custom_tree.leaf_sigmas.push_back(custom_tree.nodes[static_cast<size_t>(id)].sigma);
      return;
    }
    self(self, custom_tree.nodes[static_cast<size_t>(id)].left);
    self(self, custom_tree.nodes[static_cast<size_t>(id)].right);
  };
  number_dfs(number_dfs, 0);

  // Also evaluate the offline-trained Universal Tree so small/noisy tiles never overfit!
  MATree univ_tree = MATree::from_universal(tree_mode);
  const int n_univ_l = static_cast<int>(univ_tree.leaf_sigmas.size());
  std::vector<std::vector<int>> univ_buckets(static_cast<size_t>(n_univ_l));
  std::vector<int32_t> univ_leaf_ids(static_cast<size_t>(N));
  for (int i = 0; i < N; ++i) {
    const int lid = univ_tree.walk(&props[static_cast<size_t>(i) * static_cast<size_t>(n_props)]);
    univ_leaf_ids[static_cast<size_t>(i)] = lid;
    univ_buckets[static_cast<size_t>(lid)].push_back(i);
  }
  double univ_data = 0.0;
  int prev_s = 4;
  int univ_hdr_bits = 4 + 5;
  for (int l = 0; l < n_univ_l; ++l) {
    double lc = 0.0;
    if (!univ_buckets[static_cast<size_t>(l)].empty()) {
      prev_s = best_sigma_strided(tbl, best_nu, univ_buckets[static_cast<size_t>(l)], resid, frac, htbl.neg_log, lc);
    }
    univ_tree.leaf_sigmas[static_cast<size_t>(l)] = prev_s;
    univ_data += lc;
    if (l > 0) univ_hdr_bits += golomb_signed_bits(prev_s - univ_tree.leaf_sigmas[static_cast<size_t>(l - 1)]);
  }

  BitWriter c_hdr_probe;
  custom_tree.write_custom(c_hdr_probe);
  const int custom_hdr_bits = 4 + static_cast<int>(c_hdr_probe.finish().size()) * 8;
  const bool use_custom = (best_custom_data + custom_hdr_bits + 16.0) < (univ_data + univ_hdr_bits);

  std::vector<int32_t> final_leaf_ids;
  const std::vector<int32_t>* final_sigmas = nullptr;
  if (use_custom) {
    final_leaf_ids.resize(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
      final_leaf_ids[static_cast<size_t>(i)] =
          custom_tree.walk(&props[static_cast<size_t>(i) * static_cast<size_t>(n_props)]);
    }
    final_sigmas = &custom_tree.leaf_sigmas;
  } else {
    final_leaf_ids = std::move(univ_leaf_ids);
    final_sigmas = &univ_tree.leaf_sigmas;
  }

  const auto [use_adapt, pay] = encode_best_rans_once(resid, best_nu, final_leaf_ids, *final_sigmas, frac, effort);

  BitWriter tw;
  tw.write_bits(static_cast<uint32_t>(best_nu & 3), 2);
  tw.write_bits(use_adapt ? 1u : 0u, 1);
  if (use_custom) {
    tw.write_bits(1u, 1);
    custom_tree.write_custom(tw);
  } else {
    tw.write_bits(0u, 1);
    tw.write_bits(static_cast<uint32_t>((*final_sigmas)[0] & 31), 5);
    for (size_t l = 1; l < final_sigmas->size(); ++l) {
      tw.write_signed_golomb((*final_sigmas)[l] - (*final_sigmas)[l - 1]);
    }
  }
  const auto hdr = tw.finish();
  if (hdr.size() > 65535) throw CodecError("MA tree header exceeds uint16");

  std::vector<uint8_t> out;
  out.reserve(6 + hdr.size() + pay.size());
  const uint16_t hn = static_cast<uint16_t>(hdr.size());
  out.push_back(static_cast<uint8_t>(hn));
  out.push_back(static_cast<uint8_t>(hn >> 8));
  out.insert(out.end(), hdr.begin(), hdr.end());
  const uint32_t pn = static_cast<uint32_t>(pay.size());
  for (int k = 0; k < 4; ++k) out.push_back(static_cast<uint8_t>(pn >> (8 * k)));
  out.insert(out.end(), pay.begin(), pay.end());
  return out;
}

std::vector<uint8_t> encode_channel_raster_exact(const Plane& P, int c_idx, int rct_mode, int maxv, int effort,
                                                 const Plane* C0, const Plane* C0_err, const Plane* C1,
                                                 const Plane* C1_err, Plane& out_err_q4) {
  const int H = P.height;
  const int W = P.width;
  const int N = H * W;
  out_err_q4 = Plane(H, W, 0);
  const bool hc0 = C0 != nullptr;
  const bool hc1 = C1 != nullptr;
  const int role = !hc0 ? 0 : (!hc1 ? 1 : 2);
  const int tree_mode = !hc0 ? 0 : 1;
  const int D = !hc0 ? 21 : (!hc1 ? 30 : 36);
  const int n_sec = 16;

  // Fast single-pass causal mode at effort == 1 (1 predictor pass + 1 rANS pass, zero matrix/tree search!)
  if (effort <= 1 || N < 256) {
    const bool use_patch = true;
    const int nlms_gain = !hc0 ? priors::kNlmsGainLuma : priors::kNlmsGainChroma;
    BitWriter hw;
    hw.write_bits(use_patch ? 1u : 0u, 1);
    hw.write_bits(static_cast<uint32_t>(nlms_gain & 31), 5);
    hw.write_bits(0u, 1);  // Prior-Only (0 custom ridge bytes!)
    const auto w_bytes = hw.finish();

    std::array<std::array<int32_t, 36>, 16> wlms{};
    std::array<std::array<int16_t, 36>, 16> ws_q8{};
    for (int s = 0; s < n_sec; ++s) {
      for (int a = 0; a < D; ++a) {
        const int16_t pw = get_prior_weight(role, s, a);
        ws_q8[s][a] = pw;
        wlms[s][a] = static_cast<int32_t>(pw) << 8;
      }
    }
    MATree univ_tree = MATree::from_universal(tree_mode);
    RasterState st(H, W, c_idx, rct_mode, maxv, hc0, hc1, use_patch);
    std::vector<int32_t> resid(static_cast<size_t>(N));
    std::vector<int32_t> frac(static_cast<size_t>(N));
    std::vector<int32_t> leaf_ids(static_cast<size_t>(N));
    alignas(32) int32_t f[40]{};
    int16_t prp[14];
    int64_t pr_out[RasterState::KEXP];
    int sec = 0;

    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const int id = y * W + x;
        const int64_t pred_q4 = st.predict_features(P, C0, C0_err, C1, C1_err, y, x, f, sec, prp, pr_out);
        const auto& w8 = ws_q8[sec];
        int64_t acc = 128;
        if (st.qsh == 0) {
          int32_t acc32 = 128;
          for (int a = 0; a < 36; ++a) acc32 += f[a] * static_cast<int32_t>(w8[a]);
          acc = acc32;
        } else {
          for (int a = 0; a < D; ++a) acc += static_cast<int64_t>(f[a]) * w8[a];
        }
        const int64_t raw_mu = pred_q4 + (acc >> 8);
        const auto [blo, bhi] = pixel_bounds_q4(rct_mode, c_idx, maxv, C0, C1, y, x);
        const int64_t mu_q4 = std::clamp(raw_mu, blo, bhi);
        const int64_t mu_i = (mu_q4 + 8) >> 4;
        const int fr = std::clamp(static_cast<int>(mu_q4 - (mu_i << 4) + 8), 0, 15);
        const int32_t truth = P.samples[id];
        resid[static_cast<size_t>(id)] = truth - static_cast<int32_t>(mu_i);
        frac[static_cast<size_t>(id)] = fr;
        leaf_ids[static_cast<size_t>(id)] = univ_tree.walk(prp);
        out_err_q4.samples[id] = static_cast<int32_t>((static_cast<int64_t>(truth) << 4) - mu_q4);
        if (nlms_gain > 0) {
          nlms_update(wlms[sec].data(), ws_q8[sec].data(), D, f, (static_cast<int64_t>(truth) << 4) - raw_mu,
                      nlms_gain, st.qsh);
        }
        st.commit(y, x, truth, pred_q4, mu_q4, pr_out);
      }
    }
    const auto ma_bytes = encode_ma_stream_universal_fast(resid, frac, leaf_ids, tree_mode);
    std::vector<uint8_t> out;
    out.reserve(2 + w_bytes.size() + ma_bytes.size());
    const uint16_t wn = static_cast<uint16_t>(w_bytes.size());
    out.push_back(static_cast<uint8_t>(wn));
    out.push_back(static_cast<uint8_t>(wn >> 8));
    out.insert(out.end(), w_bytes.begin(), w_bytes.end());
    out.insert(out.end(), ma_bytes.begin(), ma_bytes.end());
    return out;
  }

  // Effort >= 2: Pass 1 accumulates SectorGrams around preloaded W_prior (0 MB all_f!).
  const bool use_patch = true;
  std::array<std::array<int16_t, 36>, 16> w_prior{};
  for (int s = 0; s < n_sec; ++s) {
    for (int a = 0; a < D; ++a) w_prior[s][a] = get_prior_weight(role, s, a);
  }

  std::vector<SectorGram> sgrams(static_cast<size_t>(n_sec));
  {
    RasterState st(H, W, c_idx, rct_mode, maxv, hc0, hc1, use_patch);
    alignas(32) int32_t f[40]{};
    int64_t pr_out[RasterState::KEXP];
    int sec = 0;
    if (effort == 2 && N >= 65536 && H >= 48) {
      constexpr int kStripH = 10;
      constexpr int kStripStep = 24;
      for (int y0 = 0; y0 < H; y0 += kStripStep) {
        st.reset_row_state();
        const int y1 = std::min(H, y0 + kStripH);
        for (int y = y0; y < y1; ++y) {
          const bool collect = (y >= y0 + 2);
          for (int x = 0; x < W; ++x) {
            const int id = y * W + x;
            const int64_t pred_q4 = st.predict_features(P, C0, C0_err, C1, C1_err, y, x, f, sec, nullptr, pr_out);
            int64_t prior_acc = 128;
            const auto& wp = w_prior[sec];
            for (int a = 0; a < D; ++a) prior_acc += static_cast<int64_t>(f[a]) * wp[a];
            const int64_t raw_mu = pred_q4 + (prior_acc >> 8);
            const auto [blo, bhi] = pixel_bounds_q4(rct_mode, c_idx, maxv, C0, C1, y, x);
            const int64_t mu_q4 = std::clamp(raw_mu, blo, bhi);
            const int32_t truth = P.samples[id];
            if (collect) {
              const double y_res = static_cast<double>((static_cast<int64_t>(truth) << 4) - raw_mu);
              sgrams[static_cast<size_t>(sec)].add(D, f, y_res);
            }
            st.commit(y, x, truth, pred_q4, mu_q4, pr_out);
          }
        }
      }
    } else {
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          const int id = y * W + x;
          const int64_t pred_q4 = st.predict_features(P, C0, C0_err, C1, C1_err, y, x, f, sec, nullptr, pr_out);
          int64_t prior_acc = 128;
          const auto& wp = w_prior[sec];
          for (int a = 0; a < D; ++a) prior_acc += static_cast<int64_t>(f[a]) * wp[a];
          const int64_t raw_mu = pred_q4 + (prior_acc >> 8);
          const auto [blo, bhi] = pixel_bounds_q4(rct_mode, c_idx, maxv, C0, C1, y, x);
          const int64_t mu_q4 = std::clamp(raw_mu, blo, bhi);
          const int32_t truth = P.samples[id];
          const double y_res = static_cast<double>((static_cast<int64_t>(truth) << 4) - raw_mu);
          sgrams[static_cast<size_t>(sec)].add(D, f, y_res);
          st.commit(y, x, truth, pred_q4, mu_q4, pr_out);
        }
      }
    }
  }

  BitWriter ridge_probe;
  std::vector<std::array<int16_t, 36>> w_final;
  const bool wrote_custom = solve_and_write_map_ridge(ridge_probe, role, n_sec, D, N, sgrams, w_final);
  const int base_nlms = !hc0 ? priors::kNlmsGainLuma : priors::kNlmsGainChroma;
  const int final_nlms = !wrote_custom ? base_nlms : (effort == 2 ? (!hc0 ? 4 : 8) : 0);

  BitWriter hw;
  hw.write_bits(use_patch ? 1u : 0u, 1);
  hw.write_bits(static_cast<uint32_t>(final_nlms & 31), 5);
  solve_and_write_map_ridge(hw, role, n_sec, D, N, sgrams, w_final);
  const auto w_bytes = hw.finish();
  if (w_bytes.size() > 65535) throw CodecError("frontier ridge header exceeds uint16");

  std::array<std::array<int32_t, 36>, 16> wlms{};
  std::array<std::array<int16_t, 36>, 16> ws_q8{};
  for (int s = 0; s < n_sec; ++s) {
    for (int a = 0; a < D; ++a) {
      const int16_t wf = w_final[static_cast<size_t>(s)][a];
      ws_q8[s][a] = wf;
      wlms[s][a] = static_cast<int32_t>(wf) << 8;
    }
  }

  std::vector<int32_t> resid(static_cast<size_t>(N));
  std::vector<int32_t> frac(static_cast<size_t>(N));
  std::vector<int16_t> props(static_cast<size_t>(N) * 14);
  {
    RasterState st(H, W, c_idx, rct_mode, maxv, hc0, hc1, use_patch);
    alignas(32) int32_t f[40]{};
    int64_t pr_out[RasterState::KEXP];
    int sec = 0;
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const int id = y * W + x;
        int16_t* prp = &props[static_cast<size_t>(id) * 14];
        const int64_t pred_q4 = st.predict_features(P, C0, C0_err, C1, C1_err, y, x, f, sec, prp, pr_out);
        const auto& w8 = ws_q8[sec];
        int64_t acc = 128;
        if (st.qsh == 0) {
          int32_t acc32 = 128;
          for (int a = 0; a < 36; ++a) acc32 += f[a] * static_cast<int32_t>(w8[a]);
          acc = acc32;
        } else {
          for (int a = 0; a < D; ++a) acc += static_cast<int64_t>(f[a]) * w8[a];
        }
        const int64_t raw_mu = pred_q4 + (acc >> 8);
        const auto [blo, bhi] = pixel_bounds_q4(rct_mode, c_idx, maxv, C0, C1, y, x);
        const int64_t mu_q4 = std::clamp(raw_mu, blo, bhi);
        const int64_t mu_i = (mu_q4 + 8) >> 4;
        const int fr = std::clamp(static_cast<int>(mu_q4 - (mu_i << 4) + 8), 0, 15);
        const int32_t truth = P.samples[id];
        resid[static_cast<size_t>(id)] = truth - static_cast<int32_t>(mu_i);
        frac[static_cast<size_t>(id)] = fr;
        out_err_q4.samples[id] = static_cast<int32_t>((static_cast<int64_t>(truth) << 4) - mu_q4);
        if (final_nlms > 0) {
          nlms_update(wlms[sec].data(), ws_q8[sec].data(), D, f, (static_cast<int64_t>(truth) << 4) - raw_mu,
                      final_nlms, st.qsh);
        }
        st.commit(y, x, truth, pred_q4, mu_q4, pr_out);
      }
    }
  }

  const int max_leaves = effort == 2 ? (N >= 100000 ? 64 : 32) : (N >= 100000 ? 80 : 40);
  const auto ma_bytes = encode_ma_stream(resid, frac, props, 14, tree_mode, max_leaves, effort);

  std::vector<uint8_t> out;
  out.reserve(2 + w_bytes.size() + ma_bytes.size());
  const uint16_t wn = static_cast<uint16_t>(w_bytes.size());
  out.push_back(static_cast<uint8_t>(wn));
  out.push_back(static_cast<uint8_t>(wn >> 8));
  out.insert(out.end(), w_bytes.begin(), w_bytes.end());
  out.insert(out.end(), ma_bytes.begin(), ma_bytes.end());
  return out;
}

__attribute__((noinline)) Plane decode_channel_raster_exact(const uint8_t* data, size_t size, size_t& pos, int H,
                                                            int W, int c_idx, int rct_mode, int maxv, const Plane* C0,
                                                            const Plane* C0_err, const Plane* C1, const Plane* C1_err,
                                                            Plane& out_err_q4) {
  if (pos + 2 > size) throw CodecError("truncated frontier raster ridge header");
  const uint16_t wn = static_cast<uint16_t>(data[pos] | (uint16_t(data[pos + 1]) << 8));
  pos += 2;
  if (wn == 0 || pos + wn > size) throw CodecError("truncated frontier raster ridge weights");
  const bool hc0 = C0 != nullptr;
  const bool hc1 = C1 != nullptr;
  const int role = !hc0 ? 0 : (!hc1 ? 1 : 2);
  const int tree_mode = !hc0 ? 0 : 1;
  const int D = !hc0 ? 21 : (!hc1 ? 30 : 36);
  const int n_sec = 16;
  BitReader wr(data + pos, wn);
  pos += wn;
  const bool use_patch = wr.read_bits(1) == 1;
  const int nlms_gain = static_cast<int>(wr.read_bits(5));
  const auto w_final = read_sector_ridge(wr, role, n_sec, D);
  wr.check_zero_tail();

  OpenedMAStream mst = open_ma_stream(data, size, pos, 14, tree_mode);
  AdaptiveStreamDecoder dec(mst.payload, mst.nu, mst.use_adapt, mst.tree.leaf_sigmas);

  std::array<std::array<int32_t, 36>, 16> wlms{};
  std::array<std::array<int16_t, 36>, 16> ws_q8{};
  for (int s = 0; s < n_sec; ++s) {
    for (int a = 0; a < D; ++a) {
      const int16_t wf = w_final[static_cast<size_t>(s)][a];
      ws_q8[s][a] = wf;
      wlms[s][a] = static_cast<int32_t>(wf) << 8;
    }
  }

  Plane P(H, W, 0);
  out_err_q4 = Plane(H, W, 0);
  RasterState st(H, W, c_idx, rct_mode, maxv, hc0, hc1, use_patch);
  alignas(32) int32_t f[40]{};
  int16_t prp[14];
  int64_t pr_out[RasterState::KEXP];
  int sec = 0;

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const int id = y * W + x;
      const int64_t pred_q4 = st.predict_features(P, C0, C0_err, C1, C1_err, y, x, f, sec, prp, pr_out);
      const auto& w8 = ws_q8[sec];
      int64_t acc = 128;
      if (st.qsh == 0) {
        int32_t acc32 = 128;
        for (int a = 0; a < 36; ++a) acc32 += f[a] * static_cast<int32_t>(w8[a]);
        acc = acc32;
      } else {
        for (int a = 0; a < D; ++a) acc += static_cast<int64_t>(f[a]) * w8[a];
      }
      const int64_t raw_mu = pred_q4 + (acc >> 8);
      const auto [blo, bhi] = pixel_bounds_q4(rct_mode, c_idx, maxv, C0, C1, y, x);
      const int64_t mu_q4 = std::clamp(raw_mu, blo, bhi);
      const int64_t mu_i = (mu_q4 + 8) >> 4;
      const int fr = std::clamp(static_cast<int>(mu_q4 - (mu_i << 4) + 8), 0, 15);
      const int leaf = mst.tree.walk(prp);
      const int32_t r = dec.read(leaf, fr);
      const int64_t truth64 = mu_i + static_cast<int64_t>(r);
      if (truth64 < -262144 || truth64 > 262144) throw CodecError("decoded frontier sample out of range");
      const int32_t truth = static_cast<int32_t>(truth64);
      P.samples[id] = truth;
      out_err_q4.samples[id] = static_cast<int32_t>((truth64 << 4) - mu_q4);
      if (nlms_gain > 0) {
        nlms_update(wlms[sec].data(), ws_q8[sec].data(), D, f, (truth64 << 4) - raw_mu, nlms_gain, st.qsh);
      }
      st.commit(y, x, truth, pred_q4, mu_q4, pr_out);
    }
  }
  dec.check_finished();
  return P;
}

std::vector<uint8_t> encode_chroma_coset_exact(const Plane& P, int c_idx, int rct_mode, int maxv, int effort,
                                               const Plane& C0, const Plane& C0_err, const Plane* C1,
                                               const Plane* C1_err, Plane& out_err_q4) {
  const int H = P.height;
  const int W = P.width;
  if (H < 4 || W < 4) {
    return encode_channel_raster_exact(P, c_idx, rct_mode, maxv, effort, &C0, &C0_err, C1, C1_err, out_err_q4);
  }
  out_err_q4 = Plane(H, W, 0);
  Plane stage_err(H, W, 0);
  std::vector<uint8_t> out;

  const int H2 = (H + 1) / 2;
  const int W2 = (W + 1) / 2;
  {
    Plane sP(H2, W2);
    Plane sG(H2, W2);
    Plane sGe(H2, W2);
    Plane sU(H2, W2);
    Plane sUe(H2, W2);
    Plane sOutE;
    for (int iy = 0; iy < H2; ++iy) {
      for (int ix = 0; ix < W2; ++ix) {
        sP.at(iy, ix) = P.at(iy * 2, ix * 2);
        sG.at(iy, ix) = C0.at(iy * 2, ix * 2);
        sGe.at(iy, ix) = C0_err.at(iy * 2, ix * 2);
        if (C1 != nullptr) {
          sU.at(iy, ix) = C1->at(iy * 2, ix * 2);
          sUe.at(iy, ix) = C1_err->at(iy * 2, ix * 2);
        }
      }
    }
    const auto s0 = encode_channel_raster_exact(sP, c_idx, rct_mode, maxv, effort, &sG, &sGe,
                                                C1 != nullptr ? &sU : nullptr, C1 != nullptr ? &sUe : nullptr, sOutE);
    out.insert(out.end(), s0.begin(), s0.end());
    for (int iy = 0; iy < H2; ++iy) {
      for (int ix = 0; ix < W2; ++ix) {
        const int32_t eq4 = sOutE.at(iy, ix);
        out_err_q4.at(iy * 2, ix * 2) = eq4;
        stage_err.at(iy * 2, ix * 2) = static_cast<int32_t>(iabs64(eq4));
      }
    }
  }

  const int qsh = maxv > 255 ? 8 : 0;
  const int py_s[4] = {0, 1, 0, 1};
  const int px_s[4] = {0, 1, 1, 0};
  for (int stg = 1; stg <= 3; ++stg) {
    const int role = (C1 == nullptr ? 3 : 6) + (stg - 1);
    const int tree_mode = 1 + stg;
    const int Knb = stg == 1 ? 12 : 10;
    const int D = Knb + 10 + (C1 != nullptr ? 6 : 0);
    const int n_sec = 4;
    const int Ns = ((H - py_s[stg] + 1) / 2) * ((W - px_s[stg] + 1) / 2);

    if (effort <= 1 || Ns < 128) {
      // Fast 1/8-subsampled MAP Ridge (~0.4ms) + Universal MA Tree so effort=1 adapts to any 4:2:0 upsampler!
      std::array<std::array<int16_t, 36>, 4> w_prior{};
      for (int s = 0; s < n_sec; ++s) {
        for (int a = 0; a < D; ++a) w_prior[s][a] = get_prior_weight(role, s, a);
      }
      std::vector<SectorGram> sgrams(static_cast<size_t>(n_sec));
      int32_t f_tmp[32];
      int sec_tmp = 0;
      if (Ns >= 128) {
        int idx = 0;
        for (int y = py_s[stg]; y < H; y += 2) {
          for (int x = px_s[stg]; x < W; x += 2, ++idx) {
            if ((idx & 3) != 0) continue;
            const int64_t b_q4 =
                interp_features_exact(P, C0, C0_err, C1, C1_err, stage_err, maxv, stg, y, x, f_tmp, sec_tmp, nullptr);
            int64_t prior_acc = 128;
            const auto& wp = w_prior[sec_tmp];
            for (int a = 0; a < D; ++a) prior_acc += static_cast<int64_t>(f_tmp[a]) * wp[a];
            const int32_t truth = P.samples[y * W + x];
            const double y_res = static_cast<double>((static_cast<int64_t>(truth) << 4) - (b_q4 + (prior_acc >> 8)));
            sgrams[static_cast<size_t>(sec_tmp)].add(D, f_tmp, y_res);
          }
        }
      }
      BitWriter cos_probe;
      std::vector<std::array<int16_t, 36>> w_q8;
      const bool wrote_custom =
          Ns >= 128 ? solve_and_write_map_ridge(cos_probe, role, n_sec, D, Ns, sgrams, w_q8) : false;
      const int nlms_gain = wrote_custom ? 4 : 12;
      BitWriter hw;
      hw.write_bits(static_cast<uint32_t>(nlms_gain & 31), 5);
      if (Ns >= 128) {
        solve_and_write_map_ridge(hw, role, n_sec, D, Ns, sgrams, w_q8);
      } else {
        hw.write_bits(0u, 1);
      }
      const auto wb = hw.finish();
      const uint16_t wn = static_cast<uint16_t>(wb.size());
      out.push_back(static_cast<uint8_t>(wn));
      out.push_back(static_cast<uint8_t>(wn >> 8));
      out.insert(out.end(), wb.begin(), wb.end());

      std::array<std::array<int32_t, 36>, 4> wlms{};
      std::array<std::array<int16_t, 36>, 4> ws_q8{};
      for (int s = 0; s < n_sec; ++s) {
        for (int a = 0; a < D; ++a) {
          const int16_t pw = Ns >= 128 ? w_q8[static_cast<size_t>(s)][a] : get_prior_weight(role, s, a);
          ws_q8[s][a] = pw;
          wlms[s][a] = static_cast<int32_t>(pw) << 8;
        }
      }
      MATree univ_tree = MATree::from_universal(tree_mode);
      std::vector<int32_t> resid(static_cast<size_t>(Ns));
      std::vector<int32_t> frac(static_cast<size_t>(Ns));
      std::vector<int32_t> leaf_ids(static_cast<size_t>(Ns));
      int16_t prp[11];
      int i = 0;
      for (int y = py_s[stg]; y < H; y += 2) {
        for (int x = px_s[stg]; x < W; x += 2, ++i) {
          const int64_t b_q4 =
              interp_features_exact(P, C0, C0_err, C1, C1_err, stage_err, maxv, stg, y, x, f_tmp, sec_tmp, prp);
          int64_t acc = 128;
          const auto& w8 = ws_q8[sec_tmp];
          for (int a = 0; a < D; ++a) acc += static_cast<int64_t>(f_tmp[a]) * w8[a];
          const int64_t raw_mu = b_q4 + (acc >> 8);
          const auto [blo, bhi] = pixel_bounds_q4(rct_mode, c_idx, maxv, &C0, C1, y, x);
          const int64_t mu_q4 = std::clamp(raw_mu, blo, bhi);
          const int64_t mu_i = (mu_q4 + 8) >> 4;
          const int fr = std::clamp(static_cast<int>(mu_q4 - (mu_i << 4) + 8), 0, 15);
          const int32_t truth = P.samples[y * W + x];
          resid[static_cast<size_t>(i)] = truth - static_cast<int32_t>(mu_i);
          frac[static_cast<size_t>(i)] = fr;
          leaf_ids[static_cast<size_t>(i)] = univ_tree.walk(prp);
          const int32_t eq4 = static_cast<int32_t>((static_cast<int64_t>(truth) << 4) - mu_q4);
          out_err_q4.samples[y * W + x] = eq4;
          stage_err.samples[y * W + x] = static_cast<int32_t>(iabs64(eq4));
          nlms_update(wlms[sec_tmp].data(), ws_q8[sec_tmp].data(), D, f_tmp,
                      (static_cast<int64_t>(truth) << 4) - raw_mu, nlms_gain, qsh);
        }
      }
      const auto mb = encode_ma_stream_universal_fast(resid, frac, leaf_ids, tree_mode);
      out.insert(out.end(), mb.begin(), mb.end());
      continue;
    }

    // Effort >= 2: Subsampled MAP Ridge around preloaded W_prior[role] (zero all_f buffer!)
    std::array<std::array<int16_t, 36>, 4> w_prior{};
    for (int s = 0; s < n_sec; ++s) {
      for (int a = 0; a < D; ++a) w_prior[s][a] = get_prior_weight(role, s, a);
    }
    const int sub_mask = effort == 2 ? 1 : 0;
    std::vector<SectorGram> sgrams(static_cast<size_t>(n_sec));
    int32_t f_tmp[32];
    int sec_tmp = 0;
    int idx = 0;
    for (int y = py_s[stg]; y < H; y += 2) {
      for (int x = px_s[stg]; x < W; x += 2, ++idx) {
        if ((idx & sub_mask) != 0) continue;
        const int64_t b_q4 =
            interp_features_exact(P, C0, C0_err, C1, C1_err, stage_err, maxv, stg, y, x, f_tmp, sec_tmp, nullptr);
        int64_t prior_acc = 128;
        const auto& wp = w_prior[sec_tmp];
        for (int a = 0; a < D; ++a) prior_acc += static_cast<int64_t>(f_tmp[a]) * wp[a];
        const int32_t truth = P.samples[y * W + x];
        const double y_res = static_cast<double>((static_cast<int64_t>(truth) << 4) - (b_q4 + (prior_acc >> 8)));
        sgrams[static_cast<size_t>(sec_tmp)].add(D, f_tmp, y_res);
      }
    }

    BitWriter cos_probe;
    std::vector<std::array<int16_t, 36>> w_q8;
    const bool wrote_custom = solve_and_write_map_ridge(cos_probe, role, n_sec, D, Ns, sgrams, w_q8);
    const int cos_nlms = wrote_custom ? 0 : 12;

    BitWriter hw;
    hw.write_bits(static_cast<uint32_t>(cos_nlms & 31), 5);
    solve_and_write_map_ridge(hw, role, n_sec, D, Ns, sgrams, w_q8);
    const auto wb = hw.finish();
    if (wb.size() > 65535) throw CodecError("frontier coset ridge header exceeds uint16");
    const uint16_t wn = static_cast<uint16_t>(wb.size());
    out.push_back(static_cast<uint8_t>(wn));
    out.push_back(static_cast<uint8_t>(wn >> 8));
    out.insert(out.end(), wb.begin(), wb.end());

    std::array<std::array<int32_t, 36>, 4> wlms{};
    std::array<std::array<int16_t, 36>, 4> ws_q8{};
    for (int s = 0; s < n_sec; ++s) {
      for (int a = 0; a < D; ++a) {
        const int16_t wf = w_q8[static_cast<size_t>(s)][a];
        ws_q8[s][a] = wf;
        wlms[s][a] = static_cast<int32_t>(wf) << 8;
      }
    }

    std::vector<int32_t> resid(static_cast<size_t>(Ns));
    std::vector<int32_t> frac(static_cast<size_t>(Ns));
    std::vector<int16_t> props(static_cast<size_t>(Ns) * 11);
    int i = 0;
    for (int y = py_s[stg]; y < H; y += 2) {
      for (int x = px_s[stg]; x < W; x += 2, ++i) {
        const int64_t b_q4 = interp_features_exact(P, C0, C0_err, C1, C1_err, stage_err, maxv, stg, y, x, f_tmp,
                                                   sec_tmp, &props[static_cast<size_t>(i) * 11]);
        int64_t acc = 128;
        const auto& w8 = ws_q8[sec_tmp];
        for (int a = 0; a < D; ++a) acc += static_cast<int64_t>(f_tmp[a]) * w8[a];
        const int64_t raw_mu = b_q4 + (acc >> 8);
        const auto [blo, bhi] = pixel_bounds_q4(rct_mode, c_idx, maxv, &C0, C1, y, x);
        const int64_t mu_q4 = std::clamp(raw_mu, blo, bhi);
        const int64_t mu_i = (mu_q4 + 8) >> 4;
        const int fr = std::clamp(static_cast<int>(mu_q4 - (mu_i << 4) + 8), 0, 15);
        const int32_t truth = P.samples[y * W + x];
        resid[static_cast<size_t>(i)] = truth - static_cast<int32_t>(mu_i);
        frac[static_cast<size_t>(i)] = fr;
        const int32_t eq4 = static_cast<int32_t>((static_cast<int64_t>(truth) << 4) - mu_q4);
        out_err_q4.samples[y * W + x] = eq4;
        stage_err.samples[y * W + x] = static_cast<int32_t>(iabs64(eq4));
        if (cos_nlms > 0) {
          nlms_update(wlms[sec_tmp].data(), ws_q8[sec_tmp].data(), D, f_tmp,
                      (static_cast<int64_t>(truth) << 4) - raw_mu, cos_nlms, qsh);
        }
      }
    }
    const auto mb = encode_ma_stream(resid, frac, props, 11, tree_mode, 32, effort);
    out.insert(out.end(), mb.begin(), mb.end());
  }
  return out;
}

__attribute__((noinline)) Plane decode_chroma_coset_exact(const uint8_t* data, size_t size, size_t& pos, int H, int W,
                                                          int c_idx, int rct_mode, int maxv, const Plane& C0,
                                                          const Plane& C0_err, const Plane* C1, const Plane* C1_err,
                                                          Plane& out_err_q4) {
  if (H < 4 || W < 4) {
    return decode_channel_raster_exact(data, size, pos, H, W, c_idx, rct_mode, maxv, &C0, &C0_err, C1, C1_err,
                                       out_err_q4);
  }
  Plane P(H, W, 0);
  out_err_q4 = Plane(H, W, 0);
  Plane stage_err(H, W, 0);

  const int H2 = (H + 1) / 2;
  const int W2 = (W + 1) / 2;
  {
    Plane sG(H2, W2);
    Plane sGe(H2, W2);
    Plane sU(H2, W2);
    Plane sUe(H2, W2);
    Plane sOutE;
    for (int iy = 0; iy < H2; ++iy) {
      for (int ix = 0; ix < W2; ++ix) {
        sG.at(iy, ix) = C0.at(iy * 2, ix * 2);
        sGe.at(iy, ix) = C0_err.at(iy * 2, ix * 2);
        if (C1 != nullptr) {
          sU.at(iy, ix) = C1->at(iy * 2, ix * 2);
          sUe.at(iy, ix) = C1_err->at(iy * 2, ix * 2);
        }
      }
    }
    const Plane sP = decode_channel_raster_exact(data, size, pos, H2, W2, c_idx, rct_mode, maxv, &sG, &sGe,
                                                 C1 != nullptr ? &sU : nullptr, C1 != nullptr ? &sUe : nullptr, sOutE);
    for (int iy = 0; iy < H2; ++iy) {
      for (int ix = 0; ix < W2; ++ix) {
        P.at(iy * 2, ix * 2) = sP.at(iy, ix);
        const int32_t eq4 = sOutE.at(iy, ix);
        out_err_q4.at(iy * 2, ix * 2) = eq4;
        stage_err.at(iy * 2, ix * 2) = static_cast<int32_t>(iabs64(eq4));
      }
    }
  }

  const int qsh = maxv > 255 ? 8 : 0;
  const int py_s[4] = {0, 1, 0, 1};
  const int px_s[4] = {0, 1, 1, 0};
  for (int stg = 1; stg <= 3; ++stg) {
    const int role = (C1 == nullptr ? 3 : 6) + (stg - 1);
    const int tree_mode = 1 + stg;
    const int Knb = stg == 1 ? 12 : 10;
    const int D = Knb + 10 + (C1 != nullptr ? 6 : 0);
    const int n_sec = 4;
    if (pos + 2 > size) throw CodecError("truncated frontier coset ridge header");
    const uint16_t wn = static_cast<uint16_t>(data[pos] | (uint16_t(data[pos + 1]) << 8));
    pos += 2;
    if (wn == 0 || pos + wn > size) throw CodecError("truncated frontier coset ridge weights");
    BitReader wr(data + pos, wn);
    pos += wn;
    const int nlms_gain = static_cast<int>(wr.read_bits(5));
    const auto w_q8 = read_sector_ridge(wr, role, n_sec, D);
    wr.check_zero_tail();

    OpenedMAStream mst = open_ma_stream(data, size, pos, 11, tree_mode);
    AdaptiveStreamDecoder dec(mst.payload, mst.nu, mst.use_adapt, mst.tree.leaf_sigmas);

    std::array<std::array<int32_t, 36>, 4> wlms{};
    std::array<std::array<int16_t, 36>, 4> ws_q8{};
    for (int s = 0; s < n_sec; ++s) {
      for (int a = 0; a < D; ++a) {
        const int16_t wf = w_q8[static_cast<size_t>(s)][a];
        ws_q8[s][a] = wf;
        wlms[s][a] = static_cast<int32_t>(wf) << 8;
      }
    }

    int32_t f_tmp[32];
    int16_t prp[11];
    int sec_tmp = 0;
    int i = 0;
    for (int y = py_s[stg]; y < H; y += 2) {
      for (int x = px_s[stg]; x < W; x += 2, ++i) {
        const int64_t b_q4 =
            interp_features_exact(P, C0, C0_err, C1, C1_err, stage_err, maxv, stg, y, x, f_tmp, sec_tmp, prp);
        int64_t acc = 128;
        const auto& w8 = ws_q8[sec_tmp];
        for (int a = 0; a < D; ++a) acc += static_cast<int64_t>(f_tmp[a]) * w8[a];
        const int64_t raw_mu = b_q4 + (acc >> 8);
        const auto [blo, bhi] = pixel_bounds_q4(rct_mode, c_idx, maxv, &C0, C1, y, x);
        const int64_t mu_q4 = std::clamp(raw_mu, blo, bhi);
        const int64_t mu_i = (mu_q4 + 8) >> 4;
        const int fr = std::clamp(static_cast<int>(mu_q4 - (mu_i << 4) + 8), 0, 15);
        const int leaf = mst.tree.walk(prp);
        const int32_t r = dec.read(leaf, fr);
        const int64_t truth64 = mu_i + static_cast<int64_t>(r);
        if (truth64 < -262144 || truth64 > 262144) throw CodecError("decoded frontier coset sample out of range");
        const int32_t truth = static_cast<int32_t>(truth64);
        P.samples[y * W + x] = truth;
        const int32_t eq4 = static_cast<int32_t>((truth64 << 4) - mu_q4);
        out_err_q4.samples[y * W + x] = eq4;
        stage_err.samples[y * W + x] = static_cast<int32_t>(iabs64(eq4));
        if (nlms_gain > 0) {
          nlms_update(wlms[sec_tmp].data(), ws_q8[sec_tmp].data(), D, f_tmp, (truth64 << 4) - raw_mu, nlms_gain, qsh);
        }
      }
    }
    dec.check_finished();
  }
  return P;
}

// Fast 0.04ms gradient-conditioned midpoint linearity screener for 2x2 Coset vs Raster Chroma.
// Measures |2*c - w - e| + |2*c - n - s| over |e - w| + |s - n| at active chroma gradients:
// bandlimited/4:2:0-upsampled chroma ramps have lin < 0.43, whereas native 4:4:4 chroma has ~0.50..0.95.
double screen_coset_vs_raster(const Plane& P, const Plane& /*C0*/, int maxv) {
  const int H = P.height;
  const int W = P.width;
  if (H < 8 || W < 8) return 1.0;
  const int thr = 8 << (maxv > 255 ? 8 : 0);
  const int32_t* ps = P.samples.data();
  int64_t sum_mid = 0;
  int64_t sum_grad2 = 0;
  int64_t count = 0;
  for (int y = 2; y + 2 < H; y += 2) {
    const int32_t* row = ps + y * W;
    const int32_t* r_up = row - W;
    const int32_t* r_dn = row + W;
    for (int x = 2; x + 2 < W; x += 2) {
      const int32_t c = row[x];
      const int32_t w = row[x - 1];
      const int32_t e = row[x + 1];
      const int32_t n = r_up[x];
      const int32_t s = r_dn[x];
      const int32_t grad2 = iabs32(e - w) + iabs32(s - n);
      if (grad2 >= thr) {
        sum_mid += iabs32(2 * c - w - e) + iabs32(2 * c - n - s);
        sum_grad2 += grad2;
        ++count;
      }
    }
  }
  if (count < 32 || sum_grad2 <= 0) return 1.0;
  return static_cast<double>(sum_mid) / static_cast<double>(sum_grad2);
}

struct RctCandidate {
  int mode;
  int perm;
  double cost;
};

// Fast 0.2ms strided MED entropy proxy across color transforms!
std::vector<RctCandidate> screen_color_transforms(const std::vector<Plane>& raw_planes) {
  const int H = raw_planes[0].height;
  const int W = raw_planes[0].width;
  const RctCandidate specs[6] = {
      {MODE_SUB, 0, 0.0},
      {kModeSubAvg, 0, 0.0},
      {kModeSubAvg, 1, 0.0},
      {kModeSubAvg, 2, 0.0},
      {kModeSubAvg, 5, 0.0},
      {MODE_IDENTITY, 0, 0.0},
  };
  std::vector<RctCandidate> out(std::begin(specs), std::end(specs));
  if (H < 4 || W < 4) return out;

  const int32_t* r = raw_planes[0].samples.data();
  const int32_t* g = raw_planes[1].samples.data();
  const int32_t* b = raw_planes[2].samples.data();

  auto med3 = [](int32_t wv, int32_t nv, int32_t nw) -> int32_t {
    const int32_t hi = std::max(wv, nv);
    const int32_t lo = std::min(wv, nv);
    return nw < lo ? hi : (nw > hi ? lo : (wv + nv - nw));
  };

  for (auto& cand : out) {
    auto sample_ch = [&](int idx, int ch_k) -> int32_t {
      if (cand.mode == MODE_IDENTITY) return ch_k == 0 ? r[idx] : (ch_k == 1 ? g[idx] : b[idx]);
      if (cand.mode == MODE_SUB) return ch_k == 0 ? g[idx] : (ch_k == 1 ? (r[idx] - g[idx]) : (b[idx] - g[idx]));
      const int ia = kSubAvgPerms[cand.perm][0];
      const int ib = kSubAvgPerms[cand.perm][1];
      const int ic = kSubAvgPerms[cand.perm][2];
      const int32_t va = ia == 0 ? r[idx] : (ia == 1 ? g[idx] : b[idx]);
      if (ch_k == 0) return va;
      const int32_t vb = ib == 0 ? r[idx] : (ib == 1 ? g[idx] : b[idx]);
      if (ch_k == 1) return vb - va;
      const int32_t vc = ic == 0 ? r[idx] : (ic == 1 ? g[idx] : b[idx]);
      return vc - ((va + vb) >> 1);
    };

    double bits = 0.0;
    for (int y = 2; y < H; y += 4) {
      for (int x = 2; x < W; x += 4) {
        const int idx = y * W + x;
        const int idxW = idx - 1;
        const int idxN = idx - W;
        const int idxNW = idx - W - 1;
        for (int ck = 0; ck < 3; ++ck) {
          const int32_t err = iabs32(sample_ch(idx, ck) -
                                     med3(sample_ch(idxW, ck), sample_ch(idxN, ck), sample_ch(idxNW, ck)));
          bits += std::log2(static_cast<double>(err) + 1.5);
        }
      }
    }
    cand.cost = bits;
  }
  std::sort(out.begin(), out.end(), [](const RctCandidate& a, const RctCandidate& b_c) { return a.cost < b_c.cost; });
  return out;
}

template <typename F1, typename F2>
void run_two_parallel(bool enable, F1&& f1, F2&& f2) {
  if (!enable) {
    f1();
    f2();
    return;
  }
  std::exception_ptr ep;
  std::thread t([&]() {
    try {
      f1();
    } catch (...) {
      ep = std::current_exception();
    }
  });
  try {
    f2();
  } catch (...) {
    t.join();
    throw;
  }
  t.join();
  if (ep) std::rethrow_exception(ep);
}

}  // namespace

std::vector<uint8_t> encode_frontier(const std::vector<Plane>& raw_planes, bool u16, int effort) {
  if (raw_planes.empty()) throw CodecError("empty frontier tile");
  const int C = static_cast<int>(raw_planes.size());
  if (C != 1 && C != 3 && C != 4) throw CodecError("frontier supports 1, 3, or 4 channels");
  const int H = raw_planes[0].height;
  const int W = raw_planes[0].width;
  if (H <= 0 || W <= 0) throw CodecError("empty frontier plane");
  const int maxv = u16 ? 65535 : 255;
  const bool par_ok = (static_cast<int64_t>(H) * W >= 100000) && (std::thread::hardware_concurrency() >= 2);

  if (C < 3) {
    Plane e0;
    const auto b0 = encode_channel_raster_exact(raw_planes[0], 0, MODE_IDENTITY, maxv, effort, nullptr, nullptr,
                                                nullptr, nullptr, e0);
    std::vector<uint8_t> out{static_cast<uint8_t>(MODE_IDENTITY), 0, 0};
    out.insert(out.end(), b0.begin(), b0.end());
    return out;
  }

  auto best_chroma = [&](const Plane& P, int c_idx, int rm, const Plane& c0, const Plane& c0e, const Plane* c1,
                         const Plane* c1e, Plane& out_e, uint8_t& flag_bit) -> std::vector<uint8_t> {
    if (rm == MODE_IDENTITY || H < 8 || W < 8) {
      flag_bit = 0;
      return encode_channel_raster_exact(P, c_idx, rm, maxv, effort, &c0, &c0e, c1, c1e, out_e);
    }
    const double lin = screen_coset_vs_raster(P, c0, maxv);
    if (effort <= 1) {
      if (lin < 0.43) {
        flag_bit = 1;
        return encode_chroma_coset_exact(P, c_idx, rm, maxv, 1, c0, c0e, c1, c1e, out_e);
      }
      flag_bit = 0;
      return encode_channel_raster_exact(P, c_idx, rm, maxv, 1, &c0, &c0e, c1, c1e, out_e);
    }
    if (effort == 2) {
      if (lin < 0.41) {
        flag_bit = 1;
        return encode_chroma_coset_exact(P, c_idx, rm, maxv, effort, c0, c0e, c1, c1e, out_e);
      }
      if (lin > 0.47) {
        flag_bit = 0;
        return encode_channel_raster_exact(P, c_idx, rm, maxv, effort, &c0, &c0e, c1, c1e, out_e);
      }
    }
    Plane er;
    Plane ec;
    std::vector<uint8_t> br;
    std::vector<uint8_t> bc;
    run_two_parallel(
        par_ok,
        [&]() { br = encode_channel_raster_exact(P, c_idx, rm, maxv, effort, &c0, &c0e, c1, c1e, er); },
        [&]() { bc = encode_chroma_coset_exact(P, c_idx, rm, maxv, effort, c0, c0e, c1, c1e, ec); });
    if (bc.size() < br.size()) {
      out_e = std::move(ec);
      flag_bit = 1;
      return bc;
    }
    out_e = std::move(er);
    flag_bit = 0;
    return br;
  };

  const auto rct_ranked = screen_color_transforms(raw_planes);

  auto encode_full_rct = [&](int mode, int perm, Plane& out_e0, uint8_t& out_cflags) -> std::vector<uint8_t> {
    const auto pl = color_fwd_frontier(raw_planes, mode, perm);
    Plane e1, e2;
    uint8_t f1 = 0, f2 = 0;
    auto b0 = encode_channel_raster_exact(pl[0], 0, mode, maxv, effort, nullptr, nullptr, nullptr, nullptr, out_e0);
    auto b1 = best_chroma(pl[1], 1, mode, pl[0], out_e0, nullptr, nullptr, e1, f1);
    auto b2 = best_chroma(pl[2], 2, mode, pl[0], out_e0, &pl[1], &e1, e2, f2);
    out_cflags = static_cast<uint8_t>(f1 | (f2 << 1));
    std::vector<uint8_t> res{static_cast<uint8_t>(mode), static_cast<uint8_t>(perm), out_cflags};
    res.reserve(3 + b0.size() + b1.size() + b2.size());
    res.insert(res.end(), b0.begin(), b0.end());
    res.insert(res.end(), b1.begin(), b1.end());
    res.insert(res.end(), b2.begin(), b2.end());
    return res;
  };

  // Effort 1: Ultra-fast single-candidate execution (exact 3 channel passes total!)
  if (effort <= 1) {
    Plane best_e0;
    uint8_t cflags = 0;
    auto out = encode_full_rct(rct_ranked[0].mode, rct_ranked[0].perm, best_e0, cflags);
    if (C == 4) {
      Plane e3;
      const auto final_pl = color_fwd_frontier(raw_planes, rct_ranked[0].mode, rct_ranked[0].perm);
      const auto b3 = encode_channel_raster_exact(final_pl[3], 3, MODE_IDENTITY, maxv, effort, &final_pl[0],
                                                  &best_e0, nullptr, nullptr, e3);
      out.insert(out.end(), b3.begin(), b3.end());
    }
    return out;
  }

  // Check if top-ranked RCT is non-G-base (e.g., uncorrelated synthetic RGB or R/B-smooth graphics)
  const bool top_is_g = (rct_ranked[0].mode == MODE_SUB && rct_ranked[0].perm == 0) ||
                        (rct_ranked[0].mode == kModeSubAvg && (rct_ranked[0].perm == 0 || rct_ranked[0].perm == 1));
  if (!top_is_g && effort == 2 && rct_ranked[0].cost * 1.01 < rct_ranked[1].cost) {
    Plane best_e0;
    uint8_t cflags = 0;
    auto out = encode_full_rct(rct_ranked[0].mode, rct_ranked[0].perm, best_e0, cflags);
    if (C == 4) {
      Plane e3;
      const auto final_pl = color_fwd_frontier(raw_planes, rct_ranked[0].mode, rct_ranked[0].perm);
      const auto b3 = encode_channel_raster_exact(final_pl[3], 3, MODE_IDENTITY, maxv, effort, &final_pl[0],
                                                  &best_e0, nullptr, nullptr, e3);
      out.insert(out.end(), b3.begin(), b3.end());
    }
    return out;
  }

  // Shared-G search: encode Ch0 = G once!
  Plane eG;
  auto bG = encode_channel_raster_exact(raw_planes[1], 0, MODE_SUB, maxv, effort, nullptr, nullptr, nullptr,
                                        nullptr, eG);

  double cost_sub = 1e30, cost_avg0 = 1e30, cost_avg1 = 1e30;
  for (const auto& c : rct_ranked) {
    if (c.mode == MODE_SUB && c.perm == 0) cost_sub = c.cost;
    if (c.mode == kModeSubAvg && c.perm == 0) cost_avg0 = c.cost;
    if (c.mode == kModeSubAvg && c.perm == 1) cost_avg1 = c.cost;
  }

  int chosen_mode = kModeSubAvg;
  int chosen_perm = 0;
  uint8_t cflags = 0;
  Plane best_e0 = eG;
  std::vector<uint8_t> best_c0 = bG;
  std::vector<uint8_t> best_c1;
  std::vector<uint8_t> best_c2;

  auto eval_perm0_chain = [&](int& out_mode, uint8_t& out_cf, std::vector<uint8_t>& out_c1,
                              std::vector<uint8_t>& out_c2) {
    const auto sub_planes = color_fwd_frontier(raw_planes, MODE_SUB, 0);
    const auto avg0_planes = color_fwd_frontier(raw_planes, kModeSubAvg, 0);
    Plane eRG, dummy_e2a, dummy_e2b;
    uint8_t f1 = 0, f2_sub = 0, f2_avg = 0;
    out_c1 = best_chroma(sub_planes[1], 1, MODE_SUB, sub_planes[0], eG, nullptr, nullptr, eRG, f1);

    if (effort == 2 && cost_avg0 * 1.001 < cost_sub) {
      out_mode = kModeSubAvg;
      out_c2 = best_chroma(avg0_planes[2], 2, kModeSubAvg, avg0_planes[0], eG, &avg0_planes[1], &eRG, dummy_e2b,
                           f2_avg);
      out_cf = static_cast<uint8_t>(f1 | (f2_avg << 1));
    } else if (effort == 2 && cost_sub * 1.001 < cost_avg0) {
      out_mode = MODE_SUB;
      out_c2 = best_chroma(sub_planes[2], 2, MODE_SUB, sub_planes[0], eG, &sub_planes[1], &eRG, dummy_e2a, f2_sub);
      out_cf = static_cast<uint8_t>(f1 | (f2_sub << 1));
    } else {
      std::vector<uint8_t> bBG_sub, bBG_avg;
      run_two_parallel(
          par_ok,
          [&]() {
            bBG_sub =
                best_chroma(sub_planes[2], 2, MODE_SUB, sub_planes[0], eG, &sub_planes[1], &eRG, dummy_e2a, f2_sub);
          },
          [&]() {
            bBG_avg = best_chroma(avg0_planes[2], 2, kModeSubAvg, avg0_planes[0], eG, &avg0_planes[1], &eRG, dummy_e2b,
                                  f2_avg);
          });
      if (bBG_avg.size() < bBG_sub.size()) {
        out_mode = kModeSubAvg;
        out_cf = static_cast<uint8_t>(f1 | (f2_avg << 1));
        out_c2 = std::move(bBG_avg);
      } else {
        out_mode = MODE_SUB;
        out_cf = static_cast<uint8_t>(f1 | (f2_sub << 1));
        out_c2 = std::move(bBG_sub);
      }
    }
  };

  auto eval_perm1_chain = [&](uint8_t& out_cf, std::vector<uint8_t>& out_c1, std::vector<uint8_t>& out_c2) {
    const auto avg1 = color_fwd_frontier(raw_planes, kModeSubAvg, 1);
    Plane eBG1, eRG1;
    uint8_t g1 = 0, g2 = 0;
    out_c1 = best_chroma(avg1[1], 1, kModeSubAvg, avg1[0], eG, nullptr, nullptr, eBG1, g1);
    out_c2 = best_chroma(avg1[2], 2, kModeSubAvg, avg1[0], eG, &avg1[1], &eBG1, eRG1, g2);
    out_cf = static_cast<uint8_t>(g1 | (g2 << 1));
  };

  const bool prefer_avg1 = cost_avg1 < std::min(cost_sub, cost_avg0);
  if (effort == 2 && prefer_avg1) {
    chosen_mode = kModeSubAvg;
    chosen_perm = 1;
    eval_perm1_chain(cflags, best_c1, best_c2);
  } else if (effort >= 3) {
    int p0_mode = kModeSubAvg;
    uint8_t p0_cf = 0, p1_cf = 0;
    std::vector<uint8_t> p0_c1, p0_c2, p1_c1, p1_c2;
    run_two_parallel(
        par_ok, [&]() { eval_perm0_chain(p0_mode, p0_cf, p0_c1, p0_c2); },
        [&]() { eval_perm1_chain(p1_cf, p1_c1, p1_c2); });
    if (p1_c1.size() + p1_c2.size() < p0_c1.size() + p0_c2.size()) {
      chosen_mode = kModeSubAvg;
      chosen_perm = 1;
      cflags = p1_cf;
      best_c1 = std::move(p1_c1);
      best_c2 = std::move(p1_c2);
    } else {
      chosen_mode = p0_mode;
      chosen_perm = 0;
      cflags = p0_cf;
      best_c1 = std::move(p0_c1);
      best_c2 = std::move(p0_c2);
    }
  } else {
    chosen_perm = 0;
    eval_perm0_chain(chosen_mode, cflags, best_c1, best_c2);
  }

  // If top pre-screened transform was non-G (e.g. Identity or R/B-base), also verify it at effort >= 2!
  if (!top_is_g) {
    Plane alt_e0;
    uint8_t alt_cf = 0;
    auto alt_full = encode_full_rct(rct_ranked[0].mode, rct_ranked[0].perm, alt_e0, alt_cf);
    if (alt_full.size() < 3 + best_c0.size() + best_c1.size() + best_c2.size()) {
      if (C == 4) {
        Plane e3;
        const auto final_pl = color_fwd_frontier(raw_planes, rct_ranked[0].mode, rct_ranked[0].perm);
        const auto b3 = encode_channel_raster_exact(final_pl[3], 3, MODE_IDENTITY, maxv, effort, &final_pl[0],
                                                    &alt_e0, nullptr, nullptr, e3);
        alt_full.insert(alt_full.end(), b3.begin(), b3.end());
      }
      return alt_full;
    }
  }

  std::vector<uint8_t> out{static_cast<uint8_t>(chosen_mode), static_cast<uint8_t>(chosen_perm), cflags};
  out.reserve(3 + best_c0.size() + best_c1.size() + best_c2.size());
  out.insert(out.end(), best_c0.begin(), best_c0.end());
  out.insert(out.end(), best_c1.begin(), best_c1.end());
  out.insert(out.end(), best_c2.begin(), best_c2.end());

  if (C == 4) {
    Plane e3;
    const auto final_pl = color_fwd_frontier(raw_planes, chosen_mode, chosen_perm);
    const auto b3 = encode_channel_raster_exact(final_pl[3], 3, MODE_IDENTITY, maxv, effort, &final_pl[0],
                                                &best_e0, nullptr, nullptr, e3);
    out.insert(out.end(), b3.begin(), b3.end());
  }
  return out;
}

std::vector<Plane> decode_frontier(const uint8_t* data, size_t size, int height, int width, int channels, bool u16) {
  if (data == nullptr || size < 3) throw CodecError("truncated frontier tile");
  if (height <= 0 || width <= 0 || (channels != 1 && channels != 3 && channels != 4)) {
    throw CodecError("invalid frontier tile geometry");
  }
  const int rct_mode = data[0];
  const int rct_perm = data[1];
  const uint8_t cflags = data[2];
  if (rct_mode != MODE_IDENTITY && rct_mode != MODE_SUB && rct_mode != kModeSubAvg) {
    throw CodecError("invalid frontier color mode");
  }
  if ((rct_mode == MODE_IDENTITY && rct_perm != 0) || (rct_mode == MODE_SUB && rct_perm > 2) ||
      (rct_mode == kModeSubAvg && rct_perm > 5)) {
    throw CodecError("invalid frontier color permutation");
  }
  if ((cflags & ~0x03u) != 0 || (channels < 3 && (rct_mode != MODE_IDENTITY || cflags != 0))) {
    throw CodecError("invalid frontier flags");
  }

  size_t pos = 3;
  const int maxv = u16 ? 65535 : 255;

  if (channels < 3) {
    Plane e0;
    Plane p0 = decode_channel_raster_exact(data, size, pos, height, width, 0, MODE_IDENTITY, maxv, nullptr,
                                           nullptr, nullptr, nullptr, e0);
    if (pos != size) throw CodecError("trailing bytes in frontier grayscale tile");
    return {std::move(p0)};
  }

  std::vector<Plane> pl(static_cast<size_t>(channels));
  Plane e0;
  Plane e1;
  Plane e2;
  pl[0] = decode_channel_raster_exact(data, size, pos, height, width, 0, rct_mode, maxv, nullptr, nullptr,
                                      nullptr, nullptr, e0);
  if (cflags & 1u) {
    pl[1] = decode_chroma_coset_exact(data, size, pos, height, width, 1, rct_mode, maxv, pl[0], e0, nullptr,
                                      nullptr, e1);
  } else {
    pl[1] = decode_channel_raster_exact(data, size, pos, height, width, 1, rct_mode, maxv, &pl[0], &e0, nullptr,
                                        nullptr, e1);
  }
  if (cflags & 2u) {
    pl[2] = decode_chroma_coset_exact(data, size, pos, height, width, 2, rct_mode, maxv, pl[0], e0, &pl[1], &e1, e2);
  } else {
    pl[2] =
        decode_channel_raster_exact(data, size, pos, height, width, 2, rct_mode, maxv, &pl[0], &e0, &pl[1], &e1, e2);
  }
  if (channels == 4) {
    Plane e3;
    pl[3] = decode_channel_raster_exact(data, size, pos, height, width, 3, MODE_IDENTITY, maxv, &pl[0], &e0,
                                        nullptr, nullptr, e3);
  }
  if (pos != size) throw CodecError("trailing bytes in frontier tile");
  return color_inv_frontier(pl, rct_mode, rct_perm);
}

}  // namespace vpxc
