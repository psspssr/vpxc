#include "vpxc/frontier.hpp"

#include "vpxc/bitio.hpp"
#include "vpxc/color.hpp"
#include "vpxc/entropy.hpp"
#include "vpxc/error.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
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

inline int flog2(uint64_t x) { return x ? (63 - __builtin_clzll(x)) : 0; }

inline int64_t iabs64(int64_t v) {
  if (v >= 0) return v;
  if (v == INT64_MIN) return INT64_MAX;
  return -v;
}

uint32_t wp_wt(uint64_t x, uint32_t maxw) {
  int shift = flog2(x + 1) - 5;
  if (shift < 0) shift = 0;
  uint64_t idx = x >> shift;
  if (idx >= 64) return 4;
  return 4u + static_cast<uint32_t>((static_cast<uint64_t>(maxw) * kDiv[idx + 1]) >> shift);
}

// Clamps `c` into `[0, len - 1]` while preserving `(c & 1)` whenever `len >= 2`,
// guaranteeing boundary taps in 2x2 coset stages never read undecoded cosets.
inline int clamp_same_parity(int c, int len) {
  if (c >= 0 && c < len) return c;
  if (len <= 1) return 0;
  int p = c & 1;
  if (c < 0) return p < len ? p : 0;
  int last = len - 1;
  if ((last & 1) != p) --last;
  return last >= 0 ? last : 0;
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

std::vector<Plane> color_inv_frontier(const std::vector<Plane>& pl, int mode, int perm) {
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

// Exact Q4 pixel-domain feasibility bounds given already-decoded channels.
std::pair<int64_t, int64_t> pixel_bounds_q4(int rct_mode, int c_idx, int maxv, const Plane* C0, const Plane* C1,
                                            int y, int x) {
  if (rct_mode == MODE_IDENTITY || c_idx == 3) {
    return {0, static_cast<int64_t>(maxv) << 4};
  }
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

// ============================================================================
// Normative Integer Adaptive Core66 Model (shared identically by Encode/Decode)
// ============================================================================
struct AdaptiveCore66 {
  static constexpr int kFracBins = 8;

  struct Slot {
    int32_t counts[CORE_BINS];
    int32_t freq[CORE_BINS];
    int32_t cum[CORE_BINS + 1];
    int32_t pending = 0;
    int64_t total = 0;

    void init(const int32_t* prior_freq) {
      total = 0;
      int32_t run = 0;
      cum[0] = 0;
      for (int i = 0; i < CORE_BINS; ++i) {
        freq[i] = prior_freq[i];
        run += freq[i];
        cum[i + 1] = run;
        int32_t c = (prior_freq[i] + 64) >> 7;
        if (c < 1) c = 1;
        counts[i] = c;
        total += c;
      }
      pending = 0;
    }

    void observe(int sym) {
      counts[sym] += 4;
      total += 4;
      if (++pending >= 16) rebuild();
    }

    void rebuild() {
      constexpr int32_t kRemPool = RANS_M - CORE_BINS;
      int32_t sum = 0;
      int best_k = 32;
      int32_t best_c = -1;
      for (int i = 0; i < CORE_BINS; ++i) {
        const int32_t f = 1 + static_cast<int32_t>((static_cast<int64_t>(counts[i]) * kRemPool) / total);
        freq[i] = f;
        sum += f;
        if (counts[i] > best_c) {
          best_c = counts[i];
          best_k = i;
        }
      }
      freq[best_k] += (RANS_M - sum);
      int32_t run = 0;
      cum[0] = 0;
      for (int i = 0; i < CORE_BINS; ++i) {
        run += freq[i];
        cum[i + 1] = run;
      }
      if (total > 8192) {
        total = 0;
        for (int i = 0; i < CORE_BINS; ++i) {
          counts[i] = (counts[i] + 1) >> 1;
          total += counts[i];
        }
      }
      pending = 0;
    }
  };

  std::vector<Slot> slots;

  void init(int nu, const std::vector<int32_t>& leaf_sigmas) {
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

  Slot& at(int leaf, int frac) {
    return slots[static_cast<size_t>(leaf) * kFracBins + static_cast<size_t>((frac >> 1) & 7)];
  }
};

std::vector<uint8_t> encode_adaptive_rans_stream(const std::vector<int32_t>& residuals, int nu,
                                                 const std::vector<int32_t>& leaf_ids,
                                                 const std::vector<int32_t>& leaf_sigmas,
                                                 const std::vector<int32_t>& frac_idx) {
  const ParametricTables& tables = get_parametric_tables();
  const size_t n = residuals.size();
  AdaptiveCore66 model;
  model.init(nu, leaf_sigmas);

  struct Op {
    int32_t start;
    int32_t freq;
  };
  std::vector<Op> ops;
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

    ops.push_back(Op{slot.cum[sym], slot.freq[sym]});
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
          if (bit == 0) ops.push_back(Op{0, p0});
          else ops.push_back(Op{p0, RANS_M - p0});
          branch = (branch << 1) | bit;
        }
        if (rem_low > 0) raw_bits.write_bits(static_cast<uint32_t>(rem & ((1 << rem_low) - 1)), rem_low);
      } else {
        raw_bits.write_bits(static_cast<uint32_t>(rem), b);
      }
    }
  }

  RansEncoder enc;
  for (size_t i = ops.size(); i-- > 0;) enc.put(ops[i].start, ops[i].freq);
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

struct AdaptiveStreamDecoder {
  RansDecoder rans;
  BitReader raw_bits;
  int nu = 1;
  bool adaptive = false;
  std::vector<int32_t> leaf_sigmas;
  AdaptiveCore66 model;

  AdaptiveStreamDecoder(std::span<const uint8_t> blob, int nu_in, bool adapt_in,
                        const std::vector<int32_t>& sigmas)
      : rans(split_rans(blob)),
        raw_bits(blob.data() + 8 + split_rans_len(blob), split_extra_len(blob)),
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

  int pull_sym(const int32_t* cum, const int32_t* freq) {
    const int target = rans.cf();
    int lo = 0;
    int hi = CORE_BINS;
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (cum[mid] <= target) lo = mid;
      else hi = mid - 1;
    }
    const int sym = std::clamp(lo, 0, CORE_BINS - 1);
    rans.advance(cum[sym], freq[sym]);
    return sym;
  }

  int32_t read(int leaf, int frac) {
    if (leaf < 0 || static_cast<size_t>(leaf) >= leaf_sigmas.size()) {
      throw CodecError("frontier leaf index out of range");
    }
    const ParametricTables& tables = get_parametric_tables();
    const int sig = std::clamp(int(leaf_sigmas[static_cast<size_t>(leaf)]), 0, NUM_SIGMAS - 1);
    const int f_clamped = std::clamp(frac, 0, NUM_FRACS - 1);
    const int b = tables.shifts[sig];
    int sym = 0;
    if (adaptive) {
      auto& slot = model.at(leaf, f_clamped);
      sym = pull_sym(slot.cum, slot.freq);
      slot.observe(sym);
    } else {
      sym = pull_sym(tables.cums[nu][sig][f_clamped], tables.freqs[nu][sig][f_clamped]);
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
};

// ============================================================================
// MA Context Tree (Encode + Normative Integer Decode)
// ============================================================================
struct MATree {
  struct Node {
    int prop = -1;
    int thr = 0;
    int sigma = 4;
    int left = -1;
    int right = -1;
    int leaf_id = -1;
  };
  std::vector<Node> nodes;
  std::vector<int32_t> leaf_sigmas;

  int walk(const int16_t* props) const {
    int cur = 0;
    while (nodes[static_cast<size_t>(cur)].left >= 0) {
      const Node& nd = nodes[static_cast<size_t>(cur)];
      cur = (props[nd.prop] <= nd.thr) ? nd.left : nd.right;
    }
    return nodes[static_cast<size_t>(cur)].leaf_id;
  }

  void write(BitWriter& w) const {
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

  static MATree read(BitReader& r, int n_props) {
    MATree t;
    auto rec = [&](auto&& self, int depth) -> int {
      if (depth > 24 || t.nodes.size() >= 512) throw CodecError("MA tree limit exceeded");
      const int id = static_cast<int>(t.nodes.size());
      t.nodes.push_back(Node{});
      if (r.read_bits(1) == 1) {
        const int sig = static_cast<int>(r.read_bits(5));
        const int lid = static_cast<int>(t.leaf_sigmas.size());
        t.leaf_sigmas.push_back(sig);
        t.nodes[static_cast<size_t>(id)].sigma = sig;
        t.nodes[static_cast<size_t>(id)].leaf_id = lid;
        return id;
      }
      const int p = static_cast<int>(r.read_bits(4));
      if (p < 0 || p >= n_props) throw CodecError("MA tree property out of range");
      const int thr = r.read_signed_golomb();
      if (thr < -32768 || thr > 32767) throw CodecError("MA tree threshold out of range");
      const int l = self(self, depth + 1);
      const int rg = self(self, depth + 1);
      t.nodes[static_cast<size_t>(id)].prop = p;
      t.nodes[static_cast<size_t>(id)].thr = thr;
      t.nodes[static_cast<size_t>(id)].left = l;
      t.nodes[static_cast<size_t>(id)].right = rg;
      return id;
    };
    rec(rec, 0);
    return t;
  }
};

// ============================================================================
// Encode-Only Linear Solver & MA Tree Fit (IEEE-754 confined strictly to encode)
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

int best_sigma_subset(const ParametricTables& tbl, int nu, const std::vector<int>& idx,
                      const std::vector<int32_t>& resid, const std::vector<int32_t>& frac,
                      const double* neg_log, double& out_cost) {
  if (idx.empty()) {
    out_cost = 0.0;
    return 4;
  }
  double sum_q8 = 0.0;
  for (int i : idx) {
    const int64_t inner = static_cast<int64_t>(resid[static_cast<size_t>(i)]) * 16 -
                          (static_cast<int64_t>(frac[static_cast<size_t>(i)]) - 8);
    sum_q8 += static_cast<double>(iabs64(inner) << 4);
  }
  const double est = (sum_q8 / static_cast<double>(idx.size())) * 0.92 + 20.0;
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
  for (int cand = std::max(0, guess - 3); cand < std::min(NUM_SIGMAS, guess + 4); ++cand) {
    const int sh = tbl.shifts[cand];
    const int64_t bias = sh > 0 ? (1LL << (sh - 1)) : 0;
    double cost = static_cast<double>(idx.size() * static_cast<size_t>(sh));
    for (int i : idx) {
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
  out_cost = best_c;
  return best_s;
}

std::vector<uint8_t> encode_ma_stream(const std::vector<int32_t>& resid, const std::vector<int32_t>& frac,
                                      const std::vector<int16_t>& props, int n_props, int max_leaves) {
  const ParametricTables& tbl = get_parametric_tables();
  static const std::vector<double> neg_log = [] {
    std::vector<double> v(65536);
    for (int i = 0; i < 65536; ++i) v[static_cast<size_t>(i)] = -std::log2((i + 1) / 65536.0);
    return v;
  }();

  const int N = static_cast<int>(resid.size());
  MATree tree;
  if (N == 0) {
    tree.nodes.push_back(MATree::Node{-1, 0, 4, -1, -1, 0});
    tree.leaf_sigmas.push_back(4);
    BitWriter tw;
    tw.write_bits(1, 2);
    tw.write_bits(0, 1);
    tree.write(tw);
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

  struct LeafWork {
    int node_idx;
    std::vector<int> indices;
    double cost;
    double best_gain;
    int best_p;
    int best_t;
    int depth;
  };

  auto eval_leaf = [&](LeafWork& lw) {
    lw.best_gain = -1.0;
    if (lw.depth >= 18 || static_cast<int>(lw.indices.size()) < 96) return;
    const int step = std::max(1, static_cast<int>(lw.indices.size()) / 2048);
    std::vector<int> sub;
    sub.reserve(lw.indices.size() / static_cast<size_t>(step) + 1);
    for (size_t k = 0; k < lw.indices.size(); k += static_cast<size_t>(step)) sub.push_back(lw.indices[k]);
    double base_sub = 0.0;
    best_sigma_subset(tbl, 1, sub, resid, frac, neg_log.data(), base_sub);

    std::vector<int16_t> vals;
    vals.reserve(sub.size());
    std::vector<int> cand_t;
    cand_t.reserve(18);
    std::vector<int> lsub;
    std::vector<int> rsub;
    lsub.reserve(sub.size());
    rsub.reserve(sub.size());

    for (int p = 0; p < n_props; ++p) {
      int16_t mn = 32767;
      int16_t mx = -32768;
      vals.clear();
      for (int i : sub) {
        const int16_t v = props[static_cast<size_t>(i) * static_cast<size_t>(n_props) + static_cast<size_t>(p)];
        mn = std::min(mn, v);
        mx = std::max(mx, v);
        vals.push_back(v);
      }
      if (mn == mx) continue;
      std::sort(vals.begin(), vals.end());
      cand_t.clear();
      if (mn <= 0 && 0 < mx) cand_t.push_back(0);
      if (mn <= 1 && 1 < mx) cand_t.push_back(1);
      for (int q = 1; q <= 15; ++q) {
        cand_t.push_back(vals[(vals.size() * static_cast<size_t>(q)) / 16]);
      }
      std::sort(cand_t.begin(), cand_t.end());
      cand_t.erase(std::unique(cand_t.begin(), cand_t.end()), cand_t.end());
      for (int thr : cand_t) {
        if (thr < mn || thr >= mx) continue;
        lsub.clear();
        rsub.clear();
        for (int i : sub) {
          if (props[static_cast<size_t>(i) * static_cast<size_t>(n_props) + static_cast<size_t>(p)] <= thr) {
            lsub.push_back(i);
          } else {
            rsub.push_back(i);
          }
        }
        if (lsub.size() < 40 || rsub.size() < 40) continue;
        double lc = 0.0;
        double rc = 0.0;
        best_sigma_subset(tbl, 1, lsub, resid, frac, neg_log.data(), lc);
        best_sigma_subset(tbl, 1, rsub, resid, frac, neg_log.data(), rc);
        const double gain = (base_sub - lc - rc) * static_cast<double>(step);
        if (gain > lw.best_gain) {
          lw.best_gain = gain;
          lw.best_p = p;
          lw.best_t = thr;
        }
      }
    }
  };

  std::vector<int> all_idx(static_cast<size_t>(N));
  std::iota(all_idx.begin(), all_idx.end(), 0);
  double root_c = 0.0;
  const int root_s = best_sigma_subset(tbl, 1, all_idx, resid, frac, neg_log.data(), root_c);
  tree.nodes.push_back(MATree::Node{-1, 0, root_s, -1, -1, -1});
  std::vector<LeafWork> leaves;
  leaves.push_back(LeafWork{0, std::move(all_idx), root_c, -1.0, 0, 0, 0});
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
    const int lid = static_cast<int>(tree.nodes.size());
    tree.nodes.push_back(MATree::Node{-1, 0, 4, -1, -1, -1});
    const int rid = static_cast<int>(tree.nodes.size());
    tree.nodes.push_back(MATree::Node{-1, 0, 4, -1, -1, -1});
    tree.nodes[static_cast<size_t>(cur.node_idx)].prop = cur.best_p;
    tree.nodes[static_cast<size_t>(cur.node_idx)].thr = cur.best_t;
    tree.nodes[static_cast<size_t>(cur.node_idx)].left = lid;
    tree.nodes[static_cast<size_t>(cur.node_idx)].right = rid;

    LeafWork lw_l{lid, std::move(li), 0.0, -1.0, 0, 0, cur.depth + 1};
    LeafWork lw_r{rid, std::move(ri), 0.0, -1.0, 0, 0, cur.depth + 1};
    eval_leaf(lw_l);
    eval_leaf(lw_r);
    leaves.push_back(std::move(lw_l));
    leaves.push_back(std::move(lw_r));
  }

  int best_nu = 1;
  double best_nu_c = std::numeric_limits<double>::infinity();
  for (int nu = 0; nu < NUM_NUS; ++nu) {
    double tc = 0.0;
    for (auto& lw : leaves) {
      double lc = 0.0;
      best_sigma_subset(tbl, nu, lw.indices, resid, frac, neg_log.data(), lc);
      tc += lc;
    }
    if (tc < best_nu_c) {
      best_nu_c = tc;
      best_nu = nu;
    }
  }

  for (auto& lw : leaves) {
    double lc = 0.0;
    tree.nodes[static_cast<size_t>(lw.node_idx)].sigma =
        best_sigma_subset(tbl, best_nu, lw.indices, resid, frac, neg_log.data(), lc);
  }
  tree.leaf_sigmas.clear();
  auto number_dfs = [&](auto&& self, int id) -> void {
    if (tree.nodes[static_cast<size_t>(id)].left < 0) {
      tree.nodes[static_cast<size_t>(id)].leaf_id = static_cast<int>(tree.leaf_sigmas.size());
      tree.leaf_sigmas.push_back(tree.nodes[static_cast<size_t>(id)].sigma);
      return;
    }
    self(self, tree.nodes[static_cast<size_t>(id)].left);
    self(self, tree.nodes[static_cast<size_t>(id)].right);
  };
  number_dfs(number_dfs, 0);

  std::vector<int32_t> sig_idx(static_cast<size_t>(N));
  std::vector<int32_t> leaf_ids(static_cast<size_t>(N));
  for (int i = 0; i < N; ++i) {
    const int lid = tree.walk(&props[static_cast<size_t>(i) * static_cast<size_t>(n_props)]);
    leaf_ids[static_cast<size_t>(i)] = lid;
    sig_idx[static_cast<size_t>(i)] = tree.leaf_sigmas[static_cast<size_t>(lid)];
  }

  const auto static_pay = encode_parametric_rans(resid, best_nu, sig_idx, frac);
  const auto adapt_pay = encode_adaptive_rans_stream(resid, best_nu, leaf_ids, tree.leaf_sigmas, frac);
  const bool use_adapt = adapt_pay.size() < static_pay.size();
  const auto& pay = use_adapt ? adapt_pay : static_pay;

  BitWriter tw;
  tw.write_bits(static_cast<uint32_t>(best_nu & 3), 2);
  tw.write_bits(use_adapt ? 1u : 0u, 1);
  tree.write(tw);
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

struct OpenedMAStream {
  MATree tree;
  int nu = 1;
  bool use_adapt = false;
  std::span<const uint8_t> payload;
};

OpenedMAStream open_ma_stream(const uint8_t* data, size_t size, size_t& pos, int n_props) {
  if (pos + 2 > size) throw CodecError("truncated MA stream header");
  const uint16_t hn = static_cast<uint16_t>(data[pos] | (uint16_t(data[pos + 1]) << 8));
  pos += 2;
  if (hn == 0 || pos + hn + 4 > size) throw CodecError("truncated MA stream header");
  BitReader hr(data + pos, hn);
  pos += hn;
  OpenedMAStream st;
  st.nu = static_cast<int>(hr.read_bits(2));
  st.use_adapt = hr.read_bits(1) == 1;
  st.tree = MATree::read(hr, n_props);
  const uint32_t pn = uint32_t(data[pos]) | (uint32_t(data[pos + 1]) << 8) |
                      (uint32_t(data[pos + 2]) << 16) | (uint32_t(data[pos + 3]) << 24);
  pos += 4;
  if (pn > size - pos) throw CodecError("truncated MA stream payload");
  st.payload = std::span<const uint8_t>(data + pos, pn);
  pos += pn;
  return st;
}

void fit_and_write_sector_ridge(BitWriter& hw, int n_sec, int D, const std::vector<int64_t>& all_f,
                                const std::vector<int64_t>& base_v, const std::vector<int>& sec_v,
                                const int32_t* target_samples, const std::pair<int, int>* coords, int W_stride,
                                std::vector<std::vector<int64_t>>& out_w) {
  const int N = static_cast<int>(base_v.size());
  out_w.assign(static_cast<size_t>(n_sec), std::vector<int64_t>(static_cast<size_t>(D), 0));
  if (N < 64 || D == 0) {
    hw.write_bits(0, 1);
    return;
  }
  auto get_y = [&](int i) -> double {
    const int32_t s = coords != nullptr
                          ? target_samples[coords[i].first * W_stride + coords[i].second]
                          : target_samples[i];
    return static_cast<double>((static_cast<int64_t>(s) << 4) - base_v[static_cast<size_t>(i)]);
  };

  std::vector<double> Gg(static_cast<size_t>(D * D), 0.0);
  std::vector<double> Rg(static_cast<size_t>(D), 0.0);
  std::vector<double> sol_g;
  for (int i = 0; i < N; ++i) {
    const double yv = get_y(i);
    const int64_t* f = &all_f[static_cast<size_t>(i) * static_cast<size_t>(D)];
    for (int a = 0; a < D; ++a) {
      const double fa = static_cast<double>(f[a]);
      Rg[static_cast<size_t>(a)] += fa * yv;
      for (int b = a; b < D; ++b) {
        Gg[static_cast<size_t>(a * D + b)] += fa * static_cast<double>(f[b]);
      }
    }
  }
  for (int a = 0; a < D; ++a) {
    for (int b = 0; b < a; ++b) Gg[static_cast<size_t>(a * D + b)] = Gg[static_cast<size_t>(b * D + a)];
    Gg[static_cast<size_t>(a * D + a)] += std::max(Gg[static_cast<size_t>(a * D + a)] * 1e-4, 512.0);
  }
  std::vector<int64_t> wg(static_cast<size_t>(D), 0);
  if (solve_spd(Gg, Rg, D, sol_g)) {
    for (int a = 0; a < D; ++a) {
      wg[static_cast<size_t>(a)] = std::clamp<int64_t>(std::llround(sol_g[static_cast<size_t>(a)] * 256.0), -2047, 2047);
    }
  }

  hw.write_bits(1, 1);
  for (int a = 0; a < D; ++a) hw.write_signed_golomb(static_cast<int>(wg[static_cast<size_t>(a)]));

  for (int s = 0; s < n_sec; ++s) {
    out_w[static_cast<size_t>(s)] = wg;
    std::vector<int> idx;
    for (int i = 0; i < N; ++i) {
      if (sec_v[static_cast<size_t>(i)] == s) idx.push_back(i);
    }
    if (static_cast<int>(idx.size()) < 96) {
      hw.write_bits(0, 1);
      continue;
    }
    std::vector<double> Gs(static_cast<size_t>(D * D), 0.0);
    std::vector<double> Rs(static_cast<size_t>(D), 0.0);
    std::vector<double> sol_s;
    double err_g = 0.0;
    for (int i : idx) {
      const double yv = get_y(i);
      const int64_t* f = &all_f[static_cast<size_t>(i) * static_cast<size_t>(D)];
      __int128 accg = 128;
      for (int a = 0; a < D; ++a) accg += static_cast<__int128>(f[a]) * wg[static_cast<size_t>(a)];
      const double yg = yv - static_cast<double>(static_cast<int64_t>(accg >> 8));
      err_g += std::fabs(yg);
      for (int a = 0; a < D; ++a) {
        const double fa = static_cast<double>(f[a]);
        Rs[static_cast<size_t>(a)] += fa * yg;
        for (int b = a; b < D; ++b) Gs[static_cast<size_t>(a * D + b)] += fa * static_cast<double>(f[b]);
      }
    }
    for (int a = 0; a < D; ++a) {
      for (int b = 0; b < a; ++b) Gs[static_cast<size_t>(a * D + b)] = Gs[static_cast<size_t>(b * D + a)];
      Gs[static_cast<size_t>(a * D + a)] += std::max(Gs[static_cast<size_t>(a * D + a)] * 5e-4, 1024.0);
    }
    if (!solve_spd(Gs, Rs, D, sol_s)) {
      hw.write_bits(0, 1);
      continue;
    }
    std::vector<int64_t> dw(static_cast<size_t>(D), 0);
    int hdr_bits = 1;
    for (int a = 0; a < D; ++a) {
      const int64_t ws = std::clamp<int64_t>(
          wg[static_cast<size_t>(a)] + std::llround(sol_s[static_cast<size_t>(a)] * 256.0), -2047, 2047);
      dw[static_cast<size_t>(a)] = ws - wg[static_cast<size_t>(a)];
      const uint64_t mag = (static_cast<uint64_t>(iabs64(dw[static_cast<size_t>(a)])) << 1) + 1;
      hdr_bits += 2 * (64 - __builtin_clzll(mag));
    }
    double err_s = 0.0;
    for (int i : idx) {
      const double yv = get_y(i);
      const int64_t* f = &all_f[static_cast<size_t>(i) * static_cast<size_t>(D)];
      __int128 accs = 128;
      for (int a = 0; a < D; ++a) {
        accs += static_cast<__int128>(f[a]) * (wg[static_cast<size_t>(a)] + dw[static_cast<size_t>(a)]);
      }
      err_s += std::fabs(yv - static_cast<double>(static_cast<int64_t>(accs >> 8)));
    }
    const double nd = static_cast<double>(idx.size());
    const double gain = nd * (std::log2(err_g / nd + 4.0) - std::log2(err_s / nd + 4.0));
    if (gain > static_cast<double>(hdr_bits) + 6.0) {
      hw.write_bits(1, 1);
      for (int a = 0; a < D; ++a) {
        out_w[static_cast<size_t>(s)][static_cast<size_t>(a)] = wg[static_cast<size_t>(a)] + dw[static_cast<size_t>(a)];
        hw.write_signed_golomb(static_cast<int>(dw[static_cast<size_t>(a)]));
      }
    } else {
      hw.write_bits(0, 1);
    }
  }
}

std::vector<std::vector<int64_t>> read_sector_ridge(BitReader& hr, int n_sec, int D) {
  std::vector<std::vector<int64_t>> out_w(static_cast<size_t>(n_sec), std::vector<int64_t>(static_cast<size_t>(D), 0));
  if (hr.read_bits(1) == 0) return out_w;
  std::vector<int64_t> wg(static_cast<size_t>(D), 0);
  for (int a = 0; a < D; ++a) {
    const int g = hr.read_signed_golomb();
    if (g < -2047 || g > 2047) throw CodecError("frontier ridge weight out of range");
    wg[static_cast<size_t>(a)] = g;
  }
  for (int s = 0; s < n_sec; ++s) {
    out_w[static_cast<size_t>(s)] = wg;
    if (hr.read_bits(1) == 1) {
      for (int a = 0; a < D; ++a) {
        const int d = hr.read_signed_golomb();
        if (d < -4096 || d > 4096) throw CodecError("frontier ridge delta out of range");
        out_w[static_cast<size_t>(s)][static_cast<size_t>(a)] =
            std::clamp<int64_t>(wg[static_cast<size_t>(a)] + d, -2047, 2047);
      }
    }
  }
  return out_w;
}

// ============================================================================
// Causal L-Template Non-Local Patch Predictor (Shared Integer Encode/Decode)
// ============================================================================
struct PatchPredictor {
  struct Slot {
    int16_t y = -1;
    int16_t x = -1;
  };
  std::vector<std::array<Slot, 4>> table;
  std::vector<uint8_t> ptr;
  int qsh = 0;

  explicit PatchPredictor(int maxv) : table(65536), ptr(65536, 0), qsh(maxv > 255 ? 8 : 0) {}

  uint16_t hash_key(int64_t d1, int64_t d2, int64_t d3, int64_t d4) const {
    const int sh = qsh + 1;
    const uint32_t h = (static_cast<uint32_t>(d1 >> sh) * 73856093u) ^
                       (static_cast<uint32_t>(d2 >> sh) * 19349663u) ^
                       (static_cast<uint32_t>(d3 >> sh) * 83492791u) ^
                       (static_cast<uint32_t>(d4 >> sh) * 2654435761u);
    return static_cast<uint16_t>((h ^ (h >> 16)) & 0xffffu);
  }

  std::pair<int64_t, int> query(const Plane& P, int y, int x, int64_t fallback_q4) const {
    if (y < 2 || x < 2 || x + 1 >= P.width) return {fallback_q4, 0};
    const int64_t Wv = P.at(y, x - 1);
    const int64_t Nv = P.at(y - 1, x);
    const int64_t NW = P.at(y - 1, x - 1);
    const int64_t NE = P.at(y - 1, x + 1);
    const int64_t WW = P.at(y, x - 2);
    const int64_t NN = P.at(y - 2, x);
    const uint16_t h = hash_key(Wv - NW, Nv - NW, NE - Nv, WW - Wv);
    const auto& bucket = table[h];
    int64_t best_num = 0;
    int64_t best_den = 0;
    int64_t best_sad = 999999;
    for (int k = 0; k < 4; ++k) {
      const int my = bucket[static_cast<size_t>(k)].y;
      const int mx = bucket[static_cast<size_t>(k)].x;
      if (my < 2 || mx < 2 || mx + 1 >= P.width) continue;
      if (std::abs(my - y) + std::abs(mx - x) > 192) continue;
      const int64_t mNW = P.at(my - 1, mx - 1);
      const int64_t mW = P.at(my, mx - 1);
      const int64_t mN = P.at(my - 1, mx);
      const int64_t sad = (iabs64((mW - mNW) - (Wv - NW)) +
                           iabs64((mN - mNW) - (Nv - NW)) +
                           iabs64((int64_t(P.at(my - 1, mx + 1)) - mN) - (NE - Nv)) +
                           iabs64((int64_t(P.at(my, mx - 2)) - mW) - (WW - Wv)) +
                           iabs64((int64_t(P.at(my - 2, mx)) - mN) - (NN - Nv))) >>
                          qsh;
      if (sad < 20) {
        const int64_t wt = 256 / (sad * sad + 4);
        const int64_t cand = NW + (static_cast<int64_t>(P.at(my, mx)) - mNW);
        best_num += wt * (cand << 4);
        best_den += wt;
        if (sad < best_sad) best_sad = sad;
      }
    }
    if (best_den == 0) return {fallback_q4, 0};
    const int conf = best_sad == 0 ? 15 : (best_sad <= 2 ? 10 : (best_sad <= 6 ? 5 : 2));
    return {best_num / best_den, conf};
  }

  void insert(const Plane& P, int y, int x) {
    if (y < 2 || x < 2 || x + 1 >= P.width || y > 32767 || x > 32767) return;
    const int64_t Wv = P.at(y, x - 1);
    const int64_t Nv = P.at(y - 1, x);
    const int64_t NW = P.at(y - 1, x - 1);
    const int64_t NE = P.at(y - 1, x + 1);
    const int64_t WW = P.at(y, x - 2);
    const uint16_t h = hash_key(Wv - NW, Nv - NW, NE - Nv, WW - Wv);
    uint8_t& p = ptr[h];
    table[h][p & 3] = Slot{static_cast<int16_t>(y), static_cast<int16_t>(x)};
    ++p;
  }
};

// ============================================================================
// Shared Raster Predictor State (Bit-Exact Encode & Decode)
// ============================================================================
struct RasterState {
  static constexpr int KEXP = 8;
  int H;
  int W;
  int c_idx;
  int rct_mode;
  int maxv;
  int qsh;
  int D;
  int n_sec;
  int stride;
  std::vector<uint32_t> perr[KEXP];
  std::vector<int64_t> wp_err;
  Plane true_err;
  PatchPredictor patcher;
  uint32_t mw[KEXP] = {0xd, 0xc, 0xc, 0xb, 0xc, 0xa, 0xa, 0xb};

  RasterState(int h, int w, int c, int rm, int mv, bool has_c0, bool has_c1)
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
        patcher(mv) {
    for (int k = 0; k < KEXP; ++k) perr[k].assign(static_cast<size_t>(stride * 2), 0);
  }

  static int64_t at_clamped(const Plane& pl, int y, int x) {
    return pl.at(std::clamp(y, 0, pl.height - 1), std::clamp(x, 0, pl.width - 1));
  }

  int64_t predict_features(const Plane& P, const Plane* C0, const Plane* C0_err, const Plane* C1,
                           const Plane* C1_err, int y, int x, int64_t* f, int& sec, int16_t* prp,
                           int64_t pr_out[KEXP]) const {
    const int cur = (y & 1) ? 0 : stride;
    const int prv = (y & 1) ? stride : 0;
    const int64_t Nv = y > 0 ? P.at(y - 1, x) : (x > 0 ? P.at(y, x - 1) : 0);
    const int64_t Wv = x > 0 ? P.at(y, x - 1) : Nv;
    const int64_t NE = y > 0 ? P.at(y - 1, std::min(x + 1, W - 1)) : Nv;
    const int64_t NW = (y > 0 && x > 0) ? P.at(y - 1, x - 1) : Nv;
    const int64_t NN = y > 1 ? P.at(y - 2, x) : Nv;
    const int64_t WW = x > 1 ? P.at(y, x - 2) : Wv;
    const int64_t NWW = (y > 0 && x > 1) ? P.at(y - 1, x - 2) : Wv;
    const int64_t NNW = (y > 1 && x > 0) ? P.at(y - 2, x - 1) : Nv;
    const int64_t NNE = (y > 1 && x + 1 < W) ? P.at(y - 2, x + 1) : NE;
    const int64_t NEE = (y > 0 && x + 2 < W) ? P.at(y - 1, x + 2) : NE;

    const int64_t hi = std::max(Wv, Nv);
    const int64_t lo = std::min(Wv, Nv);
    const int64_t med = (NW < lo) ? hi : ((NW > hi) ? lo : (Wv + Nv - NW));
    const int64_t dh_g = iabs64(Wv - WW) + iabs64(Nv - NW) + iabs64(Nv - NE);
    const int64_t dv_g = iabs64(Wv - NW) + iabs64(Nv - NN) + iabs64(NE - NNE);
    const int64_t gap_thr = 80LL << qsh;
    const int64_t gap = (dv_g - dh_g > gap_thr)
                            ? Wv
                            : ((dh_g - dv_g > gap_thr) ? Nv : (((Wv + Nv) >> 1) + ((NE - NW) >> 2)));
    const int64_t planar = (2 * Wv + 2 * Nv - NW + NE + 2) >> 2;

    const int pN = prv + x;
    const int pNE = x < W - 1 ? pN + 1 : pN;
    const int pNW = x > 0 ? pN - 1 : pN;
    const int pW = x > 0 ? cur + x - 1 : pN;
    const int64_t teW = x == 0 ? 0 : wp_err[static_cast<size_t>(cur + x - 1)];
    const int64_t teN = wp_err[static_cast<size_t>(pN)];
    const int64_t teNW = wp_err[static_cast<size_t>(pNW)];
    const int64_t teNE = wp_err[static_cast<size_t>(pNE)];
    const int64_t n = Nv << 4;
    const int64_t wv = Wv << 4;
    const int64_t ne = NE << 4;
    const int64_t nw = NW << 4;
    const int64_t nn = NN << 4;

    const auto [patch_q4, patch_conf] = patcher.query(P, y, x, med << 4);

    pr_out[0] = wv + ne - n;
    pr_out[1] = n - (((teN + teW + teNE) * 8) >> 5);
    pr_out[2] = wv - (((teN + teW + teNW) * 8) >> 5);
    pr_out[3] = n - ((teNW * 4 + teNE * 3 + (nn - n) * 23 + (nw - wv) * 2) >> 5);
    pr_out[4] = med << 4;
    pr_out[5] = gap << 4;
    pr_out[6] = planar << 4;
    pr_out[7] = patch_q4;

    uint32_t wts[KEXP];
    for (int k = 0; k < KEXP; ++k) {
      const uint64_t acc = uint64_t(perr[k][static_cast<size_t>(pN)]) +
                           uint64_t(perr[k][static_cast<size_t>(pNE)]) +
                           uint64_t(perr[k][static_cast<size_t>(pNW)]) +
                           (x > 0 ? uint64_t(perr[k][static_cast<size_t>(pW)]) : 0ull);
      wts[k] = wp_wt(acc >> qsh, mw[k]);
    }
    if (patch_conf == 0) wts[7] = 1;

    uint32_t sw = 0;
    for (int k = 0; k < KEXP; ++k) sw += wts[k];
    if (sw < 16) sw = 16;
    const int lw = flog2(sw);
    sw = 0;
    for (int k = 0; k < KEXP; ++k) {
      if (lw >= 4) wts[k] >>= (lw - 4);
      sw += wts[k];
    }
    if (sw == 0) sw = 1;
    else if (sw > 64) sw = 64;
    int64_t s = static_cast<int64_t>(sw >> 1) - 1;
    for (int k = 0; k < KEXP; ++k) s += pr_out[k] * static_cast<int64_t>(wts[k]);
    int64_t pred_q4 = static_cast<int64_t>((static_cast<__int128>(s) * kDiv[sw]) >> 24);
    if (((teN ^ teW) | (teN ^ teNW)) <= 0 && patch_conf < 10) {
      pred_q4 = std::clamp(pred_q4, std::min({wv, ne, n}), std::max({wv, ne, n}));
    }

    const int64_t reW = x > 0 ? true_err.at(y, x - 1) : 0;
    const int64_t reN = y > 0 ? true_err.at(y - 1, x) : 0;
    const int64_t reNW = (y > 0 && x > 0) ? true_err.at(y - 1, x - 1) : 0;
    const int64_t reNE = (y > 0 && x + 1 < W) ? true_err.at(y - 1, x + 1) : 0;
    const int64_t reWW = x > 1 ? true_err.at(y, x - 2) : 0;
    const int64_t reNN = y > 1 ? true_err.at(y - 2, x) : 0;

    int k = 0;
    f[k++] = wv - pred_q4;
    f[k++] = n - pred_q4;
    f[k++] = nw - pred_q4;
    f[k++] = ne - pred_q4;
    f[k++] = (WW << 4) - pred_q4;
    f[k++] = nn - pred_q4;
    f[k++] = (NWW << 4) - pred_q4;
    f[k++] = (NNW << 4) - pred_q4;
    f[k++] = (NNE << 4) - pred_q4;
    f[k++] = (NEE << 4) - pred_q4;
    f[k++] = (med << 4) - pred_q4;
    f[k++] = (gap << 4) - pred_q4;
    f[k++] = (planar << 4) - pred_q4;
    f[k++] = pr_out[0] - pred_q4;
    f[k++] = patch_conf > 0 ? (patch_q4 - pred_q4) : 0;
    f[k++] = reW;
    f[k++] = reN;
    f[k++] = reNW;
    f[k++] = reNE;
    f[k++] = reWW;
    f[k++] = reNN;

    int64_t bilat_diff = 0;
    if (C0 == nullptr) {
      const int64_t d45 = iabs64(Wv - Nv) + iabs64(NW - NN);
      const int64_t d135 = iabs64(Nv - NE) + iabs64(Wv - NW);
      const int dir = (dh_g * 2 < dv_g) ? 0 : ((dv_g * 2 < dh_g) ? 1 : ((d45 < d135) ? 2 : 3));
      const int64_t gsum = (dh_g + dv_g) >> qsh;
      const int con = (gsum < 16) ? 0 : ((gsum < 48) ? 1 : ((gsum < 128) ? 2 : 3));
      sec = con * 4 + dir;
    } else {
      const int64_t g0 = C0->at(y, x);
      const int64_t nbp[6] = {Wv, Nv, NW, NE, WW, NN};
      const int64_t nbg[6] = {at_clamped(*C0, y, x - 1),     at_clamped(*C0, y - 1, x),
                              at_clamped(*C0, y - 1, x - 1), at_clamped(*C0, y - 1, x + 1),
                              at_clamped(*C0, y, x - 2),     at_clamped(*C0, y - 2, x)};
      const int nbd[6] = {1, 1, 2, 2, 3, 3};
      int64_t bn = 0;
      int64_t bd = 0;
      for (int m = 0; m < 6; ++m) {
        int64_t dg = iabs64(g0 - nbg[m]) >> qsh;
        if (C1 != nullptr) {
          const int64_t u0 = C1->at(y, x);
          const int64_t um = m == 0   ? at_clamped(*C1, y, x - 1)
                             : m == 1 ? at_clamped(*C1, y - 1, x)
                             : m == 2 ? at_clamped(*C1, y - 1, x - 1)
                             : m == 3 ? at_clamped(*C1, y - 1, x + 1)
                             : m == 4 ? at_clamped(*C1, y, x - 2)
                                      : at_clamped(*C1, y - 2, x);
          dg += iabs64(u0 - um) >> qsh;
        }
        int64_t wt = 8192 / ((dg * dg + 8) * nbd[m]);
        if (wt < 1) wt = 1;
        bn += wt * (nbp[m] << 4);
        bd += wt;
      }
      bilat_diff = (bn / bd) - pred_q4;
      f[k++] = bilat_diff;
      const int64_t gN = at_clamped(*C0, y - 1, x);
      const int64_t gS = at_clamped(*C0, y + 1, x);
      const int64_t gW = at_clamped(*C0, y, x - 1);
      const int64_t gE = at_clamped(*C0, y, x + 1);
      const int64_t gNW = at_clamped(*C0, y - 1, x - 1);
      const int64_t gNE = at_clamped(*C0, y - 1, x + 1);
      const int64_t gSW = at_clamped(*C0, y + 1, x - 1);
      const int64_t gSE = at_clamped(*C0, y + 1, x + 1);
      f[k++] = (g0 - gW) << 4;
      f[k++] = (g0 - gN) << 4;
      f[k++] = (2 * g0 - gW - gE) << 3;
      f[k++] = (2 * g0 - gN - gS) << 3;
      f[k++] = (4 * g0 - gNW - gNE - gSW - gSE) << 2;
      f[k++] = (gN - gS) << 3;
      f[k++] = (gW - gE) << 3;
      f[k++] = C0_err->at(y, x);
      if (C1 != nullptr) {
        const int64_t u0 = C1->at(y, x);
        const int64_t uN = at_clamped(*C1, y - 1, x);
        const int64_t uS = at_clamped(*C1, y + 1, x);
        const int64_t uW = at_clamped(*C1, y, x - 1);
        const int64_t uE = at_clamped(*C1, y, x + 1);
        f[k++] = (u0 - uW) << 4;
        f[k++] = (u0 - uN) << 4;
        f[k++] = (2 * u0 - uW - uE) << 3;
        f[k++] = (2 * u0 - uN - uS) << 3;
        f[k++] = (uN - uS) << 3;
        f[k++] = C1_err->at(y, x);
      }
      const int phase = ((y & 1) << 1) | (x & 1);
      const int64_t gdh = iabs64(gW - gE) >> qsh;
      const int64_t gdv = iabs64(gN - gS) >> qsh;
      const int ori = (gdh + gdv < 16) ? 0 : ((gdh > gdv * 2) ? 1 : ((gdv > gdh * 2) ? 2 : 3));
      sec = phase * 4 + ori;
    }

    if (prp != nullptr) {
      const uint64_t ew = static_cast<uint64_t>(iabs64(reW));
      const uint64_t en = static_cast<uint64_t>(iabs64(reN));
      const uint64_t enw = static_cast<uint64_t>(iabs64(reNW));
      const uint64_t ene = static_cast<uint64_t>(iabs64(reNE));
      const uint64_t eww = static_cast<uint64_t>(iabs64(reWW));
      const uint64_t enn = static_cast<uint64_t>(iabs64(reNN));
      prp[0] = static_cast<int16_t>(flog2(2 * (ew + en) + enw + ene + ((eww + enn) >> 1) + 1));
      prp[1] = static_cast<int16_t>(flog2(ew + 1));
      prp[2] = static_cast<int16_t>(flog2(en + 1));
      prp[3] = static_cast<int16_t>(flog2(enw + ene + 1));
      prp[4] = static_cast<int16_t>(flog2(static_cast<uint64_t>(iabs64(f[0])) + 1));
      prp[5] = static_cast<int16_t>(flog2(static_cast<uint64_t>(iabs64(f[1])) + 1));
      prp[6] = static_cast<int16_t>(flog2(static_cast<uint64_t>(iabs64(f[0] - f[1])) + 1));
      prp[7] = static_cast<int16_t>(flog2(static_cast<uint64_t>(iabs64(f[10])) + 1));
      prp[8] = static_cast<int16_t>(sec);
      prp[9] = static_cast<int16_t>((reW >= 0 ? 1 : -1) * flog2(ew + 1));
      prp[10] = static_cast<int16_t>((reN >= 0 ? 1 : -1) * flog2(en + 1));
      prp[11] = static_cast<int16_t>(patch_conf);
      prp[12] = static_cast<int16_t>(
          C0 != nullptr ? flog2(static_cast<uint64_t>(iabs64(C0_err->at(y, x))) + 1) : (((y & 1) << 1) | (x & 1)));
      prp[13] = static_cast<int16_t>(
          C0 != nullptr ? flog2(static_cast<uint64_t>(iabs64(bilat_diff)) + 1)
                        : flog2(static_cast<uint64_t>(dh_g + dv_g) + 1));
    }
    return pred_q4;
  }

  void commit(const Plane& P, int y, int x, int32_t truth, int64_t pred_q4, int64_t mu_q4,
              const int64_t pr_out[KEXP]) {
    const int cur = (y & 1) ? 0 : stride;
    const int prv = (y & 1) ? stride : 0;
    const int64_t val_q4 = static_cast<int64_t>(truth) << 4;
    true_err.at(y, x) = static_cast<int32_t>(std::clamp<int64_t>(val_q4 - mu_q4, -8388608, 8388608));
    wp_err[static_cast<size_t>(cur + x)] = pred_q4 - val_q4;
    for (int m = 0; m < KEXP; ++m) {
      const uint64_t e64 = static_cast<uint64_t>((iabs64(pr_out[m] - val_q4) + 8) >> 4);
      const uint32_t e = static_cast<uint32_t>(std::min<uint64_t>(e64, 1u << 28));
      perr[m][static_cast<size_t>(cur + x)] = e;
      perr[m][static_cast<size_t>(prv + x + 1)] += e;
    }
    patcher.insert(P, y, x);
  }
};

std::vector<uint8_t> encode_channel_raster_exact(const Plane& P, int c_idx, int rct_mode, int maxv, int effort,
                                                 const Plane* C0, const Plane* C0_err, const Plane* C1,
                                                 const Plane* C1_err, Plane& out_err_q4) {
  const int H = P.height;
  const int W = P.width;
  const int N = H * W;
  out_err_q4 = Plane(H, W, 0);
  const bool hc0 = C0 != nullptr;
  const bool hc1 = C1 != nullptr;
  const int D = !hc0 ? 21 : (!hc1 ? 30 : 36);
  const int n_sec = 16;

  std::vector<int64_t> all_f(static_cast<size_t>(N) * static_cast<size_t>(D));
  std::vector<int64_t> base_v(static_cast<size_t>(N));
  std::vector<int> sec_v(static_cast<size_t>(N));

  auto collect_pass = [&](const std::vector<std::vector<int64_t>>* w_q8, std::vector<int32_t>* out_r,
                          std::vector<int32_t>* out_fr, std::vector<int16_t>* out_prp) {
    RasterState st(H, W, c_idx, rct_mode, maxv, hc0, hc1);
    int64_t pr_out[RasterState::KEXP];
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const int id = y * W + x;
        int16_t* prp = out_prp != nullptr ? &(*out_prp)[static_cast<size_t>(id) * 14] : nullptr;
        const int64_t pred_q4 = st.predict_features(
            P, C0, C0_err, C1, C1_err, y, x, &all_f[static_cast<size_t>(id) * static_cast<size_t>(D)],
            sec_v[static_cast<size_t>(id)], prp, pr_out);
        base_v[static_cast<size_t>(id)] = pred_q4;
        int64_t mu_q4 = pred_q4;
        if (w_q8 != nullptr) {
          __int128 acc = 128;
          const auto& ws = (*w_q8)[static_cast<size_t>(sec_v[static_cast<size_t>(id)])];
          for (int a = 0; a < D; ++a) {
            acc += static_cast<__int128>(all_f[static_cast<size_t>(id) * static_cast<size_t>(D) + static_cast<size_t>(a)]) *
                   ws[static_cast<size_t>(a)];
          }
          mu_q4 += static_cast<int64_t>(acc >> 8);
        }
        const auto [blo, bhi] = pixel_bounds_q4(rct_mode, c_idx, maxv, C0, C1, y, x);
        mu_q4 = std::clamp(mu_q4, blo, bhi);
        const int32_t truth = P.at(y, x);
        if (out_r != nullptr) {
          const int64_t mu_i = (mu_q4 + 8) >> 4;
          const int fr = std::clamp(static_cast<int>(mu_q4 - (mu_i << 4) + 8), 0, 15);
          (*out_r)[static_cast<size_t>(id)] = truth - static_cast<int32_t>(mu_i);
          (*out_fr)[static_cast<size_t>(id)] = fr;
          out_err_q4.at(y, x) = static_cast<int32_t>((static_cast<int64_t>(truth) << 4) - mu_q4);
        }
        st.commit(P, y, x, truth, pred_q4, mu_q4, pr_out);
      }
    }
  };

  BitWriter hw;
  std::vector<std::vector<int64_t>> w_final;
  if (effort >= 2 && N >= 256) {
    BitWriter dummy_hw;
    std::vector<std::vector<int64_t>> w1;
    collect_pass(nullptr, nullptr, nullptr, nullptr);
    fit_and_write_sector_ridge(dummy_hw, n_sec, D, all_f, base_v, sec_v, P.samples.data(), nullptr, W, w1);
    collect_pass(&w1, nullptr, nullptr, nullptr);
    fit_and_write_sector_ridge(hw, n_sec, D, all_f, base_v, sec_v, P.samples.data(), nullptr, W, w_final);
  } else {
    collect_pass(nullptr, nullptr, nullptr, nullptr);
    fit_and_write_sector_ridge(hw, n_sec, D, all_f, base_v, sec_v, P.samples.data(), nullptr, W, w_final);
  }
  const auto w_bytes = hw.finish();
  if (w_bytes.size() > 65535) throw CodecError("frontier ridge header exceeds uint16");

  std::vector<int32_t> resid(static_cast<size_t>(N));
  std::vector<int32_t> frac(static_cast<size_t>(N));
  std::vector<int16_t> props(static_cast<size_t>(N) * 14);
  collect_pass(&w_final, &resid, &frac, &props);

  const int max_leaves = effort <= 1 ? (N >= 100000 ? 48 : 24) : (N >= 100000 ? 80 : 32);
  const auto ma_bytes = encode_ma_stream(resid, frac, props, 14, max_leaves);

  std::vector<uint8_t> out;
  out.reserve(2 + w_bytes.size() + ma_bytes.size());
  const uint16_t wn = static_cast<uint16_t>(w_bytes.size());
  out.push_back(static_cast<uint8_t>(wn));
  out.push_back(static_cast<uint8_t>(wn >> 8));
  out.insert(out.end(), w_bytes.begin(), w_bytes.end());
  out.insert(out.end(), ma_bytes.begin(), ma_bytes.end());
  return out;
}

Plane decode_channel_raster_exact(const uint8_t* data, size_t size, size_t& pos, int H, int W, int c_idx,
                                  int rct_mode, int maxv, const Plane* C0, const Plane* C0_err,
                                  const Plane* C1, const Plane* C1_err, Plane& out_err_q4) {
  if (pos + 2 > size) throw CodecError("truncated frontier raster ridge header");
  const uint16_t wn = static_cast<uint16_t>(data[pos] | (uint16_t(data[pos + 1]) << 8));
  pos += 2;
  if (wn == 0 || pos + wn > size) throw CodecError("truncated frontier raster ridge weights");
  const bool hc0 = C0 != nullptr;
  const bool hc1 = C1 != nullptr;
  const int D = !hc0 ? 21 : (!hc1 ? 30 : 36);
  const int n_sec = 16;
  BitReader wr(data + pos, wn);
  pos += wn;
  const auto w_final = read_sector_ridge(wr, n_sec, D);

  OpenedMAStream mst = open_ma_stream(data, size, pos, 14);
  AdaptiveStreamDecoder dec(mst.payload, mst.nu, mst.use_adapt, mst.tree.leaf_sigmas);

  Plane P(H, W, 0);
  out_err_q4 = Plane(H, W, 0);
  RasterState st(H, W, c_idx, rct_mode, maxv, hc0, hc1);
  int64_t f[40];
  int16_t prp[14];
  int64_t pr_out[RasterState::KEXP];
  int sec = 0;

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const int64_t pred_q4 = st.predict_features(P, C0, C0_err, C1, C1_err, y, x, f, sec, prp, pr_out);
      __int128 acc = 128;
      const auto& ws = w_final[static_cast<size_t>(sec)];
      for (int a = 0; a < D; ++a) acc += static_cast<__int128>(f[a]) * ws[static_cast<size_t>(a)];
      int64_t mu_q4 = pred_q4 + static_cast<int64_t>(acc >> 8);
      const auto [blo, bhi] = pixel_bounds_q4(rct_mode, c_idx, maxv, C0, C1, y, x);
      mu_q4 = std::clamp(mu_q4, blo, bhi);
      const int64_t mu_i = (mu_q4 + 8) >> 4;
      const int fr = std::clamp(static_cast<int>(mu_q4 - (mu_i << 4) + 8), 0, 15);
      const int leaf = mst.tree.walk(prp);
      const int32_t r = dec.read(leaf, fr);
      const int64_t truth64 = mu_i + static_cast<int64_t>(r);
      if (truth64 < -262144 || truth64 > 262144) throw CodecError("decoded frontier sample out of range");
      const int32_t truth = static_cast<int32_t>(truth64);
      P.at(y, x) = truth;
      out_err_q4.at(y, x) = static_cast<int32_t>((truth64 << 4) - mu_q4);
      st.commit(P, y, x, truth, pred_q4, mu_q4, pr_out);
    }
  }
  return P;
}

// ============================================================================
// 2x2 Coset-Scheduled Chroma Interpolation (Strictly Causal Parity-Safe)
// ============================================================================
const std::vector<std::pair<int, int>>& stage_nbs(int stg) {
  static const std::vector<std::pair<int, int>> s1 = {
      {-1, -1}, {-1, 1}, {1, -1}, {1, 1}, {-3, -1}, {-3, 1}, {3, -1}, {3, 1}, {-1, -3}, {1, -3}, {-1, 3}, {1, 3}};
  static const std::vector<std::pair<int, int>> s2 = {
      {0, -1}, {0, 1}, {-1, 0}, {1, 0}, {0, -3}, {0, 3}, {-2, -1}, {-2, 1}, {2, -1}, {2, 1}};
  static const std::vector<std::pair<int, int>> s3 = {
      {-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}, {-3, 0}, {3, 0}};
  return stg == 1 ? s1 : (stg == 2 ? s2 : s3);
}

int64_t interp_features_exact(const Plane& P, const Plane& C0, const Plane& C0_err, const Plane* C1,
                              const Plane* C1_err, const Plane& stage_err, int maxv, int stg, int y, int x,
                              int64_t* f, int& sec, int16_t* prp) {
  const int H = P.height;
  const int W = P.width;
  const int qsh = maxv > 255 ? 8 : 0;
  const auto& nbs = stage_nbs(stg);
  const int Knb = static_cast<int>(nbs.size());
  auto pat = [&](int yy, int xx) -> int64_t {
    return P.at(clamp_same_parity(yy, H), clamp_same_parity(xx, W));
  };
  auto gat = [&](const Plane& pl, int yy, int xx) -> int64_t {
    return pl.at(std::clamp(yy, 0, H - 1), std::clamp(xx, 0, W - 1));
  };

  int64_t sum4 = 0;
  for (int k = 0; k < 4; ++k) sum4 += pat(y + nbs[static_cast<size_t>(k)].first, x + nbs[static_cast<size_t>(k)].second);
  const int64_t b_q4 = sum4 << 2;

  const int64_t g0 = C0.at(y, x);
  int64_t bn = 0;
  int64_t bd = 0;
  for (int k = 0; k < std::min(Knb, 8); ++k) {
    const int ny = clamp_same_parity(y + nbs[static_cast<size_t>(k)].first, H);
    const int nx = clamp_same_parity(x + nbs[static_cast<size_t>(k)].second, W);
    int64_t dg = iabs64(g0 - C0.at(ny, nx)) >> qsh;
    if (C1 != nullptr) dg += iabs64(int64_t(C1->at(y, x)) - C1->at(ny, nx)) >> qsh;
    const int dist2 = nbs[static_cast<size_t>(k)].first * nbs[static_cast<size_t>(k)].first +
                      nbs[static_cast<size_t>(k)].second * nbs[static_cast<size_t>(k)].second;
    int64_t wt = 8192 / ((dg * dg + 8) * dist2);
    if (wt < 1) wt = 1;
    bn += wt * (static_cast<int64_t>(P.at(ny, nx)) << 4);
    bd += wt;
  }
  const int64_t bilat_diff = (bn / bd) - b_q4;

  int m = 0;
  for (int k = 0; k < Knb; ++k) {
    f[m++] = (pat(y + nbs[static_cast<size_t>(k)].first, x + nbs[static_cast<size_t>(k)].second) << 4) - b_q4;
  }
  f[m++] = bilat_diff;
  const int64_t gN = gat(C0, y - 1, x);
  const int64_t gS = gat(C0, y + 1, x);
  const int64_t gW = gat(C0, y, x - 1);
  const int64_t gE = gat(C0, y, x + 1);
  const int64_t gNW = gat(C0, y - 1, x - 1);
  const int64_t gNE = gat(C0, y - 1, x + 1);
  const int64_t gSW = gat(C0, y + 1, x - 1);
  const int64_t gSE = gat(C0, y + 1, x + 1);
  f[m++] = (2 * g0 - gW - gE) << 3;
  f[m++] = (2 * g0 - gN - gS) << 3;
  f[m++] = (4 * g0 - gNW - gNE - gSW - gSE) << 2;
  f[m++] = (g0 - ((gN + gS + gW + gE) >> 2)) << 4;
  f[m++] = (gN - gS) << 3;
  f[m++] = (gW - gE) << 3;
  f[m++] = (gNW - gSE) << 3;
  f[m++] = (gNE - gSW) << 3;
  f[m++] = C0_err.at(y, x);
  if (C1 != nullptr) {
    const int64_t u0 = C1->at(y, x);
    const int64_t uN = gat(*C1, y - 1, x);
    const int64_t uS = gat(*C1, y + 1, x);
    const int64_t uW = gat(*C1, y, x - 1);
    const int64_t uE = gat(*C1, y, x + 1);
    const int64_t uNW = gat(*C1, y - 1, x - 1);
    const int64_t uNE = gat(*C1, y - 1, x + 1);
    const int64_t uSW = gat(*C1, y + 1, x - 1);
    const int64_t uSE = gat(*C1, y + 1, x + 1);
    f[m++] = (2 * u0 - uW - uE) << 3;
    f[m++] = (2 * u0 - uN - uS) << 3;
    f[m++] = (4 * u0 - uNW - uNE - uSW - uSE) << 2;
    f[m++] = (u0 - ((uN + uS + uW + uE) >> 2)) << 4;
    f[m++] = (uN - uS) << 3;
    f[m++] = C1_err->at(y, x);
  }
  const int64_t gdh = (iabs64(gW - gE) + iabs64(2 * g0 - gW - gE)) >> qsh;
  const int64_t gdv = (iabs64(gN - gS) + iabs64(2 * g0 - gN - gS)) >> qsh;
  sec = (gdh + gdv < 12) ? 0 : ((gdh > gdv * 2) ? 1 : ((gdv > gdh * 2) ? 2 : 3));

  if (prp != nullptr) {
    uint64_t prev_e = 0;
    for (int k = 0; k < 4; ++k) {
      const int ny = clamp_same_parity(y + nbs[static_cast<size_t>(k)].first, H);
      const int nx = clamp_same_parity(x + nbs[static_cast<size_t>(k)].second, W);
      prev_e += static_cast<uint64_t>(iabs64(stage_err.at(ny, nx)));
    }
    const uint64_t causal_same = (x >= 2 ? static_cast<uint64_t>(iabs64(stage_err.at(y, x - 2))) : 0ull) +
                                 (y >= 2 ? static_cast<uint64_t>(iabs64(stage_err.at(y - 2, x))) : 0ull);
    uint64_t c_span = 0;
    for (int k = 0; k < 4; ++k) c_span += static_cast<uint64_t>(iabs64(f[k]));
    prp[0] = static_cast<int16_t>(flog2(causal_same + 1));
    prp[1] = static_cast<int16_t>(flog2(prev_e + 1));
    prp[2] = static_cast<int16_t>(flog2(c_span + 1));
    prp[3] = static_cast<int16_t>(flog2(static_cast<uint64_t>(iabs64(bilat_diff)) + 1));
    prp[4] = static_cast<int16_t>(flog2(static_cast<uint64_t>(iabs64(C0_err.at(y, x))) + 1));
    prp[5] = static_cast<int16_t>(flog2(static_cast<uint64_t>(iabs64(gW - gE) + iabs64(gN - gS)) + 1));
    prp[6] = static_cast<int16_t>(sec);
    prp[7] = static_cast<int16_t>((f[0] - f[1] >= 0 ? 1 : -1) *
                                  flog2(static_cast<uint64_t>(iabs64(f[0] - f[1])) + 1));
    prp[8] = static_cast<int16_t>(C1 != nullptr ? flog2(static_cast<uint64_t>(iabs64(C1_err->at(y, x))) + 1) : 0);
    prp[9] = static_cast<int16_t>(
        flog2(static_cast<uint64_t>(iabs64(2 * g0 - gW - gE) + iabs64(2 * g0 - gN - gS)) + 1));
    prp[10] = static_cast<int16_t>(flog2(static_cast<uint64_t>(iabs64(f[0]) + iabs64(f[1])) + 1));
  }
  return b_q4;
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

  const int max_leaves_coset = effort <= 1 ? 24 : 32;
  const int py_s[4] = {0, 1, 0, 1};
  const int px_s[4] = {0, 1, 1, 0};
  for (int stg = 1; stg <= 3; ++stg) {
    std::vector<std::pair<int, int>> coords;
    for (int y = py_s[stg]; y < H; y += 2) {
      for (int x = px_s[stg]; x < W; x += 2) coords.emplace_back(y, x);
    }
    const int Ns = static_cast<int>(coords.size());
    const int Knb = static_cast<int>(stage_nbs(stg).size());
    const int D = Knb + 10 + (C1 != nullptr ? 6 : 0);
    const int n_sec = 4;
    std::vector<int64_t> all_f(static_cast<size_t>(Ns) * static_cast<size_t>(D));
    std::vector<int64_t> base_v(static_cast<size_t>(Ns));
    std::vector<int> sec_v(static_cast<size_t>(Ns));
    for (int i = 0; i < Ns; ++i) {
      base_v[static_cast<size_t>(i)] = interp_features_exact(
          P, C0, C0_err, C1, C1_err, stage_err, maxv, stg, coords[static_cast<size_t>(i)].first,
          coords[static_cast<size_t>(i)].second, &all_f[static_cast<size_t>(i) * static_cast<size_t>(D)],
          sec_v[static_cast<size_t>(i)], nullptr);
    }
    BitWriter hw;
    std::vector<std::vector<int64_t>> w_q8;
    fit_and_write_sector_ridge(hw, n_sec, D, all_f, base_v, sec_v, P.samples.data(), coords.data(), W, w_q8);
    const auto wb = hw.finish();
    if (wb.size() > 65535) throw CodecError("frontier coset ridge header exceeds uint16");
    const uint16_t wn = static_cast<uint16_t>(wb.size());
    out.push_back(static_cast<uint8_t>(wn));
    out.push_back(static_cast<uint8_t>(wn >> 8));
    out.insert(out.end(), wb.begin(), wb.end());

    std::vector<int32_t> resid(static_cast<size_t>(Ns));
    std::vector<int32_t> frac(static_cast<size_t>(Ns));
    std::vector<int16_t> props(static_cast<size_t>(Ns) * 11);
    int64_t f_tmp[32];
    int sec_tmp = 0;
    for (int i = 0; i < Ns; ++i) {
      const auto [y, x] = coords[static_cast<size_t>(i)];
      const int64_t b_q4 = interp_features_exact(P, C0, C0_err, C1, C1_err, stage_err, maxv, stg, y, x, f_tmp,
                                                 sec_tmp, &props[static_cast<size_t>(i) * 11]);
      __int128 acc = 128;
      for (int a = 0; a < D; ++a) acc += static_cast<__int128>(f_tmp[a]) * w_q8[static_cast<size_t>(sec_tmp)][static_cast<size_t>(a)];
      int64_t mu_q4 = b_q4 + static_cast<int64_t>(acc >> 8);
      const auto [blo, bhi] = pixel_bounds_q4(rct_mode, c_idx, maxv, &C0, C1, y, x);
      mu_q4 = std::clamp(mu_q4, blo, bhi);
      const int64_t mu_i = (mu_q4 + 8) >> 4;
      const int fr = std::clamp(static_cast<int>(mu_q4 - (mu_i << 4) + 8), 0, 15);
      const int32_t truth = P.at(y, x);
      resid[static_cast<size_t>(i)] = truth - static_cast<int32_t>(mu_i);
      frac[static_cast<size_t>(i)] = fr;
      const int32_t eq4 = static_cast<int32_t>((static_cast<int64_t>(truth) << 4) - mu_q4);
      out_err_q4.at(y, x) = eq4;
      stage_err.at(y, x) = static_cast<int32_t>(iabs64(eq4));
    }
    const auto mb = encode_ma_stream(resid, frac, props, 11, max_leaves_coset);
    out.insert(out.end(), mb.begin(), mb.end());
  }
  return out;
}

Plane decode_chroma_coset_exact(const uint8_t* data, size_t size, size_t& pos, int H, int W, int c_idx,
                                int rct_mode, int maxv, const Plane& C0, const Plane& C0_err, const Plane* C1,
                                const Plane* C1_err, Plane& out_err_q4) {
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

  const int py_s[4] = {0, 1, 0, 1};
  const int px_s[4] = {0, 1, 1, 0};
  for (int stg = 1; stg <= 3; ++stg) {
    const int Knb = static_cast<int>(stage_nbs(stg).size());
    const int D = Knb + 10 + (C1 != nullptr ? 6 : 0);
    const int n_sec = 4;
    if (pos + 2 > size) throw CodecError("truncated frontier coset ridge header");
    const uint16_t wn = static_cast<uint16_t>(data[pos] | (uint16_t(data[pos + 1]) << 8));
    pos += 2;
    if (wn == 0 || pos + wn > size) throw CodecError("truncated frontier coset ridge weights");
    BitReader wr(data + pos, wn);
    pos += wn;
    const auto w_q8 = read_sector_ridge(wr, n_sec, D);

    OpenedMAStream mst = open_ma_stream(data, size, pos, 11);
    AdaptiveStreamDecoder dec(mst.payload, mst.nu, mst.use_adapt, mst.tree.leaf_sigmas);

    int64_t f_tmp[32];
    int16_t prp[11];
    int sec_tmp = 0;
    for (int y = py_s[stg]; y < H; y += 2) {
      for (int x = px_s[stg]; x < W; x += 2) {
        const int64_t b_q4 =
            interp_features_exact(P, C0, C0_err, C1, C1_err, stage_err, maxv, stg, y, x, f_tmp, sec_tmp, prp);
        __int128 acc = 128;
        for (int a = 0; a < D; ++a) {
          acc += static_cast<__int128>(f_tmp[a]) * w_q8[static_cast<size_t>(sec_tmp)][static_cast<size_t>(a)];
        }
        int64_t mu_q4 = b_q4 + static_cast<int64_t>(acc >> 8);
        const auto [blo, bhi] = pixel_bounds_q4(rct_mode, c_idx, maxv, &C0, C1, y, x);
        mu_q4 = std::clamp(mu_q4, blo, bhi);
        const int64_t mu_i = (mu_q4 + 8) >> 4;
        const int fr = std::clamp(static_cast<int>(mu_q4 - (mu_i << 4) + 8), 0, 15);
        const int leaf = mst.tree.walk(prp);
        const int32_t r = dec.read(leaf, fr);
        const int64_t truth64 = mu_i + static_cast<int64_t>(r);
        if (truth64 < -262144 || truth64 > 262144) throw CodecError("decoded frontier coset sample out of range");
        const int32_t truth = static_cast<int32_t>(truth64);
        P.at(y, x) = truth;
        const int32_t eq4 = static_cast<int32_t>((truth64 << 4) - mu_q4);
        out_err_q4.at(y, x) = eq4;
        stage_err.at(y, x) = static_cast<int32_t>(iabs64(eq4));
      }
    }
  }
  return P;
}

int64_t plane_l1_grad(const Plane& p) {
  int64_t g = 0;
  for (int y = 0; y < p.height; ++y) {
    for (int x = 0; x < p.width; ++x) {
      if (x > 0) g += iabs64(int64_t(p.at(y, x)) - p.at(y, x - 1));
      if (y > 0) g += iabs64(int64_t(p.at(y, x)) - p.at(y - 1, x));
    }
  }
  return g;
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
    Plane er;
    Plane ec;
    auto br = encode_channel_raster_exact(P, c_idx, rm, maxv, effort, &c0, &c0e, c1, c1e, er);
    auto bc = encode_chroma_coset_exact(P, c_idx, rm, maxv, effort, c0, c0e, c1, c1e, ec);
    if (bc.size() < br.size()) {
      out_e = std::move(ec);
      flag_bit = 1;
      return bc;
    }
    out_e = std::move(er);
    flag_bit = 0;
    return br;
  };

  // Shared-G search: encode Ch0 = G and Ch1 = R - G once, then compare Ch2 under
  // SubtractGreen vs SubtractAverage (perm=0).
  Plane eG;
  auto bG = encode_channel_raster_exact(raw_planes[1], 0, MODE_SUB, maxv, effort, nullptr, nullptr, nullptr,
                                        nullptr, eG);

  const auto sub_planes = color_fwd_frontier(raw_planes, MODE_SUB, 0);
  const auto avg0_planes = color_fwd_frontier(raw_planes, kModeSubAvg, 0);

  Plane eRG;
  uint8_t f1 = 0;
  uint8_t f2_sub = 0;
  uint8_t f2_avg = 0;
  auto bRG = best_chroma(sub_planes[1], 1, MODE_SUB, sub_planes[0], eG, nullptr, nullptr, eRG, f1);

  Plane dummy_e2a;
  Plane dummy_e2b;
  auto bBG_sub = best_chroma(sub_planes[2], 2, MODE_SUB, sub_planes[0], eG, &sub_planes[1], &eRG, dummy_e2a, f2_sub);
  auto bBG_avg =
      best_chroma(avg0_planes[2], 2, kModeSubAvg, avg0_planes[0], eG, &avg0_planes[1], &eRG, dummy_e2b, f2_avg);

  int chosen_mode = MODE_SUB;
  int chosen_perm = 0;
  uint8_t cflags = f1;
  Plane best_e0 = eG;
  std::vector<uint8_t> best_c0 = bG;
  std::vector<uint8_t> best_c1 = std::move(bRG);
  std::vector<uint8_t> best_c2 = std::move(bBG_sub);
  if (bBG_avg.size() < best_c2.size()) {
    chosen_mode = kModeSubAvg;
    chosen_perm = 0;
    cflags = static_cast<uint8_t>(f1 | (f2_avg << 1));
    best_c2 = std::move(bBG_avg);
  } else {
    cflags = static_cast<uint8_t>(f1 | (f2_sub << 1));
  }

  // At effort >= 2, also test perm=1 (G, B-G, R - ((G+B)>>1)), reusing bG/eG for free.
  if (effort >= 2) {
    const auto avg1 = color_fwd_frontier(raw_planes, kModeSubAvg, 1);
    Plane eBG1;
    Plane eRG1;
    uint8_t g1 = 0;
    uint8_t g2 = 0;
    auto b1 = best_chroma(avg1[1], 1, kModeSubAvg, avg1[0], eG, nullptr, nullptr, eBG1, g1);
    auto b2 = best_chroma(avg1[2], 2, kModeSubAvg, avg1[0], eG, &avg1[1], &eBG1, eRG1, g2);
    if (bG.size() + b1.size() + b2.size() < best_c0.size() + best_c1.size() + best_c2.size()) {
      chosen_mode = kModeSubAvg;
      chosen_perm = 1;
      cflags = static_cast<uint8_t>(g1 | (g2 << 1));
      best_c0 = bG;
      best_c1 = std::move(b1);
      best_c2 = std::move(b2);
      best_e0 = eG;
    }
  }

  // At effort >= 3, also test the smoother non-G base channel (R-base perm=2 or B-base perm=5)
  // plus Identity (for uncorrelated synthetic RGB tiles).
  if (effort >= 3) {
    const int alt_perm = plane_l1_grad(raw_planes[0]) <= plane_l1_grad(raw_planes[2]) ? 2 : 5;
    const auto alt_pl = color_fwd_frontier(raw_planes, kModeSubAvg, alt_perm);
    Plane eA0;
    Plane eA1;
    Plane eA2;
    uint8_t h1 = 0;
    uint8_t h2 = 0;
    auto bA0 = encode_channel_raster_exact(alt_pl[0], 0, kModeSubAvg, maxv, effort, nullptr, nullptr, nullptr,
                                           nullptr, eA0);
    auto bA1 = best_chroma(alt_pl[1], 1, kModeSubAvg, alt_pl[0], eA0, nullptr, nullptr, eA1, h1);
    auto bA2 = best_chroma(alt_pl[2], 2, kModeSubAvg, alt_pl[0], eA0, &alt_pl[1], &eA1, eA2, h2);
    if (bA0.size() + bA1.size() + bA2.size() < best_c0.size() + best_c1.size() + best_c2.size()) {
      chosen_mode = kModeSubAvg;
      chosen_perm = alt_perm;
      cflags = static_cast<uint8_t>(h1 | (h2 << 1));
      best_c0 = std::move(bA0);
      best_c1 = std::move(bA1);
      best_c2 = std::move(bA2);
      best_e0 = std::move(eA0);
    }
  }

  std::vector<uint8_t> out{static_cast<uint8_t>(chosen_mode), static_cast<uint8_t>(chosen_perm), cflags};
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
