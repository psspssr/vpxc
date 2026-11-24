#include "vpxc/fast_codec.hpp"

#include "vpxc/bitio.hpp"
#include "vpxc/color.hpp"
#include "vpxc/error.hpp"
#include "vpxc/simd.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace vpxc {
namespace {

constexpr int kFastScale = 12;
constexpr int kFastM = 1 << kFastScale;          // 4096
constexpr uint32_t kFastL = 1u << 15;            // 32768; state stays in [32768, 2^31 - 1]
constexpr int kFastToks = 46;                    // covers |r| <= 262,143 (u < 524,288)
constexpr int kFastCtxs = 8;
constexpr int kRctId = 0;
constexpr int kRctSubG = 1;
constexpr int kRctSubAvg = 2;

struct FastTables {
  uint32_t tok_base[kFastToks];
  uint8_t tok_ebits[kFastToks];
  uint8_t ctx_lut[96];
  uint64_t recip_mul[kFastM + 1];
  uint8_t recip_sh[kFastM + 1];
};

constexpr FastTables make_fast_tables() {
  FastTables t{};
  for (int tok = 0; tok < 16; ++tok) {
    t.tok_base[tok] = static_cast<uint32_t>(tok);
    t.tok_ebits[tok] = 0;
  }
  for (int tok = 16; tok < kFastToks; ++tok) {
    const int b = 4 + ((tok - 16) >> 1);
    const int nb = b - 1;
    const uint32_t msb = static_cast<uint32_t>((tok - 16) & 1);
    t.tok_base[tok] = (2u + msb) << nb;
    t.tok_ebits[tok] = static_cast<uint8_t>(nb);
  }
  for (int s = 0; s < 96; ++s) {
    if (s == 0) t.ctx_lut[s] = 0;
    else if (s <= 2) t.ctx_lut[s] = 1;
    else if (s <= 5) t.ctx_lut[s] = 2;
    else if (s <= 10) t.ctx_lut[s] = 3;
    else if (s <= 18) t.ctx_lut[s] = 4;
    else if (s <= 28) t.ctx_lut[s] = 5;
    else if (s <= 38) t.ctx_lut[s] = 6;
    else t.ctx_lut[s] = 7;
  }
  t.recip_mul[0] = 0;
  t.recip_sh[0] = 0;
  t.recip_mul[1] = 1;
  t.recip_sh[1] = 0;
  for (uint32_t f = 2; f <= static_cast<uint32_t>(kFastM); ++f) {
    int k = 0;
    while ((1u << (k + 1)) <= f) ++k;
    if ((1u << k) == f) {
      t.recip_mul[f] = 1ULL;
      t.recip_sh[f] = static_cast<uint8_t>(k);
    } else {
      const int sh = k + 32;
      const uint64_t num = (1ULL << sh) + static_cast<uint64_t>(f - 1);
      t.recip_mul[f] = num / static_cast<uint64_t>(f);
      t.recip_sh[f] = static_cast<uint8_t>(sh);
    }
  }
  return t;
}

constexpr FastTables kFT = make_fast_tables();

inline uint32_t zigzag_enc(int32_t r) noexcept {
  return (static_cast<uint32_t>(r) << 1) ^ static_cast<uint32_t>(r >> 31);
}

inline int32_t zigzag_dec(uint32_t u) noexcept {
  return static_cast<int32_t>((u >> 1) ^ (~(u & 1u) + 1u));
}

inline void tokenize_u32(uint32_t u, uint8_t& tok, uint32_t& extra, uint8_t& nbits) noexcept {
  if (u < 16u) {
    tok = static_cast<uint8_t>(u);
    extra = 0;
    nbits = 0;
    return;
  }
  const int b = 31 - __builtin_clz(u);
  const int nb = b - 1;
  const uint32_t msb = (u >> nb) & 1u;
  tok = static_cast<uint8_t>(16 + ((b - 4) << 1) + static_cast<int>(msb));
  extra = u & ((1u << nb) - 1u);
  nbits = static_cast<uint8_t>(nb);
}

inline int64_t iabs64(int64_t v) noexcept { return v >= 0 ? v : -v; }

inline int32_t med3_i32(int32_t w, int32_t n, int32_t nw) noexcept {
  const int32_t hi = std::max(w, n);
  const int32_t lo = std::min(w, n);
  if (nw >= hi) return lo;
  if (nw <= lo) return hi;
  return static_cast<int32_t>(static_cast<int64_t>(w) + n - nw);
}

// Causal 3D anatomy-switched Delta-MED predictor for volumetric CT/MRI slices (100% integer, zero float).
inline int32_t pred_3d_switched(int32_t w, int32_t n, int32_t nw, int32_t z_cur, int32_t z_w, int32_t z_n,
                                int32_t z_nw, int32_t pmin, int32_t pmax) noexcept {
  const int64_t pred_2d = med3_i32(w, n, nw);
  const int32_t dw = static_cast<int32_t>(static_cast<int64_t>(w) - z_w);
  const int32_t dn = static_cast<int32_t>(static_cast<int64_t>(n) - z_n);
  const int32_t dnw = static_cast<int32_t>(static_cast<int64_t>(nw) - z_nw);
  const int64_t pred_3d = static_cast<int64_t>(z_cur) + med3_i32(dw, dn, dnw);

  const int64_t act_3d = iabs64(static_cast<int64_t>(dw) - dnw) + iabs64(static_cast<int64_t>(dn) - dnw);
  const int64_t act_2d = iabs64(static_cast<int64_t>(w) - nw) + iabs64(static_cast<int64_t>(n) - nw);

  int64_t p = 0;
  if (act_3d * 4 <= act_2d * 3 + 2) {
    p = pred_3d;
  } else if (act_2d * 2 <= act_3d) {
    p = pred_2d;
  } else {
    p = (pred_3d + pred_2d + 1) >> 1;
  }
  return static_cast<int32_t>(std::clamp<int64_t>(p, pmin, pmax));
}

struct FastBitWriter {
  uint64_t buf = 0;
  int bits = 0;
  std::vector<uint8_t> out;

  inline void put(uint32_t val, int n) {
    if (n == 0) return;
    buf |= (static_cast<uint64_t>(val) << bits);
    bits += n;
    if (bits >= 32) {
      const uint32_t w = static_cast<uint32_t>(buf & 0xFFFFFFFFu);
      const size_t o = out.size();
      out.resize(o + 4);
      out[o + 0] = static_cast<uint8_t>(w);
      out[o + 1] = static_cast<uint8_t>(w >> 8);
      out[o + 2] = static_cast<uint8_t>(w >> 16);
      out[o + 3] = static_cast<uint8_t>(w >> 24);
      buf >>= 32;
      bits -= 32;
    }
  }

  void finish() {
    while (bits > 0) {
      out.push_back(static_cast<uint8_t>(buf & 0xFFu));
      buf >>= 8;
      bits -= 8;
    }
    bits = 0;
    buf = 0;
  }
};

struct FastBitReader {
  const uint8_t* ptr = nullptr;
  const uint8_t* end = nullptr;
  uint64_t buf = 0;
  int bits = 0;

  FastBitReader() = default;
  FastBitReader(const uint8_t* d, size_t n) : ptr(d), end(d + n) { refill(); }

  inline void refill() noexcept {
    if (bits <= 32 && ptr + 4 <= end) {
      const uint32_t w = static_cast<uint32_t>(ptr[0]) | (static_cast<uint32_t>(ptr[1]) << 8) |
                         (static_cast<uint32_t>(ptr[2]) << 16) | (static_cast<uint32_t>(ptr[3]) << 24);
      buf |= (static_cast<uint64_t>(w) << bits);
      ptr += 4;
      bits += 32;
    }
    while (bits <= 56 && ptr < end) {
      buf |= (static_cast<uint64_t>(*ptr++) << bits);
      bits += 8;
    }
  }

  inline uint32_t get(int n) {
    if (n == 0) return 0;
    if (bits < n) {
      refill();
      if (bits < n) throw CodecError("truncated fast extra-bits stream");
    }
    const uint32_t v = static_cast<uint32_t>(buf & ((1ULL << n) - 1ULL));
    buf >>= n;
    bits -= n;
    return v;
  }

  void check_end() const {
    if (ptr != end || bits >= 8 || buf != 0) throw CodecError("non-canonical fast extra-bits stream");
  }
};

struct Rans12Enc {
  uint32_t state = kFastL;
  std::vector<uint16_t> words;

  inline void put(uint32_t start, uint32_t freq) {
    const uint32_t x_max = freq << 19;  // ((kFastL >> 12) << 16) * freq
    if (state >= x_max) {
      words.push_back(static_cast<uint16_t>(state & 0xFFFFu));
      state >>= 16;
    }
    const uint32_t q = (freq == 1u)
                           ? state
                           : static_cast<uint32_t>((static_cast<uint64_t>(state) * kFT.recip_mul[freq]) >>
                                                   kFT.recip_sh[freq]);
    const uint32_t r = state - q * freq;
    state = (q << kFastScale) + r + start;
  }

  void append_to(std::vector<uint8_t>& dst) const {
    const size_t o = dst.size();
    dst.resize(o + 4 + words.size() * 2);
    dst[o + 0] = static_cast<uint8_t>(state);
    dst[o + 1] = static_cast<uint8_t>(state >> 8);
    dst[o + 2] = static_cast<uint8_t>(state >> 16);
    dst[o + 3] = static_cast<uint8_t>(state >> 24);
    size_t p = o + 4;
    for (auto it = words.rbegin(); it != words.rend(); ++it) {
      const uint16_t w = *it;
      dst[p + 0] = static_cast<uint8_t>(w);
      dst[p + 1] = static_cast<uint8_t>(w >> 8);
      p += 2;
    }
  }
};

struct Rans12Dec {
  const uint8_t* ptr = nullptr;
  const uint8_t* end = nullptr;
  uint32_t state = 0;

  Rans12Dec() = default;
  Rans12Dec(const uint8_t* d, size_t n) {
    if (n < 4 || (n & 1u) != 0) throw CodecError("invalid fast rANS stream size");
    state = static_cast<uint32_t>(d[0]) | (static_cast<uint32_t>(d[1]) << 8) |
            (static_cast<uint32_t>(d[2]) << 16) | (static_cast<uint32_t>(d[3]) << 24);
    if (state < kFastL || state >= (1u << 31)) throw CodecError("invalid fast rANS initial state");
    ptr = d + 4;
    end = d + n;
  }

  inline uint32_t step(const uint32_t* ctx_lut) {
    const uint32_t entry = ctx_lut[state & 0xFFFu];
    const uint32_t sym = entry & 0x7Fu;
    const uint32_t freq = (entry >> 7) & 0x1FFFu;
    const uint32_t bias = entry >> 20;
    uint32_t x = freq * (state >> kFastScale) + bias;
    if (x < kFastL) {
      if (ptr + 2 > end) throw CodecError("truncated fast rANS stream");
      const uint32_t w = static_cast<uint32_t>(ptr[0]) | (static_cast<uint32_t>(ptr[1]) << 8);
      ptr += 2;
      x = (x << 16) | w;
    }
    state = x;
    return sym;
  }

  void check_end() const {
    if (ptr != end || state != kFastL) throw CodecError("non-canonical fast rANS final state");
  }
};

void normalize_counts_4096(const uint32_t counts[kFastToks], uint16_t freqs[kFastToks], int& max_sym) {
  max_sym = -1;
  uint64_t total = 0;
  for (int i = 0; i < kFastToks; ++i) {
    if (counts[i] > 0) {
      max_sym = i;
      total += counts[i];
    }
    freqs[i] = 0;
  }
  if (max_sym < 0 || total == 0) {
    freqs[0] = kFastM;
    max_sym = -1;
    return;
  }
  int sum_f = 0;
  int best_sym = 0;
  uint32_t best_cnt = 0;
  for (int i = 0; i <= max_sym; ++i) {
    if (counts[i] == 0) {
      freqs[i] = 0;
      continue;
    }
    if (counts[i] > best_cnt) {
      best_cnt = counts[i];
      best_sym = i;
    }
    int f = static_cast<int>((static_cast<uint64_t>(counts[i]) * kFastM + (total >> 1)) / total);
    if (f < 1) f = 1;
    if (f > kFastM) f = kFastM;
    freqs[i] = static_cast<uint16_t>(f);
    sum_f += f;
  }
  int diff = kFastM - sum_f;
  if (static_cast<int>(freqs[best_sym]) + diff >= 1) {
    freqs[best_sym] = static_cast<uint16_t>(static_cast<int>(freqs[best_sym]) + diff);
  } else {
    // Distribute negative excess across symbols with freq > 1
    while (diff < 0) {
      int k = -1;
      uint16_t mf = 1;
      for (int i = 0; i <= max_sym; ++i) {
        if (freqs[i] > mf) {
          mf = freqs[i];
          k = i;
        }
      }
      if (k < 0) break;
      --freqs[k];
      ++diff;
    }
  }
}

void write_freq_tables(BitWriter& bw, const uint16_t freqs[kFastCtxs][kFastToks], const int max_syms[kFastCtxs]) {
  for (int c = 0; c < kFastCtxs; ++c) {
    if (max_syms[c] < 0) {
      bw.write_bits(0u, 1);
      continue;
    }
    bw.write_bits(1u, 1);
    bw.write_bits(static_cast<uint32_t>(max_syms[c]), 6);
    for (int s = 0; s <= max_syms[c]; ++s) {
      const uint16_t f = freqs[c][s];
      if (f == 0) {
        bw.write_bits(0u, 1);
      } else {
        bw.write_bits(1u, 1);
        // Unsigned Exp-Golomb on (f - 1) in 0..4095
        const uint32_t v = static_cast<uint32_t>(f);  // >= 1
        const int b = 31 - __builtin_clz(v);
        for (int z = 0; z < b; ++z) bw.write_bits(0u, 1);
        bw.write_bits(v, b + 1);
      }
    }
  }
}

void read_and_build_lut(BitReader& br, uint32_t lut[kFastCtxs * kFastM]) {
  for (int c = 0; c < kFastCtxs; ++c) {
    uint32_t* dst = lut + c * kFastM;
    if (br.read_bits(1) == 0u) {
      for (int cf = 0; cf < kFastM; ++cf) {
        dst[cf] = (static_cast<uint32_t>(cf) << 20) | (static_cast<uint32_t>(kFastM) << 7) | 0u;
      }
      continue;
    }
    const int max_sym = static_cast<int>(br.read_bits(6));
    if (max_sym >= kFastToks) throw CodecError("invalid fast max_sym in header");
    int cum = 0;
    for (int s = 0; s <= max_sym; ++s) {
      if (br.read_bits(1) == 0u) continue;
      int b = 0;
      while (br.read_bits(1) == 0u) {
        ++b;
        if (b > 12) throw CodecError("corrupt fast frequency Exp-Golomb");
      }
      const uint32_t tail = b > 0 ? br.read_bits(b) : 0u;
      const int f = static_cast<int>((1u << b) | tail);
      if (f <= 0 || cum + f > kFastM) throw CodecError("fast frequency sum exceeds 4096");
      const uint32_t base_entry = (static_cast<uint32_t>(f) << 7) | static_cast<uint32_t>(s);
      for (int k = 0; k < f; ++k) {
        dst[cum + k] = (static_cast<uint32_t>(k) << 20) | base_entry;
      }
      cum += f;
    }
    if (cum != kFastM) throw CodecError("fast frequency sum does not equal 4096");
  }
  br.check_zero_tail();
}

void append_u32_le(std::vector<uint8_t>& out, uint32_t v) {
  out.push_back(static_cast<uint8_t>(v));
  out.push_back(static_cast<uint8_t>(v >> 8));
  out.push_back(static_cast<uint8_t>(v >> 16));
  out.push_back(static_cast<uint8_t>(v >> 24));
}

void append_i32_le(std::vector<uint8_t>& out, int32_t v) {
  append_u32_le(out, static_cast<uint32_t>(v));
}

uint32_t read_u32_le(const uint8_t* data, size_t size, size_t& pos) {
  if (pos + 4 > size) throw CodecError("truncated fast u32 header");
  const uint32_t v = static_cast<uint32_t>(data[pos]) | (static_cast<uint32_t>(data[pos + 1]) << 8) |
                     (static_cast<uint32_t>(data[pos + 2]) << 16) | (static_cast<uint32_t>(data[pos + 3]) << 24);
  pos += 4;
  return v;
}

int32_t read_i32_le(const uint8_t* data, size_t size, size_t& pos) {
  return static_cast<int32_t>(read_u32_le(data, size, pos));
}

int choose_fast_rct(const std::vector<Plane>& raw) {
  if (raw.size() < 3) return kRctId;
  const int H = raw[0].height;
  const int W = raw[0].width;
  if (H < 4 || W < 4) return kRctSubAvg;
  const int32_t* r = raw[0].samples.data();
  const int32_t* g = raw[1].samples.data();
  const int32_t* b = raw[2].samples.data();

  uint64_t cost_id = 0, cost_sub = 0, cost_avg = 0;
  for (int y = 1; y < H; y += 4) {
    const int row = y * W;
    const int up = (y - 1) * W;
    for (int x = 1; x < W; x += 4) {
      const int i = row + x;
      const int iw = i - 1;
      const int in = up + x;
      // Identity L1 step cost
      cost_id += static_cast<uint64_t>(iabs64(int64_t(r[i]) - r[iw]) + iabs64(int64_t(r[i]) - r[in]) +
                                       iabs64(int64_t(g[i]) - g[iw]) + iabs64(int64_t(g[i]) - g[in]) +
                                       iabs64(int64_t(b[i]) - b[iw]) + iabs64(int64_t(b[i]) - b[in]));
      const int64_t g_cost = iabs64(int64_t(g[i]) - g[iw]) + iabs64(int64_t(g[i]) - g[in]);
      const int32_t rg_i = r[i] - g[i], rg_w = r[iw] - g[iw], rg_n = r[in] - g[in];
      const int64_t rg_cost = iabs64(int64_t(rg_i) - rg_w) + iabs64(int64_t(rg_i) - rg_n);
      const int32_t bg_i = b[i] - g[i], bg_w = b[iw] - g[iw], bg_n = b[in] - g[in];
      cost_sub += static_cast<uint64_t>(g_cost + rg_cost + iabs64(int64_t(bg_i) - bg_w) + iabs64(int64_t(bg_i) - bg_n));
      const int32_t ba_i = b[i] - ((r[i] + g[i]) >> 1);
      const int32_t ba_w = b[iw] - ((r[iw] + g[iw]) >> 1);
      const int32_t ba_n = b[in] - ((r[in] + g[in]) >> 1);
      cost_avg += static_cast<uint64_t>(g_cost + rg_cost + iabs64(int64_t(ba_i) - ba_w) + iabs64(int64_t(ba_i) - ba_n));
    }
  }
  if (cost_id <= cost_sub && cost_id <= cost_avg) return kRctId;
  return (cost_avg <= cost_sub) ? kRctSubAvg : kRctSubG;
}

}  // namespace

std::vector<uint8_t> encode_fast_plane(const Plane& P, const Plane* ref_slice) {
  const int H = P.height;
  const int W = P.width;
  const int N = H * W;
  if (H <= 0 || W <= 0) throw CodecError("empty fast plane");
  if (ref_slice != nullptr && (ref_slice->height != H || ref_slice->width != W)) {
    throw CodecError("ref_slice dimension mismatch");
  }

  const int32_t* ps = P.samples.data();
  int32_t pmin = ps[0];
  int32_t pmax = ps[0];
  for (int i = 1; i < N; ++i) {
    if (ps[i] < pmin) pmin = ps[i];
    if (ps[i] > pmax) pmax = ps[i];
  }

  std::vector<uint8_t> out;
  // Constant plane bypass (5 bytes total!)
  if (pmin == pmax) {
    out.push_back(0u);
    append_i32_le(out, pmin);
    return out;
  }

  // Optional quantization stride GCD compaction on 2D planes
  uint32_t step_gcd = 1u;
  if (ref_slice == nullptr) {
    uint32_t g = 0u;
    const int probe_n = std::min(N, 64);
    for (int i = 0; i < probe_n; ++i) {
      g = std::gcd(g, static_cast<uint32_t>(static_cast<int64_t>(ps[i]) - pmin));
      if (g == 1u) break;
    }
    if (g > 1u) {
      for (int i = probe_n; i < N; ++i) {
        g = std::gcd(g, static_cast<uint32_t>(static_cast<int64_t>(ps[i]) - pmin));
        if (g == 1u) break;
      }
      if (g > 1u && g <= 4096u) step_gcd = g;
    }
  }

  Plane scaled;
  const int32_t* src = ps;
  int32_t smin = pmin;
  int32_t smax = pmax;
  if (step_gcd > 1u) {
    scaled = Plane(H, W);
    const int32_t ig = static_cast<int32_t>(step_gcd);
    for (int i = 0; i < N; ++i) {
      scaled.samples[i] = static_cast<int32_t>((static_cast<int64_t>(ps[i]) - pmin) / ig);
    }
    src = scaled.samples.data();
    smin = 0;
    smax = static_cast<int32_t>((static_cast<int64_t>(pmax) - pmin) / ig);
  }
  const int32_t mid = static_cast<int32_t>((static_cast<int64_t>(smin) + smax) >> 1);
  const int n_stripes = (H >= 16) ? 4 : 1;

  // Pass 1: Tokenize all stripes and collect empirical context counts
  uint32_t counts[kFastCtxs][kFastToks]{};
  std::vector<uint8_t> all_tok(static_cast<size_t>(N));
  std::vector<uint8_t> all_ctx(static_cast<size_t>(N));
  std::array<FastBitWriter, 4> eb_writers;

  const int32_t* zs = ref_slice != nullptr ? ref_slice->samples.data() : nullptr;
  std::vector<uint8_t> prev_tok(static_cast<size_t>(W), 0);

  for (int st = 0; st < n_stripes; ++st) {
    const int y0 = (st * H) / n_stripes;
    const int y1 = ((st + 1) * H) / n_stripes;
    std::fill(prev_tok.begin(), prev_tok.end(), 0);
    auto& ebw = eb_writers[static_cast<size_t>(st)];
    ebw.out.reserve(static_cast<size_t>((y1 - y0) * W) / 2);

    for (int y = y0; y < y1; ++y) {
      const int32_t* row = src + y * W;
      const int32_t* up = (y > y0) ? (row - W) : nullptr;
      const int32_t* zr = zs != nullptr ? (zs + y * W) : nullptr;
      const int32_t* zu = (zs != nullptr && y > y0) ? (zr - W) : nullptr;
      uint8_t left_t = 0;

      for (int x = 0; x < W; ++x) {
        int32_t pred = mid;
        if (zr == nullptr) {
          if (y == y0) {
            pred = (x == 0) ? mid : row[x - 1];
          } else if (x == 0) {
            pred = up[0];
          } else {
            pred = med3_i32(row[x - 1], up[x], up[x - 1]);
          }
        } else {
          if (y == y0) {
            if (x == 0) {
              pred = std::clamp(zr[0], smin, smax);
            } else {
              const int64_t p = static_cast<int64_t>(zr[x]) + (static_cast<int64_t>(row[x - 1]) - zr[x - 1]);
              pred = static_cast<int32_t>(std::clamp<int64_t>(p, smin, smax));
            }
          } else if (x == 0) {
            const int64_t p = static_cast<int64_t>(zr[0]) + (static_cast<int64_t>(up[0]) - zu[0]);
            pred = static_cast<int32_t>(std::clamp<int64_t>(p, smin, smax));
          } else {
            pred = pred_3d_switched(row[x - 1], up[x], up[x - 1], zr[x], zr[x - 1], zu[x], zu[x - 1], smin, smax);
          }
        }

        const int32_t r = static_cast<int32_t>(static_cast<int64_t>(row[x]) - pred);
        const uint32_t u = zigzag_enc(r);
        if (u >= 524288u) throw CodecError("fast residual exceeds 19-bit ZigZag range");
        uint8_t tok = 0, nbits = 0;
        uint32_t extra = 0;
        tokenize_u32(u, tok, extra, nbits);
        ebw.put(extra, nbits);

        const uint8_t ctx = kFT.ctx_lut[static_cast<int>(left_t) + static_cast<int>(prev_tok[static_cast<size_t>(x)])];
        const size_t idx = static_cast<size_t>(y * W + x);
        all_tok[idx] = tok;
        all_ctx[idx] = ctx;
        counts[ctx][tok]++;
        left_t = tok;
        prev_tok[static_cast<size_t>(x)] = tok;
      }
    }
    ebw.finish();
  }

  uint16_t freqs[kFastCtxs][kFastToks]{};
  uint16_t cums[kFastCtxs][kFastToks]{};
  int max_syms[kFastCtxs]{};
  for (int c = 0; c < kFastCtxs; ++c) {
    normalize_counts_4096(counts[c], freqs[c], max_syms[c]);
    uint16_t acc = 0;
    for (int s = 0; s < kFastToks; ++s) {
      cums[c][s] = acc;
      acc = static_cast<uint16_t>(acc + freqs[c][s]);
    }
  }

  BitWriter tw;
  write_freq_tables(tw, freqs, max_syms);
  const auto tbl_bytes = tw.finish();

  std::array<std::vector<uint8_t>, 4> rans_blobs;
  for (int st = 0; st < n_stripes; ++st) {
    const int y0 = (st * H) / n_stripes;
    const int y1 = ((st + 1) * H) / n_stripes;
    const int i0 = y0 * W;
    const int i1 = y1 * W;
    Rans12Enc enc;
    enc.words.reserve(static_cast<size_t>(i1 - i0));
    for (int i = i1 - 1; i >= i0; --i) {
      const uint8_t c = all_ctx[static_cast<size_t>(i)];
      const uint8_t t = all_tok[static_cast<size_t>(i)];
      enc.put(cums[c][t], freqs[c][t]);
    }
    enc.append_to(rans_blobs[static_cast<size_t>(st)]);
  }

  // Plane header:
  // byte 0: plane_flags = (1 = coded) | ((step_gcd > 1) << 1) | ((n_stripes == 4) << 2)
  const uint8_t pflags = static_cast<uint8_t>(1u | (step_gcd > 1u ? 2u : 0u) | (n_stripes == 4 ? 4u : 0u));
  out.push_back(pflags);
  append_i32_le(out, pmin);
  append_i32_le(out, pmax);
  if (step_gcd > 1u) append_u32_le(out, step_gcd);

  if (tbl_bytes.size() > 65535u) throw CodecError("fast table header overflow");
  out.push_back(static_cast<uint8_t>(tbl_bytes.size()));
  out.push_back(static_cast<uint8_t>(tbl_bytes.size() >> 8));
  out.insert(out.end(), tbl_bytes.begin(), tbl_bytes.end());

  for (int st = 0; st < n_stripes; ++st) {
    append_u32_le(out, static_cast<uint32_t>(rans_blobs[static_cast<size_t>(st)].size()));
    append_u32_le(out, static_cast<uint32_t>(eb_writers[static_cast<size_t>(st)].out.size()));
  }
  for (int st = 0; st < n_stripes; ++st) {
    const auto& rb = rans_blobs[static_cast<size_t>(st)];
    const auto& eb = eb_writers[static_cast<size_t>(st)].out;
    out.insert(out.end(), rb.begin(), rb.end());
    out.insert(out.end(), eb.begin(), eb.end());
  }
  return out;
}

__attribute__((noinline)) Plane decode_fast_plane(const uint8_t* data, size_t size, size_t& pos, int H, int W,
                                                  int max_abs_bound, const Plane* ref_slice) {
  if (H <= 0 || W <= 0 || static_cast<int64_t>(H) * W > 134217728LL) {
    throw CodecError("invalid fast plane geometry");
  }
  if (ref_slice != nullptr && (ref_slice->height != H || ref_slice->width != W)) {
    throw CodecError("ref_slice dimension mismatch");
  }
  if (pos >= size) throw CodecError("truncated fast plane flags");
  const uint8_t pflags = data[pos++];
  if (pflags == 0u) {
    const int32_t pmin = read_i32_le(data, size, pos);
    if (pmin < -max_abs_bound || pmin > max_abs_bound) throw CodecError("fast constant value out of bounds");
    return Plane(H, W, pmin);
  }
  if ((pflags & ~0x07u) != 0u || (pflags & 1u) == 0u) throw CodecError("invalid fast plane flags");

  const bool has_gcd = (pflags & 2u) != 0u;
  const int n_stripes = (pflags & 4u) != 0u ? 4 : 1;
  if (n_stripes == 4 && H < 16) throw CodecError("invalid 4-stripe flag on small plane");
  if (has_gcd && ref_slice != nullptr) throw CodecError("invalid gcd flag on 3D P-slice");

  const int32_t pmin = read_i32_le(data, size, pos);
  const int32_t pmax = read_i32_le(data, size, pos);
  if (pmin < -max_abs_bound || pmax > max_abs_bound || pmin >= pmax) {
    throw CodecError("invalid fast plane [pmin, pmax] range");
  }
  uint32_t step_gcd = 1u;
  if (has_gcd) {
    step_gcd = read_u32_le(data, size, pos);
    if (step_gcd <= 1u || step_gcd > 4096u) throw CodecError("invalid fast plane step_gcd");
  }

  if (pos + 2 > size) throw CodecError("truncated fast table length");
  const uint16_t tbl_len = static_cast<uint16_t>(data[pos] | (uint16_t(data[pos + 1]) << 8));
  pos += 2;
  if (tbl_len == 0 || pos + tbl_len > size) throw CodecError("truncated fast frequency table");
  uint32_t lut[kFastCtxs * kFastM];
  {
    BitReader br(data + pos, tbl_len);
    pos += tbl_len;
    read_and_build_lut(br, lut);
  }

  uint32_t r_len[4]{}, e_len[4]{};
  for (int st = 0; st < n_stripes; ++st) {
    r_len[st] = read_u32_le(data, size, pos);
    e_len[st] = read_u32_le(data, size, pos);
  }
  Rans12Dec rdec[4];
  FastBitReader edec[4];
  for (int st = 0; st < n_stripes; ++st) {
    if (pos + r_len[st] > size) throw CodecError("truncated fast stripe rANS payload");
    rdec[st] = Rans12Dec(data + pos, r_len[st]);
    pos += r_len[st];
    if (pos + e_len[st] > size) throw CodecError("truncated fast stripe extra-bits payload");
    edec[st] = FastBitReader(data + pos, e_len[st]);
    pos += e_len[st];
  }

  const int32_t smin = has_gcd ? 0 : pmin;
  const int32_t smax = has_gcd ? static_cast<int32_t>((static_cast<int64_t>(pmax) - pmin) / int32_t(step_gcd)) : pmax;
  const int32_t mid = static_cast<int32_t>((static_cast<int64_t>(smin) + smax) >> 1);

  Plane out(H, W);
  int32_t* dst = out.samples.data();
  const int32_t* zs = ref_slice != nullptr ? ref_slice->samples.data() : nullptr;

  // Ultra-fast 4-way lockstep fused decode path when 4 equal-height 2D stripes apply!
  if (n_stripes == 4 && (H % 4) == 0 && zs == nullptr) {
    const int sh = H / 4;
    std::vector<uint8_t> ptok(static_cast<size_t>(4 * W), 0);
    uint8_t* pt0 = ptok.data();
    uint8_t* pt1 = pt0 + W;
    uint8_t* pt2 = pt1 + W;
    uint8_t* pt3 = pt2 + W;

    for (int ly = 0; ly < sh; ++ly) {
      int32_t* r0 = dst + (0 * sh + ly) * W;
      int32_t* r1 = dst + (1 * sh + ly) * W;
      int32_t* r2 = dst + (2 * sh + ly) * W;
      int32_t* r3 = dst + (3 * sh + ly) * W;
      uint8_t lt0 = 0, lt1 = 0, lt2 = 0, lt3 = 0;

      if (ly == 0) {
        for (int x = 0; x < W; ++x) {
          const uint32_t s0 = rdec[0].step(lut + (uint32_t(kFT.ctx_lut[lt0]) << kFastScale));
          const uint32_t s1 = rdec[1].step(lut + (uint32_t(kFT.ctx_lut[lt1]) << kFastScale));
          const uint32_t s2 = rdec[2].step(lut + (uint32_t(kFT.ctx_lut[lt2]) << kFastScale));
          const uint32_t s3 = rdec[3].step(lut + (uint32_t(kFT.ctx_lut[lt3]) << kFastScale));

          const uint32_t u0 = kFT.tok_base[s0] + edec[0].get(kFT.tok_ebits[s0]);
          const uint32_t u1 = kFT.tok_base[s1] + edec[1].get(kFT.tok_ebits[s1]);
          const uint32_t u2 = kFT.tok_base[s2] + edec[2].get(kFT.tok_ebits[s2]);
          const uint32_t u3 = kFT.tok_base[s3] + edec[3].get(kFT.tok_ebits[s3]);

          const int64_t v0 = static_cast<int64_t>(x == 0 ? mid : r0[x - 1]) + zigzag_dec(u0);
          const int64_t v1 = static_cast<int64_t>(x == 0 ? mid : r1[x - 1]) + zigzag_dec(u1);
          const int64_t v2 = static_cast<int64_t>(x == 0 ? mid : r2[x - 1]) + zigzag_dec(u2);
          const int64_t v3 = static_cast<int64_t>(x == 0 ? mid : r3[x - 1]) + zigzag_dec(u3);
          if (v0 < smin || v0 > smax || v1 < smin || v1 > smax || v2 < smin || v2 > smax || v3 < smin || v3 > smax) {
            throw CodecError("decoded fast sample out of [smin, smax]");
          }
          r0[x] = static_cast<int32_t>(v0);
          r1[x] = static_cast<int32_t>(v1);
          r2[x] = static_cast<int32_t>(v2);
          r3[x] = static_cast<int32_t>(v3);
          lt0 = pt0[x] = static_cast<uint8_t>(s0);
          lt1 = pt1[x] = static_cast<uint8_t>(s1);
          lt2 = pt2[x] = static_cast<uint8_t>(s2);
          lt3 = pt3[x] = static_cast<uint8_t>(s3);
        }
      } else {
        const int32_t* u0_row = r0 - W;
        const int32_t* u1_row = r1 - W;
        const int32_t* u2_row = r2 - W;
        const int32_t* u3_row = r3 - W;
        for (int x = 0; x < W; ++x) {
          const uint32_t c0 = kFT.ctx_lut[int(lt0) + int(pt0[x])];
          const uint32_t c1 = kFT.ctx_lut[int(lt1) + int(pt1[x])];
          const uint32_t c2 = kFT.ctx_lut[int(lt2) + int(pt2[x])];
          const uint32_t c3 = kFT.ctx_lut[int(lt3) + int(pt3[x])];

          const uint32_t s0 = rdec[0].step(lut + (c0 << kFastScale));
          const uint32_t s1 = rdec[1].step(lut + (c1 << kFastScale));
          const uint32_t s2 = rdec[2].step(lut + (c2 << kFastScale));
          const uint32_t s3 = rdec[3].step(lut + (c3 << kFastScale));

          const int32_t d0 = zigzag_dec(kFT.tok_base[s0] + edec[0].get(kFT.tok_ebits[s0]));
          const int32_t d1 = zigzag_dec(kFT.tok_base[s1] + edec[1].get(kFT.tok_ebits[s1]));
          const int32_t d2 = zigzag_dec(kFT.tok_base[s2] + edec[2].get(kFT.tok_ebits[s2]));
          const int32_t d3 = zigzag_dec(kFT.tok_base[s3] + edec[3].get(kFT.tok_ebits[s3]));

          const int32_t p0 = (x == 0) ? u0_row[0] : med3_i32(r0[x - 1], u0_row[x], u0_row[x - 1]);
          const int32_t p1 = (x == 0) ? u1_row[0] : med3_i32(r1[x - 1], u1_row[x], u1_row[x - 1]);
          const int32_t p2 = (x == 0) ? u2_row[0] : med3_i32(r2[x - 1], u2_row[x], u2_row[x - 1]);
          const int32_t p3 = (x == 0) ? u3_row[0] : med3_i32(r3[x - 1], u3_row[x], u3_row[x - 1]);

          const int64_t v0 = static_cast<int64_t>(p0) + d0;
          const int64_t v1 = static_cast<int64_t>(p1) + d1;
          const int64_t v2 = static_cast<int64_t>(p2) + d2;
          const int64_t v3 = static_cast<int64_t>(p3) + d3;
          if (v0 < smin || v0 > smax || v1 < smin || v1 > smax || v2 < smin || v2 > smax || v3 < smin || v3 > smax) {
            throw CodecError("decoded fast sample out of [smin, smax]");
          }
          r0[x] = static_cast<int32_t>(v0);
          r1[x] = static_cast<int32_t>(v1);
          r2[x] = static_cast<int32_t>(v2);
          r3[x] = static_cast<int32_t>(v3);
          lt0 = pt0[x] = static_cast<uint8_t>(s0);
          lt1 = pt1[x] = static_cast<uint8_t>(s1);
          lt2 = pt2[x] = static_cast<uint8_t>(s2);
          lt3 = pt3[x] = static_cast<uint8_t>(s3);
        }
      }
    }
  } else {
    // General stripe decoder (arbitrary H and/or 3D volumetric ref_slice)
    std::vector<uint8_t> ptok(static_cast<size_t>(W), 0);
    for (int st = 0; st < n_stripes; ++st) {
      const int y0 = (st * H) / n_stripes;
      const int y1 = ((st + 1) * H) / n_stripes;
      std::fill(ptok.begin(), ptok.end(), 0);
      auto& rd = rdec[st];
      auto& ed = edec[st];

      for (int y = y0; y < y1; ++y) {
        int32_t* row = dst + y * W;
        const int32_t* up = (y > y0) ? (row - W) : nullptr;
        const int32_t* zr = zs != nullptr ? (zs + y * W) : nullptr;
        const int32_t* zu = (zs != nullptr && y > y0) ? (zr - W) : nullptr;
        uint8_t lt = 0;

        for (int x = 0; x < W; ++x) {
          const uint32_t ctx = kFT.ctx_lut[int(lt) + int(ptok[static_cast<size_t>(x)])];
          const uint32_t sym = rd.step(lut + (ctx << kFastScale));
          const int32_t diff = zigzag_dec(kFT.tok_base[sym] + ed.get(kFT.tok_ebits[sym]));

          int32_t pred = mid;
          if (zr == nullptr) {
            if (y == y0) pred = (x == 0) ? mid : row[x - 1];
            else if (x == 0) pred = up[0];
            else pred = med3_i32(row[x - 1], up[x], up[x - 1]);
          } else {
            if (y == y0) {
              if (x == 0) {
                pred = std::clamp(zr[0], smin, smax);
              } else {
                const int64_t p = static_cast<int64_t>(zr[x]) + (static_cast<int64_t>(row[x - 1]) - zr[x - 1]);
                pred = static_cast<int32_t>(std::clamp<int64_t>(p, smin, smax));
              }
            } else if (x == 0) {
              const int64_t p = static_cast<int64_t>(zr[0]) + (static_cast<int64_t>(up[0]) - zu[0]);
              pred = static_cast<int32_t>(std::clamp<int64_t>(p, smin, smax));
            } else {
              pred = pred_3d_switched(row[x - 1], up[x], up[x - 1], zr[x], zr[x - 1], zu[x], zu[x - 1], smin, smax);
            }
          }

          const int64_t val = static_cast<int64_t>(pred) + diff;
          if (val < smin || val > smax) throw CodecError("decoded fast sample out of [smin, smax]");
          row[x] = static_cast<int32_t>(val);
          lt = ptok[static_cast<size_t>(x)] = static_cast<uint8_t>(sym);
        }
      }
    }
  }

  for (int st = 0; st < n_stripes; ++st) {
    rdec[st].check_end();
    edec[st].check_end();
  }

  if (has_gcd) {
    const int64_t ig = static_cast<int64_t>(step_gcd);
    const int N = H * W;
    for (int i = 0; i < N; ++i) {
      const int64_t v = static_cast<int64_t>(pmin) + static_cast<int64_t>(dst[i]) * ig;
      if (v < pmin || v > pmax) throw CodecError("fast gcd reconstruction overflow");
      dst[i] = static_cast<int32_t>(v);
    }
  }
  return out;
}

std::vector<uint8_t> encode_fast(const std::vector<Plane>& raw_planes, bool /*u16*/) {
  const int C = static_cast<int>(raw_planes.size());
  if (C != 1 && C != 3 && C != 4) throw CodecError("fast profile supports 1, 3, or 4 channels");
  const int H = raw_planes[0].height;
  const int W = raw_planes[0].width;
  if (H <= 0 || W <= 0) throw CodecError("empty fast tile");

  const int rct = choose_fast_rct(raw_planes);
  std::vector<Plane> tr(static_cast<size_t>(C));
  if (rct == kRctId || C == 1) {
    tr = raw_planes;
  } else if (rct == kRctSubG) {
    tr[0] = raw_planes[1];
    tr[1] = Plane(H, W);
    tr[2] = Plane(H, W);
    simd::diff_rows_i32(raw_planes[0].samples.data(), raw_planes[1].samples.data(), tr[1].samples.data(), H * W);
    simd::diff_rows_i32(raw_planes[2].samples.data(), raw_planes[1].samples.data(), tr[2].samples.data(), H * W);
    if (C == 4) tr[3] = raw_planes[3];
  } else {
    tr[0] = raw_planes[1];
    tr[1] = Plane(H, W);
    tr[2] = Plane(H, W);
    const int N = H * W;
    const int32_t* r = raw_planes[0].samples.data();
    const int32_t* g = raw_planes[1].samples.data();
    const int32_t* b = raw_planes[2].samples.data();
    int32_t* c1 = tr[1].samples.data();
    int32_t* c2 = tr[2].samples.data();
    for (int i = 0; i < N; ++i) {
      c1[i] = r[i] - g[i];
      c2[i] = b[i] - ((r[i] + g[i]) >> 1);
    }
    if (C == 4) tr[3] = raw_planes[3];
  }

  std::vector<uint8_t> out;
  out.push_back(static_cast<uint8_t>(rct));
  for (int c = 0; c < C; ++c) {
    const auto pb = encode_fast_plane(tr[static_cast<size_t>(c)], nullptr);
    out.insert(out.end(), pb.begin(), pb.end());
  }
  return out;
}

__attribute__((noinline)) std::vector<Plane> decode_fast(const uint8_t* data, size_t size, int height, int width,
                                                         int channels, bool u16) {
  if (channels != 1 && channels != 3 && channels != 4) throw CodecError("invalid fast channel count");
  if (height <= 0 || width <= 0 || static_cast<int64_t>(height) * width > 134217728LL) {
    throw CodecError("invalid fast tile dimensions");
  }
  if (size < 1) throw CodecError("truncated fast tile payload");
  size_t pos = 0;
  const int rct = data[pos++];
  if (rct > kRctSubAvg || (channels == 1 && rct != kRctId)) throw CodecError("invalid fast rct mode");

  const int maxv = u16 ? 65535 : 255;
  const int max_abs = maxv * 2;
  std::vector<Plane> tr(static_cast<size_t>(channels));
  for (int c = 0; c < channels; ++c) {
    tr[static_cast<size_t>(c)] = decode_fast_plane(data, size, pos, height, width, max_abs, nullptr);
  }
  if (pos != size) throw CodecError("trailing bytes in fast tile payload");

  if (rct == kRctId || channels == 1) {
    for (int c = 0; c < channels; ++c) {
      for (int32_t v : tr[static_cast<size_t>(c)].samples) {
        if (v < 0 || v > maxv) throw CodecError("decoded fast sample out of pixel range");
      }
    }
    return tr;
  }

  std::vector<Plane> raw(static_cast<size_t>(channels));
  raw[0] = Plane(height, width);
  raw[1] = std::move(tr[0]);
  raw[2] = Plane(height, width);
  const int N = height * width;
  if (rct == kRctSubG) {
    simd::rct_sub_green_inv_row(raw[1].samples.data(), tr[1].samples.data(), tr[2].samples.data(),
                                raw[0].samples.data(), raw[2].samples.data(), N);
  } else {
    simd::rct_sub_avg_inv_row(raw[1].samples.data(), tr[1].samples.data(), tr[2].samples.data(),
                              raw[0].samples.data(), raw[2].samples.data(), N);
  }
  if (channels == 4) raw[3] = std::move(tr[3]);

  for (int c = 0; c < channels; ++c) {
    const int32_t* sp = raw[static_cast<size_t>(c)].samples.data();
    for (int i = 0; i < N; ++i) {
      if (sp[i] < 0 || sp[i] > maxv) throw CodecError("decoded fast RCT sample out of pixel range");
    }
  }
  return raw;
}

}  // namespace vpxc
