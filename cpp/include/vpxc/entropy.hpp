#pragma once

// Dyadic quotient coding: adaptive Rice, parametric Dyadic-Core66 rANS, and the
// bits-back rANS stack. Integer rules match src/vpxc/entropy.py. Decode paths
// do not use floating point. Bit packing is BitWriter / BitReader.

#include "vpxc/bitio.hpp"
#include "vpxc/image.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace vpxc {

constexpr uint64_t RANS64_L = 1ull << 31;
constexpr int RANS_SCALE = 16;
constexpr int RANS_M = 1 << RANS_SCALE;
constexpr int CORE_BINS = 66;  // quotients [-32, 32] plus escape 65
constexpr int QMAX = 32;
constexpr int TILT_QMAX = 6;
constexpr int TILT_MAX_DEPTH = 3;
constexpr int NUM_SIGMAS = 32;
constexpr int NUM_FRACS = 16;
constexpr int NUM_NUS = 4;
constexpr int TILT_QS = 2 * TILT_QMAX + 1;
constexpr int TILT_NODES = (1 << TILT_MAX_DEPTH) - 1;

// Q8 sample units. sigma = SIGMA_Q8 / 256, from 0.125 LSB through 16-bit range.
constexpr int32_t SIGMA_Q8[NUM_SIGMAS] = {
    32,     42,     56,     74,     98,     130,    172,    228,    302,    400,   530,
    702,    930,    1232,   1632,   2162,   2864,   3794,   5026,   6658,   8820,  11684,
    15478,  20504,  27162,  35982,  47666,  63144,  83648,  110810, 146792, 194458,
};

// max(0, ceil(log2((sigma_q8 / 256) / 2.5))). sigma_q8 <= 640 yields 0.
int dyadic_shift_q8(int64_t sigma_q8);
// dyadic_shift_q8(activity << 8). The shift result is capped at 12.
int dyadic_shift(int64_t activity);

struct DyadicParts {
  int64_t quotient = 0;
  // Unset when shift <= 0, matching Python's None remainder.
  std::optional<int64_t> remainder;
};

DyadicParts split_dyadic(int64_t residual, int shift);
int64_t join_dyadic(int64_t quotient, std::optional<int64_t> remainder, int shift);

// Smallest k with (count << k) >= acc, capped at 16.
int rice_k(int64_t acc, int64_t count);

// Exact integer largest-remainder normalization. The symbol count is counts.size().
// Laplace mode matches normalize_freqs: each mass is max(count, 0) + 1.
// Both overloads reject a non-positive total.
std::vector<int32_t> normalize_freqs(std::span<const int64_t> counts, int total = RANS_M);
std::vector<int32_t> normalize_freqs(std::span<const int32_t> counts, int total = RANS_M);

// cum[0] = 0, cum[i+1] = sum(freqs[0..i]). The result has freqs.size() + 1 entries.
std::vector<int32_t> cumulative(std::span<const int32_t> freqs);
std::vector<int32_t> cumulative(std::span<const int64_t> freqs);

// 64-bit rANS, 32-bit renormalization, scale 2^16.
// finish() layout: state hi, state lo, then renormalization words newest-first.
class RansEncoder {
 public:
  RansEncoder() = default;
  void put(int start, int freq);
  // starts/freqs are already ordered from the last symbol to the first.
  void put_batch(std::span<const int32_t> starts, std::span<const int32_t> freqs);
  std::vector<uint8_t> finish();

 private:
  uint64_t state_ = RANS64_L;
  std::vector<uint32_t> words_;
};

// Non-owning. `data` must outlive the decoder.
class RansDecoder {
 public:
  explicit RansDecoder(std::span<const uint8_t> data);
  inline int cf() const { return int(state_ & static_cast<uint64_t>(RANS_M - 1)); }
  inline void advance(int start, int freq) {
    const uint64_t c = state_ & static_cast<uint64_t>(RANS_M - 1);
    uint64_t x = static_cast<uint64_t>(freq) * (state_ >> RANS_SCALE) + c - static_cast<uint64_t>(start);
    while (x < RANS64_L) {
      if (pos_ + 4 > size_) throw CodecError("truncated rANS stream");
      const uint8_t* p = data_ + pos_;
      const uint32_t word =
          uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
      pos_ += 4;
      x = (x << 32) | word;
    }
    state_ = x;
  }
  void check_finished() const;

 private:
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
  size_t pos_ = 0;
  uint64_t state_ = 0;
};

// LIFO rANS word stack. `state` and `words` match the Python attributes:
// words.back() is the next word pop() renorms with.
class RansBitsBackStack {
 public:
  uint64_t state = RANS64_L;
  std::vector<uint32_t> words;

  RansBitsBackStack() = default;
  RansBitsBackStack(uint64_t state_in, std::vector<uint32_t> words_in)
      : state(state_in), words(std::move(words_in)) {}

  void push(int start, int freq);
  int pop(std::span<const int32_t> cum, std::span<const int32_t> freqs);
  std::vector<uint8_t> to_bytes() const;
  static RansBitsBackStack from_bytes(std::span<const uint8_t> data);
};

// Signed samples on valid sites, row-major. context == nullptr forces context 0.
// Context values outside 0..7 are bucket 7.
std::vector<uint8_t> encode_rice_values(const Plane& values, const Mask& valid, const Plane* context,
                                        bool causal_mu);
inline std::vector<uint8_t> encode_rice_values(const Plane& values, const Mask& valid, bool causal_mu) {
  return encode_rice_values(values, valid, nullptr, causal_mu);
}

Plane decode_rice_values(std::span<const uint8_t> data, int height, int width, const Mask& valid,
                         const Plane* context, bool causal_mu);
inline Plane decode_rice_values(std::span<const uint8_t> data, int height, int width, const Mask& valid,
                                bool causal_mu) {
  return decode_rice_values(data, height, width, valid, nullptr, causal_mu);
}

// Two-pass empirical Dyadic-Core66 rANS. `shifts` is per-sample and may be 0.
std::vector<uint8_t> encode_dyadic_rans(const Plane& residuals, const Mask& valid, const Plane& shifts);
Plane decode_dyadic_rans(std::span<const uint8_t> data, int height, int width, const Mask& valid,
                         const Plane& shifts);

// Student-t CDF in Q24 (1 << 24 == 1.0). z_q12 is round(z * 4096). nu_idx is 0..3.
int32_t eval_student_cdf_q24(int64_t z_q12, int nu_idx);

// Cached tables built with the same integer PCHIP / largest-remainder rules as Python.
// freqs[nu][sigma][frac][bin], cums[nu][sigma][frac][bin] (67 entries, leading 0),
// tilt_p0[nu][sigma][q + 6][node] is Q16 P(bit == 0), default RANS_M / 2.
struct ParametricTables {
  int32_t shifts[NUM_SIGMAS];
  int32_t freqs[NUM_NUS][NUM_SIGMAS][NUM_FRACS][CORE_BINS];
  int32_t cums[NUM_NUS][NUM_SIGMAS][NUM_FRACS][CORE_BINS + 1];
  int32_t tilt_p0[NUM_NUS][NUM_SIGMAS][TILT_QS][TILT_NODES];
};

const ParametricTables& get_parametric_tables();

// Bilateral sheaf bridge Core66 row at b == 0. sig_q8 is clamped to at least 32.
struct CoreRow {
  std::vector<int32_t> freqs;
  std::vector<int32_t> cum;
};

CoreRow build_bimodal_row(int nu_idx, int sig_q8, int mode_lo, int mode_hi);

// Batch parametric coder.
// Empty optional spans mean "absent", matching Python None.
// sig/frac are clipped to their legal ranges on this path.
// ext_freq is n * CORE_BINS and ext_cum is n * (CORE_BINS + 1), row-major, or both empty.
// When external rows are present they replace the core symbol distribution; tilt still
// uses the parametric remainder table. A bridge is (mask, lo, hi), applied only where
// the mask is set, the dyadic shift is 0, and the clamped modes differ by at least 6.
std::vector<uint8_t> encode_parametric_rans(std::span<const int32_t> r_tilde, int nu_idx,
                                            std::span<const int32_t> sig_idx,
                                            std::span<const int32_t> frac_idx,
                                            std::span<const uint8_t> bridge_mask = {},
                                            std::span<const int32_t> bridge_lo = {},
                                            std::span<const int32_t> bridge_hi = {},
                                            std::span<const int32_t> ext_freq = {},
                                            std::span<const int32_t> ext_cum = {});

std::vector<int32_t> decode_parametric_rans(std::span<const uint8_t> data, int count, int nu_idx,
                                            std::span<const int32_t> sig_idx,
                                            std::span<const int32_t> frac_idx,
                                            std::span<const uint8_t> bridge_mask = {},
                                            std::span<const int32_t> bridge_lo = {},
                                            std::span<const int32_t> bridge_hi = {},
                                            std::span<const int32_t> ext_freq = {},
                                            std::span<const int32_t> ext_cum = {});

// Pull-style inverse. Owns a copy of the payload bytes.
// sig/frac outside 0..NUM_*-1 become 0 (they are not clipped). That matches
// ParametricDecoder.read and differs from the batch clip.
class ParametricDecoder {
 public:
  explicit ParametricDecoder(std::span<const uint8_t> data, int nu_idx);
  ParametricDecoder(const ParametricDecoder&) = delete;
  ParametricDecoder& operator=(const ParametricDecoder&) = delete;
  ParametricDecoder(ParametricDecoder&&) noexcept = default;
  ParametricDecoder& operator=(ParametricDecoder&&) noexcept = default;

  int32_t read(int sig_i, int frac_i, std::optional<std::pair<int, int>> bridge = std::nullopt);
  void check_finished() const;

 private:
  struct BiKey {
    int s = 0;
    int f = 0;
    int lo = 0;
    int hi = 0;
    bool operator<(const BiKey& other) const {
      if (s != other.s) return s < other.s;
      if (f != other.f) return f < other.f;
      if (lo != other.lo) return lo < other.lo;
      return hi < other.hi;
    }
  };

  int nu_ = 0;
  std::vector<uint8_t> raw_;
  std::vector<uint8_t> extra_;
  uint64_t x_ = 0;
  size_t pos_ = 0;
  // Points at extra_. Declared after extra_ so the default move keeps the pointer.
  std::optional<BitReader> reader_;
  std::map<BiKey, CoreRow> bi_;
};

// Activity buckets 0..7 map to sigma {1, 2, 3, 5, 10, 20, 40, 80}. Other buckets stay 0.
Plane shift_map_from_activity(const Plane& activity_ctx);

}  // namespace vpxc
