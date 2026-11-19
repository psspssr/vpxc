#include "vpxc/spatial.hpp"

#include "vpxc/bitio.hpp"
#include "vpxc/error.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace vpxc {
namespace {

constexpr int kMatches = 8;

int64_t iabs(int64_t v) {
  if (v >= 0) return v;
  if (v == INT64_MIN) return INT64_MAX;
  return -v;
}

// C++20 arithmetic right shift matches Python // for powers of two.
int64_t isqrt_i64(int64_t n) {
  if (n < 2) return n < 0 ? 0 : n;
  uint64_t x = static_cast<uint64_t>(n);
  uint64_t y = (x + 1) / 2;
  while (y < x) {
    x = y;
    y = (x + static_cast<uint64_t>(n) / x) / 2;
  }
  return static_cast<int64_t>(x);
}

__int128 floor_div128(__int128 a, __int128 b) {
  __int128 q = a / b;
  __int128 r = a % b;
  if (r != 0 && a < 0) --q;
  return q;
}

int rice_k(int64_t acc, int64_t count) {
  int k = 0;
  int64_t scaled = count;
  while (scaled < acc && k < 16) {
    scaled <<= 1;
    ++k;
  }
  return k;
}

int bucket_of(int64_t grad, int limit) {
  if (grad <= 0) return 0;
  const int bit = 63 - __builtin_clzll(static_cast<uint64_t>(grad));
  return bit < limit ? bit : limit - 1;
}

int64_t median_edge(int64_t left, int64_t up, int64_t ul, bool have_l, bool have_u, bool have_ul) {
  if (!have_l && !have_u) return 0;
  if (!have_l) return up;
  if (!have_u) return left;
  if (!have_ul) return (left + up) >> 1;
  const int64_t hi = left > up ? left : up;
  const int64_t lo = left > up ? up : left;
  const int64_t pred = left + up - ul;
  if (ul < lo) return hi;
  if (ul > hi) return lo;
  return pred;
}

int64_t boundary_grad(int64_t left, int64_t up, int64_t ul, int64_t ur, int64_t ll, bool have_l,
                      bool have_u, bool have_ul, bool have_ur, bool have_ll) {
  if (have_l && have_u && have_ul) {
    int64_t g = iabs(left - ul) + iabs(up - ul);
    if (have_ur) g = (2 * g + iabs(up - ur)) >> 1;
    return g;
  }
  if (have_l && have_u) return iabs(left - up);
  if (have_l) return have_ll ? iabs(left - ll) : 0;
  if (have_u) return have_ur ? iabs(up - ur) : 0;
  return 0;
}

int64_t median_value(std::vector<int64_t>& values) {
  const size_t n = values.size();
  if (n == 0) return 0;
  const size_t mid = n / 2;
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
  if (n & 1u) return values[mid];
  const int64_t upper = values[mid];
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid - 1),
                   values.begin() + static_cast<std::ptrdiff_t>(mid));
  const int64_t lower = values[mid - 1];
  return (lower + upper) / 2;
}

int estimate_noise_var(const Plane& plane, const Mask& valid) {
  if (plane.height < 3 || plane.width < 3) return 0;
  std::vector<int64_t> lap;
  lap.reserve(static_cast<size_t>(plane.height) * static_cast<size_t>(plane.width) / 2);
  for (int y = 1; y < plane.height - 1; ++y) {
    for (int x = 1; x < plane.width - 1; ++x) {
      if (!valid.at(y, x) || !valid.at(y - 1, x) || !valid.at(y + 1, x) || !valid.at(y, x - 1) ||
          !valid.at(y, x + 1)) {
        continue;
      }
      const int64_t center = plane.at(y, x);
      const int64_t value = 4 * center - plane.at(y - 1, x) - plane.at(y + 1, x) - plane.at(y, x - 1) -
                            plane.at(y, x + 1);
      lap.push_back(iabs(value));
    }
  }
  if (lap.empty()) return 0;
  const int64_t mad = median_value(lap);
  if (mad == 0) return 0;
  int64_t est = (mad * mad + 4) / 9;
  if (est < 0) est = 0;
  if (est > 255) est = 255;
  return static_cast<int>(est);
}

struct PatchKey {
  int64_t q1 = 0;
  int64_t q2 = 0;
  int64_t q3 = 0;
  bool operator<(const PatchKey& other) const {
    if (q1 != other.q1) return q1 < other.q1;
    if (q2 != other.q2) return q2 < other.q2;
    return q3 < other.q3;
  }
};

PatchKey d8_patch_key(int64_t d1, int64_t d2, int64_t d3) {
  const int64_t q1 = d1 >> 2;
  const int64_t q2 = d2 >> 2;
  const int64_t q3 = d3 >> 2;
  if (q1 <= q2) return PatchKey{q1, q2, q3};
  return PatchKey{q2, q1, q3};
}

struct Fuse {
  int64_t pred;
  int64_t grad;
};

Fuse kalman_fuse(int64_t med, int64_t base_ref, const std::vector<int64_t>& matches, int64_t grad,
                 int s_eta2) {
  if (matches.empty()) return Fuse{med, grad};
  const int64_t noise_std = s_eta2 > 0 ? isqrt_i64(s_eta2) : 0;
  int64_t gate = 2 * grad + 3 * noise_std + 2;
  if (gate < 6) gate = 6;
  std::vector<int64_t> accepted;
  accepted.reserve(matches.size());
  for (int64_t delta : matches) {
    const int64_t pred = base_ref + delta;
    if (iabs(pred - med) <= gate) accepted.push_back(pred);
  }
  if (accepted.empty()) return Fuse{med, grad};
  if (s_eta2 == 0) {
    const int64_t copy_val = accepted.back();
    int64_t agreement = 0;
    for (int64_t value : accepted) {
      const int64_t gap = iabs(value - copy_val);
      if (gap > agreement) agreement = gap;
    }
    return Fuse{copy_val, grad < agreement ? grad : agreement};
  }
  const __int128 s_loc2 = static_cast<__int128>(grad) * static_cast<__int128>(grad) + 4;
  __int128 sum = 0;
  for (int64_t value : accepted) sum += value;
  const __int128 acc = static_cast<__int128>(med) * s_eta2 + s_loc2 * sum;
  const __int128 denom =
      static_cast<__int128>(s_eta2) + static_cast<__int128>(accepted.size()) * s_loc2;
  const __int128 mu = floor_div128(acc * 2 + denom, denom * 2);
  const __int128 s_pred2 = static_cast<__int128>(s_eta2) + (s_loc2 * s_eta2 + (denom >> 1)) / denom;
  int64_t eff = isqrt_i64(static_cast<int64_t>(s_pred2));
  if (grad < eff) eff = grad;
  return Fuse{static_cast<int64_t>(mu), eff};
}

void require_shape(const Plane& plane, const Mask& valid) {
  if (plane.height < 0 || plane.width < 0) throw CodecError("negative plane shape");
  if (plane.height != valid.height || plane.width != valid.width) {
    throw CodecError("plane and mask shapes differ");
  }
  const size_t n = static_cast<size_t>(plane.height) * static_cast<size_t>(plane.width);
  if (plane.samples.size() < n || valid.bits.size() < n) throw CodecError("plane storage is short");
}

struct SitePred {
  int64_t pred = 0;
  int64_t eff = 0;
  bool has_key = false;
  PatchKey key{};
  int64_t ul = 0;
};

SitePred predict_site(const Plane& plane, const Mask& valid, int y, int x,
                      const std::map<PatchKey, std::vector<int64_t>>& table, bool kalman, int s_eta2) {
  const bool have_l = x > 0 && valid.at(y, x - 1);
  const bool have_ll = x > 1 && valid.at(y, x - 2);
  const bool have_u = y > 0 && valid.at(y - 1, x);
  const bool have_ul = y > 0 && x > 0 && valid.at(y - 1, x - 1);
  const bool have_ur = y > 0 && x + 1 < plane.width && valid.at(y - 1, x + 1);
  const int64_t left = have_l ? plane.at(y, x - 1) : 0;
  const int64_t ll = have_ll ? plane.at(y, x - 2) : left;
  const int64_t up = have_u ? plane.at(y - 1, x) : 0;
  const int64_t ul = have_ul ? plane.at(y - 1, x - 1) : (have_l ? left : up);
  const int64_t ur = have_ur ? plane.at(y - 1, x + 1) : up;
  SitePred site;
  site.pred = median_edge(left, up, ul, have_l, have_u, have_ul);
  site.eff = boundary_grad(left, up, ul, ur, ll, have_l, have_u, have_ul, have_ur, have_ll);
  site.ul = ul;
  if (kalman && (have_l || have_u)) {
    const int64_t d1 = left - ul;
    const int64_t d2 = up - ul;
    const int64_t d3 = iabs(d1) <= iabs(d2) ? (ur - up) : (left - ll);
    site.key = d8_patch_key(d1, d2, d3);
    site.has_key = true;
    std::vector<int64_t> matches;
    const auto found = table.find(site.key);
    if (found != table.end()) matches = found->second;
    const Fuse fused = kalman_fuse(site.pred, ul, matches, site.eff, s_eta2);
    site.pred = fused.pred;
    site.eff = fused.grad;
  }
  return site;
}

void commit_match(std::map<PatchKey, std::vector<int64_t>>& table, const SitePred& site, int64_t value) {
  if (!site.has_key) return;
  auto& bucket = table[site.key];
  if (bucket.size() >= static_cast<size_t>(kMatches)) bucket.erase(bucket.begin());
  bucket.push_back(value - site.ul);
}

int rice_symbol(int64_t err) {
  if (err == INT64_MIN) throw CodecError("rice symbol out of range");
  const int64_t symbol = err < 0 ? ((-err) << 1) - 1 : (err << 1);
  if (symbol < 0 || symbol > static_cast<int64_t>(INT32_MAX)) throw CodecError("rice symbol out of range");
  return static_cast<int>(symbol);
}

int32_t narrow_sample(int64_t value) {
  if (value < static_cast<int64_t>(INT32_MIN) || value > static_cast<int64_t>(INT32_MAX)) {
    throw CodecError("sample does not fit in int32");
  }
  return static_cast<int32_t>(value);
}

}  // namespace

std::vector<uint8_t> encode_raster(const Plane& plane, const Mask& valid, bool kalman) {
  require_shape(plane, valid);
  BitWriter writer;
  const int s_eta2 = kalman ? estimate_noise_var(plane, valid) : 0;
  if (kalman) writer.write_bits(static_cast<uint32_t>(s_eta2), 8);
  int64_t acc[8];
  int64_t count[8];
  for (int i = 0; i < 8; ++i) {
    acc[i] = 4;
    count[i] = 1;
  }
  std::map<PatchKey, std::vector<int64_t>> table;
  for (int y = 0; y < plane.height; ++y) {
    for (int x = 0; x < plane.width; ++x) {
      if (!valid.at(y, x)) continue;
      const SitePred site = predict_site(plane, valid, y, x, table, kalman, s_eta2);
      const int64_t err = static_cast<int64_t>(plane.at(y, x)) - site.pred;
      const int symbol = rice_symbol(err);
      const int ctx = bucket_of(site.eff, 8);
      const int k = rice_k(acc[ctx], count[ctx]);
      writer.write_rice(symbol, k);
      acc[ctx] += symbol;
      count[ctx] += 1;
      if (count[ctx] >= 64) {
        acc[ctx] = acc[ctx] >> 1;
        if (acc[ctx] < 1) acc[ctx] = 1;
        count[ctx] >>= 1;
      }
      commit_match(table, site, plane.at(y, x));
    }
  }
  return writer.finish();
}

Plane decode_raster(const uint8_t* data, size_t size, int height, int width, const Mask& valid, bool kalman) {
  if (height < 0 || width < 0) throw CodecError("negative plane shape");
  if (valid.height != height || valid.width != width) throw CodecError("plane and mask shapes differ");
  const size_t n = static_cast<size_t>(height) * static_cast<size_t>(width);
  if (valid.bits.size() < n) throw CodecError("plane storage is short");
  Plane plane(height, width, 0);
  BitReader reader(data, size);
  const int s_eta2 = kalman ? static_cast<int>(reader.read_bits(8)) : 0;
  int64_t acc[8];
  int64_t count[8];
  for (int i = 0; i < 8; ++i) {
    acc[i] = 4;
    count[i] = 1;
  }
  std::map<PatchKey, std::vector<int64_t>> table;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (!valid.at(y, x)) continue;
      const SitePred site = predict_site(plane, valid, y, x, table, kalman, s_eta2);
      const int ctx = bucket_of(site.eff, 8);
      const int k = rice_k(acc[ctx], count[ctx]);
      const int symbol = reader.read_rice(k);
      const int err = unzigzag(symbol);
      const int64_t value = site.pred + err;
      plane.at(y, x) = narrow_sample(value);
      acc[ctx] += symbol;
      count[ctx] += 1;
      if (count[ctx] >= 64) {
        acc[ctx] = acc[ctx] >> 1;
        if (acc[ctx] < 1) acc[ctx] = 1;
        count[ctx] >>= 1;
      }
      commit_match(table, site, value);
    }
  }
  return plane;
}

}  // namespace vpxc
