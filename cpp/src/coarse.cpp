#include "vpxc/coarse.hpp"

#include "vpxc/bitio.hpp"
#include "vpxc/error.hpp"
#include "vpxc/mdl.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace vpxc {
namespace {

constexpr int kBuckets = 16;
constexpr int kMatches = 8;

int64_t iabs(int64_t v) {
  if (v >= 0) return v;
  if (v == INT64_MIN) return INT64_MAX;
  return -v;
}

int64_t shl(int64_t v, int bits) {
  if (bits <= 0) return v;
  return static_cast<int64_t>(static_cast<uint64_t>(v) << bits);
}

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

int bucket16(int64_t grad) {
  if (grad <= 0) return 0;
  const int bit = 63 - __builtin_clzll(static_cast<uint64_t>(grad));
  return bit < kBuckets ? bit : kBuckets - 1;
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

int64_t boundary_grad(int64_t left, int64_t up, int64_t ul, int64_t ur, int64_t ll, bool have_l, bool have_u,
                      bool have_ul, bool have_ur, bool have_ll) {
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
  return (values[mid - 1] + upper) / 2;
}

int estimate_noise_var(const Plane& plane, const Mask& valid) {
  if (plane.height < 3 || plane.width < 3) return 0;
  std::vector<int64_t> lap;
  for (int y = 1; y < plane.height - 1; ++y) {
    for (int x = 1; x < plane.width - 1; ++x) {
      if (!valid.at(y, x) || !valid.at(y - 1, x) || !valid.at(y + 1, x) || !valid.at(y, x - 1) ||
          !valid.at(y, x + 1)) {
        continue;
      }
      const int64_t center = plane.at(y, x);
      lap.push_back(iabs(4 * center - plane.at(y - 1, x) - plane.at(y + 1, x) - plane.at(y, x - 1) -
                         plane.at(y, x + 1)));
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

Fuse kalman_fuse(int64_t med, int64_t base_ref, const std::vector<int64_t>& matches, int64_t grad, int s_eta2) {
  if (matches.empty()) return Fuse{med, grad};
  const int64_t noise_std = s_eta2 > 0 ? isqrt_i64(s_eta2) : 0;
  int64_t gate = 2 * grad + 3 * noise_std + 2;
  if (gate < 6) gate = 6;
  std::vector<int64_t> accepted;
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
  const __int128 s_loc2 = static_cast<__int128>(grad) * grad + 4;
  __int128 sum = 0;
  for (int64_t value : accepted) sum += value;
  const __int128 acc = static_cast<__int128>(med) * s_eta2 + s_loc2 * sum;
  const __int128 denom = static_cast<__int128>(s_eta2) + static_cast<__int128>(accepted.size()) * s_loc2;
  const __int128 mu = floor_div128(acc * 2 + denom, denom * 2);
  const __int128 s_pred2 = static_cast<__int128>(s_eta2) + (s_loc2 * s_eta2 + (denom >> 1)) / denom;
  int64_t eff = isqrt_i64(static_cast<int64_t>(s_pred2));
  if (grad < eff) eff = grad;
  return Fuse{static_cast<int64_t>(mu), eff};
}

struct Grab {
  int64_t value = 0;
  bool ok = false;
};

Grab grab(const Plane& plane, const Mask& valid, int y, int x) {
  if (y < 0 || x < 0 || y >= plane.height || x >= plane.width) return {};
  if (y >= valid.height || x >= valid.width) throw CodecError("mask index out of range");
  if (!valid.at(y, x)) return {};
  return Grab{plane.at(y, x), true};
}

std::optional<int64_t> gap_predict(int64_t left, int64_t up, int64_t ul, int64_t ur, int64_t ll, int64_t nn,
                                   bool have_l, bool have_u, bool have_ul, bool have_ur, bool have_ll, bool have_nn) {
  if (!(have_l && have_u)) return std::nullopt;
  const int64_t west_west = have_ll ? ll : left;
  int64_t dh = iabs(left - west_west);
  int64_t dv = 0;
  if (have_ul) {
    dh += iabs(up - ul);
    dv += iabs(left - ul);
  }
  if (have_ur) dh += iabs(up - ur);
  if (have_nn) dv += iabs(up - nn);
  if (dv - dh > 80) return left;
  if (dh - dv > 80) return up;
  int64_t pred = (left + up) >> 1;
  if (have_ur && have_ul) pred += (ur - ul) >> 2;
  if (dv - dh > 32) pred = (pred + left) >> 1;
  else if (dh - dv > 32) pred = (pred + up) >> 1;
  return pred;
}

void split_mu(int64_t mu_q4, int64_t& mu_int, int& frac) {
  mu_int = (mu_q4 + 8) >> 4;
  int64_t f = mu_q4 - shl(mu_int, 4) + 8;
  if (f < 0) f = 0;
  else if (f > 15) f = 15;
  frac = static_cast<int>(f);
}

struct Site {
  int64_t mu_int = 0;
  int frac = 0;
  int bucket = 0;
  bool has_key = false;
  PatchKey key{};
  int64_t ul = 0;
  int n_feat = 0;
  int64_t feat[6] = {};
};

void require_plane(const Plane& plane, const Mask& valid) {
  if (plane.height < 0 || plane.width < 0) throw CodecError("negative plane shape");
  if (plane.height != valid.height || plane.width != valid.width) throw CodecError("plane and mask shapes differ");
  const size_t n = static_cast<size_t>(plane.height) * static_cast<size_t>(plane.width);
  if (plane.samples.size() < n || valid.bits.size() < n) throw CodecError("plane storage is short");
}

Site predict_site(const Plane& plane, const Mask& valid, const Plane* ref, int y, int x,
                  const std::map<PatchKey, std::vector<int64_t>>& table, int s_eta2, const int64_t* weights,
                  int n_weights) {
  const Grab left = grab(plane, valid, y, x - 1);
  const Grab ll = grab(plane, valid, y, x - 2);
  const Grab up = grab(plane, valid, y - 1, x);
  const Grab ul_g = grab(plane, valid, y - 1, x - 1);
  const Grab ur = grab(plane, valid, y - 1, x + 1);
  const Grab nn = grab(plane, valid, y - 2, x);
  const bool have_ul = ul_g.ok;
  const int64_t ul = have_ul ? ul_g.value : (left.ok ? left.value : up.value);
  const int64_t med = median_edge(left.value, up.value, ul, left.ok, up.ok, have_ul);
  const int64_t grad =
      boundary_grad(left.value, up.value, ul, ur.value, ll.value, left.ok, up.ok, have_ul, ur.ok, ll.ok);
  const int64_t west = left.ok ? left.value : med;
  const int64_t north = up.ok ? up.value : med;
  const std::optional<int64_t> gap =
      gap_predict(left.value, up.value, ul, ur.value, ll.value, nn.value, left.ok, up.ok, have_ul, ur.ok, ll.ok, nn.ok);
  const int64_t gap_v = gap ? *gap : med;
  int64_t kalman = med;
  int64_t eff = grad;
  Site site;
  if (left.ok || up.ok) {
    const int64_t d1 = left.value - ul;
    const int64_t d2 = up.value - ul;
    const int64_t d3 = iabs(d1) <= iabs(d2) ? (ur.value - up.value) : (left.value - ll.value);
    site.key = d8_patch_key(d1, d2, d3);
    site.has_key = true;
    std::vector<int64_t> matches;
    const auto found = table.find(site.key);
    if (found != table.end()) matches = found->second;
    const Fuse fused = kalman_fuse(med, ul, matches, grad, s_eta2);
    kalman = fused.pred;
    eff = fused.grad;
  }
  site.n_feat = 0;
  site.feat[site.n_feat++] = shl(west - med, 4);
  site.feat[site.n_feat++] = shl(north - med, 4);
  site.feat[site.n_feat++] = shl(gap_v - med, 4);
  site.feat[site.n_feat++] = shl(kalman - med, 4);
  if (ref != nullptr) {
    const Grab r_left = grab(*ref, valid, y, x - 1);
    const Grab r_up = grab(*ref, valid, y - 1, x);
    const Grab r_ul = grab(*ref, valid, y - 1, x - 1);
    const bool have_rul = r_ul.ok;
    const int64_t rul = have_rul ? r_ul.value : (r_left.ok ? r_left.value : r_up.value);
    const int64_t ref_med = median_edge(r_left.value, r_up.value, rul, r_left.ok, r_up.ok, have_rul);
    const Grab here = grab(*ref, valid, y, x);
    site.feat[site.n_feat++] = here.ok ? shl(here.value - ref_med, 4) : 0;
    const Grab down = grab(*ref, valid, y + 1, x);
    const Grab right = grab(*ref, valid, y, x + 1);
    int64_t lap = 0;
    if (here.ok && r_up.ok && down.ok && r_left.ok && right.ok) {
      lap = 4 * here.value - r_up.value - down.value - r_left.value - right.value;
    }
    site.feat[site.n_feat++] = shl(lap, 2);
  }
  int64_t corr = 0;
  if (weights != nullptr) {
    if (n_weights != site.n_feat) throw CodecError("coarse ridge width does not match the predictor bank");
    __int128 acc = 128;
    for (int i = 0; i < site.n_feat; ++i) acc += static_cast<__int128>(site.feat[i]) * weights[i];
    corr = static_cast<int64_t>(acc >> 8);
  }
  split_mu(shl(med, 4) + corr, site.mu_int, site.frac);
  site.bucket = bucket16(eff);
  site.ul = ul;
  return site;
}

void commit_match(std::map<PatchKey, std::vector<int64_t>>& table, const Site& site, int64_t value) {
  if (!site.has_key) return;
  auto& bucket = table[site.key];
  if (bucket.size() >= static_cast<size_t>(kMatches)) bucket.erase(bucket.begin());
  bucket.push_back(value - site.ul);
}

std::vector<std::pair<int, int>> valid_sites(const Mask& valid) {
  std::vector<std::pair<int, int>> sites;
  for (int y = 0; y < valid.height; ++y) {
    for (int x = 0; x < valid.width; ++x) {
      if (valid.at(y, x)) sites.emplace_back(y, x);
    }
  }
  return sites;
}

int32_t narrow_i32(int64_t value, const char* what) {
  if (value < static_cast<int64_t>(INT32_MIN) || value > static_cast<int64_t>(INT32_MAX)) throw CodecError(what);
  return static_cast<int32_t>(value);
}

std::vector<uint8_t> pack_header(const std::vector<uint8_t>& header, const std::vector<uint8_t>& payload) {
  if (header.size() > 65535) throw CodecError("coarse header exceeds 16 bits");
  std::vector<uint8_t> out;
  out.push_back(static_cast<uint8_t>(header.size() & 255u));
  out.push_back(static_cast<uint8_t>((header.size() >> 8) & 255u));
  out.insert(out.end(), header.begin(), header.end());
  out.insert(out.end(), payload.begin(), payload.end());
  return out;
}

void check_ref(const Plane* ref) {
  if (ref == nullptr) return;
  if (ref->height < 0 || ref->width < 0) throw CodecError("negative plane shape");
  const size_t n = static_cast<size_t>(ref->height) * static_cast<size_t>(ref->width);
  if (ref->samples.size() < n) throw CodecError("plane storage is short");
}

}  // namespace

std::vector<uint8_t> encode_coarse(const Plane& plane, const Mask& valid, const Plane* ref) {
  require_plane(plane, valid);
  check_ref(ref);
  const std::vector<std::pair<int, int>> sites = valid_sites(valid);
  if (sites.empty()) {
    BitWriter writer;
    writer.write_bits(0, 8);
    writer.write_bits(1, 2);
    writer.write_bits(0, 1);
    writer.write_bits(4, 5);
    const std::vector<uint8_t> header = writer.finish();
    const std::vector<uint8_t> payload = encode_parametric_rans({}, 1, {}, {});
    return pack_header(header, payload);
  }
  const int s_eta2 = (plane.height >= 3 && plane.width >= 3) ? estimate_noise_var(plane, valid) : 0;
  std::map<PatchKey, std::vector<int64_t>> table;
  std::vector<int64_t> base_mu;
  std::vector<int64_t> values;
  std::vector<int64_t> phi;
  int n_feat = 0;
  base_mu.reserve(sites.size());
  values.reserve(sites.size());
  for (const auto& site_yx : sites) {
    const Site site = predict_site(plane, valid, ref, site_yx.first, site_yx.second, table, s_eta2, nullptr, 0);
    if (n_feat == 0) n_feat = site.n_feat;
    base_mu.push_back(site.mu_int);
    const int64_t sample = plane.at(site_yx.first, site_yx.second);
    values.push_back(sample);
    for (int i = 0; i < site.n_feat; ++i) phi.push_back(site.feat[i]);
    commit_match(table, site, sample);
  }
  std::vector<int64_t> target(sites.size());
  for (size_t i = 0; i < sites.size(); ++i) target[i] = shl(values[i] - base_mu[i], 4);
  const std::optional<std::vector<int64_t>> weights =
      fit_ridge_q8(phi.data(), static_cast<int>(sites.size()), n_feat, target.data(), 48);

  table.clear();
  std::vector<int32_t> frac(sites.size());
  std::vector<int32_t> buckets(sites.size());
  std::vector<int64_t> resid64(sites.size());
  std::vector<int32_t> resid32(sites.size());
  const int64_t* wptr = weights ? weights->data() : nullptr;
  const int n_w = weights ? static_cast<int>(weights->size()) : 0;
  for (size_t i = 0; i < sites.size(); ++i) {
    const Site site =
        predict_site(plane, valid, ref, sites[i].first, sites[i].second, table, s_eta2, wptr, n_w);
    frac[i] = site.frac;
    buckets[i] = site.bucket;
    const int64_t sample = plane.at(sites[i].first, sites[i].second);
    resid64[i] = sample - site.mu_int;
    resid32[i] = narrow_i32(resid64[i], "coarse residual does not fit in int32");
    commit_match(table, site, sample);
  }
  const SigmaNu fit = fit_sigma_nu(resid64.data(), frac.data(), buckets.data(), static_cast<int>(sites.size()), kBuckets);
  std::vector<int32_t> sig_idx(sites.size());
  for (size_t i = 0; i < sites.size(); ++i) sig_idx[i] = fit.sigma[static_cast<size_t>(buckets[i])];
  const std::vector<uint8_t> payload =
      encode_parametric_rans(resid32, fit.nu, sig_idx, frac);

  BitWriter writer;
  writer.write_bits(static_cast<uint32_t>(s_eta2) & 255u, 8);
  writer.write_bits(static_cast<uint32_t>(fit.nu) & 3u, 2);
  writer.write_bits(weights ? 1u : 0u, 1);
  if (weights) {
    writer.write_bits(static_cast<uint32_t>(weights->size()), 4);
    for (int64_t weight : *weights) {
      writer.write_signed_golomb(narrow_i32(weight, "coarse weight out of range"));
    }
  }
  int prev = fit.sigma[0] & 31;
  writer.write_bits(static_cast<uint32_t>(prev), 5);
  for (int bucket = 1; bucket < kBuckets; ++bucket) {
    const int cur = fit.sigma[static_cast<size_t>(bucket)] & 31;
    writer.write_signed_golomb(cur - prev);
    prev = cur;
  }
  return pack_header(writer.finish(), payload);
}

Plane decode_coarse(const uint8_t* data, size_t size, int height, int width, const Mask& valid, const Plane* ref) {
  if (height < 0 || width < 0) throw CodecError("negative plane shape");
  if (valid.height != height || valid.width != width) throw CodecError("plane and mask shapes differ");
  const size_t n = static_cast<size_t>(height) * static_cast<size_t>(width);
  if (valid.bits.size() < n) throw CodecError("plane storage is short");
  check_ref(ref);
  if (data == nullptr || size < 2) throw CodecError("truncated coarse header");
  const uint32_t header_n = static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8);
  if (static_cast<uint64_t>(2) + header_n > size) throw CodecError("truncated coarse header");
  BitReader reader(data + 2, header_n);
  const uint8_t* payload = data + 2 + header_n;
  const size_t payload_n = size - 2 - header_n;
  const int s_eta2 = static_cast<int>(reader.read_bits(8));
  const int nu = static_cast<int>(reader.read_bits(2));
  const bool use_w = reader.read_bits(1) == 1;
  std::vector<int64_t> weights;
  if (use_w) {
    const int n_w = static_cast<int>(reader.read_bits(4));
    weights.resize(static_cast<size_t>(n_w));
    for (int i = 0; i < n_w; ++i) weights[static_cast<size_t>(i)] = reader.read_signed_golomb();
  }
  std::vector<int> sig_map(kBuckets);
  int prev = static_cast<int>(reader.read_bits(5));
  sig_map[0] = prev;
  for (int bucket = 1; bucket < kBuckets; ++bucket) {
    const int cur = prev + reader.read_signed_golomb();
    if (cur < 0 || cur > 31) throw CodecError("coarse sigma index out of range");
    sig_map[static_cast<size_t>(bucket)] = cur;
    prev = cur;
  }
  Plane out(height, width, 0);
  const std::vector<std::pair<int, int>> sites = valid_sites(valid);
  if (sites.empty()) return out;
  ParametricDecoder decoder(std::span<const uint8_t>(payload, payload_n), nu);
  std::map<PatchKey, std::vector<int64_t>> table;
  const int64_t* wptr = use_w ? weights.data() : nullptr;
  const int n_w = static_cast<int>(weights.size());
  for (const auto& site_yx : sites) {
    const Site site = predict_site(out, valid, ref, site_yx.first, site_yx.second, table, s_eta2, wptr, n_w);
    const int residual = decoder.read(sig_map[static_cast<size_t>(site.bucket)], site.frac);
    const int64_t value = site.mu_int + residual;
    out.at(site_yx.first, site_yx.second) = narrow_i32(value, "sample does not fit in int32");
    commit_match(table, site, value);
  }
  return out;
}

}  // namespace vpxc
