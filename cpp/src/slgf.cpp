#include "vpxc/slgf.hpp"

#include "vpxc/bitio.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace vpxc {
namespace {

constexpr int kWeightMax = 2047;

int64_t iabs64(int64_t v) {
  if (v >= 0) return v;
  if (v == std::numeric_limits<int64_t>::min()) return std::numeric_limits<int64_t>::max();
  return -v;
}

int32_t i32_wrap(int64_t v) {
  const auto bits = static_cast<uint32_t>(static_cast<uint64_t>(v));
  int32_t out = 0;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

int bit_length_u(uint64_t v) {
  if (v == 0) return 0;
  return 64 - __builtin_clzll(v);
}

__int128 floor_div(__int128 a, __int128 b) {
  __int128 q = a / b;
  const __int128 r = a % b;
  if (r != 0 && ((a < 0) != (b < 0))) --q;
  return q;
}

int64_t isqrt_int(int64_t n) {
  if (n <= 0) return 0;
  int64_t x = n;
  int64_t y = (x + 1) >> 1;
  while (y < x) {
    x = y;
    y = (x + n / x) >> 1;
  }
  return x;
}

int64_t edge_at(const Plane& plane, int y, int x) {
  if (y < 0) y = 0;
  if (x < 0) x = 0;
  if (y >= plane.height) y = plane.height - 1;
  if (x >= plane.width) x = plane.width - 1;
  return plane.samples[static_cast<size_t>(y) * static_cast<size_t>(plane.width) + static_cast<size_t>(x)];
}

bool mask_at(const Mask& mask, int y, int x) {
  if (y < 0 || x < 0 || y >= mask.height || x >= mask.width) return false;
  return mask.at(y, x);
}

size_t lin(int width, int y, int x) {
  return static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
}

int64_t dot_q8(const int64_t* row, int cols, const int64_t* weights) {
  __int128 acc = 0;
  for (int j = 0; j < cols; ++j) acc += static_cast<__int128>(row[j]) * weights[j];
  acc += 128;
  return static_cast<int64_t>(acc >> 8);
}

int golomb_bits(int64_t value) {
  uint64_t mag = value < 0 ? static_cast<uint64_t>(-value) : static_cast<uint64_t>(value);
  mag = (mag << 1) + 1;
  return 2 * bit_length_u(mag);
}

int64_t round_half_even(double x) {
  if (!std::isfinite(x)) return 0;
  const double ax = std::fabs(x);
  const double ip = std::floor(ax);
  const double frac = ax - ip;
  const auto base = static_cast<int64_t>(ip);
  int64_t mag = base;
  if (frac > 0.5) mag = base + 1;
  else if (frac == 0.5) mag = (base % 2 == 0) ? base : base + 1;
  return x < 0 ? -mag : mag;
}

bool solve_ge(std::vector<double> a, std::vector<double> b, int n, std::vector<double>& x) {
  double scale = 0.0;
  for (double v : a) scale = std::max(scale, std::fabs(v));
  if (!(scale > 0.0)) return false;
  for (int col = 0; col < n; ++col) {
    int piv = col;
    double best = std::fabs(a[static_cast<size_t>(col * n + col)]);
    for (int row = col + 1; row < n; ++row) {
      const double v = std::fabs(a[static_cast<size_t>(row * n + col)]);
      if (v > best) {
        best = v;
        piv = row;
      }
    }
    if (!(best > scale * 1e-14)) return false;
    if (piv != col) {
      for (int j = col; j < n; ++j) {
        std::swap(a[static_cast<size_t>(col * n + j)], a[static_cast<size_t>(piv * n + j)]);
      }
      std::swap(b[static_cast<size_t>(col)], b[static_cast<size_t>(piv)]);
    }
    const double diag = a[static_cast<size_t>(col * n + col)];
    for (int row = col + 1; row < n; ++row) {
      const double factor = a[static_cast<size_t>(row * n + col)] / diag;
      a[static_cast<size_t>(row * n + col)] = 0.0;
      for (int j = col + 1; j < n; ++j) {
        a[static_cast<size_t>(row * n + j)] -= factor * a[static_cast<size_t>(col * n + j)];
      }
      b[static_cast<size_t>(row)] -= factor * b[static_cast<size_t>(col)];
    }
  }
  x.assign(static_cast<size_t>(n), 0.0);
  for (int row = n - 1; row >= 0; --row) {
    double sum = b[static_cast<size_t>(row)];
    for (int j = row + 1; j < n; ++j) sum -= a[static_cast<size_t>(row * n + j)] * x[static_cast<size_t>(j)];
    const double diag = a[static_cast<size_t>(row * n + row)];
    if (!(std::fabs(diag) > scale * 1e-14)) return false;
    x[static_cast<size_t>(row)] = sum / diag;
  }
  return true;
}

int64_t chord_weight(int64_t kappa2, int64_t delta) {
  __int128 den = static_cast<__int128>(kappa2) + static_cast<__int128>(delta) * delta;
  if (den < 1) den = 1;
  __int128 q = (static_cast<__int128>(kappa2) << 10) / den;
  if (q < 1) q = 1;
  return static_cast<int64_t>(q);
}

void append_u16(std::vector<uint8_t>& out, uint16_t value) {
  out.push_back(static_cast<uint8_t>(value & 255));
  out.push_back(static_cast<uint8_t>((value >> 8) & 255));
}

void append_u32(std::vector<uint8_t>& out, uint32_t value) {
  out.push_back(static_cast<uint8_t>(value & 255));
  out.push_back(static_cast<uint8_t>((value >> 8) & 255));
  out.push_back(static_cast<uint8_t>((value >> 16) & 255));
  out.push_back(static_cast<uint8_t>((value >> 24) & 255));
}

void append_bytes(std::vector<uint8_t>& out, const std::vector<uint8_t>& blob) {
  out.insert(out.end(), blob.begin(), blob.end());
}

struct Reader {
  const uint8_t* data = nullptr;
  size_t size = 0;
  size_t pos = 0;

  void need(size_t n, const char* what) const {
    if (n > size - pos) throw CodecError(what);
  }
  uint8_t u8(const char* what) {
    need(1, what);
    return data[pos++];
  }
  uint16_t u16(const char* what) {
    need(2, what);
    const uint16_t v = static_cast<uint16_t>(data[pos] | (uint16_t(data[pos + 1]) << 8));
    pos += 2;
    return v;
  }
  uint32_t u32(const char* what) {
    need(4, what);
    const uint32_t v = uint32_t(data[pos]) | (uint32_t(data[pos + 1]) << 8) | (uint32_t(data[pos + 2]) << 16) |
                       (uint32_t(data[pos + 3]) << 24);
    pos += 4;
    return v;
  }
  std::span<const uint8_t> take(size_t n, const char* what) {
    need(n, what);
    std::span<const uint8_t> view(data + pos, n);
    pos += n;
    return view;
  }
};

int argmin_sigma(double target) {
  int best = 0;
  double best_gap = std::fabs(static_cast<double>(SIGMA_Q8[0]) - target);
  for (int s = 1; s < NUM_SIGMAS; ++s) {
    const double gap = std::fabs(static_cast<double>(SIGMA_Q8[s]) - target);
    if (gap < best_gap) {
      best_gap = gap;
      best = s;
    }
  }
  return best;
}

double neg_log_prob(int freq) {
  static const std::vector<double> table = [] {
    std::vector<double> values(65536);
    for (int i = 0; i < 65536; ++i) values[static_cast<size_t>(i)] = -std::log2((i + 1) / 65536.0);
    return values;
  }();
  if (freq < 1) freq = 1;
  if (freq > 65536) freq = 65536;
  return table[static_cast<size_t>(freq - 1)];
}

int symbol_of(int64_t residual, int shift) {
  __int128 quot = residual;
  if (shift > 0) {
    const __int128 bias = __int128{1} << (shift - 1);
    quot = (static_cast<__int128>(residual) + bias) >> shift;
  }
  const __int128 mag = quot < 0 ? -quot : quot;
  if (mag > 32) return 65;
  return static_cast<int>(quot + 32);
}

double bucket_cost(const ParametricTables& tables, int sigma, const int64_t* residual, const int32_t* frac,
                   const std::vector<int>& which) {
  const int shift = tables.shifts[sigma];
  double cost = static_cast<double>(which.size()) * static_cast<double>(shift);
  for (int index : which) {
    const int sym = symbol_of(residual[index], shift);
    const int32_t freq = tables.freqs[1][sigma][frac[index]][sym];
    cost += neg_log_prob(freq);
    if (sym == 65) {
      __int128 quot = residual[index];
      if (shift > 0) quot = (static_cast<__int128>(residual[index]) + (__int128{1} << (shift - 1))) >> shift;
      const __int128 mag = quot < 0 ? -quot : quot;
      cost += 2.0 * std::log2(static_cast<double>(mag - 31)) + 3.0;
    }
  }
  return cost;
}

struct SubMeta {
  bool use_ridge = false;
  bool use_bridge = false;
  int nu = 1;
  std::vector<int64_t> weights;
  std::vector<int32_t> sigma;
};

void write_meta(BitWriter& writer, int n_pts, const SubMeta& meta) {
  writer.write_bits(meta.use_ridge ? 1u : 0u, 1);
  writer.write_bits(meta.use_bridge ? 1u : 0u, 1);
  writer.write_bits(static_cast<uint32_t>(meta.nu & 3), 2);
  if (meta.use_ridge) {
    for (int64_t weight : meta.weights) writer.write_signed_golomb(static_cast<int>(weight));
  }
  if (n_pts < 64) {
    writer.write_bits(static_cast<uint32_t>(meta.sigma[0] & 31), 5);
    return;
  }
  int prev = meta.sigma[0] & 31;
  writer.write_bits(static_cast<uint32_t>(prev), 5);
  for (int bucket = 1; bucket < kMaxBuckets; ++bucket) {
    const int cur = meta.sigma[static_cast<size_t>(bucket)] & 31;
    writer.write_signed_golomb(cur - prev);
    prev = cur;
  }
}

SubMeta read_meta(BitReader& reader, int n_pts, int n_feat) {
  SubMeta meta;
  meta.use_ridge = reader.read_bits(1) == 1;
  meta.use_bridge = reader.read_bits(1) == 1;
  meta.nu = static_cast<int>(reader.read_bits(2));
  if (meta.use_ridge) {
    meta.weights.resize(static_cast<size_t>(n_feat));
    for (int j = 0; j < n_feat; ++j) meta.weights[static_cast<size_t>(j)] = reader.read_signed_golomb();
  }
  meta.sigma.assign(kMaxBuckets, 0);
  if (n_pts < 64) {
    const int s0 = static_cast<int>(reader.read_bits(5));
    std::fill(meta.sigma.begin(), meta.sigma.end(), s0);
    return meta;
  }
  int prev = static_cast<int>(reader.read_bits(5));
  meta.sigma[0] = prev;
  for (int bucket = 1; bucket < kMaxBuckets; ++bucket) {
    const int cur = prev + reader.read_signed_golomb();
    if (cur < 0 || cur >= NUM_SIGMAS) throw CodecError("decoded sigma index out of range");
    meta.sigma[static_cast<size_t>(bucket)] = cur;
    prev = cur;
  }
  return meta;
}

std::vector<uint8_t> bridge_or_plain(const std::vector<int32_t>& residual, int nu, const std::vector<int32_t>& sig,
                                     const std::vector<int32_t>& frac, const std::vector<uint8_t>& bridge,
                                     const std::vector<int32_t>& lo, const std::vector<int32_t>& hi, bool& use_bridge) {
  use_bridge = false;
  auto plain = encode_parametric_rans(residual, nu, sig, frac);
  int count = 0;
  for (uint8_t bit : bridge) count += bit ? 1 : 0;
  if (count < 24) return plain;
  double zero_sum = 0.0;
  double mode_sum = 0.0;
  for (size_t i = 0; i < residual.size(); ++i) {
    if (!bridge[i]) continue;
    const int64_t rt = residual[i];
    const int64_t dz = iabs64(rt);
    const int64_t dlo = iabs64(rt - lo[i]);
    const int64_t dhi = iabs64(rt - hi[i]);
    zero_sum += static_cast<double>(dz);
    mode_sum += static_cast<double>(dlo < dhi ? dlo : dhi);
  }
  const double n = static_cast<double>(count);
  if (!(mode_sum / n + 1.5 < zero_sum / n)) return plain;
  auto mixed = encode_parametric_rans(residual, nu, sig, frac, bridge, lo, hi);
  if (mixed.size() < plain.size()) {
    use_bridge = true;
    return mixed;
  }
  return plain;
}

}  // namespace

LevelFeatures level_geometry_features(const Plane& spatial_even, const Mask& valid) {
  if (spatial_even.height != valid.height || spatial_even.width != valid.width) {
    throw CodecError("SLGF feature shape mismatch");
  }
  LevelFeatures feat;
  feat.height = spatial_even.height;
  feat.width = spatial_even.width;
  const int h = feat.height;
  const int w = feat.width;
  const size_t n = static_cast<size_t>(h) * static_cast<size_t>(w);
  if (h <= 0 || w <= 0) return feat;
  feat.mu0_q4.assign(n, 0);
  feat.f_sheaf_q4.assign(n, 0);
  feat.f_cub_v_q4.assign(n, 0);
  feat.f_cub_h_q4.assign(n, 0);
  feat.f_ring5_q4.assign(n, 0);
  feat.f_ssm_h_q4.assign(n, 0);
  feat.f_ssm_v_q4.assign(n, 0);
  feat.n_minus_s.assign(n, 0);
  feat.w_minus_e.assign(n, 0);
  feat.wv.assign(n, 1);
  feat.wh.assign(n, 1);
  feat.curv_q4.assign(n, 0);
  feat.lo_nb.assign(n, 0);
  feat.hi_nb.assign(n, 0);
  feat.is_bridge.assign(n, 0);
  feat.flat_pred.assign(n, 0);

  const ParityMasks par = parity_masks(valid);
  const Plane flat_plane = predict(spatial_even, valid, false);
  bool any_even = false;
  int32_t even_min = 0;
  int32_t even_max = 0;
  __int128 even_sum = 0;
  int64_t even_count = 0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (!par.even.at(y, x)) continue;
      const int32_t sample = spatial_even.at(y, x);
      if (!any_even) {
        even_min = even_max = sample;
        any_even = true;
      } else {
        even_min = std::min(even_min, sample);
        even_max = std::max(even_max, sample);
      }
      even_sum += sample;
      ++even_count;
    }
  }
  int64_t kappa2 = 144;
  if (any_even) {
    const int64_t span = static_cast<int64_t>(even_max) - static_cast<int64_t>(even_min);
    if (span > 512) {
      const int shift = std::max(0, bit_length_u(static_cast<uint64_t>(span)) - 9);
      const int64_t kappa = int64_t{12} << shift;
      kappa2 = kappa * kappa;
    }
  }
  const int64_t mean = even_count ? static_cast<int64_t>(floor_div(even_sum, even_count)) : 0;
  feat.act_shift = std::max(0, (bit_length_u(static_cast<uint64_t>(kappa2)) - 8) / 2);

  Plane filled = spatial_even;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const size_t i = lin(w, y, x);
      if (par.odd.at(y, x)) filled.samples[i] = flat_plane.samples[i];
      if (!valid.at(y, x)) filled.samples[i] = i32_wrap(mean);
      feat.flat_pred[i] = flat_plane.samples[i];
    }
  }
  const int edge_kappa = std::max(12, static_cast<int>(isqrt_int(kappa2)));
  const std::array<Plane, 4> scans = scan_4way(filled, 53248, edge_kappa);

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const size_t i = lin(w, y, x);
      const int64_t n1 = edge_at(filled, y - 1, x);
      const int64_t s1 = edge_at(filled, y + 1, x);
      const int64_t w1 = edge_at(filled, y, x - 1);
      const int64_t e1 = edge_at(filled, y, x + 1);
      const bool vn = mask_at(par.even, y - 1, x);
      const bool vs = mask_at(par.even, y + 1, x);
      const bool vw = mask_at(par.even, y, x - 1);
      const bool ve = mask_at(par.even, y, x + 1);
      const int64_t flat = feat.flat_pred[i];
      const int64_t n_eff = vn ? n1 : (vs ? s1 : flat);
      const int64_t s_eff = vs ? s1 : (vn ? n1 : flat);
      const int64_t w_eff = vw ? w1 : (ve ? e1 : flat);
      const int64_t e_eff = ve ? e1 : (vw ? w1 : flat);
      int64_t acc = 0;
      int cnt = 0;
      if (vn) { acc += n1; ++cnt; }
      if (vs) { acc += s1; ++cnt; }
      if (vw) { acc += w1; ++cnt; }
      if (ve) { acc += e1; ++cnt; }
      const int64_t mean_q4 = cnt ? static_cast<int64_t>(floor_div(static_cast<__int128>(acc) * 16, cnt)) : 0;
      const int64_t m0 = 4 * (n_eff + s_eff + w_eff + e_eff);
      const int64_t dv = iabs64(n_eff - s_eff);
      const int64_t dh = iabs64(w_eff - e_eff);
      const int64_t wv = chord_weight(kappa2, dv);
      const int64_t wh = chord_weight(kappa2, dh);
      const __int128 dir_num =
          __int128{8} * (static_cast<__int128>(wv) * (n_eff + s_eff) + static_cast<__int128>(wh) * (w_eff + e_eff));
      const int64_t dir = static_cast<int64_t>(floor_div(dir_num, static_cast<__int128>(wv) + wh));
      const int64_t n3 = edge_at(filled, y - 3, x);
      const int64_t s3 = edge_at(filled, y + 3, x);
      const int64_t w3 = edge_at(filled, y, x - 3);
      const int64_t e3 = edge_at(filled, y, x + 3);
      int64_t ring = 0;
      ring += edge_at(filled, y - 2, x - 1);
      ring += edge_at(filled, y - 2, x + 1);
      ring += edge_at(filled, y + 2, x - 1);
      ring += edge_at(filled, y + 2, x + 1);
      ring += edge_at(filled, y - 1, x - 2);
      ring += edge_at(filled, y + 1, x - 2);
      ring += edge_at(filled, y - 1, x + 2);
      ring += edge_at(filled, y + 1, x + 2);
      const int64_t f_ring = m0 - 2 * ring;
      const int64_t f_ssm_h = ((edge_at(scans[0], y, x - 1) + edge_at(scans[1], y, x + 1)) >> 1) - m0;
      const int64_t f_ssm_v = ((edge_at(scans[2], y - 1, x) + edge_at(scans[3], y + 1, x)) >> 1) - m0;
      const int64_t saddle = iabs64((n_eff + s_eff) - (w_eff + e_eff));
      const int64_t min_chord = dv < dh ? dv : dh;
      int64_t lo = n_eff;
      int64_t hi = n_eff;
      for (int64_t sample : {s_eff, w_eff, e_eff}) {
        lo = std::min(lo, sample);
        hi = std::max(hi, sample);
      }
      feat.mu0_q4[i] = mean_q4 - 16 * flat;
      feat.f_sheaf_q4[i] = dir - m0;
      feat.f_cub_v_q4[i] = 8 * (n_eff + s_eff) - 8 * (n3 + s3);
      feat.f_cub_h_q4[i] = 8 * (w_eff + e_eff) - 8 * (w3 + e3);
      feat.f_ring5_q4[i] = f_ring;
      feat.f_ssm_h_q4[i] = f_ssm_h;
      feat.f_ssm_v_q4[i] = f_ssm_v;
      feat.n_minus_s[i] = n_eff - s_eff;
      feat.w_minus_e[i] = w_eff - e_eff;
      feat.wv[i] = wv;
      feat.wh[i] = wh;
      feat.curv_q4[i] = 8 * saddle + 6 * min_chord + iabs64(f_ring);
      feat.lo_nb[i] = lo - flat;
      feat.hi_nb[i] = hi - flat;
      feat.is_bridge[i] = (min_chord >= 20 && cnt == 4) ? 1 : 0;
    }
  }
  return feat;
}

std::vector<Mask> subcoset_masks(const Mask& valid) {
  const ParityMasks par = parity_masks(valid);
  int n_odd = 0;
  for (uint8_t bit : par.odd.bits) n_odd += bit ? 1 : 0;
  if (n_odd == 0) return {};
  if (n_odd < 96) return {par.odd};
  Mask c1(valid.height, valid.width, false);
  Mask c2(valid.height, valid.width, false);
  Mask c3(valid.height, valid.width, false);
  bool any1 = false;
  bool any2 = false;
  bool any3 = false;
  for (int y = 0; y < valid.height; ++y) {
    for (int x = 0; x < valid.width; ++x) {
      if (!par.odd.at(y, x)) continue;
      if ((y & 1) == 0) {
        if (((((y >> 1) + (x >> 1)) & 1) == 0)) {
          c1.at(y, x) = 1;
          any1 = true;
        } else {
          c2.at(y, x) = 1;
          any2 = true;
        }
      } else {
        c3.at(y, x) = 1;
        any3 = true;
      }
    }
  }
  std::vector<Mask> masks;
  if (any1) masks.push_back(std::move(c1));
  if (any2) masks.push_back(std::move(c2));
  if (any3) masks.push_back(std::move(c3));
  return masks;
}

FeatureRows assemble_features_and_activity(const LevelFeatures& base, int sub_idx, const std::vector<int>& sites,
                                           const std::vector<int64_t>& decoded_odd,
                                           const std::vector<int64_t>* coarse_proj, const Plane* ch0_spatial,
                                           const Plane* ch0_resid, const Plane* ch1_resid) {
  FeatureRows out;
  const int h = base.height;
  const int w = base.width;
  const int n = static_cast<int>(sites.size());
  out.rows = n;
  out.act_q4.assign(static_cast<size_t>(n), 1);
  if (n == 0) return out;
  auto at_odd = [&](int y, int x) -> int64_t {
    if (y < 0 || x < 0 || y >= h || x >= w) return 0;
    return decoded_odd[lin(w, y, x)];
  };
  const bool use_chroma = ch0_spatial != nullptr && ch0_resid != nullptr;
  if (use_chroma && (ch0_spatial->height != h || ch0_spatial->width != w || ch0_resid->height != h ||
                     ch0_resid->width != w)) {
    throw CodecError("SLGF cross-channel feature shape mismatch");
  }
  if (ch1_resid != nullptr && (ch1_resid->height != h || ch1_resid->width != w)) {
    throw CodecError("SLGF cross-channel feature shape mismatch");
  }
  std::vector<int64_t> packed;
  int cols = 0;
  for (int row = 0; row < n; ++row) {
    const int site = sites[static_cast<size_t>(row)];
    const int y = site / w;
    const int x = site - y * w;
    std::vector<int64_t> feat;
    feat.push_back(base.f_sheaf_q4[static_cast<size_t>(site)]);
    feat.push_back(base.f_cub_v_q4[static_cast<size_t>(site)]);
    feat.push_back(base.f_cub_h_q4[static_cast<size_t>(site)]);
    feat.push_back(base.f_ring5_q4[static_cast<size_t>(site)]);
    feat.push_back(base.f_ssm_h_q4[static_cast<size_t>(site)]);
    feat.push_back(base.f_ssm_v_q4[static_cast<size_t>(site)]);
    int64_t nb_mag = 0;
    bool have_nb = false;
    if (sub_idx == 1) {
      const int64_t rv1 = at_odd(y - 2, x);
      const int64_t rv2 = at_odd(y + 2, x);
      const int64_t rh1 = at_odd(y, x - 2);
      const int64_t rh2 = at_odd(y, x + 2);
      feat.push_back(4 * (rv1 + rv2 + rh1 + rh2));
      feat.push_back(8 * ((rv1 + rv2) - (rh1 + rh2)));
      nb_mag = 4 * (iabs64(rv1) + iabs64(rv2) + iabs64(rh1) + iabs64(rh2));
      have_nb = true;
    } else if (sub_idx == 2) {
      const int64_t rnw = at_odd(y - 1, x - 1);
      const int64_t rne = at_odd(y - 1, x + 1);
      const int64_t rsw = at_odd(y + 1, x - 1);
      const int64_t rse = at_odd(y + 1, x + 1);
      const int64_t sum4 = rnw + rne + rsw + rse;
      const int64_t wv = base.wv[static_cast<size_t>(site)];
      const int64_t wh = base.wh[static_cast<size_t>(site)];
      feat.push_back(4 * sum4);
      feat.push_back(8 * ((rnw + rse) - (rne + rsw)));
      feat.push_back(static_cast<int64_t>(
          floor_div(__int128{4} * (wv - wh) * sum4, static_cast<__int128>(wv) + wh)));
      nb_mag = 4 * (iabs64(rnw) + iabs64(rne) + iabs64(rsw) + iabs64(rse));
      have_nb = true;
    }
    if (coarse_proj != nullptr) feat.push_back(8 * (*coarse_proj)[static_cast<size_t>(site)]);
    if (use_chroma) {
      const int64_t par_v = ((y & 2) == 0) ? 1 : -1;
      const int64_t par_h = ((x & 2) == 0) ? 1 : -1;
      const int64_t north = edge_at(*ch0_spatial, y - 1, x);
      const int64_t south = edge_at(*ch0_spatial, y + 1, x);
      const int64_t west = edge_at(*ch0_spatial, y, x - 1);
      const int64_t east = edge_at(*ch0_spatial, y, x + 1);
      const int64_t center = ch0_spatial->samples[static_cast<size_t>(site)];
      const int64_t lap = 4 * center - north - south - west - east;
      const int64_t r0 = ch0_resid->samples[static_cast<size_t>(site)];
      feat.push_back(16 * r0);
      feat.push_back(4 * lap);
      feat.push_back(8 * base.n_minus_s[static_cast<size_t>(site)] * par_v);
      feat.push_back(8 * base.w_minus_e[static_cast<size_t>(site)] * par_h);
      feat.push_back(8 * (north - south) * par_v);
      feat.push_back(8 * (west - east) * par_h);
      if (ch1_resid != nullptr) feat.push_back(16 * ch1_resid->samples[static_cast<size_t>(site)]);
    }
    if (row == 0) {
      cols = static_cast<int>(feat.size());
      packed.assign(static_cast<size_t>(n) * static_cast<size_t>(cols), 0);
    }
    for (int c = 0; c < cols; ++c) packed[static_cast<size_t>(row * cols + c)] = feat[static_cast<size_t>(c)];
    const int64_t curv = base.curv_q4[static_cast<size_t>(site)];
    int64_t act = 0;
    if (have_nb) act = ((3 * curv) >> 3) + ((7 * nb_mag) >> 3);
    else if (coarse_proj != nullptr) act = ((5 * curv) >> 3) + 6 * iabs64((*coarse_proj)[static_cast<size_t>(site)]);
    else act = (curv * 5) >> 3;
    if (use_chroma) {
      const int64_t r0_abs = 16 * iabs64(ch0_resid->samples[static_cast<size_t>(site)]);
      act = ((5 * act) >> 3) + ((2 * r0_abs) >> 3);
    }
    out.act_q4[static_cast<size_t>(row)] = std::max<int64_t>(act, 1);
  }
  out.cols = cols;
  out.phi = std::move(packed);
  return out;
}

void split_mu_q4(const int64_t* mu_q4, size_t n, int64_t* mu_int, int32_t* frac_idx) {
  for (size_t i = 0; i < n; ++i) {
    const int64_t mu = mu_q4[i];
    const int64_t rounded = (mu + 8) >> 4;
    int64_t frac = mu - (rounded << 4) + 8;
    if (frac < 0) frac = 0;
    if (frac > 15) frac = 15;
    mu_int[i] = rounded;
    frac_idx[i] = static_cast<int32_t>(frac);
  }
}

void bucket_from_act_q4(const int64_t* act_q4, size_t n, int act_shift, int32_t* buckets) {
  std::vector<int64_t> values(n);
  const int64_t cap = int64_t{1} << 28;
  for (size_t i = 0; i < n; ++i) {
    int64_t scaled = act_q4[i] >> act_shift;
    if (scaled < 1) scaled = 1;
    if (scaled > cap) scaled = cap;
    values[i] = (scaled * scaled) >> 4;
  }
  const std::vector<int32_t> logs = ilog2_int64(values);
  for (size_t i = 0; i < n; ++i) {
    int bucket = logs[i];
    if (bucket < 0) bucket = 0;
    if (bucket > kMaxBuckets - 1) bucket = kMaxBuckets - 1;
    buckets[i] = bucket;
  }
}

std::optional<std::vector<int64_t>> coarse_resid_projection(const Plane* prev_resid, const PackInfo& info,
                                                           const Mask& valid) {
  if (prev_resid == nullptr) return std::nullopt;
  const Plane unpacked = unpack_even(*prev_resid, info, valid);
  const ParityMasks par = parity_masks(valid);
  const int h = valid.height;
  const int w = valid.width;
  std::vector<int64_t> out(static_cast<size_t>(h) * static_cast<size_t>(w), 0);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int64_t count = 0;
      int64_t acc = 0;
      const int ny[4] = {y - 1, y + 1, y, y};
      const int nx[4] = {x, x, x - 1, x + 1};
      for (int k = 0; k < 4; ++k) {
        if (!mask_at(par.even, ny[k], nx[k])) continue;
        ++count;
        acc += iabs64(unpacked.at(ny[k], nx[k]));
      }
      if (count) out[lin(w, y, x)] = acc / count;
    }
  }
  return out;
}

SigmaNuFit select_sigma_nu(const int64_t* r_tilde, const int32_t* frac_idx, const int32_t* buckets, size_t n,
                           int n_buckets, int early_below) {
  if (n_buckets <= 0) throw CodecError("sigma bucket count is empty");
  SigmaNuFit fit;
  fit.sig_map.assign(static_cast<size_t>(n_buckets), 0);
  auto abs_err = [&](size_t i) {
    const __int128 value = (static_cast<__int128>(r_tilde[i]) << 4) - (frac_idx[i] - 8);
    __int128 mag = value < 0 ? -value : value;
    mag <<= 4;
    return static_cast<double>(mag);
  };
  if (n < static_cast<size_t>(early_below)) {
    double mad = 64.0;
    if (n > 0) {
      double sum = 0.0;
      for (size_t i = 0; i < n; ++i) sum += abs_err(i);
      mad = (sum / static_cast<double>(n)) * 0.92 + 24.0;
    }
    const int s0 = argmin_sigma(mad);
    std::fill(fit.sig_map.begin(), fit.sig_map.end(), s0);
    fit.nu_idx = 1;
    return fit;
  }
  const ParametricTables& tables = get_parametric_tables();
  int prev = 4;
  for (int bucket = 0; bucket < n_buckets; ++bucket) {
    std::vector<int> which;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
      if (buckets[i] != bucket) continue;
      which.push_back(static_cast<int>(i));
      sum += abs_err(i);
    }
    if (static_cast<int>(which.size()) < 8) {
      fit.sig_map[static_cast<size_t>(bucket)] = prev;
      continue;
    }
    const double est = (sum / static_cast<double>(which.size())) * 0.92 + 20.0;
    const int guess = argmin_sigma(est);
    int best_s = guess;
    double best_cost = std::numeric_limits<double>::infinity();
    const int begin = std::max(0, guess - 2);
    const int end = std::min(NUM_SIGMAS, guess + 3);
    for (int cand = begin; cand < end; ++cand) {
      const double cost = bucket_cost(tables, cand, r_tilde, frac_idx, which);
      if (cost < best_cost) {
        best_cost = cost;
        best_s = cand;
      }
    }
    fit.sig_map[static_cast<size_t>(bucket)] = best_s;
    prev = best_s;
  }
  std::vector<int> symbols(n);
  std::vector<int> chosen(n);
  for (size_t i = 0; i < n; ++i) {
    const int bucket = buckets[i];
    const int sigma = fit.sig_map[static_cast<size_t>(bucket)];
    chosen[i] = sigma;
    symbols[i] = symbol_of(r_tilde[i], tables.shifts[sigma]);
  }
  int best_nu = 0;
  double best_nu_cost = std::numeric_limits<double>::infinity();
  for (int nu = 0; nu < NUM_NUS; ++nu) {
    double cost = 0.0;
    for (size_t i = 0; i < n; ++i) {
      const int32_t freq = tables.freqs[nu][chosen[i]][frac_idx[i]][symbols[i]];
      cost += neg_log_prob(freq);
    }
    if (cost < best_nu_cost) {
      best_nu_cost = cost;
      best_nu = nu;
    }
  }
  fit.nu_idx = best_nu;
  return fit;
}

SubcosetEstimate estimate_subcoset_model(const int32_t* r_target, const int64_t* mu0_q4, const int64_t* phi, int rows,
                                         int cols, const int32_t* buckets) {
  SubcosetEstimate model;
  model.w_q8.assign(static_cast<size_t>(std::max(cols, 0)), 0);
  model.sig_map.assign(kMaxBuckets, 0);
  if (rows < 0 || cols < 0) throw CodecError("SLGF sub-coset shape is invalid");
  std::vector<int64_t> mu(static_cast<size_t>(std::max(rows, 0)));
  for (int i = 0; i < rows; ++i) mu[static_cast<size_t>(i)] = mu0_q4[i];
  if (rows >= 128 && cols > 0) {
    std::vector<double> energy(static_cast<size_t>(cols), 0.0);
    std::vector<double> gram(static_cast<size_t>(cols * cols), 0.0);
    std::vector<double> rhs(static_cast<size_t>(cols), 0.0);
    for (int row = 0; row < rows; ++row) {
      const int64_t* src = phi + static_cast<size_t>(row * cols);
      const double y = 16.0 * static_cast<double>(r_target[row]) - static_cast<double>(mu0_q4[row]);
      for (int j = 0; j < cols; ++j) {
        const double xj = static_cast<double>(src[j]);
        energy[static_cast<size_t>(j)] += xj * xj;
        rhs[static_cast<size_t>(j)] += xj * y;
        for (int k = 0; k < cols; ++k) {
          gram[static_cast<size_t>(j * cols + k)] += xj * static_cast<double>(src[k]);
        }
      }
    }
    for (int j = 0; j < cols; ++j) {
      gram[static_cast<size_t>(j * cols + j)] += std::max(energy[static_cast<size_t>(j)] * 1e-4, 64.0);
    }
    std::vector<double> solved;
    if (solve_ge(gram, rhs, cols, solved)) {
      std::vector<int64_t> cand(static_cast<size_t>(cols));
      bool nonzero = false;
      for (int j = 0; j < cols; ++j) {
        int64_t weight = round_half_even(solved[static_cast<size_t>(j)] * 256.0);
        if (weight > kWeightMax) weight = kWeightMax;
        if (weight < -kWeightMax) weight = -kWeightMax;
        cand[static_cast<size_t>(j)] = weight;
        if (weight != 0) nonzero = true;
      }
      if (nonzero) {
        double sum0 = 0.0;
        double sum1 = 0.0;
        for (int row = 0; row < rows; ++row) {
          const int64_t corr = dot_q8(phi + static_cast<size_t>(row * cols), cols, cand.data());
          const int64_t mu1 = mu0_q4[row] + corr;
          const int64_t err0 = iabs64(16 * static_cast<int64_t>(r_target[row]) - mu0_q4[row]);
          const int64_t err1 = iabs64(16 * static_cast<int64_t>(r_target[row]) - mu1);
          sum0 += static_cast<double>(err0);
          sum1 += static_cast<double>(err1);
        }
        int header = 0;
        for (int64_t weight : cand) header += golomb_bits(weight);
        const double nd = static_cast<double>(rows);
        const double gain = nd * (std::log2(sum0 / nd + 4.0) - std::log2(sum1 / nd + 4.0));
        if (gain > static_cast<double>(header) + 8.0) {
          model.use_ridge = true;
          model.w_q8 = std::move(cand);
          for (int row = 0; row < rows; ++row) {
            mu[static_cast<size_t>(row)] =
                mu0_q4[row] + dot_q8(phi + static_cast<size_t>(row * cols), cols, model.w_q8.data());
          }
        }
      }
    }
  }
  std::vector<int64_t> mu_int(static_cast<size_t>(std::max(rows, 0)));
  std::vector<int32_t> frac(static_cast<size_t>(std::max(rows, 0)));
  std::vector<int64_t> residual(static_cast<size_t>(std::max(rows, 0)));
  if (rows > 0) {
    split_mu_q4(mu.data(), static_cast<size_t>(rows), mu_int.data(), frac.data());
    for (int i = 0; i < rows; ++i) residual[static_cast<size_t>(i)] = static_cast<int64_t>(r_target[i]) - mu_int[static_cast<size_t>(i)];
  }
  const SigmaNuFit picked = select_sigma_nu(residual.data(), frac.data(), buckets, static_cast<size_t>(std::max(rows, 0)),
                                             kMaxBuckets, 64);
  model.nu_idx = picked.nu_idx;
  model.sig_map = picked.sig_map;
  return model;
}

namespace {

void check_planes(const std::vector<Plane>& planes) {
  if (planes.empty()) throw CodecError("empty SLGF tile");
  const int h = planes[0].height;
  const int w = planes[0].width;
  if (h <= 0 || w <= 0) throw CodecError("empty image");
  for (const Plane& plane : planes) {
    if (plane.height != h || plane.width != w) throw CodecError("SLGF planes disagree");
  }
}

}  // namespace

std::vector<uint8_t> encode_slgf_tile(const std::vector<Plane>& planes, int max_levels) {
  check_planes(planes);
  const int height = planes[0].height;
  const int width = planes[0].width;
  const auto built = build_levels(height, width, max_levels);
  const std::vector<Level>& geom = built.first;
  const int nlevels = static_cast<int>(geom.size());
  std::vector<uint8_t> out;
  out.push_back(static_cast<uint8_t>(nlevels));
  if (nlevels == 0) {
    const Mask mask(height, width, true);
    for (const Plane& plane : planes) {
      const std::vector<uint8_t> blob = encode_raster(plane, mask, true);
      append_u32(out, static_cast<uint32_t>(blob.size()));
      append_bytes(out, blob);
    }
    return out;
  }
  std::vector<Tower> towers;
  towers.reserve(planes.size());
  for (const Plane& plane : planes) towers.push_back(forward_tower(plane, nlevels, false));
  std::vector<std::vector<Plane>> ch_recons;
  std::vector<std::vector<Plane>> ch_resids;
  for (size_t ch = 0; ch < planes.size(); ++ch) {
    const Tower& tower = towers[ch];
    if (static_cast<int>(tower.details.size()) != nlevels || static_cast<int>(tower.levels.size()) != nlevels) {
      throw CodecError("SLGF tower depth mismatch");
    }
    const std::vector<uint8_t> med = encode_raster(tower.coarse, built.second, false);
    const std::vector<uint8_t> kal = encode_raster(tower.coarse, built.second, true);
    const bool kalman = kal.size() < med.size();
    const std::vector<uint8_t>& coarse_blob = kalman ? kal : med;
    out.push_back(kalman ? 1 : 0);
    append_u32(out, static_cast<uint32_t>(coarse_blob.size()));
    append_bytes(out, coarse_blob);

    Plane current = tower.coarse;
    std::vector<Plane> level_recons(static_cast<size_t>(nlevels), Plane(1, 1));
    std::vector<Plane> level_resids(static_cast<size_t>(nlevels), Plane(1, 1));
    const Plane* prev = nullptr;
    for (int level = nlevels - 1; level >= 0; --level) {
      const Level& spec = geom[static_cast<size_t>(level)];
      const Plane& lifted = tower.details[static_cast<size_t>(level)];
      Plane spatial = unpack_even(current, spec.info, spec.valid);
      const ParityMasks par = parity_masks(spec.valid);
      const LevelFeatures base = level_geometry_features(spatial, spec.valid);
      const auto coarse = coarse_resid_projection(prev, spec.info, spec.valid);
      const Plane* ch0_sp = ch >= 1 ? &ch_recons[0][static_cast<size_t>(level)] : nullptr;
      const Plane* ch0_rs = ch >= 1 ? &ch_resids[0][static_cast<size_t>(level)] : nullptr;
      const Plane* ch1_rs = ch >= 2 ? &ch_resids[1][static_cast<size_t>(level)] : nullptr;
      const std::vector<Mask> subs = subcoset_masks(spec.valid);
      std::vector<int64_t> decoded(static_cast<size_t>(spatial.height) * static_cast<size_t>(spatial.width), 0);
      BitWriter meta_writer;
      std::vector<std::vector<uint8_t>> payloads;
      for (size_t sub = 0; sub < subs.size(); ++sub) {
        const std::vector<int> sites = true_sites(subs[sub]);
        const FeatureRows rows = assemble_features_and_activity(base, static_cast<int>(sub), sites, decoded,
                                                                coarse ? &*coarse : nullptr, ch0_sp, ch0_rs, ch1_rs);
        std::vector<int32_t> bucket(sites.size());
        std::vector<int64_t> mu0(sites.size());
        std::vector<int32_t> target(sites.size());
        for (size_t i = 0; i < sites.size(); ++i) {
          mu0[i] = base.mu0_q4[static_cast<size_t>(sites[i])];
          target[i] = lifted.samples[static_cast<size_t>(sites[i])];
        }
        bucket_from_act_q4(rows.act_q4.data(), rows.act_q4.size(), base.act_shift, bucket.data());
        const SubcosetEstimate model =
            estimate_subcoset_model(target.data(), mu0.data(), rows.phi.data(), rows.rows, rows.cols, bucket.data());
        std::vector<int64_t> mu = mu0;
        if (model.use_ridge) {
          for (int i = 0; i < rows.rows; ++i) {
            mu[static_cast<size_t>(i)] =
                mu0[static_cast<size_t>(i)] + dot_q8(rows.phi.data() + static_cast<size_t>(i * rows.cols), rows.cols,
                                                     model.w_q8.data());
          }
        }
        std::vector<int64_t> mu_int(sites.size());
        std::vector<int32_t> frac(sites.size());
        split_mu_q4(mu.data(), mu.size(), mu_int.data(), frac.data());
        std::vector<int32_t> residual(sites.size());
        std::vector<int32_t> sig(sites.size());
        for (size_t i = 0; i < sites.size(); ++i) {
          residual[i] = i32_wrap(static_cast<int64_t>(target[i]) - mu_int[i]);
          sig[i] = model.sig_map[static_cast<size_t>(bucket[i])];
        }
        const ParametricTables& tables = get_parametric_tables();
        std::vector<uint8_t> bridge(sites.size(), 0);
        std::vector<int32_t> lo(sites.size());
        std::vector<int32_t> hi(sites.size());
        for (size_t i = 0; i < sites.size(); ++i) {
          const int site = sites[i];
          const bool shift0 = tables.shifts[sig[i]] == 0;
          bridge[i] = (base.is_bridge[static_cast<size_t>(site)] && shift0) ? 1 : 0;
          lo[i] = i32_wrap(base.lo_nb[static_cast<size_t>(site)] - mu_int[i]);
          hi[i] = i32_wrap(base.hi_nb[static_cast<size_t>(site)] - mu_int[i]);
        }
        bool use_bridge = false;
        const std::vector<uint8_t> chosen =
            bridge_or_plain(residual, model.nu_idx, sig, frac, bridge, lo, hi, use_bridge);
        SubMeta meta;
        meta.use_ridge = model.use_ridge;
        meta.use_bridge = use_bridge;
        meta.nu = model.nu_idx;
        meta.weights = model.w_q8;
        meta.sigma = model.sig_map;
        write_meta(meta_writer, static_cast<int>(sites.size()), meta);
        payloads.push_back(chosen);
        for (size_t i = 0; i < sites.size(); ++i) decoded[static_cast<size_t>(sites[i])] = target[i];
      }
      std::vector<uint8_t> meta_bytes = meta_writer.finish();
      if (meta_bytes.size() > 65535) throw CodecError("SLGF metadata exceeds 65535 bytes");
      append_u16(out, static_cast<uint16_t>(meta_bytes.size()));
      append_bytes(out, meta_bytes);
      for (const auto& payload : payloads) {
        append_u32(out, static_cast<uint32_t>(payload.size()));
        append_bytes(out, payload);
      }
      Plane full = spatial;
      Plane resid(spatial.height, spatial.width);
      for (size_t i = 0; i < par.odd.bits.size(); ++i) {
        if (!par.odd.bits[i]) continue;
        const int32_t sample = i32_wrap(decoded[i]);
        full.samples[i] = sample;
        resid.samples[i] = sample;
      }
      current = lift_inverse(full, spec.valid, false);
      level_recons[static_cast<size_t>(level)] = current;
      level_resids[static_cast<size_t>(level)] = std::move(resid);
      prev = &level_resids[static_cast<size_t>(level)];
    }
    ch_recons.push_back(std::move(level_recons));
    ch_resids.push_back(std::move(level_resids));
  }
  return out;
}

std::vector<Plane> decode_slgf_tile(const uint8_t* data, size_t size, int height, int width, int channels) {
  if (data == nullptr && size != 0) throw CodecError("truncated SLGF tile");
  if (size < 1) throw CodecError("truncated SLGF tile");
  if (channels < 1) throw CodecError("unsupported sample format");
  Reader reader{data, size, 0};
  const int nlevels = reader.u8("truncated SLGF tile");
  if (nlevels == 0) {
    std::vector<Plane> planes;
    const Mask mask(height, width, true);
    for (int ch = 0; ch < channels; ++ch) {
      const uint32_t blen = reader.u32("truncated level-0 SLGF plane");
      const auto blob = reader.take(blen, "truncated level-0 SLGF data");
      planes.push_back(decode_raster(blob.data(), blob.size(), height, width, mask, true));
    }
    if (reader.pos != reader.size) throw CodecError("trailing bytes in level-0 SLGF tile");
    return planes;
  }
  const auto built = build_levels(height, width, nlevels);
  if (static_cast<int>(built.first.size()) != nlevels) throw CodecError("SLGF quincunx depth mismatch");
  std::vector<std::vector<Plane>> ch_recons;
  std::vector<std::vector<Plane>> ch_resids;
  std::vector<Plane> out;
  for (int ch = 0; ch < channels; ++ch) {
    if (reader.pos + 5 > reader.size) throw CodecError("truncated SLGF coarse header");
    const int c_flag = reader.u8("truncated SLGF coarse header");
    const uint32_t c_len = reader.u32("truncated SLGF coarse header");
    const auto coarse_blob = reader.take(c_len, "truncated SLGF coarse payload");
    Plane current = decode_raster(coarse_blob.data(), coarse_blob.size(), built.second.height, built.second.width,
                                  built.second, c_flag == 1);
    std::vector<Plane> level_recons(static_cast<size_t>(nlevels), Plane(1, 1));
    std::vector<Plane> level_resids(static_cast<size_t>(nlevels), Plane(1, 1));
    const Plane* prev = nullptr;
    for (int level = nlevels - 1; level >= 0; --level) {
      const Level& spec = built.first[static_cast<size_t>(level)];
      Plane spatial = unpack_even(current, spec.info, spec.valid);
      const ParityMasks par = parity_masks(spec.valid);
      const LevelFeatures base = level_geometry_features(spatial, spec.valid);
      const auto coarse = coarse_resid_projection(prev, spec.info, spec.valid);
      const Plane* ch0_sp = ch >= 1 ? &ch_recons[0][static_cast<size_t>(level)] : nullptr;
      const Plane* ch0_rs = ch >= 1 ? &ch_resids[0][static_cast<size_t>(level)] : nullptr;
      const Plane* ch1_rs = ch >= 2 ? &ch_resids[1][static_cast<size_t>(level)] : nullptr;
      const uint16_t meta_len = reader.u16("truncated SLGF level metadata length");
      const auto meta_bytes = reader.take(meta_len, "truncated SLGF level metadata");
      BitReader meta_reader(meta_bytes.data(), meta_bytes.size());
      const std::vector<Mask> subs = subcoset_masks(spec.valid);
      std::vector<int64_t> decoded(static_cast<size_t>(spatial.height) * static_cast<size_t>(spatial.width), 0);
      for (size_t sub = 0; sub < subs.size(); ++sub) {
        const std::vector<int> sites = true_sites(subs[sub]);
        const FeatureRows rows = assemble_features_and_activity(base, static_cast<int>(sub), sites, decoded,
                                                                coarse ? &*coarse : nullptr, ch0_sp, ch0_rs, ch1_rs);
        std::vector<int64_t> mu0(sites.size());
        for (size_t i = 0; i < sites.size(); ++i) mu0[i] = base.mu0_q4[static_cast<size_t>(sites[i])];
        const SubMeta meta = read_meta(meta_reader, static_cast<int>(sites.size()), rows.cols);
        std::vector<int64_t> mu = mu0;
        if (meta.use_ridge) {
          if (static_cast<int>(meta.weights.size()) != rows.cols) throw CodecError("SLGF ridge weight count mismatch");
          for (int i = 0; i < rows.rows; ++i) {
            mu[static_cast<size_t>(i)] =
                mu0[static_cast<size_t>(i)] +
                dot_q8(rows.phi.data() + static_cast<size_t>(i * rows.cols), rows.cols, meta.weights.data());
          }
        }
        std::vector<int64_t> mu_int(sites.size());
        std::vector<int32_t> frac(sites.size());
        split_mu_q4(mu.data(), mu.size(), mu_int.data(), frac.data());
        std::vector<int32_t> bucket(sites.size());
        bucket_from_act_q4(rows.act_q4.data(), rows.act_q4.size(), base.act_shift, bucket.data());
        std::vector<int32_t> sig(sites.size());
        for (size_t i = 0; i < sites.size(); ++i) sig[i] = meta.sigma[static_cast<size_t>(bucket[i])];
        const uint32_t sub_len = reader.u32("truncated SLGF sub-coset length");
        const auto sub_blob = reader.take(sub_len, "truncated SLGF sub-coset stream");
        std::vector<int32_t> residual;
        if (meta.use_bridge) {
          const ParametricTables& tables = get_parametric_tables();
          std::vector<uint8_t> bridge(sites.size(), 0);
          std::vector<int32_t> lo(sites.size());
          std::vector<int32_t> hi(sites.size());
          for (size_t i = 0; i < sites.size(); ++i) {
            const int site = sites[i];
            bridge[i] = (base.is_bridge[static_cast<size_t>(site)] && tables.shifts[sig[i]] == 0) ? 1 : 0;
            lo[i] = i32_wrap(base.lo_nb[static_cast<size_t>(site)] - mu_int[i]);
            hi[i] = i32_wrap(base.hi_nb[static_cast<size_t>(site)] - mu_int[i]);
          }
          residual = decode_parametric_rans(sub_blob, static_cast<int>(sites.size()), meta.nu, sig, frac, bridge, lo, hi);
        } else {
          residual = decode_parametric_rans(sub_blob, static_cast<int>(sites.size()), meta.nu, sig, frac);
        }
        if (residual.size() != sites.size()) throw CodecError("SLGF sub-coset length mismatch");
        for (size_t i = 0; i < sites.size(); ++i) {
          decoded[static_cast<size_t>(sites[i])] = static_cast<int64_t>(residual[i]) + mu_int[i];
        }
      }
      Plane full = spatial;
      Plane resid(spatial.height, spatial.width);
      for (size_t i = 0; i < par.odd.bits.size(); ++i) {
        if (!par.odd.bits[i]) continue;
        const int32_t sample = i32_wrap(decoded[i]);
        full.samples[i] = sample;
        resid.samples[i] = sample;
      }
      current = lift_inverse(full, spec.valid, false);
      level_recons[static_cast<size_t>(level)] = current;
      level_resids[static_cast<size_t>(level)] = std::move(resid);
      prev = &level_resids[static_cast<size_t>(level)];
    }
    ch_recons.push_back(std::move(level_recons));
    ch_resids.push_back(std::move(level_resids));
    out.push_back(std::move(current));
  }
  if (reader.pos != reader.size) throw CodecError("trailing bytes in SLGF tile");
  return out;
}

}  // namespace vpxc
