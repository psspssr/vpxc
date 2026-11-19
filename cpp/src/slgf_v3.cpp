#include "vpxc/slgf_v3.hpp"

#include "vpxc/bitio.hpp"
#include "vpxc/mdl.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace vpxc {
namespace {

constexpr int kD8 = 2;
constexpr int kHead = 4;
constexpr int kIssm = 8;

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

size_t lin(int width, int y, int x) { return static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x); }

int64_t dot_q8(const int64_t* row, int cols, const int64_t* weights) {
  __int128 acc = 0;
  for (int j = 0; j < cols; ++j) acc += static_cast<__int128>(row[j]) * weights[j];
  acc += 128;
  return static_cast<int64_t>(acc >> 8);
}

__int128 l1_q4(const int64_t* target, const int64_t* mu, int n) {
  __int128 sum = 0;
  for (int i = 0; i < n; ++i) {
    const __int128 delta = (static_cast<__int128>(target[i]) << 4) - mu[i];
    sum += delta < 0 ? -delta : delta;
  }
  return sum;
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

std::optional<Plane> even_pad(const std::optional<Plane>& src, bool axis_rows) {
  if (!src || src->height <= 0 || src->width <= 0 || src->samples.empty()) return std::nullopt;
  Plane out = *src;
  if (axis_rows && (out.height & 1)) {
    out.height -= 1;
    out.samples.resize(static_cast<size_t>(out.height) * static_cast<size_t>(out.width));
  }
  if (!axis_rows && (out.width & 1)) {
    Plane cropped(out.height, out.width - 1);
    for (int y = 0; y < cropped.height; ++y) {
      std::copy_n(out.samples.data() + static_cast<size_t>(y) * static_cast<size_t>(out.width), cropped.width,
                  cropped.samples.data() + static_cast<size_t>(y) * static_cast<size_t>(cropped.width));
    }
    out = std::move(cropped);
  }
  if (out.height <= 0 || out.width <= 0 || out.samples.empty()) return std::nullopt;
  return out;
}

LevelFeatures crop_features(const LevelFeatures& src, int y0, int x0, int height, int width) {
  LevelFeatures out;
  out.height = height;
  out.width = width;
  out.act_shift = src.act_shift;
  auto crop64 = [&](const std::vector<int64_t>& in) {
    std::vector<int64_t> dest(static_cast<size_t>(height) * static_cast<size_t>(width));
    for (int y = 0; y < height; ++y) {
      std::copy_n(in.data() + static_cast<size_t>(y0 + y) * static_cast<size_t>(src.width) + static_cast<size_t>(x0),
                  width, dest.data() + static_cast<size_t>(y) * static_cast<size_t>(width));
    }
    return dest;
  };
  auto crop8 = [&](const std::vector<uint8_t>& in) {
    std::vector<uint8_t> dest(static_cast<size_t>(height) * static_cast<size_t>(width));
    for (int y = 0; y < height; ++y) {
      std::copy_n(in.data() + static_cast<size_t>(y0 + y) * static_cast<size_t>(src.width) + static_cast<size_t>(x0),
                  width, dest.data() + static_cast<size_t>(y) * static_cast<size_t>(width));
    }
    return dest;
  };
  out.mu0_q4 = crop64(src.mu0_q4);
  out.f_sheaf_q4 = crop64(src.f_sheaf_q4);
  out.f_cub_v_q4 = crop64(src.f_cub_v_q4);
  out.f_cub_h_q4 = crop64(src.f_cub_h_q4);
  out.f_ring5_q4 = crop64(src.f_ring5_q4);
  out.f_ssm_h_q4 = crop64(src.f_ssm_h_q4);
  out.f_ssm_v_q4 = crop64(src.f_ssm_v_q4);
  out.n_minus_s = crop64(src.n_minus_s);
  out.w_minus_e = crop64(src.w_minus_e);
  out.wv = crop64(src.wv);
  out.wh = crop64(src.wh);
  out.curv_q4 = crop64(src.curv_q4);
  out.lo_nb = crop64(src.lo_nb);
  out.hi_nb = crop64(src.hi_nb);
  out.flat_pred = crop64(src.flat_pred);
  out.is_bridge = crop8(src.is_bridge);
  return out;
}

LevelFeatures features_with_halo(const Plane& spatial, const Mask& valid, const Halo* halo) {
  if (halo == nullptr) return level_geometry_features(spatial, valid);
  const std::optional<Plane> top = even_pad(halo->top, true);
  const std::optional<Plane> left = even_pad(halo->left, false);
  const int pt = top ? top->height : 0;
  const int pl = left ? left->width : 0;
  if (pt == 0 && pl == 0) return level_geometry_features(spatial, valid);
  const int height = spatial.height;
  const int width = spatial.width;
  Plane big(height + pt, width + pl);
  Mask big_valid(big.height, big.width, false);
  if (pt && top) {
    const int use_w = std::min(width, top->width);
    for (int y = 0; y < pt; ++y) {
      for (int x = 0; x < use_w; ++x) {
        big.at(y, pl + x) = top->at(y, x);
        big_valid.at(y, pl + x) = 1;
      }
    }
  }
  if (pl && left) {
    const int use_h = std::min(height, left->height);
    for (int y = 0; y < use_h; ++y) {
      for (int x = 0; x < pl; ++x) {
        big.at(pt + y, x) = left->at(y, x);
        big_valid.at(pt + y, x) = 1;
      }
    }
  }
  if (pt && pl && halo->corner) {
    const int ch = std::min(pt, halo->corner->height);
    const int cw = std::min(pl, halo->corner->width);
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        big.at(y, x) = halo->corner->at(y, x);
        big_valid.at(y, x) = 1;
      }
    }
  }
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      big.at(pt + y, pl + x) = spatial.at(y, x);
      big_valid.at(pt + y, pl + x) = valid.at(y, x) ? 1 : 0;
    }
  }
  return crop_features(level_geometry_features(big, big_valid), pt, pl, height, width);
}

std::vector<int32_t> sectors_of(const LevelFeatures& base, const std::vector<int>& sites) {
  std::vector<int32_t> sectors(sites.size());
  for (size_t i = 0; i < sites.size(); ++i) {
    const int64_t ns = base.n_minus_s[static_cast<size_t>(sites[i])];
    const int64_t we = base.w_minus_e[static_cast<size_t>(sites[i])];
    const uint64_t ay = static_cast<uint64_t>(iabs64(ns));
    const uint64_t ax = static_cast<uint64_t>(iabs64(we));
    const bool horiz = ax >= ay;
    const bool positive_x = we >= 0;
    const bool positive_y = ns >= 0;
    int sector = horiz ? (positive_x ? 0 : 4) : (positive_y ? 2 : 6);
    const bool diag = ax > 0 && ay > 0 && static_cast<__int128>(std::min(ax, ay)) * 3 >= static_cast<__int128>(std::max(ax, ay));
    if (diag) sector = positive_x ? (positive_y ? 1 : 7) : (positive_y ? 3 : 5);
    sectors[i] = sector;
  }
  return sectors;
}

int64_t odd_at(const std::vector<int64_t>& decoded, int height, int width, int y, int x) {
  if (y < 0 || x < 0 || y >= height || x >= width) return 0;
  return decoded[lin(width, y, x)];
}

std::vector<int16_t> make_props(const LevelFeatures& base, const std::vector<int>& sites,
                                const std::vector<int64_t>& decoded, int sub_idx,
                                const std::vector<int64_t>* coarse, const Plane* ch0) {
  const int width = base.width;
  const int height = base.height;
  const int n = static_cast<int>(sites.size());
  std::vector<int64_t> raw(static_cast<size_t>(n) * 11, 0);
  for (int i = 0; i < n; ++i) {
    const int site = sites[static_cast<size_t>(i)];
    const int y = site / width;
    const int x = site - y * width;
    const int64_t ns = base.n_minus_s[static_cast<size_t>(site)];
    const int64_t we = base.w_minus_e[static_cast<size_t>(site)];
    // Python _props always calls _product(..., 2): the diagonal pair, on every sub-coset.
    (void)sub_idx;
    const int64_t prod = static_cast<int64_t>(
        (static_cast<__int128>(odd_at(decoded, height, width, y - 1, x - 1)) *
         odd_at(decoded, height, width, y + 1, x + 1)) >>
        6);
    int64_t* row = raw.data() + static_cast<size_t>(i) * 11;
    row[0] = ns;
    row[1] = we;
    row[2] = iabs64(ns);
    row[3] = iabs64(we);
    row[4] = base.curv_q4[static_cast<size_t>(site)];
    row[5] = base.f_ring5_q4[static_cast<size_t>(site)];
    row[6] = prod;
    row[7] = coarse ? (*coarse)[static_cast<size_t>(site)] : 0;
    row[8] = ch0 ? ch0->samples[static_cast<size_t>(site)] : 0;
    row[9] = ch0 ? iabs64(ch0->samples[static_cast<size_t>(site)]) : 0;
    row[10] = ((y & 1) << 1) | (x & 1);
  }
  return quantize_props(raw.data(), raw.size());
}

std::vector<int64_t> apply_w(const std::vector<int64_t>& mu0, const int64_t* phi, int cols,
                             const std::vector<int32_t>& sectors, const std::vector<int64_t>* global,
                             const std::optional<std::vector<int64_t>>* overrides) {
  std::vector<int64_t> mu = mu0;
  if (global != nullptr) {
    bool any = false;
    for (int64_t weight : *global) {
      if (weight != 0) any = true;
    }
    if (any) {
      for (int i = 0; i < static_cast<int>(mu0.size()); ++i) {
        mu[static_cast<size_t>(i)] =
            mu0[static_cast<size_t>(i)] + dot_q8(phi + static_cast<size_t>(i * cols), cols, global->data());
      }
    }
  }
  for (int sector = 0; sector < 8; ++sector) {
    if (!overrides[sector]) continue;
    const std::vector<int64_t>& weights = *overrides[sector];
    for (int i = 0; i < static_cast<int>(mu0.size()); ++i) {
      if (sectors[static_cast<size_t>(i)] != sector) continue;
      mu[static_cast<size_t>(i)] =
          mu0[static_cast<size_t>(i)] + dot_q8(phi + static_cast<size_t>(i * cols), cols, weights.data());
    }
  }
  return mu;
}

void write_sig(BitWriter& writer, const std::vector<int32_t>& sig_map, int n_pts) {
  if (n_pts < 64) {
    writer.write_bits(static_cast<uint32_t>(sig_map[0] & 31), 5);
    return;
  }
  int prev = sig_map[0] & 31;
  writer.write_bits(static_cast<uint32_t>(prev), 5);
  for (int bucket = 1; bucket < kMaxBuckets; ++bucket) {
    const int cur = sig_map[static_cast<size_t>(bucket)] & 31;
    writer.write_signed_golomb(cur - prev);
    prev = cur;
  }
}

std::vector<int32_t> read_sig(BitReader& reader, int n_pts) {
  std::vector<int32_t> sig(kMaxBuckets, 0);
  if (n_pts < 64) {
    std::fill(sig.begin(), sig.end(), static_cast<int>(reader.read_bits(5)));
    return sig;
  }
  int prev = static_cast<int>(reader.read_bits(5));
  sig[0] = prev;
  for (int bucket = 1; bucket < kMaxBuckets; ++bucket) {
    const int cur = prev + reader.read_signed_golomb();
    if (cur < 0 || cur >= NUM_SIGMAS) throw CodecError("sigma index out of range");
    sig[static_cast<size_t>(bucket)] = cur;
    prev = cur;
  }
  return sig;
}

std::vector<int64_t> read_weights(BitReader& reader, int n_feat) {
  std::vector<int64_t> weights(static_cast<size_t>(n_feat));
  for (int i = 0; i < n_feat; ++i) weights[static_cast<size_t>(i)] = reader.read_signed_golomb();
  return weights;
}

Plane plane_from_i64(int height, int width, const std::vector<int64_t>& values) {
  Plane plane(height, width);
  for (size_t i = 0; i < values.size() && i < plane.samples.size(); ++i) plane.samples[i] = i32_wrap(values[i]);
  return plane;
}

std::vector<uint8_t> seed_mask(const std::vector<int>& sites, int width) {
  std::vector<uint8_t> seed(sites.size(), 0);
  for (size_t i = 0; i < sites.size(); ++i) {
    const int y = sites[i] / width;
    const int x = sites[i] - y * width;
    seed[i] = (y % 16 == 1 && x % 14 == 0) ? 1 : 0;
  }
  return seed;
}

bool any_override(const std::optional<std::vector<int64_t>>* overrides) {
  for (int sector = 0; sector < 8; ++sector) {
    if (overrides[sector]) return true;
  }
  return false;
}

struct Prepared {
  std::vector<int32_t> residual;
  int nu = 1;
  std::vector<int32_t> sig_idx;
  std::vector<int32_t> sig_map;
  std::vector<int32_t> frac;
  std::optional<std::vector<int64_t>> global_w;
  std::optional<std::vector<int64_t>> overrides[8];
  bool use_dir = false;
  bool use_phi_mu = false;
  bool use_d8 = false;
  bool use_head = false;
  bool use_issm = false;
  bool use_bridge = false;
  std::vector<uint8_t> bridge_blob;
  bool use_tree = false;
  std::unique_ptr<TreeNode> tree;
  bool use_hist = false;
  std::vector<uint8_t> hist_bytes;
  std::vector<int32_t> ext_freq;
  std::vector<int32_t> ext_cum;
  bool use_s3 = false;
  std::vector<uint8_t> s3_blob;
};

void check_planes(const std::vector<Plane>& planes) {
  if (planes.empty()) throw CodecError("empty SLGF tile");
  for (const Plane& plane : planes) {
    if (plane.height != planes[0].height || plane.width != planes[0].width) throw CodecError("SLGF planes disagree");
  }
}

}  // namespace

std::vector<uint8_t> encode_slgf_v3(const std::vector<Plane>& planes, int max_levels, int effort,
                                    const std::vector<Halo>* halos, bool use_static, int sheaf_mode) {
  check_planes(planes);
  const int height = planes[0].height;
  const int width = planes[0].width;
  const auto built = build_levels(height, width, max_levels);
  const int nlevels = static_cast<int>(built.first.size());
  std::vector<uint8_t> out;
  out.push_back(static_cast<uint8_t>(nlevels));
  if (nlevels == 0) {
    const Mask mask(height, width, true);
    for (const Plane& plane : planes) {
      const auto blob = encode_raster(plane, mask, true);
      append_u32(out, static_cast<uint32_t>(blob.size()));
      append_bytes(out, blob);
    }
    return out;
  }
  int feature_mask = 0;
  // D8 search and the MANIAC sweep do not pay for themselves on photographs:
  // measured tool use on Kodak was under one sub-coset in thirty. Directional
  // sheaf weights and the global ridge do. Those stay at effort 2.
  if (effort >= 3 && std::min(height, width) >= 8) {
    feature_mask |= kD8;
  }
  if (effort >= 2 && use_static && std::min(height, width) >= 8) feature_mask |= kHead;
  if (effort >= 3 && use_static) feature_mask |= kIssm;

  std::vector<Tower> towers;
  for (const Plane& plane : planes) towers.push_back(forward_tower(plane, nlevels, sheaf_mode == 1));
  std::vector<std::vector<Plane>> ch_recons;
  std::vector<std::vector<Plane>> ch_resids;
  std::vector<Plane> coarse_ref;
  for (size_t ch = 0; ch < planes.size(); ++ch) {
    const Tower& tower = towers[ch];
    if (static_cast<int>(tower.details.size()) != nlevels) throw CodecError("SLGF tower depth mismatch");
    const Plane* ref = nullptr;
    if (ch > 0 && coarse_ref[0].height == tower.coarse.height && coarse_ref[0].width == tower.coarse.width) {
      ref = &coarse_ref[0];
    }
    const auto med = encode_raster(tower.coarse, built.second, false);
    const auto kal = encode_raster(tower.coarse, built.second, true);
    const auto bank = encode_coarse(tower.coarse, built.second, ref);
    int c_flag = 0;
    const std::vector<uint8_t>* c_blob = &med;
    if (kal.size() < c_blob->size()) {
      c_flag = 1;
      c_blob = &kal;
    }
    if (bank.size() < c_blob->size()) {
      c_flag = 2;
      c_blob = &bank;
    }
    out.push_back(static_cast<uint8_t>(c_flag));
    append_u32(out, static_cast<uint32_t>(c_blob->size()));
    append_bytes(out, *c_blob);
    coarse_ref.push_back(tower.coarse);

    Plane current = tower.coarse;
    std::vector<Plane> level_recons(static_cast<size_t>(nlevels), Plane(1, 1));
    std::vector<Plane> level_resids(static_cast<size_t>(nlevels), Plane(1, 1));
    const Plane* prev = nullptr;
    const Halo* channel_halo = nullptr;
    if (halos != nullptr) {
      if (ch >= halos->size()) throw CodecError("SLGF halo count mismatch");
      channel_halo = &(*halos)[ch];
    }
    for (int level = nlevels - 1; level >= 0; --level) {
      const Level& spec = built.first[static_cast<size_t>(level)];
      const Plane& lifted = tower.details[static_cast<size_t>(level)];
      Plane spatial = unpack_even(current, spec.info, spec.valid);
      const ParityMasks par = parity_masks(spec.valid);
      const LevelFeatures base = features_with_halo(spatial, spec.valid, level == 0 ? channel_halo : nullptr);
      const auto coarse = coarse_resid_projection(prev, spec.info, spec.valid);
      const Plane* ch0_sp = ch >= 1 ? &ch_recons[0][static_cast<size_t>(level)] : nullptr;
      const Plane* ch0_rs = ch >= 1 ? &ch_resids[0][static_cast<size_t>(level)] : nullptr;
      const Plane* ch1_rs = ch >= 2 ? &ch_resids[1][static_cast<size_t>(level)] : nullptr;

      std::optional<PhiAcq> phi_model;
      std::optional<Plane> phi_pred;
      int odd_count = 0;
      for (uint8_t bit : par.odd.bits) odd_count += bit ? 1 : 0;
      if (effort >= 3 && ch0_sp != nullptr && ch0_sp->height == spatial.height && ch0_sp->width == spatial.width &&
          odd_count >= 2048) {
        Plane truth = spatial;
        for (size_t i = 0; i < par.odd.bits.size(); ++i) {
          if (!par.odd.bits[i]) continue;
          truth.samples[i] = i32_wrap(base.flat_pred[i] + lifted.samples[i]);
        }
        const Plane flat_plane = plane_from_i64(base.height, base.width, base.flat_pred);
        phi_model = fit_phi(*ch0_sp, truth, flat_plane, par.odd);
        if (phi_model) phi_pred = apply_phi(*ch0_sp, *phi_model);
      }
      std::optional<std::vector<int64_t>> issm_map;
      if (feature_mask & kIssm) {
        std::vector<int64_t> filled(base.flat_pred.size());
        for (size_t i = 0; i < filled.size(); ++i) filled[i] = spatial.samples[i];
        for (size_t i = 0; i < par.odd.bits.size(); ++i) {
          if (par.odd.bits[i]) filled[i] = base.flat_pred[i];
        }
        issm_map = context_map(filled.data(), spatial.height, spatial.width);
      }

      const std::vector<Mask> subs = subcoset_masks(spec.valid);
      std::vector<int64_t> decoded(static_cast<size_t>(spatial.height) * static_cast<size_t>(spatial.width), 0);
      Mask known(spatial.height, spatial.width, false);
      std::vector<Prepared> prepared;
      bool phi_used = false;
      for (size_t sub = 0; sub < subs.size(); ++sub) {
        const Mask& mask = subs[sub];
        const std::vector<int> sites = true_sites(mask);
        const int n = static_cast<int>(sites.size());
        std::optional<Plane> d8;
        if (feature_mask & kD8) {
          Plane samples = spatial;
          for (size_t i = 0; i < known.bits.size(); ++i) {
            if (!known.bits[i]) continue;
            samples.samples[i] = i32_wrap(base.flat_pred[i] + decoded[i]);
          }
          const Plane flat_plane = plane_from_i64(base.height, base.width, base.flat_pred);
          d8 = sheaf_kalman_delta(samples, known, mask, flat_plane);
        }
        const FeatureRows rows =
            assemble_features_and_activity(base, static_cast<int>(sub), sites, decoded, coarse ? &*coarse : nullptr,
                                           ch0_sp, ch0_rs, ch1_rs);
        std::vector<int64_t> mu0(static_cast<size_t>(n));
        std::vector<int64_t> target(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
          mu0[static_cast<size_t>(i)] = base.mu0_q4[static_cast<size_t>(sites[static_cast<size_t>(i)])];
          target[static_cast<size_t>(i)] = lifted.samples[static_cast<size_t>(sites[static_cast<size_t>(i)])];
        }
        Prepared item;
        if (phi_pred) {
          std::vector<int64_t> mu_phi(static_cast<size_t>(n));
          for (int i = 0; i < n; ++i) {
            const int site = sites[static_cast<size_t>(i)];
            mu_phi[static_cast<size_t>(i)] =
                (static_cast<int64_t>(phi_pred->samples[static_cast<size_t>(site)]) - base.flat_pred[static_cast<size_t>(site)])
                << 4;
          }
          if (l1_q4(target.data(), mu_phi.data(), n) + 64 < l1_q4(target.data(), mu0.data(), n)) {
            mu0 = std::move(mu_phi);
            item.use_phi_mu = true;
            phi_used = true;
          }
        }
        if (effort >= 2) {
          std::vector<int64_t> mu_dir = mu0;
          for (int i = 0; i < n; ++i) {
            mu_dir[static_cast<size_t>(i)] += base.f_sheaf_q4[static_cast<size_t>(sites[static_cast<size_t>(i)])] >> 1;
          }
          if (l1_q4(target.data(), mu_dir.data(), n) + 16 < l1_q4(target.data(), mu0.data(), n)) {
            mu0 = std::move(mu_dir);
            item.use_dir = true;
          }
        }
        if (d8) {
          int nonzero = 0;
          for (int site : sites) {
            if (d8->samples[static_cast<size_t>(site)] != 0) ++nonzero;
          }
          if (nonzero) {
            std::vector<int64_t> mu_d8 = mu0;
            for (int i = 0; i < n; ++i) {
              mu_d8[static_cast<size_t>(i)] +=
                  static_cast<int64_t>(d8->samples[static_cast<size_t>(sites[static_cast<size_t>(i)])]) << 4;
            }
            if (l1_q4(target.data(), mu_d8.data(), n) + 16 < l1_q4(target.data(), mu0.data(), n)) {
              mu0 = std::move(mu_d8);
              item.use_d8 = true;
            }
          }
        }
        std::vector<int64_t> head;
        if ((feature_mask & kHead) && rows.cols > 0) {
          head = project_head(rows.phi.data(), rows.rows, rows.cols);
          if (static_cast<int>(head.size()) == n) {
            std::vector<int64_t> mu_h = mu0;
            for (int i = 0; i < n; ++i) mu_h[static_cast<size_t>(i)] += head[static_cast<size_t>(i)];
            if (l1_q4(target.data(), mu_h.data(), n) + 16 < l1_q4(target.data(), mu0.data(), n)) {
              mu0 = std::move(mu_h);
              item.use_head = true;
            }
          }
        }
        if (issm_map && static_cast<int>(issm_map->size()) == spatial.height * spatial.width) {
          std::vector<int64_t> mu_i = mu0;
          for (int i = 0; i < n; ++i) {
            mu_i[static_cast<size_t>(i)] += (*issm_map)[static_cast<size_t>(sites[static_cast<size_t>(i)])];
          }
          if (l1_q4(target.data(), mu_i.data(), n) + 16 < l1_q4(target.data(), mu0.data(), n)) {
            mu0 = std::move(mu_i);
            item.use_issm = true;
          }
        }

        const std::vector<int32_t> sectors = sectors_of(base, sites);
        std::vector<int32_t> buckets(static_cast<size_t>(n));
        bucket_from_act_q4(rows.act_q4.data(), rows.act_q4.size(), base.act_shift, buckets.data());
        std::vector<int32_t> target32(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) target32[static_cast<size_t>(i)] = i32_wrap(target[static_cast<size_t>(i)]);
        const SubcosetEstimate model = estimate_subcoset_model(target32.data(), mu0.data(), rows.phi.data(), rows.rows,
                                                               rows.cols, buckets.data());
        if (model.use_ridge) item.global_w = model.w_q8;
        if (effort >= 3 && n >= 200 && rows.cols > 0) {
          std::vector<int64_t> target_q4(static_cast<size_t>(n));
          for (int i = 0; i < n; ++i) {
            target_q4[static_cast<size_t>(i)] =
                static_cast<int64_t>((static_cast<__int128>(target[static_cast<size_t>(i)]) << 4) - mu0[static_cast<size_t>(i)]);
          }
          for (int sector = 0; sector < 8; ++sector) {
            std::vector<int> sel;
            for (int i = 0; i < n; ++i) {
              if (sectors[static_cast<size_t>(i)] == sector) sel.push_back(i);
            }
            if (static_cast<int>(sel.size()) < 64) continue;
            std::vector<int64_t> phi_s(sel.size() * static_cast<size_t>(rows.cols));
            std::vector<int64_t> y_s(sel.size());
            for (size_t k = 0; k < sel.size(); ++k) {
              y_s[k] = target_q4[static_cast<size_t>(sel[k])];
              std::copy_n(rows.phi.data() + static_cast<size_t>(sel[k] * rows.cols), rows.cols,
                          phi_s.data() + k * static_cast<size_t>(rows.cols));
            }
            auto local = fit_ridge_q8(phi_s.data(), static_cast<int>(sel.size()), rows.cols, y_s.data(), 64);
            if (local) item.overrides[sector] = std::move(*local);
          }
        }
        std::optional<std::vector<int64_t>> none[8];
        const std::vector<int64_t>* global = item.global_w ? &*item.global_w : nullptr;
        if (any_override(item.overrides)) {
          const std::vector<int64_t> mu_global = apply_w(mu0, rows.phi.data(), rows.cols, sectors, global, none);
          std::vector<int64_t> mu_i_g(static_cast<size_t>(n));
          std::vector<int32_t> frac_g(static_cast<size_t>(n));
          split_mu_q4(mu_global.data(), static_cast<size_t>(n), mu_i_g.data(), frac_g.data());
          std::vector<int32_t> resid_g(static_cast<size_t>(n));
          std::vector<int64_t> resid_g64(static_cast<size_t>(n));
          for (int i = 0; i < n; ++i) {
            resid_g[static_cast<size_t>(i)] = i32_wrap(target[static_cast<size_t>(i)] - mu_i_g[static_cast<size_t>(i)]);
            resid_g64[static_cast<size_t>(i)] = resid_g[static_cast<size_t>(i)];
          }
          const SigmaNu fit_g = fit_sigma_nu(resid_g64.data(), frac_g.data(), buckets.data(), n, kMaxBuckets);
          std::vector<int32_t> sig_g(static_cast<size_t>(n));
          for (int i = 0; i < n; ++i) {
            sig_g[static_cast<size_t>(i)] = n < 64 ? fit_g.sigma[0] : fit_g.sigma[static_cast<size_t>(buckets[static_cast<size_t>(i)])];
          }
          const auto plain = encode_parametric_rans(resid_g, fit_g.nu, sig_g, frac_g);

          const std::vector<int64_t> mu_mix = apply_w(mu0, rows.phi.data(), rows.cols, sectors, global, item.overrides);
          std::vector<int64_t> mu_int(static_cast<size_t>(n));
          std::vector<int32_t> frac(static_cast<size_t>(n));
          split_mu_q4(mu_mix.data(), static_cast<size_t>(n), mu_int.data(), frac.data());
          std::vector<int32_t> residual(static_cast<size_t>(n));
          std::vector<int64_t> residual64(static_cast<size_t>(n));
          for (int i = 0; i < n; ++i) {
            residual[static_cast<size_t>(i)] = i32_wrap(target[static_cast<size_t>(i)] - mu_int[static_cast<size_t>(i)]);
            residual64[static_cast<size_t>(i)] = residual[static_cast<size_t>(i)];
          }
          const SigmaNu fit_m = fit_sigma_nu(residual64.data(), frac.data(), buckets.data(), n, kMaxBuckets);
          std::vector<int32_t> sig_idx(static_cast<size_t>(n));
          for (int i = 0; i < n; ++i) {
            sig_idx[static_cast<size_t>(i)] =
                n < 64 ? fit_m.sigma[0] : fit_m.sigma[static_cast<size_t>(buckets[static_cast<size_t>(i)])];
          }
          const auto mixed = encode_parametric_rans(residual, fit_m.nu, sig_idx, frac);
          int tax = 8;
          for (int sector = 0; sector < 8; ++sector) {
            if (item.overrides[sector]) tax += 2 * static_cast<int>(item.overrides[sector]->size());
          }
          if (mixed.size() + static_cast<size_t>(tax) >= plain.size()) {
            for (int sector = 0; sector < 8; ++sector) item.overrides[sector].reset();
            item.residual = std::move(resid_g);
            item.frac = std::move(frac_g);
            item.nu = fit_g.nu;
            item.sig_map = fit_g.sigma;
            item.sig_idx = std::move(sig_g);
          } else {
            item.residual = std::move(residual);
            item.frac = std::move(frac);
            item.nu = fit_m.nu;
            item.sig_map = fit_m.sigma;
            item.sig_idx = std::move(sig_idx);
          }
        } else {
          const std::vector<int64_t> mu_q = apply_w(mu0, rows.phi.data(), rows.cols, sectors, global, none);
          std::vector<int64_t> mu_int(static_cast<size_t>(n));
          item.frac.resize(static_cast<size_t>(n));
          split_mu_q4(mu_q.data(), static_cast<size_t>(n), mu_int.data(), item.frac.data());
          item.residual.resize(static_cast<size_t>(n));
          item.sig_idx.resize(static_cast<size_t>(n));
          item.sig_map = model.sig_map;
          item.nu = model.nu_idx;
          for (int i = 0; i < n; ++i) {
            item.residual[static_cast<size_t>(i)] = i32_wrap(target[static_cast<size_t>(i)] - mu_int[static_cast<size_t>(i)]);
            item.sig_idx[static_cast<size_t>(i)] =
                n < 64 ? model.sig_map[0] : model.sig_map[static_cast<size_t>(buckets[static_cast<size_t>(i)])];
          }
        }

        if (effort >= 3 && n >= 80) {
          const std::vector<int16_t> qprops =
              make_props(base, sites, decoded, static_cast<int>(sub), coarse ? &*coarse : nullptr, ch0_rs);
          std::vector<int64_t> resid64(static_cast<size_t>(n));
          for (int i = 0; i < n; ++i) resid64[static_cast<size_t>(i)] = item.residual[static_cast<size_t>(i)];
          auto tree = build_sigma_tree(qprops.data(), n, 11, resid64.data());
          if (tree) {
            const TreeMap mapped = apply_tree(*tree, qprops.data(), n, 11);
            const auto blob_b = encode_parametric_rans(item.residual, item.nu, item.sig_idx, item.frac);
            const auto blob_t = encode_parametric_rans(item.residual, item.nu, mapped.sigma, item.frac);
            BitWriter tree_header;
            write_tree(tree_header, *tree);
            if (blob_t.size() + tree_header.finish().size() < blob_b.size()) {
              item.use_tree = true;
              item.sig_idx = mapped.sigma;
              item.tree = std::move(tree);
            }
          }
        }
        if (item.use_tree && item.tree && n >= 400) {
          const std::vector<int16_t> qprops =
              make_props(base, sites, decoded, static_cast<int>(sub), coarse ? &*coarse : nullptr, ch0_rs);
          std::vector<int64_t> resid64(static_cast<size_t>(n));
          for (int i = 0; i < n; ++i) resid64[static_cast<size_t>(i)] = item.residual[static_cast<size_t>(i)];
          auto packed = clustered_histograms(qprops.data(), n, 11, *item.tree, resid64.data(), item.nu);
          if (packed) {
            const auto blob_h = encode_parametric_rans(item.residual, item.nu, item.sig_idx, item.frac, {}, {}, {},
                                                       packed->freq, packed->cum);
            const auto blob_p = encode_parametric_rans(item.residual, item.nu, item.sig_idx, item.frac);
            if (blob_h.size() + packed->header.size() < blob_p.size()) {
              item.use_hist = true;
              item.hist_bytes = std::move(packed->header);
              item.ext_freq = std::move(packed->freq);
              item.ext_cum = std::move(packed->cum);
            }
          }
        }
        if (!item.use_hist) {
          const ParametricTables& tables = get_parametric_tables();
          std::vector<uint8_t> bridge(static_cast<size_t>(n), 0);
          std::vector<int32_t> lo(static_cast<size_t>(n));
          std::vector<int32_t> hi(static_cast<size_t>(n));
          std::vector<int64_t> mu_int(static_cast<size_t>(n));
          std::vector<int32_t> frac_check(static_cast<size_t>(n));
          split_mu_q4(apply_w(mu0, rows.phi.data(), rows.cols, sectors, global, item.overrides).data(),
                      static_cast<size_t>(n), mu_int.data(), frac_check.data());
          int bridges = 0;
          for (int i = 0; i < n; ++i) {
            const int site = sites[static_cast<size_t>(i)];
            const bool on = base.is_bridge[static_cast<size_t>(site)] && tables.shifts[item.sig_idx[static_cast<size_t>(i)]] == 0;
            bridge[static_cast<size_t>(i)] = on ? 1 : 0;
            bridges += on ? 1 : 0;
            lo[static_cast<size_t>(i)] = i32_wrap(base.lo_nb[static_cast<size_t>(site)] - mu_int[static_cast<size_t>(i)]);
            hi[static_cast<size_t>(i)] = i32_wrap(base.hi_nb[static_cast<size_t>(site)] - mu_int[static_cast<size_t>(i)]);
          }
          if (bridges >= 24) {
            double zero_sum = 0.0;
            double mode_sum = 0.0;
            for (int i = 0; i < n; ++i) {
              if (!bridge[static_cast<size_t>(i)]) continue;
              const int64_t rt = item.residual[static_cast<size_t>(i)];
              zero_sum += static_cast<double>(iabs64(rt));
              const int64_t dlo = iabs64(rt - lo[static_cast<size_t>(i)]);
              const int64_t dhi = iabs64(rt - hi[static_cast<size_t>(i)]);
              mode_sum += static_cast<double>(dlo < dhi ? dlo : dhi);
            }
            const double nd = static_cast<double>(bridges);
            if (mode_sum / nd + 1.5 < zero_sum / nd) {
              const auto plain = encode_parametric_rans(item.residual, item.nu, item.sig_idx, item.frac);
              const auto mixed = encode_parametric_rans(item.residual, item.nu, item.sig_idx, item.frac, bridge, lo, hi);
              if (mixed.size() < plain.size()) {
                item.use_bridge = true;
                item.bridge_blob = mixed;
              }
            }
          }
        }
        if (effort >= 3 && level == 0 && std::min(height, width) >= 128 && !item.use_hist && !item.use_bridge && n >= 256) {
          const std::vector<uint8_t> seed = seed_mask(sites, width);
          std::vector<int64_t> resid64(static_cast<size_t>(n));
          for (int i = 0; i < n; ++i) resid64[static_cast<size_t>(i)] = item.residual[static_cast<size_t>(i)];
          auto trial = encode_s3(resid64.data(), item.sig_idx.data(), item.frac.data(), item.nu, seed.data(),
                                 static_cast<size_t>(n));
          const auto plain = encode_parametric_rans(item.residual, item.nu, item.sig_idx, item.frac);
          if (trial && trial->size() < plain.size()) {
            item.use_s3 = true;
            item.s3_blob = std::move(*trial);
          }
        }
        prepared.push_back(std::move(item));
        for (int i = 0; i < n; ++i) decoded[static_cast<size_t>(sites[static_cast<size_t>(i)])] = target[static_cast<size_t>(i)];
        for (size_t i = 0; i < mask.bits.size(); ++i) {
          if (mask.bits[i]) known.bits[i] = 1;
        }
      }

      if (!phi_used) {
        phi_model.reset();
        for (const Prepared& item : prepared) {
          if (item.use_phi_mu) phi_used = true;
        }
        if (!phi_used) phi_model.reset();
      }
      BitWriter level_header;
      level_header.write_bits(static_cast<uint32_t>(feature_mask & 15), 4);
      if (!phi_model || !phi_used) {
        level_header.write_bits(0, 1);
        phi_model.reset();
      } else {
        level_header.write_bits(1, 1);
        write_phi(level_header, *phi_model);
      }
      const std::vector<uint8_t> header_bytes = level_header.finish();
      BitWriter meta;
      std::vector<std::vector<uint8_t>> payloads;
      for (Prepared& item : prepared) {
        bool use_phi_mu = item.use_phi_mu;
        if (!phi_model) use_phi_mu = false;
        meta.write_bits(item.use_dir ? 1u : 0u, 1);
        meta.write_bits(use_phi_mu ? 1u : 0u, 1);
        meta.write_bits(item.use_d8 ? 1u : 0u, 1);
        meta.write_bits(item.use_head ? 1u : 0u, 1);
        meta.write_bits(item.use_issm ? 1u : 0u, 1);
        const bool use_ridge = item.global_w.has_value();
        meta.write_bits(use_ridge ? 1u : 0u, 1);
        meta.write_bits(item.use_bridge ? 1u : 0u, 1);
        int regime_bits = 0;
        for (int sector = 0; sector < 8; ++sector) {
          if (item.overrides[sector]) regime_bits |= 1 << sector;
        }
        meta.write_bits(regime_bits ? 1u : 0u, 1);
        meta.write_bits(static_cast<uint32_t>(item.nu & 3), 2);
        if (use_ridge) {
          for (int64_t weight : *item.global_w) meta.write_signed_golomb(static_cast<int>(weight));
        }
        if (regime_bits) {
          meta.write_bits(static_cast<uint32_t>(regime_bits), 8);
          for (int sector = 0; sector < 8; ++sector) {
            if ((regime_bits & (1 << sector)) == 0) continue;
            for (int64_t weight : *item.overrides[sector]) meta.write_signed_golomb(static_cast<int>(weight));
          }
        }
        meta.write_bits(item.use_tree ? 1u : 0u, 1);
        if (item.use_tree && item.tree) write_tree(meta, *item.tree);
        else write_sig(meta, item.sig_map, static_cast<int>(item.residual.size()));
        meta.write_bits(item.use_hist ? 1u : 0u, 1);
        meta.write_bits(item.use_s3 ? 1u : 0u, 1);
        std::vector<uint8_t> chosen;
        if (item.use_s3) chosen = std::move(item.s3_blob);
        else if (item.use_bridge) chosen = std::move(item.bridge_blob);
        else if (item.use_hist) {
          const auto rans = encode_parametric_rans(item.residual, item.nu, item.sig_idx, item.frac, {}, {}, {},
                                                   item.ext_freq, item.ext_cum);
          if (item.hist_bytes.size() > 65535) throw CodecError("SLGF histogram exceeds 65535 bytes");
          append_u16(chosen, static_cast<uint16_t>(item.hist_bytes.size()));
          append_bytes(chosen, item.hist_bytes);
          append_bytes(chosen, rans);
        } else {
          chosen = encode_parametric_rans(item.residual, item.nu, item.sig_idx, item.frac);
        }
        payloads.push_back(std::move(chosen));
      }
      const std::vector<uint8_t> meta_bytes = meta.finish();
      if (header_bytes.size() > 65535 || meta_bytes.size() > 65535) {
        throw CodecError("SLGF metadata exceeds 65535 bytes");
      }
      append_u16(out, static_cast<uint16_t>(header_bytes.size()));
      append_bytes(out, header_bytes);
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
      current = lift_inverse(full, spec.valid, sheaf_mode == 1);
      level_recons[static_cast<size_t>(level)] = current;
      level_resids[static_cast<size_t>(level)] = std::move(resid);
      prev = &level_resids[static_cast<size_t>(level)];
    }
    ch_recons.push_back(std::move(level_recons));
    ch_resids.push_back(std::move(level_resids));
  }
  return out;
}

std::vector<Plane> decode_slgf_v3(const uint8_t* data, size_t size, int height, int width, int channels,
                                  const std::vector<Halo>* halos) {
  if (size < 1) throw CodecError("truncated SLGF tile");
  Reader reader{data, size, 0};
  const int head = reader.u8("truncated SLGF tile");
  // Bit 4 selects the sheaf-weighted lift. Older streams leave it clear.
  const int sheaf_mode = (head & 0x10) ? 1 : 0;
  const int nlevels = head & 0x0f;
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
  std::vector<Plane> coarse_ref;
  std::vector<Plane> out;
  for (int ch = 0; ch < channels; ++ch) {
    if (reader.pos + 5 > reader.size) throw CodecError("truncated SLGF coarse header");
    const int c_flag = reader.u8("truncated SLGF coarse header");
    const uint32_t c_len = reader.u32("truncated SLGF coarse header");
    const auto blob = reader.take(c_len, "truncated SLGF coarse payload");
    const Plane* ref = nullptr;
    if (ch > 0 && coarse_ref[0].height == built.second.height && coarse_ref[0].width == built.second.width) {
      ref = &coarse_ref[0];
    }
    Plane current;
    if (c_flag == 2) {
      current = decode_coarse(blob.data(), blob.size(), built.second.height, built.second.width, built.second, ref);
    } else if (c_flag == 0 || c_flag == 1) {
      current = decode_raster(blob.data(), blob.size(), built.second.height, built.second.width, built.second, c_flag == 1);
    } else {
      throw CodecError("unknown coarse coder");
    }
    coarse_ref.push_back(current);
    std::vector<Plane> level_recons(static_cast<size_t>(nlevels), Plane(1, 1));
    std::vector<Plane> level_resids(static_cast<size_t>(nlevels), Plane(1, 1));
    const Plane* prev = nullptr;
    const Halo* channel_halo = nullptr;
    if (halos != nullptr) {
      if (static_cast<size_t>(ch) >= halos->size()) throw CodecError("SLGF halo count mismatch");
      channel_halo = &(*halos)[static_cast<size_t>(ch)];
    }
    for (int level = nlevels - 1; level >= 0; --level) {
      const Level& spec = built.first[static_cast<size_t>(level)];
      Plane spatial = unpack_even(current, spec.info, spec.valid);
      const ParityMasks par = parity_masks(spec.valid);
      const LevelFeatures base = features_with_halo(spatial, spec.valid, level == 0 ? channel_halo : nullptr);
      const auto coarse = coarse_resid_projection(prev, spec.info, spec.valid);
      const Plane* ch0_sp = ch >= 1 ? &ch_recons[0][static_cast<size_t>(level)] : nullptr;
      const Plane* ch0_rs = ch >= 1 ? &ch_resids[0][static_cast<size_t>(level)] : nullptr;
      const Plane* ch1_rs = ch >= 2 ? &ch_resids[1][static_cast<size_t>(level)] : nullptr;
      const uint16_t header_n = reader.u16("truncated level header");
      const auto header_bytes = reader.take(header_n, "truncated level header");
      BitReader header(header_bytes.data(), header_bytes.size());
      const int feature_mask = static_cast<int>(header.read_bits(4));
      const bool has_phi = header.read_bits(1) == 1;
      std::optional<PhiAcq> phi_model;
      std::optional<Plane> phi_pred;
      if (has_phi) {
        phi_model = read_phi(header);
        if (ch0_sp != nullptr && ch0_sp->height == spatial.height && ch0_sp->width == spatial.width) {
          phi_pred = apply_phi(*ch0_sp, *phi_model);
        }
      }
      std::optional<std::vector<int64_t>> issm_map;
      if (feature_mask & kIssm) {
        std::vector<int64_t> filled(static_cast<size_t>(spatial.height) * static_cast<size_t>(spatial.width));
        for (size_t i = 0; i < filled.size(); ++i) filled[i] = spatial.samples[i];
        for (size_t i = 0; i < par.odd.bits.size() && i < filled.size(); ++i) {
          if (par.odd.bits[i]) filled[i] = base.flat_pred[i];
        }
        issm_map = context_map(filled.data(), spatial.height, spatial.width);
      }
      const uint16_t meta_n = reader.u16("truncated sub-coset metadata");
      const auto meta_bytes = reader.take(meta_n, "truncated sub-coset metadata");
      BitReader meta(meta_bytes.data(), meta_bytes.size());
      const std::vector<Mask> subs = subcoset_masks(spec.valid);
      std::vector<int64_t> decoded(static_cast<size_t>(spatial.height) * static_cast<size_t>(spatial.width), 0);
      Mask known(spatial.height, spatial.width, false);
      for (size_t sub = 0; sub < subs.size(); ++sub) {
        const Mask& mask = subs[sub];
        const std::vector<int> sites = true_sites(mask);
        const int n = static_cast<int>(sites.size());
        std::optional<Plane> d8;
        if (feature_mask & kD8) {
          Plane samples = spatial;
          for (size_t i = 0; i < known.bits.size(); ++i) {
            if (!known.bits[i]) continue;
            samples.samples[i] = i32_wrap(base.flat_pred[i] + decoded[i]);
          }
          const Plane flat_plane = plane_from_i64(base.height, base.width, base.flat_pred);
          d8 = sheaf_kalman_delta(samples, known, mask, flat_plane);
        }
        const FeatureRows rows =
            assemble_features_and_activity(base, static_cast<int>(sub), sites, decoded, coarse ? &*coarse : nullptr,
                                           ch0_sp, ch0_rs, ch1_rs);
        std::vector<int64_t> mu0(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) mu0[static_cast<size_t>(i)] = base.mu0_q4[static_cast<size_t>(sites[static_cast<size_t>(i)])];
        const bool use_dir = meta.read_bits(1) == 1;
        const bool use_phi_mu = meta.read_bits(1) == 1;
        const bool use_d8 = meta.read_bits(1) == 1;
        const bool use_head = meta.read_bits(1) == 1;
        const bool use_issm = meta.read_bits(1) == 1;
        const bool use_ridge = meta.read_bits(1) == 1;
        const bool use_bridge = meta.read_bits(1) == 1;
        const bool use_regimes = meta.read_bits(1) == 1;
        const int nu = static_cast<int>(meta.read_bits(2));
        if (use_phi_mu && phi_pred) {
          for (int i = 0; i < n; ++i) {
            const int site = sites[static_cast<size_t>(i)];
            mu0[static_cast<size_t>(i)] =
                (static_cast<int64_t>(phi_pred->samples[static_cast<size_t>(site)]) - base.flat_pred[static_cast<size_t>(site)])
                << 4;
          }
        }
        if (use_dir) {
          for (int i = 0; i < n; ++i) {
            mu0[static_cast<size_t>(i)] += base.f_sheaf_q4[static_cast<size_t>(sites[static_cast<size_t>(i)])] >> 1;
          }
        }
        if (use_d8 && d8) {
          for (int i = 0; i < n; ++i) {
            mu0[static_cast<size_t>(i)] +=
                static_cast<int64_t>(d8->samples[static_cast<size_t>(sites[static_cast<size_t>(i)])]) << 4;
          }
        }
        if (use_head && rows.cols > 0) {
          const std::vector<int64_t> head = project_head(rows.phi.data(), rows.rows, rows.cols);
          if (static_cast<int>(head.size()) == n) {
            for (int i = 0; i < n; ++i) mu0[static_cast<size_t>(i)] += head[static_cast<size_t>(i)];
          }
        }
        if (use_issm && issm_map) {
          for (int i = 0; i < n; ++i) {
            mu0[static_cast<size_t>(i)] += (*issm_map)[static_cast<size_t>(sites[static_cast<size_t>(i)])];
          }
        }
        const std::vector<int64_t> global_w = use_ridge ? read_weights(meta, rows.cols) : std::vector<int64_t>{};
        std::optional<std::vector<int64_t>> overrides[8];
        if (use_regimes) {
          const int bits = static_cast<int>(meta.read_bits(8));
          for (int sector = 0; sector < 8; ++sector) {
            if (bits & (1 << sector)) overrides[sector] = read_weights(meta, rows.cols);
          }
        }
        const std::vector<int32_t> sectors = sectors_of(base, sites);
        const std::vector<int64_t> mu_q =
            apply_w(mu0, rows.phi.data(), rows.cols, sectors, use_ridge ? &global_w : nullptr, overrides);
        std::vector<int64_t> mu_int(static_cast<size_t>(n));
        std::vector<int32_t> frac(static_cast<size_t>(n));
        split_mu_q4(mu_q.data(), static_cast<size_t>(n), mu_int.data(), frac.data());
        const bool use_tree = meta.read_bits(1) == 1;
        std::vector<int32_t> sig_idx;
        std::vector<int32_t> leaf_ids;
        std::unique_ptr<TreeNode> tree;
        if (use_tree) {
          tree = read_tree(meta);
          number_tree(*tree);
          const std::vector<int16_t> qprops =
              make_props(base, sites, decoded, static_cast<int>(sub), coarse ? &*coarse : nullptr, ch0_rs);
          const TreeMap mapped = apply_tree(*tree, qprops.data(), n, 11);
          sig_idx = mapped.sigma;
          leaf_ids = mapped.leaf;
        } else {
          const std::vector<int32_t> sig_map = read_sig(meta, n);
          std::vector<int32_t> buckets(static_cast<size_t>(n));
          bucket_from_act_q4(rows.act_q4.data(), rows.act_q4.size(), base.act_shift, buckets.data());
          sig_idx.resize(static_cast<size_t>(n));
          for (int i = 0; i < n; ++i) sig_idx[static_cast<size_t>(i)] = sig_map[static_cast<size_t>(buckets[static_cast<size_t>(i)])];
        }
        const bool use_hist = meta.read_bits(1) == 1;
        const bool use_s3 = meta.read_bits(1) == 1;
        const uint32_t sub_n = reader.u32("truncated sub-coset length");
        const auto raw = reader.take(sub_n, "truncated sub-coset stream");
        std::vector<int32_t> residual;
        if (use_s3) {
          const std::vector<uint8_t> seed = seed_mask(sites, spatial.width);
          residual = decode_s3(raw.data(), raw.size(), n, sig_idx.data(), frac.data(), nu, seed.data());
        } else if (use_hist) {
          if (!tree) throw CodecError("histogram without a context tree");
          if (raw.size() < 2) throw CodecError("truncated histogram");
          const int hist_n = raw[0] | (raw[1] << 8);
          if (static_cast<size_t>(2 + hist_n) > raw.size()) throw CodecError("truncated histogram");
          BitReader hist_reader(raw.data() + 2, static_cast<size_t>(hist_n));
          const HistogramTables tables = read_histograms(hist_reader, leaf_ids.data(), n, nu);
          residual = decode_parametric_rans(raw.subspan(static_cast<size_t>(2 + hist_n)), n, nu, sig_idx, frac, {}, {}, {},
                                            tables.freq, tables.cum);
        } else if (use_bridge) {
          const ParametricTables& tables = get_parametric_tables();
          std::vector<uint8_t> bridge(static_cast<size_t>(n), 0);
          std::vector<int32_t> lo(static_cast<size_t>(n));
          std::vector<int32_t> hi(static_cast<size_t>(n));
          for (int i = 0; i < n; ++i) {
            const int site = sites[static_cast<size_t>(i)];
            bridge[static_cast<size_t>(i)] =
                (base.is_bridge[static_cast<size_t>(site)] && tables.shifts[sig_idx[static_cast<size_t>(i)]] == 0) ? 1 : 0;
            lo[static_cast<size_t>(i)] = i32_wrap(base.lo_nb[static_cast<size_t>(site)] - mu_int[static_cast<size_t>(i)]);
            hi[static_cast<size_t>(i)] = i32_wrap(base.hi_nb[static_cast<size_t>(site)] - mu_int[static_cast<size_t>(i)]);
          }
          residual = decode_parametric_rans(raw, n, nu, sig_idx, frac, bridge, lo, hi);
        } else {
          residual = decode_parametric_rans(raw, n, nu, sig_idx, frac);
        }
        if (static_cast<int>(residual.size()) != n) throw CodecError("SLGF sub-coset length mismatch");
        for (int i = 0; i < n; ++i) {
          decoded[static_cast<size_t>(sites[static_cast<size_t>(i)])] =
              static_cast<int64_t>(residual[static_cast<size_t>(i)]) + mu_int[static_cast<size_t>(i)];
        }
        for (size_t i = 0; i < mask.bits.size(); ++i) {
          if (mask.bits[i]) known.bits[i] = 1;
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
      current = lift_inverse(full, spec.valid, sheaf_mode == 1);
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

#ifndef VPXC_HAS_ISSM_INT8
// issm_int8.hpp is not in the tree. These match issm_int8.py on the default weights.
std::vector<int64_t> context_map(const int64_t* filled, int height, int width) {
  const size_t n = static_cast<size_t>(std::max(height, 0)) * static_cast<size_t>(std::max(width, 0));
  std::vector<int64_t> out(n, 0);
  if (height <= 0 || width <= 0 || filled == nullptr) return out;
  static const int8_t dw[8][9] = {
      {0, -8, 0, -8, 32, -8, 0, -8, 0}, {-1, -2, -1, 0, 0, 0, 1, 2, 1}, {-1, 0, 1, -2, 0, 2, -1, 0, 1},
      {0, -4, 4, -4, 0, 4, 4, -4, 0},   {4, -4, 0, 4, 0, -4, 0, -4, 4}, {-2, 0, 4, 0, -4, 0, 4, 0, -2},
      {0, 0, 0, 8, -16, 8, 0, 0, 0},    {0, 8, 0, 0, -16, 0, 0, 8, 0},
  };
  static const int8_t pw[8] = {24, 8, 8, 4, 4, 4, 6, 6};
  auto at = [&](int y, int x) -> int64_t {
    if (y < 0) y = 0;
    if (x < 0) x = 0;
    if (y >= height) y = height - 1;
    if (x >= width) x = width - 1;
    return filled[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)];
  };
  std::vector<int64_t> mixed(n, 0);
  for (int k = 0; k < 8; ++k) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int64_t acc = 0;
        for (int dy = 0; dy < 3; ++dy) {
          for (int dx = 0; dx < 3; ++dx) {
            acc += static_cast<int64_t>(dw[k][dy * 3 + dx]) * at(y + dy - 1, x + dx - 1);
          }
        }
        mixed[lin(width, y, x)] += static_cast<int64_t>(pw[k]) * acc;
      }
    }
  }
  for (int64_t& value : mixed) value = (value + 4) >> 3;
  for (int y = 0; y < height; ++y) {
    std::vector<int64_t> decay(static_cast<size_t>(width), 60000);
    std::vector<int64_t> bias(static_cast<size_t>(width));
    for (int x = 0; x < width; ++x) {
      int64_t b = mixed[lin(width, y, x)];
      if (b < -2000) b = -2000;
      if (b > 2000) b = 2000;
      bias[static_cast<size_t>(x)] = b;
    }
    const std::vector<int64_t> state = tree(decay, bias);
    for (int x = 0; x < width; ++x) out[lin(width, y, x)] = (state[static_cast<size_t>(x)] >> 4) + mixed[lin(width, y, x)];
  }
  return out;
}

std::vector<int64_t> project_head(const int64_t* phi_q4, int rows, int cols) {
  static const int8_t head[16] = {16, -12, -12, 8, 8, -6, 10, -8, 4, -4, 6, -6, 0, 0, 0, 0};
  std::vector<int64_t> out(static_cast<size_t>(std::max(rows, 0)), 0);
  const int n = std::min(std::max(cols, 0), 16);
  if (n <= 0 || rows <= 0 || phi_q4 == nullptr) return out;
  for (int row = 0; row < rows; ++row) {
    __int128 acc = 128;
    for (int j = 0; j < n; ++j) {
      acc += static_cast<__int128>(phi_q4[static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(j)]) *
             head[j];
    }
    out[static_cast<size_t>(row)] = static_cast<int64_t>(acc >> 8);
  }
  return out;
}
#endif

}  // namespace vpxc
