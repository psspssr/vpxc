#include "vpxc/lattice.hpp"

#include "vpxc/error.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>

namespace vpxc {
namespace {

constexpr int kKappa8 = 12;
constexpr int kDirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

int32_t wadd32(int32_t a, int32_t b) {
  return static_cast<int32_t>(static_cast<uint32_t>(a) + static_cast<uint32_t>(b));
}

int32_t wsub32(int32_t a, int32_t b) {
  return static_cast<int32_t>(static_cast<uint32_t>(a) - static_cast<uint32_t>(b));
}

int64_t wmul(int64_t a, int64_t b) {
  return static_cast<int64_t>(static_cast<uint64_t>(a) * static_cast<uint64_t>(b));
}

int64_t wadd(int64_t a, int64_t b) {
  return static_cast<int64_t>(static_cast<uint64_t>(a) + static_cast<uint64_t>(b));
}

int64_t wsub(int64_t a, int64_t b) {
  return static_cast<int64_t>(static_cast<uint64_t>(a) - static_cast<uint64_t>(b));
}

int64_t floor_div(int64_t n, int64_t d) {
  int64_t q = n / d;
  const int64_t r = n % d;
  if (r != 0 && ((n < 0) != (d < 0))) --q;
  return q;
}

int64_t i64_abs(int64_t v) { return v < 0 ? -v : v; }

int32_t narrow_i32(int64_t v) { return static_cast<int32_t>(v); }

void check_plane(const Plane& p) {
  if (p.height < 0 || p.width < 0) throw CodecError("negative plane shape");
  const size_t n = static_cast<size_t>(p.height) * static_cast<size_t>(p.width);
  if (p.samples.size() != n) throw CodecError("plane buffer size mismatch");
}

void check_mask(const Mask& m) {
  if (m.height < 0 || m.width < 0) throw CodecError("negative mask shape");
  const size_t n = static_cast<size_t>(m.height) * static_cast<size_t>(m.width);
  if (m.bits.size() != n) throw CodecError("mask buffer size mismatch");
}

void check_pair(const Plane& img, const Mask& valid) {
  check_plane(img);
  check_mask(valid);
  if (img.height != valid.height || img.width != valid.width) throw CodecError("plane/mask shape mismatch");
}

int to_int(int64_t v) {
  if (v > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
      v < static_cast<int64_t>(std::numeric_limits<int>::min())) {
    throw CodecError("lattice coordinate overflow");
  }
  return static_cast<int>(v);
}

int bit_length_u64(uint64_t x) {
  if (x == 0) return 0;
  return 64 - std::countl_zero(x);
}

int32_t ilog2_one(int64_t value) {
  uint64_t x = value < 1 ? 1ull : static_cast<uint64_t>(value);
  int32_t res = 0;
  for (int shift : {32, 16, 8, 4, 2, 1}) {
    if (x >= (1ull << shift)) {
      x >>= shift;
      res += shift;
    }
  }
  return res;
}

int64_t adaptive_kappa2(bool any, int32_t mn, int32_t mx) {
  const int64_t base = static_cast<int64_t>(kKappa8) * kKappa8;
  if (!any) return base;
  const int64_t span = static_cast<int64_t>(mx) - static_cast<int64_t>(mn);
  if (span <= 512) return base;
  const int shift = std::max(0, bit_length_u64(static_cast<uint64_t>(span)) - 9);
  const int64_t kappa = static_cast<int64_t>(kKappa8) << shift;
  return kappa * kappa;
}

int64_t chord_weight(int64_t kappa2, int64_t delta, int shift) {
  int64_t den = wadd(kappa2, wmul(delta, delta));
  if (den < 1) den = 1;
  const int64_t num = kappa2 << shift;
  const int64_t w = floor_div(num, den);
  return w < 1 ? 1 : w;
}

struct Neigh {
  int64_t v[4];
  bool ok[4];
  int cnt = 0;
  int64_t acc = 0;
};

Neigh gather(const Plane& img, const Mask& accept, int y, int x) {
  Neigh n;
  for (int i = 0; i < 4; ++i) {
    const int ny = y + kDirs[i][0];
    const int nx = x + kDirs[i][1];
    if (ny < 0 || nx < 0 || ny >= img.height || nx >= img.width || !accept.at(ny, nx)) {
      n.v[i] = 0;
      n.ok[i] = false;
      continue;
    }
    n.v[i] = img.at(ny, nx);
    n.ok[i] = true;
    n.acc += n.v[i];
    ++n.cnt;
  }
  return n;
}

Plane update_term(const Plane& residual, const Mask& valid) {
  check_pair(residual, valid);
  const ParityMasks par = parity_masks(valid);
  Plane out(residual.height, residual.width, 0);
  for (int y = 0; y < residual.height; ++y) {
    for (int x = 0; x < residual.width; ++x) {
      int64_t acc = 0;
      for (int i = 0; i < 4; ++i) {
        const int ny = y + kDirs[i][0];
        const int nx = x + kDirs[i][1];
        if (ny < 0 || nx < 0 || ny >= residual.height || nx >= residual.width) continue;
        if (!par.odd.at(ny, nx)) continue;
        acc += residual.at(ny, nx);
      }
      out.at(y, x) = narrow_i32((acc + 4) >> 3);
    }
  }
  return out;
}

void even_range(const Mask& even, int64_t& u_min, int64_t& u_max, int64_t& v_min, int64_t& v_max, int64_t& count) {
  count = 0;
  for (int y = 0; y < even.height; ++y) {
    for (int x = 0; x < even.width; ++x) {
      if (!even.at(y, x)) continue;
      const int64_t u = (static_cast<int64_t>(y) + x) >> 1;
      const int64_t v = (static_cast<int64_t>(y) - x) >> 1;
      if (count == 0) {
        u_min = u_max = u;
        v_min = v_max = v;
      } else {
        u_min = std::min(u_min, u);
        u_max = std::max(u_max, u);
        v_min = std::min(v_min, v);
        v_max = std::max(v_max, v);
      }
      ++count;
    }
  }
}

}  // namespace

std::vector<int32_t> ilog2_int64(const std::vector<int64_t>& values) {
  std::vector<int32_t> out(values.size());
  for (size_t i = 0; i < values.size(); ++i) out[i] = ilog2_one(values[i]);
  return out;
}

ParityMasks parity_masks(const Mask& valid) {
  check_mask(valid);
  ParityMasks out;
  out.odd = Mask(valid.height, valid.width, false);
  out.even = Mask(valid.height, valid.width, false);
  for (int y = 0; y < valid.height; ++y) {
    for (int x = 0; x < valid.width; ++x) {
      if (!valid.at(y, x)) continue;
      if (((y + x) & 1) == 1) out.odd.at(y, x) = 1;
      else out.even.at(y, x) = 1;
    }
  }
  return out;
}

Plane predict(const Plane& img, const Mask& valid, bool sheaf) {
  check_pair(img, valid);
  const int h = img.height;
  const int w = img.width;
  const ParityMasks par = parity_masks(valid);
  int64_t kappa2 = 0;
  if (sheaf) {
    bool any = false;
    int32_t mn = 0;
    int32_t mx = 0;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        if (!par.even.at(y, x)) continue;
        const int32_t v = img.at(y, x);
        if (!any) {
          mn = mx = v;
          any = true;
        } else {
          mn = std::min(mn, v);
          mx = std::max(mx, v);
        }
      }
    }
    kappa2 = adaptive_kappa2(any, mn, mx);
  }

  Plane pred(h, w, 0);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const Neigh nb = gather(img, par.even, y, x);
      if (nb.cnt <= 0) continue;
      const int64_t flat = floor_div(nb.acc * 2 + nb.cnt, nb.cnt * 2);
      int64_t result = flat;
      if (sheaf && nb.cnt == 4) {
        const int64_t nu = nb.v[0];
        const int64_t nd = nb.v[1];
        const int64_t nl = nb.v[2];
        const int64_t nr = nb.v[3];
        const int64_t dv = i64_abs(wsub(nu, nd));
        const int64_t dh = i64_abs(wsub(nl, nr));
        const int64_t d_nw = wadd(i64_abs(wsub(nu, nl)), i64_abs(wsub(nd, nr)));
        const int64_t d_ne = wadd(i64_abs(wsub(nu, nr)), i64_abs(wsub(nd, nl)));
        const int64_t wv = chord_weight(kappa2, dv, 10);
        const int64_t wh = chord_weight(kappa2, dh, 10);
        const int64_t num2 = wadd(wmul(wv, wadd(nu, nd)), wmul(wh, wadd(nl, nr)));
        const int64_t den2 = wmul(2, wadd(wv, wh));
        result = floor_div(wadd(wmul(2, num2), den2), wmul(2, den2));
        const int64_t d2 = wmul(dv, dv);
        const int64_t four_k = wmul(4, kappa2);
        const bool diag = dv == dh && d2 > four_k && (wmul(d_nw, 4) < dv || wmul(d_ne, 4) < dv);
        if (diag) {
          result = d_nw <= d_ne ? (wadd(nu, nl) >> 1) : (wadd(nu, nr) >> 1);
        }
      } else if (sheaf && nb.cnt == 3) {
        constexpr int64_t kBig = int64_t{1} << 60;
        int64_t vmin = kBig;
        int64_t vmax = -kBig;
        for (int i = 0; i < 4; ++i) {
          if (!nb.ok[i]) continue;
          vmin = std::min(vmin, nb.v[i]);
          vmax = std::max(vmax, nb.v[i]);
        }
        const int64_t med = nb.acc - vmin - vmax;
        int64_t wsum = 0;
        int64_t accw = 0;
        for (int i = 0; i < 4; ++i) {
          if (!nb.ok[i]) continue;
          const int64_t wt = chord_weight(kappa2, wsub(nb.v[i], med), 8);
          wsum = wadd(wsum, wt);
          accw = wadd(accw, wmul(wt, nb.v[i]));
        }
        result = floor_div(wadd(wmul(2, accw), wsum), wmul(2, wsum));
      }
      pred.at(y, x) = narrow_i32(result);
    }
  }
  return pred;
}

Plane lift_forward(const Plane& samples, const Mask& valid, bool sheaf) {
  check_pair(samples, valid);
  const Plane pred = predict(samples, valid, sheaf);
  const ParityMasks par = parity_masks(valid);
  Plane out = samples;
  const int h = samples.height;
  const int w = samples.width;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      if (par.odd.at(y, x)) out.at(y, x) = wsub32(samples.at(y, x), pred.at(y, x));
  const Plane upd = update_term(out, valid);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      if (par.even.at(y, x)) out.at(y, x) = wadd32(samples.at(y, x), upd.at(y, x));
  return out;
}

Plane lift_inverse(const Plane& lifted, const Mask& valid, bool sheaf) {
  check_pair(lifted, valid);
  const ParityMasks par = parity_masks(valid);
  const Plane upd = update_term(lifted, valid);
  Plane restored = lifted;
  const int h = lifted.height;
  const int w = lifted.width;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      if (par.even.at(y, x)) restored.at(y, x) = wsub32(lifted.at(y, x), upd.at(y, x));
  const Plane pred = predict(restored, valid, sheaf);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      if (par.odd.at(y, x)) restored.at(y, x) = wadd32(lifted.at(y, x), pred.at(y, x));
  return restored;
}

PackEven pack_even(const Plane& img, const Mask& valid) {
  check_pair(img, valid);
  const ParityMasks par = parity_masks(valid);
  int64_t u_min = 0, u_max = 0, v_min = 0, v_max = 0, count = 0;
  even_range(par.even, u_min, u_max, v_min, v_max, count);
  PackEven result;
  result.info.parent_h = valid.height;
  result.info.parent_w = valid.width;
  if (count == 0) {
    result.packed = Plane(1, 1, 0);
    result.mask = Mask(1, 1, false);
    result.info.u_shift = 0;
    result.info.v_shift = 0;
    result.info.packed_h = 1;
    result.info.packed_w = 1;
    return result;
  }
  const int64_t u_shift = -u_min;
  const int64_t v_shift = -v_min;
  const int hs = to_int(u_max + u_shift + 1);
  const int ws = to_int(v_max + v_shift + 1);
  result.info.u_shift = to_int(u_shift);
  result.info.v_shift = to_int(v_shift);
  result.info.packed_h = hs;
  result.info.packed_w = ws;
  result.packed = Plane(hs, ws, 0);
  result.mask = Mask(hs, ws, false);
  for (int y = 0; y < valid.height; ++y) {
    for (int x = 0; x < valid.width; ++x) {
      if (!par.even.at(y, x)) continue;
      const int64_t u = (static_cast<int64_t>(y) + x) >> 1;
      const int64_t v = (static_cast<int64_t>(y) - x) >> 1;
      const int us = to_int(u + u_shift);
      const int vs = to_int(v + v_shift);
      result.packed.at(us, vs) = img.at(y, x);
      result.mask.at(us, vs) = 1;
    }
  }
  return result;
}

Plane unpack_even(const Plane& packed, const PackInfo& info, const Mask& parent_valid) {
  check_plane(packed);
  check_mask(parent_valid);
  if (parent_valid.height != info.parent_h || parent_valid.width != info.parent_w) {
    throw CodecError("unpack parent shape mismatch");
  }
  if (packed.height != info.packed_h || packed.width != info.packed_w) {
    throw CodecError("unpack packed shape mismatch");
  }
  Plane spatial(info.parent_h, info.parent_w, 0);
  const ParityMasks par = parity_masks(parent_valid);
  for (int y = 0; y < parent_valid.height; ++y) {
    for (int x = 0; x < parent_valid.width; ++x) {
      if (!par.even.at(y, x)) continue;
      const int64_t u = (static_cast<int64_t>(y) + x) >> 1;
      const int64_t v = (static_cast<int64_t>(y) - x) >> 1;
      const int us = to_int(u + info.u_shift);
      const int vs = to_int(v + info.v_shift);
      if (!packed.inside(us, vs)) throw CodecError("unpack index out of range");
      spatial.at(y, x) = packed.at(us, vs);
    }
  }
  return spatial;
}

ContextMap activity_context(const Plane& lowpass, const Mask& valid) {
  check_pair(lowpass, valid);
  const ParityMasks par = parity_masks(valid);
  ContextMap ctx;
  ctx.height = lowpass.height;
  ctx.width = lowpass.width;
  ctx.samples.assign(static_cast<size_t>(ctx.height) * static_cast<size_t>(ctx.width), 0);
  for (int y = 0; y < ctx.height; ++y) {
    for (int x = 0; x < ctx.width; ++x) {
      const Neigh nb = gather(lowpass, par.even, y, x);
      const bool has_v = nb.ok[0] && nb.ok[1];
      const bool has_h = nb.ok[2] && nb.ok[3];
      int64_t dv = 0;
      int64_t dh = 0;
      if (has_v) dv = i64_abs(wsub(nb.v[0], nb.v[1]));
      if (has_h) dh = i64_abs(wsub(nb.v[2], nb.v[3]));
      int64_t saddle = 0;
      int64_t min_chord = 0;
      int64_t one = 0;
      if (has_v && has_h) {
        saddle = i64_abs(wsub(wadd(nb.v[0], nb.v[1]), wadd(nb.v[2], nb.v[3])));
        min_chord = std::min(dv, dh);
      } else if (has_v) {
        one = dv >> 1;
      } else if (has_h) {
        one = dh >> 1;
      }
      const int64_t act = (saddle >> 1) + (min_chord >> 1) + one;
      if (act > 0) {
        int bits = ilog2_one(act);
        if (bits > 7) bits = 7;
        if (bits < 0) bits = 0;
        ctx.samples[static_cast<size_t>(y) * static_cast<size_t>(ctx.width) + static_cast<size_t>(x)] =
            static_cast<uint8_t>(bits);
      }
    }
  }
  return ctx;
}

std::pair<std::vector<Level>, Mask> build_levels(int h, int w, int max_levels) {
  if (h < 0 || w < 0) throw CodecError("negative shape");
  Mask valid(h, w, true);
  std::vector<Level> levels;
  if (max_levels < 0) max_levels = 0;
  for (int n = 0; n < max_levels; ++n) {
    const ParityMasks par = parity_masks(valid);
    int64_t n_odd = 0;
    int64_t n_even = 0;
    for (uint8_t bit : par.odd.bits)
      if (bit) ++n_odd;
    for (uint8_t bit : par.even.bits)
      if (bit) ++n_even;
    if (n_odd < 1 || n_even < 4) break;
    if (std::min(valid.height, valid.width) < 3) break;
    const Plane dummy(valid.height, valid.width, 0);
    PackEven packed = pack_even(dummy, valid);
    const int64_t psize = static_cast<int64_t>(packed.mask.height) * packed.mask.width;
    const int64_t vsize = static_cast<int64_t>(valid.height) * valid.width;
    if (vsize > 64 && psize > 8 * vsize) break;
    Level level;
    level.info = packed.info;
    level.packed_mask = packed.mask;
    level.valid = valid;
    valid = packed.mask;
    levels.push_back(std::move(level));
  }
  return {std::move(levels), std::move(valid)};
}

Tower forward_tower(const Plane& plane, int max_levels, bool sheaf) {
  check_plane(plane);
  Tower tower;
  tower.levels = build_levels(plane.height, plane.width, max_levels).first;
  if (tower.levels.empty()) {
    tower.coarse = plane;
    tower.coarse_valid = Mask(plane.height, plane.width, true);
    return tower;
  }
  Plane current = plane;
  tower.details.reserve(tower.levels.size());
  for (const Level& level : tower.levels) {
    Plane lifted = lift_forward(current, level.valid, sheaf);
    tower.details.push_back(lifted);
    current = pack_even(lifted, level.valid).packed;
  }
  tower.coarse = std::move(current);
  tower.coarse_valid = tower.levels.back().packed_mask;
  return tower;
}

Plane inverse_tower(const Plane& coarse, const std::vector<Plane>& details, const std::vector<Level>& levels,
                    bool sheaf) {
  check_plane(coarse);
  if (details.size() != levels.size()) throw CodecError("tower rank mismatch");
  Plane current = coarse;
  if (levels.empty()) return current;
  for (int i = static_cast<int>(levels.size()) - 1; i >= 0; --i) {
    const Level& level = levels[static_cast<size_t>(i)];
    const Plane& lifted = details[static_cast<size_t>(i)];
    check_pair(lifted, level.valid);
    Plane spatial = unpack_even(current, level.info, level.valid);
    const ParityMasks par = parity_masks(level.valid);
    for (int y = 0; y < level.valid.height; ++y)
      for (int x = 0; x < level.valid.width; ++x)
        if (par.odd.at(y, x)) spatial.at(y, x) = lifted.at(y, x);
    current = lift_inverse(spatial, level.valid, sheaf);
  }
  return current;
}

}  // namespace vpxc
