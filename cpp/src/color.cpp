#include "vpxc/color.hpp"

#include "vpxc/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace vpxc {
namespace {

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

int32_t r12_i32(int64_t m, int32_t z) {
  return static_cast<int32_t>(wadd(wmul(m, z), kQ12Bias) >> kQ12);
}

int32_t i32_abs(int32_t v) {
  if (v >= 0) return v;
  if (v == INT32_MIN) return v;
  return static_cast<int32_t>(-v);
}

void check_plane(const Plane& p) {
  if (p.height < 0 || p.width < 0) throw CodecError("negative plane shape");
  const size_t n = static_cast<size_t>(p.height) * static_cast<size_t>(p.width);
  if (p.samples.size() != n) throw CodecError("plane buffer size mismatch");
}

void check_same(const Plane& a, const Plane& b) {
  check_plane(a);
  check_plane(b);
  if (a.height != b.height || a.width != b.width) throw CodecError("channel shape mismatch");
}

int base_index(int perm) {
  if (perm == 0) return 1;
  if (perm == 1) return 0;
  if (perm == 2) return 2;
  int m = perm % 3;
  if (m < 0) m += 3;
  return m;
}

void other_indexes(int base, int& i0, int& i1) {
  int tmp[2];
  int n = 0;
  for (int i = 0; i < 3; ++i)
    if (i != base) tmp[n++] = i;
  i0 = tmp[0];
  i1 = tmp[1];
}

int sub_len(int n, int step) {
  if (n <= 0) return 0;
  return (n + step - 1) / step;
}

void ycocg_forward_px(int32_t r, int32_t g, int32_t b, int32_t& y, int32_t& co, int32_t& cg) {
  co = wsub32(r, b);
  const int32_t t = wadd32(b, co >> 1);
  cg = wsub32(g, t);
  y = wadd32(t, cg >> 1);
}

void ycocg_inverse_px(int32_t y, int32_t co, int32_t cg, int32_t& r, int32_t& g, int32_t& b) {
  const int32_t t = wsub32(y, cg >> 1);
  g = wadd32(cg, t);
  b = wsub32(t, co >> 1);
  r = wadd32(b, co);
}

void forward_rgb(const Plane rgb[3], const ColorTransform& tr, Plane out[3]) {
  const int h = rgb[0].height;
  const int w = rgb[0].width;
  const size_t n = rgb[0].samples.size();
  if (tr.mode == MODE_IDENTITY) {
    out[0] = rgb[0];
    out[1] = rgb[1];
    out[2] = rgb[2];
    return;
  }
  if (tr.mode == MODE_YCOCG) {
    out[0] = Plane(h, w);
    out[1] = Plane(h, w);
    out[2] = Plane(h, w);
    for (size_t i = 0; i < n; ++i) {
      int32_t y, co, cg;
      ycocg_forward_px(rgb[0].samples[i], rgb[1].samples[i], rgb[2].samples[i], y, co, cg);
      out[0].samples[i] = y;
      out[1].samples[i] = co;
      out[2].samples[i] = cg;
    }
    return;
  }
  if (tr.mode == MODE_SUB) {
    const int base = base_index(tr.perm);
    int i0, i1;
    other_indexes(base, i0, i1);
    out[0] = rgb[base];
    out[1] = Plane(h, w);
    out[2] = Plane(h, w);
    for (size_t i = 0; i < n; ++i) {
      out[1].samples[i] = wsub32(rgb[i0].samples[i], rgb[base].samples[i]);
      out[2].samples[i] = wsub32(rgb[i1].samples[i], rgb[base].samples[i]);
    }
    return;
  }
  if (tr.mode == MODE_SHEAR) {
    const int base = base_index(tr.perm);
    int i0, i1;
    other_indexes(base, i0, i1);
    const int a = tr.shear[0];
    const int c = tr.shear[1];
    const int d = tr.shear[2];
    out[0] = rgb[base];
    out[1] = Plane(h, w);
    out[2] = Plane(h, w);
    for (size_t i = 0; i < n; ++i) {
      const int32_t src = rgb[base].samples[i];
      const int32_t x = wsub32(rgb[i0].samples[i], r12_i32(a, src));
      const int32_t y = wsub32(wsub32(rgb[i1].samples[i], r12_i32(c, src)), r12_i32(d, x));
      out[1].samples[i] = x;
      out[2].samples[i] = y;
    }
    return;
  }
  if (tr.mode == MODE_WSIL) {
    sl3_forward_arrays(rgb, out, tr.wsil);
    return;
  }
  throw CodecError("unknown color mode " + std::to_string(tr.mode));
}

void inverse_rgb(const Plane head[3], const ColorTransform& tr, Plane out[3]) {
  const int h = head[0].height;
  const int w = head[0].width;
  const size_t n = head[0].samples.size();
  if (tr.mode == MODE_IDENTITY) {
    out[0] = head[0];
    out[1] = head[1];
    out[2] = head[2];
    return;
  }
  if (tr.mode == MODE_YCOCG) {
    out[0] = Plane(h, w);
    out[1] = Plane(h, w);
    out[2] = Plane(h, w);
    for (size_t i = 0; i < n; ++i) {
      int32_t r, g, b;
      ycocg_inverse_px(head[0].samples[i], head[1].samples[i], head[2].samples[i], r, g, b);
      out[0].samples[i] = r;
      out[1].samples[i] = g;
      out[2].samples[i] = b;
    }
    return;
  }
  if (tr.mode == MODE_SUB) {
    const int base = base_index(tr.perm);
    int i0, i1;
    other_indexes(base, i0, i1);
    Plane slots[3];
    slots[base] = head[0];
    slots[i0] = Plane(h, w);
    slots[i1] = Plane(h, w);
    for (size_t i = 0; i < n; ++i) {
      slots[i0].samples[i] = wadd32(head[1].samples[i], head[0].samples[i]);
      slots[i1].samples[i] = wadd32(head[2].samples[i], head[0].samples[i]);
    }
    out[0] = std::move(slots[0]);
    out[1] = std::move(slots[1]);
    out[2] = std::move(slots[2]);
    return;
  }
  if (tr.mode == MODE_SHEAR) {
    const int base = base_index(tr.perm);
    int i0, i1;
    other_indexes(base, i0, i1);
    const int a = tr.shear[0];
    const int c = tr.shear[1];
    const int d = tr.shear[2];
    Plane slots[3];
    slots[base] = head[0];
    slots[i0] = Plane(h, w);
    slots[i1] = Plane(h, w);
    for (size_t i = 0; i < n; ++i) {
      const int32_t src = head[0].samples[i];
      const int32_t x = head[1].samples[i];
      const int32_t y = head[2].samples[i];
      slots[i0].samples[i] = wadd32(x, r12_i32(a, src));
      slots[i1].samples[i] = wadd32(wadd32(y, r12_i32(d, x)), r12_i32(c, src));
    }
    out[0] = std::move(slots[0]);
    out[1] = std::move(slots[1]);
    out[2] = std::move(slots[2]);
    return;
  }
  if (tr.mode == MODE_WSIL) {
    sl3_inverse_arrays(head, out, tr.wsil);
    return;
  }
  throw CodecError("unknown color mode " + std::to_string(tr.mode));
}

bool weights_nonzero(const int w[3]) { return w[0] != 0 || w[1] != 0 || w[2] != 0; }

Plane extra_pred(const Plane head[3], const int weights[3]) {
  Plane out(head[0].height, head[0].width, 0);
  if (!weights_nonzero(weights)) return out;
  const size_t n = head[0].samples.size();
  for (size_t i = 0; i < n; ++i) {
    int64_t acc = wmul(weights[0], head[0].samples[i]);
    acc = wadd(acc, wmul(weights[1], head[1].samples[i]));
    acc = wadd(acc, wmul(weights[2], head[2].samples[i]));
    acc = wadd(acc, kQ12Bias);
    out.samples[i] = static_cast<int32_t>(acc >> kQ12);
  }
  return out;
}

double mad_plane(const Plane& p, int sy, int sx) {
  const int hs = sub_len(p.height, sy);
  const int ws = sub_len(p.width, sx);
  auto at = [&](int iy, int ix) -> int32_t { return p.at(iy * sy, ix * sx); };
  int64_t acc = 0;
  int count = 0;
  if (hs >= 3 && ws >= 3) {
    count = (hs - 2) * (ws - 2);
    for (int iy = 1; iy < hs - 1; ++iy) {
      for (int ix = 1; ix < ws - 1; ++ix) {
        int32_t sum = at(iy - 1, ix);
        sum = wadd32(sum, at(iy + 1, ix));
        sum = wadd32(sum, at(iy, ix - 1));
        sum = wadd32(sum, at(iy, ix + 1));
        sum = wadd32(sum, 2);
        const int32_t pred = sum >> 2;
        acc += static_cast<int64_t>(i32_abs(wsub32(at(iy, ix), pred)));
      }
    }
  } else if (ws >= 2) {
    count = hs * (ws - 1);
    for (int iy = 0; iy < hs; ++iy)
      for (int ix = 0; ix < ws - 1; ++ix)
        acc += static_cast<int64_t>(i32_abs(wsub32(at(iy, ix + 1), at(iy, ix))));
  } else if (hs >= 2) {
    count = (hs - 1) * ws;
    for (int iy = 0; iy < hs - 1; ++iy)
      for (int ix = 0; ix < ws; ++ix)
        acc += static_cast<int64_t>(i32_abs(wsub32(at(iy + 1, ix), at(iy, ix))));
  } else {
    count = hs * ws;
    for (int iy = 0; iy < hs; ++iy)
      for (int ix = 0; ix < ws; ++ix) acc += static_cast<int64_t>(i32_abs(at(iy, ix)));
  }
  if (count <= 0) return 0.0;
  return static_cast<double>(acc) / static_cast<double>(count);
}

double score_planes(const Plane* planes, int nplanes) {
  const Plane* used[8];
  int n = 0;
  for (int i = 0; i < nplanes; ++i) {
    if (planes[i].height > 0 && planes[i].width > 0) used[n++] = &planes[i];
  }
  if (n == 0) return 0.0;
  const int sy = std::max(1, used[0]->height / 256);
  const int sx = std::max(1, used[0]->width / 256);
  double total = 0.0;
  for (int i = 0; i < n; ++i) total += std::log2(mad_plane(*used[i], sy, sx) + 0.35);
  return total / static_cast<double>(n);
}

std::vector<double> subsample(const Plane& p, int sy, int sx, int& hs, int& ws) {
  hs = sub_len(p.height, sy);
  ws = sub_len(p.width, sx);
  std::vector<double> sub(static_cast<size_t>(hs) * static_cast<size_t>(ws));
  for (int iy = 0; iy < hs; ++iy)
    for (int ix = 0; ix < ws; ++ix)
      sub[static_cast<size_t>(iy) * static_cast<size_t>(ws) + static_cast<size_t>(ix)] =
          static_cast<double>(p.at(iy * sy, ix * sx));
  return sub;
}

std::vector<double> diffs_from_sub(const std::vector<double>& sub, int hs, int ws) {
  if (sub.empty()) return {};
  std::vector<double> parts;
  if (ws >= 2) {
    parts.reserve(parts.size() + static_cast<size_t>(hs) * static_cast<size_t>(ws - 1));
    for (int iy = 0; iy < hs; ++iy)
      for (int ix = 0; ix < ws - 1; ++ix) {
        const size_t i0 = static_cast<size_t>(iy) * static_cast<size_t>(ws) + static_cast<size_t>(ix);
        parts.push_back(sub[i0 + 1] - sub[i0]);
      }
  }
  if (hs >= 2) {
    for (int iy = 0; iy < hs - 1; ++iy)
      for (int ix = 0; ix < ws; ++ix) {
        const size_t i0 = static_cast<size_t>(iy) * static_cast<size_t>(ws) + static_cast<size_t>(ix);
        parts.push_back(sub[i0 + static_cast<size_t>(ws)] - sub[i0]);
      }
  }
  if (!parts.empty()) return parts;
  double mean = 0.0;
  for (double v : sub) mean += v;
  mean /= static_cast<double>(sub.size());
  parts.resize(sub.size());
  for (size_t i = 0; i < sub.size(); ++i) parts[i] = sub[i] - mean;
  return parts;
}

std::vector<double> diamond_residual(const Plane& p, int sy, int sx) {
  int hs = 0;
  int ws = 0;
  const std::vector<double> sub = subsample(p, sy, sx, hs, ws);
  if (hs >= 3 && ws >= 3) {
    std::vector<double> g(static_cast<size_t>(hs - 2) * static_cast<size_t>(ws - 2));
    size_t t = 0;
    for (int iy = 1; iy < hs - 1; ++iy) {
      for (int ix = 1; ix < ws - 1; ++ix) {
        const size_t c = static_cast<size_t>(iy) * static_cast<size_t>(ws) + static_cast<size_t>(ix);
        const double neigh = sub[static_cast<size_t>(iy - 1) * ws + ix] + sub[static_cast<size_t>(iy + 1) * ws + ix] +
                             sub[static_cast<size_t>(iy) * ws + (ix - 1)] + sub[static_cast<size_t>(iy) * ws + (ix + 1)];
        g[t++] = sub[c] - 0.25 * neigh;
      }
    }
    return g;
  }
  return diffs_from_sub(sub, hs, ws);
}

std::vector<double> diffs_1d(const Plane& p, int sy, int sx) {
  int hs = 0;
  int ws = 0;
  const std::vector<double> sub = subsample(p, sy, sx, hs, ws);
  return diffs_from_sub(sub, hs, ws);
}

bool solve_linear(double* A, double* b, int n) {
  for (int k = 0; k < n; ++k) {
    int piv = k;
    double best = std::fabs(A[k * n + k]);
    for (int i = k + 1; i < n; ++i) {
      const double v = std::fabs(A[i * n + k]);
      if (v > best) {
        best = v;
        piv = i;
      }
    }
    if (!(best > 0.0) || !std::isfinite(best)) return false;
    if (piv != k) {
      for (int j = k; j < n; ++j) std::swap(A[k * n + j], A[piv * n + j]);
      std::swap(b[k], b[piv]);
    }
    const double diag = A[k * n + k];
    for (int i = k + 1; i < n; ++i) {
      const double f = A[i * n + k] / diag;
      A[i * n + k] = 0.0;
      for (int j = k + 1; j < n; ++j) A[i * n + j] -= f * A[k * n + j];
      b[i] -= f * b[k];
    }
  }
  for (int i = n - 1; i >= 0; --i) {
    double s = b[i];
    for (int j = i + 1; j < n; ++j) s -= A[i * n + j] * b[j];
    const double diag = A[i * n + i];
    if (diag == 0.0 || !std::isfinite(diag)) return false;
    b[i] = s / diag;
  }
  return true;
}

void estimate_shear(const Plane& base, const Plane& x, const Plane& y, int shear[3]) {
  shear[0] = shear[1] = shear[2] = 0;
  const int sy = std::max(1, base.height / 256);
  const int sx = std::max(1, base.width / 256);
  const std::vector<double> db = diamond_residual(base, sy, sx);
  const std::vector<double> dx = diamond_residual(x, sy, sx);
  const std::vector<double> dy = diamond_residual(y, sy, sx);
  if (db.empty() || db.size() != dx.size() || db.size() != dy.size()) return;
  double bb = 0.0;
  double bd = 0.0;
  for (size_t i = 0; i < db.size(); ++i) {
    bb += db[i] * db[i];
    bd += db[i] * dx[i];
  }
  bb += 1e-6;
  const double a_val = bd / bb;
  const int aq = q12(a_val);
  const double scale = static_cast<double>(aq) / 4096.0;
  double g00 = 1e-6;
  double g01 = 0.0;
  double g11 = 1e-6;
  double r0 = 0.0;
  double r1 = 0.0;
  for (size_t i = 0; i < db.size(); ++i) {
    const double xres = dx[i] - scale * db[i];
    g00 += db[i] * db[i];
    g01 += db[i] * xres;
    g11 += xres * xres;
    r0 += db[i] * dy[i];
    r1 += xres * dy[i];
  }
  double A[4] = {g00, g01, g01, g11};
  double b[2] = {r0, r1};
  if (!solve_linear(A, b, 2) || !std::isfinite(b[0]) || !std::isfinite(b[1])) {
    shear[0] = aq;
    return;
  }
  shear[0] = aq;
  shear[1] = q12(b[0]);
  shear[2] = q12(b[1]);
}

void estimate_extra(const Plane head[3], const Plane& extra, int weights[3]) {
  weights[0] = weights[1] = weights[2] = 0;
  const int sy = std::max(1, extra.height / 256);
  const int sx = std::max(1, extra.width / 256);
  const std::vector<double> de = diffs_1d(extra, sy, sx);
  if (de.size() < 4) return;
  double abs_sum = 0.0;
  for (double v : de) abs_sum += std::fabs(v);
  if (abs_sum / static_cast<double>(de.size()) == 0.0) return;
  const std::vector<double> d0 = diffs_1d(head[0], sy, sx);
  const std::vector<double> d1 = diffs_1d(head[1], sy, sx);
  const std::vector<double> d2 = diffs_1d(head[2], sy, sx);
  if (d0.size() != de.size() || d1.size() != de.size() || d2.size() != de.size()) return;
  double G[9] = {};
  double rhs[3] = {};
  for (size_t i = 0; i < de.size(); ++i) {
    const double row[3] = {d0[i], d1[i], d2[i]};
    for (int a = 0; a < 3; ++a) {
      rhs[a] += row[a] * de[i];
      for (int b = 0; b < 3; ++b) G[a * 3 + b] += row[a] * row[b];
    }
  }
  G[0] += 1.0;
  G[4] += 1.0;
  G[8] += 1.0;
  if (!solve_linear(G, rhs, 3)) return;
  int trial[3] = {q12(rhs[0]), q12(rhs[1]), q12(rhs[2])};
  Plane lifted = extra;
  const Plane pred = extra_pred(head, trial);
  for (size_t i = 0; i < lifted.samples.size(); ++i) lifted.samples[i] = wsub32(lifted.samples[i], pred.samples[i]);
  if (score_planes(&lifted, 1) + 0.02 < score_planes(&extra, 1)) {
    weights[0] = trial[0];
    weights[1] = trial[1];
    weights[2] = trial[2];
  }
}

struct Candidate {
  double score = 0.0;
  int index = 0;
  ColorTransform transform{};
};

}  // namespace

ColorTransform choose_color_transform(const std::vector<Plane>& channels) {
  if (channels.size() < 3) return {};
  check_same(channels[0], channels[1]);
  check_same(channels[0], channels[2]);
  const Plane rgb[3] = {channels[0], channels[1], channels[2]};

  std::vector<Candidate> cands;
  cands.reserve(9);
  {
    Candidate c;
    c.score = score_planes(rgb, 3);
    c.index = 0;
    c.transform.mode = MODE_IDENTITY;
    cands.push_back(c);
  }
  {
    Plane ycg[3];
    ColorTransform id;
    id.mode = MODE_YCOCG;
    forward_rgb(rgb, id, ycg);
    Candidate c;
    c.score = score_planes(ycg, 3);
    c.index = 1;
    c.transform = id;
    cands.push_back(c);
  }
  const int perms[3] = {0, 1, 2};
  for (int idx = 0; idx < 3; ++idx) {
    const int perm = perms[idx];
    ColorTransform sub;
    sub.mode = MODE_SUB;
    sub.perm = perm;
    Plane sub_p[3];
    forward_rgb(rgb, sub, sub_p);
    Candidate cs;
    cs.score = score_planes(sub_p, 3) + 0.035;
    cs.index = 2 + idx;
    cs.transform = sub;
    cands.push_back(cs);

    const int base = base_index(perm);
    int i0, i1;
    other_indexes(base, i0, i1);
    int sh[3];
    estimate_shear(rgb[base], rgb[i0], rgb[i1], sh);
    ColorTransform shear;
    shear.mode = MODE_SHEAR;
    shear.perm = perm;
    shear.shear[0] = sh[0];
    shear.shear[1] = sh[1];
    shear.shear[2] = sh[2];
    Plane sh_p[3];
    forward_rgb(rgb, shear, sh_p);
    Candidate ch;
    ch.score = score_planes(sh_p, 3) + 0.04;
    ch.index = 5 + idx;
    ch.transform = shear;
    cands.push_back(ch);
  }
  {
    ColorTransform ws;
    ws.mode = MODE_WSIL;
    ws.wsil = estimate_sl3_wsil(rgb);
    Plane wp[3];
    forward_rgb(rgb, ws, wp);
    Candidate c;
    c.score = score_planes(wp, 3) + 0.06;
    c.index = 9;
    c.transform = ws;
    cands.push_back(c);
  }
  std::sort(cands.begin(), cands.end(), [](const Candidate& a, const Candidate& b) {
    if (a.score < b.score) return true;
    if (b.score < a.score) return false;
    return a.index < b.index;
  });
  ColorTransform chosen = cands.front().transform;
  if (channels.size() == 4) {
    check_same(channels[0], channels[3]);
    Plane head[3];
    forward_rgb(rgb, chosen, head);
    int ew[3];
    estimate_extra(head, channels[3], ew);
    chosen.extra[0] = ew[0];
    chosen.extra[1] = ew[1];
    chosen.extra[2] = ew[2];
  }
  return chosen;
}

std::vector<Plane> apply_color_forward(const std::vector<Plane>& channels, const ColorTransform& transform) {
  if (channels.size() < 3) return channels;
  check_same(channels[0], channels[1]);
  check_same(channels[0], channels[2]);
  const Plane rgb[3] = {channels[0], channels[1], channels[2]};
  Plane head[3];
  forward_rgb(rgb, transform, head);
  std::vector<Plane> out;
  out.reserve(channels.size());
  out.push_back(std::move(head[0]));
  out.push_back(std::move(head[1]));
  out.push_back(std::move(head[2]));
  if (channels.size() == 3) return out;
  for (size_t i = 3; i < channels.size(); ++i) {
    check_plane(channels[i]);
    out.push_back(channels[i]);
  }
  if (weights_nonzero(transform.extra)) {
    check_same(out[0], out[3]);
    Plane refs[3] = {out[0], out[1], out[2]};
    const Plane pred = extra_pred(refs, transform.extra);
    for (size_t i = 0; i < out[3].samples.size(); ++i)
      out[3].samples[i] = wsub32(out[3].samples[i], pred.samples[i]);
  }
  return out;
}

std::vector<Plane> apply_color_inverse(const std::vector<Plane>& planes, const ColorTransform& transform) {
  if (planes.size() < 3) return planes;
  check_same(planes[0], planes[1]);
  check_same(planes[0], planes[2]);
  Plane head[3] = {planes[0], planes[1], planes[2]};
  std::vector<Plane> extra;
  for (size_t i = 3; i < planes.size(); ++i) {
    check_plane(planes[i]);
    extra.push_back(planes[i]);
  }
  if (!extra.empty() && weights_nonzero(transform.extra)) {
    check_same(head[0], extra[0]);
    const Plane pred = extra_pred(head, transform.extra);
    for (size_t i = 0; i < extra[0].samples.size(); ++i)
      extra[0].samples[i] = wadd32(extra[0].samples[i], pred.samples[i]);
  }
  Plane rgb[3];
  inverse_rgb(head, transform, rgb);
  std::vector<Plane> out;
  out.reserve(planes.size());
  out.push_back(std::move(rgb[0]));
  out.push_back(std::move(rgb[1]));
  out.push_back(std::move(rgb[2]));
  for (Plane& e : extra) out.push_back(std::move(e));
  return out;
}

}  // namespace vpxc
