#include "vpxc/wsil.hpp"

#include "vpxc/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace vpxc {
namespace {

int32_t narrow_i32(int64_t v) { return static_cast<int32_t>(v); }

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

void check_plane(const Plane& p) {
  if (p.height < 0 || p.width < 0) throw CodecError("negative plane shape");
  const size_t n = static_cast<size_t>(p.height) * static_cast<size_t>(p.width);
  if (p.samples.size() != n) throw CodecError("plane buffer size mismatch");
}

bool perm_ok(const int perm[3]) {
  bool seen[3] = {};
  for (int i = 0; i < 3; ++i) {
    if (perm[i] < 0 || perm[i] > 2 || seen[perm[i]]) return false;
    seen[perm[i]] = true;
  }
  return true;
}

// NumPy int64 r12, then astype(int32).
int32_t r12_i32(int64_t m, int32_t z) {
  const int64_t shifted = wadd(wmul(m, z), kQ12Bias) >> kQ12;
  return narrow_i32(shifted);
}

int32_t r12_fused(int m1, int32_t z1, int m2, int32_t z2) {
  const int64_t shifted = wadd(wadd(wmul(m1, z1), wmul(m2, z2)), kQ12Bias) >> kQ12;
  return narrow_i32(shifted);
}

void wsil2_forward(int32_t a1, int32_t a2, int s, int s_inv, int32_t& y1, int32_t& y2) {
  const int32_t u2 = wsub32(a2, r12_i32(s, a1));
  const int32_t u1 = wadd32(a1, r12_i32(static_cast<int64_t>(s_inv) - kQ12One, u2));
  y2 = wadd32(u2, u1);
  y1 = wadd32(u1, r12_i32(static_cast<int64_t>(s) - kQ12One, y2));
}

void wsil2_inverse(int32_t b1, int32_t b2, int s, int s_inv, int32_t& x1, int32_t& x2) {
  const int32_t u1 = wsub32(b1, r12_i32(static_cast<int64_t>(s) - kQ12One, b2));
  const int32_t u2 = wsub32(b2, u1);
  x1 = wsub32(u1, r12_i32(static_cast<int64_t>(s_inv) - kQ12One, u2));
  x2 = wadd32(u2, r12_i32(s, x1));
}

void check_trio(const Plane in[3]) {
  for (int i = 0; i < 3; ++i) check_plane(in[i]);
  if (in[1].height != in[0].height || in[1].width != in[0].width || in[2].height != in[0].height ||
      in[2].width != in[0].width) {
    throw CodecError("SL3 planes differ in shape");
  }
}

double det3(const double m[3][3]) {
  return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
         m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
         m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

// Cyclic Jacobi. Columns of V are eigenvectors; eigenvalues land on the diagonal of A.
void jacobi_sym3(double A[3][3], double eval[3], double V[3][3]) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) V[i][j] = (i == j) ? 1.0 : 0.0;
  for (int iter = 0; iter < 64; ++iter) {
    int p = 0;
    int q = 1;
    double best = std::fabs(A[0][1]);
    if (std::fabs(A[0][2]) > best) {
      best = std::fabs(A[0][2]);
      p = 0;
      q = 2;
    }
    if (std::fabs(A[1][2]) > best) {
      best = std::fabs(A[1][2]);
      p = 1;
      q = 2;
    }
    const double scale = std::fabs(A[0][0]) + std::fabs(A[1][1]) + std::fabs(A[2][2]) + 1.0;
    if (best <= 1e-15 * scale) break;
    const double apq = A[p][q];
    const double tau = (A[q][q] - A[p][p]) / (2.0 * apq);
    const double t = std::copysign(1.0, tau) / (std::fabs(tau) + std::sqrt(1.0 + tau * tau));
    const double c = 1.0 / std::sqrt(1.0 + t * t);
    const double s = t * c;
    const double app = A[p][p] - t * apq;
    const double aqq = A[q][q] + t * apq;
    A[p][p] = app;
    A[q][q] = aqq;
    A[p][q] = A[q][p] = 0.0;
    for (int r = 0; r < 3; ++r) {
      if (r == p || r == q) continue;
      const double arp = A[r][p];
      const double arq = A[r][q];
      A[r][p] = A[p][r] = c * arp - s * arq;
      A[r][q] = A[q][r] = c * arq + s * arp;
    }
    for (int r = 0; r < 3; ++r) {
      const double vrp = V[r][p];
      const double vrq = V[r][q];
      V[r][p] = c * vrp - s * vrq;
      V[r][q] = s * vrp + c * vrq;
    }
  }
  eval[0] = A[0][0];
  eval[1] = A[1][1];
  eval[2] = A[2][2];
}

int sub_len(int n, int step) {
  if (n <= 0) return 0;
  return (n + step - 1) / step;
}

}  // namespace

void sl3_forward_arrays(const Plane in[3], Plane out[3], const SL3Params& params) {
  check_trio(in);
  if (!perm_ok(params.perm)) throw CodecError("SL3 perm is not a permutation");
  const int h = in[0].height;
  const int w = in[0].width;
  const size_t n = static_cast<size_t>(h) * static_cast<size_t>(w);
  const int32_t* x0 = in[params.perm[0]].samples.data();
  const int32_t* x1 = in[params.perm[1]].samples.data();
  const int32_t* x2 = in[params.perm[2]].samples.data();
  Plane y0(h, w);
  Plane y1(h, w);
  Plane y2(h, w);
  for (size_t i = 0; i < n; ++i) {
    const int32_t c0 = wadd32(x0[i], r12_fused(params.u01, x1[i], params.u02, x2[i]));
    const int32_t c1 = wadd32(x1[i], r12_i32(params.u12, x2[i]));
    const int32_t c2 = x2[i];
    int32_t c1p = 0;
    int32_t c2p = 0;
    wsil2_forward(c1, c2, params.s2, params.s2_inv, c1p, c2p);
    int32_t c0pp = 0;
    int32_t c1pp = 0;
    wsil2_forward(c0, c1p, params.s1, params.s1_inv, c0pp, c1pp);
    y0.samples[i] = c0pp;
    y1.samples[i] = wadd32(c1pp, r12_i32(params.l10, c0pp));
    y2.samples[i] = wadd32(c2p, r12_fused(params.l20, c0pp, params.l21, c1pp));
  }
  out[0] = std::move(y0);
  out[1] = std::move(y1);
  out[2] = std::move(y2);
}

void sl3_inverse_arrays(const Plane in[3], Plane out[3], const SL3Params& params) {
  check_trio(in);
  if (!perm_ok(params.perm)) throw CodecError("SL3 perm is not a permutation");
  const int h = in[0].height;
  const int w = in[0].width;
  const size_t n = static_cast<size_t>(h) * static_cast<size_t>(w);
  const int32_t* y0 = in[0].samples.data();
  const int32_t* y1 = in[1].samples.data();
  const int32_t* y2 = in[2].samples.data();
  Plane xp0(h, w);
  Plane xp1(h, w);
  Plane xp2(h, w);
  for (size_t i = 0; i < n; ++i) {
    const int32_t c0pp = y0[i];
    const int32_t c1pp = wsub32(y1[i], r12_i32(params.l10, c0pp));
    const int32_t c2p = wsub32(y2[i], r12_fused(params.l20, c0pp, params.l21, c1pp));
    int32_t c0 = 0;
    int32_t c1p = 0;
    wsil2_inverse(c0pp, c1pp, params.s1, params.s1_inv, c0, c1p);
    int32_t c1 = 0;
    int32_t c2 = 0;
    wsil2_inverse(c1p, c2p, params.s2, params.s2_inv, c1, c2);
    xp2.samples[i] = c2;
    xp1.samples[i] = wsub32(c1, r12_i32(params.u12, c2));
    xp0.samples[i] = wsub32(c0, r12_fused(params.u01, xp1.samples[i], params.u02, c2));
  }
  Plane planes[3] = {std::move(xp0), std::move(xp1), std::move(xp2)};
  out[params.perm[0]] = std::move(planes[0]);
  out[params.perm[1]] = std::move(planes[1]);
  out[params.perm[2]] = std::move(planes[2]);
}

SL3Params factor_sl3_matrix(const double matrix[9]) {
  double mat[3][3];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) mat[r][c] = matrix[r * 3 + c];
  double det = det3(mat);
  if (!std::isfinite(det) || std::fabs(det) < 1e-8) return {};
  if (det < 0.0) {
    mat[2][0] = -mat[2][0];
    mat[2][1] = -mat[2][1];
    mat[2][2] = -mat[2][2];
    det = -det;
  }
  const double scale = std::pow(det, 1.0 / 3.0);
  if (!(scale > 0.0) || !std::isfinite(scale)) return {};
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) mat[r][c] /= scale;

  const int perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  const double flips[4][3] = {{1.0, 1.0, 1.0}, {1.0, -1.0, -1.0}, {-1.0, 1.0, -1.0}, {-1.0, -1.0, 1.0}};
  double best_score = std::numeric_limits<double>::infinity();
  SL3Params best{};
  for (const auto& perm : perms) {
    const int inv = (perm[0] > perm[1]) + (perm[0] > perm[2]) + (perm[1] > perm[2]);
    double wp[3][3];
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) wp[r][c] = mat[r][perm[c]];
    if (inv & 1) {
      wp[2][0] = -wp[2][0];
      wp[2][1] = -wp[2][1];
      wp[2][2] = -wp[2][2];
    }
    for (const auto& sf : flips) {
      double m[3][3];
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) m[r][c] = wp[r][c] * sf[r];
      const double d0 = m[0][0];
      if (d0 <= 0.15 || d0 >= 6.5) continue;
      const double u01 = m[0][1] / d0;
      const double u02 = m[0][2] / d0;
      const double l10 = m[1][0] / d0;
      const double l20 = m[2][0] / d0;
      const double schur11 = m[1][1] - l10 * d0 * u01;
      if (schur11 <= 0.15 || schur11 >= 6.5) continue;
      const double d1 = schur11;
      const double u12 = (m[1][2] - l10 * d0 * u02) / d1;
      const double l21 = (m[2][1] - l20 * d0 * u01) / d1;
      const double s1 = d0;
      const double s2 = d0 * d1;
      if (s2 <= 0.15 || s2 >= 6.5) continue;
      const double shears[6] = {u01, u02, u12, l10, l20, l21};
      double shear_mag = 0.0;
      for (double s : shears) shear_mag = std::max(shear_mag, std::fabs(s));
      if (shear_mag > 7.5) continue;
      const double score = std::fabs(std::log(s1)) + std::fabs(std::log(s2)) + 0.15 * shear_mag;
      if (score < best_score) {
        best_score = score;
        best = {};
        best.perm[0] = perm[0];
        best.perm[1] = perm[1];
        best.perm[2] = perm[2];
        best.u01 = q12(u01);
        best.u02 = q12(u02);
        best.u12 = q12(u12);
        best.l10 = q12(l10);
        best.l20 = q12(l20);
        best.l21 = q12(l21);
        best.s1 = std::max(1, q12(s1));
        best.s1_inv = std::max(1, q12(1.0 / s1));
        best.s2 = std::max(1, q12(s2));
        best.s2_inv = std::max(1, q12(1.0 / s2));
      }
    }
  }
  return best;
}

SL3Params estimate_sl3_wsil(const Plane rgb[3]) {
  check_trio(rgb);
  const int h = rgb[0].height;
  const int w = rgb[0].width;
  if (h == 0 || w == 0) return {};
  const int step_y = std::max(1, h / 256);
  const int step_x = std::max(1, w / 256);
  const int hs = sub_len(h, step_y);
  const int ws = sub_len(w, step_x);
  if (hs == 0 || ws == 0) return {};

  std::vector<double> sub(3 * static_cast<size_t>(hs) * static_cast<size_t>(ws));
  for (int c = 0; c < 3; ++c) {
    double* dst = sub.data() + static_cast<size_t>(c) * static_cast<size_t>(hs) * static_cast<size_t>(ws);
    for (int iy = 0; iy < hs; ++iy)
      for (int ix = 0; ix < ws; ++ix)
        dst[static_cast<size_t>(iy) * static_cast<size_t>(ws) + static_cast<size_t>(ix)] =
            static_cast<double>(rgb[c].at(iy * step_y, ix * step_x));
  }

  auto plane = [&](int c) -> const double* {
    return sub.data() + static_cast<size_t>(c) * static_cast<size_t>(hs) * static_cast<size_t>(ws);
  };

  std::vector<double> g;
  if (hs >= 3 && ws >= 3) {
    const int gh = hs - 2;
    const int gw = ws - 2;
    g.resize(static_cast<size_t>(gh) * static_cast<size_t>(gw) * 3);
    size_t row = 0;
    for (int iy = 1; iy < hs - 1; ++iy) {
      for (int ix = 1; ix < ws - 1; ++ix) {
        for (int c = 0; c < 3; ++c) {
          const double* s = plane(c);
          const size_t center = static_cast<size_t>(iy) * static_cast<size_t>(ws) + static_cast<size_t>(ix);
          const double neigh = s[static_cast<size_t>(iy - 1) * ws + ix] + s[static_cast<size_t>(iy + 1) * ws + ix] +
                               s[static_cast<size_t>(iy) * ws + (ix - 1)] + s[static_cast<size_t>(iy) * ws + (ix + 1)];
          g[row * 3 + static_cast<size_t>(c)] = s[center] - 0.25 * neigh;
        }
        ++row;
      }
    }
  } else {
    const bool horiz = ws >= 2;
    const bool vert = hs >= 2;
    if (horiz || vert) {
      const size_t n_h = horiz ? static_cast<size_t>(hs) * static_cast<size_t>(ws - 1) : 0;
      const size_t n_v = vert ? static_cast<size_t>(hs - 1) * static_cast<size_t>(ws) : 0;
      g.resize((n_h + n_v) * 3);
      size_t row = 0;
      if (horiz) {
        for (int iy = 0; iy < hs; ++iy) {
          for (int ix = 0; ix < ws - 1; ++ix) {
            for (int c = 0; c < 3; ++c) {
              const double* s = plane(c);
              const size_t i0 = static_cast<size_t>(iy) * ws + static_cast<size_t>(ix);
              g[row * 3 + static_cast<size_t>(c)] = s[i0 + 1] - s[i0];
            }
            ++row;
          }
        }
      }
      if (vert) {
        for (int iy = 0; iy < hs - 1; ++iy) {
          for (int ix = 0; ix < ws; ++ix) {
            for (int c = 0; c < 3; ++c) {
              const double* s = plane(c);
              const size_t i0 = static_cast<size_t>(iy) * ws + static_cast<size_t>(ix);
              g[row * 3 + static_cast<size_t>(c)] = s[i0 + static_cast<size_t>(ws)] - s[i0];
            }
            ++row;
          }
        }
      }
    } else {
      g.resize(static_cast<size_t>(hs) * static_cast<size_t>(ws) * 3);
      for (int c = 0; c < 3; ++c) {
        const double* s = plane(c);
        const size_t n = static_cast<size_t>(hs) * static_cast<size_t>(ws);
        double mean = 0.0;
        for (size_t i = 0; i < n; ++i) mean += s[i];
        mean /= static_cast<double>(n);
        for (size_t i = 0; i < n; ++i) g[i * 3 + static_cast<size_t>(c)] = s[i] - mean;
      }
    }
  }

  const size_t nrows = g.size() / 3;
  if (nrows < 4) return {};

  double cov[3][3] = {};
  for (size_t i = 0; i < nrows; ++i) {
    const double* row = g.data() + i * 3;
    for (int a = 0; a < 3; ++a)
      for (int b = a; b < 3; ++b) cov[a][b] += row[a] * row[b];
  }
  const double inv_n = 1.0 / static_cast<double>(nrows);
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      cov[a][b] *= inv_n;
      cov[b][a] = cov[a][b];
    }
    cov[a][a] += 1e-4;
  }

  double work[3][3];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) work[r][c] = cov[r][c];
  double eval[3];
  double evec[3][3];
  jacobi_sym3(work, eval, evec);
  if (!std::isfinite(eval[0]) || !std::isfinite(eval[1]) || !std::isfinite(eval[2])) return {};

  int order[3] = {0, 1, 2};
  std::sort(order, order + 3, [&](int i, int j) {
    if (eval[i] != eval[j]) return eval[i] > eval[j];
    return i < j;
  });
  double rot[3][3];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) rot[r][c] = evec[c][order[r]];
  if (det3(rot) < 0.0) {
    rot[2][0] = -rot[2][0];
    rot[2][1] = -rot[2][1];
    rot[2][2] = -rot[2][2];
  }
  double flat[9];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) flat[r * 3 + c] = rot[r][c];
  return factor_sl3_matrix(flat);
}

}  // namespace vpxc
