#include "vpxc/acquisition.hpp"

#include "vpxc/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace vpxc {
namespace {

constexpr int kKnotX[8] = {0, 36, 73, 109, 146, 182, 219, 255};
constexpr int kGammas[6] = {256, 410, 461, 512, 563, 614};
constexpr int kDy[24] = {-2, -2, -2, -2, -2, -1, -1, -1, -1, -1, 0, 0, 0, 0,
                         1,  1,  1,  1,  1,  2,  2,  2,  2,  2};
constexpr int kDx[24] = {-2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 1, 2,
                         -2, -1, 0, 1, 2, -2, -1, 0, 1, 2};

int64_t floor_div(int64_t a, int64_t b) {
  const int64_t q = a / b;
  const int64_t r = a % b;
  if (r != 0 && a < 0) return q - 1;
  return q;
}

int64_t round_half_even(double value) {
  if (!std::isfinite(value)) return 0;
  const double down = std::floor(value);
  const double frac = value - down;
  if (frac > 0.5) return static_cast<int64_t>(down) + 1;
  if (frac < 0.5) return static_cast<int64_t>(down);
  const int64_t base = static_cast<int64_t>(down);
  if ((base & 1LL) == 0) return base;
  return base + 1;
}

int py_bit_length(int64_t value) {
  uint64_t mag;
  if (value > 0) mag = static_cast<uint64_t>(value);
  else if (value == 0) return 0;
  else if (value == INT64_MIN) mag = 1ull << 63;
  else mag = static_cast<uint64_t>(-value);
  return 64 - __builtin_clzll(mag);
}

void lut_from_knots(const int knots[8], int64_t lut[256]) {
  int64_t y[8];
  for (int i = 0; i < 8; ++i) y[i] = knots[i];
  int64_t secant[7];
  for (int i = 0; i < 7; ++i) {
    const int64_t delta = y[i + 1] - y[i];
    const int64_t h = kKnotX[i + 1] - kKnotX[i];
    secant[i] = floor_div(delta * 256, h > 0 ? h : 1);
  }
  int64_t slope[8] = {};
  for (int i = 1; i < 7; ++i) {
    const int64_t s0 = secant[i - 1];
    const int64_t s1 = secant[i];
    if (s0 > 0 && s1 > 0) slope[i] = floor_div(2 * s0 * s1 + ((s0 + s1) >> 1), s0 + s1);
  }
  slope[0] = secant[0];
  slope[7] = secant[6];
  for (int seg = 0; seg < 7; ++seg) {
    const int x0 = kKnotX[seg];
    const int x1 = kKnotX[seg + 1];
    const int64_t y0 = y[seg];
    const int64_t y1 = y[seg + 1];
    const int span = x1 - x0;
    const int64_t lo = y0 < y1 ? y0 : y1;
    const int64_t hi = y1 > y0 ? y1 : y0;
    const int64_t m0 = slope[seg];
    const int64_t m1 = slope[seg + 1];
    const int last = seg == 6 ? x1 : x1 - 1;
    const int64_t dy = y1 - y0;
    for (int tpos = x0; tpos <= last; ++tpos) {
      const int64_t t = (static_cast<int64_t>(tpos - x0) << 8) / span;
      const int64_t c2 = 3 * dy - (((2 * m0 + m1) * span) >> 8);
      const int64_t c3 = (((m0 + m1) * span) >> 8) - 2 * dy;
      const int64_t p1 = ((c3 * t) >> 8) + c2;
      const int64_t p2 = ((p1 * t) >> 8) + ((m0 * span) >> 8);
      int64_t val = y0 + ((p2 * t) >> 8);
      if (val < lo) val = lo;
      else if (val > hi) val = hi;
      lut[tpos] = val;
    }
  }
}

void inverse_lut(const int64_t lut[256], int inv[4096]) {
  for (int i = 0; i < 4096; ++i) inv[i] = 255;
  int pos = 0;
  for (int i = 0; i < 256; ++i) {
    int end = static_cast<int>(lut[i]);
    if (end < 0) end = 0;
    if (end > 4095) end = 4095;
    if (end >= pos) {
      for (int k = pos; k <= end; ++k) inv[k] = i;
      pos = end + 1;
    }
  }
}

int64_t clamp_shift_index(int64_t sample, int shift) {
  int64_t idx = sample >> shift;
  if (idx < 0) return 0;
  if (idx > 255) return 255;
  return idx;
}

struct Factor {
  int n = 0;
  std::vector<double> V;
  std::vector<double> uc;
  std::vector<double> sig;
};

void jacobi_svd(std::vector<double> A, const std::vector<double>& rhs, int m, int n, Factor& out) {
  out.n = n;
  out.V.assign(static_cast<size_t>(n * n), 0.0);
  for (int i = 0; i < n; ++i) out.V[static_cast<size_t>(i * n + i)] = 1.0;
  const double eps = std::numeric_limits<double>::epsilon();
  const double eps_sq = eps * eps;
  for (int sweep = 0; sweep < 50; ++sweep) {
    bool rotated = false;
    for (int p = 0; p < n; ++p) {
      for (int q = p + 1; q < n; ++q) {
        double alpha = 0.0;
        double beta = 0.0;
        double gamma = 0.0;
        for (int i = 0; i < m; ++i) {
          const double ap = A[static_cast<size_t>(i * n + p)];
          const double aq = A[static_cast<size_t>(i * n + q)];
          alpha += ap * ap;
          beta += aq * aq;
          gamma += ap * aq;
        }
        if (gamma == 0.0 || gamma * gamma <= eps_sq * alpha * beta) continue;
        rotated = true;
        const double tau = (beta - alpha) / (2.0 * gamma);
        const double t = tau >= 0.0 ? 1.0 / (tau + std::sqrt(1.0 + tau * tau))
                                    : -1.0 / (-tau + std::sqrt(1.0 + tau * tau));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;
        for (int i = 0; i < m; ++i) {
          const size_t ip = static_cast<size_t>(i * n + p);
          const size_t iq = static_cast<size_t>(i * n + q);
          const double ap = A[ip];
          const double aq = A[iq];
          A[ip] = c * ap - s * aq;
          A[iq] = s * ap + c * aq;
        }
        for (int i = 0; i < n; ++i) {
          const size_t ip = static_cast<size_t>(i * n + p);
          const size_t iq = static_cast<size_t>(i * n + q);
          const double vp = out.V[ip];
          const double vq = out.V[iq];
          out.V[ip] = c * vp - s * vq;
          out.V[iq] = s * vp + c * vq;
        }
      }
    }
    if (!rotated) break;
  }
  out.sig.assign(static_cast<size_t>(n), 0.0);
  out.uc.assign(static_cast<size_t>(n), 0.0);
  for (int j = 0; j < n; ++j) {
    double ss = 0.0;
    double dot = 0.0;
    for (int i = 0; i < m; ++i) {
      const double a = A[static_cast<size_t>(i * n + j)];
      ss += a * a;
      dot += a * rhs[static_cast<size_t>(i)];
    }
    out.sig[static_cast<size_t>(j)] = std::sqrt(ss);
    out.uc[static_cast<size_t>(j)] = dot;
  }
}

void householder_qr(std::vector<double>& A, std::vector<double>& b, int m, int n) {
  const int steps = m < n ? m : n;
  for (int k = 0; k < steps; ++k) {
    double norm2 = 0.0;
    for (int i = k; i < m; ++i) {
      const double v = A[static_cast<size_t>(i) * static_cast<size_t>(n) + k];
      norm2 += v * v;
    }
    const double normx = std::sqrt(norm2);
    if (normx == 0.0) continue;
    const double x0 = A[static_cast<size_t>(k) * static_cast<size_t>(n) + k];
    const double sign = x0 >= 0.0 ? 1.0 : -1.0;
    std::vector<double> v(static_cast<size_t>(m - k));
    v[0] = x0 + sign * normx;
    for (int i = 1; i < m - k; ++i) {
      v[static_cast<size_t>(i)] = A[static_cast<size_t>(k + i) * static_cast<size_t>(n) + k];
    }
    double v2 = 0.0;
    for (double t : v) v2 += t * t;
    if (v2 == 0.0) continue;
    const double beta = 2.0 / v2;
    for (int j = k; j < n; ++j) {
      double dot = 0.0;
      for (int i = 0; i < m - k; ++i) {
        dot += v[static_cast<size_t>(i)] * A[static_cast<size_t>(k + i) * static_cast<size_t>(n) + j];
      }
      const double w = beta * dot;
      for (int i = 0; i < m - k; ++i) {
        A[static_cast<size_t>(k + i) * static_cast<size_t>(n) + j] -= v[static_cast<size_t>(i)] * w;
      }
    }
    double dotb = 0.0;
    for (int i = 0; i < m - k; ++i) dotb += v[static_cast<size_t>(i)] * b[static_cast<size_t>(k + i)];
    const double wb = beta * dotb;
    for (int i = 0; i < m - k; ++i) b[static_cast<size_t>(k + i)] -= v[static_cast<size_t>(i)] * wb;
  }
}

Factor factorize(std::vector<double> A, std::vector<double> b, int m, int n) {
  Factor out;
  out.n = n;
  if (m <= 0 || n <= 0) {
    out.V.assign(static_cast<size_t>(n * n), 0.0);
    out.uc.assign(static_cast<size_t>(n), 0.0);
    out.sig.assign(static_cast<size_t>(n), 0.0);
    return out;
  }
  if (m >= n) {
    householder_qr(A, b, m, n);
    std::vector<double> R(static_cast<size_t>(n * n));
    std::vector<double> c(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      c[static_cast<size_t>(i)] = b[static_cast<size_t>(i)];
      for (int j = 0; j < n; ++j) {
        R[static_cast<size_t>(i * n + j)] = A[static_cast<size_t>(i) * static_cast<size_t>(n) + j];
      }
    }
    jacobi_svd(std::move(R), c, n, n, out);
    return out;
  }
  jacobi_svd(std::move(A), b, m, n, out);
  return out;
}

void apply_factor(const Factor& factor, double tol, double* x) {
  for (int i = 0; i < factor.n; ++i) x[i] = 0.0;
  if (factor.sig.empty()) return;
  std::vector<double> y(static_cast<size_t>(factor.n), 0.0);
  for (int j = 0; j < factor.n; ++j) {
    const double sig = factor.sig[static_cast<size_t>(j)];
    if (sig > tol) y[static_cast<size_t>(j)] = factor.uc[static_cast<size_t>(j)] / (sig * sig);
  }
  for (int i = 0; i < factor.n; ++i) {
    double acc = 0.0;
    for (int j = 0; j < factor.n; ++j) {
      acc += factor.V[static_cast<size_t>(i * factor.n + j)] * y[static_cast<size_t>(j)];
    }
    x[i] = acc;
  }
}

void knots_for_gamma(int gamma_q8, int knots[8]) {
  const double gamma = static_cast<double>(gamma_q8) / 256.0;
  int acc = 0;
  for (int i = 0; i < 8; ++i) {
    int value = 0;
    if (i != 0) {
      const double pos = static_cast<double>(kKnotX[i]) / 255.0;
      value = static_cast<int>(round_half_even(std::pow(pos, gamma) * 4095.0));
    }
    if (value < acc) value = acc;
    acc = value;
    knots[i] = value;
  }
  if (knots[7] < 1) knots[7] = 1;
}

size_t count_odd(const Mask& odd) {
  size_t n = 0;
  for (int y = 0; y < odd.height; ++y) {
    for (int x = 0; x < odd.width; ++x) {
      if (odd.at(y, x)) ++n;
    }
  }
  return n;
}

}  // namespace

Plane apply_phi(const Plane& luma, const PhiAcq& phi) {
  if (phi.shift < 0 || phi.shift > 31) throw CodecError("acquisition shift out of range");
  if (luma.height < 0 || luma.width < 0) throw CodecError("negative plane shape");
  const size_t n = static_cast<size_t>(luma.height) * static_cast<size_t>(luma.width);
  if (luma.samples.size() < n) throw CodecError("plane storage is short");
  Plane out(luma.height, luma.width, 0);
  if (luma.height == 0 || luma.width == 0) return out;
  int64_t lut[256];
  lut_from_knots(phi.knots, lut);
  int inv[4096];
  inverse_lut(lut, inv);
  const int height = luma.height;
  const int width = luma.width;
  auto lin_at = [&](int y, int x) -> int64_t {
    if (y < 0) y = 0;
    if (x < 0) x = 0;
    if (y >= height) y = height - 1;
    if (x >= width) x = width - 1;
    return lut[clamp_shift_index(luma.at(y, x), phi.shift)];
  };
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int phase = ((y & 1) << 1) | (x & 1);
      int64_t acc = 0;
      for (int k = 0; k < 24; ++k) {
        acc += static_cast<int64_t>(phi.taps[phase][k]) * lin_at(y + kDy[k], x + kDx[k]);
      }
      int64_t pred = acc >> 6;
      if (pred < 0) pred = 0;
      else if (pred > 4095) pred = 4095;
      const int64_t value = static_cast<int64_t>(inv[pred]) << phi.shift;
      if (value < static_cast<int64_t>(INT32_MIN) || value > static_cast<int64_t>(INT32_MAX)) {
        throw CodecError("acquisition prediction does not fit in int32");
      }
      out.at(y, x) = static_cast<int32_t>(value);
    }
  }
  return out;
}

void write_phi(BitWriter& writer, const PhiAcq& phi) {
  writer.write_bits(static_cast<uint32_t>(phi.shift) & 15u, 4);
  for (int knot : phi.knots) writer.write_bits(static_cast<uint32_t>(knot) & 4095u, 12);
  for (int phase = 0; phase < 4; ++phase) {
    for (int k = 0; k < 24; ++k) writer.write_signed_golomb(phi.taps[phase][k]);
  }
}

PhiAcq read_phi(BitReader& reader) {
  PhiAcq phi;
  phi.shift = static_cast<int>(reader.read_bits(4));
  for (int i = 0; i < 8; ++i) phi.knots[i] = static_cast<int>(reader.read_bits(12));
  for (int i = 0; i < 7; ++i) {
    if (phi.knots[i] > phi.knots[i + 1]) throw CodecError("acquisition knots are not monotone");
  }
  for (int i = 0; i < 4 * 24; ++i) {
    const int value = reader.read_signed_golomb();
    if (value < -4096 || value > 4096) throw CodecError("acquisition tap out of range");
    phi.taps[i / 24][i % 24] = value;
  }
  return phi;
}

std::optional<PhiAcq> fit_phi(const Plane& luma, const Plane& chroma, const Plane& flat_pred, const Mask& odd) {
  if (luma.height != chroma.height || luma.width != chroma.width || luma.height != flat_pred.height ||
      luma.width != flat_pred.width || luma.height != odd.height || luma.width != odd.width) {
    throw CodecError("acquisition fit shapes differ");
  }
  if (luma.height < 0 || luma.width < 0) throw CodecError("negative plane shape");
  const size_t n = static_cast<size_t>(luma.height) * static_cast<size_t>(luma.width);
  if (luma.samples.size() < n || chroma.samples.size() < n || flat_pred.samples.size() < n || odd.bits.size() < n) {
    throw CodecError("plane storage is short");
  }
  if (count_odd(odd) < 2048) return std::nullopt;
  int64_t peak = luma.samples.empty() ? 0 : static_cast<int64_t>(luma.samples[0]);
  for (int y = 0; y < luma.height; ++y) {
    for (int x = 0; x < luma.width; ++x) {
      const int64_t a = luma.at(y, x);
      const int64_t b = chroma.at(y, x);
      if (a > peak) peak = a;
      if (b > peak) peak = b;
    }
  }
  const int shift = std::max(0, py_bit_length(peak) - 8);
  double err0 = 0.0;
  int n_odd = 0;
  for (int y = 0; y < luma.height; ++y) {
    for (int x = 0; x < luma.width; ++x) {
      if (!odd.at(y, x)) continue;
      const int64_t gap = static_cast<int64_t>(chroma.at(y, x)) - static_cast<int64_t>(flat_pred.at(y, x));
      err0 += static_cast<double>(gap >= 0 ? gap : -gap);
      ++n_odd;
    }
  }
  const double mad0 = err0 / static_cast<double>(n_odd) + 1.0;
  std::optional<PhiAcq> best;
  double best_mad = 0.0;
  constexpr int kCols = 24;
  for (int gamma : kGammas) {
    PhiAcq phi;
    phi.shift = shift;
    knots_for_gamma(gamma, phi.knots);
    int64_t lut[256];
    lut_from_knots(phi.knots, lut);
    std::vector<int64_t> lin(n);
    for (int y = 0; y < luma.height; ++y) {
      for (int x = 0; x < luma.width; ++x) {
        lin[static_cast<size_t>(y * luma.width + x)] =
            lut[clamp_shift_index(luma.at(y, x), shift)];
      }
    }
    auto lin_at = [&](int y, int x) -> double {
      if (y < 0) y = 0;
      if (x < 0) x = 0;
      if (y >= luma.height) y = luma.height - 1;
      if (x >= luma.width) x = luma.width - 1;
      return static_cast<double>(lin[static_cast<size_t>(y * luma.width + x)]);
    };
    std::vector<double> design[4];
    std::vector<double> target[4];
    int rows[4] = {};
    for (int y = 0; y < luma.height; ++y) {
      for (int x = 0; x < luma.width; ++x) {
        if (!odd.at(y, x)) continue;
        const int phase = ((y & 1) << 1) | (x & 1);
        for (int k = 0; k < kCols; ++k) design[phase].push_back(lin_at(y + kDy[k], x + kDx[k]));
        const int64_t idx = clamp_shift_index(chroma.at(y, x), shift);
        target[phase].push_back(static_cast<double>(lut[idx]));
        ++rows[phase];
      }
    }
    Factor factors[4];
    double s_max = 0.0;
    for (int phase = 0; phase < 4; ++phase) {
      factors[phase] = factorize(std::move(design[phase]), target[phase], rows[phase], kCols);
      for (double sig : factors[phase].sig) {
        if (sig > s_max) s_max = sig;
      }
    }
    const double tol =
        std::numeric_limits<double>::epsilon() * static_cast<double>(std::max(n_odd, 96)) * s_max;
    for (int phase = 0; phase < 4; ++phase) {
      double coef[kCols];
      apply_factor(factors[phase], tol, coef);
      for (int k = 0; k < kCols; ++k) {
        int64_t tap = round_half_even(coef[k] * 64.0);
        if (tap > 511) tap = 511;
        if (tap < -512) tap = -512;
        phi.taps[phase][k] = static_cast<int>(tap);
      }
    }
    const Plane pred = apply_phi(luma, phi);
    double err1 = 0.0;
    for (int y = 0; y < luma.height; ++y) {
      for (int x = 0; x < luma.width; ++x) {
        if (!odd.at(y, x)) continue;
        const int64_t gap = static_cast<int64_t>(chroma.at(y, x)) - static_cast<int64_t>(pred.at(y, x));
        err1 += static_cast<double>(gap >= 0 ? gap : -gap);
      }
    }
    const double mad1 = err1 / static_cast<double>(n_odd) + 1.0;
    if (!best || mad1 < best_mad) {
      best = phi;
      best_mad = mad1;
    }
  }
  if (!best) return std::nullopt;
  const double gain =
      static_cast<double>(n_odd) * (std::log2(mad0) - std::log2(best_mad));
  if (!(gain > 4.0 + 96.0 + 96.0 * 8.0 + 32.0)) return std::nullopt;
  return best;
}

}  // namespace vpxc
