#include "vpxc/issm.hpp"

#include "vpxc/error.hpp"

#include <algorithm>
#include <cstdint>

namespace vpxc {
namespace {

constexpr int kChunk = 16;
constexpr int64_t kQ16One = 1 << 16;
constexpr int64_t kQ16MaxContractive = kQ16One - 64;
constexpr int64_t kQ30One = 1 << 30;

int64_t wmul(int64_t a, int64_t b) {
  return static_cast<int64_t>(static_cast<uint64_t>(a) * static_cast<uint64_t>(b));
}

int64_t wadd(int64_t a, int64_t b) {
  return static_cast<int64_t>(static_cast<uint64_t>(a) + static_cast<uint64_t>(b));
}

// NumPy int64 floor division (toward -infinity).
int64_t floor_div(int64_t n, int64_t d) {
  int64_t q = n / d;
  const int64_t r = n % d;
  if (r != 0 && ((n < 0) != (d < 0))) --q;
  return q;
}

int64_t step_q16(int64_t a, int64_t state, int64_t b) {
  return wadd(wadd(wmul(a, state), int64_t{1} << 15) >> 16, b);
}

int64_t compose_alpha(int64_t alpha_r, int64_t alpha_l) {
  return wadd(wmul(alpha_r, alpha_l), int64_t{1} << 29) >> 30;
}

int64_t compose_beta(int64_t alpha_r, int64_t beta_l, int64_t beta_r) {
  return wadd(wadd(wmul(alpha_r, beta_l), int64_t{1} << 29) >> 30, beta_r);
}

void check_len(const std::vector<int64_t>& decay, const std::vector<int64_t>& bias) {
  if (decay.size() != bias.size()) throw CodecError("issm decay/bias length mismatch");
  if (decay.size() > static_cast<size_t>(INT32_MAX)) throw CodecError("issm sequence too long");
}

int32_t store_i32(int64_t v) { return static_cast<int32_t>(v); }

}  // namespace

std::vector<int64_t> sequential(const std::vector<int64_t>& decay, const std::vector<int64_t>& bias) {
  check_len(decay, bias);
  std::vector<int64_t> out(decay.size());
  int64_t state = 0;
  for (size_t t = 0; t < decay.size(); ++t) {
    state = step_q16(decay[t], state, bias[t]);
    out[t] = state;
  }
  return out;
}

std::vector<int64_t> tree(const std::vector<int64_t>& decay, const std::vector<int64_t>& bias) {
  check_len(decay, bias);
  const int length = static_cast<int>(decay.size());
  if (length == 0) return {};

  const int n_chunks = (length + kChunk - 1) / kChunk;
  int width = 1;
  while (width < n_chunks) width *= 2;

  std::vector<int64_t> alpha(static_cast<size_t>(width), kQ30One);
  std::vector<int64_t> beta(static_cast<size_t>(width), 0);
  for (int c = 0; c < n_chunks; ++c) {
    int64_t a_work = kQ30One;
    int64_t b_work = 0;
    for (int k = 0; k < kChunk; ++k) {
      const int t = c * kChunk + k;
      const int64_t ak = t < length ? decay[static_cast<size_t>(t)] : kQ16One;
      const int64_t bk = t < length ? bias[static_cast<size_t>(t)] : 0;
      a_work = wadd(wmul(ak, a_work), int64_t{1} << 15) >> 16;
      b_work = step_q16(ak, b_work, bk);
    }
    alpha[static_cast<size_t>(c)] = a_work;
    beta[static_cast<size_t>(c)] = b_work;
  }

  for (int step = 1; step < width; step *= 2) {
    for (int left = step - 1; left < width; left += 2 * step) {
      const int right = left + step;
      const int64_t ar = alpha[static_cast<size_t>(right)];
      const int64_t br = beta[static_cast<size_t>(right)];
      const int64_t al = alpha[static_cast<size_t>(left)];
      const int64_t bl = beta[static_cast<size_t>(left)];
      alpha[static_cast<size_t>(right)] = compose_alpha(ar, al);
      beta[static_cast<size_t>(right)] = compose_beta(ar, bl, br);
    }
  }

  alpha[static_cast<size_t>(width - 1)] = kQ30One;
  beta[static_cast<size_t>(width - 1)] = 0;

  for (int step = width / 2; step >= 1; step /= 2) {
    for (int left = step - 1; left < width; left += 2 * step) {
      const int right = left + step;
      const int64_t saved_a = alpha[static_cast<size_t>(left)];
      const int64_t saved_b = beta[static_cast<size_t>(left)];
      alpha[static_cast<size_t>(left)] = alpha[static_cast<size_t>(right)];
      beta[static_cast<size_t>(left)] = beta[static_cast<size_t>(right)];
      const int64_t ar = alpha[static_cast<size_t>(right)];
      const int64_t br = beta[static_cast<size_t>(right)];
      alpha[static_cast<size_t>(right)] = compose_alpha(saved_a, ar);
      beta[static_cast<size_t>(right)] = compose_beta(saved_a, br, saved_b);
    }
  }

  std::vector<int64_t> out(static_cast<size_t>(length));
  for (int c = 0; c < n_chunks; ++c) {
    int64_t state = beta[static_cast<size_t>(c)];
    for (int k = 0; k < kChunk; ++k) {
      const int t = c * kChunk + k;
      const int64_t ak = t < length ? decay[static_cast<size_t>(t)] : kQ16One;
      const int64_t bk = t < length ? bias[static_cast<size_t>(t)] : 0;
      state = step_q16(ak, state, bk);
      if (t < length) out[static_cast<size_t>(t)] = state;
    }
  }
  return out;
}

std::array<Plane, 4> scan_4way(const Plane& plane, int base_decay, int edge_kappa) {
  if (plane.height < 0 || plane.width < 0) throw CodecError("negative plane shape");
  const size_t n = static_cast<size_t>(plane.height) * static_cast<size_t>(plane.width);
  if (plane.samples.size() != n) throw CodecError("plane buffer size mismatch");
  const int h = plane.height;
  const int w = plane.width;
  std::array<Plane, 4> out = {Plane(h, w), Plane(h, w), Plane(h, w), Plane(h, w)};
  if (h == 0 || w == 0) return out;

  int64_t a0 = base_decay;
  if (a0 < 0) a0 = 0;
  if (a0 > kQ16MaxContractive) a0 = kQ16MaxContractive;
  const int64_t ek = edge_kappa;
  int64_t k2 = ek * ek;
  if (k2 < 1) k2 = 1;
  // Numerator is a Python int. Reject values NumPy cannot cast into the int64 decay.
  if (a0 > 0 && k2 > INT64_MAX / a0) throw CodecError("edge_kappa exceeds int64 decay");
  const int64_t num = a0 * k2;

  auto sweep = [&](const int64_t* row, int len, int64_t* dest) {
    std::vector<int64_t> decay(static_cast<size_t>(len));
    std::vector<int64_t> bias(static_cast<size_t>(len));
    decay[0] = 0;
    bias[0] = wmul(kQ16One, row[0]);
    for (int x = 1; x < len; ++x) {
      int64_t diff = row[x] - row[x - 1];
      if (diff < 0) diff = -diff;
      int64_t den = wadd(k2, wmul(diff, diff));
      if (den < 1) den = 1;
      const int64_t dec = floor_div(num, den);
      decay[static_cast<size_t>(x)] = dec;
      bias[static_cast<size_t>(x)] = wmul(kQ16One - dec, row[x]);
    }
    const std::vector<int64_t> states = tree(decay, bias);
    for (int x = 0; x < len; ++x) dest[x] = wadd(states[static_cast<size_t>(x)], int64_t{1} << 11) >> 12;
  };

  std::vector<int64_t> seq(static_cast<size_t>(std::max(h, w)));
  std::vector<int64_t> dest(seq.size());
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) seq[static_cast<size_t>(x)] = plane.at(y, x);
    sweep(seq.data(), w, dest.data());
    for (int x = 0; x < w; ++x) out[0].at(y, x) = store_i32(dest[static_cast<size_t>(x)]);
    for (int x = 0; x < w; ++x) seq[static_cast<size_t>(x)] = plane.at(y, w - 1 - x);
    sweep(seq.data(), w, dest.data());
    for (int x = 0; x < w; ++x) out[1].at(y, w - 1 - x) = store_i32(dest[static_cast<size_t>(x)]);
  }
  for (int x = 0; x < w; ++x) {
    for (int y = 0; y < h; ++y) seq[static_cast<size_t>(y)] = plane.at(y, x);
    sweep(seq.data(), h, dest.data());
    for (int y = 0; y < h; ++y) out[2].at(y, x) = store_i32(dest[static_cast<size_t>(y)]);
    for (int y = 0; y < h; ++y) seq[static_cast<size_t>(y)] = plane.at(h - 1 - y, x);
    sweep(seq.data(), h, dest.data());
    for (int y = 0; y < h; ++y) out[3].at(h - 1 - y, x) = store_i32(dest[static_cast<size_t>(y)]);
  }
  return out;
}

}  // namespace vpxc
