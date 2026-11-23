#include "vpxc/mdl.hpp"

#include "vpxc/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace vpxc {
namespace {

int64_t iabs(int64_t v) {
  if (v >= 0) return v;
  if (v == INT64_MIN) return INT64_MAX;
  return -v;
}

int bit_length_u64(uint64_t value) {
  if (value == 0) return 0;
  return 64 - __builtin_clzll(value);
}

int golomb_bits(int64_t value) {
  uint64_t mag;
  if (value >= 0) mag = (static_cast<uint64_t>(value) << 1) + 1;
  else if (value == INT64_MIN) mag = (static_cast<uint64_t>(INT64_MAX) << 1) + 1;
  else mag = (static_cast<uint64_t>(-value) << 1) + 1;
  return 2 * bit_length_u64(mag);
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

bool solve_ge(std::vector<double>& a, std::vector<double>& b, int n, std::vector<double>& x) {
  for (int k = 0; k < n; ++k) {
    int pivot = k;
    double best = std::fabs(a[static_cast<size_t>(k * n + k)]);
    for (int i = k + 1; i < n; ++i) {
      const double value = std::fabs(a[static_cast<size_t>(i * n + k)]);
      if (value > best) {
        best = value;
        pivot = i;
      }
    }
    if (!(best > 0.0) || !std::isfinite(best)) return false;
    if (pivot != k) {
      for (int j = k; j < n; ++j) {
        std::swap(a[static_cast<size_t>(k * n + j)], a[static_cast<size_t>(pivot * n + j)]);
      }
      std::swap(b[static_cast<size_t>(k)], b[static_cast<size_t>(pivot)]);
    }
    const double akk = a[static_cast<size_t>(k * n + k)];
    for (int i = k + 1; i < n; ++i) {
      const double factor = a[static_cast<size_t>(i * n + k)] / akk;
      for (int j = k + 1; j < n; ++j) {
        a[static_cast<size_t>(i * n + j)] -= factor * a[static_cast<size_t>(k * n + j)];
      }
      a[static_cast<size_t>(i * n + k)] = 0.0;
      b[static_cast<size_t>(i)] -= factor * b[static_cast<size_t>(k)];
    }
  }
  x.assign(static_cast<size_t>(n), 0.0);
  for (int i = n - 1; i >= 0; --i) {
    double sum = b[static_cast<size_t>(i)];
    for (int j = i + 1; j < n; ++j) sum -= a[static_cast<size_t>(i * n + j)] * x[static_cast<size_t>(j)];
    const double diag = a[static_cast<size_t>(i * n + i)];
    if (diag == 0.0 || !std::isfinite(diag) || !std::isfinite(sum)) return false;
    x[static_cast<size_t>(i)] = sum / diag;
    if (!std::isfinite(x[static_cast<size_t>(i)])) return false;
  }
  return true;
}

const double* neg_log_table() {
  static const std::vector<double> table = [] {
    std::vector<double> values(65536);
    for (int i = 0; i < 65536; ++i) values[static_cast<size_t>(i)] = -std::log2((i + 1) / 65536.0);
    return values;
  }();
  return table.data();
}

const int32_t* freq_row(int nu, int sigma, int frac) {
  return get_parametric_tables().freqs[nu][sigma][frac];
}

int argmin_sigma(double target) {
  int best = 0;
  double best_d = std::fabs(static_cast<double>(SIGMA_Q8[0]) - target);
  for (int i = 1; i < NUM_SIGMAS; ++i) {
    const double gap = std::fabs(static_cast<double>(SIGMA_Q8[i]) - target);
    if (gap < best_d) {
      best_d = gap;
      best = i;
    }
  }
  return best;
}

double abs_err_q8(int64_t residual, int32_t frac) {
  const __int128 inner =
      static_cast<__int128>(residual) * 16 - (static_cast<__int128>(frac) - 8);
  const __int128 mag = inner < 0 ? -inner : inner;
  return static_cast<double>(mag << 4);
}

int symbol_of(int64_t residual, int shift) {
  __int128 quot;
  if (shift <= 0) quot = residual;
  else {
    const __int128 bias = __int128{1} << (shift - 1);
    quot = (static_cast<__int128>(residual) + bias) >> shift;
  }
  const __int128 mag = quot < 0 ? -quot : quot;
  if (mag > 32) return 65;
  return static_cast<int>(quot + 32);
}

}  // namespace

std::optional<std::vector<int64_t>> fit_ridge_q8(const int64_t* phi, int rows, int cols,
                                                 const int64_t* target, int min_count) {
  if (rows < 0 || cols < 0) throw CodecError("ridge shapes do not match");
  if ((rows > 0 && (phi == nullptr || target == nullptr)) || (cols > 0 && rows > 0 && phi == nullptr)) {
    throw CodecError("ridge shapes do not match");
  }
  if (rows < min_count || cols == 0) return std::nullopt;
  bool any = false;
  for (int i = 0; i < rows * cols; ++i) {
    if (phi[i] != 0) {
      any = true;
      break;
    }
  }
  if (!any) return std::nullopt;

  std::vector<double> energy(static_cast<size_t>(cols), 0.0);
  std::vector<double> gram(static_cast<size_t>(cols * cols), 0.0);
  std::vector<double> rhs(static_cast<size_t>(cols), 0.0);
  for (int row = 0; row < rows; ++row) {
    const int64_t* src = phi + static_cast<size_t>(row * cols);
    const double y = static_cast<double>(target[row]);
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
  if (!solve_ge(gram, rhs, cols, solved)) return std::nullopt;

  std::vector<int64_t> weights(static_cast<size_t>(cols));
  bool nonzero = false;
  for (int j = 0; j < cols; ++j) {
    int64_t weight = round_half_even(solved[static_cast<size_t>(j)] * 256.0);
    if (weight > kWeightMaxQ8) weight = kWeightMaxQ8;
    if (weight < -kWeightMaxQ8) weight = -kWeightMaxQ8;
    weights[static_cast<size_t>(j)] = weight;
    if (weight != 0) nonzero = true;
  }
  if (!nonzero) return std::nullopt;

  double sum0 = 0.0;
  double sum1 = 0.0;
  for (int row = 0; row < rows; ++row) {
    const int64_t* src = phi + static_cast<size_t>(row * cols);
    __int128 dot = 0;
    for (int k = 0; k < cols; ++k) dot += static_cast<__int128>(src[k]) * weights[static_cast<size_t>(k)];
    const __int128 corr = (dot + 128) >> 8;
    const __int128 diff = static_cast<__int128>(target[row]) - corr;
    const __int128 ad = diff < 0 ? -diff : diff;
    sum0 += static_cast<double>(iabs(target[row]));
    sum1 += static_cast<double>(ad);
  }
  const double n = static_cast<double>(rows);
  const double gain = n * (std::log2(sum0 / n + 4.0) - std::log2(sum1 / n + 4.0));
  int header = 8;
  for (int64_t weight : weights) header += golomb_bits(weight);
  if (!(gain > static_cast<double>(header))) return std::nullopt;
  return weights;
}

SigmaNu fit_sigma_nu(const int64_t* residual, const int32_t* frac, const int32_t* buckets, int count,
                     int n_buckets) {
  if (n_buckets <= 0) throw CodecError("sigma bucket count is empty");
  if (count < 0) throw CodecError("sigma fit count is negative");
  if (count > 0 && (residual == nullptr || frac == nullptr || buckets == nullptr)) {
    throw CodecError("sigma fit is missing samples");
  }
  SigmaNu fit;
  if (count < 48) {
    double mad = 64.0;
    if (count > 0) {
      double sum = 0.0;
      for (int i = 0; i < count; ++i) sum += abs_err_q8(residual[i], frac[i]);
      mad = (sum / static_cast<double>(count)) * 0.92 + 24.0;
    }
    const int s0 = argmin_sigma(mad);
    fit.nu = 1;
    fit.sigma.assign(static_cast<size_t>(n_buckets), s0);
    return fit;
  }

  const int32_t* shifts = get_parametric_tables().shifts;
  const double* neg_log = neg_log_table();
  fit.sigma.assign(static_cast<size_t>(n_buckets), 0);
  int prev = 4;
  for (int bucket = 0; bucket < n_buckets; ++bucket) {
    std::vector<int> which;
    double sum = 0.0;
    for (int i = 0; i < count; ++i) {
      if (buckets[i] != bucket) continue;
      which.push_back(i);
      sum += abs_err_q8(residual[i], frac[i]);
    }
    if (static_cast<int>(which.size()) < 8) {
      fit.sigma[static_cast<size_t>(bucket)] = prev;
      continue;
    }
    const double est = (sum / static_cast<double>(which.size())) * 0.92 + 20.0;
    const int guess = argmin_sigma(est);
    int best_s = guess;
    double best_cost = std::numeric_limits<double>::infinity();
    const int begin = std::max(0, guess - 2);
    const int end = std::min(NUM_SIGMAS, guess + 3);
    for (int cand = begin; cand < end; ++cand) {
      const int shift = shifts[cand];
      double cost = static_cast<double>(which.size()) * static_cast<double>(shift);
      for (int index : which) {
        if (frac[index] < 0 || frac[index] >= NUM_FRACS) throw CodecError("fractional phase out of range");
        const int sym = symbol_of(residual[index], shift);
        const int32_t freq = freq_row(1, cand, frac[index])[sym];
        if (freq < 1 || freq > 65536) throw CodecError("parametric frequency out of range");
        cost += neg_log[freq - 1];
        if (sym == 65) {
          __int128 quot;
          if (shift <= 0) quot = residual[index];
          else quot = (static_cast<__int128>(residual[index]) + (__int128{1} << (shift - 1))) >> shift;
          const __int128 mag = quot < 0 ? -quot : quot;
          cost += 2.0 * std::log2(static_cast<double>(mag - 31)) + 3.0;
        }
      }
      if (cost < best_cost) {
        best_cost = cost;
        best_s = cand;
      }
    }
    fit.sigma[static_cast<size_t>(bucket)] = best_s;
    prev = best_s;
  }

  std::vector<int> chosen(static_cast<size_t>(count));
  std::vector<int> symbols(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    if (buckets[i] < 0 || buckets[i] >= n_buckets) throw CodecError("sigma bucket out of range");
    if (frac[i] < 0 || frac[i] >= NUM_FRACS) throw CodecError("fractional phase out of range");
    chosen[static_cast<size_t>(i)] = fit.sigma[static_cast<size_t>(buckets[i])];
    symbols[static_cast<size_t>(i)] = symbol_of(residual[i], shifts[chosen[static_cast<size_t>(i)]]);
  }
  int best_nu = 1;
  double best_nu_cost = std::numeric_limits<double>::infinity();
  for (int nu = 0; nu < NUM_NUS; ++nu) {
    double cost = 0.0;
    for (int i = 0; i < count; ++i) {
      const int32_t freq =
          freq_row(nu, chosen[static_cast<size_t>(i)], frac[i])[symbols[static_cast<size_t>(i)]];
      if (freq < 1 || freq > 65536) throw CodecError("parametric frequency out of range");
      cost += neg_log[freq - 1];
    }
    if (cost < best_nu_cost) {
      best_nu_cost = cost;
      best_nu = nu;
    }
  }
  fit.nu = best_nu;
  return fit;
}

}  // namespace vpxc
