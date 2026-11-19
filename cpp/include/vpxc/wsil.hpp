#pragma once

#include "vpxc/image.hpp"

#include <cmath>
#include <cstdint>

namespace vpxc {

inline constexpr int kQ12 = 12;
inline constexpr int kQ12One = 1 << kQ12;    // 4096
inline constexpr int kQ12Bias = 1 << (kQ12 - 1);  // 2048

// Appendix A.1 multipliers. perm defaults to identity; squeezes default to 1.0 in Q12.
struct SL3Params {
  int perm[3] = {0, 1, 2};
  int u01 = 0;
  int u02 = 0;
  int u12 = 0;
  int l10 = 0;
  int l20 = 0;
  int l21 = 0;
  int s1 = kQ12One;
  int s1_inv = kQ12One;
  int s2 = kQ12One;
  int s2_inv = kQ12One;
};

// Round-to-nearest Q12 product, ties toward +infinity: (m * z + 2048) >> 12.
inline int r12(int m, int z) {
  const std::int64_t shifted =
      (static_cast<std::int64_t>(m) * static_cast<std::int64_t>(z) + kQ12Bias) >> kQ12;
  return static_cast<int>(shifted);
}

// Quantize a real multiplier to Q12 and clamp to int16. Encode-only.
// Half-integers use round-half-to-even, matching NumPy np.round.
inline int q12(double value) {
  if (!std::isfinite(value)) return 0;
  const double scaled = value * static_cast<double>(kQ12One);
  if (!std::isfinite(scaled)) return scaled > 0.0 ? 32767 : -32768;
  const double rounded = std::nearbyint(scaled);
  if (rounded > 32767.0) return 32767;
  if (rounded < -32768.0) return -32768;
  return static_cast<int>(rounded);
}

// Vectorized Appendix A.1. in/out are three equal-shaped planes.
void sl3_forward_arrays(const Plane in[3], Plane out[3], const SL3Params& params);
void sl3_inverse_arrays(const Plane in[3], Plane out[3], const SL3Params& params);

// Row-major 3x3. Encode-only (uses double). Returns identity params if det is degenerate.
SL3Params factor_sl3_matrix(const double matrix[9]);

// Encode-only decorrelating eigenframe from 2-D diamond Laplacians.
SL3Params estimate_sl3_wsil(const Plane rgb[3]);

}  // namespace vpxc
