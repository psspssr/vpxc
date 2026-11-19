#pragma once

#include "vpxc/entropy.hpp"

#include <cstdint>
#include <optional>
#include <vector>

// Encode-only fits. Decoding never calls this header.

namespace vpxc {

inline constexpr int kWeightMaxQ8 = 2047;

// Q8 weights for phi (rows x cols, row-major) explaining target, or nullopt.
// Normal equations match mdl.py: gram = Phi^T Phi + diag(max(1e-4 * column energy, 64)),
// solved with partial-pivoting Gaussian elimination.
std::optional<std::vector<int64_t>> fit_ridge_q8(const int64_t* phi, int rows, int cols,
                                                 const int64_t* target, int min_count = 64);

struct SigmaNu {
  int nu = 1;
  std::vector<int32_t> sigma;
};

// Student-t nu and one SIGMA_Q8 index per activity bucket.
SigmaNu fit_sigma_nu(const int64_t* residual, const int32_t* frac, const int32_t* buckets, int count,
                     int n_buckets = 16);

}  // namespace vpxc
