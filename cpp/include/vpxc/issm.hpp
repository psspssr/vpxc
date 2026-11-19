#pragma once

#include "vpxc/image.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace vpxc {

// Left-to-right recurrence h = ((a * h + 2^15) >> 16) + b.
std::vector<int64_t> sequential(const std::vector<int64_t>& decay, const std::vector<int64_t>& bias);

// Normative balanced-tree rollout. Bit-matches issm.tree_rows on one row,
// including the Q1.30 Blelloch composition (not merely sequential within 1 LSB).
std::vector<int64_t> tree(const std::vector<int64_t>& decay, const std::vector<int64_t>& bias);

// East, west, south, north. Each plane holds the Q4 state in int32.
// base_decay defaults to 53248 and edge_kappa to 16. No floating point.
std::array<Plane, 4> scan_4way(const Plane& plane, int base_decay = 53248, int edge_kappa = 16);

}  // namespace vpxc
