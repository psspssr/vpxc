#pragma once

#include "vpxc/image.hpp"

namespace vpxc {

// Integer Sheaf-Kalman correction on query sites: match - flat_pred, else 0.
// Python returns int64; this stores the same value in a Plane (int32). A correction
// outside int32 keeps the low 32 bits.
Plane sheaf_kalman_delta(const Plane& samples, const Mask& known, const Mask& query,
                         const Plane& flat_pred, int radius = 128);

}  // namespace vpxc
