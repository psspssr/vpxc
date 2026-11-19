#pragma once

#include "vpxc/entropy.hpp"
#include "vpxc/image.hpp"

#include <cstdint>
#include <vector>

namespace vpxc {

// Predictor-bank coder for one packed coarse plane. ref may be null.
// Decode applies the stored integers and never uses floating point.
std::vector<uint8_t> encode_coarse(const Plane& plane, const Mask& valid, const Plane* ref);
Plane decode_coarse(const uint8_t* data, size_t size, int height, int width, const Mask& valid,
                    const Plane* ref);

}  // namespace vpxc
