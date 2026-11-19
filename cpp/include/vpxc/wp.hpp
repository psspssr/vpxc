#pragma once

#include "vpxc/image.hpp"

#include <cstdint>
#include <vector>

namespace vpxc {

// JPEG XL 4-tap self-correcting weighted predictor, lossless-8 preset,
// with the prediction error coded by Dyadic-Core66. Decode is integer-only.
std::vector<uint8_t> encode_wp(const std::vector<Plane>& planes);
std::vector<Plane> decode_wp(const uint8_t* data, size_t size, int height, int width, int channels);

}  // namespace vpxc
