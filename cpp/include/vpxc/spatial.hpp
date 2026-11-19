#pragma once

#include "vpxc/image.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vpxc {

// Causal MED raster coder with optional D8 Sheaf-Kalman. Decode is integer-only.
// Rice symbols go through BitWriter::write_rice / BitReader::read_rice.
std::vector<uint8_t> encode_raster(const Plane& plane, const Mask& valid, bool kalman);
Plane decode_raster(const uint8_t* data, size_t size, int height, int width, const Mask& valid,
                    bool kalman);

}  // namespace vpxc
