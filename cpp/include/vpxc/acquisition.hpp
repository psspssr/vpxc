#pragma once

#include "vpxc/bitio.hpp"
#include "vpxc/image.hpp"

#include <optional>

namespace vpxc {

// Pre-gamma acquisition header. knots are monotone 12-bit linear-light codes.
// taps are Q6, indexed [phase][offset] with phase = ((y & 1) << 1) | (x & 1).
struct PhiAcq {
  int shift = 0;
  int knots[8] = {};
  int taps[4][24] = {};
};

// Integer predicted chroma field from decoded luma. No floating point.
Plane apply_phi(const Plane& luma, const PhiAcq& phi);

// Encode-only. Returns nullopt when the header does not pay for itself.
std::optional<PhiAcq> fit_phi(const Plane& luma, const Plane& chroma, const Plane& flat_pred,
                              const Mask& odd);

void write_phi(BitWriter& writer, const PhiAcq& phi);
PhiAcq read_phi(BitReader& reader);

}  // namespace vpxc
