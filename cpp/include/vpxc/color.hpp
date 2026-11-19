#pragma once

#include "vpxc/wsil.hpp"

#include <vector>

namespace vpxc {

inline constexpr int MODE_IDENTITY = 0;
inline constexpr int MODE_YCOCG = 1;
inline constexpr int MODE_SUB = 2;
inline constexpr int MODE_SHEAR = 3;
inline constexpr int MODE_WSIL = 4;

// perm: 0 = green base (channel 1), 1 = red base (channel 0), 2 = blue base (channel 2).
// shear is the Q12 triple (a, c, d). extra is the Q12 coupling of channel 3 onto the lifted head.
struct ColorTransform {
  int mode = MODE_IDENTITY;
  int perm = 0;
  int shear[3] = {0, 0, 0};
  SL3Params wsil{};
  int extra[3] = {0, 0, 0};
};

// Encode-only. May use double to score candidates. Fewer than 3 channels returns identity.
ColorTransform choose_color_transform(const std::vector<Plane>& channels);

std::vector<Plane> apply_color_forward(const std::vector<Plane>& channels, const ColorTransform& transform);
std::vector<Plane> apply_color_inverse(const std::vector<Plane>& planes, const ColorTransform& transform);

}  // namespace vpxc
