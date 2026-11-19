#pragma once

// Tile container (src/vpxc/codec.py).

#include "vpxc/image.hpp"
#include "vpxc/slgf.hpp"
#include "vpxc/slgf_v3.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#if __has_include("vpxc/color.hpp")
#include "vpxc/color.hpp"
#define VPXC_HAS_COLOR 1
#endif

#if __has_include("vpxc/wsil.hpp")
#include "vpxc/wsil.hpp"
#define VPXC_HAS_WSIL 1
#endif

namespace vpxc {

#ifndef VPXC_HAS_COLOR
#ifndef VPXC_HAS_WSIL
struct SL3Params {
  int perm[3] = {0, 1, 2};
  int u01 = 0;
  int u02 = 0;
  int u12 = 0;
  int l10 = 0;
  int l20 = 0;
  int l21 = 0;
  int s1 = 4096;
  int s1_inv = 4096;
  int s2 = 4096;
  int s2_inv = 4096;
};
#endif

inline constexpr int MODE_IDENTITY = 0;
inline constexpr int MODE_YCOCG = 1;
inline constexpr int MODE_SUB = 2;
inline constexpr int MODE_SHEAR = 3;
inline constexpr int MODE_WSIL = 4;

struct ColorTransform {
  int mode = MODE_IDENTITY;
  int perm = 0;
  int shear[3] = {0, 0, 0};
  SL3Params wsil{};
  int extra_weights[3] = {0, 0, 0};
};

ColorTransform choose_color_transform(const std::vector<Plane>& channels);
std::vector<Plane> apply_color_forward(const std::vector<Plane>& channels, const ColorTransform& transform);
std::vector<Plane> apply_color_inverse(const std::vector<Plane>& planes, const ColorTransform& transform);
#endif

#ifndef VPXC_HAS_ENTROPY
std::vector<uint8_t> encode_dyadic_rans(const Plane& residuals, const Mask& valid,
                                        const std::vector<uint8_t>& shifts);
Plane decode_dyadic_rans(const uint8_t* data, size_t size, int height, int width, const Mask& valid,
                         const std::vector<uint8_t>& shifts);
std::vector<uint8_t> encode_rice_values(const Plane& values, const Mask& valid,
                                        const std::vector<uint8_t>& context, bool causal_mu);
Plane decode_rice_values(const uint8_t* data, size_t size, int height, int width, const Mask& valid,
                         const std::vector<uint8_t>& context, bool causal_mu);
std::vector<uint8_t> shift_map_from_activity(const std::vector<uint8_t>& activity, int height, int width);
#endif

// Container entry points. encode/decode match the migration note; encode_image/decode_image
// are the same functions under the names the C++ port was asked to export.
std::vector<uint8_t> encode_image(const Image& image, const EncodeOptions& opt);
Image decode_image(const uint8_t* data, size_t size);

inline std::vector<uint8_t> encode(const Image& image, const EncodeOptions& opt) {
  return encode_image(image, opt);
}
inline Image decode(std::span<const uint8_t> bytes) { return decode_image(bytes.data(), bytes.size()); }

}  // namespace vpxc
