#pragma once

#include "vpxc/error.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vpxc {

#ifdef VPXC_FUZZ_MAX_SAMPLES
inline constexpr size_t kMaxImageSamples = VPXC_FUZZ_MAX_SAMPLES;
#else
inline constexpr size_t kMaxImageSamples = 134217728ULL;  // 256 MiB uint16 sample budget
#endif
inline constexpr int kMaxImageDim = 16384;

// Row-major plane. Samples are signed because lifting residuals leave the
// original uint8 / uint16 range.
struct Plane {
  int height = 0;
  int width = 0;
  std::vector<int32_t> samples;

  Plane() = default;
  Plane(int h, int w, int32_t fill = 0) : height(h), width(w) {
    if (h < 0 || w < 0) throw CodecError("negative plane dimensions");
    const size_t n = static_cast<size_t>(h) * static_cast<size_t>(w);
    if (n > kMaxImageSamples) throw CodecError("plane sample count exceeds safety limit");
    samples.assign(n, fill);
  }

  int32_t& at(int y, int x) { return samples[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)]; }
  int32_t at(int y, int x) const { return samples[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)]; }
  bool inside(int y, int x) const { return y >= 0 && x >= 0 && y < height && x < width; }
};

struct Mask {
  int height = 0;
  int width = 0;
  std::vector<uint8_t> bits;

  Mask() = default;
  Mask(int h, int w, bool fill = true) : height(h), width(w) {
    if (h < 0 || w < 0) throw CodecError("negative mask dimensions");
    const size_t n = static_cast<size_t>(h) * static_cast<size_t>(w);
    if (n > kMaxImageSamples) throw CodecError("mask sample count exceeds safety limit");
    bits.assign(n, fill ? 1 : 0);
  }

  uint8_t& at(int y, int x) { return bits[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)]; }
  bool at(int y, int x) const { return bits[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] != 0; }
};

struct Image {
  int height = 0;
  int width = 0;
  int channels = 1;
  bool u16 = false;
  // Interleaved HWC samples. uint8 values are stored in the low 8 bits.
  std::vector<uint16_t> samples;

  size_t plane_size() const { return static_cast<size_t>(height) * static_cast<size_t>(width); }
};

struct EncodeOptions {
  int tile = 384;
  int effort = 1;
  int version = 3;
  // Empty means automatic profile selection.
  const char* profile = nullptr;
};

}  // namespace vpxc
