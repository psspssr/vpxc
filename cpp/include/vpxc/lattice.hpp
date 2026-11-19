#pragma once

#include "vpxc/image.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace vpxc {

// Q = [[1, 1], [1, -1]] even-site packing: U = (r + c) / 2, V = (r - c) / 2,
// shifted so the packed origin is (0, 0).
struct PackInfo {
  int u_shift = 0;
  int v_shift = 0;
  int parent_h = 0;
  int parent_w = 0;
  int packed_h = 0;
  int packed_w = 0;
};

// odd = valid & ((r + c) & 1); even = valid & ~odd.
struct ParityMasks {
  Mask odd;
  Mask even;
};

ParityMasks parity_masks(const Mask& valid);

// Integer prediction of every site from orthogonal even neighbors.
// sheaf=false: rounded mean. sheaf=true: directional conductances.
Plane predict(const Plane& img, const Mask& valid, bool sheaf);

Plane lift_forward(const Plane& samples, const Mask& valid, bool sheaf);
Plane lift_inverse(const Plane& lifted, const Mask& valid, bool sheaf);

struct PackEven {
  Plane packed;
  Mask mask;
  PackInfo info;
};

PackEven pack_even(const Plane& img, const Mask& valid);
Plane unpack_even(const Plane& packed, const PackInfo& info, const Mask& parent_valid);

// floor(log2(max(value, 1))) for each entry, matching the pure int64 bit loop.
std::vector<int32_t> ilog2_int64(const std::vector<int64_t>& values);

// Curvature / contour bucket in 0..7. samples is row-major, same shape as lowpass.
struct ContextMap {
  int height = 0;
  int width = 0;
  std::vector<uint8_t> samples;

  uint8_t at(int y, int x) const { return samples[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)]; }
};

ContextMap activity_context(const Plane& lowpass, const Mask& valid);

struct Level {
  Mask valid;
  PackInfo info;
  Mask packed_mask;
};

// levels followed by the coarse valid mask. Stops on the Python early-stop rules.
std::pair<std::vector<Level>, Mask> build_levels(int h, int w, int max_levels);

struct Tower {
  Plane coarse;
  Mask coarse_valid;
  std::vector<Plane> details;
  std::vector<Level> levels;
};

Tower forward_tower(const Plane& plane, int max_levels, bool sheaf);
Plane inverse_tower(const Plane& coarse, const std::vector<Plane>& details, const std::vector<Level>& levels,
                    bool sheaf);

}  // namespace vpxc
