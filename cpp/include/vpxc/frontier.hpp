#pragma once

#include "vpxc/image.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vpxc {

// Channel-causal hybrid lossless tile coder combining:
// - Per-tile zero/minimal-rounding RCT selection (SubtractBase & SubtractAverage)
//   with exact pixel-domain feasibility clamping,
// - 8-expert self-correcting predictor + causal L-template patch predictor +
//   2-pass MDL-gated directional Q8 Ridge regression,
// - Hybrid raster vs 2x2-coset exact chroma interpolation with full-plane
//   bilateral edge-affinity Cross-Component Prediction (CfL),
// - MDL-pruned MA context tree + prior-seeded adaptive Dyadic-Core66 rANS.
// All decode paths use strictly integer arithmetic with zero floating-point ops.
std::vector<uint8_t> encode_frontier(const std::vector<Plane>& raw_planes, bool u16, int effort);
std::vector<Plane> decode_frontier(const uint8_t* data, size_t size, int height, int width, int channels, bool u16);

// Cross-slice 3D volumetric P-slice coder conditioned on causal previous slice `ref_slice`
// (and optional `prev_ref_slice` for 2nd-order Z-velocity context/prediction), 100% integer decode.
std::vector<uint8_t> encode_frontier_slice_3d(const Plane& cur_slice, const Plane& ref_slice,
                                              const Plane* prev_ref_slice, bool u16, int effort);
Plane decode_frontier_slice_3d(const uint8_t* data, size_t size, int height, int width, const Plane& ref_slice,
                               const Plane* prev_ref_slice, bool u16);

}  // namespace vpxc
