#pragma once

// High-throughput lossless tile & slice coder ("fast" profile, kProfileFast = 9).
//
// Decouples residual entropy coding from spatial prediction via:
//   1. Exact reversible RCT (Identity, Subtract-Green, Subtract-Average) with SIMD inverse,
//   2. Per-plane dynamic-range [pmin, pmax] + integer quantization GCD compaction + O(1) constant bypass,
//   3. 4-stripe parallel integer MED (or cross-slice 3D anatomy-switched Delta-MED for medical volumes),
//   4. 4-way interleaved 12-bit flat-LUT rANS (single uint32_t table lookup, single-branch 16-bit renorm,
//      zero binary search, zero per-pixel adaptive CDF rebuilds, 100% IEEE-754 float-free decode).

#include "vpxc/image.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vpxc {

// Encode a multi-channel 8-bit or 16-bit tile using the high-throughput 4-way interleaved rANS + MED engine.
std::vector<uint8_t> encode_fast(const std::vector<Plane>& raw_planes, bool u16);

// Decode a "fast" profile tile payload using strictly integer operations (zero floating-point instructions).
std::vector<Plane> decode_fast(const uint8_t* data, size_t size, int height, int width, int channels, bool u16);

// Single-plane fast residual stream helpers reused by the 3D volumetric medical P-slice coder.
std::vector<uint8_t> encode_fast_plane(const Plane& P, const Plane* ref_slice = nullptr);
Plane decode_fast_plane(const uint8_t* data, size_t size, size_t& pos, int height, int width, int max_abs_bound,
                        const Plane* ref_slice = nullptr);

}  // namespace vpxc
