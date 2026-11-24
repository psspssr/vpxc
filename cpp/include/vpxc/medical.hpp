#pragma once

// VPXV 3D Volumetric Medical Lossless Container & Random-Access Slice Codec.
// Supports uint8, uint16, and signed int16 (Hounsfield Units) CT/MRI stacks
// with Group-of-Slices (GoS) O(1) random-access slice seeking, per-slice and
// whole-volume CRC-32 integrity verification, and 100% integer decoding.

#include "vpxc/image.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vpxc {

enum class VolumeDType : uint8_t {
  U8 = 1,
  U16 = 2,
  I16 = 3,  // Signed 16-bit two's-complement (e.g., CT Hounsfield Units -1024..+3071)
};

struct Volume {
  int depth = 0;
  int height = 0;
  int width = 0;
  VolumeDType dtype = VolumeDType::U16;
  // Row-major ZYX samples.
  // - U8: stored in low 8 bits [0..255]
  // - U16: unsigned [0..65535]
  // - I16: raw two's-complement uint16_t bit pattern of int16_t
  std::vector<uint16_t> samples;

  size_t slice_voxels() const noexcept { return static_cast<size_t>(height) * static_cast<size_t>(width); }
  size_t total_voxels() const noexcept { return static_cast<size_t>(depth) * slice_voxels(); }
};

struct VolumeOptions {
  int effort = 0;    // 0 = real-time PACS fast mode (<5ms/slice decode); 1..3 = frontier 3D archival mode
  int gos_size = 8;  // Group-of-Slices random-access interval [1..64]
};

struct VolumeHeader {
  int version = 1;
  VolumeDType dtype = VolumeDType::U16;
  int effort_mode = 0;
  int gos_size = 8;
  int depth = 0;
  int height = 0;
  int width = 0;
  uint32_t volume_crc32 = 0;
};

VolumeHeader inspect_volume(const uint8_t* data, size_t size);
std::vector<uint8_t> encode_volume(const Volume& vol, const VolumeOptions& opt = {});
Volume decode_volume(const uint8_t* data, size_t size);

// O(1) Group-of-Slices random-access decode of a single axial slice `z_index` [0 .. depth-1]
// with per-slice CRC-32 verification, without decoding the rest of the 3D volume.
Volume decode_volume_slice(const uint8_t* data, size_t size, int z_index);

}  // namespace vpxc
