#include "vpxc/codec.hpp"
#include "vpxc/error.hpp"
#include "vpxc/medical.hpp"
#include "vpxc/simd.hpp"

#include <cstdint>
#include <cstring>
#include <new>
#include <string>

using vpxc::CodecError;
using vpxc::EncodeOptions;
using vpxc::Image;

namespace {

thread_local std::string g_error;

void set_error(const std::exception& ex) { g_error = ex.what(); }

void check_geometry(int h, int w, int c, const void* samples) {
  if (c != 1 && c != 3 && c != 4) throw CodecError("channel count must be 1, 3, or 4");
  if (h <= 0 || w <= 0) throw CodecError("empty image");
  if (h > vpxc::kMaxImageDim || w > vpxc::kMaxImageDim) throw CodecError("image dimension exceeds safety limit");
  const size_t n = static_cast<size_t>(h) * static_cast<size_t>(w) * static_cast<size_t>(c);
  if (n > vpxc::kMaxImageSamples) throw CodecError("image sample count exceeds safety limit");
  if (samples == nullptr) throw CodecError("null image");
}

Image image_from(const uint16_t* samples, int h, int w, int c, int u16) {
  check_geometry(h, w, c, samples);
  Image image;
  image.height = h;
  image.width = w;
  image.channels = c;
  image.u16 = u16 != 0;
  const size_t n = static_cast<size_t>(h) * static_cast<size_t>(w) * static_cast<size_t>(c);
  image.samples.assign(samples, samples + n);
  return image;
}

}  // namespace

extern "C" {

const char* vpxc_last_error() { return g_error.c_str(); }

void vpxc_free(void* p) { delete[] static_cast<uint8_t*>(p); }

int vpxc_encode(const uint16_t* samples, int h, int w, int c, int u16, int tile, int effort, int version,
                const char* profile, uint8_t** out_bytes, size_t* out_n) {
  g_error.clear();
  if (out_bytes) *out_bytes = nullptr;
  if (out_n) *out_n = 0;
  try {
    EncodeOptions opt;
    opt.tile = tile;
    opt.effort = effort;
    opt.version = version;
    opt.profile = (profile != nullptr && profile[0] != '\0') ? profile : nullptr;
    const std::vector<uint8_t> bytes = vpxc::encode_image(image_from(samples, h, w, c, u16), opt);
    auto* buf = new uint8_t[bytes.size()];
    if (!bytes.empty()) std::memcpy(buf, bytes.data(), bytes.size());
    *out_bytes = buf;
    *out_n = bytes.size();
    return 0;
  } catch (const CodecError& ex) {
    set_error(ex);
    return 1;
  } catch (const std::exception& ex) {
    set_error(ex);
    return 2;
  }
}

int vpxc_decode(const uint8_t* data, size_t n, uint16_t** samples, int* h, int* w, int* c, int* u16) {
  g_error.clear();
  try {
    const Image image = vpxc::decode_image(data, n);
    const size_t count = image.plane_size() * static_cast<size_t>(image.channels);
    auto* buf = new uint16_t[count];
    if (count) std::memcpy(buf, image.samples.data(), count * sizeof(uint16_t));
    *samples = buf;
    *h = image.height;
    *w = image.width;
    *c = image.channels;
    *u16 = image.u16 ? 1 : 0;
    return 0;
  } catch (const CodecError& ex) {
    set_error(ex);
    return 1;
  } catch (const std::exception& ex) {
    set_error(ex);
    return 2;
  }
}

void vpxc_free_samples(void* p) { delete[] static_cast<uint16_t*>(p); }

const char* vpxc_arch_name() { return vpxc::simd::arch_name(); }

int vpxc_encode_volume(const uint16_t* samples, int depth, int height, int width, int dtype, int effort, int gos_size,
                       uint8_t** out_bytes, size_t* out_n) {
  g_error.clear();
  if (out_bytes) *out_bytes = nullptr;
  if (out_n) *out_n = 0;
  try {
    if (samples == nullptr || out_bytes == nullptr || out_n == nullptr) throw CodecError("null volume pointer");
    if (depth <= 0 || height <= 0 || width <= 0) throw CodecError("empty medical volume");
    if (dtype < 1 || dtype > 3) throw CodecError("invalid medical volume dtype");
    vpxc::Volume vol;
    vol.depth = depth;
    vol.height = height;
    vol.width = width;
    vol.dtype = static_cast<vpxc::VolumeDType>(dtype);
    const size_t total = vol.total_voxels();
    vol.samples.assign(samples, samples + total);
    vpxc::VolumeOptions opt;
    opt.effort = effort;
    opt.gos_size = gos_size;
    const std::vector<uint8_t> bytes = vpxc::encode_volume(vol, opt);
    auto* buf = new uint8_t[bytes.size()];
    if (!bytes.empty()) std::memcpy(buf, bytes.data(), bytes.size());
    *out_bytes = buf;
    *out_n = bytes.size();
    return 0;
  } catch (const CodecError& ex) {
    set_error(ex);
    return 1;
  } catch (const std::exception& ex) {
    set_error(ex);
    return 2;
  }
}

int vpxc_decode_volume(const uint8_t* data, size_t n, uint16_t** samples, int* depth, int* height, int* width,
                       int* dtype) {
  g_error.clear();
  try {
    if (samples == nullptr || depth == nullptr || height == nullptr || width == nullptr || dtype == nullptr) {
      throw CodecError("null output pointer");
    }
    const vpxc::Volume vol = vpxc::decode_volume(data, n);
    const size_t total = vol.total_voxels();
    auto* buf = new uint16_t[total];
    if (total > 0) std::memcpy(buf, vol.samples.data(), total * sizeof(uint16_t));
    *samples = buf;
    *depth = vol.depth;
    *height = vol.height;
    *width = vol.width;
    *dtype = static_cast<int>(vol.dtype);
    return 0;
  } catch (const CodecError& ex) {
    set_error(ex);
    return 1;
  } catch (const std::exception& ex) {
    set_error(ex);
    return 2;
  }
}

int vpxc_decode_volume_slice(const uint8_t* data, size_t n, int z_index, uint16_t** samples, int* height, int* width,
                             int* dtype) {
  g_error.clear();
  try {
    if (samples == nullptr || height == nullptr || width == nullptr || dtype == nullptr) {
      throw CodecError("null output pointer");
    }
    const vpxc::Volume slice = vpxc::decode_volume_slice(data, n, z_index);
    const size_t total = slice.slice_voxels();
    auto* buf = new uint16_t[total];
    if (total > 0) std::memcpy(buf, slice.samples.data(), total * sizeof(uint16_t));
    *samples = buf;
    *height = slice.height;
    *width = slice.width;
    *dtype = static_cast<int>(slice.dtype);
    return 0;
  } catch (const CodecError& ex) {
    set_error(ex);
    return 1;
  } catch (const std::exception& ex) {
    set_error(ex);
    return 2;
  }
}

int vpxc_inspect_volume(const uint8_t* data, size_t n, int* depth, int* height, int* width, int* dtype,
                        int* effort_mode, int* gos_size) {
  g_error.clear();
  try {
    const vpxc::VolumeHeader hdr = vpxc::inspect_volume(data, n);
    if (depth) *depth = hdr.depth;
    if (height) *height = hdr.height;
    if (width) *width = hdr.width;
    if (dtype) *dtype = static_cast<int>(hdr.dtype);
    if (effort_mode) *effort_mode = hdr.effort_mode;
    if (gos_size) *gos_size = hdr.gos_size;
    return 0;
  } catch (const CodecError& ex) {
    set_error(ex);
    return 1;
  } catch (const std::exception& ex) {
    set_error(ex);
    return 2;
  }
}

}  // extern "C"
