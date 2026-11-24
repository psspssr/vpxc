// libFuzzer + AddressSanitizer + UndefinedBehaviorSanitizer harness for vpxc.
//
// Build & run example:
//   clang++ --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13 -std=c++20 -O2 -g \
//     -DVPXC_FUZZ_MAX_SAMPLES=1048576 \
//     -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all \
//     -Icpp/include cpp/fuzz/fuzz_decode.cpp cpp/src/bitio.cpp cpp/src/entropy.cpp \
//     cpp/src/wsil.cpp cpp/src/color.cpp cpp/src/lattice.cpp cpp/src/issm.cpp \
//     cpp/src/spatial.cpp cpp/src/coarse.cpp cpp/src/matching.cpp \
//     cpp/src/acquisition.cpp cpp/src/trees.cpp cpp/src/bitsback.cpp \
//     cpp/src/mdl.cpp cpp/src/slgf.cpp cpp/src/slgf_v3.cpp cpp/src/wp.cpp \
//     cpp/src/frontier.cpp cpp/src/fast_codec.cpp cpp/src/medical.cpp \
//     cpp/src/codec.cpp -lz -pthread -o /tmp/fuzz_vpxc
//
// Deliberately catches ONLY vpxc::CodecError so any UB, OOB read/write, memory
// leak, OOM, or unexpected std::exception immediately aborts with a stack trace.

#include "vpxc/codec.hpp"
#include "vpxc/error.hpp"
#include "vpxc/fast_codec.hpp"
#include "vpxc/frontier.hpp"
#include "vpxc/medical.hpp"

#include <cstddef>
#include <cstdint>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  // 1. Container-level 2D image decode (all profiles: raw, raster, kalman, qx, palette, slgf, wp, frontier, fast)
  try {
    (void)vpxc::decode_image(data, size);
  } catch (const vpxc::CodecError&) {
  }

  // 2. Container-level 3D volumetric medical decode + O(1) slice random access
  try {
    const auto hdr = vpxc::inspect_volume(data, size);
    const int z_probe = hdr.depth > 0 ? static_cast<int>((size > 0 ? data[size - 1] : 0) % hdr.depth) : 0;
    (void)vpxc::decode_volume_slice(data, size, z_probe);
    (void)vpxc::decode_volume(data, size);
  } catch (const vpxc::CodecError&) {
  }

  // 3. Direct raw tile / slice decoder fuzzing on small bounded geometries
  if (size >= 2) {
    const int h = 1 + (data[0] & 15);
    const int w = 1 + ((data[0] >> 4) & 15);
    const int c_sel = data[1] & 3;
    const int channels = (c_sel == 0) ? 1 : ((c_sel == 1) ? 3 : 4);
    const bool u16 = (data[1] & 4) != 0;
    const uint8_t* payload = data + 2;
    const size_t plen = size - 2;

    try {
      (void)vpxc::decode_fast(payload, plen, h, w, channels, u16);
    } catch (const vpxc::CodecError&) {
    }

    try {
      (void)vpxc::decode_frontier(payload, plen, h, w, channels, u16);
    } catch (const vpxc::CodecError&) {
    }

    try {
      vpxc::Plane ref(h, w, u16 ? 1024 : 64);
      size_t pos = 0;
      (void)vpxc::decode_fast_plane(payload, plen, pos, h, w, u16 ? 65535 : 255, &ref);
    } catch (const vpxc::CodecError&) {
    }

    try {
      vpxc::Plane ref(h, w, u16 ? 1024 : 64);
      vpxc::Plane prev2(h, w, u16 ? 1020 : 60);
      (void)vpxc::decode_frontier_slice_3d(payload, plen, h, w, ref, &prev2, u16);
    } catch (const vpxc::CodecError&) {
    }
  }

  return 0;
}
