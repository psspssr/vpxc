#include "vpxc/medical.hpp"

#include "vpxc/error.hpp"
#include "vpxc/fast_codec.hpp"
#include "vpxc/frontier.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <mutex>
#include <span>
#include <thread>
#include <utility>
#include <vector>
#include <zlib.h>

namespace vpxc {
namespace {

constexpr uint8_t kMagic[4] = {'V', 'P', 'X', 'V'};
constexpr uint8_t kVersion = 1;
constexpr int kMaxVolDepth = 4096;
constexpr int kMaxVolDim = 8192;

#ifdef VPXC_FUZZ_MAX_SAMPLES
constexpr size_t kMaxVolumeVoxels = VPXC_FUZZ_MAX_SAMPLES;
#else
constexpr size_t kMaxVolumeVoxels = 268435456ULL;  // 512 MiB uint16 voxel budget
#endif

constexpr uint8_t kSliceFast2D = 0;
constexpr uint8_t kSliceFast3D = 1;
constexpr uint8_t kSliceFrontier2D = 2;
constexpr uint8_t kSliceFrontier3D = 3;
constexpr uint8_t kSliceRaw = 4;

inline void write_u32_le(std::vector<uint8_t>& out, uint32_t v) {
  out.push_back(static_cast<uint8_t>(v & 0xFFu));
  out.push_back(static_cast<uint8_t>((v >> 8) & 0xFFu));
  out.push_back(static_cast<uint8_t>((v >> 16) & 0xFFu));
  out.push_back(static_cast<uint8_t>((v >> 24) & 0xFFu));
}

inline uint32_t read_u32_le(const uint8_t* p) noexcept {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

uint32_t crc32_span(const uint8_t* data, size_t n) noexcept {
  uLong crc = ::crc32(0L, Z_NULL, 0);
  while (n > 0) {
    const uInt chunk = n > (1u << 20) ? (1u << 20) : static_cast<uInt>(n);
    crc = ::crc32(crc, data, chunk);
    data += chunk;
    n -= chunk;
  }
  return static_cast<uint32_t>(crc);
}

uint32_t crc32_slice_samples(const uint16_t* samples, size_t n, VolumeDType dtype) {
  if (dtype == VolumeDType::U8) {
    std::vector<uint8_t> buf(n);
    for (size_t i = 0; i < n; ++i) buf[i] = static_cast<uint8_t>(samples[i] & 0xFFu);
    return crc32_span(buf.data(), buf.size());
  }
  std::vector<uint8_t> buf(n * 2);
  for (size_t i = 0; i < n; ++i) {
    const uint16_t v = samples[i];
    buf[2 * i] = static_cast<uint8_t>(v & 0xFFu);
    buf[2 * i + 1] = static_cast<uint8_t>(v >> 8);
  }
  return crc32_span(buf.data(), buf.size());
}

uint32_t crc32_volume_samples(const uint16_t* samples, size_t total_voxels, VolumeDType dtype) {
  uLong crc = ::crc32(0L, Z_NULL, 0);
  constexpr size_t kBlock = 1u << 16;
  std::vector<uint8_t> scratch(dtype == VolumeDType::U8 ? kBlock : kBlock * 2);
  size_t off = 0;
  while (off < total_voxels) {
    const size_t take = std::min(kBlock, total_voxels - off);
    if (dtype == VolumeDType::U8) {
      for (size_t i = 0; i < take; ++i) scratch[i] = static_cast<uint8_t>(samples[off + i] & 0xFFu);
      crc = ::crc32(crc, scratch.data(), static_cast<uInt>(take));
    } else {
      for (size_t i = 0; i < take; ++i) {
        const uint16_t v = samples[off + i];
        scratch[2 * i] = static_cast<uint8_t>(v & 0xFFu);
        scratch[2 * i + 1] = static_cast<uint8_t>(v >> 8);
      }
      crc = ::crc32(crc, scratch.data(), static_cast<uInt>(take * 2));
    }
    off += take;
  }
  return static_cast<uint32_t>(crc);
}

inline uint16_t to_unsigned_domain(uint16_t raw, VolumeDType dtype) {
  if (dtype == VolumeDType::U8) {
    if (raw > 255u) throw CodecError("U8 volume sample exceeds 255");
    return raw;
  }
  if (dtype == VolumeDType::I16) {
    // Lossless bijection int16_t [-32768..32767] <-> uint16_t [0..65535]
    return static_cast<uint16_t>(raw ^ 0x8000u);
  }
  return raw;
}

inline uint16_t from_unsigned_domain(int32_t uval, VolumeDType dtype) {
  if (dtype == VolumeDType::U8) {
    if (uval < 0 || uval > 255) throw CodecError("decoded U8 voxel out of range");
    return static_cast<uint16_t>(uval);
  }
  if (uval < 0 || uval > 65535) throw CodecError("decoded 16-bit voxel out of range");
  const uint16_t u16 = static_cast<uint16_t>(uval);
  return dtype == VolumeDType::I16 ? static_cast<uint16_t>(u16 ^ 0x8000u) : u16;
}

Plane extract_slice_plane(const Volume& vol, int z) {
  Plane p(vol.height, vol.width, 0);
  const size_t sv = vol.slice_voxels();
  const uint16_t* src = vol.samples.data() + static_cast<size_t>(z) * sv;
  for (size_t i = 0; i < sv; ++i) {
    p.samples[i] = static_cast<int32_t>(to_unsigned_domain(src[i], vol.dtype));
  }
  return p;
}

std::vector<uint8_t> make_raw_slice_bytes(const Plane& p, bool u16) {
  const size_t n = p.samples.size();
  std::vector<uint8_t> out(u16 ? n * 2 : n);
  if (!u16) {
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<uint8_t>(p.samples[i] & 0xFF);
  } else {
    for (size_t i = 0; i < n; ++i) {
      const uint16_t v = static_cast<uint16_t>(p.samples[i]);
      out[2 * i] = static_cast<uint8_t>(v & 0xFFu);
      out[2 * i + 1] = static_cast<uint8_t>(v >> 8);
    }
  }
  return out;
}

Plane decode_raw_slice_bytes(const uint8_t* data, size_t size, int H, int W, bool u16) {
  const size_t n = static_cast<size_t>(H) * static_cast<size_t>(W);
  const size_t need = u16 ? n * 2 : n;
  if (size != need) throw CodecError("invalid raw medical slice length");
  Plane p(H, W, 0);
  if (!u16) {
    for (size_t i = 0; i < n; ++i) p.samples[i] = data[i];
  } else {
    for (size_t i = 0; i < n; ++i) {
      p.samples[i] = static_cast<int32_t>(data[2 * i] | (uint32_t(data[2 * i + 1]) << 8));
    }
  }
  return p;
}

struct ParsedVolume {
  VolumeHeader hdr;
  size_t num_gos = 0;
  size_t payload_offset = 0;
  std::vector<uint32_t> gos_lengths;
  std::vector<size_t> gos_offsets;
};

ParsedVolume parse_volume_directory(const uint8_t* data, size_t size) {
  // Header is 24 bytes:
  // [0..3]  "VPXV"
  // [4]     version (1)
  // [5]     dtype (1=U8, 2=U16, 3=I16)
  // [6]     effort_mode (0..3)
  // [7]     gos_size (1..64)
  // [8..11] depth
  // [12..15] height
  // [16..19] width
  // [20..23] volume_crc32
  if (data == nullptr || size < 24) throw CodecError("truncated VPXV header");
  if (std::memcmp(data, kMagic, 4) != 0) throw CodecError("not a readable VPXV volume");
  const int version = data[4];
  if (version != kVersion) throw CodecError("unsupported VPXV version");
  const uint8_t dt_raw = data[5];
  if (dt_raw < 1 || dt_raw > 3) throw CodecError("unsupported VPXV voxel dtype");
  const int effort_mode = data[6];
  if (effort_mode > 3) throw CodecError("invalid VPXV effort mode");
  const int gos_size = data[7];
  if (gos_size < 1 || gos_size > 64) throw CodecError("invalid VPXV GoS size");

  const uint32_t depth_u32 = read_u32_le(data + 8);
  const uint32_t height_u32 = read_u32_le(data + 12);
  const uint32_t width_u32 = read_u32_le(data + 16);
  const uint32_t vol_crc = read_u32_le(data + 20);

  if (depth_u32 == 0 || height_u32 == 0 || width_u32 == 0 ||
      depth_u32 > static_cast<uint32_t>(kMaxVolDepth) ||
      height_u32 > static_cast<uint32_t>(kMaxVolDim) ||
      width_u32 > static_cast<uint32_t>(kMaxVolDim)) {
    throw CodecError("invalid VPXV geometry");
  }

  const size_t slice_vox = static_cast<size_t>(height_u32) * static_cast<size_t>(width_u32);
  if (slice_vox > kMaxImageSamples) throw CodecError("VPXV slice exceeds safety limit");
  if (static_cast<size_t>(depth_u32) > kMaxVolumeVoxels / slice_vox) {
    throw CodecError("VPXV volume voxel count exceeds safety limit");
  }

  const size_t num_gos = (static_cast<size_t>(depth_u32) + static_cast<size_t>(gos_size) - 1) /
                         static_cast<size_t>(gos_size);
  if (num_gos > (size - 24) / 4) throw CodecError("truncated VPXV GoS directory");
  const size_t dir_bytes = num_gos * 4;
  const size_t payload_start = 24 + dir_bytes;

  ParsedVolume pv;
  pv.hdr.version = version;
  pv.hdr.dtype = static_cast<VolumeDType>(dt_raw);
  pv.hdr.effort_mode = effort_mode;
  pv.hdr.gos_size = gos_size;
  pv.hdr.depth = static_cast<int>(depth_u32);
  pv.hdr.height = static_cast<int>(height_u32);
  pv.hdr.width = static_cast<int>(width_u32);
  pv.hdr.volume_crc32 = vol_crc;
  pv.num_gos = num_gos;
  pv.payload_offset = payload_start;
  pv.gos_lengths.resize(num_gos);
  pv.gos_offsets.resize(num_gos);

  size_t running = payload_start;
  for (size_t g = 0; g < num_gos; ++g) {
    const uint32_t glen = read_u32_le(data + 24 + g * 4);
    if (glen < 9 || glen > size - running) throw CodecError("truncated or invalid VPXV GoS payload");
    pv.gos_lengths[g] = glen;
    pv.gos_offsets[g] = running;
    running += glen;
  }
  if (running != size) throw CodecError("trailing bytes in VPXV container");
  return pv;
}

Plane decode_single_slice_record(uint8_t codec_tag, const uint8_t* payload, size_t len, int H, int W, bool u16,
                                 const Plane* ref_slice, const Plane* prev_ref_slice) {
  const int max_abs = u16 ? 65535 : 255;
  if (codec_tag == kSliceFast2D) {
    size_t pos = 0;
    Plane p = decode_fast_plane(payload, len, pos, H, W, max_abs, nullptr);
    if (pos != len) throw CodecError("trailing bytes in fast 2D medical slice");
    return p;
  }
  if (codec_tag == kSliceFast3D) {
    if (ref_slice == nullptr) throw CodecError("VPXV keyframe slice cannot use 3D P-mode");
    size_t pos = 0;
    Plane p = decode_fast_plane(payload, len, pos, H, W, max_abs, ref_slice);
    if (pos != len) throw CodecError("trailing bytes in fast 3D medical slice");
    return p;
  }
  if (codec_tag == kSliceFrontier2D) {
    auto planes = decode_frontier(payload, len, H, W, 1, u16);
    return std::move(planes[0]);
  }
  if (codec_tag == kSliceFrontier3D) {
    if (ref_slice == nullptr) throw CodecError("VPXV keyframe slice cannot use frontier 3D P-mode");
    return decode_frontier_slice_3d(payload, len, H, W, *ref_slice, prev_ref_slice, u16);
  }
  if (codec_tag == kSliceRaw) {
    return decode_raw_slice_bytes(payload, len, H, W, u16);
  }
  throw CodecError("unknown VPXV slice codec tag");
}

void store_plane_to_samples(const Plane& p, VolumeDType dtype, uint16_t* dst) {
  const size_t n = p.samples.size();
  for (size_t i = 0; i < n; ++i) {
    dst[i] = from_unsigned_domain(p.samples[i], dtype);
  }
}

void validate_gos_slice_headers(const ParsedVolume& pv, const uint8_t* data) {
  for (size_t g = 0; g < pv.num_gos; ++g) {
    const int z0 = static_cast<int>(g) * pv.hdr.gos_size;
    const int z1 = std::min(pv.hdr.depth, z0 + pv.hdr.gos_size);
    const size_t g_end = pv.gos_offsets[g] + pv.gos_lengths[g];
    size_t pos = pv.gos_offsets[g];
    for (int z = z0; z < z1; ++z) {
      if (pos + 9 > g_end) throw CodecError("truncated VPXV slice header");
      const uint8_t tag = data[pos];
      if (tag > kSliceRaw) throw CodecError("unknown VPXV slice codec tag");
      if (z == z0 && (tag == kSliceFast3D || tag == kSliceFrontier3D)) {
        throw CodecError("VPXV GoS first slice must be an intra slice");
      }
      const uint32_t slen = read_u32_le(data + pos + 1);
      pos += 9;
      if (slen > g_end - pos) throw CodecError("truncated VPXV slice payload");
      pos += slen;
    }
    if (pos != g_end) throw CodecError("trailing bytes in VPXV GoS block");
  }
}

}  // namespace

VolumeHeader inspect_volume(const uint8_t* data, size_t size) {
  return parse_volume_directory(data, size).hdr;
}

std::vector<uint8_t> encode_volume(const Volume& vol, const VolumeOptions& opt) {
  if (vol.depth <= 0 || vol.height <= 0 || vol.width <= 0) throw CodecError("empty medical volume");
  if (vol.depth > kMaxVolDepth || vol.height > kMaxVolDim || vol.width > kMaxVolDim) {
    throw CodecError("medical volume dimensions exceed safety limits");
  }
  const uint8_t dt_raw = static_cast<uint8_t>(vol.dtype);
  if (dt_raw < 1 || dt_raw > 3) throw CodecError("invalid medical volume dtype");
  if (opt.effort < 0 || opt.effort > 3) throw CodecError("volume effort must be 0, 1, 2, or 3");
  if (opt.gos_size < 1 || opt.gos_size > 64) throw CodecError("volume gos_size must be between 1 and 64");

  const size_t slice_vox = vol.slice_voxels();
  if (slice_vox > kMaxImageSamples || static_cast<size_t>(vol.depth) > kMaxVolumeVoxels / slice_vox) {
    throw CodecError("medical volume voxel count exceeds safety limit");
  }
  const size_t total_vox = vol.total_voxels();
  if (vol.samples.size() < total_vox) throw CodecError("medical volume sample buffer too small");

  const bool u16 = vol.dtype != VolumeDType::U8;
  const uint32_t vol_crc = crc32_volume_samples(vol.samples.data(), total_vox, vol.dtype);
  const size_t num_gos =
      (static_cast<size_t>(vol.depth) + static_cast<size_t>(opt.gos_size) - 1) / static_cast<size_t>(opt.gos_size);

  std::vector<std::vector<uint8_t>> gos_blobs(num_gos);

  auto encode_one_gos = [&](size_t g) {
    const int z0 = static_cast<int>(g) * opt.gos_size;
    const int z1 = std::min(vol.depth, z0 + opt.gos_size);
    std::vector<uint8_t> g_out;
    Plane prev_p;
    Plane prev2_p;
    const size_t raw_slice_bytes = slice_vox * (u16 ? 2u : 1u);

    for (int z = z0; z < z1; ++z) {
      Plane cur_p = extract_slice_plane(vol, z);
      const uint32_t scrc =
          crc32_slice_samples(vol.samples.data() + static_cast<size_t>(z) * slice_vox, slice_vox, vol.dtype);

      uint8_t best_tag = kSliceFast2D;
      std::vector<uint8_t> best_bytes;

      if (z == z0) {
        best_bytes = encode_fast_plane(cur_p, nullptr);
        best_tag = kSliceFast2D;
        if (opt.effort >= 1) {
          auto fr2d = encode_frontier({cur_p}, u16, opt.effort);
          if (fr2d.size() < best_bytes.size()) {
            best_bytes = std::move(fr2d);
            best_tag = kSliceFrontier2D;
          }
        }
      } else {
        best_bytes = encode_fast_plane(cur_p, &prev_p);
        best_tag = kSliceFast3D;
        if (opt.effort == 0) {
          // Also check 2D fast in case of abrupt multi-series FOV cut
          auto f2d = encode_fast_plane(cur_p, nullptr);
          if (f2d.size() < best_bytes.size()) {
            best_bytes = std::move(f2d);
            best_tag = kSliceFast2D;
          }
        } else {
          const Plane* p2_ptr = (z >= z0 + 2) ? &prev2_p : nullptr;
          auto fr3d = encode_frontier_slice_3d(cur_p, prev_p, p2_ptr, u16, opt.effort);
          if (fr3d.size() < best_bytes.size()) {
            best_bytes = std::move(fr3d);
            best_tag = kSliceFrontier3D;
          }
          if (opt.effort >= 2) {
            auto fr2d = encode_frontier({cur_p}, u16, 1);
            if (fr2d.size() < best_bytes.size()) {
              best_bytes = std::move(fr2d);
              best_tag = kSliceFrontier2D;
            }
          }
        }
      }

      if (best_bytes.size() >= raw_slice_bytes) {
        best_bytes = make_raw_slice_bytes(cur_p, u16);
        best_tag = kSliceRaw;
      }

      g_out.push_back(best_tag);
      write_u32_le(g_out, static_cast<uint32_t>(best_bytes.size()));
      write_u32_le(g_out, scrc);
      g_out.insert(g_out.end(), best_bytes.begin(), best_bytes.end());

      prev2_p = std::move(prev_p);
      prev_p = std::move(cur_p);
    }
    gos_blobs[g] = std::move(g_out);
  };

  const unsigned hw_threads = std::max(1u, std::thread::hardware_concurrency());
  if (num_gos >= 2 && hw_threads >= 2) {
    std::atomic<size_t> next_g{0};
    std::mutex err_mu;
    std::exception_ptr first_err;
    const size_t n_workers = std::min<size_t>(num_gos, hw_threads);
    std::vector<std::thread> workers;
    workers.reserve(n_workers - 1);
    auto worker_fn = [&]() {
      while (true) {
        const size_t g = next_g.fetch_add(1, std::memory_order_relaxed);
        if (g >= num_gos) break;
        try {
          encode_one_gos(g);
        } catch (...) {
          std::lock_guard<std::mutex> lk(err_mu);
          if (!first_err) first_err = std::current_exception();
          next_g.store(num_gos, std::memory_order_relaxed);
          break;
        }
      }
    };
    for (size_t w = 0; w + 1 < n_workers; ++w) workers.emplace_back(worker_fn);
    worker_fn();
    for (auto& t : workers) t.join();
    if (first_err) std::rethrow_exception(first_err);
  } else {
    for (size_t g = 0; g < num_gos; ++g) encode_one_gos(g);
  }

  size_t total_payload = 0;
  for (const auto& b : gos_blobs) {
    if (b.size() >= (size_t(1) << 32)) throw CodecError("VPXV GoS exceeds 4 GiB");
    total_payload += b.size();
  }

  std::vector<uint8_t> out;
  out.reserve(24 + num_gos * 4 + total_payload);
  out.insert(out.end(), std::begin(kMagic), std::end(kMagic));
  out.push_back(kVersion);
  out.push_back(dt_raw);
  out.push_back(static_cast<uint8_t>(opt.effort));
  out.push_back(static_cast<uint8_t>(opt.gos_size));
  write_u32_le(out, static_cast<uint32_t>(vol.depth));
  write_u32_le(out, static_cast<uint32_t>(vol.height));
  write_u32_le(out, static_cast<uint32_t>(vol.width));
  write_u32_le(out, vol_crc);
  for (const auto& b : gos_blobs) {
    write_u32_le(out, static_cast<uint32_t>(b.size()));
  }
  for (const auto& b : gos_blobs) {
    out.insert(out.end(), b.begin(), b.end());
  }
  return out;
}

Volume decode_volume(const uint8_t* data, size_t size) {
  const ParsedVolume pv = parse_volume_directory(data, size);
  validate_gos_slice_headers(pv, data);

  Volume out;
  out.depth = pv.hdr.depth;
  out.height = pv.hdr.height;
  out.width = pv.hdr.width;
  out.dtype = pv.hdr.dtype;
  const size_t slice_vox = out.slice_voxels();
  const size_t total_vox = out.total_voxels();
  out.samples.assign(total_vox, 0);

  const bool u16 = out.dtype != VolumeDType::U8;

  auto decode_one_gos = [&](size_t g) {
    const int z0 = static_cast<int>(g) * pv.hdr.gos_size;
    const int z1 = std::min(out.depth, z0 + pv.hdr.gos_size);
    size_t pos = pv.gos_offsets[g];
    Plane prev_p;
    Plane prev2_p;
    for (int z = z0; z < z1; ++z) {
      const uint8_t tag = data[pos];
      const uint32_t slen = read_u32_le(data + pos + 1);
      const uint32_t scrc = read_u32_le(data + pos + 5);
      pos += 9;
      const Plane* ref_ptr = (z > z0) ? &prev_p : nullptr;
      const Plane* prev2_ptr = (z >= z0 + 2) ? &prev2_p : nullptr;
      Plane cur_p = decode_single_slice_record(tag, data + pos, slen, out.height, out.width, u16, ref_ptr, prev2_ptr);
      pos += slen;

      uint16_t* dst = out.samples.data() + static_cast<size_t>(z) * slice_vox;
      store_plane_to_samples(cur_p, out.dtype, dst);
      if (crc32_slice_samples(dst, slice_vox, out.dtype) != scrc) {
        throw CodecError("VPXV slice CRC-32 mismatch");
      }
      prev2_p = std::move(prev_p);
      prev_p = std::move(cur_p);
    }
  };

  const unsigned hw_threads = std::max(1u, std::thread::hardware_concurrency());
  if (pv.num_gos >= 2 && hw_threads >= 2) {
    std::atomic<size_t> next_g{0};
    std::mutex err_mu;
    std::exception_ptr first_err;
    const size_t n_workers = std::min<size_t>(pv.num_gos, hw_threads);
    std::vector<std::thread> workers;
    workers.reserve(n_workers - 1);
    auto worker_fn = [&]() {
      while (true) {
        const size_t g = next_g.fetch_add(1, std::memory_order_relaxed);
        if (g >= pv.num_gos) break;
        try {
          decode_one_gos(g);
        } catch (...) {
          std::lock_guard<std::mutex> lk(err_mu);
          if (!first_err) first_err = std::current_exception();
          next_g.store(pv.num_gos, std::memory_order_relaxed);
          break;
        }
      }
    };
    for (size_t w = 0; w + 1 < n_workers; ++w) workers.emplace_back(worker_fn);
    worker_fn();
    for (auto& t : workers) t.join();
    if (first_err) std::rethrow_exception(first_err);
  } else {
    for (size_t g = 0; g < pv.num_gos; ++g) decode_one_gos(g);
  }

  if (crc32_volume_samples(out.samples.data(), total_vox, out.dtype) != pv.hdr.volume_crc32) {
    throw CodecError("VPXV volume CRC-32 mismatch");
  }
  return out;
}

Volume decode_volume_slice(const uint8_t* data, size_t size, int z_index) {
  const ParsedVolume pv = parse_volume_directory(data, size);
  if (z_index < 0 || z_index >= pv.hdr.depth) {
    throw CodecError("VPXV slice index out of range");
  }

  const size_t g = static_cast<size_t>(z_index / pv.hdr.gos_size);
  const int z0 = static_cast<int>(g) * pv.hdr.gos_size;
  const int z1 = std::min(pv.hdr.depth, z0 + pv.hdr.gos_size);
  const size_t g_end = pv.gos_offsets[g] + pv.gos_lengths[g];

  // Validate headers in target GoS before decoding
  {
    size_t scan = pv.gos_offsets[g];
    for (int z = z0; z < z1; ++z) {
      if (scan + 9 > g_end) throw CodecError("truncated VPXV slice header");
      const uint8_t tag = data[scan];
      if (tag > kSliceRaw) throw CodecError("unknown VPXV slice codec tag");
      if (z == z0 && (tag == kSliceFast3D || tag == kSliceFrontier3D)) {
        throw CodecError("VPXV GoS first slice must be an intra slice");
      }
      const uint32_t slen = read_u32_le(data + scan + 1);
      scan += 9;
      if (slen > g_end - scan) throw CodecError("truncated VPXV slice payload");
      scan += slen;
    }
    if (scan != g_end) throw CodecError("trailing bytes in VPXV GoS block");
  }

  const bool u16 = pv.hdr.dtype != VolumeDType::U8;
  const size_t slice_vox = static_cast<size_t>(pv.hdr.height) * static_cast<size_t>(pv.hdr.width);
  size_t pos = pv.gos_offsets[g];
  Plane prev_p;
  Plane prev2_p;
  uint32_t target_crc = 0;

  for (int z = z0; z <= z_index; ++z) {
    const uint8_t tag = data[pos];
    const uint32_t slen = read_u32_le(data + pos + 1);
    const uint32_t scrc = read_u32_le(data + pos + 5);
    pos += 9;
    const Plane* ref_ptr = (z > z0) ? &prev_p : nullptr;
    const Plane* prev2_ptr = (z >= z0 + 2) ? &prev2_p : nullptr;
    Plane cur_p =
        decode_single_slice_record(tag, data + pos, slen, pv.hdr.height, pv.hdr.width, u16, ref_ptr, prev2_ptr);
    pos += slen;
    target_crc = scrc;
    prev2_p = std::move(prev_p);
    prev_p = std::move(cur_p);
  }

  Volume out;
  out.depth = 1;
  out.height = pv.hdr.height;
  out.width = pv.hdr.width;
  out.dtype = pv.hdr.dtype;
  out.samples.resize(slice_vox);
  store_plane_to_samples(prev_p, out.dtype, out.samples.data());
  if (crc32_slice_samples(out.samples.data(), slice_vox, out.dtype) != target_crc) {
    throw CodecError("VPXV slice CRC-32 mismatch");
  }
  return out;
}

}  // namespace vpxc
