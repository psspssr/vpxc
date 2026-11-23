#include "vpxc/codec.hpp"
#include "vpxc/frontier.hpp"
#include "vpxc/wp.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <zlib.h>

namespace vpxc {
namespace {

constexpr int kMaxLevels = 4;
constexpr int kMaxPalette = 512;
constexpr uint8_t kProfileRaw = 0;
constexpr uint8_t kProfileRaster = 1;
constexpr uint8_t kProfileKalman = 2;
constexpr uint8_t kProfileQxRice = 3;
constexpr uint8_t kProfileQxRans = 4;
constexpr uint8_t kProfilePalette = 5;
constexpr uint8_t kProfileSlgf = 6;
constexpr uint8_t kProfileWp = 7;
constexpr uint8_t kProfileFrontier = 8;
constexpr uint8_t kFlagSheaf = 1;
constexpr uint8_t kColorExtra = 1;

void append_u16(std::vector<uint8_t>& out, uint16_t value) {
  out.push_back(static_cast<uint8_t>(value & 255));
  out.push_back(static_cast<uint8_t>((value >> 8) & 255));
}

void append_u32(std::vector<uint8_t>& out, uint32_t value) {
  out.push_back(static_cast<uint8_t>(value & 255));
  out.push_back(static_cast<uint8_t>((value >> 8) & 255));
  out.push_back(static_cast<uint8_t>((value >> 16) & 255));
  out.push_back(static_cast<uint8_t>((value >> 24) & 255));
}

void append_i16(std::vector<uint8_t>& out, int value) {
  if (value < -32768 || value > 32767) throw CodecError("color coefficient does not fit in int16");
  append_u16(out, static_cast<uint16_t>(static_cast<int16_t>(value)));
}

void append_bytes(std::vector<uint8_t>& out, const uint8_t* data, size_t n) { out.insert(out.end(), data, data + n); }

void append_prefixed(std::vector<uint8_t>& out, const std::vector<uint8_t>& blob) {
  if (blob.size() >= (size_t(1) << 32)) throw CodecError("tile payload exceeds 4 GiB");
  append_u32(out, static_cast<uint32_t>(blob.size()));
  append_bytes(out, blob.data(), blob.size());
}

struct Reader {
  const uint8_t* data = nullptr;
  size_t size = 0;
  size_t pos = 0;
  void need(size_t n, const char* what) const {
    if (n > size - pos) throw CodecError(what);
  }
  uint8_t u8(const char* what) {
    need(1, what);
    return data[pos++];
  }
  uint16_t u16(const char* what) {
    need(2, what);
    const uint16_t v = static_cast<uint16_t>(data[pos] | (uint16_t(data[pos + 1]) << 8));
    pos += 2;
    return v;
  }
  int16_t i16(const char* what) { return static_cast<int16_t>(u16(what)); }
  uint32_t u32(const char* what) {
    need(4, what);
    const uint32_t v = uint32_t(data[pos]) | (uint32_t(data[pos + 1]) << 8) | (uint32_t(data[pos + 2]) << 16) |
                       (uint32_t(data[pos + 3]) << 24);
    pos += 4;
    return v;
  }
  std::span<const uint8_t> take(size_t n, const char* what) {
    need(n, what);
    std::span<const uint8_t> view(data + pos, n);
    pos += n;
    return view;
  }
};

uint32_t crc32_bytes(const uint8_t* data, size_t n) {
  uLong crc = ::crc32(0L, Z_NULL, 0);
  while (n) {
    const uInt chunk = n > (1u << 20) ? (1u << 20) : static_cast<uInt>(n);
    crc = ::crc32(crc, data, chunk);
    data += chunk;
    n -= chunk;
  }
  return static_cast<uint32_t>(crc);
}

void check_image(const Image& image) {
  if (image.channels != 1 && image.channels != 3 && image.channels != 4) {
    throw CodecError("channel count must be 1, 3, or 4");
  }
  if (image.height <= 0 || image.width <= 0) throw CodecError("empty image");
  if (image.height > 20000 || image.width > 20000) throw CodecError("image dimension exceeds 20000");
  const size_t need = image.plane_size() * static_cast<size_t>(image.channels);
  if (image.samples.size() < need) throw CodecError("image sample buffer has the wrong size");
}

std::vector<uint8_t> image_bytes(const Image& image) {
  const size_t n = image.plane_size() * static_cast<size_t>(image.channels);
  std::vector<uint8_t> out;
  if (!image.u16) {
    out.resize(n);
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<uint8_t>(image.samples[i] & 255);
  } else {
    out.resize(n * 2);
    for (size_t i = 0; i < n; ++i) {
      const uint16_t v = image.samples[i];
      out[2 * i] = static_cast<uint8_t>(v & 255);
      out[2 * i + 1] = static_cast<uint8_t>(v >> 8);
    }
  }
  return out;
}

std::vector<uint8_t> tile_bytes(const Image& image, int y0, int y1, int x0, int x1) {
  const int height = y1 - y0;
  const int width = x1 - x0;
  const int channels = image.channels;
  const size_t pixels = static_cast<size_t>(height) * static_cast<size_t>(width);
  std::vector<uint8_t> out(pixels * static_cast<size_t>(channels) * (image.u16 ? 2 : 1));
  size_t o = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t pix = static_cast<size_t>(y0 + y) * static_cast<size_t>(image.width) + static_cast<size_t>(x0 + x);
      for (int c = 0; c < channels; ++c) {
        const uint16_t sample = channels == 1 ? image.samples[pix] : image.samples[pix * static_cast<size_t>(channels) + static_cast<size_t>(c)];
        if (!image.u16) {
          out[o++] = static_cast<uint8_t>(sample & 255);
        } else {
          out[o++] = static_cast<uint8_t>(sample & 255);
          out[o++] = static_cast<uint8_t>(sample >> 8);
        }
      }
    }
  }
  return out;
}

std::vector<Plane> split_region(const Image& image, int y0, int y1, int x0, int x1) {
  const int height = y1 - y0;
  const int width = x1 - x0;
  std::vector<Plane> planes;
  planes.reserve(static_cast<size_t>(image.channels));
  for (int c = 0; c < image.channels; ++c) {
    Plane plane(height, width);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const size_t pix = static_cast<size_t>(y0 + y) * static_cast<size_t>(image.width) + static_cast<size_t>(x0 + x);
        const uint16_t sample = image.channels == 1 ? image.samples[pix]
                                                    : image.samples[pix * static_cast<size_t>(image.channels) + static_cast<size_t>(c)];
        plane.at(y, x) = image.u16 ? static_cast<int32_t>(sample) : static_cast<int32_t>(sample & 255);
      }
    }
    planes.push_back(std::move(plane));
  }
  return planes;
}

void store_sample(Image& dest, int y, int x, int c, uint16_t value) {
  const size_t pix = static_cast<size_t>(y) * static_cast<size_t>(dest.width) + static_cast<size_t>(x);
  if (dest.channels == 1) dest.samples[pix] = value;
  else dest.samples[pix * static_cast<size_t>(dest.channels) + static_cast<size_t>(c)] = value;
}

void store_planes(Image& dest, int y0, int x0, const std::vector<Plane>& planes) {
  if (planes.empty()) throw CodecError("decoded tile is empty");
  int32_t lo = planes[0].samples.empty() ? 0 : planes[0].samples[0];
  int32_t hi = lo;
  for (const Plane& plane : planes) {
    for (int32_t sample : plane.samples) {
      lo = std::min(lo, sample);
      hi = std::max(hi, sample);
    }
  }
  const int limit = dest.u16 ? 65535 : 255;
  if (lo < 0 || hi > limit) {
    throw CodecError("decoded sample " + std::to_string(lo) + ".." + std::to_string(hi) + " outside " +
                     (dest.u16 ? "uint16" : "uint8"));
  }
  for (int y = 0; y < planes[0].height; ++y) {
    for (int x = 0; x < planes[0].width; ++x) {
      for (int c = 0; c < static_cast<int>(planes.size()); ++c) {
        store_sample(dest, y0 + y, x0 + x, c, static_cast<uint16_t>(planes[static_cast<size_t>(c)].at(y, x)));
      }
    }
  }
}

void store_bytes(Image& dest, int y0, int y1, int x0, int x1, const uint8_t* raw, size_t n) {
  const int height = y1 - y0;
  const int width = x1 - x0;
  const int channels = dest.channels;
  const size_t need = static_cast<size_t>(height) * static_cast<size_t>(width) * static_cast<size_t>(channels) * (dest.u16 ? 2 : 1);
  if (n < need) throw CodecError("truncated raw tile");
  size_t o = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < channels; ++c) {
        uint16_t value = raw[o++];
        if (dest.u16) value = static_cast<uint16_t>(value | (uint16_t(raw[o++]) << 8));
        store_sample(dest, y0 + y, x0 + x, c, value);
      }
    }
  }
}

Plane med_residuals(const Plane& index) {
  const int h = index.height;
  const int w = index.width;
  Plane out(h, w);
  if (h <= 0 || w <= 0) return out;
  out.at(0, 0) = index.at(0, 0);
  for (int x = 1; x < w; ++x) out.at(0, x) = index.at(0, x) - index.at(0, x - 1);
  for (int y = 1; y < h; ++y) out.at(y, 0) = index.at(y, 0) - index.at(y - 1, 0);
  for (int y = 1; y < h; ++y) {
    for (int x = 1; x < w; ++x) {
      const int32_t left = index.at(y, x - 1);
      const int32_t up = index.at(y - 1, x);
      const int32_t ul = index.at(y - 1, x - 1);
      const int32_t hi = std::max(left, up);
      const int32_t lo = std::min(left, up);
      int32_t pred = left + up - ul;
      if (ul < lo) pred = hi;
      else if (ul > hi) pred = lo;
      out.at(y, x) = index.at(y, x) - pred;
    }
  }
  return out;
}

Plane med_reconstruct(const Plane& resid) {
  const int h = resid.height;
  const int w = resid.width;
  Plane out(h, w);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (y == 0 && x == 0) out.at(0, 0) = resid.at(0, 0);
      else if (y == 0) out.at(0, x) = out.at(0, x - 1) + resid.at(0, x);
      else if (x == 0) out.at(y, 0) = out.at(y - 1, 0) + resid.at(y, 0);
      else {
        const int32_t left = out.at(y, x - 1);
        const int32_t up = out.at(y - 1, x);
        const int32_t ul = out.at(y - 1, x - 1);
        const int32_t hi = left > up ? left : up;
        const int32_t lo = left > up ? up : left;
        int32_t pred = left + up - ul;
        if (ul < lo) pred = hi;
        else if (ul > hi) pred = lo;
        out.at(y, x) = pred + resid.at(y, x);
      }
    }
  }
  return out;
}

Plane activity_plane(const ContextMap& ctx) {
  Plane plane(ctx.height, ctx.width);
  const size_t n = std::min(ctx.samples.size(), plane.samples.size());
  for (size_t i = 0; i < n; ++i) plane.samples[i] = ctx.samples[i];
  return plane;
}

std::vector<uint8_t> raw_record(const Image& image, int y0, int y1, int x0, int x1) {
  std::vector<uint8_t> out{kProfileRaw};
  const auto bytes = tile_bytes(image, y0, y1, x0, x1);
  append_bytes(out, bytes.data(), bytes.size());
  return out;
}

std::vector<uint8_t> raster_record(const std::vector<Plane>& planes, bool kalman) {
  std::vector<uint8_t> body;
  const Mask mask(planes[0].height, planes[0].width, true);
  for (const Plane& plane : planes) append_prefixed(body, encode_raster(plane, mask, kalman));
  std::vector<uint8_t> out{kalman ? kProfileKalman : kProfileRaster, 0, 0};
  append_prefixed(out, body);
  return out;
}

std::vector<Halo> make_halos(const std::vector<Plane>& lifted, int y0, int x0, int y1, int x1) {
  int pt = std::min(16, y0);
  int pl = std::min(16, x0);
  pt &= ~1;
  pl &= ~1;
  const int height = y1 - y0;
  const int width = x1 - x0;
  std::vector<Halo> halos;
  halos.reserve(lifted.size());
  for (const Plane& plane : lifted) {
    Halo halo;
    if (pt) {
      Plane top(pt, width);
      for (int y = 0; y < pt; ++y)
        for (int x = 0; x < width; ++x) top.at(y, x) = plane.at(y0 - pt + y, x0 + x);
      halo.top = std::move(top);
    }
    if (pl) {
      Plane left(height, pl);
      for (int y = 0; y < height; ++y)
        for (int x = 0; x < pl; ++x) left.at(y, x) = plane.at(y0 + y, x0 - pl + x);
      halo.left = std::move(left);
    }
    if (pt && pl) {
      Plane corner(pt, pl);
      for (int y = 0; y < pt; ++y)
        for (int x = 0; x < pl; ++x) corner.at(y, x) = plane.at(y0 - pt + y, x0 - pl + x);
      halo.corner = std::move(corner);
    }
    halos.push_back(std::move(halo));
  }
  return halos;
}

std::vector<Halo> halos_from_decoded(const Image& dest, int y0, int x0, int y1, int x1, const ColorTransform& transform) {
  int pt = std::min(16, y0);
  int pl = std::min(16, x0);
  pt &= ~1;
  pl &= ~1;
  if (pt == 0 && pl == 0) return std::vector<Halo>(static_cast<size_t>(dest.channels));
  const std::vector<Plane> tops = pt ? apply_color_forward(split_region(dest, y0 - pt, y0, x0, x1), transform) : std::vector<Plane>{};
  const std::vector<Plane> lefts = pl ? apply_color_forward(split_region(dest, y0, y1, x0 - pl, x0), transform) : std::vector<Plane>{};
  const std::vector<Plane> corners =
      (pt && pl) ? apply_color_forward(split_region(dest, y0 - pt, y0, x0 - pl, x0), transform) : std::vector<Plane>{};
  const size_t count = !tops.empty() ? tops.size() : (!lefts.empty() ? lefts.size() : corners.size());
  std::vector<Halo> halos(count);
  for (size_t i = 0; i < count; ++i) {
    if (!tops.empty()) halos[i].top = tops[i];
    if (!lefts.empty()) halos[i].left = lefts[i];
    if (!corners.empty()) halos[i].corner = corners[i];
  }
  return halos;
}

std::vector<uint8_t> slgf_record(const std::vector<Plane>& planes, int effort, int version, const std::vector<Halo>* halos,
                                bool use_static) {
  std::vector<uint8_t> payload;
  const int64_t area = static_cast<int64_t>(planes[0].height) * planes[0].width;
  if (version <= 2) {
    payload = encode_slgf_tile(planes, kMaxLevels);
  } else {
    auto flat = encode_slgf_v3(planes, kMaxLevels, effort, halos, use_static, 0);
    auto sheaf = encode_slgf_v3(planes, kMaxLevels, effort, halos, use_static, 1);
    if (!sheaf.empty() && sheaf.size() < flat.size()) {
      sheaf[0] = static_cast<uint8_t>(sheaf[0] | 0x10);
      payload = std::move(sheaf);
    } else {
      payload = std::move(flat);
    }
    if (!payload.empty()) payload[0] = static_cast<uint8_t>(payload[0] | 0x80);
    if (effort >= 3 || area < 128 * 128) {
      auto legacy = encode_slgf_tile(planes, kMaxLevels);
      if (payload.empty() || legacy.size() < payload.size()) payload = std::move(legacy);
    }
  }
  std::vector<uint8_t> out{kProfileSlgf, 0, 0};
  append_prefixed(out, payload);
  return out;
}

double tower_cost(const std::vector<Tower>& towers) {
  __int128 total = 0;
  int64_t count = 0;
  for (const Tower& tower : towers) {
    const size_t levels = std::min(tower.details.size(), tower.levels.size());
    for (size_t level = 0; level < levels; ++level) {
      const ParityMasks par = parity_masks(tower.levels[level].valid);
      const Plane& lifted = tower.details[level];
      for (size_t i = 0; i < par.odd.bits.size() && i < lifted.samples.size(); ++i) {
        if (!par.odd.bits[i]) continue;
        const int32_t value = lifted.samples[i];
        total += value < 0 ? -static_cast<__int128>(value) : value;
        ++count;
      }
    }
  }
  if (count == 0) return 0.0;
  return static_cast<double>(total) / static_cast<double>(count);
}

std::vector<uint8_t> qx_record(const std::vector<Tower>& towers, bool sheaf, bool rans) {
  if (towers.empty()) throw CodecError("quincunx tile is empty");
  const int nlevels = static_cast<int>(towers[0].levels.size());
  std::vector<uint8_t> body;
  for (const Tower& tower : towers) {
    if (static_cast<int>(tower.levels.size()) != nlevels || static_cast<int>(tower.details.size()) != nlevels) {
      throw CodecError("planes disagree on quincunx depth");
    }
    append_prefixed(body, encode_raster(tower.coarse, tower.coarse_valid, false));
    for (int level = nlevels - 1; level >= 0; --level) {
      const Plane& lifted = tower.details[static_cast<size_t>(level)];
      const Mask& valid = tower.levels[static_cast<size_t>(level)].valid;
      const ParityMasks par = parity_masks(valid);
      const Plane activity = activity_plane(activity_context(lifted, valid));
      std::vector<uint8_t> blob;
      if (rans) {
        const Plane shifts = shift_map_from_activity(activity);
        blob = encode_dyadic_rans(lifted, par.odd, shifts);
      } else {
        blob = encode_rice_values(lifted, par.odd, &activity, false);
      }
      append_prefixed(body, blob);
    }
  }
  std::vector<uint8_t> out{rans ? kProfileQxRans : kProfileQxRice, static_cast<uint8_t>(sheaf ? kFlagSheaf : 0),
                           static_cast<uint8_t>(nlevels)};
  append_prefixed(out, body);
  return out;
}

std::vector<std::vector<uint8_t>> quincunx_candidates(const std::vector<Plane>& planes, const char* entropy) {
  const auto built = build_levels(planes[0].height, planes[0].width, kMaxLevels);
  if (built.first.empty()) return {};
  struct Analysis {
    bool sheaf = false;
    std::vector<Tower> towers;
    double cost = 0;
  };
  std::vector<Analysis> analyses;
  for (bool sheaf : {false, true}) {
    Analysis item;
    item.sheaf = sheaf;
    for (const Plane& plane : planes) item.towers.push_back(forward_tower(plane, kMaxLevels, sheaf));
    item.cost = tower_cost(item.towers);
    analyses.push_back(std::move(item));
  }
  std::stable_sort(analyses.begin(), analyses.end(),
                   [](const Analysis& a, const Analysis& b) { return a.cost < b.cost; });
  std::vector<Analysis> chosen;
  chosen.push_back(std::move(analyses[0]));
  if (analyses.size() == 2 && analyses[1].cost <= chosen[0].cost * 1.02 + 0.05) chosen.push_back(std::move(analyses[1]));
  std::vector<std::vector<uint8_t>> records;
  for (const Analysis& item : chosen) {
    if (entropy == nullptr || std::strcmp(entropy, "rice") == 0) records.push_back(qx_record(item.towers, item.sheaf, false));
    if (entropy == nullptr || std::strcmp(entropy, "rans") == 0) records.push_back(qx_record(item.towers, item.sheaf, true));
  }
  return records;
}

std::optional<std::vector<uint8_t>> palette_record(const Image& image, int y0, int y1, int x0, int x1, int effort) {
  const int height = y1 - y0;
  const int width = x1 - x0;
  const int channels = image.channels;
  const int pixels = height * width;
  if (pixels <= 0) return std::nullopt;
  const int bpp = (image.u16 ? 2 : 1) * channels;
  std::vector<uint8_t> keys(static_cast<size_t>(pixels) * static_cast<size_t>(bpp));
  std::vector<uint16_t> samples(static_cast<size_t>(pixels) * static_cast<size_t>(channels));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int pix = y * width + x;
      const size_t src = static_cast<size_t>(y0 + y) * static_cast<size_t>(image.width) + static_cast<size_t>(x0 + x);
      for (int c = 0; c < channels; ++c) {
        const uint16_t sample = channels == 1 ? image.samples[src] : image.samples[src * static_cast<size_t>(channels) + static_cast<size_t>(c)];
        const uint16_t stored = image.u16 ? sample : static_cast<uint16_t>(sample & 255);
        samples[static_cast<size_t>(pix) * static_cast<size_t>(channels) + static_cast<size_t>(c)] = stored;
        if (!image.u16) keys[static_cast<size_t>(pix) * static_cast<size_t>(bpp) + static_cast<size_t>(c)] = static_cast<uint8_t>(stored);
        else {
          keys[static_cast<size_t>(pix) * static_cast<size_t>(bpp) + static_cast<size_t>(2 * c)] = static_cast<uint8_t>(stored & 255);
          keys[static_cast<size_t>(pix) * static_cast<size_t>(bpp) + static_cast<size_t>(2 * c + 1)] = static_cast<uint8_t>(stored >> 8);
        }
      }
    }
  }
  std::vector<int> order(static_cast<size_t>(pixels));
  for (int i = 0; i < pixels; ++i) order[static_cast<size_t>(i)] = i;
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return std::memcmp(keys.data() + static_cast<size_t>(a) * static_cast<size_t>(bpp),
                       keys.data() + static_cast<size_t>(b) * static_cast<size_t>(bpp), static_cast<size_t>(bpp)) < 0;
  });
  std::vector<int> uniq;
  std::vector<int> inv(static_cast<size_t>(pixels));
  for (int rank = 0; rank < pixels; ++rank) {
    const int pix = order[static_cast<size_t>(rank)];
    if (uniq.empty() || std::memcmp(keys.data() + static_cast<size_t>(pix) * static_cast<size_t>(bpp),
                                    keys.data() + static_cast<size_t>(uniq.back()) * static_cast<size_t>(bpp),
                                    static_cast<size_t>(bpp)) != 0) {
      uniq.push_back(pix);
    }
    inv[static_cast<size_t>(pix)] = static_cast<int>(uniq.size()) - 1;
  }
  if (uniq.empty() || static_cast<int>(uniq.size()) > kMaxPalette) return std::nullopt;
  const int ncolors = static_cast<int>(uniq.size());
  std::vector<int64_t> sums(static_cast<size_t>(ncolors), 0);
  for (int i = 0; i < ncolors; ++i) {
    const int pix = uniq[static_cast<size_t>(i)];
    for (int c = 0; c < channels; ++c) sums[static_cast<size_t>(i)] += samples[static_cast<size_t>(pix) * static_cast<size_t>(channels) + static_cast<size_t>(c)];
  }
  std::vector<int> lum(static_cast<size_t>(ncolors));
  for (int i = 0; i < ncolors; ++i) lum[static_cast<size_t>(i)] = i;
  std::stable_sort(lum.begin(), lum.end(), [&](int a, int b) { return sums[static_cast<size_t>(a)] < sums[static_cast<size_t>(b)]; });
  std::vector<int> rank(static_cast<size_t>(ncolors));
  for (int i = 0; i < ncolors; ++i) rank[static_cast<size_t>(lum[static_cast<size_t>(i)])] = i;
  Plane index(height, width);
  for (int i = 0; i < pixels; ++i) index.samples[static_cast<size_t>(i)] = rank[static_cast<size_t>(inv[static_cast<size_t>(i)])];
  const Mask mask(height, width, true);
  const Plane zeros(height, width);
  const Plane med = med_residuals(index);
  std::vector<uint8_t> chosen = encode_dyadic_rans(med, mask, zeros);
  int mode = 2;
  const auto raster = encode_raster(index, mask, false);
  if (raster.size() < chosen.size()) {
    mode = 0;
    chosen = raster;
  }
  if (effort >= 2 && pixels <= 65536) {
    const auto kalman = encode_raster(index, mask, true);
    if (kalman.size() < chosen.size()) {
      mode = 1;
      chosen = kalman;
    }
  }
  std::vector<uint8_t> payload;
  append_u16(payload, static_cast<uint16_t>(ncolors));
  for (int color = 0; color < ncolors; ++color) {
    const int pix = uniq[static_cast<size_t>(lum[static_cast<size_t>(color)])];
    for (int c = 0; c < channels; ++c) {
      const uint16_t sample = samples[static_cast<size_t>(pix) * static_cast<size_t>(channels) + static_cast<size_t>(c)];
      if (!image.u16) payload.push_back(static_cast<uint8_t>(sample));
      else append_u16(payload, sample);
    }
  }
  append_prefixed(payload, chosen);
  std::vector<uint8_t> out{kProfilePalette, static_cast<uint8_t>(mode), 0};
  append_prefixed(out, payload);
  return out;
}

std::vector<uint8_t> wp_record(const std::vector<Plane>& planes) {
  const auto payload = encode_wp(planes);
  std::vector<uint8_t> out{kProfileWp, 0, 0};
  append_prefixed(out, payload);
  return out;
}

std::vector<uint8_t> frontier_record(const Image& image, int y0, int y1, int x0, int x1, int effort) {
  const auto raw_planes = split_region(image, y0, y1, x0, x1);
  const auto payload = encode_frontier(raw_planes, image.u16, effort);
  std::vector<uint8_t> out{kProfileFrontier, 0, 0};
  append_prefixed(out, payload);
  return out;
}

bool needs_legacy_planes(const Image& image, const EncodeOptions& opt) {
  if (opt.profile != nullptr && opt.profile[0] != '\0') {
    return std::strcmp(opt.profile, "raw") != 0 && std::strcmp(opt.profile, "palette") != 0 &&
           std::strcmp(opt.profile, "frontier") != 0;
  }
  if (opt.version <= 2) return true;
  for (int y0 = 0; y0 < image.height; y0 += opt.tile) {
    for (int x0 = 0; x0 < image.width; x0 += opt.tile) {
      const int th = std::min(image.height, y0 + opt.tile) - y0;
      const int tw = std::min(image.width, x0 + opt.tile) - x0;
      if (static_cast<int64_t>(th) * tw <= 65536) return true;
    }
  }
  return false;
}

std::vector<uint8_t> shortest(std::vector<std::vector<uint8_t>> records) {
  if (records.empty()) throw CodecError("no tile candidate");
  size_t best = 0;
  for (size_t i = 1; i < records.size(); ++i) {
    if (records[i].size() < records[best].size()) best = i;
  }
  return std::move(records[best]);
}

std::vector<uint8_t> encode_tile(const Image& image, int y0, int y1, int x0, int x1, const ColorTransform& transform,
                                int effort, const char* profile, int version, const std::vector<Halo>* halos) {
  const int height = y1 - y0;
  const int width = x1 - x0;
  const int64_t area = static_cast<int64_t>(height) * width;
  if (profile != nullptr && profile[0] != '\0') {
    if (std::strcmp(profile, "raw") == 0) return raw_record(image, y0, y1, x0, x1);
    if (std::strcmp(profile, "palette") == 0) {
      auto palette = palette_record(image, y0, y1, x0, x1, 2);
      if (!palette) throw CodecError(std::string("profile ") + profile + " does not apply to this tile");
      return std::move(*palette);
    }
    if (std::strcmp(profile, "frontier") == 0) {
      if (version <= 2) throw CodecError("profile frontier requires bitstream version 3");
      return frontier_record(image, y0, y1, x0, x1, effort);
    }
    const std::vector<Plane> planes = apply_color_forward(split_region(image, y0, y1, x0, x1), transform);
    if (std::strcmp(profile, "raster") == 0) return raster_record(planes, false);
    if (std::strcmp(profile, "kalman") == 0) return raster_record(planes, true);
    if (std::strcmp(profile, "slgf") == 0) return slgf_record(planes, effort, version, halos, true);
    if (std::strcmp(profile, "wp") == 0) return wp_record(planes);
    if (std::strcmp(profile, "slgf-zero") == 0) return slgf_record(planes, std::min(effort, 2), version, halos, false);
    if (std::strcmp(profile, "qx-rice") == 0 || std::strcmp(profile, "qx-rans") == 0) {
      auto records = quincunx_candidates(planes, std::strcmp(profile, "qx-rans") == 0 ? "rans" : "rice");
      if (records.empty()) return raster_record(planes, false);
      return shortest(std::move(records));
    }
    throw CodecError(std::string("unknown profile ") + profile);
  }
  std::vector<std::vector<uint8_t>> candidates;
  candidates.push_back(raw_record(image, y0, y1, x0, x1));
  if (version >= 3) {
    candidates.push_back(frontier_record(image, y0, y1, x0, x1, effort));
  }
  if (auto palette = palette_record(image, y0, y1, x0, x1, effort)) candidates.push_back(std::move(*palette));
  if (version <= 2 || area <= 65536) {
    const std::vector<Plane> planes = apply_color_forward(split_region(image, y0, y1, x0, x1), transform);
    candidates.push_back(slgf_record(planes, effort, version, halos, true));
    if ((effort >= 2 || std::min(height, width) < 32) && area <= 65536) {
      candidates.push_back(raster_record(planes, true));
    }
  }
  return shortest(std::move(candidates));
}

std::vector<Plane> decode_raster_planes(std::span<const uint8_t> payload, int height, int width, int channels, bool kalman) {
  Reader reader{payload.data(), payload.size(), 0};
  std::vector<Plane> planes;
  const Mask mask(height, width, true);
  for (int c = 0; c < channels; ++c) {
    const uint32_t len = reader.u32("truncated length prefix");
    const auto blob = reader.take(len, "truncated prefixed blob");
    planes.push_back(decode_raster(blob.data(), blob.size(), height, width, mask, kalman));
  }
  if (reader.pos != reader.size) throw CodecError("raster tile has trailing bytes");
  return planes;
}

std::vector<Plane> decode_qx_planes(std::span<const uint8_t> payload, int height, int width, int channels, int nlevels,
                                   bool sheaf, bool rans) {
  const auto built = build_levels(height, width, nlevels);
  if (static_cast<int>(built.first.size()) != nlevels) throw CodecError("quincunx depth does not match this tile");
  Reader reader{payload.data(), payload.size(), 0};
  std::vector<Plane> planes;
  for (int c = 0; c < channels; ++c) {
    const uint32_t len = reader.u32("truncated length prefix");
    const auto blob = reader.take(len, "truncated prefixed blob");
    if (nlevels == 0) {
      const Mask mask(height, width, true);
      planes.push_back(decode_raster(blob.data(), blob.size(), height, width, mask, false));
      continue;
    }
    const Mask& coarse_valid = built.first.back().packed_mask;
    Plane current = decode_raster(blob.data(), blob.size(), coarse_valid.height, coarse_valid.width, coarse_valid, false);
    for (int level = nlevels - 1; level >= 0; --level) {
      const Level& spec = built.first[static_cast<size_t>(level)];
      const uint32_t detail_len = reader.u32("truncated length prefix");
      const auto detail = reader.take(detail_len, "truncated prefixed blob");
      Plane spatial = unpack_even(current, spec.info, spec.valid);
      const ParityMasks par = parity_masks(spec.valid);
      const Plane activity = activity_plane(activity_context(spatial, spec.valid));
      Plane resid;
      if (rans) {
        const Plane shifts = shift_map_from_activity(activity);
        resid = decode_dyadic_rans(detail, spatial.height, spatial.width, par.odd, shifts);
      } else {
        resid = decode_rice_values(detail, spatial.height, spatial.width, par.odd, &activity, false);
      }
      for (size_t i = 0; i < par.odd.bits.size() && i < spatial.samples.size() && i < resid.samples.size(); ++i) {
        if (par.odd.bits[i]) spatial.samples[i] = resid.samples[i];
      }
      current = lift_inverse(spatial, spec.valid, sheaf);
    }
    planes.push_back(std::move(current));
  }
  if (reader.pos != reader.size) throw CodecError("quincunx tile has trailing bytes");
  return planes;
}

void decode_palette(Image& dest, int y0, int x0, std::span<const uint8_t> payload, int height, int width, int mode) {
  if (payload.size() < 2) throw CodecError("truncated palette");
  const int ncolors = payload[0] | (payload[1] << 8);
  if (ncolors <= 0 || ncolors > kMaxPalette) throw CodecError("palette size out of range");
  const int channels = dest.channels;
  const int sample_bytes = dest.u16 ? 2 : 1;
  const size_t nbytes = static_cast<size_t>(ncolors) * static_cast<size_t>(channels) * static_cast<size_t>(sample_bytes);
  if (2 + nbytes > payload.size()) throw CodecError("truncated palette colors");
  std::vector<uint16_t> colors(static_cast<size_t>(ncolors) * static_cast<size_t>(channels));
  size_t o = 2;
  for (size_t i = 0; i < colors.size(); ++i) {
    uint16_t value = payload[o++];
    if (dest.u16) value = static_cast<uint16_t>(value | (uint16_t(payload[o++]) << 8));
    colors[i] = value;
  }
  Reader reader{payload.data(), payload.size(), o};
  const uint32_t index_len = reader.u32("truncated length prefix");
  const auto index_blob = reader.take(index_len, "truncated prefixed blob");
  if (reader.pos != reader.size) throw CodecError("palette tile has trailing bytes");
  const Mask mask(height, width, true);
  Plane index;
  if (mode == 2) {
    const Plane zeros(height, width);
    const Plane resid = decode_dyadic_rans(index_blob, height, width, mask, zeros);
    index = med_reconstruct(resid);
  } else if (mode == 0 || mode == 1) {
    index = decode_raster(index_blob.data(), index_blob.size(), height, width, mask, mode == 1);
  } else {
    throw CodecError("unknown palette mode flag " + std::to_string(mode));
  }
  for (int32_t sample : index.samples) {
    if (sample < 0 || sample >= ncolors) throw CodecError("palette index out of range");
  }
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int32_t id = index.at(y, x);
      for (int c = 0; c < channels; ++c) {
        store_sample(dest, y0 + y, x0 + x, c, colors[static_cast<size_t>(id) * static_cast<size_t>(channels) + static_cast<size_t>(c)]);
      }
    }
  }
}

}  // namespace

std::vector<uint8_t> encode_image(const Image& image, const EncodeOptions& opt) {
  check_image(image);
  if (opt.tile < 8 || opt.tile > 4096) throw CodecError("tile size must be between 8 and 4096");
  if (opt.version != 2 && opt.version != 3) throw CodecError("unsupported bitstream version");
  if (opt.version == 2 && opt.effort != 1 && opt.effort != 2) throw CodecError("version 2 effort must be 1 or 2");
  if (opt.version == 3 && (opt.effort < 1 || opt.effort > 3)) throw CodecError("effort must be 1, 2, or 3");
  const bool use_legacy_planes = needs_legacy_planes(image, opt);
  const ColorTransform transform =
      (image.channels >= 3 && use_legacy_planes)
          ? choose_color_transform(split_region(image, 0, image.height, 0, image.width))
          : ColorTransform{};
  const std::vector<uint8_t> samples = image_bytes(image);
  const uint32_t crc = crc32_bytes(samples.data(), samples.size());
  bool has_extra = false;
  if (image.channels == 4) {
    for (int weight : transform.extra) {
      if (weight != 0) has_extra = true;
    }
  }
  std::vector<uint8_t> out;
  out.push_back('V');
  out.push_back('P');
  out.push_back('X');
  out.push_back('C');
  out.push_back(static_cast<uint8_t>(opt.version));
  out.push_back(image.u16 ? 2 : 1);
  out.push_back(static_cast<uint8_t>(image.channels));
  out.push_back(static_cast<uint8_t>(transform.mode));
  out.push_back(static_cast<uint8_t>(transform.perm));
  out.push_back(has_extra ? kColorExtra : 0);
  append_u32(out, static_cast<uint32_t>(image.width));
  append_u32(out, static_cast<uint32_t>(image.height));
  append_u32(out, crc);
  append_u16(out, static_cast<uint16_t>(opt.tile));
  if (transform.mode == MODE_WSIL) {
    const SL3Params& w = transform.wsil;
    out.push_back(static_cast<uint8_t>(w.perm[0]));
    out.push_back(static_cast<uint8_t>(w.perm[1]));
    out.push_back(static_cast<uint8_t>(w.perm[2]));
    for (int value : {w.u01, w.u02, w.u12, w.l10, w.l20, w.l21, w.s1, w.s1_inv, w.s2, w.s2_inv}) append_i16(out, value);
  } else if (transform.mode != MODE_IDENTITY) {
    for (int value : transform.shear) append_i16(out, value);
  }
  if (has_extra) {
    for (int value : transform.extra) append_i16(out, value);
  }
  std::vector<Plane> lifted;
  if (opt.version == 3 && use_legacy_planes) {
    lifted = apply_color_forward(split_region(image, 0, image.height, 0, image.width), transform);
  }
  for (int y0 = 0; y0 < image.height; y0 += opt.tile) {
    for (int x0 = 0; x0 < image.width; x0 += opt.tile) {
      const int y1 = std::min(image.height, y0 + opt.tile);
      const int x1 = std::min(image.width, x0 + opt.tile);
      std::vector<Halo> halos;
      const std::vector<Halo>* halo_ptr = nullptr;
      if (opt.version == 3 && !lifted.empty()) {
        halos = make_halos(lifted, y0, x0, y1, x1);
        halo_ptr = &halos;
      }
      const auto tile = encode_tile(image, y0, y1, x0, x1, transform, opt.effort, opt.profile, opt.version, halo_ptr);
      append_bytes(out, tile.data(), tile.size());
    }
  }
  return out;
}

Image decode_image(const uint8_t* data, size_t size) {
  if (data == nullptr || size < 24) throw CodecError("truncated header");
  Reader reader{data, size, 0};
  const auto magic = reader.take(4, "truncated header");
  if (magic[0] != 'V' || magic[1] != 'P' || magic[2] != 'X' || magic[3] != 'C') {
    throw CodecError("not a readable vpxc bitstream");
  }
  const int version = reader.u8("truncated header");
  const int bps = reader.u8("truncated header");
  const int channels = reader.u8("truncated header");
  const int mode = reader.u8("truncated header");
  const int perm = reader.u8("truncated header");
  const int cflags = reader.u8("truncated header");
  const int width = static_cast<int>(reader.u32("truncated header"));
  const int height = static_cast<int>(reader.u32("truncated header"));
  const uint32_t crc = reader.u32("truncated header");
  const int tile = reader.u16("truncated header");
  if (version != 2 && version != 3) throw CodecError("not a readable vpxc bitstream");
  if ((bps != 1 && bps != 2) || (channels != 1 && channels != 3 && channels != 4)) {
    throw CodecError("unsupported sample format");
  }
  if (width <= 0 || height <= 0 || tile < 8 || width > 20000 || height > 20000) throw CodecError("invalid geometry");
  ColorTransform transform;
  transform.mode = mode;
  transform.perm = perm;
  if (mode == MODE_WSIL) {
    if (reader.pos + 23 > reader.size) throw CodecError("truncated WSIL header");
    const int p0 = reader.u8("truncated WSIL header");
    const int p1 = reader.u8("truncated WSIL header");
    const int p2 = reader.u8("truncated WSIL header");
    if (!(p0 != p1 && p0 != p2 && p1 != p2 && p0 >= 0 && p0 <= 2 && p1 >= 0 && p1 <= 2 && p2 >= 0 && p2 <= 2)) {
      throw CodecError("invalid WSIL channel permutation");
    }
    transform.wsil.perm[0] = p0;
    transform.wsil.perm[1] = p1;
    transform.wsil.perm[2] = p2;
    transform.wsil.u01 = reader.i16("truncated WSIL header");
    transform.wsil.u02 = reader.i16("truncated WSIL header");
    transform.wsil.u12 = reader.i16("truncated WSIL header");
    transform.wsil.l10 = reader.i16("truncated WSIL header");
    transform.wsil.l20 = reader.i16("truncated WSIL header");
    transform.wsil.l21 = reader.i16("truncated WSIL header");
    transform.wsil.s1 = reader.i16("truncated WSIL header");
    transform.wsil.s1_inv = reader.i16("truncated WSIL header");
    transform.wsil.s2 = reader.i16("truncated WSIL header");
    transform.wsil.s2_inv = reader.i16("truncated WSIL header");
  } else if (mode != MODE_IDENTITY) {
    transform.shear[0] = reader.i16("truncated color header");
    transform.shear[1] = reader.i16("truncated color header");
    transform.shear[2] = reader.i16("truncated color header");
  }
  if (cflags & kColorExtra) {
    transform.extra[0] = reader.i16("truncated 4th-channel color header");
    transform.extra[1] = reader.i16("truncated 4th-channel color header");
    transform.extra[2] = reader.i16("truncated 4th-channel color header");
  }
  Image out;
  out.height = height;
  out.width = width;
  out.channels = channels;
  out.u16 = bps == 2;
  out.samples.assign(static_cast<size_t>(height) * static_cast<size_t>(width) * static_cast<size_t>(channels), 0);
  for (int y0 = 0; y0 < height; y0 += tile) {
    for (int x0 = 0; x0 < width; x0 += tile) {
      const int y1 = std::min(height, y0 + tile);
      const int x1 = std::min(width, x0 + tile);
      const int th = y1 - y0;
      const int tw = x1 - x0;
      if (reader.pos >= reader.size) throw CodecError("truncated tile");
      const int prof = reader.u8("truncated tile");
      if (prof == kProfileRaw) {
        const size_t nbytes = static_cast<size_t>(th) * static_cast<size_t>(tw) * static_cast<size_t>(channels) * (bps == 1 ? 1 : 2);
        const auto raw = reader.take(nbytes, "truncated raw tile");
        store_bytes(out, y0, y1, x0, x1, raw.data(), raw.size());
        continue;
      }
      if (reader.pos + 6 > reader.size) throw CodecError("truncated tile header");
      const int flags = reader.u8("truncated tile header");
      const int nlevels = reader.u8("truncated tile header");
      const uint32_t length = reader.u32("truncated tile header");
      const auto payload = reader.take(length, "truncated tile payload");
      if (prof == kProfileSlgf) {
        std::vector<Plane> planes;
        if (version >= 3 && !payload.empty() && (payload[0] & 0x80)) {
          std::vector<uint8_t> modern(payload.begin(), payload.end());
          modern[0] = static_cast<uint8_t>(modern[0] & 0x7f);
          const std::vector<Halo> halos = halos_from_decoded(out, y0, x0, y1, x1, transform);
          planes = decode_slgf_v3(modern.data(), modern.size(), th, tw, channels, &halos);
        } else {
          planes = decode_slgf_tile(payload.data(), payload.size(), th, tw, channels);
        }
        store_planes(out, y0, x0, apply_color_inverse(planes, transform));
      } else if (prof == kProfileRaster || prof == kProfileKalman) {
        auto planes = decode_raster_planes(payload, th, tw, channels, prof == kProfileKalman);
        store_planes(out, y0, x0, apply_color_inverse(planes, transform));
      } else if (prof == kProfileQxRice || prof == kProfileQxRans) {
        auto planes = decode_qx_planes(payload, th, tw, channels, nlevels, (flags & kFlagSheaf) != 0, prof == kProfileQxRans);
        store_planes(out, y0, x0, apply_color_inverse(planes, transform));
      } else if (prof == kProfileWp) {
        auto planes = decode_wp(payload.data(), payload.size(), th, tw, channels);
        store_planes(out, y0, x0, apply_color_inverse(planes, transform));
      } else if (prof == kProfileFrontier) {
        if (version <= 2) throw CodecError("frontier tile requires bitstream version 3");
        auto planes = decode_frontier(payload.data(), payload.size(), th, tw, channels, bps == 2);
        store_planes(out, y0, x0, planes);
      } else if (prof == kProfilePalette) {
        decode_palette(out, y0, x0, payload, th, tw, flags);
      } else {
        throw CodecError("unknown tile profile " + std::to_string(prof));
      }
    }
  }
  if (reader.pos != reader.size) throw CodecError("trailing bytes after the last tile");
  const std::vector<uint8_t> got_bytes = image_bytes(out);
  if (crc32_bytes(got_bytes.data(), got_bytes.size()) != crc) throw CodecError("CRC mismatch");
  return out;
}

}  // namespace vpxc
