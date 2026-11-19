#include "vpxc/matching.hpp"

#include "vpxc/error.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <vector>

namespace vpxc {
namespace {

int64_t iabs(int64_t v) {
  if (v >= 0) return v;
  if (v == INT64_MIN) return INT64_MAX;
  return -v;
}

__int128 floor_div128(__int128 a, __int128 b) {
  __int128 q = a / b;
  __int128 r = a % b;
  if (r != 0 && a < 0) --q;
  return q;
}

int64_t median_value(std::vector<int64_t>& values) {
  const size_t n = values.size();
  if (n == 0) return 0;
  const size_t mid = n / 2;
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
  if (n & 1u) return values[mid];
  const int64_t upper = values[mid];
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid - 1),
                   values.begin() + static_cast<std::ptrdiff_t>(mid));
  return (values[mid - 1] + upper) / 2;
}

// Same estimator as spatial._estimate_noise_var. Unknown sites never enter the Laplacian.
int estimate_noise_var(const Plane& plane, const Mask& valid) {
  if (plane.height < 3 || plane.width < 3) return 0;
  std::vector<int64_t> lap;
  for (int y = 1; y < plane.height - 1; ++y) {
    for (int x = 1; x < plane.width - 1; ++x) {
      if (!valid.at(y, x) || !valid.at(y - 1, x) || !valid.at(y + 1, x) || !valid.at(y, x - 1) ||
          !valid.at(y, x + 1)) {
        continue;
      }
      const int64_t center = plane.at(y, x);
      const int64_t value = 4 * center - plane.at(y - 1, x) - plane.at(y + 1, x) - plane.at(y, x - 1) -
                            plane.at(y, x + 1);
      lap.push_back(iabs(value));
    }
  }
  if (lap.empty()) return 0;
  const int64_t mad = median_value(lap);
  if (mad == 0) return 0;
  int64_t est = (mad * mad + 4) / 9;
  if (est < 0) est = 0;
  if (est > 255) est = 255;
  return static_cast<int>(est);
}

using Key = std::array<int64_t, 4>;

struct Hit {
  int y;
  int x;
  int64_t high;
};

// Eight dihedral copies, including the duplicates produced by four turns of
// (identity, reflection, rotate). Those duplicates are inserted twice.
void orientations(int64_t north, int64_t south, int64_t west, int64_t east, Key out[8]) {
  int64_t cur[4] = {north, south, west, east};
  int k = 0;
  for (int turn = 0; turn < 4; ++turn) {
    out[k++] = Key{cur[0], cur[1], cur[2], cur[3]};
    out[k++] = Key{cur[1], cur[0], cur[3], cur[2]};
    const int64_t next0 = cur[2];
    const int64_t next1 = cur[3];
    const int64_t next2 = cur[1];
    const int64_t next3 = cur[0];
    cur[0] = next0;
    cur[1] = next1;
    cur[2] = next2;
    cur[3] = next3;
  }
}

void require_same(const Plane& a, const Mask& b, const char* what) {
  if (a.height != b.height || a.width != b.width) throw CodecError(what);
}

size_t count_true(const Mask& mask) {
  size_t n = 0;
  for (int y = 0; y < mask.height; ++y) {
    for (int x = 0; x < mask.width; ++x) {
      if (mask.at(y, x)) ++n;
    }
  }
  return n;
}

}  // namespace

Plane sheaf_kalman_delta(const Plane& samples, const Mask& known, const Mask& query, const Plane& flat_pred,
                         int radius) {
  if (samples.height < 0 || samples.width < 0) throw CodecError("negative plane shape");
  require_same(samples, known, "sheaf known mask shape differs");
  require_same(samples, query, "sheaf query mask shape differs");
  if (flat_pred.height != samples.height || flat_pred.width != samples.width) {
    throw CodecError("sheaf flat predictor shape differs");
  }
  const size_t n = static_cast<size_t>(samples.height) * static_cast<size_t>(samples.width);
  if (samples.samples.size() < n || known.bits.size() < n || query.bits.size() < n ||
      flat_pred.samples.size() < n) {
    throw CodecError("plane storage is short");
  }
  Plane out(samples.height, samples.width, 0);
  const size_t known_n = count_true(known);
  const size_t query_n = count_true(query);
  if (query_n < 16 || known_n < 8) return out;

  const int height = samples.height;
  const int width = samples.width;
  std::map<Key, std::vector<Hit>> table;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (!known.at(y, x)) continue;
      if (y < 1 || x < 1 || y + 1 >= height || x + 1 >= width) continue;
      const int64_t north = samples.at(y - 1, x);
      const int64_t south = samples.at(y + 1, x);
      const int64_t west = samples.at(y, x - 1);
      const int64_t east = samples.at(y, x + 1);
      const int64_t dc = (north + south + west + east) >> 2;
      const int64_t qn = (north - dc) >> 3;
      const int64_t qs = (south - dc) >> 3;
      const int64_t qw = (west - dc) >> 3;
      const int64_t qe = (east - dc) >> 3;
      const int64_t high = static_cast<int64_t>(samples.at(y, x)) - dc;
      Key keys[8];
      orientations(qn, qs, qw, qe, keys);
      for (const Key& key : keys) {
        auto& bucket = table[key];
        if (bucket.size() < 8) bucket.push_back(Hit{y, x, high});
      }
    }
  }
  if (table.empty()) return out;

  const int s_eta2 = known_n >= 16 ? estimate_noise_var(samples, known) : 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (!query.at(y, x)) continue;
      if (y < 1 || x < 1 || y + 1 >= height || x + 1 >= width) continue;
      const int64_t north = samples.at(y - 1, x);
      const int64_t south = samples.at(y + 1, x);
      const int64_t west = samples.at(y, x - 1);
      const int64_t east = samples.at(y, x + 1);
      const int64_t dc = (north + south + west + east) >> 2;
      const Key key{(north - dc) >> 3, (south - dc) >> 3, (west - dc) >> 3, (east - dc) >> 3};
      const auto found = table.find(key);
      if (found == table.end() || found->second.empty()) continue;
      const int64_t local = flat_pred.at(y, x);
      const int64_t grad = iabs(north - south) + iabs(west - east);
      std::vector<int64_t> accepted;
      for (const Hit& hit : found->second) {
        const int dy = hit.y > y ? hit.y - y : y - hit.y;
        const int dx = hit.x > x ? hit.x - x : x - hit.x;
        if ((dy > dx ? dy : dx) > radius) continue;
        const int64_t pred = dc + hit.high;
        const int64_t err = pred - local;
        if (s_eta2 <= 0) {
          const int64_t limit = grad >> 2;
          if (iabs(err) > (limit > 2 ? limit : 2)) continue;
        } else if (static_cast<__int128>(err) * err > static_cast<__int128>(14) * s_eta2 + grad) {
          continue;
        }
        accepted.push_back(pred);
      }
      if (accepted.empty()) continue;
      int64_t fused;
      if (s_eta2 <= 0) {
        fused = accepted.back();
      } else {
        const __int128 s_loc2 = static_cast<__int128>(grad) * grad + 4;
        __int128 sum = 0;
        for (int64_t pred : accepted) sum += pred;
        const __int128 acc = static_cast<__int128>(local) * s_eta2 + s_loc2 * sum;
        const __int128 denom =
            static_cast<__int128>(s_eta2) + static_cast<__int128>(accepted.size()) * s_loc2;
        fused = static_cast<int64_t>(floor_div128(acc * 2 + denom, denom * 2));
      }
      const int64_t delta = fused - local;
      out.at(y, x) = static_cast<int32_t>(delta);
    }
  }
  return out;
}

}  // namespace vpxc
