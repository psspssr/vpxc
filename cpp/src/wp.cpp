#include "vpxc/wp.hpp"

#include "vpxc/entropy.hpp"
#include "vpxc/error.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <vector>

namespace vpxc {
namespace {

constexpr int kExtra = 3;
constexpr int kRound = (1 << kExtra) / 2;
constexpr uint32_t kDiv[64] = {
    16777216, 8388608, 5592405, 4194304, 3355443, 2796202, 2396745, 2097152, 1864135, 1677721, 1525201,
    1398101,  1290555, 1198372, 1118481, 1048576, 986895,  932067,  883011,  838860,  798915,  762600,
    729444,   699050,  671088,  645277,  621378,  599186,  578524,  559240,  541200,  524288,  508400,
    493447,   479349,  466033,  453438,  441505,  430185,  419430,  409200,  399457,  390167,  381300,
    372827,   364722,  356962,  349525,  342392,  335544,  328965,  322638,  316551,  310689,  305040,
    299593,   294337,  289262,  284359,  279620,  275036,  270600,  266305,  262144};

struct WpHeader {
  uint32_t w[4] = {0xd, 0xc, 0xc, 0xb};
  int p1C = 8;
  int p2C = 8;
  int p3Ca = 4;
  int p3Cb = 0;
  int p3Cc = 3;
  int p3Cd = 23;
  int p3Ce = 2;
};

int floor_log2(uint64_t x) { return x ? (63 - __builtin_clzll(x)) : 0; }

uint32_t error_weight(uint64_t x, uint32_t maxweight) {
  int shift = floor_log2(x + 1) - 5;
  if (shift < 0) shift = 0;
  const uint64_t idx = x >> shift;
  if (idx >= 64) return 4;
  return 4u + static_cast<uint32_t>((static_cast<uint64_t>(maxweight) * kDiv[idx]) >> shift);
}

int64_t weighted_average(const int64_t pred[4], uint32_t w[4]) {
  uint32_t sum_w = 0;
  for (int i = 0; i < 4; ++i) sum_w += w[i];
  if (sum_w < 16) sum_w = 16;
  const uint32_t log_w = static_cast<uint32_t>(floor_log2(sum_w));
  sum_w = 0;
  for (int i = 0; i < 4; ++i) {
    if (log_w >= 4) w[i] >>= log_w - 4;
    sum_w += w[i];
  }
  if (sum_w == 0) sum_w = 1;
  if (sum_w > 64) sum_w = 64;
  int64_t sum = static_cast<int64_t>(sum_w >> 1) - 1;
  for (int i = 0; i < 4; ++i) sum += pred[i] * static_cast<int64_t>(w[i]);
  return (sum * static_cast<int64_t>(kDiv[sum_w - 1])) >> 24;
}

int sigma_from_q8_target(int64_t target) {
  int best = 0;
  int64_t best_d = std::llabs(static_cast<int64_t>(SIGMA_Q8[0]) - target);
  for (int i = 1; i < NUM_SIGMAS; ++i) {
    const int64_t gap = std::llabs(static_cast<int64_t>(SIGMA_Q8[i]) - target);
    if (gap < best_d) {
      best_d = gap;
      best = i;
    }
  }
  return best;
}

struct Walk {
  const WpHeader header{};
  int width = 0;
  int stride = 0;
  std::vector<uint32_t> perr[4];
  std::vector<int> error;

  explicit Walk(int w) : width(w), stride(w + 2) {
    for (auto& row : perr) row.assign(static_cast<size_t>(stride * 2), 0);
    error.assign(static_cast<size_t>(stride * 2), 0);
  }

  int sample_at(const Plane& plane, int y, int x) const {
    if (y < 0 || x < 0 || y >= plane.height || x >= plane.width) return 0;
    return plane.at(y, x);
  }

  void predict(const Plane& plane, int y, int x, int& guess, int& frac) {
    const int N = sample_at(plane, y - 1, x);
    const int W = sample_at(plane, y, x - 1);
    const int NE = y > 0 ? sample_at(plane, y - 1, std::min(x + 1, width - 1)) : 0;
    const int NW = (y > 0 && x > 0) ? sample_at(plane, y - 1, x - 1) : (y > 0 ? N : W);
    const int NN = y > 1 ? sample_at(plane, y - 2, x) : N;
    const int cur = (y & 1) ? 0 : stride;
    const int prev = (y & 1) ? stride : 0;
    const int pos_N = prev + x;
    const int pos_NE = x < width - 1 ? pos_N + 1 : pos_N;
    const int pos_NW = x > 0 ? pos_N - 1 : pos_N;
    uint32_t weights[4];
    for (int i = 0; i < 4; ++i) {
      const uint64_t acc = static_cast<uint64_t>(perr[i][static_cast<size_t>(pos_N)]) +
                           perr[i][static_cast<size_t>(pos_NE)] + perr[i][static_cast<size_t>(pos_NW)];
      weights[i] = error_weight(acc, header.w[i]);
    }
    const int64_t n = static_cast<int64_t>(N) << kExtra;
    const int64_t wv = static_cast<int64_t>(W) << kExtra;
    const int64_t ne = static_cast<int64_t>(NE) << kExtra;
    const int64_t nw = static_cast<int64_t>(NW) << kExtra;
    const int64_t nn = static_cast<int64_t>(NN) << kExtra;
    const int teW = x == 0 ? 0 : error[static_cast<size_t>(cur + x - 1)];
    const int teN = error[static_cast<size_t>(pos_N)];
    const int teNW = error[static_cast<size_t>(pos_NW)];
    const int teNE = error[static_cast<size_t>(pos_NE)];
    const int sumWN = teN + teW;
    int64_t prediction[4];
    prediction[0] = wv + ne - n;
    prediction[1] = n - ((static_cast<int64_t>(sumWN + teNE) * header.p1C) >> 5);
    prediction[2] = wv - ((static_cast<int64_t>(sumWN + teNW) * header.p2C) >> 5);
    prediction[3] = n - ((static_cast<int64_t>(teNW) * header.p3Ca + static_cast<int64_t>(teN) * header.p3Cb +
                          static_cast<int64_t>(teNE) * header.p3Cc + (nn - n) * header.p3Cd + (nw - wv) * header.p3Ce) >>
                         5);
    int64_t pred = weighted_average(prediction, weights);
    if (((teN ^ teW) | (teN ^ teNW)) <= 0) {
      const int64_t mx = std::max(wv, std::max(ne, n));
      const int64_t mn = std::min(wv, std::min(ne, n));
      pred = std::max(mn, std::min(mx, pred));
    }
    const int64_t pred_q4 = pred << 1;
    const int64_t gi = (pred_q4 + 8) >> 4;
    guess = static_cast<int>(gi);
    frac = std::clamp(static_cast<int>(pred_q4 - (gi << 4) + 8), 0, 15);
    const int neighbor = std::max({std::abs(teW), std::abs(teN), std::abs(teNW), std::abs(teNE)}) >> kExtra;
    last_mag = std::max(neighbor, (run_abs + 2) >> 2);
    saved_pred = pred;
    for (int i = 0; i < 4; ++i) saved_p[i] = prediction[i];
  }

  void update(int truth, int y, int x) {
    const int cur = (y & 1) ? 0 : stride;
    const int prev = (y & 1) ? stride : 0;
    const int64_t val = static_cast<int64_t>(truth) << kExtra;
    error[static_cast<size_t>(cur + x)] = static_cast<int>(saved_pred - val);
    const int pixel_err = std::abs(truth - static_cast<int>((saved_pred + kRound) >> kExtra));
    run_abs = (run_abs * 3 + pixel_err) >> 2;
    for (int i = 0; i < 4; ++i) {
      const int err = static_cast<int>((std::llabs(saved_p[i] - val) + kRound) >> kExtra);
      perr[i][static_cast<size_t>(cur + x)] = static_cast<uint32_t>(err);
      perr[i][static_cast<size_t>(prev + x + 1)] += static_cast<uint32_t>(err);
    }
  }

  int64_t saved_pred = 0;
  int64_t saved_p[4]{};
  int run_abs = 0;
  int last_mag = 0;
};

int log_bucket(int mag, int limit) {
  int bucket = 0;
  while (mag > 1 && bucket < limit) {
    mag >>= 1;
    ++bucket;
  }
  return bucket;
}

void context_pair(const Walk& walk, const Plane& plane, int y, int x, int guess, int& eb, int& gb) {
  const int W = x > 0 ? plane.at(y, x - 1) : guess;
  const int N = y > 0 ? plane.at(y - 1, x) : guess;
  eb = log_bucket(walk.last_mag, 7);
  gb = log_bucket(std::abs(W - N), 7);
}

struct Sample {
  int32_t residual = 0;
  int32_t frac = 8;
  uint8_t ctx = 0;
};

std::vector<Sample> collect(const Plane& plane) {
  Walk walk(plane.width);
  std::vector<Sample> samples(static_cast<size_t>(plane.height * plane.width));
  for (int y = 0; y < plane.height; ++y) {
    for (int x = 0; x < plane.width; ++x) {
      int guess = 0;
      int frac = 8;
      walk.predict(plane, y, x, guess, frac);
      int eb = 0;
      int gb = 0;
      context_pair(walk, plane, y, x, guess, eb, gb);
      Sample& sample = samples[static_cast<size_t>(y * plane.width + x)];
      sample.residual = plane.at(y, x) - guess;
      sample.frac = frac;
      sample.ctx = static_cast<uint8_t>((eb << 3) | gb);
      walk.update(plane.at(y, x), y, x);
    }
  }
  return samples;
}

std::array<int, 64> sigma_table(const std::vector<Sample>& samples) {
  int64_t abs_q4_sum[64]{};
  int64_t count[64]{};
  for (const Sample& sample : samples) {
    const int ctx = sample.ctx;
    const int64_t inner = static_cast<int64_t>(sample.residual) * 16 - (static_cast<int64_t>(sample.frac) - 8);
    abs_q4_sum[ctx] += inner < 0 ? -inner : inner;
    count[ctx] += 1;
  }
  std::array<int, 64> sigma{};
  int prev = 4;
  for (int ctx = 0; ctx < 64; ++ctx) {
    if (count[ctx] == 0) {
      sigma[ctx] = prev;
      continue;
    }
    const int64_t target_q8 = ((abs_q4_sum[ctx] << 4) * 236) / (count[ctx] * 256) + 20;
    sigma[ctx] = sigma_from_q8_target(target_q8);
    prev = sigma[ctx];
  }
  return sigma;
}

}  // namespace

std::vector<uint8_t> encode_wp(const std::vector<Plane>& planes) {
  if (planes.empty()) throw CodecError("empty weighted-predictor tile");
  std::vector<uint8_t> out;
  for (const Plane& plane : planes) {
    if (plane.height <= 0 || plane.width <= 0) throw CodecError("empty weighted-predictor plane");
    const std::vector<Sample> samples = collect(plane);
    const std::array<int, 64> table = sigma_table(samples);
    std::vector<int32_t> residual(samples.size());
    std::vector<int32_t> sigma(samples.size());
    std::vector<int32_t> frac(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
      residual[i] = samples[i].residual;
      frac[i] = samples[i].frac;
      sigma[i] = table[samples[i].ctx];
    }
    int best_nu = 0;
    std::vector<uint8_t> best;
    for (int nu = 0; nu < NUM_NUS; ++nu) {
      auto blob = encode_parametric_rans(residual, nu, sigma, frac);
      if (best.empty() || blob.size() < best.size()) {
        best_nu = nu;
        best = std::move(blob);
      }
    }
    out.push_back(static_cast<uint8_t>(best_nu));
    for (int s : table) out.push_back(static_cast<uint8_t>(s));
    if (best.size() >= (1ull << 32)) throw CodecError("weighted-predictor plane exceeds 4 GiB");
    out.push_back(static_cast<uint8_t>(best.size()));
    out.push_back(static_cast<uint8_t>(best.size() >> 8));
    out.push_back(static_cast<uint8_t>(best.size() >> 16));
    out.push_back(static_cast<uint8_t>(best.size() >> 24));
    out.insert(out.end(), best.begin(), best.end());
  }
  return out;
}

std::vector<Plane> decode_wp(const uint8_t* data, size_t size, int height, int width, int channels) {
  if (data == nullptr || size < static_cast<size_t>(channels)) throw CodecError("truncated weighted-predictor tile");
  size_t pos = 0;
  std::vector<Plane> planes;
  planes.reserve(static_cast<size_t>(channels));
  for (int c = 0; c < channels; ++c) {
    if (pos >= size) throw CodecError("truncated weighted-predictor channel");
    const int nu = data[pos++];
    if (nu < 0 || nu >= NUM_NUS) throw CodecError("weighted-predictor nu out of range");
    if (pos + 64 + 4 > size) throw CodecError("truncated weighted-predictor context table");
    int table[64];
    for (int i = 0; i < 64; ++i) table[i] = data[pos++];
    const uint32_t len = static_cast<uint32_t>(data[pos]) | (static_cast<uint32_t>(data[pos + 1]) << 8) |
                         (static_cast<uint32_t>(data[pos + 2]) << 16) | (static_cast<uint32_t>(data[pos + 3]) << 24);
    pos += 4;
    if (pos + len > size) throw CodecError("truncated weighted-predictor payload");
    Plane plane(height, width);
    Walk walk(width);
    ParametricDecoder decoder(std::span<const uint8_t>(data + pos, len), nu);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int guess = 0;
        int frac = 8;
        walk.predict(plane, y, x, guess, frac);
        int eb = 0;
        int gb = 0;
        context_pair(walk, plane, y, x, guess, eb, gb);
        const int residual = decoder.read(table[(eb << 3) | gb], frac);
        const int64_t truth = static_cast<int64_t>(guess) + residual;
        if (truth < -2147483647 || truth > 2147483647) throw CodecError("weighted predictor overflow");
        plane.at(y, x) = static_cast<int32_t>(truth);
        walk.update(plane.at(y, x), y, x);
      }
    }
    pos += len;
    planes.push_back(std::move(plane));
  }
  if (pos != size) throw CodecError("trailing bytes in weighted-predictor tile");
  return planes;
}

}  // namespace vpxc
