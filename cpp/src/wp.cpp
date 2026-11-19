#include "vpxc/wp.hpp"

#include "vpxc/entropy.hpp"
#include "vpxc/error.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <vector>

namespace vpxc {
namespace {

constexpr int kExtra = 3;
constexpr int kRound = (1 << kExtra) / 2 - 1;
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

int floor_log2(uint64_t x) { return 63 - __builtin_clzll(x); }

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
  const uint32_t log_w = floor_log2(sum_w);
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

int sigma_scale() {
  static const int scale = [] {
    const char* env = std::getenv("VPXC_SIGMA_SCALE");
    return env != nullptr && env[0] != '\0' ? std::atoi(env) : 160;
  }();
  return scale;
}

int sigma_of_mag(int mag) {
  if (mag < 0) mag = 0;
  const int target = mag == 0 ? SIGMA_Q8[0] : mag * sigma_scale();
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

  // Returns the integer prediction and the sigma index. Does not update state.
  void predict(const Plane& plane, int y, int x, int& guess, int& sigma) {
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
      const uint64_t acc = static_cast<uint64_t>(perr[i][static_cast<size_t>(pos_N)]) + perr[i][static_cast<size_t>(pos_NE)] +
                           perr[i][static_cast<size_t>(pos_NW)];
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
    guess = static_cast<int>((pred + kRound) >> kExtra);
    const int neighbor = std::max({std::abs(teW), std::abs(teN), std::abs(teNW), std::abs(teNE)}) >> kExtra;
    // Running absolute error tracks the local scale better than a single neighbor max.
    const int mag = std::max(neighbor, (run_abs + 2) >> 2);
    last_mag = mag;
    sigma = sigma_of_mag(mag);
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

struct Coded {
  int nu = 1;
  std::vector<int32_t> residual;
  std::vector<int32_t> sigma;
};

int log_bucket(int mag, int limit) {
  int bucket = 0;
  while (mag > 1 && bucket < limit) {
    mag >>= 1;
    ++bucket;
  }
  return bucket;
}

struct Adapt {
  int count[42]{};
  int total = 0;
  double emit(int value) {
    int mag = value < 0 ? -value : value;
    int sym = 0;
    int extra = 0;
    if (mag <= 20) sym = value < 0 ? mag * 2 - 1 : mag * 2;
    else {
      sym = 41;
      int bits = 0;
      int m = mag;
      while (m > 1) {
        m >>= 1;
        ++bits;
      }
      extra = 1 + bits;
    }
    const double p = static_cast<double>(count[sym] + 1) / static_cast<double>(total + 42);
    count[sym] += 1;
    total += 1;
    return -std::log2(p) + extra;
  }
};

Coded analyze(const Plane& plane, double* adapt_bits) {
  Walk walk(plane.width);
  Coded out;
  const int n = plane.height * plane.width;
  out.residual.resize(static_cast<size_t>(n));
  out.sigma.resize(static_cast<size_t>(n));
  Adapt fine[8][8];
  double bits = 0;
  for (int y = 0; y < plane.height; ++y) {
    for (int x = 0; x < plane.width; ++x) {
      int guess = 0;
      int sigma = 0;
      walk.predict(plane, y, x, guess, sigma);
      const int truth = plane.at(y, x);
      const int err = truth - guess;
      out.residual[static_cast<size_t>(y * plane.width + x)] = err;
      out.sigma[static_cast<size_t>(y * plane.width + x)] = sigma;
      if (adapt_bits) {
        const int W = x ? plane.at(y, x - 1) : guess;
        const int N = y ? plane.at(y - 1, x) : guess;
        const int eb = log_bucket(walk.last_mag, 7);
        const int gb = log_bucket(std::abs(W - N), 7);
        bits += fine[eb][gb].emit(err);
      }
      walk.update(truth, y, x);
    }
  }
  if (adapt_bits) *adapt_bits += bits;
  return out;
}

struct Freq {
  int count[42]{};
  int total = 0;
  void rescale() {
    if (total < 8192) return;
    total = 0;
    for (int& c : count) {
      c = (c + 1) >> 1;
      total += c;
    }
  }
  void model(int sym, uint32_t& cum, uint32_t& freq, uint32_t& tot) const {
    cum = 0;
    for (int i = 0; i < sym; ++i) cum += static_cast<uint32_t>(count[i] + 1);
    freq = static_cast<uint32_t>(count[sym] + 1);
    tot = static_cast<uint32_t>(total + 42);
  }
  void seen(int sym) {
    count[sym] += 1;
    total += 1;
    rescale();
  }
};

struct ArithEnc {
  uint32_t low = 0;
  uint32_t high = 0xffffffffu;
  std::vector<uint8_t> out;
  void encode(uint32_t cum, uint32_t freq, uint32_t tot) {
    const uint64_t span = static_cast<uint64_t>(high) - low + 1;
    const uint64_t step = span / tot;
    if (step == 0) throw CodecError("arithmetic range collapsed");
    const uint32_t new_low = static_cast<uint32_t>(low + step * cum);
    const uint32_t new_high = static_cast<uint32_t>(low + step * (cum + freq) - 1);
    low = new_low;
    high = new_high;
    renorm();
  }
  void renorm() {
    while (((low ^ high) & 0xff000000u) == 0) {
      out.push_back(static_cast<uint8_t>(low >> 24));
      low <<= 8;
      high = (high << 8) | 0xffu;
    }
  }
  std::vector<uint8_t> finish() {
    for (int i = 0; i < 4; ++i) {
      out.push_back(static_cast<uint8_t>(low >> 24));
      low <<= 8;
    }
    return out;
  }
};

struct ArithDec {
  uint32_t low = 0;
  uint32_t high = 0xffffffffu;
  uint32_t code = 0;
  const uint8_t* data = nullptr;
  size_t size = 0;
  size_t pos = 0;
  explicit ArithDec(const uint8_t* bytes, size_t n) : data(bytes), size(n) {
    for (int i = 0; i < 4; ++i) code = (code << 8) | next();
  }
  uint8_t next() { return pos < size ? data[pos++] : 0; }
  uint32_t target(uint32_t tot) const {
    const uint64_t span = static_cast<uint64_t>(high) - low + 1;
    const uint64_t step = span / tot;
    if (step == 0) throw CodecError("arithmetic range collapsed");
    return static_cast<uint32_t>((static_cast<uint64_t>(code - low)) / step);
  }
  void update(uint32_t cum, uint32_t freq, uint32_t tot) {
    const uint64_t span = static_cast<uint64_t>(high) - low + 1;
    const uint64_t step = span / tot;
    const uint32_t base = low;
    low = static_cast<uint32_t>(base + step * cum);
    high = static_cast<uint32_t>(base + step * (static_cast<uint64_t>(cum) + freq) - 1);
    while (((low ^ high) & 0xff000000u) == 0) {
      low <<= 8;
      high = (high << 8) | 0xffu;
      code = (code << 8) | next();
    }
  }
};

int floor_log2_int(int mag) { return 31 - __builtin_clz(static_cast<unsigned>(mag)); }

void write_escape(ArithEnc& enc, int value) {
  const int mag = value < 0 ? -value : value;
  const int bits = floor_log2_int(mag);
  const int rest = mag - (1 << bits);
  enc.encode(value < 0 ? 1u : 0u, 1u, 2u);
  for (int i = bits - 1; i >= 0; --i) enc.encode((rest >> i) & 1, 1u, 2u);
}

int read_escape(ArithDec& dec, int bits_hint_unused) {
  (void)bits_hint_unused;
  const uint32_t sign = dec.target(2);
  dec.update(sign, 1u, 2u);
  // The bit length is known from the symbol path: caller passes it.
  return static_cast<int>(sign);
}

struct Esc {
  int sign = 0;
  int rest = 0;
  int bits = 0;
};

Esc split_escape(int value) {
  const int mag = value < 0 ? -value : value;
  Esc e;
  e.sign = value < 0 ? 1 : 0;
  e.bits = floor_log2_int(mag);
  e.rest = mag - (1 << e.bits);
  return e;
}

void put_symbol(ArithEnc& enc, Freq& freq, int value) {
  int sym = 0;
  Esc esc;
  const int mag = value < 0 ? -value : value;
  if (mag <= 20) sym = value < 0 ? mag * 2 - 1 : mag * 2;
  else {
    sym = 41;
    esc = split_escape(value);
  }
  uint32_t cum = 0, fr = 0, tot = 0;
  freq.model(sym, cum, fr, tot);
  enc.encode(cum, fr, tot);
  freq.seen(sym);
  if (sym == 41) {
    enc.encode(static_cast<uint32_t>(esc.sign), 1u, 2u);
    for (int i = 4; i >= 0; --i) enc.encode(static_cast<uint32_t>((esc.bits >> i) & 1), 1u, 2u);
    for (int i = esc.bits - 1; i >= 0; --i) enc.encode(static_cast<uint32_t>((esc.rest >> i) & 1), 1u, 2u);
  }
}

int get_symbol(ArithDec& dec, Freq& freq) {
  uint32_t cum0 = 0, fr0 = 0, tot = 0;
  freq.model(0, cum0, fr0, tot);
  const uint32_t tgt = dec.target(tot);
  int sym = 41;
  uint32_t cum = 0;
  for (int i = 0; i < 42; ++i) {
    const uint32_t fr = static_cast<uint32_t>(freq.count[i] + 1);
    if (tgt < cum + fr) {
      sym = i;
      dec.update(cum, fr, tot);
      break;
    }
    cum += fr;
  }
  freq.seen(sym);
  if (sym < 41) return (sym & 1) ? -((sym + 1) >> 1) : (sym >> 1);
  const uint32_t sign = dec.target(2);
  dec.update(sign, 1u, 2u);
  // Bit length is not stored; the symbol only says "escape". Recover mag by reading
  // a unary-free form: the probe's rate assumed the decoder knows the bit length.
  // Store the bit length in 5 unary-free bits, enough for |residual| up to 2^31.
  int bits = 0;
  for (int i = 4; i >= 0; --i) {
    const uint32_t bit = dec.target(2);
    dec.update(bit, 1u, 2u);
    bits = (bits << 1) | static_cast<int>(bit);
  }
  int rest = 0;
  for (int i = bits - 1; i >= 0; --i) {
    const uint32_t bit = dec.target(2);
    dec.update(bit, 1u, 2u);
    rest = (rest << 1) | static_cast<int>(bit);
  }
  const int mag = (1 << bits) + rest;
  return sign ? -mag : mag;
}

}  // namespace

int context_pair(const Walk& walk, const Plane& plane, int y, int x, int guess, int& eb, int& gb) {
  const int W = x > 0 ? plane.at(y, x - 1) : guess;
  const int N = y > 0 ? plane.at(y - 1, x) : guess;
  eb = log_bucket(walk.last_mag, 7);
  gb = log_bucket(std::abs(W - N), 7);
  return 0;
}

struct Sample {
  int32_t residual = 0;
  uint8_t ctx = 0;
};

std::vector<Sample> collect(const Plane& plane) {
  Walk walk(plane.width);
  std::vector<Sample> samples(static_cast<size_t>(plane.height * plane.width));
  for (int y = 0; y < plane.height; ++y) {
    for (int x = 0; x < plane.width; ++x) {
      int guess = 0;
      int sigma = 0;
      walk.predict(plane, y, x, guess, sigma);
      int eb = 0;
      int gb = 0;
      context_pair(walk, plane, y, x, guess, eb, gb);
      Sample& sample = samples[static_cast<size_t>(y * plane.width + x)];
      sample.residual = plane.at(y, x) - guess;
      sample.ctx = static_cast<uint8_t>((eb << 3) | gb);
      walk.update(plane.at(y, x), y, x);
      (void)sigma;
    }
  }
  return samples;
}

std::array<int, 64> sigma_table(const std::vector<Sample>& samples) {
  int64_t abs_sum[64]{};
  int count[64]{};
  for (const Sample& sample : samples) {
    const int ctx = sample.ctx;
    const int32_t r = sample.residual;
    abs_sum[ctx] += r < 0 ? -static_cast<int64_t>(r) : r;
    count[ctx] += 1;
  }
  std::array<int, 64> sigma{};
  for (int ctx = 0; ctx < 64; ++ctx) {
    const int mag = count[ctx] ? static_cast<int>(abs_sum[ctx] / count[ctx]) : 0;
    sigma[ctx] = sigma_of_mag(mag);
  }
  return sigma;
}

std::vector<uint8_t> encode_wp(const std::vector<Plane>& planes) {
  if (planes.empty()) throw CodecError("empty weighted-predictor tile");
  std::vector<uint8_t> out;
  for (const Plane& plane : planes) {
    if (plane.height <= 0 || plane.width <= 0) throw CodecError("empty weighted-predictor plane");
    const std::vector<Sample> samples = collect(plane);
    const std::array<int, 64> table = sigma_table(samples);
    std::vector<int32_t> residual(samples.size());
    std::vector<int32_t> sigma(samples.size());
    std::vector<int32_t> frac(samples.size(), 8);
    for (size_t i = 0; i < samples.size(); ++i) {
      residual[i] = samples[i].residual;
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
        int sig = 0;
        walk.predict(plane, y, x, guess, sig);
        int eb = 0;
        int gb = 0;
        context_pair(walk, plane, y, x, guess, eb, gb);
        const int residual = decoder.read(table[(eb << 3) | gb], 8);
        const int64_t truth = static_cast<int64_t>(guess) + residual;
        if (truth < -2147483647 || truth > 2147483647) throw CodecError("weighted predictor overflow");
        plane.at(y, x) = static_cast<int32_t>(truth);
        walk.update(plane.at(y, x), y, x);
        (void)sig;
      }
    }
    pos += len;
    planes.push_back(std::move(plane));
  }
  if (pos != size) throw CodecError("trailing bytes in weighted-predictor tile");
  return planes;
}

}  // namespace vpxc
