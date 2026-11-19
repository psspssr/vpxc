#include "vpxc/entropy.hpp"

#include "vpxc/error.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>

namespace vpxc {
namespace {

void append_u32_le(std::vector<uint8_t>& out, uint32_t word) {
  out.push_back(static_cast<uint8_t>(word));
  out.push_back(static_cast<uint8_t>(word >> 8));
  out.push_back(static_cast<uint8_t>(word >> 16));
  out.push_back(static_cast<uint8_t>(word >> 24));
}

void append_u16_le(std::vector<uint8_t>& out, uint16_t word) {
  out.push_back(static_cast<uint8_t>(word));
  out.push_back(static_cast<uint8_t>(word >> 8));
}

uint32_t read_u32_le(const uint8_t* p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

uint16_t read_u16_le(const uint8_t* p) { return uint16_t(p[0]) | (uint16_t(p[1]) << 8); }

// Floor division for a positive divisor. Matches Python //.
int64_t floor_div_pos(int64_t n, int64_t d) {
  int64_t q = n / d;
  if (n % d < 0) --q;
  return q;
}

// Exact v * 2^shift for results that fit in int64. shift == 0 returns v.
int64_t mul_pow2(int64_t v, int shift) {
  if (shift <= 0) return v;
  if (shift >= 63) throw CodecError("dyadic shift out of range");
  if (v >= 0) return v << shift;
  if (v == INT64_MIN) throw CodecError("dyadic shift out of range");
  return -((-v) << shift);
}

int clamp_int(int v, int lo, int hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

constexpr int32_t kCdfNu25[257] = {
    8388608, 9144624, 9884460, 10593521, 11260046, 11875794, 12436143, 12939713, 13387706, 13783155,
    14130222, 14433625, 14698213, 14928680, 15129393, 15304303, 15456920, 15590317, 15707156, 15809735,
    15900020, 15979694, 16050195, 16112749, 16168404, 16218054, 16262466, 16302297, 16338110, 16370390,
    16399557, 16425973, 16449951, 16471765, 16491651, 16509818, 16526446, 16541696, 16555707, 16568603,
    16580493, 16591474, 16601631, 16611041, 16619771, 16627883, 16635431, 16642463, 16649024, 16655151,
    16660882, 16666248, 16671278, 16675999, 16680433, 16684603, 16688529, 16692227, 16695716, 16699009,
    16702121, 16705063, 16707848, 16710486, 16712987, 16715360, 16717613, 16719753, 16721788, 16723725,
    16725568, 16727325, 16728999, 16730596, 16732121, 16733577, 16734969, 16736299, 16737572, 16738791,
    16739958, 16741076, 16742148, 16743177, 16744163, 16745111, 16746021, 16746895, 16747736, 16748544,
    16749322, 16750071, 16750792, 16751487, 16752156, 16752802, 16753424, 16754025, 16754604, 16755164,
    16755704, 16756227, 16756731, 16757219, 16757691, 16758147, 16758589, 16759017, 16759431, 16759832,
    16760220, 16760596, 16760961, 16761315, 16761658, 16761991, 16762314, 16762628, 16762932, 16763228,
    16763515, 16763794, 16764065, 16764328, 16764585, 16764834, 16765076, 16765312, 16765541, 16765764,
    16765982, 16766193, 16766400, 16766600, 16766796, 16766987, 16767172, 16767353, 16767530, 16767702,
    16767870, 16768034, 16768194, 16768350, 16768502, 16768650, 16768795, 16768937, 16769076, 16769211,
    16769343, 16769472, 16769598, 16769721, 16769841, 16769959, 16770074, 16770187, 16770297, 16770405,
    16770510, 16770613, 16770714, 16770813, 16770910, 16771005, 16771097, 16771188, 16771277, 16771364,
    16771449, 16771533, 16771615, 16771695, 16771774, 16771851, 16771926, 16772001, 16772073, 16772144,
    16772214, 16772283, 16772350, 16772416, 16772481, 16772544, 16772607, 16772668, 16772728, 16772787,
    16772845, 16772901, 16772957, 16773012, 16773066, 16773118, 16773170, 16773221, 16773271, 16773321,
    16773369, 16773417, 16773463, 16773509, 16773554, 16773599, 16773642, 16773685, 16773727, 16773769,
    16773810, 16773850, 16773889, 16773928, 16773966, 16774004, 16774041, 16774077, 16774113, 16774148,
    16774183, 16774217, 16774250, 16774283, 16774316, 16774348, 16774379, 16774411, 16774441, 16774471,
    16774501, 16774530, 16774559, 16774587, 16774615, 16774643, 16774670, 16774697, 16774723, 16774749,
    16774774, 16774800, 16774824, 16774849, 16774873, 16774897, 16774920, 16774943, 16774966, 16774989,
    16775011, 16775033, 16775054, 16775075, 16775096, 16775117, 16775138,
};

constexpr int32_t kCdfNu35[257] = {
    8388608, 9165672, 9927408, 10659768, 11351041, 11992522, 12578734, 13107268, 13578327, 13994137,
    14358320, 14675337, 14950025, 15187253, 15391692, 15567668, 15719089, 15849424, 15961705, 16058559,
    16142243, 16214687, 16277534, 16332178, 16379804, 16421414, 16457859, 16489861, 16518031, 16542891,
    16564883, 16584385, 16601721, 16617168, 16630962, 16643308, 16654383, 16664337, 16673304, 16681396,
    16688715, 16695345, 16701363, 16706836, 16711820, 16716368, 16720524, 16724328, 16727816, 16731018,
    16733962, 16736672, 16739171, 16741478, 16743610, 16745584, 16747413, 16749109, 16750685, 16752150,
    16753514, 16754785, 16755971, 16757078, 16758112, 16759080, 16759986, 16760836, 16761632, 16762380,
    16763083, 16763744, 16764366, 16764952, 16765504, 16766025, 16766516, 16766981, 16767420, 16767835,
    16768228, 16768600, 16768953, 16769287, 16769604, 16769906, 16770192, 16770464, 16770723, 16770969,
    16771203, 16771426, 16771639, 16771842, 16772035, 16772220, 16772396, 16772565, 16772726, 16772880,
    16773027, 16773168, 16773303, 16773432, 16773556, 16773674, 16773788, 16773897, 16774002, 16774102,
    16774199, 16774291, 16774381, 16774466, 16774549, 16774628, 16774704, 16774777, 16774848, 16774916,
    16774982, 16775045, 16775106, 16775164, 16775221, 16775276, 16775329, 16775379, 16775429, 16775476,
    16775522, 16775566, 16775609, 16775651, 16775691, 16775730, 16775767, 16775803, 16775839, 16775873,
    16775906, 16775938, 16775969, 16775999, 16776028, 16776056, 16776083, 16776110, 16776135, 16776160,
    16776185, 16776208, 16776231, 16776253, 16776275, 16776296, 16776316, 16776336, 16776355, 16776374,
    16776392, 16776409, 16776427, 16776443, 16776460, 16776475, 16776491, 16776506, 16776520, 16776535,
    16776548, 16776562, 16776575, 16776588, 16776600, 16776612, 16776624, 16776636, 16776647, 16776658,
    16776669, 16776679, 16776689, 16776699, 16776709, 16776719, 16776728, 16776737, 16776746, 16776754,
    16776763, 16776771, 16776779, 16776787, 16776794, 16776802, 16776809, 16776816, 16776823, 16776830,
    16776837, 16776843, 16776850, 16776856, 16776862, 16776868, 16776874, 16776880, 16776885, 16776891,
    16776896, 16776901, 16776907, 16776912, 16776916, 16776921, 16776926, 16776931, 16776935, 16776940,
    16776944, 16776948, 16776952, 16776957, 16776961, 16776965, 16776968, 16776972, 16776976, 16776980,
    16776983, 16776987, 16776990, 16776993, 16776997, 16777000, 16777003, 16777006, 16777009, 16777012,
    16777015, 16777018, 16777021, 16777024, 16777026, 16777029, 16777032, 16777034, 16777037, 16777039,
    16777042, 16777044, 16777047, 16777049, 16777051, 16777053, 16777056,
};

constexpr int32_t kCdfNu60[257] = {
    8388608, 9188827, 9974667, 10732733, 11451457, 12121708, 12737113, 13294077, 13791562, 14230692,
    14614266, 14946259, 15231357, 15474566, 15680904, 15855188, 16001888, 16125048, 16228255, 16314642,
    16386909, 16447359, 16497945, 16540307, 16575821, 16605633, 16630698, 16651808, 16669621, 16684682,
    16697442, 16708277, 16717498, 16725362, 16732084, 16737844, 16742790, 16747047, 16750719, 16753893,
    16756644, 16759033, 16761112, 16762924, 16764508, 16765895, 16767113, 16768182, 16769125, 16769957,
    16770692, 16771343, 16771921, 16772434, 16772892, 16773299, 16773664, 16773989, 16774281, 16774543,
    16774779, 16774990, 16775181, 16775354, 16775509, 16775650, 16775778, 16775893, 16775999, 16776094,
    16776181, 16776261, 16776333, 16776399, 16776459, 16776515, 16776566, 16776612, 16776655, 16776694,
    16776730, 16776763, 16776794, 16776822, 16776848, 16776872, 16776895, 16776915, 16776935, 16776952,
    16776969, 16776984, 16776998, 16777011, 16777024, 16777035, 16777046, 16777056, 16777065, 16777073,
    16777082, 16777089, 16777096, 16777103, 16777109, 16777115, 16777120, 16777125, 16777130, 16777135,
    16777139, 16777143, 16777147, 16777150, 16777153, 16777156, 16777159, 16777162, 16777165, 16777167,
    16777170, 16777172, 16777174, 16777176, 16777178, 16777179, 16777181, 16777183, 16777184, 16777186,
    16777187, 16777188, 16777190, 16777191, 16777192, 16777193, 16777194, 16777195, 16777196, 16777196,
    16777197, 16777198, 16777199, 16777199, 16777200, 16777201, 16777201, 16777202, 16777203, 16777203,
    16777204, 16777204, 16777204, 16777205, 16777205, 16777206, 16777206, 16777206, 16777207, 16777207,
    16777207, 16777208, 16777208, 16777208, 16777209, 16777209, 16777209, 16777209, 16777210, 16777210,
    16777210, 16777210, 16777210, 16777211, 16777211, 16777211, 16777211, 16777211, 16777211, 16777212,
    16777212, 16777212, 16777212, 16777212, 16777212, 16777212, 16777213, 16777213, 16777213, 16777213,
    16777213, 16777213, 16777213, 16777213, 16777213, 16777213, 16777213, 16777214, 16777214, 16777214,
    16777214, 16777214, 16777214, 16777214, 16777214, 16777214, 16777214, 16777214, 16777214, 16777214,
    16777214, 16777214, 16777214, 16777214, 16777214, 16777215, 16777215, 16777215, 16777215, 16777215,
    16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215,
    16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215,
    16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215,
    16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215,
};

constexpr int32_t kCdfNuInf[257] = {
    8388608, 9223077, 10044625, 10840930, 11600815, 12314719, 12975040, 13576342, 14115423, 14591237,
    15004707, 15358439, 15656377, 15903437, 16105135, 16267250, 16395532, 16495471, 16572123, 16630005,
    16673035, 16704530, 16727224, 16743324, 16754568, 16762301, 16767535, 16771024, 16773313, 16774792,
    16775733, 16776322, 16776685, 16776905, 16777037, 16777114, 16777159, 16777185, 16777199, 16777207,
    16777211, 16777214, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215,
    16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215,
    16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215,
    16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215,
    16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215,
    16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215,
    16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215,
    16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215,
    16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215,
    16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215,
    16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215,
    16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215,
    16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215,
    16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215,
    16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215,
    16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215,
    16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215,
    16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215,
    16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215,
    16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215,
    16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215,
    16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215,
};

struct CdfModel {
  const int32_t* y[NUM_NUS];
  int64_t slope[NUM_NUS][257];
};

void fill_slopes(const int32_t* y, int64_t* slope) {
  int64_t delta[256];
  for (int i = 0; i < 256; ++i) delta[i] = int64_t(y[i + 1]) - int64_t(y[i]);
  for (int i = 0; i < 257; ++i) slope[i] = 0;
  // Interior knots 1..255. Both adjacent secants must be strictly positive.
  for (int i = 0; i < 255; ++i) {
    int64_t d0 = delta[i];
    int64_t d1 = delta[i + 1];
    if (d0 > 0 && d1 > 0) {
      slope[i + 1] = (2 * d0 * d1 + ((d0 + d1) >> 1)) / (d0 + d1);
    }
  }
  slope[0] = delta[0];
  slope[256] = delta[255];
}

const CdfModel& cdf_model() {
  static const CdfModel model = [] {
    CdfModel m{};
    m.y[0] = kCdfNu25;
    m.y[1] = kCdfNu35;
    m.y[2] = kCdfNu60;
    m.y[3] = kCdfNuInf;
    for (int nu = 0; nu < NUM_NUS; ++nu) fill_slopes(m.y[nu], m.slope[nu]);
    return m;
  }();
  return model;
}

int32_t eval_cdf(int64_t z, int nu) {
  if (nu < 0 || nu >= NUM_NUS) throw CodecError("nu index out of range");
  const CdfModel& model = cdf_model();
  const int32_t* y = model.y[nu];
  const int64_t* slope = model.slope[nu];
  const bool neg = z < 0;
  int64_t az;
  if (z == INT64_MIN) {
    az = INT64_MAX;
  } else {
    az = neg ? -z : z;
  }
  int64_t val;
  if (az >= (int64_t(256) << 9)) {
    val = y[256];
  } else {
    const int idx = int(az >> 9);
    const int t = int(az & 511);
    const int64_t y0 = y[idx];
    const int64_t y1 = y[idx + 1];
    const int64_t m0 = slope[idx];
    const int64_t m1 = slope[idx + 1];
    const int64_t dy = y1 - y0;
    const int64_t c2 = 3 * dy - 2 * m0 - m1;
    const int64_t c3 = m0 + m1 - 2 * dy;
    const int64_t p1 = ((c3 * t + 256) >> 9) + c2;
    const int64_t p2 = ((p1 * t + 256) >> 9) + m0;
    val = y0 + ((p2 * t + 256) >> 9);
  }
  const int64_t lo = int64_t(1) << 23;
  const int64_t hi = (int64_t(1) << 24) - 1;
  if (val < lo) val = lo;
  if (val > hi) val = hi;
  if (neg) val = (int64_t(1) << 24) - val;
  return int32_t(val);
}

// laplace: mass = max(count, 0) + 1 and the result must partition `total`.
// otherwise: mass = max(count, 1), matching _normalize_masses_exact.
std::vector<int32_t> normalize_impl(const int64_t* counts, int nsym, int total, bool laplace, bool check) {
  if (nsym <= 0 || total <= 0) throw CodecError("frequency normalization failed");
  std::vector<int64_t> base(static_cast<size_t>(nsym));
  int64_t mass = 0;
  for (int i = 0; i < nsym; ++i) {
    int64_t c = counts[i];
    if (laplace) {
      if (c < 0) c = 0;
      if (c == INT64_MAX) throw CodecError("frequency normalization failed");
      ++c;
    } else if (c < 1) {
      c = 1;
    }
    base[static_cast<size_t>(i)] = c;
    if (mass > INT64_MAX - c) throw CodecError("frequency normalization failed");
    mass += c;
  }
  std::vector<int64_t> freqs(static_cast<size_t>(nsym));
  std::vector<int64_t> rem(static_cast<size_t>(nsym));
  int64_t sum = 0;
  for (int i = 0; i < nsym; ++i) {
    const __int128 numer = static_cast<__int128>(base[static_cast<size_t>(i)]) * total;
    __int128 q = numer / mass;
    if (q < 1) q = 1;
    if (q > INT64_MAX) throw CodecError("frequency normalization failed");
    const int64_t freq = static_cast<int64_t>(q);
    freqs[static_cast<size_t>(i)] = freq;
    const __int128 r = numer - static_cast<__int128>(freq) * mass;
    if (r > INT64_MAX || r < INT64_MIN) throw CodecError("frequency normalization failed");
    rem[static_cast<size_t>(i)] = static_cast<int64_t>(r);
    sum += freq;
  }
  const int64_t deficit = int64_t(total) - sum;
  std::vector<int> order(static_cast<size_t>(nsym));
  std::iota(order.begin(), order.end(), 0);
  if (deficit > 0) {
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      return rem[static_cast<size_t>(a)] > rem[static_cast<size_t>(b)];
    });
    for (int64_t i = 0; i < deficit; ++i) {
      freqs[static_cast<size_t>(order[static_cast<size_t>(i % nsym)])] += 1;
    }
  } else if (deficit < 0) {
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      return rem[static_cast<size_t>(a)] < rem[static_cast<size_t>(b)];
    });
    int64_t left = -deficit;
    while (left > 0) {
      bool progressed = false;
      for (int raw : order) {
        int64_t& freq = freqs[static_cast<size_t>(raw)];
        if (freq > 1) {
          --freq;
          --left;
          progressed = true;
          if (left == 0) break;
        }
      }
      if (!progressed) throw CodecError("frequency normalization failed");
    }
  }
  std::vector<int32_t> out(static_cast<size_t>(nsym));
  int64_t check_sum = 0;
  for (int i = 0; i < nsym; ++i) {
    const int64_t freq = freqs[static_cast<size_t>(i)];
    if (freq < 1 || freq > INT32_MAX) {
      throw CodecError(check ? "frequency table is not a full partition" : "frequency normalization failed");
    }
    check_sum += freq;
    out[static_cast<size_t>(i)] = static_cast<int32_t>(freq);
  }
  if (check && check_sum != total) throw CodecError("frequency table is not a full partition");
  return out;
}

int bisect_right(const int32_t* values, int n, int target) {
  int lo = 0;
  int hi = n;
  while (lo < hi) {
    const int mid = lo + ((hi - lo) >> 1);
    if (values[mid] <= target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

void rans_push(uint64_t& state, std::vector<uint32_t>& words, int start, int freq) {
  if (freq <= 0) throw CodecError("rANS frequency must be positive");
  uint64_t x = state;
  const uint64_t base = (RANS64_L >> RANS_SCALE) << 32;
  const unsigned __int128 x_max = static_cast<unsigned __int128>(base) * static_cast<unsigned __int128>(freq);
  while (static_cast<unsigned __int128>(x) >= x_max) {
    words.push_back(static_cast<uint32_t>(x));
    x >>= 32;
  }
  const uint64_t f = static_cast<uint64_t>(freq);
  const uint64_t q = x / f;
  const uint64_t r = x % f;
  state = (q << RANS_SCALE) + static_cast<uint64_t>(start) + r;
}

uint64_t rans_advance_state(uint64_t x, int start, int freq, const uint8_t* data, size_t size, size_t& pos) {
  const uint64_t cf = x & static_cast<uint64_t>(RANS_M - 1);
  x = static_cast<uint64_t>(freq) * (x >> RANS_SCALE) + cf - static_cast<uint64_t>(start);
  while (x < RANS64_L) {
    if (pos + 4 > size) throw CodecError("truncated rANS stream");
    const uint32_t word = read_u32_le(data + pos);
    pos += 4;
    x = (x << 32) | word;
  }
  return x;
}

void write_wide(BitWriter& writer, uint64_t value, int width) {
  if (width < 0) throw CodecError("negative bit width");
  if (width == 0) return;
  if (width <= 32) {
    writer.write_bits(static_cast<uint32_t>(value), width);
    return;
  }
  int left = width;
  while (left > 0) {
    const int chunk = left > 16 ? 16 : left;
    const int shift = left - chunk;
    const uint32_t mask = chunk == 16 ? 0xffffu : ((1u << chunk) - 1u);
    writer.write_bits(static_cast<uint32_t>((value >> shift) & mask), chunk);
    left -= chunk;
  }
}

uint64_t read_wide(BitReader& reader, int width) {
  if (width < 0) throw CodecError("negative bit width");
  if (width == 0) return 0;
  if (width <= 32) return reader.read_bits(width);
  uint64_t value = 0;
  int left = width;
  while (left > 0) {
    const int chunk = left > 16 ? 16 : left;
    value = (value << chunk) | reader.read_bits(chunk);
    left -= chunk;
  }
  return value;
}

void write_escape(BitWriter& writer, int64_t quotient) {
  writer.write_bits(quotient < 0 ? 1u : 0u, 1);
  const int64_t mag = quotient < 0 ? -quotient : quotient;
  const int64_t code = mag - (QMAX + 1);
  // write_exp_golomb adds one in uint32. UINT32_MAX would wrap that sum.
  if (code < 0 || code >= int64_t(UINT32_MAX)) throw CodecError("exp-golomb value must be non-negative");
  writer.write_exp_golomb(static_cast<uint32_t>(code));
}

int64_t read_escape(BitReader& reader) {
  const uint32_t sign = reader.read_bits(1);
  const uint64_t mag = static_cast<uint64_t>(reader.read_exp_golomb()) + static_cast<uint64_t>(QMAX + 1);
  return sign ? -static_cast<int64_t>(mag) : static_cast<int64_t>(mag);
}

void require_plane(const Plane& plane) {
  if (plane.height < 0 || plane.width < 0 ||
      plane.samples.size() < static_cast<size_t>(plane.height) * static_cast<size_t>(plane.width)) {
    throw CodecError("plane storage is short");
  }
}

void require_mask(const Mask& mask) {
  if (mask.height < 0 || mask.width < 0 ||
      mask.bits.size() < static_cast<size_t>(mask.height) * static_cast<size_t>(mask.width)) {
    throw CodecError("mask storage is short");
  }
}

struct Site {
  int y;
  int x;
};

std::vector<Site> valid_sites(const Mask& mask) {
  std::vector<Site> sites;
  sites.reserve(mask.bits.size());
  for (int y = 0; y < mask.height; ++y) {
    for (int x = 0; x < mask.width; ++x) {
      if (mask.at(y, x)) sites.push_back(Site{y, x});
    }
  }
  return sites;
}

bool symbol_escaped(int64_t quotient) { return quotient > QMAX || quotient < -int64_t(QMAX); }

int symbol_of(int64_t quotient) {
  if (symbol_escaped(quotient)) return CORE_BINS - 1;
  return int(quotient + QMAX);
}

std::unique_ptr<ParametricTables> build_parametric_tables() {
  auto tables = std::make_unique<ParametricTables>();
  constexpr int kTiltCount = NUM_NUS * NUM_SIGMAS * TILT_QS * TILT_NODES;
  static_assert(sizeof(ParametricTables::tilt_p0) / sizeof(int32_t) ==
                NUM_NUS * NUM_SIGMAS * TILT_QS * TILT_NODES);
  std::fill_n(&tables->tilt_p0[0][0][0][0], kTiltCount, RANS_M >> 1);

  for (int s = 0; s < NUM_SIGMAS; ++s) {
    const int b = dyadic_shift_q8(SIGMA_Q8[s]);
    tables->shifts[s] = b;
    const int64_t bias = b > 0 ? (1LL << (b - 1)) : 0;
    const int64_t sig = SIGMA_Q8[s];
    for (int nu = 0; nu < NUM_NUS; ++nu) {
      for (int f = 0; f < NUM_FRACS; ++f) {
        const int64_t offset = int64_t(f) - 8;
        int64_t masses[CORE_BINS];
        int32_t prev = 0;
        int64_t core_sum = 0;
        for (int edge = 0; edge < CORE_BINS; ++edge) {
          const int64_t q_edge = int64_t(-QMAX + edge);
          const int64_t base = (mul_pow2(q_edge, b) - bias) * 16 - 8;
          const int64_t z = floor_div_pos(mul_pow2(base - offset, 16), sig);
          const int32_t cdf = eval_cdf(z, nu);
          if (edge > 0) {
            int64_t diff = int64_t(cdf) - int64_t(prev);
            if (diff < 1) diff = 1;
            masses[edge - 1] = diff;
            core_sum += diff;
          }
          prev = cdf;
        }
        const int64_t esc = (int64_t(1) << 24) - core_sum;
        masses[CORE_BINS - 1] = esc < 1 ? 1 : esc;
        const std::vector<int32_t> freqs = normalize_impl(masses, CORE_BINS, RANS_M, false, false);
        const std::vector<int32_t> cum = cumulative(freqs);
        for (int i = 0; i < CORE_BINS; ++i) tables->freqs[nu][s][f][i] = freqs[static_cast<size_t>(i)];
        for (int i = 0; i < CORE_BINS + 1; ++i) tables->cums[nu][s][f][i] = cum[static_cast<size_t>(i)];
      }
      if (b <= 0) continue;
      const int depth = b < TILT_MAX_DEPTH ? b : TILT_MAX_DEPTH;
      const int n_sub = 1 << depth;
      const int sub_step = 1 << (b - depth);
      for (int qi = 0; qi < TILT_QS; ++qi) {
        const int q_val = qi - TILT_QMAX;
        const int64_t lo_r = mul_pow2(q_val, b) - bias;
        int32_t sub_cdf[9];
        for (int k = 0; k <= n_sub; ++k) {
          const int64_t edge = (lo_r + int64_t(k) * sub_step) * 16 - 8;
          const int64_t z = floor_div_pos(mul_pow2(edge, 16), sig);
          sub_cdf[k] = eval_cdf(z, nu);
        }
        int64_t sub_mass[8];
        for (int k = 0; k < n_sub; ++k) {
          int64_t diff = int64_t(sub_cdf[k + 1]) - int64_t(sub_cdf[k]);
          sub_mass[k] = diff < 1 ? 1 : diff;
        }
        for (int lvl = 0; lvl < depth; ++lvl) {
          const int span = n_sub >> (lvl + 1);
          const int base_node = (1 << lvl) - 1;
          for (int branch = 0; branch < (1 << lvl); ++branch) {
            const int start_i = branch * 2 * span;
            int64_t m0 = 0;
            int64_t m1 = 0;
            for (int k = 0; k < span; ++k) m0 += sub_mass[start_i + k];
            for (int k = 0; k < span; ++k) m1 += sub_mass[start_i + span + k];
            int64_t p0 = (m0 * RANS_M + ((m0 + m1) >> 1)) / (m0 + m1);
            if (p0 < 1) p0 = 1;
            if (p0 > RANS_M - 1) p0 = RANS_M - 1;
            tables->tilt_p0[nu][s][qi][base_node + branch] = static_cast<int32_t>(p0);
          }
        }
      }
    }
  }
  return tables;
}

CoreRow mix_bridge_row(int nu, int s, int f, int lo, int hi) {
  const ParametricTables& tables = get_parametric_tables();
  const int sig = SIGMA_Q8[std::max(0, s - 3)];
  const CoreRow bi = build_bimodal_row(nu, sig, lo, hi);
  int64_t masses[CORE_BINS];
  for (int i = 0; i < CORE_BINS; ++i) {
    masses[i] = int64_t(tables.freqs[nu][s][f][i]) + int64_t(bi.freqs[static_cast<size_t>(i)]);
  }
  CoreRow mix;
  mix.freqs = normalize_impl(masses, CORE_BINS, RANS_M, false, false);
  mix.cum = cumulative(std::span<const int32_t>(mix.freqs));
  return mix;
}

template <typename Map, typename Key>
const CoreRow& lookup_bridge(Map& cache, const Key& key, int nu, int s, int f, int lo, int hi) {
  auto found = cache.find(key);
  if (found == cache.end()) found = cache.emplace(key, mix_bridge_row(nu, s, f, lo, hi)).first;
  return found->second;
}

struct BitCursor {
  const uint8_t* raw = nullptr;
  size_t raw_size = 0;
  size_t pos = 0;
  uint64_t x = 0;
  BitReader* bits = nullptr;
};

int64_t pull_residual(BitCursor& cursor, const ParametricTables& tables, int nu, int s, int b,
                      const int32_t* f_row, const int32_t* c_row) {
  const int cf = int(cursor.x & static_cast<uint64_t>(RANS_M - 1));
  const int sym = bisect_right(c_row, CORE_BINS + 1, cf) - 1;
  if (sym < 0 || sym >= CORE_BINS) throw CodecError("parametric rANS symbol out of range");
  cursor.x = rans_advance_state(cursor.x, c_row[sym], f_row[sym], cursor.raw, cursor.raw_size, cursor.pos);
  int64_t quotient;
  if (sym == CORE_BINS - 1) quotient = read_escape(*cursor.bits);
  else quotient = int64_t(sym) - QMAX;
  if (b <= 0) return quotient;
  int64_t remainder;
  if (quotient >= -TILT_QMAX && quotient <= TILT_QMAX) {
    const int depth = b < TILT_MAX_DEPTH ? b : TILT_MAX_DEPTH;
    const int rem_low = b - depth;
    const int32_t* tilt = tables.tilt_p0[nu][s][int(quotient) + TILT_QMAX];
    int branch = 0;
    for (int lvl = 0; lvl < depth; ++lvl) {
      const int p0 = tilt[((1 << lvl) - 1) + branch];
      const int cf_bit = int(cursor.x & static_cast<uint64_t>(RANS_M - 1));
      int bit;
      int start;
      int freq;
      if (cf_bit < p0) {
        bit = 0;
        start = 0;
        freq = p0;
      } else {
        bit = 1;
        start = p0;
        freq = RANS_M - p0;
      }
      cursor.x = rans_advance_state(cursor.x, start, freq, cursor.raw, cursor.raw_size, cursor.pos);
      branch = (branch << 1) | bit;
    }
    const uint32_t low = rem_low > 0 ? cursor.bits->read_bits(rem_low) : 0;
    remainder = (int64_t(branch) << rem_low) | int64_t(low);
  } else {
    remainder = int64_t(read_wide(*cursor.bits, b));
  }
  return join_dyadic(quotient, remainder, b);
}

bool span_any(std::span<const uint8_t> mask) {
  for (uint8_t bit : mask) {
    if (bit) return true;
  }
  return false;
}

struct BridgeView {
  bool active = false;
  std::span<const uint8_t> mask;
  std::span<const int32_t> lo;
  std::span<const int32_t> hi;
};

BridgeView prepare_bridge(int n, bool allow, std::span<const uint8_t> mask, std::span<const int32_t> lo,
                          std::span<const int32_t> hi) {
  BridgeView view;
  if (!allow || mask.empty() || lo.empty() || hi.empty() || !span_any(mask)) return view;
  if (mask.size() != static_cast<size_t>(n) || lo.size() != static_cast<size_t>(n) ||
      hi.size() != static_cast<size_t>(n)) {
    throw CodecError("bridge context length mismatch");
  }
  view.active = true;
  view.mask = mask;
  view.lo = lo;
  view.hi = hi;
  return view;
}

void require_external(int n, std::span<const int32_t> ext_freq, std::span<const int32_t> ext_cum) {
  const bool freq_set = !ext_freq.empty();
  const bool cum_set = !ext_cum.empty();
  if (freq_set == cum_set) {
    if (!freq_set) return;
    if (ext_freq.size() != static_cast<size_t>(n) * CORE_BINS ||
        ext_cum.size() != static_cast<size_t>(n) * (CORE_BINS + 1)) {
      throw CodecError("external Core66 rows have the wrong shape");
    }
    for (int i = 0; i < n; ++i) {
      int64_t sum = 0;
      for (int bin = 0; bin < CORE_BINS; ++bin) {
        const int32_t freq = ext_freq[static_cast<size_t>(i) * CORE_BINS + static_cast<size_t>(bin)];
        if (freq < 1) throw CodecError("external Core66 rows are not a full partition");
        sum += freq;
      }
      if (sum != RANS_M) throw CodecError("external Core66 rows are not a full partition");
    }
    return;
  }
  throw CodecError("external Core66 rows must include frequencies and cumulatives");
}

std::vector<uint8_t> frame_parts(const std::vector<uint8_t>& rans, const std::vector<uint8_t>& extra) {
  std::vector<uint8_t> out;
  out.reserve(8 + rans.size() + extra.size());
  append_u32_le(out, static_cast<uint32_t>(rans.size()));
  append_u32_le(out, static_cast<uint32_t>(extra.size()));
  out.insert(out.end(), rans.begin(), rans.end());
  out.insert(out.end(), extra.begin(), extra.end());
  return out;
}

}  // namespace

int dyadic_shift_q8(int64_t sigma_q8) {
  if (sigma_q8 <= 640) return 0;
  int b = 0;
  int64_t acc = 640;
  while (acc < sigma_q8 && b < 12) {
    acc <<= 1;
    ++b;
  }
  return b;
}

int dyadic_shift(int64_t activity) {
  if (activity > (INT64_MAX >> 8)) return 12;
  if (activity < (INT64_MIN >> 8)) return 0;
  return dyadic_shift_q8(activity << 8);
}

DyadicParts split_dyadic(int64_t residual, int shift) {
  DyadicParts out;
  if (shift <= 0) {
    out.quotient = residual;
    return out;
  }
  if (shift > 62) throw CodecError("dyadic shift out of range");
  const int64_t bias = 1LL << (shift - 1);
  out.quotient = floor_div_pos(residual + bias, 1LL << shift);
  out.remainder = residual - mul_pow2(out.quotient, shift) + bias;
  return out;
}

int64_t join_dyadic(int64_t quotient, std::optional<int64_t> remainder, int shift) {
  if (shift <= 0) return quotient;
  if (!remainder) throw CodecError("missing dyadic remainder");
  if (shift > 62) throw CodecError("dyadic shift out of range");
  const int64_t bias = 1LL << (shift - 1);
  return mul_pow2(quotient, shift) - bias + *remainder;
}

int rice_k(int64_t acc, int64_t count) {
  int k = 0;
  int64_t scaled = count;
  while (scaled < acc && k < 16) {
    if (scaled > (INT64_MAX >> 1)) break;
    scaled <<= 1;
    ++k;
  }
  return k;
}

std::vector<int32_t> normalize_freqs(std::span<const int64_t> counts, int total) {
  if (counts.size() > static_cast<size_t>(INT_MAX)) throw CodecError("frequency normalization failed");
  return normalize_impl(counts.data(), int(counts.size()), total, true, true);
}

std::vector<int32_t> normalize_freqs(std::span<const int32_t> counts, int total) {
  std::vector<int64_t> widened(counts.begin(), counts.end());
  return normalize_freqs(std::span<const int64_t>(widened), total);
}

std::vector<int32_t> cumulative(std::span<const int32_t> freqs) {
  std::vector<int32_t> cum(freqs.size() + 1, 0);
  int64_t acc = 0;
  for (size_t i = 0; i < freqs.size(); ++i) {
    acc += freqs[i];
    cum[i + 1] = static_cast<int32_t>(acc);
  }
  return cum;
}

std::vector<int32_t> cumulative(std::span<const int64_t> freqs) {
  std::vector<int32_t> cum(freqs.size() + 1, 0);
  int64_t acc = 0;
  for (size_t i = 0; i < freqs.size(); ++i) {
    acc += freqs[i];
    cum[i + 1] = static_cast<int32_t>(acc);
  }
  return cum;
}

void RansEncoder::put(int start, int freq) { rans_push(state_, words_, start, freq); }

void RansEncoder::put_batch(std::span<const int32_t> starts, std::span<const int32_t> freqs) {
  if (starts.size() != freqs.size()) throw CodecError("rANS batch length mismatch");
  for (size_t i = 0; i < starts.size(); ++i) rans_push(state_, words_, starts[i], freqs[i]);
}

std::vector<uint8_t> RansEncoder::finish() {
  words_.push_back(static_cast<uint32_t>(state_));
  words_.push_back(static_cast<uint32_t>(state_ >> 32));
  std::vector<uint8_t> out;
  out.reserve(words_.size() * 4);
  append_u32_le(out, words_.back());
  append_u32_le(out, words_[words_.size() - 2]);
  if (words_.size() >= 3) {
    for (size_t i = words_.size() - 3;; --i) {
      append_u32_le(out, words_[i]);
      if (i == 0) break;
    }
  }
  return out;
}

RansDecoder::RansDecoder(std::span<const uint8_t> data) : data_(data.data()), size_(data.size()), pos_(8) {
  if (data.size() < 8) throw CodecError("truncated rANS stream");
  const uint32_t hi = read_u32_le(data.data());
  const uint32_t lo = read_u32_le(data.data() + 4);
  state_ = (uint64_t(hi) << 32) | lo;
}

int RansDecoder::cf() const { return int(state_ & static_cast<uint64_t>(RANS_M - 1)); }

void RansDecoder::advance(int start, int freq) {
  state_ = rans_advance_state(state_, start, freq, data_, size_, pos_);
}

void RansBitsBackStack::push(int start, int freq) { rans_push(state, words, start, freq); }

int RansBitsBackStack::pop(std::span<const int32_t> cum, std::span<const int32_t> freqs) {
  const int cf = int(state & static_cast<uint64_t>(RANS_M - 1));
  const int sym = bisect_right(cum.data(), int(cum.size()), cf) - 1;
  if (sym < 0 || sym >= int(freqs.size())) throw CodecError("bits-back pop symbol out of range");
  const int freq = freqs[static_cast<size_t>(sym)];
  const int start = cum[static_cast<size_t>(sym)];
  uint64_t x = static_cast<uint64_t>(freq) * (state >> RANS_SCALE) + static_cast<uint64_t>(cf) -
               static_cast<uint64_t>(start);
  while (x < RANS64_L) {
    if (words.empty()) throw CodecError("bits-back seed stack underflow");
    x = (x << 32) | words.back();
    words.pop_back();
  }
  state = x;
  return sym;
}

std::vector<uint8_t> RansBitsBackStack::to_bytes() const {
  std::vector<uint8_t> out;
  out.reserve(12 + words.size() * 4);
  append_u32_le(out, static_cast<uint32_t>(state >> 32));
  append_u32_le(out, static_cast<uint32_t>(state));
  append_u32_le(out, static_cast<uint32_t>(words.size()));
  for (size_t i = words.size(); i-- > 0;) append_u32_le(out, words[i]);
  return out;
}

RansBitsBackStack RansBitsBackStack::from_bytes(std::span<const uint8_t> data) {
  if (data.size() < 12) throw CodecError("truncated bits-back rANS stream");
  const uint32_t hi = read_u32_le(data.data());
  const uint32_t lo = read_u32_le(data.data() + 4);
  const uint32_t nwords = read_u32_le(data.data() + 8);
  if (uint64_t(12) + uint64_t(nwords) * 4 > data.size()) throw CodecError("truncated bits-back word stack");
  std::vector<uint32_t> stack(nwords);
  for (uint32_t i = 0; i < nwords; ++i) stack[i] = read_u32_le(data.data() + 12 + 4 * i);
  std::reverse(stack.begin(), stack.end());
  return RansBitsBackStack((uint64_t(hi) << 32) | lo, std::move(stack));
}

std::vector<uint8_t> encode_rice_values(const Plane& values, const Mask& valid, const Plane* context,
                                        bool causal_mu) {
  require_plane(values);
  require_mask(valid);
  if (values.height != valid.height || values.width != valid.width) throw CodecError("rice plane shape mismatch");
  if (context) {
    require_plane(*context);
    if (context->height != valid.height || context->width != valid.width) {
      throw CodecError("rice context shape mismatch");
    }
  }
  BitWriter writer;
  int64_t acc[8];
  int64_t count[8];
  for (int i = 0; i < 8; ++i) {
    acc[i] = 4;
    count[i] = 1;
  }
  int32_t prev = 0;
  for (Site site : valid_sites(valid)) {
    int ctx = 0;
    if (context) ctx = context->at(site.y, site.x);
    if (ctx < 0 || ctx > 7) ctx = 7;
    const int32_t sample = values.at(site.y, site.x);
    const int32_t mu = causal_mu ? prev : 0;
    const int64_t err = int64_t(sample) - int64_t(mu);
    const int64_t symbol = err < 0 ? ((-err) << 1) - 1 : (err << 1);
    if (symbol < 0 || symbol > INT32_MAX) throw CodecError("rice symbol exceeds 32-bit range");
    const int k = rice_k(acc[ctx], count[ctx]);
    writer.write_rice(static_cast<int>(symbol), k);
    acc[ctx] += symbol;
    count[ctx] += 1;
    if (count[ctx] >= 64) {
      acc[ctx] = acc[ctx] >> 1;
      if (acc[ctx] < 1) acc[ctx] = 1;
      count[ctx] >>= 1;
    }
    prev = sample;
  }
  return writer.finish();
}

Plane decode_rice_values(std::span<const uint8_t> data, int height, int width, const Mask& valid,
                         const Plane* context, bool causal_mu) {
  require_mask(valid);
  if (valid.height != height || valid.width != width) throw CodecError("rice plane shape mismatch");
  if (context) {
    require_plane(*context);
    if (context->height != height || context->width != width) throw CodecError("rice context shape mismatch");
  }
  BitReader reader(data.data(), data.size());
  int64_t acc[8];
  int64_t count[8];
  for (int i = 0; i < 8; ++i) {
    acc[i] = 4;
    count[i] = 1;
  }
  Plane out(height, width, 0);
  int32_t prev = 0;
  for (Site site : valid_sites(valid)) {
    int ctx = 0;
    if (context) ctx = context->at(site.y, site.x);
    if (ctx < 0 || ctx > 7) ctx = 7;
    const int32_t mu = causal_mu ? prev : 0;
    const int k = rice_k(acc[ctx], count[ctx]);
    const int symbol = reader.read_rice(k);
    const int err = unzigzag(symbol);
    const int32_t value = static_cast<int32_t>(int64_t(err) + int64_t(mu));
    out.at(site.y, site.x) = value;
    acc[ctx] += symbol;
    count[ctx] += 1;
    if (count[ctx] >= 64) {
      acc[ctx] = acc[ctx] >> 1;
      if (acc[ctx] < 1) acc[ctx] = 1;
      count[ctx] >>= 1;
    }
    prev = value;
  }
  return out;
}

std::vector<uint8_t> encode_dyadic_rans(const Plane& residuals, const Mask& valid, const Plane& shifts) {
  require_plane(residuals);
  require_plane(shifts);
  require_mask(valid);
  if (residuals.height != valid.height || residuals.width != valid.width || shifts.height != valid.height ||
      shifts.width != valid.width) {
    throw CodecError("dyadic plane shape mismatch");
  }
  const std::vector<Site> sites = valid_sites(valid);
  std::vector<int> symbols;
  symbols.reserve(sites.size());
  struct Extra {
    int64_t quotient;
    std::optional<int64_t> remainder;
    int shift;
    bool escaped;
  };
  std::vector<Extra> extras;
  extras.reserve(sites.size());
  int64_t counts[CORE_BINS] = {};
  for (Site site : sites) {
    const int shift = shifts.at(site.y, site.x);
    const DyadicParts parts = split_dyadic(residuals.at(site.y, site.x), shift);
    const bool escaped = symbol_escaped(parts.quotient);
    const int symbol = symbol_of(parts.quotient);
    symbols.push_back(symbol);
    counts[symbol] += 1;
    extras.push_back(Extra{parts.quotient, parts.remainder, shift, escaped});
  }
  const std::vector<int32_t> freqs = normalize_impl(counts, CORE_BINS, RANS_M, true, true);
  const std::vector<int32_t> cum = cumulative(freqs);
  RansEncoder encoder;
  for (size_t i = symbols.size(); i-- > 0;) {
    const int symbol = symbols[i];
    encoder.put(cum[static_cast<size_t>(symbol)], freqs[static_cast<size_t>(symbol)]);
  }
  const std::vector<uint8_t> rans = encoder.finish();
  BitWriter writer;
  for (const Extra& extra : extras) {
    if (extra.escaped) write_escape(writer, extra.quotient);
    if (extra.shift > 0) {
      if (!extra.remainder || *extra.remainder < 0 || *extra.remainder >= (1LL << extra.shift)) {
        throw CodecError("dyadic remainder out of range");
      }
      write_wide(writer, static_cast<uint64_t>(*extra.remainder), extra.shift);
    }
  }
  const std::vector<uint8_t> side = writer.finish();
  std::vector<uint8_t> out;
  out.reserve(CORE_BINS * 2 + 8 + rans.size() + side.size());
  for (int32_t freq : freqs) {
    if (freq < 0 || freq > 65535) throw CodecError("invalid dyadic frequency table");
    append_u16_le(out, static_cast<uint16_t>(freq));
  }
  append_u32_le(out, static_cast<uint32_t>(rans.size()));
  append_u32_le(out, static_cast<uint32_t>(side.size()));
  out.insert(out.end(), rans.begin(), rans.end());
  out.insert(out.end(), side.begin(), side.end());
  return out;
}

Plane decode_dyadic_rans(std::span<const uint8_t> data, int height, int width, const Mask& valid,
                         const Plane& shifts) {
  require_mask(valid);
  require_plane(shifts);
  if (valid.height != height || valid.width != width || shifts.height != height || shifts.width != width) {
    throw CodecError("dyadic plane shape mismatch");
  }
  if (data.size() < static_cast<size_t>(CORE_BINS) * 2 + 8) throw CodecError("truncated dyadic rANS header");
  int32_t freq_row[CORE_BINS];
  int64_t freq_sum = 0;
  size_t pos = 0;
  for (int i = 0; i < CORE_BINS; ++i) {
    freq_row[i] = read_u16_le(data.data() + pos);
    pos += 2;
    if (freq_row[i] < 1) throw CodecError("invalid dyadic frequency table");
    freq_sum += freq_row[i];
  }
  if (freq_sum != RANS_M) throw CodecError("invalid dyadic frequency table");
  const uint32_t rans_n = read_u32_le(data.data() + pos);
  pos += 4;
  const uint32_t extra_n = read_u32_le(data.data() + pos);
  pos += 4;
  if (uint64_t(pos) + rans_n + extra_n > data.size()) throw CodecError("truncated dyadic payload");
  const std::span<const int32_t> freqs(freq_row, CORE_BINS);
  const std::vector<int32_t> cum = cumulative(freqs);
  const std::vector<Site> sites = valid_sites(valid);
  RansDecoder decoder(data.subspan(pos, rans_n));
  std::vector<int> symbols(sites.size(), 0);
  for (size_t i = 0; i < sites.size(); ++i) {
    const int cf = decoder.cf();
    const int symbol = bisect_right(cum.data(), int(cum.size()), cf) - 1;
    if (symbol < 0 || symbol >= CORE_BINS) throw CodecError("rANS symbol out of range");
    if (!(cum[static_cast<size_t>(symbol)] <= cf && cf < cum[static_cast<size_t>(symbol) + 1])) {
      throw CodecError("rANS cumulative mismatch");
    }
    decoder.advance(cum[static_cast<size_t>(symbol)], freq_row[symbol]);
    symbols[i] = symbol;
  }
  BitReader reader(data.data() + pos + rans_n, extra_n);
  Plane out(height, width, 0);
  for (size_t i = 0; i < sites.size(); ++i) {
    const int shift = shifts.at(sites[i].y, sites[i].x);
    int64_t quotient;
    std::optional<int64_t> remainder;
    if (symbols[i] == CORE_BINS - 1) quotient = read_escape(reader);
    else quotient = int64_t(symbols[i]) - QMAX;
    if (shift > 0) remainder = int64_t(read_wide(reader, shift));
    out.at(sites[i].y, sites[i].x) = static_cast<int32_t>(join_dyadic(quotient, remainder, shift));
  }
  return out;
}

int32_t eval_student_cdf_q24(int64_t z_q12, int nu_idx) { return eval_cdf(z_q12, nu_idx); }

const ParametricTables& get_parametric_tables() {
  static const std::unique_ptr<ParametricTables> tables = build_parametric_tables();
  return *tables;
}

CoreRow build_bimodal_row(int nu_idx, int sig_q8, int mode_lo, int mode_hi) {
  if (nu_idx < 0 || nu_idx >= NUM_NUS) throw CodecError("nu index out of range");
  const int64_t sig = sig_q8 > 32 ? int64_t(sig_q8) : 32;
  int32_t cdf[CORE_BINS];
  for (int edge = 0; edge < CORE_BINS; ++edge) {
    const int64_t boundary = int64_t(-QMAX + edge) * 16 - 8;
    const int64_t z_lo = floor_div_pos(mul_pow2(boundary - int64_t(mode_lo) * 16, 16), sig);
    const int64_t z_hi = floor_div_pos(mul_pow2(boundary - int64_t(mode_hi) * 16, 16), sig);
    const int32_t lo = eval_cdf(z_lo, nu_idx);
    const int32_t hi = eval_cdf(z_hi, nu_idx);
    cdf[edge] = static_cast<int32_t>((int64_t(lo) + int64_t(hi)) >> 1);
  }
  int64_t masses[CORE_BINS];
  int64_t core_sum = 0;
  for (int i = 0; i < CORE_BINS - 1; ++i) {
    int64_t diff = int64_t(cdf[i + 1]) - int64_t(cdf[i]);
    if (diff < 1) diff = 1;
    masses[i] = diff;
    core_sum += diff;
  }
  const int64_t esc = (int64_t(1) << 24) - core_sum;
  masses[CORE_BINS - 1] = esc < 1 ? 1 : esc;
  CoreRow row;
  row.freqs = normalize_impl(masses, CORE_BINS, RANS_M, false, false);
  row.cum = cumulative(std::span<const int32_t>(row.freqs));
  return row;
}

std::vector<uint8_t> encode_parametric_rans(std::span<const int32_t> r_tilde, int nu_idx,
                                            std::span<const int32_t> sig_idx, std::span<const int32_t> frac_idx,
                                            std::span<const uint8_t> bridge_mask, std::span<const int32_t> bridge_lo,
                                            std::span<const int32_t> bridge_hi, std::span<const int32_t> ext_freq,
                                            std::span<const int32_t> ext_cum) {
  if (r_tilde.empty()) return std::vector<uint8_t>(8, 0);
  const int n = int(r_tilde.size());
  if (sig_idx.size() != r_tilde.size() || frac_idx.size() != r_tilde.size()) {
    throw CodecError("parametric context length mismatch");
  }
  if (nu_idx < 0 || nu_idx >= NUM_NUS) throw CodecError("nu index out of range");
  const bool use_ext = !ext_freq.empty() || !ext_cum.empty();
  require_external(n, ext_freq, ext_cum);
  const BridgeView bridge = prepare_bridge(n, !use_ext, bridge_mask, bridge_lo, bridge_hi);
  const ParametricTables& tables = get_parametric_tables();
  std::vector<int32_t> starts;
  std::vector<int32_t> freqs;
  starts.reserve(static_cast<size_t>(n) * 4);
  freqs.reserve(static_cast<size_t>(n) * 4);
  BitWriter writer;
  std::map<std::tuple<int, int, int, int>, CoreRow> cache;
  for (int i = 0; i < n; ++i) {
    const int s = clamp_int(sig_idx[static_cast<size_t>(i)], 0, NUM_SIGMAS - 1);
    const int f = clamp_int(frac_idx[static_cast<size_t>(i)], 0, NUM_FRACS - 1);
    const int b = tables.shifts[s];
    const DyadicParts parts = split_dyadic(r_tilde[static_cast<size_t>(i)], b);
    const int64_t quotient = parts.quotient;
    const int64_t remainder = parts.remainder.value_or(0);
    const int symbol = symbol_of(quotient);
    const int32_t* f_row = use_ext ? ext_freq.data() + static_cast<size_t>(i) * CORE_BINS : tables.freqs[nu_idx][s][f];
    const int32_t* c_row =
        use_ext ? ext_cum.data() + static_cast<size_t>(i) * (CORE_BINS + 1) : tables.cums[nu_idx][s][f];
    const CoreRow* mix = nullptr;
    if (bridge.active && bridge.mask[static_cast<size_t>(i)] && b == 0) {
      const int lo = clamp_int(bridge.lo[static_cast<size_t>(i)], -28, 28);
      const int hi = clamp_int(bridge.hi[static_cast<size_t>(i)], -28, 28);
      if (hi - lo >= 6) {
        mix = &lookup_bridge(cache, std::make_tuple(s, f, lo, hi), nu_idx, s, f, lo, hi);
        f_row = mix->freqs.data();
        c_row = mix->cum.data();
      }
    }
    starts.push_back(c_row[symbol]);
    freqs.push_back(f_row[symbol]);
    if (symbol == CORE_BINS - 1) write_escape(writer, quotient);
    if (b > 0) {
      if (quotient >= -TILT_QMAX && quotient <= TILT_QMAX) {
        const int depth = b < TILT_MAX_DEPTH ? b : TILT_MAX_DEPTH;
        const int rem_low = b - depth;
        const int top = int(remainder >> rem_low);
        const int32_t* tilt = tables.tilt_p0[nu_idx][s][int(quotient) + TILT_QMAX];
        int branch = 0;
        for (int lvl = 0; lvl < depth; ++lvl) {
          const int bit = (top >> (depth - 1 - lvl)) & 1;
          const int p0 = tilt[((1 << lvl) - 1) + branch];
          if (bit == 0) {
            starts.push_back(0);
            freqs.push_back(p0);
          } else {
            starts.push_back(p0);
            freqs.push_back(RANS_M - p0);
          }
          branch = (branch << 1) | bit;
        }
        if (rem_low > 0) writer.write_bits(static_cast<uint32_t>(remainder), rem_low);
      } else {
        writer.write_bits(static_cast<uint32_t>(remainder), b);
      }
    }
  }
  std::reverse(starts.begin(), starts.end());
  std::reverse(freqs.begin(), freqs.end());
  RansEncoder encoder;
  encoder.put_batch(starts, freqs);
  return frame_parts(encoder.finish(), writer.finish());
}

std::vector<int32_t> decode_parametric_rans(std::span<const uint8_t> data, int count, int nu_idx,
                                            std::span<const int32_t> sig_idx, std::span<const int32_t> frac_idx,
                                            std::span<const uint8_t> bridge_mask, std::span<const int32_t> bridge_lo,
                                            std::span<const int32_t> bridge_hi, std::span<const int32_t> ext_freq,
                                            std::span<const int32_t> ext_cum) {
  if (data.size() < 8) throw CodecError("truncated parametric rANS header");
  const uint32_t rans_n = read_u32_le(data.data());
  const uint32_t extra_n = read_u32_le(data.data() + 4);
  if (uint64_t(8) + rans_n + extra_n > data.size()) throw CodecError("truncated parametric rANS payload");
  if (count == 0) return {};
  if (count < 0) throw CodecError("parametric context length mismatch");
  if (sig_idx.size() != static_cast<size_t>(count) || frac_idx.size() != static_cast<size_t>(count)) {
    throw CodecError("parametric context length mismatch");
  }
  if (nu_idx < 0 || nu_idx >= NUM_NUS) throw CodecError("nu index out of range");
  if (rans_n < 8) throw CodecError("truncated rANS stream");
  const bool use_ext = !ext_freq.empty() || !ext_cum.empty();
  require_external(count, ext_freq, ext_cum);
  const BridgeView bridge = prepare_bridge(count, !use_ext, bridge_mask, bridge_lo, bridge_hi);
  const ParametricTables& tables = get_parametric_tables();
  const uint8_t* raw = data.data() + 8;
  BitCursor cursor;
  cursor.raw = raw;
  cursor.raw_size = rans_n;
  cursor.pos = 8;
  cursor.x = (uint64_t(read_u32_le(raw)) << 32) | read_u32_le(raw + 4);
  BitReader reader(data.data() + 8 + rans_n, extra_n);
  cursor.bits = &reader;
  std::map<std::tuple<int, int, int, int>, CoreRow> cache;
  std::vector<int32_t> out(static_cast<size_t>(count), 0);
  for (int i = 0; i < count; ++i) {
    const int s = clamp_int(sig_idx[static_cast<size_t>(i)], 0, NUM_SIGMAS - 1);
    const int f = clamp_int(frac_idx[static_cast<size_t>(i)], 0, NUM_FRACS - 1);
    const int b = tables.shifts[s];
    const int32_t* f_row = use_ext ? ext_freq.data() + static_cast<size_t>(i) * CORE_BINS : tables.freqs[nu_idx][s][f];
    const int32_t* c_row =
        use_ext ? ext_cum.data() + static_cast<size_t>(i) * (CORE_BINS + 1) : tables.cums[nu_idx][s][f];
    const CoreRow* mix = nullptr;
    if (bridge.active && bridge.mask[static_cast<size_t>(i)] && b == 0) {
      const int lo = clamp_int(bridge.lo[static_cast<size_t>(i)], -28, 28);
      const int hi = clamp_int(bridge.hi[static_cast<size_t>(i)], -28, 28);
      if (hi - lo >= 6) {
        mix = &lookup_bridge(cache, std::make_tuple(s, f, lo, hi), nu_idx, s, f, lo, hi);
        f_row = mix->freqs.data();
        c_row = mix->cum.data();
      }
    }
    const int64_t value = pull_residual(cursor, tables, nu_idx, s, b, f_row, c_row);
    out[static_cast<size_t>(i)] = static_cast<int32_t>(value);
  }
  return out;
}

ParametricDecoder::ParametricDecoder(std::span<const uint8_t> data, int nu_idx) {
  if (data.size() < 8) throw CodecError("truncated parametric rANS header");
  const uint32_t rans_n = read_u32_le(data.data());
  const uint32_t extra_n = read_u32_le(data.data() + 4);
  if (uint64_t(8) + rans_n + extra_n > data.size()) throw CodecError("truncated parametric rANS payload");
  if (nu_idx < 0 || nu_idx >= NUM_NUS) throw CodecError("nu index out of range");
  if (rans_n < 8) throw CodecError("truncated rANS stream");
  (void)get_parametric_tables();
  nu_ = nu_idx;
  raw_.assign(data.begin() + 8, data.begin() + 8 + rans_n);
  extra_.assign(data.begin() + 8 + rans_n, data.begin() + 8 + rans_n + extra_n);
  x_ = (uint64_t(read_u32_le(raw_.data())) << 32) | read_u32_le(raw_.data() + 4);
  pos_ = 8;
  reader_.emplace(extra_.data(), extra_.size());
}

int32_t ParametricDecoder::read(int sig_i, int frac_i, std::optional<std::pair<int, int>> bridge) {
  const ParametricTables& tables = get_parametric_tables();
  const int s = (sig_i >= 0 && sig_i < NUM_SIGMAS) ? sig_i : 0;
  const int f = (frac_i >= 0 && frac_i < NUM_FRACS) ? frac_i : 0;
  const int b = tables.shifts[s];
  const int32_t* f_row = tables.freqs[nu_][s][f];
  const int32_t* c_row = tables.cums[nu_][s][f];
  const CoreRow* mix = nullptr;
  if (bridge && b == 0) {
    const int lo = clamp_int(bridge->first, -28, 28);
    const int hi = clamp_int(bridge->second, -28, 28);
    if (hi - lo >= 6) {
      mix = &lookup_bridge(bi_, BiKey{s, f, lo, hi}, nu_, s, f, lo, hi);
      f_row = mix->freqs.data();
      c_row = mix->cum.data();
    }
  }
  BitCursor cursor;
  cursor.raw = raw_.data();
  cursor.raw_size = raw_.size();
  cursor.pos = pos_;
  cursor.x = x_;
  cursor.bits = &*reader_;
  const int64_t value = pull_residual(cursor, tables, nu_, s, b, f_row, c_row);
  pos_ = cursor.pos;
  x_ = cursor.x;
  return static_cast<int32_t>(value);
}

Plane shift_map_from_activity(const Plane& activity_ctx) {
  require_plane(activity_ctx);
  static const int kSigma[8] = {1, 2, 3, 5, 10, 20, 40, 80};
  int mapped[8];
  for (int bucket = 0; bucket < 8; ++bucket) mapped[bucket] = dyadic_shift(kSigma[bucket]);
  Plane out(activity_ctx.height, activity_ctx.width, 0);
  for (int y = 0; y < activity_ctx.height; ++y) {
    for (int x = 0; x < activity_ctx.width; ++x) {
      const int32_t bucket = activity_ctx.at(y, x);
      if (bucket >= 0 && bucket < 8) out.at(y, x) = mapped[bucket];
    }
  }
  return out;
}

}  // namespace vpxc
