"""Dyadic quotient coding: adaptive Rice, parametric Dyadic-Core66 rANS, and S^3-BB-rANS.

Implements Section 3.5, Section 4.1 Eq. (6), Appendix A.3, and Appendix A.4 of
the SLGF specification in 100% IEEE-754-free integer arithmetic:

- Exact integer largest-remainder frequency normalization (zero float division).
- Normative 2 KB Q24 monotone PCHIP cumulative Student-t lookup tables for
  ``nu in {2.5, 3.5, 6.0, inf}`` over sub-integer fractional means and 32
  log-spaced Q8 scales covering 8-bit through 16-bit dynamic ranges.
- Bimodal bilateral sheaf bridge mixtures across strong discontinuities.
- Tilted binary rANS sub-bucket remainder tree on ``|q| <= 6`` for ``b > 0``.
- Tail-Seeded Split-Lattice Bits-Back rANS (``s3_bb_encode`` / ``s3_bb_decode``)
  achieving zero cold-start auxiliary bit waste on single images.
"""

from __future__ import annotations

from bisect import bisect_right

import numpy as np

from vpxc.errors import CodecError

RICE_ESCAPE = 32
RANS64_L = 1 << 31
RANS_SCALE = 16
RANS_M = 1 << RANS_SCALE
CORE_BINS = 66  # 65 centered quotients [-32..32] + escape symbol 65
QMAX = 32
TILT_QMAX = 6
TILT_MAX_DEPTH = 3

# 32 log-spaced scales in Q8 sample units (sigma = SIGMA_Q8 / 256.0), spanning
# 0.125 LSB (ultra-smooth chroma) up to 724 LSB (16-bit medical / RAW).
SIGMA_Q8 = (
    32, 42, 56, 74, 98, 130, 172, 228,
    302, 400, 530, 702, 930, 1232, 1632, 2162,
    2864, 3794, 5026, 6658, 8820, 11684, 15478, 20504,
    27162, 35982, 47666, 63144, 83648, 110810, 146792, 194458,
)
NUM_SIGMAS = len(SIGMA_Q8)
NUM_FRACS = 16  # delta_mu in {-8/16 .. +7/16}
NUM_NUS = 4     # {2.5, 3.5, 6.0, inf}

# Normative 2 KB Q24 Student-t cumulative knots F_nu(z) at z = k / 8 for k = 0..256.
_CDF_NU25 = (
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
)
_CDF_NU35 = (
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
)
_CDF_NU60 = (
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
)
_CDF_NUINF = (
    8388608, 9223077, 10044625, 10840930, 11600815, 12314719, 12975040, 13576342, 14115423, 14591237,
    15004707, 15358439, 15656377, 15903437, 16105135, 16267250, 16395532, 16495471, 16572123, 16630005,
    16673035, 16704530, 16727224, 16743324, 16754568, 16762301, 16767535, 16771024, 16773313, 16774792,
    16775733, 16776322, 16776685, 16776905, 16777037, 16777114, 16777159, 16777185, 16777199, 16777207,
    16777211, 16777214, 16777215,
) + (16777215,) * 214

_KNOT_TABLES = tuple(
    np.asarray(tbl, dtype=np.int64)
    for tbl in (_CDF_NU25, _CDF_NU35, _CDF_NU60, _CDF_NUINF)
)


def _pchip_slopes_int(y: np.ndarray) -> np.ndarray:
    delta = y[1:] - y[:-1]
    d = np.zeros_like(y)
    d0 = delta[:-1]
    d1 = delta[1:]
    pos = (d0 > 0) & (d1 > 0)
    d[1:-1][pos] = (2 * d0[pos] * d1[pos] + ((d0[pos] + d1[pos]) >> 1)) // (
        d0[pos] + d1[pos]
    )
    d[0] = delta[0]
    d[-1] = delta[-1]
    return d


_SLOPE_TABLES = tuple(_pchip_slopes_int(y) for y in _KNOT_TABLES)


def eval_student_cdf_q24(z_q12: np.ndarray, nu_idx: int) -> np.ndarray:
    """Normative integer PCHIP Student-t CDF in Q24 (``1 << 24 == 1.0``).

    ``z_q12`` holds ``round(z * 4096)`` as signed ``int64``; ``nu_idx in 0..3``
    selects ``nu in {2.5, 3.5, 6.0, inf}``. Zero floating-point operations.
    """
    y = _KNOT_TABLES[int(nu_idx)]
    m = _SLOPE_TABLES[int(nu_idx)]
    z = np.asarray(z_q12, dtype=np.int64)
    neg = z < 0
    az = np.abs(z)
    clamped = az >= (256 << 9)
    idx = np.minimum(az >> 9, 255)
    t = az & 511
    y0 = y[idx]
    y1 = y[idx + 1]
    m0 = m[idx]
    m1 = m[idx + 1]
    dy = y1 - y0
    c2 = 3 * dy - 2 * m0 - m1
    c3 = m0 + m1 - 2 * dy
    p1 = ((c3 * t + 256) >> 9) + c2
    p2 = ((p1 * t + 256) >> 9) + m0
    val = y0 + ((p2 * t + 256) >> 9)
    val = np.where(clamped, y[256], val)
    val = np.clip(val, 1 << 23, (1 << 24) - 1)
    return np.where(neg, (1 << 24) - val, val)


def dyadic_shift_q8(sigma_q8: int) -> int:
    """Exact ``max(0, ceil(log2((sigma_q8 / 256) / 2.5)))`` in pure integers."""
    s = int(sigma_q8)
    if s <= 640:  # 2.5 * 256 = 640
        return 0
    b = 0
    acc = 640
    while acc < s and b < 12:
        acc <<= 1
        b += 1
    return b


def zigzag(err: int) -> int:
    if err < 0:
        return ((-err) << 1) - 1
    return err << 1


def unzigzag(symbol: int) -> int:
    if symbol & 1:
        return -((symbol + 1) >> 1)
    return symbol >> 1


def dyadic_shift(activity: int) -> int:
    """``b = max(0, ceil(log2(sigma / 2.5)))`` with integer ``sigma = activity``."""
    return dyadic_shift_q8(int(activity) << 8)


def split_dyadic(residual: int, shift: int) -> tuple[int, int | None]:
    """Return ``(quotient, remainder)``. Remainder is ``None`` when ``shift == 0``."""
    residual = int(residual)
    if shift <= 0:
        return residual, None
    bias = 1 << (shift - 1)
    quotient = (residual + bias) >> shift
    remainder = residual - (quotient << shift) + bias
    return quotient, remainder


def join_dyadic(quotient: int, remainder: int | None, shift: int) -> int:
    if shift <= 0:
        return int(quotient)
    if remainder is None:
        raise CodecError("missing dyadic remainder")
    bias = 1 << (shift - 1)
    return (int(quotient) << shift) - bias + int(remainder)


class BitWriter:
    def __init__(self) -> None:
        self._buf = 0
        self._n = 0
        self._out = bytearray()

    def write_bits(self, value: int, width: int) -> None:
        if width < 0:
            raise CodecError("negative bit width")
        if width == 0:
            return
        mask = (1 << width) - 1
        self._buf = (self._buf << width) | (int(value) & mask)
        self._n += width
        while self._n >= 8:
            self._n -= 8
            self._out.append((self._buf >> self._n) & 0xFF)
            if self._n:
                self._buf &= (1 << self._n) - 1
            else:
                self._buf = 0

    def write_rice(self, symbol: int, k: int) -> None:
        if symbol < 0:
            raise CodecError("rice symbol must be non-negative")
        quotient = symbol >> k if k else symbol
        if quotient < RICE_ESCAPE:
            if quotient:
                self.write_bits((1 << quotient) - 1, quotient)
            self.write_bits(0, 1)
            if k:
                self.write_bits(symbol & ((1 << k) - 1), k)
        else:
            self.write_bits((1 << RICE_ESCAPE) - 1, RICE_ESCAPE)
            self.write_bits(0, 1)
            self.write_exp_golomb(symbol)

    def write_exp_golomb(self, value: int) -> None:
        if value < 0:
            raise CodecError("exp-golomb value must be non-negative")
        code = value + 1
        zeros = code.bit_length() - 1
        if zeros:
            self.write_bits(0, zeros)
        self.write_bits(code, zeros + 1)

    def write_signed_golomb(self, value: int) -> None:
        self.write_exp_golomb(zigzag(int(value)))

    def finish(self) -> bytes:
        if self._n:
            self._out.append((self._buf << (8 - self._n)) & 0xFF)
            self._n = 0
            self._buf = 0
        return bytes(self._out)


class BitReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0
        self._buf = 0
        self._n = 0

    def read_bits(self, width: int) -> int:
        if width < 0:
            raise CodecError("negative bit width")
        if width == 0:
            return 0
        while self._n < width:
            if self._pos >= len(self._data):
                raise CodecError("truncated bitstream")
            self._buf = (self._buf << 8) | self._data[self._pos]
            self._pos += 1
            self._n += 8
        self._n -= width
        value = (self._buf >> self._n) & ((1 << width) - 1)
        if self._n:
            self._buf &= (1 << self._n) - 1
        else:
            self._buf = 0
        return value

    def read_rice(self, k: int) -> int:
        quotient = 0
        while quotient < RICE_ESCAPE:
            if self.read_bits(1) == 0:
                rem = self.read_bits(k) if k else 0
                return (quotient << k) | rem
            quotient += 1
        if self.read_bits(1) != 0:
            raise CodecError("bad rice escape")
        return self.read_exp_golomb()

    def read_exp_golomb(self) -> int:
        zeros = 0
        while self.read_bits(1) == 0:
            zeros += 1
            if zeros > 31:
                raise CodecError("exp-golomb overflow")
        if zeros == 0:
            return 0
        rest = self.read_bits(zeros)
        return ((1 << zeros) | rest) - 1

    def read_signed_golomb(self) -> int:
        return unzigzag(self.read_exp_golomb())


def rice_k(acc: int, count: int) -> int:
    k = 0
    scaled = count
    while scaled < acc and k < 16:
        scaled <<= 1
        k += 1
    return k


def _iter_valid(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(valid)
    return ys, xs


def encode_rice_values(
    values: np.ndarray,
    valid: np.ndarray,
    context: np.ndarray | None,
    causal_mu: bool,
) -> bytes:
    """Code signed ``values`` on ``valid`` sites in row-major order."""
    writer = BitWriter()
    acc = [4] * 8
    count = [1] * 8
    ys, xs = _iter_valid(valid)
    prev = 0
    for y, x in zip(ys.tolist(), xs.tolist()):
        ctx = 0 if context is None else int(context[y, x])
        if ctx < 0 or ctx > 7:
            ctx = 7
        mu = prev if causal_mu else 0
        err = int(values[y, x]) - mu
        symbol = zigzag(err)
        k = rice_k(acc[ctx], count[ctx])
        writer.write_rice(symbol, k)
        acc[ctx] += symbol
        count[ctx] += 1
        if count[ctx] >= 64:
            acc[ctx] = max(1, acc[ctx] >> 1)
            count[ctx] >>= 1
        prev = int(values[y, x])
    return writer.finish()


def decode_rice_values(
    data: bytes,
    shape: tuple[int, int],
    valid: np.ndarray,
    context: np.ndarray | None,
    causal_mu: bool,
) -> np.ndarray:
    reader = BitReader(data)
    acc = [4] * 8
    count = [1] * 8
    out = np.zeros(shape, np.int32)
    ys, xs = _iter_valid(valid)
    prev = 0
    for y, x in zip(ys.tolist(), xs.tolist()):
        ctx = 0 if context is None else int(context[y, x])
        if ctx < 0 or ctx > 7:
            ctx = 7
        mu = prev if causal_mu else 0
        k = rice_k(acc[ctx], count[ctx])
        symbol = reader.read_rice(k)
        err = unzigzag(symbol)
        value = err + mu
        out[y, x] = value
        acc[ctx] += symbol
        count[ctx] += 1
        if count[ctx] >= 64:
            acc[ctx] = max(1, acc[ctx] >> 1)
            count[ctx] >>= 1
        prev = value
    return out


def normalize_freqs(
    counts: np.ndarray, nsym: int = CORE_BINS, total: int = RANS_M
) -> np.ndarray:
    """Exact int64 largest-remainder normalization (zero IEEE-754 floats)."""
    base = np.maximum(np.asarray(counts, dtype=np.int64).reshape(nsym), 0) + 1
    mass = int(base.sum())
    numer = base * np.int64(total)
    freqs = np.maximum(numer // mass, 1)
    deficit = int(total - int(freqs.sum()))
    # Exact signed fractional remainder scaled by ``mass``:
    signed_rem = numer - freqs * np.int64(mass)
    if deficit > 0:
        order = np.argsort(-signed_rem, kind="stable")
        for i in range(deficit):
            freqs[int(order[i % nsym])] += 1
    elif deficit < 0:
        order = np.argsort(signed_rem, kind="stable")
        left = -deficit
        while left > 0:
            progressed = False
            for raw_idx in order.tolist():
                if freqs[raw_idx] > 1:
                    freqs[raw_idx] -= 1
                    left -= 1
                    progressed = True
                    if left == 0:
                        break
            if not progressed:
                raise CodecError("frequency normalization failed")
    if int(freqs.sum()) != total or np.any(freqs < 1):
        raise CodecError("frequency table is not a full partition")
    return freqs.astype(np.int32)


def _normalize_masses_exact(masses: np.ndarray, total: int = RANS_M) -> np.ndarray:
    """Normalize positive integer CDF bin masses to ``total`` with ``f_k >= 1``."""
    nsym = int(masses.size)
    base = np.maximum(masses.astype(np.int64, copy=False), 1)
    mass = int(base.sum())
    numer = base * np.int64(total)
    freqs = np.maximum(numer // mass, 1)
    deficit = int(total - int(freqs.sum()))
    signed_rem = numer - freqs * np.int64(mass)
    if deficit > 0:
        order = np.argsort(-signed_rem, kind="stable")
        for i in range(deficit):
            freqs[int(order[i % nsym])] += 1
    elif deficit < 0:
        order = np.argsort(signed_rem, kind="stable")
        left = -deficit
        while left > 0:
            progressed = False
            for raw_idx in order.tolist():
                if freqs[raw_idx] > 1:
                    freqs[raw_idx] -= 1
                    left -= 1
                    progressed = True
                    if left == 0:
                        break
            if not progressed:
                raise CodecError("frequency normalization failed")
    return freqs.astype(np.int32)


def cumulative(freqs: np.ndarray) -> np.ndarray:
    cum = np.zeros(len(freqs) + 1, np.int32)
    cum[1:] = np.cumsum(freqs, dtype=np.int64).astype(np.int32)
    return cum


class RansEncoder:
    """64-bit rANS, 32-bit renormalization, scale ``2^16``. Symbols are given in reverse order."""

    def __init__(self) -> None:
        self.state = RANS64_L
        self._words: list[int] = []

    def put(self, start: int, freq: int) -> None:
        f = int(freq)
        if f <= 0:
            raise CodecError("rANS frequency must be positive")
        x = self.state
        x_max = ((RANS64_L >> RANS_SCALE) << 32) * f
        while x >= x_max:
            self._words.append(int(x & 0xFFFFFFFF))
            x >>= 32
        self.state = ((x // f) << RANS_SCALE) + (int(start) + (x % f))

    def put_batch(self, starts: list[int], freqs: list[int]) -> None:
        """Encode ``(starts, freqs)`` already ordered from last symbol to first."""
        x = self.state
        words = self._words
        append = words.append
        base_max = (RANS64_L >> RANS_SCALE) << 32
        for s, f in zip(starts, freqs):
            x_max = base_max * f
            while x >= x_max:
                append(x & 0xFFFFFFFF)
                x >>= 32
            q, r = divmod(x, f)
            x = (q << RANS_SCALE) + s + r
        self.state = x

    def finish(self) -> bytes:
        x = self.state
        self._words.append(int(x & 0xFFFFFFFF))
        self._words.append(int((x >> 32) & 0xFFFFFFFF))
        out = bytearray()
        out += int(self._words[-1]).to_bytes(4, "little")
        out += int(self._words[-2]).to_bytes(4, "little")
        for word in reversed(self._words[:-2]):
            out += int(word).to_bytes(4, "little")
        return bytes(out)


class RansDecoder:
    def __init__(self, data: bytes) -> None:
        if len(data) < 8:
            raise CodecError("truncated rANS stream")
        self._data = data
        self._pos = 8
        hi = int.from_bytes(data[0:4], "little")
        lo = int.from_bytes(data[4:8], "little")
        self.state = (hi << 32) | lo

    def cf(self) -> int:
        return int(self.state & (RANS_M - 1))

    def advance(self, start: int, freq: int) -> None:
        x = self.state
        mask = RANS_M - 1
        x = int(freq) * (x >> RANS_SCALE) + (x & mask) - int(start)
        while x < RANS64_L:
            if self._pos + 4 > len(self._data):
                raise CodecError("truncated rANS stream")
            word = int.from_bytes(self._data[self._pos : self._pos + 4], "little")
            self._pos += 4
            x = (x << 32) | word
        self.state = x


class RansBitsBackStack:
    """Bidirectional LIFO rANS word stack for S^3-BB-rANS (Appendix A.3)."""

    def __init__(self, state: int = RANS64_L, words: list[int] | None = None) -> None:
        self.state = int(state)
        self.words: list[int] = [] if words is None else list(words)

    def push(self, start: int, freq: int) -> None:
        f = int(freq)
        if f <= 0:
            raise CodecError("rANS frequency must be positive")
        x = self.state
        x_max = ((RANS64_L >> RANS_SCALE) << 32) * f
        while x >= x_max:
            self.words.append(int(x & 0xFFFFFFFF))
            x >>= 32
        self.state = ((x // f) << RANS_SCALE) + int(start) + (x % f)

    def pop(self, cum: np.ndarray, freqs: np.ndarray) -> int:
        cf = int(self.state & (RANS_M - 1))
        sym = int(np.searchsorted(cum, cf, side="right") - 1)
        if sym < 0 or sym >= len(freqs):
            raise CodecError("bits-back pop symbol out of range")
        f = int(freqs[sym])
        s = int(cum[sym])
        x = f * (self.state >> RANS_SCALE) + cf - s
        while x < RANS64_L:
            if not self.words:
                raise CodecError("bits-back seed stack underflow")
            x = (x << 32) | int(self.words.pop())
        self.state = x
        return sym

    def to_bytes(self) -> bytes:
        out = bytearray()
        out += int((self.state >> 32) & 0xFFFFFFFF).to_bytes(4, "little")
        out += int(self.state & 0xFFFFFFFF).to_bytes(4, "little")
        out += len(self.words).to_bytes(4, "little")
        for w in reversed(self.words):
            out += int(w).to_bytes(4, "little")
        return bytes(out)

    @classmethod
    def from_bytes(cls, data: bytes) -> RansBitsBackStack:
        if len(data) < 12:
            raise CodecError("truncated bits-back rANS stream")
        hi = int.from_bytes(data[0:4], "little")
        lo = int.from_bytes(data[4:8], "little")
        nwords = int.from_bytes(data[8:12], "little")
        if 12 + 4 * nwords > len(data):
            raise CodecError("truncated bits-back word stack")
        words = [
            int.from_bytes(data[12 + 4 * i : 16 + 4 * i], "little")
            for i in range(nwords)
        ]
        words.reverse()
        return cls((hi << 32) | lo, words)


def s3_bb_encode(
    seed_symbols: list[int],
    seed_freqs_fn,
    posterior_fn,
    main_symbols_fn,
    prior_freqs: np.ndarray,
    x_main_ctx,
) -> tuple[bytes, int]:
    """Appendix A.3 Tail-Seeded Split-Lattice Bits-Back rANS encoder.

    Executes:
      1. ``S_0 = Enc(empty, x_seed; Q(. | x_main))``
      2. ``(S_1, z) = Dec(S_0; q_Psi(. | x_main))``
      3. ``S_2 = Enc(S_1, x_main; P(. | z))``
      4. ``S_3 = Enc(S_2, z; P_Z)``
    with identically 0 auxiliary cold-start padding bits.
    """
    stack = RansBitsBackStack()
    q_seed_freqs = seed_freqs_fn(x_main_ctx)
    q_seed_cum = cumulative(q_seed_freqs)
    for sym in reversed(seed_symbols):
        stack.push(int(q_seed_cum[sym]), int(q_seed_freqs[sym]))

    post_freqs = posterior_fn(x_main_ctx)
    post_cum = cumulative(post_freqs)
    z_latent = stack.pop(post_cum, post_freqs)

    main_syms, main_starts, main_freqs = main_symbols_fn(z_latent)
    for start, freq in zip(reversed(main_starts), reversed(main_freqs)):
        stack.push(int(start), int(freq))

    prior_cum = cumulative(prior_freqs)
    stack.push(int(prior_cum[z_latent]), int(prior_freqs[z_latent]))
    return stack.to_bytes(), z_latent


def s3_bb_decode(
    blob: bytes,
    n_main: int,
    n_seed: int,
    seed_freqs_fn,
    posterior_fn,
    decode_main_fn,
    prior_freqs: np.ndarray,
) -> tuple[list[int], list[int], int]:
    """Exact algebraic inverse of :func:`s3_bb_encode`."""
    stack = RansBitsBackStack.from_bytes(blob)
    prior_cum = cumulative(prior_freqs)
    z_latent = stack.pop(prior_cum, prior_freqs)

    main_syms, x_main_ctx = decode_main_fn(stack, z_latent, n_main)

    post_freqs = posterior_fn(x_main_ctx)
    post_cum = cumulative(post_freqs)
    stack.push(int(post_cum[z_latent]), int(post_freqs[z_latent]))

    q_seed_freqs = seed_freqs_fn(x_main_ctx)
    q_seed_cum = cumulative(q_seed_freqs)
    seed_syms = [stack.pop(q_seed_cum, q_seed_freqs) for _ in range(n_seed)]
    if stack.state != RANS64_L or stack.words:
        raise CodecError("bits-back final stack invariant violated")
    return main_syms, seed_syms, z_latent


# Lazy-cached normative parametric Core66 + tilted remainder tables.
_PARAM_CACHE: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None


def get_parametric_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build and cache ``(shifts, freqs, cums, tilt_p0)`` in pure integer math.

    - ``shifts``: ``(32,)`` int32 dyadic shift ``b`` for each ``SIGMA_Q8``
    - ``freqs``: ``(4, 32, 16, 66)`` int32 Core66 frequencies
    - ``cums``: ``(4, 32, 16, 67)`` int32 Core66 cumulative starts
    - ``tilt_p0``: ``(4, 32, 13, 7)`` int32 Q16 ``P(bit == 0)`` for up to 3-bit
      tilted binary remainder trees on ``q in [-6..6]`` when ``b > 0``.
    """
    global _PARAM_CACHE
    if _PARAM_CACHE is not None:
        return _PARAM_CACHE

    shifts = np.array([dyadic_shift_q8(s) for s in SIGMA_Q8], dtype=np.int32)
    freqs = np.empty((NUM_NUS, NUM_SIGMAS, NUM_FRACS, CORE_BINS), dtype=np.int32)
    cums = np.zeros((NUM_NUS, NUM_SIGMAS, NUM_FRACS, CORE_BINS + 1), dtype=np.int32)
    tilt_p0 = np.full(
        (NUM_NUS, NUM_SIGMAS, 2 * TILT_QMAX + 1, (1 << TILT_MAX_DEPTH) - 1),
        RANS_M >> 1,
        dtype=np.int32,
    )

    q_edges = np.arange(-QMAX, QMAX + 2, dtype=np.int64)  # 66 edges for 65 core bins
    frac_offsets_q4 = np.arange(-8, 8, dtype=np.int64)    # delta_mu * 16 in [-8..7]

    for s_idx, sig_q8 in enumerate(SIGMA_Q8):
        b = int(shifts[s_idx])
        bias = (1 << (b - 1)) if b > 0 else 0
        # Exact half-integer bin boundaries in Q4 units:
        # r_tilde boundary is ((q << b) - bias - 0.5), minus delta_mu = frac_q4 / 16:
        # in Q4 units: 16 * ((q << b) - bias) - 8 - frac_q4!
        base_edges_q4 = ((q_edges << b) - bias) * 16 - 8  # (66,)
        # Broadcast across 16 fractional phases: (16, 66)
        all_edges_q4 = base_edges_q4[None, :] - frac_offsets_q4[:, None]
        # Convert Q4 residual divided by Q8 sigma into Q12 z = (edge_q4 * 256 * 4096 / 16) // sig_q8 = (edge_q4 << 16) // sig_q8!
        z_q12 = (all_edges_q4 << 16) // int(sig_q8)

        for nu_idx in range(NUM_NUS):
            cdf_q24 = eval_student_cdf_q24(z_q12, nu_idx)  # (16, 66)
            core_m = np.maximum(cdf_q24[:, 1:] - cdf_q24[:, :-1], 1)  # (16, 65)
            esc_m = np.maximum((1 << 24) - core_m.sum(axis=1, keepdims=True), 1)
            masses = np.concatenate([core_m, esc_m], axis=1)  # (16, 66)
            for f_idx in range(NUM_FRACS):
                f_row = _normalize_masses_exact(masses[f_idx])
                freqs[nu_idx, s_idx, f_idx] = f_row
                cums[nu_idx, s_idx, f_idx, 1:] = np.cumsum(f_row, dtype=np.int64)

            if b > 0:
                depth = min(b, TILT_MAX_DEPTH)
                n_sub = 1 << depth
                sub_step = 1 << (b - depth)
                for qi, q_val in enumerate(range(-TILT_QMAX, TILT_QMAX + 1)):
                    lo_r = (q_val << b) - bias
                    sub_edges_q4 = np.array(
                        [(lo_r + k * sub_step) * 16 - 8 for k in range(n_sub + 1)],
                        dtype=np.int64,
                    )
                    sub_z_q12 = (sub_edges_q4 << 16) // int(sig_q8)
                    sub_cdf = eval_student_cdf_q24(sub_z_q12, nu_idx)
                    sub_mass = np.maximum(sub_cdf[1:] - sub_cdf[:-1], 1)
                    # Populate heap-indexed internal binary tree nodes 0..(2^depth - 2):
                    for lvl in range(depth):
                        span = n_sub >> (lvl + 1)
                        base_node = (1 << lvl) - 1
                        for branch in range(1 << lvl):
                            start_i = branch * 2 * span
                            m0 = int(sub_mass[start_i : start_i + span].sum())
                            m1 = int(sub_mass[start_i + span : start_i + 2 * span].sum())
                            p0 = (m0 * RANS_M + ((m0 + m1) >> 1)) // (m0 + m1)
                            p0 = min(max(int(p0), 1), RANS_M - 1)
                            tilt_p0[nu_idx, s_idx, qi, base_node + branch] = p0

    _PARAM_CACHE = (shifts, freqs, cums, tilt_p0)
    return _PARAM_CACHE


def build_bimodal_row(
    nu_idx: int, sig_q8: int, mode_lo: int, mode_hi: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic integer 2-component bilateral sheaf bridge Core66 row (b == 0)."""
    edges_q4 = np.arange(-QMAX, QMAX + 2, dtype=np.int64) * 16 - 8
    z_lo = ((edges_q4 - int(mode_lo) * 16) << 16) // max(int(sig_q8), 32)
    z_hi = ((edges_q4 - int(mode_hi) * 16) << 16) // max(int(sig_q8), 32)
    cdf_bi = (
        eval_student_cdf_q24(z_lo, nu_idx) + eval_student_cdf_q24(z_hi, nu_idx)
    ) >> 1
    core_m = np.maximum(cdf_bi[1:] - cdf_bi[:-1], 1)
    esc_m = np.maximum((1 << 24) - int(core_m.sum()), 1)
    masses = np.append(core_m, esc_m)
    f_row = _normalize_masses_exact(masses)
    c_row = cumulative(f_row)
    return f_row, c_row


def _symbolize(quotient: int) -> tuple[int, int | None]:
    if abs(quotient) > QMAX:
        return CORE_BINS - 1, quotient
    return quotient + QMAX, None


def _encode_extra(
    writer: BitWriter,
    quotient: int,
    remainder: int | None,
    shift: int,
    escaped: bool,
) -> None:
    if escaped:
        writer.write_bits(1 if quotient < 0 else 0, 1)
        writer.write_exp_golomb(abs(quotient) - (QMAX + 1))
    if shift > 0:
        if remainder is None or remainder < 0 or remainder >= (1 << shift):
            raise CodecError("dyadic remainder out of range")
        writer.write_bits(remainder, shift)


def _decode_extra(
    reader: BitReader, symbol: int, shift: int
) -> tuple[int, int | None]:
    if symbol == CORE_BINS - 1:
        sign = reader.read_bits(1)
        magnitude = reader.read_exp_golomb() + (QMAX + 1)
        quotient = -magnitude if sign else magnitude
    else:
        quotient = symbol - QMAX
    remainder = reader.read_bits(shift) if shift > 0 else None
    return quotient, remainder


def encode_parametric_rans(
    r_tilde: np.ndarray,
    nu_idx: int,
    sig_idx: np.ndarray,
    frac_idx: np.ndarray,
    bridge_mask: np.ndarray | None = None,
    bridge_lo: np.ndarray | None = None,
    bridge_hi: np.ndarray | None = None,
    ext_freq: np.ndarray | None = None,
    ext_cum: np.ndarray | None = None,
) -> bytes:
    """Zero-table-overhead parametric Student-t Dyadic-Core66 rANS encoder.

    Implements Appendix A.4 in full:
    - 65 centered quotients ``[-32..32]`` + escape symbol 65
    - Sub-integer fractional mean phases ``frac_idx in 0..15``
    - Optional 2-component bilateral sheaf bridge on discontinuity sites
    - Depth-``min(b, 3)`` tilted binary rANS tree on ``|q| <= 6`` when ``b > 0``,
      plus raw bypass bits for remaining LSBs and ``|q| > 6``.
    """
    shifts_tbl, freqs_tbl, cums_tbl, tilt_tbl = get_parametric_tables()
    rt = np.asarray(r_tilde, dtype=np.int64).ravel()
    si = np.clip(np.asarray(sig_idx, dtype=np.int32).ravel(), 0, NUM_SIGMAS - 1)
    fi = np.clip(np.asarray(frac_idx, dtype=np.int32).ravel(), 0, NUM_FRACS - 1)
    n = int(rt.size)
    if n == 0:
        return (0).to_bytes(4, "little") + (0).to_bytes(4, "little")

    b_arr = shifts_tbl[si]
    # Vectorized dyadic split across all N samples:
    pos_b = b_arr > 0
    bias_arr = np.where(pos_b, np.int64(1) << np.maximum(b_arr - 1, 0), 0)
    q_arr = np.where(pos_b, (rt + bias_arr) >> b_arr, rt)
    u_arr = np.where(pos_b, rt - (q_arr << b_arr) + bias_arr, 0)
    abs_q = np.abs(q_arr)
    escaped_arr = abs_q > QMAX
    sym_arr = np.where(escaped_arr, CORE_BINS - 1, q_arr + QMAX).astype(np.int32)

    nu = int(nu_idx)
    f_sub = freqs_tbl[nu]  # (32, 16, 66)
    c_sub = cums_tbl[nu]   # (32, 16, 67)
    t_sub = tilt_tbl[nu]   # (32, 13, 7)

    # Vectorized lookup for all standard (non-bridge) symbols:
    if ext_freq is not None or ext_cum is not None:
        if ext_freq is None or ext_cum is None:
            raise CodecError("external Core66 rows must include frequencies and cumulatives")
        ef = np.asarray(ext_freq, dtype=np.int32)
        ec = np.asarray(ext_cum, dtype=np.int32)
        if ef.shape != (n, CORE_BINS) or ec.shape != (n, CORE_BINS + 1):
            raise CodecError("external Core66 rows have the wrong shape")
        if int(ef.min()) < 1 or not np.all(ef.sum(axis=1) == RANS_M):
            raise CodecError("external Core66 rows are not a full partition")
        idx = np.arange(n)
        sym_starts = ec[idx, sym_arr].tolist()
        sym_freqs = ef[idx, sym_arr].tolist()
        has_bridge = False
    else:
        sym_starts = c_sub[si, fi, sym_arr].tolist()
        sym_freqs = f_sub[si, fi, sym_arr].tolist()
        has_bridge = (
            bridge_mask is not None
            and bridge_lo is not None
            and bridge_hi is not None
            and bool(np.any(bridge_mask))
        )
    if has_bridge:
        bm_list = np.asarray(bridge_mask, dtype=np.bool_).ravel().tolist()
        blo_list = np.asarray(bridge_lo, dtype=np.int32).ravel().tolist()
        bhi_list = np.asarray(bridge_hi, dtype=np.int32).ravel().tolist()
        bi_cache: dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray]] = {}
        si_list = si.tolist()
        fi_list = fi.tolist()
        sym_list = sym_arr.tolist()
        b_list_check = b_arr.tolist()
        for idx in range(n):
            if bm_list[idx] and b_list_check[idx] == 0:
                s_v = si_list[idx]
                f_v = fi_list[idx]
                lo_v = max(-28, min(28, blo_list[idx]))
                hi_v = max(-28, min(28, bhi_list[idx]))
                if hi_v - lo_v >= 6:
                    key = (s_v, f_v, lo_v, hi_v)
                    pair = bi_cache.get(key)
                    if pair is None:
                        bi_sig = SIGMA_Q8[max(0, s_v - 3)]
                        f_bi, _ = build_bimodal_row(nu, bi_sig, lo_v, hi_v)
                        f_uni = f_sub[s_v, f_v]
                        f_mix = _normalize_masses_exact(f_uni.astype(np.int64) + f_bi.astype(np.int64))
                        pair = (f_mix, cumulative(f_mix))
                        bi_cache[key] = pair
                    sy = sym_list[idx]
                    sym_freqs[idx] = int(pair[0][sy])
                    sym_starts[idx] = int(pair[1][sy])

    need_extra = bool(np.any(pos_b) or np.any(escaped_arr))
    writer = BitWriter()
    if not need_extra:
        # Fast path when all shifts == 0 and no escapes: 100% pure rANS batch!
        encoder = RansEncoder()
        sym_starts.reverse()
        sym_freqs.reverse()
        encoder.put_batch(sym_starts, sym_freqs)
        rans_bytes = encoder.finish()
        extra_bytes = b""
    else:
        rans_starts: list[int] = []
        rans_freqs: list[int] = []
        q_list = q_arr.tolist()
        u_list = u_arr.tolist()
        b_list = b_arr.tolist()
        si_list = si.tolist()
        esc_list = escaped_arr.tolist()
        for i in range(n):
            rans_starts.append(sym_starts[i])
            rans_freqs.append(sym_freqs[i])
            qv = q_list[i]
            bv = b_list[i]
            if esc_list[i]:
                writer.write_bits(1 if qv < 0 else 0, 1)
                writer.write_exp_golomb(abs(qv) - (QMAX + 1))
            if bv > 0:
                uv = u_list[i]
                if -TILT_QMAX <= qv <= TILT_QMAX:
                    depth = bv if bv < TILT_MAX_DEPTH else TILT_MAX_DEPTH
                    rem_low = bv - depth
                    top_bits = uv >> rem_low
                    qi = qv + TILT_QMAX
                    t_row = t_sub[si_list[i], qi]
                    branch = 0
                    for lvl in range(depth):
                        bit = (top_bits >> (depth - 1 - lvl)) & 1
                        p0 = int(t_row[((1 << lvl) - 1) + branch])
                        if bit == 0:
                            rans_starts.append(0)
                            rans_freqs.append(p0)
                        else:
                            rans_starts.append(p0)
                            rans_freqs.append(RANS_M - p0)
                        branch = (branch << 1) | bit
                    if rem_low > 0:
                        writer.write_bits(uv & ((1 << rem_low) - 1), rem_low)
                else:
                    writer.write_bits(uv, bv)

        encoder = RansEncoder()
        rans_starts.reverse()
        rans_freqs.reverse()
        encoder.put_batch(rans_starts, rans_freqs)
        rans_bytes = encoder.finish()
        extra_bytes = writer.finish()

    out = bytearray()
    out += len(rans_bytes).to_bytes(4, "little")
    out += len(extra_bytes).to_bytes(4, "little")
    out += rans_bytes
    out += extra_bytes
    return bytes(out)


def decode_parametric_rans(
    data: bytes,
    count: int,
    nu_idx: int,
    sig_idx: np.ndarray,
    frac_idx: np.ndarray,
    bridge_mask: np.ndarray | None = None,
    bridge_lo: np.ndarray | None = None,
    bridge_hi: np.ndarray | None = None,
    ext_freq: np.ndarray | None = None,
    ext_cum: np.ndarray | None = None,
) -> np.ndarray:
    """Exact pure-integer inverse of :func:`encode_parametric_rans`."""
    if len(data) < 8:
        raise CodecError("truncated parametric rANS header")
    rans_n = int.from_bytes(data[0:4], "little")
    extra_n = int.from_bytes(data[4:8], "little")
    if 8 + rans_n + extra_n > len(data):
        raise CodecError("truncated parametric rANS payload")
    if count == 0:
        return np.zeros(0, dtype=np.int32)

    shifts_tbl, freqs_tbl, cums_tbl, tilt_tbl = get_parametric_tables()
    si = np.clip(np.asarray(sig_idx, dtype=np.int32).ravel(), 0, NUM_SIGMAS - 1)
    fi = np.clip(np.asarray(frac_idx, dtype=np.int32).ravel(), 0, NUM_FRACS - 1)
    b_arr = shifts_tbl[si]

    nu = int(nu_idx)
    f_sub = freqs_tbl[nu]
    c_sub = cums_tbl[nu]
    t_sub = tilt_tbl[nu]

    raw_rans = data[8 : 8 + rans_n]
    if len(raw_rans) < 8:
        raise CodecError("truncated rANS stream")
    x_state = (int.from_bytes(raw_rans[0:4], "little") << 32) | int.from_bytes(
        raw_rans[4:8], "little"
    )
    r_pos = 8
    r_end = len(raw_rans)
    rans_l = RANS64_L
    rans_mask = RANS_M - 1
    rans_scale = RANS_SCALE
    from_bytes = int.from_bytes

    reader = BitReader(data[8 + rans_n : 8 + rans_n + extra_n])

    if ext_freq is not None or ext_cum is not None:
        if ext_freq is None or ext_cum is None:
            raise CodecError("external Core66 rows must include frequencies and cumulatives")
        ef = np.asarray(ext_freq, dtype=np.int32)
        ec = np.asarray(ext_cum, dtype=np.int32)
        if ef.shape != (count, CORE_BINS) or ec.shape != (count, CORE_BINS + 1):
            raise CodecError("external Core66 rows have the wrong shape")
        ext_f_lists = ef.tolist()
        ext_c_lists = ec.tolist()
        has_bridge = False
    else:
        ext_f_lists = None
        ext_c_lists = None
        has_bridge = (
            bridge_mask is not None
            and bridge_lo is not None
            and bridge_hi is not None
            and bool(np.any(bridge_mask))
        )
    bm_list = (
        np.asarray(bridge_mask, dtype=np.bool_).ravel().tolist() if has_bridge else None
    )
    blo_list = (
        np.asarray(bridge_lo, dtype=np.int32).ravel().tolist() if has_bridge else None
    )
    bhi_list = (
        np.asarray(bridge_hi, dtype=np.int32).ravel().tolist() if has_bridge else None
    )
    bi_cache: dict[tuple[int, int, int, int], tuple[list[int], list[int]]] = {}
    f_lists = f_sub.tolist()
    c_lists = c_sub.tolist()
    t_lists = t_sub.tolist()

    out = [0] * count
    si_list = si.tolist()
    fi_list = fi.tolist()
    b_list = b_arr.tolist()
    # 65536-entry alias tables remove bisect from the zero-shift inner loop.
    lut_cache: dict[tuple, np.ndarray] = {}

    def _alias(s_v: int, f_v: int, c_row: list[int]) -> np.ndarray:
        key = (s_v, f_v)
        cached = lut_cache.get(key)
        if cached is not None:
            return cached
        table = np.empty(RANS_M, dtype=np.uint8)
        for sym_i in range(len(c_row) - 1):
            start_i = c_row[sym_i]
            stop_i = c_row[sym_i + 1]
            if stop_i > start_i:
                table[start_i:stop_i] = sym_i
        lut_cache[key] = table
        return table

    if not has_bridge and not bool(np.any(b_arr > 0)) and ext_f_lists is None:
        # Fast path: zero shifts and no bridge sites (pure integer inner loop).
        for i in range(count):
            s_v = si_list[i]
            f_v = fi_list[i]
            c_row = c_lists[s_v][f_v]
            f_row = f_lists[s_v][f_v]
            cf = x_state & rans_mask
            sym = int(_alias(s_v, f_v, c_row)[cf])
            if sym < 0 or sym >= CORE_BINS:
                raise CodecError("parametric rANS symbol out of range")
            x_state = f_row[sym] * (x_state >> rans_scale) + cf - c_row[sym]
            while x_state < rans_l:
                if r_pos + 4 > r_end:
                    raise CodecError("truncated rANS stream")
                x_state = (x_state << 32) | from_bytes(raw_rans[r_pos : r_pos + 4], "little")
                r_pos += 4
            if sym == CORE_BINS - 1:
                sign = reader.read_bits(1)
                mag = reader.read_exp_golomb() + (QMAX + 1)
                out[i] = -mag if sign else mag
            else:
                out[i] = sym - QMAX
        return np.asarray(out, dtype=np.int32)

    for i in range(count):
        s_v = si_list[i]
        f_v = fi_list[i]
        bv = b_list[i]
        if ext_f_lists is not None and ext_c_lists is not None:
            f_row = ext_f_lists[i]
            c_row = ext_c_lists[i]
        else:
            f_row = f_lists[s_v][f_v]
            c_row = c_lists[s_v][f_v]

        if (
            bm_list is not None
            and blo_list is not None
            and bhi_list is not None
            and bm_list[i]
            and bv == 0
        ):
            lo_v = max(-28, min(28, blo_list[i]))
            hi_v = max(-28, min(28, bhi_list[i]))
            if hi_v - lo_v >= 6:
                key = (s_v, f_v, lo_v, hi_v)
                pair = bi_cache.get(key)
                if pair is None:
                    bi_sig = SIGMA_Q8[max(0, s_v - 3)]
                    f_bi, _ = build_bimodal_row(nu, bi_sig, lo_v, hi_v)
                    f_mix = _normalize_masses_exact(
                        f_sub[s_v, f_v].astype(np.int64) + f_bi.astype(np.int64)
                    )
                    pair = (f_mix.tolist(), cumulative(f_mix).tolist())
                    bi_cache[key] = pair
                f_row, c_row = pair

        cf = x_state & rans_mask
        sym = bisect_right(c_row, cf) - 1
        if sym < 0 or sym >= CORE_BINS:
            raise CodecError("parametric rANS symbol out of range")
        x_state = f_row[sym] * (x_state >> rans_scale) + cf - c_row[sym]
        while x_state < rans_l:
            if r_pos + 4 > r_end:
                raise CodecError("truncated rANS stream")
            x_state = (x_state << 32) | from_bytes(raw_rans[r_pos : r_pos + 4], "little")
            r_pos += 4

        if sym == CORE_BINS - 1:
            sign = reader.read_bits(1)
            mag = reader.read_exp_golomb() + (QMAX + 1)
            qv = -mag if sign else mag
        else:
            qv = sym - QMAX

        if bv == 0:
            out[i] = qv
        else:
            if -TILT_QMAX <= qv <= TILT_QMAX:
                depth = bv if bv < TILT_MAX_DEPTH else TILT_MAX_DEPTH
                rem_low = bv - depth
                t_row = t_lists[s_v][qv + TILT_QMAX]
                branch = 0
                for lvl in range(depth):
                    p0 = t_row[((1 << lvl) - 1) + branch]
                    cf_b = x_state & rans_mask
                    if cf_b < p0:
                        bit = 0
                        x_state = p0 * (x_state >> rans_scale) + cf_b
                    else:
                        bit = 1
                        x_state = (RANS_M - p0) * (x_state >> rans_scale) + cf_b - p0
                    while x_state < rans_l:
                        if r_pos + 4 > r_end:
                            raise CodecError("truncated rANS stream")
                        x_state = (x_state << 32) | from_bytes(
                            raw_rans[r_pos : r_pos + 4], "little"
                        )
                        r_pos += 4
                    branch = (branch << 1) | bit
                low_bits = reader.read_bits(rem_low) if rem_low > 0 else 0
                uv = (branch << rem_low) | low_bits
            else:
                uv = reader.read_bits(bv)
            bias = 1 << (bv - 1)
            out[i] = (qv << bv) - bias + uv
    return np.asarray(out, dtype=np.int32)


class ParametricDecoder:
    """Pull-style inverse of :func:`encode_parametric_rans` for causal predictors.

    Context (sigma, fractional mean, optional bridge) may depend on previously
    reconstructed samples. Symbols are pulled in the same order the batch
    decoder consumes them, so the bytes match :func:`decode_parametric_rans`.
    """

    def __init__(self, data: bytes, nu_idx: int) -> None:
        if len(data) < 8:
            raise CodecError("truncated parametric rANS header")
        rans_n = int.from_bytes(data[0:4], "little")
        extra_n = int.from_bytes(data[4:8], "little")
        if 8 + rans_n + extra_n > len(data):
            raise CodecError("truncated parametric rANS payload")
        shifts_tbl, freqs_tbl, cums_tbl, tilt_tbl = get_parametric_tables()
        self._shifts = shifts_tbl
        nu = int(nu_idx)
        self._nu = nu
        self._f_sub = freqs_tbl[nu]
        self._c_sub = cums_tbl[nu]
        self._t_sub = tilt_tbl[nu]
        self._f_lists = self._f_sub.tolist()
        self._c_lists = self._c_sub.tolist()
        self._t_lists = self._t_sub.tolist()
        raw = data[8 : 8 + rans_n]
        if len(raw) < 8:
            raise CodecError("truncated rANS stream")
        self._raw = raw
        self._x = (int.from_bytes(raw[0:4], "little") << 32) | int.from_bytes(raw[4:8], "little")
        self._pos = 8
        self._reader = BitReader(data[8 + rans_n : 8 + rans_n + extra_n])
        self._bi: dict[tuple[int, int, int, int], tuple[list[int], list[int]]] = {}

    def _renorm(self, x: int) -> int:
        raw = self._raw
        pos = self._pos
        while x < RANS64_L:
            if pos + 4 > len(raw):
                raise CodecError("truncated rANS stream")
            x = (x << 32) | int.from_bytes(raw[pos : pos + 4], "little")
            pos += 4
        self._pos = pos
        return x

    def read(
        self,
        sig_i: int,
        frac_i: int,
        bridge: tuple[int, int] | None = None,
    ) -> int:
        s_v = int(sig_i) if 0 <= int(sig_i) < NUM_SIGMAS else 0
        f_v = int(frac_i) if 0 <= int(frac_i) < NUM_FRACS else 0
        bv = int(self._shifts[s_v])
        f_row = self._f_lists[s_v][f_v]
        c_row = self._c_lists[s_v][f_v]
        if bridge is not None and bv == 0:
            lo_v = max(-28, min(28, int(bridge[0])))
            hi_v = max(-28, min(28, int(bridge[1])))
            if hi_v - lo_v >= 6:
                key = (s_v, f_v, lo_v, hi_v)
                pair = self._bi.get(key)
                if pair is None:
                    bi_sig = SIGMA_Q8[max(0, s_v - 3)]
                    f_bi, _ = build_bimodal_row(self._nu, bi_sig, lo_v, hi_v)
                    f_mix = _normalize_masses_exact(
                        self._f_sub[s_v, f_v].astype(np.int64) + f_bi.astype(np.int64)
                    )
                    pair = (f_mix.tolist(), cumulative(f_mix).tolist())
                    self._bi[key] = pair
                f_row, c_row = pair
        x = self._x
        cf = x & (RANS_M - 1)
        sym = bisect_right(c_row, cf) - 1
        if sym < 0 or sym >= CORE_BINS:
            raise CodecError("parametric rANS symbol out of range")
        x = f_row[sym] * (x >> RANS_SCALE) + cf - c_row[sym]
        x = self._renorm(x)
        if sym == CORE_BINS - 1:
            sign = self._reader.read_bits(1)
            mag = self._reader.read_exp_golomb() + (QMAX + 1)
            qv = -mag if sign else mag
        else:
            qv = sym - QMAX
        if bv == 0:
            self._x = x
            return int(qv)
        if -TILT_QMAX <= qv <= TILT_QMAX:
            depth = bv if bv < TILT_MAX_DEPTH else TILT_MAX_DEPTH
            rem_low = bv - depth
            t_row = self._t_lists[s_v][qv + TILT_QMAX]
            branch = 0
            for lvl in range(depth):
                p0 = t_row[((1 << lvl) - 1) + branch]
                cf_b = x & (RANS_M - 1)
                if cf_b < p0:
                    bit = 0
                    x = p0 * (x >> RANS_SCALE) + cf_b
                else:
                    bit = 1
                    x = (RANS_M - p0) * (x >> RANS_SCALE) + cf_b - p0
                x = self._renorm(x)
                branch = (branch << 1) | bit
            low_bits = self._reader.read_bits(rem_low) if rem_low > 0 else 0
            uv = (branch << rem_low) | low_bits
        else:
            uv = self._reader.read_bits(bv)
        bias = 1 << (bv - 1)
        self._x = x
        return int((qv << bv) - bias + uv)


def encode_dyadic_rans(
    residuals: np.ndarray,
    valid: np.ndarray,
    shifts: np.ndarray,
) -> bytes:
    """Two-pass empirical Dyadic-Core66 rANS of signed residuals."""
    ys, xs = _iter_valid(valid)
    symbols: list[int] = []
    extras: list[tuple[int, int | None, int, bool]] = []
    counts = np.zeros(CORE_BINS, np.int64)
    for y, x in zip(ys.tolist(), xs.tolist()):
        shift = int(shifts[y, x])
        quotient, remainder = split_dyadic(int(residuals[y, x]), shift)
        symbol, escaped_q = _symbolize(quotient)
        symbols.append(symbol)
        counts[symbol] += 1
        extras.append((quotient, remainder, shift, escaped_q is not None))
    freqs = normalize_freqs(counts)
    cum = cumulative(freqs)
    encoder = RansEncoder()
    for symbol in reversed(symbols):
        encoder.put(int(cum[symbol]), int(freqs[symbol]))
    rans_bytes = encoder.finish()
    writer = BitWriter()
    for quotient, remainder, shift, escaped in extras:
        _encode_extra(writer, quotient, remainder, shift, escaped)
    extra = writer.finish()
    out = bytearray()
    for freq in freqs.tolist():
        out += int(freq).to_bytes(2, "little")
    out += len(rans_bytes).to_bytes(4, "little")
    out += len(extra) .to_bytes(4, "little")
    out += rans_bytes
    out += extra
    return bytes(out)


def decode_dyadic_rans(
    data: bytes,
    shape: tuple[int, int],
    valid: np.ndarray,
    shifts: np.ndarray,
) -> np.ndarray:
    if len(data) < CORE_BINS * 2 + 8:
        raise CodecError("truncated dyadic rANS header")
    freqs = np.empty(CORE_BINS, np.int32)
    pos = 0
    for i in range(CORE_BINS):
        freqs[i] = int.from_bytes(data[pos : pos + 2], "little")
        pos += 2
    if int(freqs.sum()) != RANS_M or np.any(freqs < 1):
        raise CodecError("invalid dyadic frequency table")
    rans_n = int.from_bytes(data[pos : pos + 4], "little")
    pos += 4
    extra_n = int.from_bytes(data[pos : pos + 4], "little")
    pos += 4
    if pos + rans_n + extra_n > len(data):
        raise CodecError("truncated dyadic payload")
    rans_bytes = data[pos : pos + rans_n]
    extra = data[pos + rans_n : pos + rans_n + extra_n]
    cum = cumulative(freqs)
    ys, xs = _iter_valid(valid)
    nsym = int(ys.size)
    decoder = RansDecoder(rans_bytes)
    symbols = [0] * nsym
    cum_list = cum.tolist()
    for i in range(nsym):
        cf = decoder.cf()
        symbol = int(np.searchsorted(cum, cf, side="right") - 1)
        if symbol < 0 or symbol >= CORE_BINS:
            raise CodecError("rANS symbol out of range")
        if not (cum_list[symbol] <= cf < cum_list[symbol + 1]):
            raise CodecError("rANS cumulative mismatch")
        decoder.advance(int(cum[symbol]), int(freqs[symbol]))
        symbols[i] = symbol
    reader = BitReader(extra)
    out = np.zeros(shape, np.int32)
    for i, (y, x) in enumerate(zip(ys.tolist(), xs.tolist())):
        shift = int(shifts[y, x])
        quotient, remainder = _decode_extra(reader, symbols[i], shift)
        out[y, x] = join_dyadic(quotient, remainder, shift)
    return out


def shift_map_from_activity(activity_ctx: np.ndarray) -> np.ndarray:
    """Map curvature activity buckets ``0..7`` to conservative ``sigma`` shifts."""
    table = np.array([1, 2, 3, 5, 10, 20, 40, 80], np.int32)
    shifts = np.zeros(activity_ctx.shape, np.uint8)
    for bucket, sig in enumerate(table.tolist()):
        shifts[activity_ctx == bucket] = dyadic_shift(int(sig))
    return shifts
