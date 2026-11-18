"""Dyadic quotient coding: adaptive Rice and Dyadic-Core66 rANS.

Both coders turn a signed residual into a quotient and a ``b``-bit remainder.
Rice writes the quotient in unary (escape to Exp-Golomb past 32) and adapts the
shift from the mean absolute quotient in each context. rANS writes a 66-symbol
quotient alphabet (bins ``[-32, 32]`` plus one escape) under a 16-bit
frequency table, which is the zero-weight Dyadic-Core66 stage in the spec.
"""

from __future__ import annotations

import numpy as np

from vpxc.errors import CodecError

RICE_ESCAPE = 32
RANS64_L = 1 << 31
RANS_SCALE = 16
RANS_M = 1 << RANS_SCALE
CORE_BINS = 66  # 65 centered quotients + escape
QMAX = 32


def zigzag(err: int) -> int:
    if err < 0:
        return ((-err) << 1) - 1
    return err << 1


def unzigzag(symbol: int) -> int:
    if symbol & 1:
        return -((symbol + 1) >> 1)
    return symbol >> 1


def dyadic_shift(activity: int) -> int:
    """``b = max(0, ceil(log2(sigma / 2.5)))`` with ``sigma = activity``.

    Integer form: smallest ``b`` in ``0..12`` such that ``5 * 2^b >= 2 * sigma``.
    Matches the audit table: 0.8→0, 2.5→0, 8→2, 25→4, 120→6 when ``sigma`` is
    that value (integers use the same threshold).
    """
    sigma = int(activity)
    if sigma <= 2:
        return 0
    b = 0
    acc = 5
    two_sigma = sigma * 2
    while acc < two_sigma and b < 12:
        acc <<= 1
        b += 1
    return b


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


def rice_k(acc: int, count: int) -> int:
    k = 0
    scaled = count
    while scaled < acc and k < 16:
        scaled <<= 1
        k += 1
    return k


def _iter_valid(valid: np.ndarray):
    ys, xs = np.nonzero(valid)
    return ys, xs


def encode_rice_values(
    values: np.ndarray,
    valid: np.ndarray,
    context: np.ndarray | None,
    causal_mu: bool,
) -> bytes:
    """Code signed ``values`` on ``valid`` sites in row-major order.

    ``context`` selects one of 8 adaptive Rice states. When it is ``None``,
    context 0 is used for every site (the caller is expected to be a raster
    coder that passes per-pixel contexts another way — this helper is for
    already-predicted residuals).
    """
    writer = BitWriter()
    acc = [16] * 8
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
    acc = [16] * 8
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


def normalize_freqs(counts: np.ndarray, nsym: int = CORE_BINS, total: int = RANS_M) -> np.ndarray:
    """Largest-remainder normalization. Every symbol gets frequency at least 1."""
    base = np.asarray(counts, dtype=np.int64).reshape(nsym) + 1
    mass = int(base.sum())
    freqs = (base * total) // mass
    freqs = np.maximum(freqs, 1)
    # Largest remainder on the fractional part ``base/mass - freqs/total``.
    deficit = int(total - int(freqs.sum()))
    frac = base / mass - freqs / total
    if deficit > 0:
        order = np.argsort(-frac, kind="stable")
        for i in range(deficit):
            freqs[int(order[i % nsym])] += 1
    elif deficit < 0:
        order = np.argsort(frac, kind="stable")
        left = -deficit
        cursor = 0
        while left > 0:
            idx = int(order[cursor % nsym])
            cursor += 1
            if freqs[idx] > 1:
                freqs[idx] -= 1
                left -= 1
            if cursor > nsym * 4 and left > 0:
                raise CodecError("frequency normalization failed")
    if int(freqs.sum()) != total or np.any(freqs < 1):
        raise CodecError("frequency table is not a full partition")
    return freqs.astype(np.int32)


def cumulative(freqs: np.ndarray) -> np.ndarray:
    cum = np.zeros(len(freqs) + 1, np.int32)
    running = 0
    for i, freq in enumerate(freqs.tolist()):
        cum[i] = running
        running += int(freq)
    cum[-1] = running
    return cum


class RansEncoder:
    """64-bit rANS, 32-bit renormalization, scale ``2^16``. Symbols are given in forward order."""

    def __init__(self) -> None:
        self.state = RANS64_L
        self._words: list[int] = []

    def put(self, start: int, freq: int) -> None:
        if freq <= 0:
            raise CodecError("rANS frequency must be positive")
        x = self.state
        x_max = ((RANS64_L >> RANS_SCALE) << 32) * int(freq)
        while x >= x_max:
            self._words.append(int(x & 0xFFFFFFFF))
            x >>= 32
        self.state = ((x // int(freq)) << RANS_SCALE) + (int(start) + (x % int(freq)))

    def finish(self) -> bytes:
        x = self.state
        self._words.append(int(x & 0xFFFFFFFF))
        self._words.append(int((x >> 32) & 0xFFFFFFFF))
        # Encoder emits backward. The flush (final state) occupies the front of
        # the byte stream; renormalization words follow in reverse emission order.
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
        lo = int.from_bytes(data[0:4], "little")
        hi = int.from_bytes(data[4:8], "little")
        # finish() wrote high word then low word at the front — see note in finish.
        # Match finish(): words[-1] is the high half, words[-2] is the low half,
        # written in that order. So byte 0 is the HIGH half.
        self.state = (lo << 32) | hi

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


def _symbolize(quotient: int) -> tuple[int, int | None]:
    if abs(quotient) > QMAX:
        return CORE_BINS - 1, quotient
    return quotient + QMAX, None


def _encode_extra(writer: BitWriter, quotient: int, remainder: int | None, shift: int, escaped: bool) -> None:
    if escaped:
        writer.write_bits(1 if quotient < 0 else 0, 1)
        writer.write_exp_golomb(abs(quotient) - (QMAX + 1))
    if shift > 0:
        if remainder is None or remainder < 0 or remainder >= (1 << shift):
            raise CodecError("dyadic remainder out of range")
        writer.write_bits(remainder, shift)


def _decode_extra(reader: BitReader, symbol: int, shift: int) -> tuple[int, int | None]:
    if symbol == CORE_BINS - 1:
        sign = reader.read_bits(1)
        magnitude = reader.read_exp_golomb() + (QMAX + 1)
        quotient = -magnitude if sign else magnitude
    else:
        quotient = symbol - QMAX
    remainder = reader.read_bits(shift) if shift > 0 else None
    return quotient, remainder


def encode_dyadic_rans(
    residuals: np.ndarray,
    valid: np.ndarray,
    shifts: np.ndarray,
) -> bytes:
    """Two-pass Dyadic-Core66 rANS of signed residuals.

    ``shifts`` is the per-pixel dyadic shift ``b``, already known to the decoder
    from causal context. The frequency table is stored in front of the stream.
    """
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
    out += len(extra).to_bytes(4, "little")
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
    # Linear search is fine: 66 bins, and searchsorted is used instead.
    cum_list = cum.tolist()
    for i in range(nsym):
        cf = decoder.cf()
        # cum is nondecreasing; rightmost start <= cf.
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
    """Map a log-activity bucket back to a dyadic shift.

    Bucket ``k`` means activity was in ``[2^k, 2^{k+1})`` (bucket 0 also holds
    activity 0 and 1). Use the top of the bucket as a conservative ``sigma``.
    """
    # sigma representative: 1, 2, 4, 8, 16, 32, 64, 128 for buckets 0..7
    table = np.array([1, 3, 6, 12, 24, 48, 96, 192], np.int32)
    sigma = table[np.clip(activity_ctx, 0, 7)]
    shifts = np.zeros(activity_ctx.shape, np.uint8)
    # Vectorized copy of dyadic_shift for the 8 possible sigmas.
    for bucket, sig in enumerate(table.tolist()):
        shifts[activity_ctx == bucket] = dyadic_shift(int(sig))
    return shifts
