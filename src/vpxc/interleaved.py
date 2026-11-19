"""32-lane interleaved rANS, pure integer, matching the optional native kernel.

Symbols in decode order are striped across 32 independent 64-bit states
(``lane = index mod 32``). The byte layout is 32 concatenations of the scalar
``RansEncoder`` framing, so a lane can be checked against ``RansEncoder``
directly. This layout is not the default tile coder: the per-lane headers cost
more than one shared state, and the encoder keeps whichever stream is smaller.
"""

from __future__ import annotations

from bisect import bisect_right

import numpy as np

from vpxc.entropy import RANS64_L, RANS_M, RansDecoder, RansEncoder
from vpxc.errors import CodecError

LANES = 32


def encode_pairs(pairs: list[tuple[int, int]]) -> bytes:
    """Encode ``(start, freq)`` pairs given in decode order."""
    lanes = [RansEncoder() for _ in range(LANES)]
    for index in range(len(pairs) - 1, -1, -1):
        start, freq = pairs[index]
        lanes[index % LANES].put(int(start), int(freq))
    out = bytearray()
    for lane in lanes:
        blob = lane.finish()
        out += len(blob).to_bytes(4, "little")
        out += blob
    return bytes(out)


def decode_pairs(
    data: bytes,
    count: int,
    cum: np.ndarray,
    freqs: np.ndarray,
) -> list[int]:
    """Inverse of :func:`encode_pairs` for one shared frequency row."""
    if len(cum) != len(freqs) + 1:
        raise CodecError("interleaved cumulative length mismatch")
    pos = 0
    decoders: list[RansDecoder] = []
    for _lane in range(LANES):
        if pos + 4 > len(data):
            raise CodecError("truncated interleaved rANS header")
        n = int.from_bytes(data[pos : pos + 4], "little")
        pos += 4
        if pos + n > len(data):
            raise CodecError("truncated interleaved rANS lane")
        decoders.append(RansDecoder(data[pos : pos + n]))
        pos += n
    if pos != len(data):
        raise CodecError("trailing bytes after interleaved rANS")
    cum_list = cum.tolist()
    freq_list = freqs.tolist()
    out = [0] * count
    for index in range(count):
        decoder = decoders[index % LANES]
        cf = decoder.cf()
        sym = bisect_right(cum_list, cf) - 1
        if sym < 0 or sym >= len(freq_list):
            raise CodecError("interleaved symbol out of range")
        decoder.advance(int(cum_list[sym]), int(freq_list[sym]))
        out[index] = sym
    return out


def encode_symbols(symbols: list[int], freqs: np.ndarray, cum: np.ndarray) -> bytes:
    pairs = [(int(cum[sym]), int(freqs[sym])) for sym in symbols]
    return encode_pairs(pairs)


def symbol_lut(cum: np.ndarray) -> np.ndarray:
    """65536-entry alias table. ``lut[cf]`` is the symbol whose bin contains ``cf``."""
    lut = np.empty(RANS_M, dtype=np.uint8)
    for sym in range(len(cum) - 1):
        start = int(cum[sym])
        stop = int(cum[sym + 1])
        if stop < start or stop > RANS_M:
            raise CodecError("cumulative table is not ordered")
        if stop > start:
            lut[start:stop] = sym
    return lut
