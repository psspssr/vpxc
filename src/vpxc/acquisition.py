"""Pre-gamma acquisition header ``phi_acq``.

An 8-knot monotone inverse-tone LUT (96 bits) and ``4 x 24`` Q4.6
parity-conditioned taps are applied only inside the causal chroma predictor.
The pixel lattice itself is not resampled, so the lifting bijection is
unchanged and decode stays on integer LUT interpolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from vpxc.entropy import BitReader, BitWriter
from vpxc.errors import CodecError

OFFSETS = tuple(
    (dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if (dy, dx) != (0, 0)
)
assert len(OFFSETS) == 24
_KNOT_X = np.array([0, 36, 73, 109, 146, 182, 219, 255], dtype=np.int64)
_GAMMAS = (256, 410, 461, 512, 563, 614)  # 1.0 .. 2.4 in Q8


@dataclass(frozen=True)
class PhiAcq:
    shift: int
    knots: np.ndarray  # (8,) uint12 linear-light codes
    taps: np.ndarray  # (4, 24) Q6


def lut_from_knots(knots: np.ndarray) -> np.ndarray:
    """Integer monotone interpolant of the 8 stored knots over ``0..255``.

    Slopes are the integer PCHIP weights of the knot values, then each segment
    is clamped to its endpoint range so the LUT cannot overshoot.
    """
    y = np.asarray(knots, dtype=np.int64).reshape(8)
    x = _KNOT_X
    delta = y[1:] - y[:-1]
    h = x[1:] - x[:-1]
    # Segment slopes in "dy per dx", stored as dy (PCHIP harmonic mean of
    # neighboring secants) scaled back onto the following sample step of 1.
    secant = (delta << 8) // np.maximum(h, 1)
    slope = np.zeros(8, dtype=np.int64)
    for i in range(1, 7):
        s0 = int(secant[i - 1])
        s1 = int(secant[i])
        if s0 > 0 and s1 > 0:
            slope[i] = (2 * s0 * s1 + ((s0 + s1) >> 1)) // (s0 + s1)
        else:
            slope[i] = 0
    slope[0] = int(secant[0])
    slope[7] = int(secant[-1])
    lut = np.empty(256, dtype=np.int64)
    for seg in range(7):
        x0 = int(x[seg])
        x1 = int(x[seg + 1])
        y0 = int(y[seg])
        y1 = int(y[seg + 1])
        span = x1 - x0
        lo = y0 if y0 < y1 else y1
        hi = y1 if y1 > y0 else y0
        m0 = int(slope[seg])
        m1 = int(slope[seg + 1])
        last = x1 if seg == 6 else x1 - 1
        for tpos in range(x0, last + 1):
            t = ((tpos - x0) << 8) // span  # 0..256
            # Hermite in Q8, then clamp into the knot range.
            dy = y1 - y0
            c2 = 3 * dy - ((2 * m0 + m1) * span >> 8)
            c3 = ((m0 + m1) * span >> 8) - 2 * dy
            p1 = (c3 * t) // 256 + c2
            p2 = (p1 * t) // 256 + ((m0 * span) >> 8)
            val = y0 + (p2 * t) // 256
            if val < lo:
                val = lo
            elif val > hi:
                val = hi
            lut[tpos] = val
    return lut


def inverse_lut(lut: np.ndarray) -> np.ndarray:
    """Right-inverse of a monotone ``0..255 -> 0..4095`` LUT, length 4096."""
    inv = np.full(4096, 255, dtype=np.int32)
    pos = 0
    for i in range(256):
        end = int(lut[i])
        if end < 0:
            end = 0
        if end > 4095:
            end = 4095
        if end >= pos:
            inv[pos : end + 1] = i
            pos = end + 1
    if pos < 4096:
        inv[pos:] = 255
    return inv


def apply_phi(luma: np.ndarray, phi: PhiAcq) -> np.ndarray:
    """Integer predicted sample field for one chroma plane, from decoded luma."""
    lut = lut_from_knots(phi.knots)
    inv = inverse_lut(lut)
    idx = np.clip(np.asarray(luma, dtype=np.int64) >> int(phi.shift), 0, 255).astype(np.int32)
    lin = lut[idx]
    padded = np.pad(lin, 2, mode="edge")
    height, width = luma.shape
    rr, cc = np.indices((height, width))
    phase = ((rr & 1) << 1) | (cc & 1)
    acc = np.zeros((height, width), dtype=np.int64)
    taps = np.asarray(phi.taps, dtype=np.int64)
    for k, (dy, dx) in enumerate(OFFSETS):
        neigh = padded[2 + dy : 2 + dy + height, 2 + dx : 2 + dx + width]
        weight = taps[:, k][phase]
        acc += weight * neigh
    pred_lin = np.clip(acc >> 6, 0, 4095).astype(np.int32)
    pred_idx = inv[pred_lin]
    return pred_idx.astype(np.int64) << int(phi.shift)


def _knots_for_gamma(gamma_q8: int) -> np.ndarray:
    pos = _KNOT_X.astype(np.float64) / 255.0
    gamma = float(gamma_q8) / 256.0
    values = np.round((pos ** gamma) * 4095.0).astype(np.int64)
    values[0] = 0
    values = np.maximum.accumulate(values)
    if int(values[-1]) < 1:
        values[-1] = 1
    return values


def write_phi(writer: BitWriter, phi: PhiAcq) -> None:
    writer.write_bits(int(phi.shift) & 15, 4)
    for knot in np.asarray(phi.knots).tolist():
        writer.write_bits(int(knot) & 4095, 12)
    for row in np.asarray(phi.taps).reshape(-1).tolist():
        writer.write_signed_golomb(int(row))


def read_phi(reader: BitReader) -> PhiAcq:
    shift = reader.read_bits(4)
    knots = np.array([reader.read_bits(12) for _ in range(8)], dtype=np.int64)
    for i in range(7):
        if int(knots[i]) > int(knots[i + 1]):
            raise CodecError("acquisition knots are not monotone")
    taps = np.empty((4, 24), dtype=np.int32)
    for i in range(4 * 24):
        value = reader.read_signed_golomb()
        if abs(value) > 4096:
            raise CodecError("acquisition tap out of range")
        taps[i // 24, i % 24] = value
    return PhiAcq(shift=shift, knots=knots, taps=taps)


def fit_phi(
    luma: np.ndarray,
    chroma: np.ndarray,
    flat_pred: np.ndarray,
    odd: np.ndarray,
) -> PhiAcq | None:
    """Return a gated ``phi_acq`` or ``None`` when the header does not pay."""
    if int(odd.sum()) < 2048:
        return None
    shift = max(0, int(max(int(luma.max()), int(chroma.max()))).bit_length() - 8)
    samples = chroma[odd].astype(np.int64)
    base = flat_pred[odd].astype(np.int64)
    err0 = np.abs(samples - base)
    mad0 = float(err0.mean()) + 1.0
    best: tuple[float, PhiAcq] | None = None
    rr, cc = np.indices(luma.shape)
    phase = ((rr & 1) << 1) | (cc & 1)
    phase_odd = phase[odd]
    for gamma in _GAMMAS:
        knots = _knots_for_gamma(gamma)
        lut = lut_from_knots(knots)
        idx = np.clip(luma.astype(np.int64) >> shift, 0, 255).astype(np.int32)
        lin = lut[idx]
        padded = np.pad(lin, 2, mode="edge")
        height, width = luma.shape
        design = np.zeros((int(odd.sum()), 96), dtype=np.float64)
        target = lut[np.clip(chroma.astype(np.int64) >> shift, 0, 255).astype(np.int32)][odd]
        col = 0
        for p in range(4):
            sel = phase_odd == p
            for dy, dx in OFFSETS:
                neigh = padded[2 + dy : 2 + dy + height, 2 + dx : 2 + dx + width][odd]
                if np.any(sel):
                    design[sel, col] = neigh[sel]
                col += 1
        coef, *_rest = np.linalg.lstsq(design, target.astype(np.float64), rcond=None)
        taps = np.clip(np.round(coef * 64.0), -512, 511).astype(np.int32).reshape(4, 24)
        # Design columns are linear-light samples. Taps are Q6, so a coefficient
        # of 1.0 is stored as 64 and the integer dot shifts back by 6.
        phi = PhiAcq(shift=shift, knots=knots, taps=taps)
        pred = apply_phi(luma, phi)[odd]
        mad1 = float(np.abs(samples - pred).mean()) + 1.0
        if best is None or mad1 < best[0]:
            best = (mad1, phi)
    if best is None:
        return None
    mad1, phi = best
    gain = float(samples.size) * (math.log2(mad0) - math.log2(mad1))
    # 4 + 96 knot bits + a pessimistic 8 bits per tap.
    if gain <= 4 + 96 + 96 * 8 + 32:
        return None
    return phi
