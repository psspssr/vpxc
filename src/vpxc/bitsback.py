"""Tile-level S^3-BB-rANS coupling for a sub-coset of residuals.

Seed sites are pushed under the parametric Core66 distribution, a 2-bit latent
is popped from that stack, the main sites are pushed under a latent-shifted
scale, and the latent is pushed under a fixed prior. Cold-start padding is not
added: if the seed stack underflows, the caller keeps the ordinary stream.
Decode is the exact stack inverse and uses no floating-point operations.
"""

from __future__ import annotations

import numpy as np

from vpxc.entropy import (
    NUM_SIGMAS,
    BitReader,
    BitWriter,
    RansBitsBackStack,
    cumulative,
    get_parametric_tables,
    normalize_freqs,
)
from vpxc.errors import CodecError

_PRIOR = normalize_freqs(np.array([2, 8, 2, 1], dtype=np.int64), nsym=4)


def _pairs(
    residuals: np.ndarray,
    sig: np.ndarray,
    frac: np.ndarray,
    nu: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int | None]]]:
    """Parametric (start, freq) pairs plus raw remainder/escape side records."""
    shifts, freqs, cums, _tilt = get_parametric_tables()
    starts: list[tuple[int, int]] = []
    extras: list[tuple[int, int, int | None]] = []
    for residual, sigma, phase in zip(residuals.tolist(), sig.tolist(), frac.tolist()):
        s_v = int(sigma)
        f_v = int(phase)
        if s_v < 0 or s_v >= NUM_SIGMAS or f_v < 0 or f_v >= 16:
            raise CodecError("bits-back context out of range")
        bv = int(shifts[s_v])
        if bv == 0:
            quot = int(residual)
            rem = 0
        else:
            bias = 1 << (bv - 1)
            quot = (int(residual) + bias) >> bv
            rem = int(residual) - (quot << bv) + bias
        if abs(quot) > 32:
            sym = 65
            esc: int | None = quot
        else:
            sym = quot + 32
            esc = None
        starts.append((int(cums[nu, s_v, f_v, sym]), int(freqs[nu, s_v, f_v, sym])))
        extras.append((bv, rem, esc))
    return starts, extras


def _write_extras(writer: BitWriter, extras: list[tuple[int, int, int | None]]) -> None:
    for bv, rem, esc in extras:
        if esc is not None:
            writer.write_bits(1 if esc < 0 else 0, 1)
            writer.write_exp_golomb(abs(esc) - 33)
        if bv > 0:
            writer.write_bits(int(rem), bv)


def _read_residual(reader: BitReader, sym: int, bv: int) -> int:
    if sym == 65:
        sign = reader.read_bits(1)
        quot = reader.read_exp_golomb() + 33
        if sign:
            quot = -quot
    else:
        quot = sym - 32
    if bv <= 0:
        return int(quot)
    rem = reader.read_bits(bv)
    bias = 1 << (bv - 1)
    return (int(quot) << bv) - bias + rem


def _posterior(abs_sum: int, count: int) -> np.ndarray:
    avg = abs_sum // max(count, 1)
    if avg < 2:
        hat = 0
    elif avg < 6:
        hat = 1
    elif avg < 16:
        hat = 2
    else:
        hat = 3
    counts = np.ones(4, dtype=np.int64)
    counts[hat] = 12
    return normalize_freqs(counts, nsym=4)


def _abs_quotients(residuals: np.ndarray, sig: np.ndarray) -> int:
    shifts, _f, _c, _t = get_parametric_tables()
    total = 0
    for residual, sigma in zip(residuals.tolist(), sig.tolist()):
        bv = int(shifts[int(sigma)])
        if bv == 0:
            quot = int(residual)
        else:
            quot = (int(residual) + (1 << (bv - 1))) >> bv
        total += abs(quot)
    return total


def encode_s3(
    residuals: np.ndarray,
    sig_idx: np.ndarray,
    frac_idx: np.ndarray,
    nu_idx: int,
    seed_mask: np.ndarray,
) -> bytes | None:
    """Bits-back encode. ``None`` when the seed cannot carry the latent."""
    resid = np.asarray(residuals, dtype=np.int64)
    sig = np.asarray(sig_idx, dtype=np.int32)
    frac = np.asarray(frac_idx, dtype=np.int32)
    seed = np.asarray(seed_mask, dtype=bool).ravel()
    if int(seed.sum()) < 8 or int((~seed).sum()) < 32:
        return None
    nu = int(nu_idx) & 3
    seed_pairs, seed_extra = _pairs(resid[seed], sig[seed], frac[seed], nu)
    try:
        stack = RansBitsBackStack()
        for start, freq in reversed(seed_pairs):
            stack.push(start, freq)
        # Provisional posterior from the known main residuals at the unshifted scale.
        # The latent then biases the main scale; the posterior itself stays a
        # function of the main samples after that bias, recomputed below.
        main = resid[~seed]
        main_sig0 = sig[~seed]
        post = _posterior(_abs_quotients(main, main_sig0), int(main.size))
        z = stack.pop(cumulative(post), post)
        shifted = np.clip(main_sig0.astype(np.int32) + (int(z) - 1), 0, NUM_SIGMAS - 1)
        main_pairs, main_extra = _pairs(main, shifted, frac[~seed], nu)
        for start, freq in reversed(main_pairs):
            stack.push(start, freq)
        # Posterior must match what decode recomputes from the shifted main.
        post_z = _posterior(_abs_quotients(main, shifted), int(main.size))
        if not np.array_equal(post_z, post):
            # The popped latent was drawn under the unshifted posterior. Rebuild
            # from scratch under the shifted posterior so encode and decode agree.
            return _encode_consistent(resid, sig, frac, nu, seed, post_z)
        prior_c = cumulative(_PRIOR)
        stack.push(int(prior_c[z]), int(_PRIOR[z]))
    except CodecError:
        return None
    writer = BitWriter()
    _write_extras(writer, main_extra)
    _write_extras(writer, seed_extra)
    body = stack.to_bytes()
    extra = writer.finish()
    out = bytearray()
    out += len(body).to_bytes(4, "little")
    out += body
    out += extra
    return bytes(out)


def _encode_consistent(resid, sig, frac, nu, seed, post) -> bytes | None:
    """Encode when the posterior is defined on the latent-shifted main scale.

    The posterior depends on the shifted main, which depends on ``z``, and ``z``
    is popped from the seed under that same posterior. We search the four
    latents: the pop is a pure function of the seed bits and the posterior, so
    at most one ``z`` is consistent, and it is exactly what decode will read.
    """
    seed_pairs, seed_extra = _pairs(resid[seed], sig[seed], frac[seed], nu)
    try:
        stack = RansBitsBackStack()
        for start, freq in reversed(seed_pairs):
            stack.push(start, freq)
        z = stack.pop(cumulative(post), post)
    except CodecError:
        return None
    main = resid[~seed]
    shifted = np.clip(sig[~seed].astype(np.int32) + (int(z) - 1), 0, NUM_SIGMAS - 1)
    if not np.array_equal(_posterior(_abs_quotients(main, shifted), int(main.size)), post):
        return None
    main_pairs, main_extra = _pairs(main, shifted, frac[~seed], nu)
    for start, freq in reversed(main_pairs):
        stack.push(start, freq)
    prior_c = cumulative(_PRIOR)
    stack.push(int(prior_c[z]), int(_PRIOR[z]))
    writer = BitWriter()
    _write_extras(writer, main_extra)
    _write_extras(writer, seed_extra)
    body = stack.to_bytes()
    extra = writer.finish()
    return len(body).to_bytes(4, "little") + body + extra


def decode_s3(
    data: bytes,
    count: int,
    sig_idx: np.ndarray,
    frac_idx: np.ndarray,
    nu_idx: int,
    seed_mask: np.ndarray,
) -> np.ndarray:
    """Exact inverse of :func:`encode_s3`."""
    if len(data) < 4:
        raise CodecError("truncated bits-back residual stream")
    body_n = int.from_bytes(data[0:4], "little")
    if 4 + body_n > len(data):
        raise CodecError("truncated bits-back body")
    # The stack serializer stores its own length; pass only that slice.
    stack = RansBitsBackStack.from_bytes(data[4 : 4 + body_n])
    extra = data[4 + body_n :]
    sig = np.asarray(sig_idx, dtype=np.int32).ravel()
    frac = np.asarray(frac_idx, dtype=np.int32).ravel()
    seed = np.asarray(seed_mask, dtype=bool).ravel()
    if int(sig.size) != count or int(seed.size) != count:
        raise CodecError("bits-back context length mismatch")
    nu = int(nu_idx) & 3
    main_sig0 = sig[~seed]
    # Decode does not know the main residuals yet, but the posterior used at
    # pop time is the one encode committed. Encode only emits a stream when the
    # shifted posterior equals the posterior that was popped, and that posterior
    # is recomputed from the shifted main after ``z`` is known. Chicken-and-egg:
    # ``z`` is coded under the prior at the top of the stack, so pop the prior
    # first (that is the decode order of S^3), then form the shifted posterior.
    prior_c = cumulative(_PRIOR)
    z = stack.pop(prior_c, _PRIOR)
    shifted = np.clip(main_sig0.astype(np.int32) + (int(z) - 1), 0, NUM_SIGMAS - 1)
    reader = BitReader(extra)
    shifts, freqs, cums, _tilt = get_parametric_tables()
    main_frac = frac[~seed]
    main_out = np.empty(int((~seed).sum()), dtype=np.int64)
    for i, (s_v, f_v) in enumerate(zip(shifted.tolist(), main_frac.tolist())):
        row_f = freqs[nu, int(s_v), int(f_v)]
        row_c = cums[nu, int(s_v), int(f_v)]
        sym = stack.pop(row_c, row_f)
        main_out[i] = _read_residual(reader, sym, int(shifts[int(s_v)]))
    post = _posterior(_abs_quotients(main_out, shifted), int(main_out.size))
    post_c = cumulative(post)
    stack.push(int(post_c[z]), int(post[z]))
    seed_sig = sig[seed]
    seed_frac = frac[seed]
    seed_out = np.empty(int(seed.sum()), dtype=np.int64)
    for i, (s_v, f_v) in enumerate(zip(seed_sig.tolist(), seed_frac.tolist())):
        row_f = freqs[nu, int(s_v), int(f_v)]
        row_c = cums[nu, int(s_v), int(f_v)]
        sym = stack.pop(row_c, row_f)
        seed_out[i] = _read_residual(reader, sym, int(shifts[int(s_v)]))
    if stack.state != (1 << 31) or stack.words:
        raise CodecError("bits-back final stack invariant violated")
    out = np.empty(count, dtype=np.int32)
    out[np.flatnonzero(seed)] = seed_out
    out[np.flatnonzero(~seed)] = main_out
    return out
