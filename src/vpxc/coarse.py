"""Coarse-band ``y_4`` coder: predictor bank, cross-channel ridge, Core66 rANS.

Replaces scalar adaptive Rice on the packed base band. The walk is causal and
identical on encode and decode. Encode may fit Q8 weights in floating point;
decode applies those integers with a pull-style rANS decoder and never uses
IEEE-754 arithmetic.
"""

from __future__ import annotations

import numpy as np

from vpxc.entropy import (
    BitReader,
    BitWriter,
    ParametricDecoder,
    encode_parametric_rans,
)
from vpxc.errors import CodecError
from vpxc.mdl import fit_ridge_q8, fit_sigma_nu
from vpxc.spatial import (
    _boundary_grad,
    _d8_patch_key,
    _estimate_noise_var,
    _kalman_fuse,
    _median,
)

_N_BUCKETS = 16


def _bucket16(grad: int) -> int:
    if grad <= 0:
        return 0
    bit = int(grad).bit_length() - 1
    return bit if bit < _N_BUCKETS else _N_BUCKETS - 1


def _grab(plane: np.ndarray, valid: np.ndarray, y: int, x: int) -> tuple[int, bool]:
    h, w = plane.shape
    if y < 0 or x < 0 or y >= h or x >= w or not bool(valid[y, x]):
        return 0, False
    return int(plane[y, x]), True


def _gap(
    left: int,
    up: int,
    ul: int,
    ur: int,
    ll: int,
    nn: int,
    have_l: bool,
    have_u: bool,
    have_ul: bool,
    have_ur: bool,
    have_ll: bool,
    have_nn: bool,
) -> int | None:
    if not (have_l and have_u):
        return None
    west_west = ll if have_ll else left
    dh = abs(left - west_west)
    dv = 0
    if have_ul:
        dh += abs(up - ul)
        dv += abs(left - ul)
    if have_ur:
        dh += abs(up - ur)
    if have_nn:
        dv += abs(up - nn)
    if dv - dh > 80:
        return left
    if dh - dv > 80:
        return up
    pred = (left + up) // 2
    if have_ur and have_ul:
        pred += (ur - ul) // 4
    if dv - dh > 32:
        pred = (pred + left) // 2
    elif dh - dv > 32:
        pred = (pred + up) // 2
    return int(pred)


def _split_mu(mu_q4: int) -> tuple[int, int]:
    mu_int = (mu_q4 + 8) >> 4
    delta = mu_q4 - (mu_int << 4)
    frac = delta + 8
    if frac < 0:
        frac = 0
    elif frac > 15:
        frac = 15
    return int(mu_int), int(frac)


def _site(
    plane: np.ndarray,
    valid: np.ndarray,
    ref: np.ndarray | None,
    y: int,
    x: int,
    table: dict,
    s_eta2: int,
    weights: np.ndarray | None,
) -> tuple[int, int, int, tuple | None, int, list[int]]:
    """Return ``(mu_int, frac, bucket, kalman_key, ul, feats)`` for one causal site."""
    left, have_l = _grab(plane, valid, y, x - 1)
    ll, have_ll = _grab(plane, valid, y, x - 2)
    up, have_u = _grab(plane, valid, y - 1, x)
    ul, have_ul = _grab(plane, valid, y - 1, x - 1)
    ur, have_ur = _grab(plane, valid, y - 1, x + 1)
    nn, have_nn = _grab(plane, valid, y - 2, x)
    if not have_ul:
        ul = left if have_l else up
    med = _median(left, up, ul, have_l, have_u, have_ul)
    grad = _boundary_grad(
        left, up, ul, ur, ll, have_l, have_u, have_ul, have_ur, have_ll
    )
    west = left if have_l else med
    north = up if have_u else med
    gap = _gap(left, up, ul, ur, ll, nn, have_l, have_u, have_ul, have_ur, have_ll, have_nn)
    if gap is None:
        gap = med
    kalman = med
    eff = grad
    key = None
    if have_l or have_u:
        d1 = left - ul
        d2 = up - ul
        d3 = (ur - up) if abs(d1) <= abs(d2) else (left - ll)
        key, _reflected = _d8_patch_key(d1, d2, d3)
        kalman, eff = _kalman_fuse(med, ul, table.get(key, []), grad, s_eta2)
    feats = [
        (west - med) << 4,
        (north - med) << 4,
        (gap - med) << 4,
        (kalman - med) << 4,
    ]
    if ref is not None:
        r_left, r_l = _grab(ref, valid, y, x - 1)
        r_up, r_u = _grab(ref, valid, y - 1, x)
        r_ul, r_ul_ok = _grab(ref, valid, y - 1, x - 1)
        if not r_ul_ok:
            r_ul = r_left if r_l else r_up
        ref_med = _median(r_left, r_up, r_ul, r_l, r_u, r_ul_ok)
        here, here_ok = _grab(ref, valid, y, x)
        feats.append(((here - ref_med) << 4) if here_ok else 0)
        down, d_ok = _grab(ref, valid, y + 1, x)
        right, rt_ok = _grab(ref, valid, y, x + 1)
        if here_ok and r_u and d_ok and r_l and rt_ok:
            lap = 4 * here - r_up - down - r_left - right
        else:
            lap = 0
        feats.append(int(lap) << 2)
    corr = 0
    if weights is not None:
        if len(weights) != len(feats):
            raise CodecError("coarse ridge width does not match the predictor bank")
        acc = 128
        for feat, weight in zip(feats, weights.tolist()):
            acc += int(feat) * int(weight)
        corr = acc >> 8
    mu_int, frac = _split_mu((med << 4) + int(corr))
    return mu_int, frac, _bucket16(eff), key, ul, feats


def _commit(table: dict, key, value: int, ul: int) -> None:
    if key is None:
        return
    delta = int(value) - int(ul)
    bucket = table.get(key)
    if bucket is None:
        table[key] = [delta]
        return
    if len(bucket) >= 8:
        del bucket[0]
    bucket.append(delta)


def _valid_sites(valid: np.ndarray) -> list[tuple[int, int]]:
    ys, xs = np.nonzero(valid)
    return list(zip(ys.tolist(), xs.tolist()))


def encode_coarse(
    plane: np.ndarray,
    valid: np.ndarray,
    ref: np.ndarray | None = None,
) -> bytes:
    """Encode one packed coarse plane with the 4-predictor bank and Core66 rANS."""
    sites = _valid_sites(valid)
    writer = BitWriter()
    if not sites:
        writer.write_bits(0, 8)
        writer.write_bits(1, 2)
        writer.write_bits(0, 1)
        writer.write_bits(4, 5)
        hdr = writer.finish()
        payload = encode_parametric_rans(
            np.zeros(0, np.int32), 1, np.zeros(0, np.int32), np.zeros(0, np.int32)
        )
        return len(hdr).to_bytes(2, "little") + hdr + payload

    s_eta2 = _estimate_noise_var(plane, valid) if min(plane.shape) >= 3 else 0
    table = {}
    base_mu: list[int] = []
    values: list[int] = []
    feat_rows: list[list[int]] = []
    for y, x in sites:
        mu_int, _frac, _bucket, key, ul_k, feats = _site(
            plane, valid, ref, y, x, table, s_eta2, None
        )
        base_mu.append(mu_int)
        values.append(int(plane[y, x]))
        feat_rows.append(feats)
        _commit(table, key, int(plane[y, x]), ul_k)

    phi = np.asarray(feat_rows, dtype=np.int64)
    # Weight-free ``mu_int`` is MED, so this is the Q4 error of that base predictor.
    target = (np.asarray(values, dtype=np.int64) - np.asarray(base_mu, dtype=np.int64)) << 4
    weights = fit_ridge_q8(phi, target, min_count=48)

    table = {}
    frac_i: list[int] = []
    buck_i: list[int] = []
    resid: list[int] = []
    for y, x in sites:
        mu_int, frac, bucket, key, ul_k, _feats = _site(
            plane, valid, ref, y, x, table, s_eta2, weights
        )
        frac_i.append(frac)
        buck_i.append(bucket)
        resid.append(int(plane[y, x]) - mu_int)
        _commit(table, key, int(plane[y, x]), ul_k)

    r_arr = np.asarray(resid, dtype=np.int32)
    f_arr = np.asarray(frac_i, dtype=np.int32)
    b_arr = np.asarray(buck_i, dtype=np.int32)
    nu, sig_map = fit_sigma_nu(r_arr, f_arr, b_arr, _N_BUCKETS)
    sig_idx = sig_map[b_arr]
    payload = encode_parametric_rans(r_arr, nu, sig_idx, f_arr)

    writer.write_bits(int(s_eta2) & 255, 8)
    writer.write_bits(int(nu) & 3, 2)
    writer.write_bits(1 if weights is not None else 0, 1)
    if weights is not None:
        writer.write_bits(len(weights), 4)
        for value in weights.tolist():
            writer.write_signed_golomb(int(value))
    first = int(sig_map[0]) & 31
    writer.write_bits(first, 5)
    prev = first
    for b_i in range(1, _N_BUCKETS):
        cur = int(sig_map[b_i]) & 31
        writer.write_signed_golomb(cur - prev)
        prev = cur
    hdr = writer.finish()
    return len(hdr).to_bytes(2, "little") + hdr + payload


def decode_coarse(
    data: bytes,
    shape: tuple[int, int],
    valid: np.ndarray,
    ref: np.ndarray | None = None,
) -> np.ndarray:
    """Exact integer inverse of :func:`encode_coarse`."""
    if len(data) < 2:
        raise CodecError("truncated coarse header")
    hdr_n = int.from_bytes(data[0:2], "little")
    if 2 + hdr_n > len(data):
        raise CodecError("truncated coarse header")
    reader = BitReader(data[2 : 2 + hdr_n])
    payload = data[2 + hdr_n :]
    s_eta2 = reader.read_bits(8)
    nu = reader.read_bits(2)
    use_w = reader.read_bits(1) == 1
    weights = None
    if use_w:
        n_w = reader.read_bits(4)
        weights = np.array([reader.read_signed_golomb() for _ in range(n_w)], dtype=np.int64)
    sig_map = np.empty(_N_BUCKETS, dtype=np.int32)
    prev = reader.read_bits(5)
    sig_map[0] = prev
    for b_i in range(1, _N_BUCKETS):
        cur = prev + reader.read_signed_golomb()
        if cur < 0 or cur > 31:
            raise CodecError("coarse sigma index out of range")
        sig_map[b_i] = cur
        prev = cur

    out = np.zeros(shape, dtype=np.int32)
    sites = _valid_sites(valid)
    if not sites:
        return out
    decoder = ParametricDecoder(payload, nu)
    table: dict = {}
    for y, x in sites:
        mu_int, frac, bucket, key, ul_k, _feats = _site(
            out, valid, ref, y, x, table, s_eta2, weights
        )
        residual = decoder.read(int(sig_map[bucket]), frac)
        value = mu_int + residual
        out[y, x] = value
        _commit(table, key, value, ul_k)
    return out
