"""Causal raster coder: Median Edge Detection plus D8-equivariant Sheaf-Kalman.

Implements Section 3.3 Eq. (4) of the SLGF specification on causal raster scans:
- Fixes boundary gradient estimation on top row / left column so missing
  neighbors do not pollute high-activity contexts with ``|sample - 0|``.
- Canonicalizes causal 4-pixel L-patches across dihedral symmetries ``g in D8``
  modulo local DC offset ``beta_m`` (the ``Z^2 x| (D8 x Aff_1)`` patch bundle).
- Adapts noise variance ``sigma_eta^2 >= 0`` per plane so noiseless synthetic /
  UI content collapses ``sigma_eta^2 -> 0`` into exact D8 patch copy while
  noisy photographs fuse up to ``M <= 8`` matches at rate ``O(1/M)`` and shrink
  predictive scale ``sigma_pred^2 = sigma_eta^2 + 1 / (sigma_loc^-2 + M sigma_eta^-2)``.
"""

from __future__ import annotations

import math

import numpy as np

from vpxc.entropy import BitReader, BitWriter, rice_k, unzigzag, zigzag

MATCHES = 8


def _bucket(grad: int) -> int:
    if grad <= 0:
        return 0
    bit = grad.bit_length() - 1
    return bit if bit < 7 else 7


def _median(
    left: int, up: int, ul: int, have_l: bool, have_u: bool, have_ul: bool
) -> int:
    if not have_l and not have_u:
        return 0
    if not have_l:
        return up
    if not have_u:
        return left
    if not have_ul:
        return (left + up) >> 1
    if left > up:
        hi, lo = left, up
    else:
        hi, lo = up, left
    pred = left + up - ul
    if ul < lo:
        return hi
    if ul > hi:
        return lo
    return pred


def _estimate_noise_var(plane: np.ndarray, valid: np.ndarray) -> int:
    """Estimate integer sensor shot-noise variance ``sigma_eta^2`` in ``0..255``."""
    if plane.shape[0] < 3 or plane.shape[1] < 3:
        return 0
    v_mid = (
        valid[1:-1, 1:-1]
        & valid[:-2, 1:-1]
        & valid[2:, 1:-1]
        & valid[1:-1, :-2]
        & valid[1:-1, 2:]
    )
    if not np.any(v_mid):
        return 0
    p = plane.astype(np.int64, copy=False)
    lap = (
        4 * p[1:-1, 1:-1]
        - p[:-2, 1:-1]
        - p[2:, 1:-1]
        - p[1:-1, :-2]
        - p[1:-1, 2:]
    )[v_mid]
    # For independent noise var s^2, Var(4x - N - S - W - E) = 20 s^2;
    # MAD of Gaussian is ~0.6745 * std => std_lap = MAD / 0.6745, s^2 ≈ MAD^2 / 9.1.
    mad = int(np.median(np.abs(lap)))
    if mad == 0:
        return 0
    est = (mad * mad + 4) // 9
    return min(max(est, 0), 255)


def _d8_patch_key(
    d1: int, d2: int, d3: int
) -> tuple[tuple[int, int, int], bool]:
    """Canonicalize causal corner differences modulo dihedral reflection ``(L <-> U)``.

    Around target ``X``, ``(L - UL, U - UL, UR - U)`` and its transposed
    reflection ``(U - UL, L - UL, LL - L)`` form the causal projection of D8
    symmetries modulo affine DC offset ``beta_m = UL``.
    """
    q1 = d1 >> 2
    q2 = d2 >> 2
    q3 = d3 >> 2
    if q1 <= q2:
        return (q1, q2, q3), False
    return (q2, q1, q3), True


def _kalman_fuse(
    med: int,
    base_ref: int,
    matches: list[int],
    grad: int,
    s_eta2: int,
) -> tuple[int, int]:
    """Return ``(mu_sk, effective_grad)`` under Section 3.3 Eq. (4)."""
    if not matches:
        return med, grad
    # Mahalanobis / shot-noise distance gate ||y_m - mu_loc|| <= gate.
    noise_std = math.isqrt(s_eta2) if s_eta2 > 0 else 0
    gate = max(6, 2 * grad + 3 * noise_std + 2)
    accepted = [base_ref + delta for delta in matches if abs((base_ref + delta) - med) <= gate]
    if not accepted:
        return med, grad

    m_cnt = len(accepted)
    if s_eta2 == 0:
        # Exact noiseless limit (sigma_eta -> 0): collapse to D8 affine match copy!
        copy_val = accepted[-1]
        agreement = max(abs(v - copy_val) for v in accepted)
        eff_grad = min(grad, agreement)
        return int(copy_val), int(eff_grad)

    s_loc2 = grad * grad + 4
    acc = med * s_eta2 + s_loc2 * sum(accepted)
    denom = s_eta2 + m_cnt * s_loc2
    mu_sk = int((2 * acc + denom) // (2 * denom))
    # Posterior variance Eq. (4): s_pred^2 = s_eta^2 + (s_loc^2 * s_eta^2) / (s_eta^2 + M * s_loc^2)
    s_pred2 = s_eta2 + (s_loc2 * s_eta2 + (denom >> 1)) // denom
    # Map posterior variance back to integer gradient scale for Rice context selection.
    eff_grad = min(grad, math.isqrt(s_pred2))
    return mu_sk, eff_grad


def _boundary_grad(
    left: int,
    up: int,
    ul: int,
    ur: int,
    ll: int,
    have_l: bool,
    have_u: bool,
    have_ul: bool,
    have_ur: bool,
    have_ll: bool,
) -> int:
    if have_l and have_u and have_ul:
        g = abs(left - ul) + abs(up - ul)
        if have_ur:
            g = (2 * g + abs(up - ur)) // 2
        return g
    if have_l and have_u:
        return abs(left - up)
    if have_l:
        return abs(left - ll) if have_ll else 0
    if have_u:
        return abs(up - ur) if have_ur else 0
    return 0


def encode_raster(plane: np.ndarray, valid: np.ndarray, kalman: bool) -> bytes:
    rows = plane.astype(np.int32, copy=False).tolist()
    vrows = valid.tolist()
    height = len(rows)
    width = len(rows[0]) if height else 0
    writer = BitWriter()
    s_eta2 = _estimate_noise_var(plane, valid) if kalman else 0
    if kalman:
        writer.write_bits(s_eta2, 8)
    acc = [4] * 8
    count = [1] * 8
    table: dict[tuple[int, int, int], list[int]] = {}
    for y in range(height):
        row = rows[y]
        vrow = vrows[y]
        up_row = rows[y - 1] if y else None
        up_valid = vrows[y - 1] if y else None
        for x in range(width):
            if not vrow[x]:
                continue
            have_l = x > 0 and vrow[x - 1]
            have_ll = x > 1 and vrow[x - 2]
            have_u = up_valid is not None and up_valid[x]
            have_ul = up_valid is not None and x > 0 and up_valid[x - 1]
            have_ur = up_valid is not None and x + 1 < width and up_valid[x + 1]
            left = row[x - 1] if have_l else 0
            ll = row[x - 2] if have_ll else left
            up = up_row[x] if have_u and up_row is not None else 0
            ul = up_row[x - 1] if have_ul and up_row is not None else (left if have_l else up)
            ur = up_row[x + 1] if have_ur and up_row is not None else up

            pred = _median(left, up, ul, have_l, have_u, have_ul)
            grad = _boundary_grad(
                left, up, ul, ur, ll, have_l, have_u, have_ul, have_ur, have_ll
            )
            eff_grad = grad
            key = None
            if kalman and (have_l or have_u):
                d1 = left - ul
                d2 = up - ul
                d3 = (ur - up) if abs(d1) <= abs(d2) else (left - ll)
                key, _reflected = _d8_patch_key(d1, d2, d3)
                pred, eff_grad = _kalman_fuse(
                    pred, ul, table.get(key, []), grad, s_eta2
                )

            err = row[x] - pred
            symbol = zigzag(err)
            ctx = _bucket(eff_grad)
            k = rice_k(acc[ctx], count[ctx])
            writer.write_rice(symbol, k)
            acc[ctx] += symbol
            count[ctx] += 1
            if count[ctx] >= 64:
                acc[ctx] = max(1, acc[ctx] >> 1)
                count[ctx] >>= 1
            if kalman and key is not None:
                delta = row[x] - ul
                bucket = table.get(key)
                if bucket is None:
                    table[key] = [delta]
                else:
                    if len(bucket) >= MATCHES:
                        del bucket[0]
                    bucket.append(delta)
    return writer.finish()


def decode_raster(
    data: bytes, shape: tuple[int, int], valid: np.ndarray, kalman: bool
) -> np.ndarray:
    height, width = shape
    vrows = valid.tolist()
    rows = [[0] * width for _ in range(height)]
    reader = BitReader(data)
    s_eta2 = reader.read_bits(8) if kalman else 0
    acc = [4] * 8
    count = [1] * 8
    table: dict[tuple[int, int, int], list[int]] = {}
    for y in range(height):
        row = rows[y]
        vrow = vrows[y]
        up_row = rows[y - 1] if y else None
        up_valid = vrows[y - 1] if y else None
        for x in range(width):
            if not vrow[x]:
                continue
            have_l = x > 0 and vrow[x - 1]
            have_ll = x > 1 and vrow[x - 2]
            have_u = up_valid is not None and up_valid[x]
            have_ul = up_valid is not None and x > 0 and up_valid[x - 1]
            have_ur = up_valid is not None and x + 1 < width and up_valid[x + 1]
            left = row[x - 1] if have_l else 0
            ll = row[x - 2] if have_ll else left
            up = up_row[x] if have_u and up_row is not None else 0
            ul = up_row[x - 1] if have_ul and up_row is not None else (left if have_l else up)
            ur = up_row[x + 1] if have_ur and up_row is not None else up

            pred = _median(left, up, ul, have_l, have_u, have_ul)
            grad = _boundary_grad(
                left, up, ul, ur, ll, have_l, have_u, have_ul, have_ur, have_ll
            )
            eff_grad = grad
            key = None
            if kalman and (have_l or have_u):
                d1 = left - ul
                d2 = up - ul
                d3 = (ur - up) if abs(d1) <= abs(d2) else (left - ll)
                key, _reflected = _d8_patch_key(d1, d2, d3)
                pred, eff_grad = _kalman_fuse(
                    pred, ul, table.get(key, []), grad, s_eta2
                )

            ctx = _bucket(eff_grad)
            k = rice_k(acc[ctx], count[ctx])
            symbol = reader.read_rice(k)
            err = unzigzag(symbol)
            value = pred + err
            row[x] = value
            acc[ctx] += symbol
            count[ctx] += 1
            if count[ctx] >= 64:
                acc[ctx] = max(1, acc[ctx] >> 1)
                count[ctx] >>= 1
            if kalman and key is not None:
                delta = value - ul
                bucket = table.get(key)
                if bucket is None:
                    table[key] = [delta]
                else:
                    if len(bucket) >= MATCHES:
                        del bucket[0]
                    bucket.append(delta)
    return np.asarray(rows, dtype=np.int32)
