"""One-step quincunx predict/update lifting on the Q^2 = 2I filtration.

Uses the commensurable reflection generator from Section 3.2 of the SLGF
specification:

    Q = [[1, 1],
         [1, -1]],   det(Q) = -2,   Q^2 = 2 I,

mapping even sites ``(r + c) mod 2 == 0`` via

    U = (r + c) / 2,   V = (r - c) / 2.

Because ``Q^2 = 2 I`` and both ``U`` and ``V`` are origin-shifted to zero,
every second quincunx level packs onto a 100%-dense unrotated ``ceil(H/2) x
ceil(W/2)`` Cartesian subgrid with zero dead rows.

Every operator in this module is pure integer arithmetic (zero IEEE-754
instructions), preserving exact cross-ISA determinism.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KAPPA_8BIT = 12
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def ilog2_int64(values: np.ndarray) -> np.ndarray:
    """Exact floor(log2(max(values, 1))) in pure int64 bit operations."""
    x = np.maximum(values.astype(np.int64, copy=False), 1).copy()
    res = np.zeros(x.shape, dtype=np.int32)
    for shift in (32, 16, 8, 4, 2, 1):
        mask = x >= (np.int64(1) << shift)
        if np.any(mask):
            x[mask] >>= shift
            res[mask] += shift
    return res


def parity_masks(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = valid.shape
    rr, cc = np.indices((h, w))
    odd_site = ((rr + cc) & 1) == 1
    odd = valid & odd_site
    even = valid & ~odd_site
    return odd, even


def _neighbors(
    img: np.ndarray, accept: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape
    padded = np.zeros((h + 2, w + 2), np.int64)
    vpad = np.zeros((h + 2, w + 2), np.bool_)
    padded[1:-1, 1:-1] = img
    vpad[1:-1, 1:-1] = accept
    neigh = np.empty((4, h, w), np.int64)
    nval = np.empty((4, h, w), np.bool_)
    for i, (dr, dc) in enumerate(_DIRS):
        neigh[i] = padded[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
        nval[i] = vpad[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
    return neigh, nval


def _adaptive_kappa2(even_vals: np.ndarray) -> int:
    """Scale sheaf edge threshold deterministically from integer even-site span."""
    if even_vals.size == 0:
        return KAPPA_8BIT * KAPPA_8BIT
    span = int(even_vals.max()) - int(even_vals.min())
    if span <= 512:
        return KAPPA_8BIT * KAPPA_8BIT
    # Scale proportionally for high-bit-depth (12..16-bit) planes using integer shift.
    shift = max(0, int(span).bit_length() - 9)
    kappa = KAPPA_8BIT << shift
    return kappa * kappa


def predict(img: np.ndarray, valid: np.ndarray, sheaf: bool) -> np.ndarray:
    """Integer prediction of odd sites from orthogonal even neighbors.

    When ``sheaf=False`` (the RITC-optimal linear Stage-1 lift), predicts the
    rounded mean of available diamond neighbors. When ``sheaf=True``, evaluates
    directional Ambrosio-Tortorelli conductances across opposing vertical
    ``(N, S)`` and horizontal ``(W, E)`` chords, breaking symmetry along edges
    and breaking ties deterministically on 2-vs-2 diagonal jumps.
    """
    _, even = parity_masks(valid)
    neigh, nval = _neighbors(img, even)
    acc = (neigh * nval).sum(axis=0)
    cnt = nval.sum(axis=0).astype(np.int64)
    pred = np.zeros(img.shape, np.int32)
    good = cnt > 0
    if not np.any(good):
        return pred
    flat_pred = np.zeros(img.shape, np.int64)
    flat_pred[good] = (2 * acc[good] + cnt[good]) // (2 * cnt[good])
    if not sheaf:
        pred[good] = flat_pred[good].astype(np.int32)
        return pred

    kappa2 = np.int64(_adaptive_kappa2(img[even]))
    pred = flat_pred.astype(np.int32, copy=True)

    has_v = nval[0] & nval[1]
    has_h = nval[2] & nval[3]
    both_chords = has_v & has_h
    if np.any(both_chords):
        n_up, n_dn, n_lt, n_rt = neigh[0], neigh[1], neigh[2], neigh[3]
        dv = np.abs(n_up - n_dn)
        dh = np.abs(n_lt - n_rt)
        # Opposing diagonal chords inside the 4-neighbor diamond (NW-axis vs NE-axis):
        d_nw = np.abs(n_up - n_lt) + np.abs(n_dn - n_rt)
        d_ne = np.abs(n_up - n_rt) + np.abs(n_dn - n_lt)

        wv = np.maximum((kappa2 << 10) // np.maximum(kappa2 + dv * dv, 1), 1)
        wh = np.maximum((kappa2 << 10) // np.maximum(kappa2 + dh * dh, 1), 1)
        # Sum of midpoints in 2x units: mv2 = N + S, mh2 = W + E.
        num2 = wv * (n_up + n_dn) + wh * (n_lt + n_rt)
        den2 = 2 * (wv + wh)
        dir_pred = (2 * num2 + den2) // (2 * den2)

        # When both axial chords jump equally (a 45-degree step edge where N==W
        # and S==E, or N==E and S==W), pick the coherent corner mode rather than
        # blurring across the jump; Stage 3's bimodal bridge models both modes.
        diag_jump = (
            both_chords
            & (dv == dh)
            & (dv * dv > 4 * kappa2)
            & ((d_nw * 4 < dv) | (d_ne * 4 < dv))
        )
        if np.any(diag_jump):
            corner = np.where(d_nw <= d_ne, (n_up + n_lt) >> 1, (n_up + n_rt) >> 1)
            dir_pred = np.where(diag_jump, corner, dir_pred)

        pred[both_chords] = dir_pred[both_chords].astype(np.int32)

    # Boundary 3-neighbor sites: weight each valid neighbor against the median of the 3.
    three = (cnt == 3) & ~both_chords
    if np.any(three):
        # Sum minus min minus max gives the exact median of the 3 valid neighbors.
        big = np.int64(1) << 60
        vmin = np.where(nval, neigh, big).min(axis=0)
        vmax = np.where(nval, neigh, -big).max(axis=0)
        med3 = acc - vmin - vmax
        diff = neigh - med3
        w3 = np.maximum((kappa2 << 8) // np.maximum(kappa2 + diff * diff, 1), 1) * nval
        wsum = w3.sum(axis=0)
        accw = (w3 * neigh).sum(axis=0)
        pred[three] = ((2 * accw[three] + wsum[three]) // (2 * wsum[three])).astype(
            np.int32
        )
    return pred


def update_term(residual_img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``floor(sum(odd neighbor residuals) / 8 + 1/2)`` via arithmetic shift."""
    odd, _ = parity_masks(valid)
    neigh, nval = _neighbors(residual_img, odd)
    acc = (neigh * nval).sum(axis=0)
    return ((acc + 4) >> 3).astype(np.int32)


def lift_forward(samples: np.ndarray, valid: np.ndarray, sheaf: bool) -> np.ndarray:
    src = samples.astype(np.int32, copy=True)
    pred = predict(src, valid, sheaf)
    out = src.copy()
    odd, even = parity_masks(valid)
    out[odd] = src[odd] - pred[odd]
    upd = update_term(out, valid)
    out[even] = src[even] + upd[even]
    return out


def lift_inverse(lifted: np.ndarray, valid: np.ndarray, sheaf: bool) -> np.ndarray:
    odd, even = parity_masks(valid)
    upd = update_term(lifted, valid)
    restored = lifted.astype(np.int32, copy=True)
    restored[even] = lifted[even] - upd[even]
    pred = predict(restored, valid, sheaf)
    restored[odd] = lifted[odd] + pred[odd]
    return restored


@dataclass(frozen=True)
class PackInfo:
    u_shift: int
    v_shift: int
    parent_shape: tuple[int, int]
    packed_shape: tuple[int, int]


def _even_coords(
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, even = parity_masks(valid)
    rr, cc = np.nonzero(even)
    u = (rr + cc) >> 1
    v = (rr - cc) >> 1
    return rr, cc, u, v


def pack_even(
    img: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, PackInfo]:
    rr, cc, u, v = _even_coords(valid)
    parent = (int(valid.shape[0]), int(valid.shape[1]))
    if rr.size == 0:
        packed = np.zeros((1, 1), np.int32)
        mask = np.zeros((1, 1), np.bool_)
        return packed, mask, PackInfo(0, 0, parent, (1, 1))
    u_shift = int(-u.min())
    v_shift = int(-v.min())
    us = u + u_shift
    vs = v + v_shift
    hs = int(us.max()) + 1
    ws = int(vs.max()) + 1
    packed = np.zeros((hs, ws), np.int32)
    mask = np.zeros((hs, ws), np.bool_)
    packed[us, vs] = img[rr, cc]
    mask[us, vs] = True
    return packed, mask, PackInfo(u_shift, v_shift, parent, (hs, ws))


def unpack_even(
    packed: np.ndarray, info: PackInfo, parent_valid: np.ndarray
) -> np.ndarray:
    spatial = np.zeros(info.parent_shape, np.int32)
    rr, cc, u, v = _even_coords(parent_valid)
    if rr.size == 0:
        return spatial
    us = u + info.u_shift
    vs = v + info.v_shift
    spatial[rr, cc] = packed[us, vs]
    return spatial


def activity_context(lowpass: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Pure-integer curvature & contour-disagreement bucket in ``0..7``.

    Measures second-order diamond saddle curvature ``|(N + S) - (W + E)|`` plus
    minimum axial chord jump ``min(|N - S|, |W - E|)`` so smooth linear ramps
    map to bucket 0 rather than inflating scale estimates.
    """
    _, even = parity_masks(valid)
    neigh, nval = _neighbors(lowpass, even)
    has_v = nval[0] & nval[1]
    has_h = nval[2] & nval[3]
    both = has_v & has_h
    dv = np.abs(neigh[0] - neigh[1]) * has_v
    dh = np.abs(neigh[2] - neigh[3]) * has_h
    saddle = np.abs((neigh[0] + neigh[1]) - (neigh[2] + neigh[3])) * both
    min_chord = np.minimum(dv, dh) * both
    # Boundary fallback when only one chord is present:
    one_chord = (dv * (has_v & ~has_h) + dh * (has_h & ~has_v)) >> 1
    act = (saddle >> 1) + (min_chord >> 1) + one_chord

    ctx = np.zeros(lowpass.shape, np.uint8)
    nz = act > 0
    if np.any(nz):
        bits = ilog2_int64(act[nz])
        ctx[nz] = np.clip(bits, 0, 7).astype(np.uint8)
    return ctx


def build_levels(
    height: int, width: int, max_levels: int
) -> tuple[list[tuple[np.ndarray, PackInfo, np.ndarray]], np.ndarray]:
    """Return ``(levels, coarse_valid)``."""
    valid = np.ones((height, width), np.bool_)
    levels: list[tuple[np.ndarray, PackInfo, np.ndarray]] = []
    for _ in range(max_levels):
        odd, even = parity_masks(valid)
        if int(odd.sum()) < 1 or int(even.sum()) < 4:
            break
        if min(valid.shape) < 3:
            break
        dummy = np.zeros(valid.shape, np.int32)
        _packed, pmask, info = pack_even(dummy, valid)
        if pmask.size > 8 * valid.size and valid.size > 64:
            break
        levels.append((valid, info, pmask))
        valid = pmask
    return levels, valid


def forward_tower(
    plane: np.ndarray, max_levels: int, sheaf: bool
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[tuple[np.ndarray, PackInfo, np.ndarray]],
]:
    """Lift ``plane`` and return ``(coarse, coarse_valid, detail_grids, geom)``."""
    geom, _final = build_levels(plane.shape[0], plane.shape[1], max_levels)
    current = plane.astype(np.int32, copy=False)
    details: list[np.ndarray] = []
    if not geom:
        mask = np.ones(current.shape, np.bool_)
        return current, mask, details, geom
    for valid, _info, _pmask in geom:
        lifted = lift_forward(current, valid, sheaf)
        details.append(lifted)
        packed, _mask, _info2 = pack_even(lifted, valid)
        current = packed
    coarse_valid = geom[-1][2]
    return current, coarse_valid, details, geom


def inverse_tower(
    coarse: np.ndarray,
    details: list[np.ndarray],
    geom: list[tuple[np.ndarray, PackInfo, np.ndarray]],
    sheaf: bool,
) -> np.ndarray:
    current = coarse.astype(np.int32, copy=False)
    if not geom:
        return current
    for lifted, (valid, info, _pmask) in zip(reversed(details), reversed(geom)):
        spatial = unpack_even(current, info, valid)
        odd, _even = parity_masks(valid)
        spatial[odd] = lifted[odd]
        current = lift_inverse(spatial, valid, sheaf)
    return current
