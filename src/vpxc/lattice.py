"""One-step quincunx predict/update lifting.

Odd sites ``(r + c)`` odd are predicted from their orthogonal even neighbors.
Even sites are then updated from those residuals with the fixed LeGall-style
coefficient ``1/8`` per neighbor (missing neighbors contribute 0). The even
lowpass is repacked by the quincunx change of basis

    U = (r + c) / 2,  V = (c - r) / 2

so the next level's orthogonal neighbors are the previous level's diagonal
neighbors. The mask of valid UV cells is a function of shape only, so the
decoder rebuilds it without side information.

Sheaf prediction replaces the flat mean by an integer bilateral average whose
weights depend only on the even neighbors. ``kappa`` is the edge scale in
sample units. Both predictors are exact bijections.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KAPPA = 12
KAPPA2 = KAPPA * KAPPA
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


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


def predict(img: np.ndarray, valid: np.ndarray, sheaf: bool) -> np.ndarray:
    """Integer prediction of every site from orthogonal even neighbors."""
    _, even = parity_masks(valid)
    neigh, nval = _neighbors(img, even)
    acc = (neigh * nval).sum(axis=0)
    cnt = nval.sum(axis=0).astype(np.int64)
    pred = np.zeros(img.shape, np.int32)
    if not sheaf:
        good = cnt > 0
        pred[good] = ((2 * acc[good] + cnt[good]) // (2 * cnt[good])).astype(
            np.int32
        )
        return pred
    c0 = np.zeros(img.shape, np.int64)
    good = cnt > 0
    c0[good] = (2 * acc[good] + cnt[good]) // (2 * cnt[good])
    diff = neigh - c0
    denom = KAPPA2 + diff * diff
    weight = (KAPPA2 << 8) // np.maximum(denom, 1)
    weight = np.maximum(weight, 1)
    weight = weight * nval
    wsum = weight.sum(axis=0)
    accw = (weight * neigh).sum(axis=0)
    ok = wsum > 0
    pred[ok] = ((2 * accw[ok] + wsum[ok]) // (2 * wsum[ok])).astype(np.int32)
    return pred


def update_term(residual_img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``floor(sum(odd neighbor residuals) / 8 + 1/2)``."""
    odd, _ = parity_masks(valid)
    neigh, nval = _neighbors(residual_img, odd)
    acc = (neigh * nval).sum(axis=0)
    return ((acc + 4) // 8).astype(np.int32)


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
    v_shift: int
    parent_shape: tuple[int, int]
    packed_shape: tuple[int, int]


def _even_coords(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, even = parity_masks(valid)
    rr, cc = np.nonzero(even)
    u = (rr + cc) // 2
    v = (cc - rr) // 2
    return rr, cc, u, v


def pack_even(
    img: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, PackInfo]:
    rr, cc, u, v = _even_coords(valid)
    parent = (int(valid.shape[0]), int(valid.shape[1]))
    if rr.size == 0:
        packed = np.zeros((1, 1), np.int32)
        mask = np.zeros((1, 1), np.bool_)
        return packed, mask, PackInfo(0, parent, (1, 1))
    v_shift = int(-v.min())
    vs = v + v_shift
    hs = int(u.max()) + 1
    ws = int(vs.max()) + 1
    packed = np.zeros((hs, ws), np.int32)
    mask = np.zeros((hs, ws), np.bool_)
    packed[u, vs] = img[rr, cc]
    mask[u, vs] = True
    return packed, mask, PackInfo(v_shift, parent, (hs, ws))


def unpack_even(packed: np.ndarray, info: PackInfo, parent_valid: np.ndarray) -> np.ndarray:
    spatial = np.zeros(info.parent_shape, np.int32)
    rr, cc, u, v = _even_coords(parent_valid)
    if rr.size == 0:
        return spatial
    vs = v + info.v_shift
    spatial[rr, cc] = packed[u, vs]
    return spatial


def activity_context(lowpass: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Log-bucket of vertical vs horizontal even-neighbor disagreement.

    Meaningful on odd sites, where those neighbors are the prediction support.
    ``0`` means a locally flat lowpass (shift ``b = 0``).
    """
    _, even = parity_masks(valid)
    neigh, nval = _neighbors(lowpass, even)
    act = np.zeros(lowpass.shape, np.int64)
    for i, j in ((0, 1), (2, 3)):
        both = nval[i] & nval[j]
        act += np.abs(neigh[i] - neigh[j]) * both
    ctx = np.zeros(lowpass.shape, np.uint8)
    nz = act > 0
    if np.any(nz):
        bits = np.floor(np.log2(act[nz].astype(np.float64))).astype(np.int32)
        ctx[nz] = np.clip(bits, 0, 7).astype(np.uint8)
    return ctx


def build_levels(
    height: int, width: int, max_levels: int
) -> tuple[list[tuple[np.ndarray, PackInfo, np.ndarray]], np.ndarray]:
    """Return ``(levels, coarse_valid)``.

    Each level is ``(parent_valid, pack_info, packed_valid)``. ``coarse_valid``
    is the mask of the array that remains after the last pack (or the original
    full mask when ``levels`` is empty).
    """
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
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[tuple[np.ndarray, PackInfo, np.ndarray]]]:
    """Lift ``plane`` and return ``(coarse, coarse_valid, detail_grids, geom)``.

    ``detail_grids[0]`` is the finest lifted grid. On that grid, odd sites hold
    residuals and even sites hold the updated lowpass.
    """
    geom, _final = build_levels(plane.shape[0], plane.shape[1], max_levels)
    current = plane.astype(np.int32, copy=False)
    details: list[np.ndarray] = []
    if not geom:
        mask = np.ones(current.shape, np.bool_)
        return current, mask, details, geom
    for valid, _info, _pmask in geom:
        lifted = lift_forward(current, valid, sheaf)
        details.append(lifted)
        packed, mask, _info2 = pack_even(lifted, valid)
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
