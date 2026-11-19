"""Reversible channel lifts used at Stage 0.

Candidates are:
- ``MODE_IDENTITY``: bypass channel mixing
- ``MODE_YCOCG``: exact YCoCg-R lifting
- ``MODE_SUB``: subtract a base channel (SubtractGreen / R / B)
- ``MODE_SHEAR``: 2-D-gradient-estimated Q12 unitriangular Cholesky shear
- ``MODE_WSIL``: full 10-multiplier normative SL(3)-WSIL eigenframe lift

For 4-channel images (RGBA or 4-plane Bayer ``G1, R, B, G2``), an optional
reversible Q12 linear prediction on channel 3 conditioned on the lifted first
three channels removes cross-plane correlation without loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from vpxc.wsil import (
    Q12,
    Q12_BIAS,
    SL3Params,
    estimate_sl3_wsil,
    q12,
    r12_array,
    sl3_forward_arrays,
    sl3_inverse_arrays,
)

MODE_IDENTITY = 0
MODE_YCOCG = 1
MODE_SUB = 2
MODE_SHEAR = 3
MODE_WSIL = 4

PERM_G = 0
PERM_R = 1
PERM_B = 2
_BASE_CHANNELS = {PERM_G: 1, PERM_R: 0, PERM_B: 2}


@dataclass(frozen=True)
class ColorTransform:
    mode: int = MODE_IDENTITY
    perm: int = 0
    shear: tuple[int, int, int] = (0, 0, 0)
    wsil: SL3Params = SL3Params()
    extra_weights: tuple[int, int, int] = (0, 0, 0)


def ycocg_forward(
    r: np.ndarray, g: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rr = r.astype(np.int32, copy=False)
    gg = g.astype(np.int32, copy=False)
    bb = b.astype(np.int32, copy=False)
    co = rr - bb
    t = bb + (co >> 1)
    cg = gg - t
    y = t + (cg >> 1)
    return y, co, cg


def ycocg_inverse(
    y: np.ndarray, co: np.ndarray, cg: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = y - (cg >> 1)
    g = cg + t
    b = t - (co >> 1)
    r = b + co
    return r, g, b


def _base_index(perm: int) -> int:
    return _BASE_CHANNELS.get(int(perm), int(perm) % 3)


def _other_indexes(base_idx: int) -> list[int]:
    return [i for i in range(3) if i != base_idx]


def sub_forward(channels: list[np.ndarray], perm: int) -> list[np.ndarray]:
    base_idx = _base_index(perm)
    src = channels[base_idx]
    return [src] + [channels[i] - src for i in _other_indexes(base_idx)]


def sub_inverse(planes: list[np.ndarray], perm: int) -> list[np.ndarray]:
    base_idx = _base_index(perm)
    out: list[np.ndarray | None] = [None, None, None]
    out[base_idx] = planes[0]
    for plane, idx in zip(planes[1:], _other_indexes(base_idx)):
        out[idx] = plane + planes[0]
    return [p for p in out if p is not None]


def shear_forward(
    channels: list[np.ndarray], perm: int, a: int, c: int, d: int
) -> list[np.ndarray]:
    base_idx = _base_index(perm)
    src = channels[base_idx]
    i0, i1 = _other_indexes(base_idx)
    x = channels[i0] - r12_array(a, src)
    y = channels[i1] - r12_array(c, src) - r12_array(d, x)
    return [src, x, y]


def shear_inverse(
    planes: list[np.ndarray], perm: int, a: int, c: int, d: int
) -> list[np.ndarray]:
    base_idx = _base_index(perm)
    src, x, y = planes
    i0, i1 = _other_indexes(base_idx)
    ch_x = x + r12_array(a, src)
    ch_y = y + r12_array(d, x) + r12_array(c, src)
    out: list[np.ndarray | None] = [None, None, None]
    out[base_idx] = src
    out[i0] = ch_x
    out[i1] = ch_y
    return [p for p in out if p is not None]


def _extra_pred(head: list[np.ndarray], weights: tuple[int, int, int]) -> np.ndarray:
    w0, w1, w2 = int(weights[0]), int(weights[1]), int(weights[2])
    if w0 == 0 and w1 == 0 and w2 == 0:
        return np.zeros(head[0].shape, dtype=np.int32)
    acc = (
        np.int64(w0) * head[0].astype(np.int64, copy=False)
        + np.int64(w1) * head[1].astype(np.int64, copy=False)
        + np.int64(w2) * head[2].astype(np.int64, copy=False)
    )
    return ((acc + Q12_BIAS) >> Q12).astype(np.int32)


def _diamond_residual_1d(arr: np.ndarray, step_y: int, step_x: int) -> np.ndarray:
    sub = arr[::step_y, ::step_x].astype(np.float64, copy=False)
    if sub.shape[0] >= 3 and sub.shape[1] >= 3:
        pred = 0.25 * (
            sub[:-2, 1:-1] + sub[2:, 1:-1] + sub[1:-1, :-2] + sub[1:-1, 2:]
        )
        return (sub[1:-1, 1:-1] - pred).ravel()
    return _diffs_1d(arr, step_y, step_x)


def _score(planes: Iterable[np.ndarray]) -> float:
    """Stage-1 integer diamond-prediction residual log-rate proxy."""
    plist = [p.astype(np.int32, copy=False) for p in planes if p.size > 0]
    if not plist:
        return 0.0
    h, w = plist[0].shape
    sy = max(1, h // 256)
    sx = max(1, w // 256)
    total = 0.0
    for p in plist:
        sub = p[::sy, ::sx]
        if sub.shape[0] >= 3 and sub.shape[1] >= 3:
            pred = (
                sub[:-2, 1:-1]
                + sub[2:, 1:-1]
                + sub[1:-1, :-2]
                + sub[1:-1, 2:]
                + 2
            ) >> 2
            mad = float(np.abs(sub[1:-1, 1:-1] - pred).mean())
        elif sub.shape[1] >= 2:
            mad = float(np.abs(np.diff(sub, axis=1)).mean())
        elif sub.shape[0] >= 2:
            mad = float(np.abs(np.diff(sub, axis=0)).mean())
        else:
            mad = float(np.abs(sub).mean())
        total += float(np.log2(mad + 0.35))
    return total / len(plist)


def _diffs_1d(arr: np.ndarray, step_y: int, step_x: int) -> np.ndarray:
    sub = arr[::step_y, ::step_x].astype(np.float64, copy=False)
    parts = []
    if sub.shape[1] >= 2:
        parts.append(np.diff(sub, axis=1).ravel())
    if sub.shape[0] >= 2:
        parts.append(np.diff(sub, axis=0).ravel())
    if parts:
        return np.concatenate(parts, axis=0)
    return sub.ravel() - float(sub.mean())


def _estimate_shear(
    base: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[int, int, int]:
    """Fit unitriangular Q12 shear on Stage-1 2-D diamond residuals (zero DC bias)."""
    h, w = base.shape
    sy = max(1, h // 256)
    sx = max(1, w // 256)
    db = _diamond_residual_1d(base, sy, sx)
    dx = _diamond_residual_1d(x, sy, sx)
    dy = _diamond_residual_1d(y, sy, sx)
    if db.size == 0:
        return 0, 0, 0
    bb = float(np.dot(db, db)) + 1e-6
    a_val = float(np.dot(db, dx) / bb)
    aq = q12(a_val)
    xres = dx - (aq / 4096.0) * db
    design = np.stack([db, xres], axis=1)
    gram = design.T @ design + 1e-6 * np.eye(2)
    coef = np.linalg.solve(gram, design.T @ dy)
    return aq, q12(float(coef[0])), q12(float(coef[1]))


def _estimate_extra_weights(
    head: list[np.ndarray], extra_plane: np.ndarray
) -> tuple[int, int, int]:
    h, w = extra_plane.shape
    sy = max(1, h // 256)
    sx = max(1, w // 256)
    de = _diffs_1d(extra_plane, sy, sx)
    if de.size < 4 or float(np.abs(de).mean()) == 0.0:
        return 0, 0, 0
    d0 = _diffs_1d(head[0], sy, sx)
    d1 = _diffs_1d(head[1], sy, sx)
    d2 = _diffs_1d(head[2], sy, sx)
    design = np.stack([d0, d1, d2], axis=1)
    gram = design.T @ design + 1.0 * np.eye(3)
    coef = np.linalg.solve(gram, design.T @ de)
    weights = (q12(float(coef[0])), q12(float(coef[1])), q12(float(coef[2])))
    lifted = extra_plane.astype(np.int32, copy=False) - _extra_pred(head, weights)
    if _score([lifted]) + 0.02 < _score([extra_plane]):
        return weights
    return 0, 0, 0


def choose_color_transform(channels: list[np.ndarray]) -> ColorTransform:
    """Select optimal Stage-0 reversible channel transform over ``Z^C``."""
    if len(channels) < 3:
        return ColorTransform()
    rgb = [c.astype(np.int32, copy=False) for c in channels[:3]]
    cands: list[tuple[float, int, ColorTransform]] = []
    cands.append((_score(rgb), 0, ColorTransform(mode=MODE_IDENTITY)))

    y, co, cg = ycocg_forward(rgb[0], rgb[1], rgb[2])
    cands.append((_score((y, co, cg)), 1, ColorTransform(mode=MODE_YCOCG)))

    for idx, perm in enumerate((PERM_G, PERM_R, PERM_B)):
        sub_p = sub_forward(rgb, perm)
        cands.append(
            (_score(sub_p) + 0.035, 2 + idx, ColorTransform(mode=MODE_SUB, perm=perm))
        )

        base_idx = _base_index(perm)
        i0, i1 = _other_indexes(base_idx)
        a, c, d = _estimate_shear(rgb[base_idx], rgb[i0], rgb[i1])
        sh_p = shear_forward(rgb, perm, a, c, d)
        cands.append(
            (
                _score(sh_p) + 0.04,
                5 + idx,
                ColorTransform(mode=MODE_SHEAR, perm=perm, shear=(a, c, d)),
            )
        )

    wsil_params = estimate_sl3_wsil(rgb)
    wsil_planes = sl3_forward_arrays(rgb, wsil_params)
    cands.append(
        (
            _score(wsil_planes) + 0.06,
            9,
            ColorTransform(mode=MODE_WSIL, wsil=wsil_params),
        )
    )

    cands.sort(key=lambda item: (item[0], item[1]))
    chosen = cands[0][2]
    if len(channels) == 4:
        head = apply_color_forward(rgb, chosen)
        ew = _estimate_extra_weights(head, channels[3].astype(np.int32, copy=False))
        chosen = ColorTransform(
            mode=chosen.mode,
            perm=chosen.perm,
            shear=chosen.shear,
            wsil=chosen.wsil,
            extra_weights=ew,
        )
    return chosen


def apply_color_forward(
    channels: list[np.ndarray], transform: ColorTransform
) -> list[np.ndarray]:
    if len(channels) < 3:
        return [c.astype(np.int32, copy=False) for c in channels]
    rgb = [c.astype(np.int32, copy=False) for c in channels[:3]]
    mode = transform.mode
    if mode == MODE_IDENTITY:
        head = rgb
    elif mode == MODE_YCOCG:
        y, co, cg = ycocg_forward(rgb[0], rgb[1], rgb[2])
        head = [y, co, cg]
    elif mode == MODE_SUB:
        head = sub_forward(rgb, transform.perm)
    elif mode == MODE_SHEAR:
        a, c, d = transform.shear
        head = shear_forward(rgb, transform.perm, a, c, d)
    elif mode == MODE_WSIL:
        head = sl3_forward_arrays(rgb, transform.wsil)
    else:
        raise ValueError(f"unknown color mode {mode}")

    if len(channels) == 3:
        return head
    extra = [c.astype(np.int32, copy=False) for c in channels[3:]]
    if any(w != 0 for w in transform.extra_weights):
        extra[0] = extra[0] - _extra_pred(head, transform.extra_weights)
    return head + extra


def apply_color_inverse(
    planes: list[np.ndarray], transform: ColorTransform
) -> list[np.ndarray]:
    if len(planes) < 3:
        return planes
    head = [p.astype(np.int32, copy=False) for p in planes[:3]]
    extra = [p.astype(np.int32, copy=False) for p in planes[3:]]
    if extra and any(w != 0 for w in transform.extra_weights):
        extra[0] = extra[0] + _extra_pred(head, transform.extra_weights)

    mode = transform.mode
    if mode == MODE_IDENTITY:
        rgb = head
    elif mode == MODE_YCOCG:
        r, g, b = ycocg_inverse(head[0], head[1], head[2])
        rgb = [r, g, b]
    elif mode == MODE_SUB:
        rgb = sub_inverse(head, transform.perm)
    elif mode == MODE_SHEAR:
        a, c, d = transform.shear
        rgb = shear_inverse(head, transform.perm, a, c, d)
    elif mode == MODE_WSIL:
        rgb = sl3_inverse_arrays(head, transform.wsil)
    else:
        raise ValueError(f"unknown color mode {mode}")
    return rgb + extra


def choose_color(
    rgb: list[np.ndarray],
) -> tuple[int, int, tuple[int, int, int]]:
    """Legacy 3-tuple selector returning ``(mode, perm, shear)``."""
    assert len(rgb) == 3
    tr = choose_color_transform(rgb)
    if tr.mode == MODE_WSIL:
        # When caller uses the 3-parameter API without SL3Params storage, pick
        # the best among Identity / YCoCg-R / Sub / Shear.
        cands: list[tuple[float, int, int, tuple[int, int, int]]] = [
            (_score(rgb), MODE_IDENTITY, 0, (0, 0, 0)),
            (_score(ycocg_forward(rgb[0], rgb[1], rgb[2])), MODE_YCOCG, 0, (0, 0, 0)),
        ]
        for perm in (PERM_G, PERM_R, PERM_B):
            cands.append((_score(sub_forward(rgb, perm)), MODE_SUB, perm, (0, 0, 0)))
            base_idx = _base_index(perm)
            i0, i1 = _other_indexes(base_idx)
            sh = _estimate_shear(rgb[base_idx], rgb[i0], rgb[i1])
            cands.append(
                (_score(shear_forward(rgb, perm, *sh)) + 0.002, MODE_SHEAR, perm, sh)
            )
        cands.sort(key=lambda item: (item[0], item[1], item[2]))
        return cands[0][1], cands[0][2], cands[0][3]
    return tr.mode, tr.perm, tr.shear


def forward_channels(
    channels: list[np.ndarray],
    mode: int,
    perm: int,
    shear: tuple[int, int, int],
    wsil: SL3Params | None = None,
    extra_weights: tuple[int, int, int] = (0, 0, 0),
) -> list[np.ndarray]:
    tr = ColorTransform(
        mode=mode,
        perm=perm,
        shear=shear,
        wsil=wsil if wsil is not None else SL3Params(),
        extra_weights=extra_weights,
    )
    return apply_color_forward(channels, tr)


def inverse_channels(
    planes: list[np.ndarray],
    mode: int,
    perm: int,
    shear: tuple[int, int, int],
    wsil: SL3Params | None = None,
    extra_weights: tuple[int, int, int] = (0, 0, 0),
) -> list[np.ndarray]:
    tr = ColorTransform(
        mode=mode,
        perm=perm,
        shear=shear,
        wsil=wsil if wsil is not None else SL3Params(),
        extra_weights=extra_weights,
    )
    return apply_color_inverse(planes, tr)
