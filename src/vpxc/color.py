"""Reversible channel lifts used at Stage 0.

Candidates are identity, YCoCg-R, subtract-a-base-channel, and a Q12
unitriangular shear (the integer Cholesky step of SL(3)-WSIL). The encoder
stores the candidate with the smallest mean absolute neighbor difference.
Every candidate is an exact bijection on ``Z^3``.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from vpxc.wsil import q12, r12_array

MODE_IDENTITY = 0
MODE_YCOCG = 1
MODE_SUB = 2
MODE_SHEAR = 3

# Base channel for MODE_SUB / MODE_SHEAR.
PERM_G = 0
PERM_R = 1
PERM_B = 2


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


def _other_indexes(base: int) -> list[int]:
    return [i for i in range(3) if i != base]


def sub_forward(
    channels: list[np.ndarray], base: int
) -> list[np.ndarray]:
    src = channels[base]
    return [src] + [channels[i] - src for i in _other_indexes(base)]


def sub_inverse(planes: list[np.ndarray], base: int) -> list[np.ndarray]:
    out: list[np.ndarray | None] = [None, None, None]
    out[base] = planes[0]
    for plane, idx in zip(planes[1:], _other_indexes(base)):
        out[idx] = plane + planes[0]
    return [p for p in out if p is not None]


def shear_forward(
    channels: list[np.ndarray], base: int, a: int, c: int, d: int
) -> list[np.ndarray]:
    src = channels[base]
    i0, i1 = _other_indexes(base)
    x = channels[i0] - r12_array(a, src)
    y = channels[i1] - r12_array(c, src) - r12_array(d, x)
    return [src, x, y]


def shear_inverse(
    planes: list[np.ndarray], base: int, a: int, c: int, d: int
) -> list[np.ndarray]:
    src, x, y = planes
    i0, i1 = _other_indexes(base)
    ch_x = x + r12_array(a, src)
    ch_y = y + r12_array(d, x) + r12_array(c, src)
    out: list[np.ndarray | None] = [None, None, None]
    out[base] = src
    out[i0] = ch_x
    out[i1] = ch_y
    return [p for p in out if p is not None]


def _score(planes: Iterable[np.ndarray]) -> float:
    total = 0.0
    count = 0
    for plane in planes:
        p = plane.astype(np.int32, copy=False)
        if p.shape[1] >= 2:
            total += float(np.abs(np.diff(p, axis=1)).mean())
            count += 1
        elif p.size:
            total += float(np.abs(p).mean())
            count += 1
    if count == 0:
        return 0.0
    return total / count


def _estimate_shear(
    base: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[int, int, int]:
    b = base.astype(np.float64, copy=False).ravel()
    xr = x.astype(np.float64, copy=False).ravel()
    yr = y.astype(np.float64, copy=False).ravel()
    step = max(1, b.size // 80000)
    b = b[::step]
    xr = xr[::step]
    yr = yr[::step]
    bb = float(np.dot(b, b)) + 1e-6
    a = float(np.dot(b, xr) / bb)
    xres = xr - a * b
    design = np.stack([b, xres], axis=1)
    coef, _, _, _ = np.linalg.lstsq(design, yr, rcond=None)
    return q12(a), q12(float(coef[0])), q12(float(coef[1]))


def choose_color(
    rgb: list[np.ndarray],
) -> tuple[int, int, tuple[int, int, int]]:
    """Pick ``(mode, perm, (a, c, d))`` by a reversible-domain smoothness proxy."""
    assert len(rgb) == 3
    cands: list[tuple[float, int, int, tuple[int, int, int]]] = []
    cands.append((_score(rgb), MODE_IDENTITY, 0, (0, 0, 0)))
    y, co, cg = ycocg_forward(rgb[0], rgb[1], rgb[2])
    cands.append((_score((y, co, cg)), MODE_YCOCG, 0, (0, 0, 0)))
    for base in (PERM_G, PERM_R, PERM_B):
        cands.append(
            (_score(sub_forward(rgb, base)), MODE_SUB, base, (0, 0, 0))
        )
        i0, i1 = _other_indexes(base)
        a, c, d = _estimate_shear(rgb[base], rgb[i0], rgb[i1])
        planes = shear_forward(rgb, base, a, c, d)
        cands.append((_score(planes), MODE_SHEAR, base, (a, c, d)))
    cands.sort(key=lambda item: (item[0], item[1], item[2]))
    _, mode, perm, shear = cands[0]
    return mode, perm, shear


def forward_channels(
    channels: list[np.ndarray], mode: int, perm: int, shear: tuple[int, int, int]
) -> list[np.ndarray]:
    if len(channels) < 3 or mode == MODE_IDENTITY:
        return [c.astype(np.int32, copy=False) for c in channels]
    rgb = [c.astype(np.int32, copy=False) for c in channels[:3]]
    extra = [c.astype(np.int32, copy=False) for c in channels[3:]]
    if mode == MODE_YCOCG:
        y, co, cg = ycocg_forward(rgb[0], rgb[1], rgb[2])
        planes = [y, co, cg]
    elif mode == MODE_SUB:
        planes = sub_forward(rgb, perm)
    elif mode == MODE_SHEAR:
        a, c, d = shear
        planes = shear_forward(rgb, perm, a, c, d)
    else:
        raise ValueError(f"unknown color mode {mode}")
    return planes + extra


def inverse_channels(
    planes: list[np.ndarray], mode: int, perm: int, shear: tuple[int, int, int]
) -> list[np.ndarray]:
    if len(planes) < 3 or mode == MODE_IDENTITY:
        return planes
    head = planes[:3]
    extra = planes[3:]
    if mode == MODE_YCOCG:
        r, g, b = ycocg_inverse(head[0], head[1], head[2])
        rgb = [r, g, b]
    elif mode == MODE_SUB:
        rgb = sub_inverse(head, perm)
    elif mode == MODE_SHEAR:
        a, c, d = shear
        rgb = shear_inverse(head, perm, a, c, d)
    else:
        raise ValueError(f"unknown color mode {mode}")
    return rgb + extra
