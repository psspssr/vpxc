"""Causal D8 Sheaf-Kalman matches from already-decoded odd sites.

Odd residuals inside one sub-coset are coded together, so a match may only use
odds from earlier sub-cosets (and the even lattice, which is known). Each
known center is indexed by the dihedral copies of its four even neighbors.
The returned correction is ``match - flat_pred`` and is zero when nothing
passes the shot-noise gate.
"""

from __future__ import annotations

import numpy as np

from vpxc.spatial import _estimate_noise_var


def _orientations(north: int, south: int, west: int, east: int) -> list[tuple[int, int, int, int]]:
    cur = (north, south, west, east)
    found = []
    for _turn in range(4):
        found.append(cur)
        north, south, west, east = cur
        found.append((south, north, east, west))
        cur = (west, east, south, north)
    return found


def sheaf_kalman_delta(
    samples: np.ndarray,
    known: np.ndarray,
    query: np.ndarray,
    flat_pred: np.ndarray,
    *,
    radius: int = 128,
) -> np.ndarray:
    """Integer Sheaf-Kalman correction on ``query`` sites.

    ``samples`` holds true values on ``known`` sites and on even neighbors.
    ``known`` and ``query`` must be disjoint.
    """
    out = np.zeros(samples.shape, dtype=np.int64)
    if int(query.sum()) < 16 or int(known.sum()) < 8:
        return out
    height, width = samples.shape
    src = np.asarray(samples, dtype=np.int64)
    flat = np.asarray(flat_pred, dtype=np.int64)
    table: dict[tuple[int, int, int, int], list[tuple[int, int, int]]] = {}
    kys, kxs = np.nonzero(known)
    for y, x in zip(kys.tolist(), kxs.tolist()):
        if y < 1 or x < 1 or y + 1 >= height or x + 1 >= width:
            continue
        north = int(src[y - 1, x])
        south = int(src[y + 1, x])
        west = int(src[y, x - 1])
        east = int(src[y, x + 1])
        dc = (north + south + west + east) >> 2
        qn, qs, qw, qe = (north - dc) >> 3, (south - dc) >> 3, (west - dc) >> 3, (east - dc) >> 3
        high = int(src[y, x]) - dc
        for key in _orientations(qn, qs, qw, qe):
            bucket = table.get(key)
            if bucket is None:
                table[key] = [(y, x, high)]
            elif len(bucket) < 8:
                bucket.append((y, x, high))
    if not table:
        return out

    # Noise scale from known samples; zero means exact-copy gating.
    noise_src = np.where(known, src, 0).astype(np.int32)
    s_eta2 = _estimate_noise_var(noise_src, known) if int(known.sum()) >= 16 else 0
    qys, qxs = np.nonzero(query)
    for y, x in zip(qys.tolist(), qxs.tolist()):
        if y < 1 or x < 1 or y + 1 >= height or x + 1 >= width:
            continue
        north = int(src[y - 1, x])
        south = int(src[y + 1, x])
        west = int(src[y, x - 1])
        east = int(src[y, x + 1])
        dc = (north + south + west + east) >> 2
        key = ((north - dc) >> 3, (south - dc) >> 3, (west - dc) >> 3, (east - dc) >> 3)
        hits = table.get(key)
        if not hits:
            continue
        local = int(flat[y, x])
        grad = abs(north - south) + abs(west - east)
        accepted: list[int] = []
        for ay, ax, high in hits:
            if max(abs(ay - y), abs(ax - x)) > radius:
                continue
            pred = dc + int(high)
            err = pred - local
            if s_eta2 <= 0:
                if abs(err) > max(2, grad >> 2):
                    continue
            elif err * err > 14 * int(s_eta2) + grad:
                continue
            accepted.append(pred)
        if not accepted:
            continue
        if s_eta2 <= 0:
            fused = accepted[-1]
        else:
            s_loc2 = grad * grad + 4
            acc = local * s_eta2 + s_loc2 * sum(accepted)
            denom = s_eta2 + len(accepted) * s_loc2
            fused = int((2 * acc + denom) // (2 * denom))
        out[y, x] = int(fused) - local
    return out
