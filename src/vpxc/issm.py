"""Canonical chunked integer state-space scan (Appendix A.2).

Intra-chunk affine summaries use Q15.16 rounding. Across chunks they compose
in Q1.30 on a balanced binary tree. The tree is the normative parallel
schedule: it is integer-only, so every platform produces the same state.

Against a left-associated composition of the same chunk maps the tree stays
within a few LSBs (Q30 reassociation). Against the per-sample recurrence the
gap is 0 for ordinary decays. When every gate stays near 1, each 16-step map
drops about one ulp and that ulp is remembered, so the gap can reach a few
tens of LSBs on a length-256 scan. The monograph's 1 LSB figure matches the
ordinary-decay case and the parallel-versus-left-associated comparison, not
the near-unity stress case.

``A[t]`` is a Q0.16 decay in ``0 .. 65536`` (65536 is the exact identity used
to pad a partial last chunk). ``b[t]`` is added in the same integer units as
the state.
"""

from __future__ import annotations

import numpy as np

CHUNK = 16
Q30_ONE = 1 << 30
IDENT = (Q30_ONE, 0)


def sequential(decay: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Left-to-right recurrence ``h = ((A * h + 2^15) >> 16) + b``."""
    state = 0
    out = np.empty(len(decay), np.int64)
    for t in range(len(decay)):
        state = ((int(decay[t]) * state + (1 << 15)) >> 16) + int(bias[t])
        out[t] = state
    return out


def _compose(right: tuple[int, int], left: tuple[int, int]) -> tuple[int, int]:
    """Apply ``left`` first, then ``right`` (Q1.30)."""
    alpha_r, beta_r = right
    alpha_l, beta_l = left
    alpha = (alpha_r * alpha_l + (1 << 29)) >> 30
    beta = ((alpha_r * beta_l + (1 << 29)) >> 30) + beta_r
    return alpha, beta


def _chunk_map(decay: np.ndarray, bias: np.ndarray, start: int, length: int) -> tuple[int, int]:
    alpha = Q30_ONE
    beta = 0
    n = len(decay)
    for k in range(length):
        t = start + k
        if t < n:
            a_t = int(decay[t])
            b_t = int(bias[t])
        else:
            a_t = 1 << 16
            b_t = 0
        alpha = (a_t * alpha + (1 << 15)) >> 16
        beta = ((a_t * beta + (1 << 15)) >> 16) + b_t
    return alpha, beta


def _exclusive_scan(maps: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Blelloch exclusive scan under ``_compose`` (identity is ``(2^30, 0)``)."""
    n = len(maps)
    width = 1
    while width < n:
        width *= 2
    buf = [IDENT] * width
    for i, item in enumerate(maps):
        buf[i] = item
    step = 1
    while step < width:
        i = 0
        while i < width:
            left = i + step - 1
            right = i + 2 * step - 1
            buf[right] = _compose(buf[right], buf[left])
            i += 2 * step
        step *= 2
    buf[width - 1] = IDENT
    step = width // 2
    while step >= 1:
        i = 0
        while i < width:
            left = i + step - 1
            right = i + 2 * step - 1
            saved = buf[left]
            buf[left] = buf[right]
            # Parent prefix runs first, then the left sibling.
            buf[right] = _compose(saved, buf[right])
            i += 2 * step
        step //= 2
    return buf[:n]


def tree(decay: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Normative balanced-tree rollout. Same length as ``decay``."""
    n = len(decay)
    if n == 0:
        return np.zeros(0, np.int64)
    n_chunks = (n + CHUNK - 1) // CHUNK
    maps = [_chunk_map(decay, bias, c * CHUNK, CHUNK) for c in range(n_chunks)]
    prefixes = _exclusive_scan(maps)
    out = np.empty(n, np.int64)
    for c in range(n_chunks):
        # Prefix beta is the incoming state: apply(prefix, 0) == beta.
        state = prefixes[c][1]
        for k in range(CHUNK):
            t = c * CHUNK + k
            if t >= n:
                break
            state = ((int(decay[t]) * state + (1 << 15)) >> 16) + int(bias[t])
            out[t] = state
    return out
