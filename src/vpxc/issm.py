"""Canonical chunked integer state-space scan (Section 3.5 & Appendix A.2).

Intra-chunk affine summaries (width ``CHUNK = 16``) evolve in ``Q15.16``.
Across chunks, summaries compose in ``Q1.30`` on a fixed power-of-two balanced
Blelloch binary tree. Both 1-D scalar sequences and vectorized 2-D row/4-way
lattice scans execute the exact same integer tree schedule, guaranteeing 0-LSB
cross-ISA bit-identity.
"""

from __future__ import annotations

import numpy as np

CHUNK = 16
Q16_ONE = 1 << 16
Q16_MAX_CONTRACTIVE = Q16_ONE - 64
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


def _compose_vec(
    alpha_r: np.ndarray,
    beta_r: np.ndarray,
    alpha_l: np.ndarray,
    beta_l: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = (alpha_r * alpha_l + (1 << 29)) >> 30
    beta = ((alpha_r * beta_l + (1 << 29)) >> 30) + beta_r
    return alpha, beta


def _chunk_map(
    decay: np.ndarray, bias: np.ndarray, start: int, length: int
) -> tuple[int, int]:
    alpha = Q30_ONE
    beta = 0
    n = len(decay)
    for k in range(length):
        t = start + k
        if t < n:
            a_t = int(decay[t])
            b_t = int(bias[t])
        else:
            a_t = Q16_ONE
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
            buf[right] = _compose(saved, buf[right])
            i += 2 * step
        step //= 2
    return buf[:n]


def tree(decay: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Normative 1-D balanced-tree rollout. Same length as ``decay``."""
    n = len(decay)
    if n == 0:
        return np.zeros(0, np.int64)
    if n <= CHUNK:
        return sequential(decay, bias)
    row = tree_rows(
        np.asarray(decay, dtype=np.int64)[None, :],
        np.asarray(bias, dtype=np.int64)[None, :],
    )
    return row[0]


def tree_rows(decay_2d: np.ndarray, bias_2d: np.ndarray) -> np.ndarray:
    """Vectorized normative balanced-tree scan across ``(batch, length)`` rows.

    Produces states bit-for-bit identical to calling :func:`tree` on every row.
    """
    dec = np.asarray(decay_2d, dtype=np.int64)
    bi = np.asarray(bias_2d, dtype=np.int64)
    if dec.ndim != 2 or bi.shape != dec.shape:
        raise ValueError("decay_2d and bias_2d must have matching 2-D shapes")
    batch, length = dec.shape
    if length == 0 or batch == 0:
        return np.zeros((batch, length), dtype=np.int64)

    n_chunks = (length + CHUNK - 1) // CHUNK
    padded_len = n_chunks * CHUNK
    if padded_len != length:
        dec_pad = np.full((batch, padded_len), Q16_ONE, dtype=np.int64)
        bi_pad = np.zeros((batch, padded_len), dtype=np.int64)
        dec_pad[:, :length] = dec
        bi_pad[:, :length] = bi
    else:
        dec_pad = dec
        bi_pad = bi

    dec_c = dec_pad.reshape(batch, n_chunks, CHUNK)
    bi_c = bi_pad.reshape(batch, n_chunks, CHUNK)

    width = 1
    while width < n_chunks:
        width *= 2

    alpha = np.full((batch, width), Q30_ONE, dtype=np.int64)
    beta = np.zeros((batch, width), dtype=np.int64)
    a_work = np.full((batch, n_chunks), Q30_ONE, dtype=np.int64)
    b_work = np.zeros((batch, n_chunks), dtype=np.int64)
    for k in range(CHUNK):
        ak = dec_c[:, :, k]
        bk = bi_c[:, :, k]
        a_work = (ak * a_work + (1 << 15)) >> 16
        b_work = ((ak * b_work + (1 << 15)) >> 16) + bk
    alpha[:, :n_chunks] = a_work
    beta[:, :n_chunks] = b_work

    step = 1
    while step < width:
        left = np.arange(step - 1, width, 2 * step)
        right = left + step
        alpha[:, right], beta[:, right] = _compose_vec(
            alpha[:, right], beta[:, right], alpha[:, left], beta[:, left]
        )
        step *= 2

    alpha[:, width - 1] = Q30_ONE
    beta[:, width - 1] = 0
    step = width // 2
    while step >= 1:
        left = np.arange(step - 1, width, 2 * step)
        right = left + step
        saved_a = alpha[:, left].copy()
        saved_b = beta[:, left].copy()
        alpha[:, left] = alpha[:, right]
        beta[:, left] = beta[:, right]
        alpha[:, right], beta[:, right] = _compose_vec(
            saved_a, saved_b, alpha[:, right], beta[:, right]
        )
        step //= 2

    out_pad = np.empty((batch, n_chunks, CHUNK), dtype=np.int64)
    state = beta[:, :n_chunks].copy()
    for k in range(CHUNK):
        ak = dec_c[:, :, k]
        bk = bi_c[:, :, k]
        state = ((ak * state + (1 << 15)) >> 16) + bk
        out_pad[:, :, k] = state
    return out_pad.reshape(batch, padded_len)[:, :length].copy()


def scan_4way(
    plane: np.ndarray,
    base_decay: int = 53248,
    edge_kappa: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normative 4-directional edge-gated Q15.16/Q1.30 I-SSM scan over a 2-D plane.

    Returns ``(h_east, h_west, h_south, h_north)`` scaled in Q4 integer units
    (i.e. ``16 * sample_units``) so downstream integer features preserve
    sub-pixel precision with zero floating-point operations.
    """
    src = plane.astype(np.int64, copy=False)
    h, w = src.shape
    if h == 0 or w == 0:
        z = np.zeros((h, w), dtype=np.int64)
        return z, z, z, z

    a0 = min(max(int(base_decay), 0), Q16_MAX_CONTRACTIVE)
    k2 = max(1, int(edge_kappa) * int(edge_kappa))

    def _sweep_horiz(arr: np.ndarray) -> np.ndarray:
        diff = np.zeros_like(arr)
        if arr.shape[1] >= 2:
            diff[:, 1:] = np.abs(arr[:, 1:] - arr[:, :-1])
        # Input-gated contractive decay A_t in [0, Q16_MAX_CONTRACTIVE]:
        # resets smoothly across strong edges so states track piecewise-smooth stalks.
        decay = (a0 * k2) // np.maximum(k2 + diff * diff, 1)
        decay[:, 0] = 0
        # Bias in Q16*Q4 = Q20? Keep state in Q16 sample units (bias = (65536 - decay) * arr),
        # then shift >> 12 to emit Q4 sample units!
        bias = (Q16_ONE - decay) * arr
        states_q16 = tree_rows(decay, bias)
        return (states_q16 + (1 << 11)) >> 12

    h_east = _sweep_horiz(src)
    h_west = _sweep_horiz(src[:, ::-1])[:, ::-1]
    h_south = _sweep_horiz(src.T).T
    h_north = _sweep_horiz(src.T[:, ::-1])[:, ::-1].T
    return h_east, h_west, h_south, h_north
