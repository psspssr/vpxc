"""Encode-only MDL fits for Q8 ridge weights and Core66 scale tables.

Decoding never imports this module. Every comparison here may use IEEE-754
arithmetic; the weights and sigma indexes that survive the gate are integers.
"""

from __future__ import annotations

import math

import numpy as np

from vpxc.entropy import NUM_NUS, NUM_SIGMAS, SIGMA_Q8, get_parametric_tables

WEIGHT_MAX_Q8 = 2047


def _golomb_bits(value: int) -> int:
    mag = (abs(int(value)) << 1) + 1
    return 2 * mag.bit_length()


def fit_ridge_q8(
    phi_q4: np.ndarray,
    target_q4: np.ndarray,
    *,
    min_count: int = 64,
) -> np.ndarray | None:
    """Return Q8 weights explaining ``target_q4`` or ``None`` when MDL rejects them.

    ``target_q4`` is the Q4 residual left after the integer base predictor
    (``16 * sample_error``). ``phi_q4`` columns are Q4 features.
    """
    phi = np.asarray(phi_q4, dtype=np.int64)
    target = np.asarray(target_q4, dtype=np.int64)
    if phi.ndim != 2 or target.shape != (phi.shape[0],):
        raise ValueError("ridge shapes do not match")
    n, n_feat = int(phi.shape[0]), int(phi.shape[1])
    if n < min_count or n_feat == 0:
        return None
    if not np.any(phi):
        return None
    xf = phi.astype(np.float64)
    yf = target.astype(np.float64)
    diag = np.maximum(np.sum(xf * xf, axis=0) * 1e-4, 64.0)
    gram = xf.T @ xf + np.diag(diag)
    try:
        solved = np.linalg.solve(gram, xf.T @ yf)
    except np.linalg.LinAlgError:
        return None
    weights = np.clip(np.round(solved), -WEIGHT_MAX_Q8, WEIGHT_MAX_Q8).astype(np.int64)
    if not np.any(weights):
        return None
    corr = (phi @ weights + 128) >> 8
    err0 = np.abs(target)
    err1 = np.abs(target - corr)
    hdr = sum(_golomb_bits(int(v)) for v in weights.tolist()) + 8
    gain = float(n) * (
        math.log2(float(err0.mean()) + 4.0) - math.log2(float(err1.mean()) + 4.0)
    )
    if gain <= hdr:
        return None
    return weights


def fit_sigma_nu(
    r_tilde: np.ndarray,
    frac_idx: np.ndarray,
    buckets: np.ndarray,
    n_buckets: int = 16,
) -> tuple[int, np.ndarray]:
    """Pick a Student-t ``nu`` and a per-bucket ``SIGMA_Q8`` index."""
    rt = np.asarray(r_tilde, dtype=np.int64).ravel()
    frac = np.asarray(frac_idx, dtype=np.int32).ravel()
    bkt = np.asarray(buckets, dtype=np.int32).ravel()
    n = int(rt.size)
    shifts_tbl, freqs_tbl, _cums, _tilt = get_parametric_tables()
    sig_table = np.asarray(SIGMA_Q8, dtype=np.float64)
    abs_err_q8 = np.abs((rt << 4) - (frac.astype(np.int64) - 8)) << 4
    if n < 48:
        mad = float(np.mean(abs_err_q8)) * 0.92 + 24.0 if n else 64.0
        s0 = int(np.argmin(np.abs(sig_table - mad)))
        return 1, np.full(n_buckets, s0, dtype=np.int32)

    sig_map = np.zeros(n_buckets, dtype=np.int32)
    prev = 4
    neg_log = -np.log2(np.arange(1, 65537, dtype=np.float64) / 65536.0)
    for b_i in range(n_buckets):
        mask = bkt == b_i
        cnt = int(mask.sum())
        if cnt < 8:
            sig_map[b_i] = prev
            continue
        est = float(np.mean(abs_err_q8[mask])) * 0.92 + 20.0
        guess = int(np.argmin(np.abs(sig_table - est)))
        rt_b = rt[mask]
        fi_b = frac[mask]
        best_s = guess
        best_cost = float("inf")
        for cand in range(max(0, guess - 2), min(NUM_SIGMAS, guess + 3)):
            bv = int(shifts_tbl[cand])
            if bv == 0:
                q_b = rt_b
            else:
                bias = 1 << (bv - 1)
                q_b = (rt_b + bias) >> bv
            sym = np.where(np.abs(q_b) > 32, 65, q_b + 32)
            freqs = freqs_tbl[1, cand, fi_b, sym]
            cost = float(neg_log[freqs - 1].sum()) + float(cnt * bv)
            escaped = np.abs(q_b) > 32
            if np.any(escaped):
                cost += float(np.sum(2.0 * np.log2(np.abs(q_b[escaped]) - 31.0) + 3.0))
            if cost < best_cost:
                best_cost = cost
                best_s = cand
        sig_map[b_i] = best_s
        prev = best_s

    si = sig_map[bkt]
    bv_all = shifts_tbl[si]
    pos = bv_all > 0
    bias_all = np.where(pos, np.int64(1) << np.maximum(bv_all - 1, 0), 0)
    q_all = np.where(pos, (rt + bias_all) >> bv_all, rt)
    sym_all = np.where(np.abs(q_all) > 32, 65, q_all + 32)
    best_nu = 1
    best_nu_cost = float("inf")
    for nu in range(NUM_NUS):
        freqs = freqs_tbl[nu, si, frac, sym_all]
        cost = float(neg_log[freqs - 1].sum())
        if cost < best_nu_cost:
            best_nu_cost = cost
            best_nu = nu
    return int(best_nu), sig_map
