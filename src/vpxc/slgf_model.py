"""Unified Stage 1-3 SLGF multiscale context, Cholesky MDL head, and Core66 coder.

Implements the complete zero-weight SLGF pipeline from Section 4.1 of the
specification:
- Stage 1: RITC-optimal 1-predict/1-update linear quincunx tower on ``Q^2 = 2I``.
- Stage 2: 3 causal spatial sub-cosets ``r_s^(1..3)``, directional cellular-sheaf
  conductances, 4-way normative ``Q15.16``/``Q1.30`` ``I-SSM`` context sweeps on
  decoded coarse cosets, cross-channel + mod-2/mod-4 CFA/4:2:0 parity taps
  (``phi_acq``), and closed-form Cholesky MDL Ridge head (``phi_head``) gated
  by ``Delta L_MDL < 0``.
- Stage 3: Fractional-mean bimodal Student-t ``Dyadic-Core66 rANS`` with tilted
  sub-bucket binary remainder trees.

Every decoding operation is 100% pure integer arithmetic (zero IEEE-754
instructions), guaranteeing bit-exact reconstruction across all ISAs.
"""

from __future__ import annotations

import math

import numpy as np

from vpxc.entropy import (
    BitReader,
    BitWriter,
    NUM_FRACS,
    NUM_NUS,
    NUM_SIGMAS,
    RANS_M,
    SIGMA_Q8,
    decode_parametric_rans,
    encode_parametric_rans,
    get_parametric_tables,
)
from vpxc.errors import CodecError
from vpxc.issm import scan_4way
from vpxc.lattice import (
    PackInfo,
    build_levels,
    forward_tower,
    ilog2_int64,
    lift_inverse,
    parity_masks,
    predict,
    unpack_even,
)
from vpxc.spatial import decode_raster, encode_raster

MAX_BUCKETS = 16
WEIGHT_MAX_Q8 = 2047

# Precomputed log2(f / 65536) cost lookup for fast exact MDL bit accounting on encoder.
_NEG_LOG2_PROB = -np.log2(np.arange(1, RANS_M + 1, dtype=np.float64) / float(RANS_M))


def _pad_mirror_2d(arr: np.ndarray, pad: int) -> np.ndarray:
    h, w = arr.shape
    if h >= 2 and w >= 2:
        return np.pad(arr, pad, mode="edge")
    return np.pad(arr, pad, mode="edge")


def _even_filled(spatial_even: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill odd/invalid sites from even neighbors using pure integer arithmetic."""
    odd, even = parity_masks(valid)
    out = spatial_even.astype(np.int64, copy=True)
    flat = predict(spatial_even, valid, sheaf=False).astype(np.int64)
    out[odd] = flat[odd]
    invalid = ~valid
    if np.any(invalid):
        # Fill exterior cells of diamond grids with integer mean of valid sites
        # so edge padding never injects artificial step jumps at zero.
        ev_vals = spatial_even[even]
        mean_v = int(ev_vals.sum() // max(1, ev_vals.size)) if ev_vals.size else 0
        out[invalid] = mean_v
    return out


def _subcoset_masks(valid: np.ndarray) -> list[np.ndarray]:
    """Partition odd sites into 3 causal sub-cosets ``r_s^(1), r_s^(2), r_s^(3)``."""
    odd, _ = parity_masks(valid)
    n_odd = int(odd.sum())
    if n_odd == 0:
        return []
    if n_odd < 96:
        return [odd]
    h, w = valid.shape
    rr, cc = np.indices((h, w))
    r_even = (rr & 1) == 0
    c1 = odd & r_even & ((((rr >> 1) + (cc >> 1)) & 1) == 0)
    c2 = odd & r_even & ~c1
    c3 = odd & ~r_even
    return [m for m in (c1, c2, c3) if np.any(m)]


def _adaptive_kappa2_int(even_vals: np.ndarray) -> int:
    if even_vals.size == 0:
        return 144
    span = int(even_vals.max()) - int(even_vals.min())
    if span <= 512:
        return 144
    shift = max(0, int(span).bit_length() - 9)
    k = 12 << shift
    return k * k


def _level_geometry_features(
    spatial_even: np.ndarray,
    valid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute pure-integer Stage-1/2 coarse features at all sites of ``spatial_even``."""
    odd, even = parity_masks(valid)
    filled = _even_filled(spatial_even, valid)
    kappa2 = np.int64(_adaptive_kappa2_int(spatial_even[even]))
    kappa = max(12, math.isqrt(int(kappa2)))

    # 4-directional normative Q15.16/Q1.30 I-SSM scan on filled even field:
    ssm_e, ssm_w, ssm_s, ssm_n = scan_4way(filled, base_decay=53248, edge_kappa=kappa)

    ep = _pad_mirror_2d(filled, 3)
    vp = np.pad(even, 3, mode="constant", constant_values=False)

    # Orthogonal distance-1 neighbors (N, S, W, E):
    n1 = ep[2:-4, 3:-3]
    s1 = ep[4:-2, 3:-3]
    w1 = ep[3:-3, 2:-4]
    e1 = ep[3:-3, 4:-2]
    vn = vp[2:-4, 3:-3]
    vs = vp[4:-2, 3:-3]
    vw = vp[3:-3, 2:-4]
    ve = vp[3:-3, 4:-2]

    #Exact linear Stage-1 integer prediction that was subtracted in lift_forward:
    flat_pred = predict(spatial_even, valid, sheaf=False).astype(np.int64)

    # Mirror missing opposite chord neighbor at boundaries using valid neighbor or flat_pred:
    n_eff = np.where(vn, n1, np.where(vs, s1, flat_pred))
    s_eff = np.where(vs, s1, np.where(vn, n1, flat_pred))
    w_eff = np.where(vw, w1, np.where(ve, e1, flat_pred))
    e_eff = np.where(ve, e1, np.where(vw, w1, flat_pred))

    m0_q4 = 4 * (n_eff + s_eff + w_eff + e_eff)
    # Exact deterministic rounding-phase offset relative to integer flat_pred:
    cnt_valid = vn.astype(np.int64) + vs.astype(np.int64) + vw.astype(np.int64) + ve.astype(np.int64)
    acc_valid = n1 * vn + s1 * vs + w1 * vw + e1 * ve
    mean_exact_q4 = np.where(
        cnt_valid > 0, (16 * acc_valid) // np.maximum(cnt_valid, 1), 0
    )
    mu0_q4 = mean_exact_q4 - 16 * flat_pred

    dv = np.abs(n_eff - s_eff)
    dh = np.abs(w_eff - e_eff)
    wv = np.maximum((kappa2 << 10) // np.maximum(kappa2 + dv * dv, 1), 1)
    wh = np.maximum((kappa2 << 10) // np.maximum(kappa2 + dh * dh, 1), 1)
    dir_q4 = (8 * (wv * (n_eff + s_eff) + wh * (w_eff + e_eff))) // (wv + wh)
    f_sheaf_q4 = dir_q4 - m0_q4

    # Distance-3 axial even neighbors (N3, S3, W3, E3):
    n3 = ep[0:-6, 3:-3]
    s3 = ep[6:, 3:-3]
    w3 = ep[3:-3, 0:-6]
    e3 = ep[3:-3, 6:]
    f_cub_v_q4 = 8 * (n_eff + s_eff) - 8 * (n3 + s3)
    f_cub_h_q4 = 8 * (w_eff + e_eff) - 8 * (w3 + e3)

    # 8-site sqrt(5) even ring:
    ring_sum = (
        ep[1:-5, 2:-4]
        + ep[1:-5, 4:-2]
        + ep[5:-1, 2:-4]
        + ep[5:-1, 4:-2]
        + ep[2:-4, 1:-5]
        + ep[4:-2, 1:-5]
        + ep[2:-4, 5:-1]
        + ep[4:-2, 5:-1]
    )
    f_ring5_q4 = m0_q4 - 2 * ring_sum

    # 4-way I-SSM states evaluated on causal orthogonal neighbors (so odd center is never read directly):
    se_p = _pad_mirror_2d(ssm_e, 1)
    sw_p = _pad_mirror_2d(ssm_w, 1)
    ss_p = _pad_mirror_2d(ssm_s, 1)
    sn_p = _pad_mirror_2d(ssm_n, 1)
    # West neighbor's East-going state + East neighbor's West-going state:
    f_ssm_h_q4 = ((se_p[1:-1, :-2] + sw_p[1:-1, 2:]) >> 1) - m0_q4
    f_ssm_v_q4 = ((ss_p[:-2, 1:-1] + sn_p[2:, 1:-1]) >> 1) - m0_q4

    saddle = np.abs((n_eff + s_eff) - (w_eff + e_eff))
    min_chord = np.minimum(dv, dh)
    curv_q4 = 8 * saddle + 6 * min_chord + np.abs(f_ring5_q4)

    # Opposing modes for bilateral sheaf bridge across strong jumps:
    lo_nb = np.minimum(np.minimum(n_eff, s_eff), np.minimum(w_eff, e_eff)) - flat_pred
    hi_nb = np.maximum(np.maximum(n_eff, s_eff), np.maximum(w_eff, e_eff)) - flat_pred
    is_bridge = (min_chord >= 20) & (cnt_valid == 4)

    return {
        "mu0_q4": mu0_q4,
        "f_sheaf_q4": f_sheaf_q4,
        "f_cub_v_q4": f_cub_v_q4,
        "f_cub_h_q4": f_cub_h_q4,
        "f_ring5_q4": f_ring5_q4,
        "f_ssm_h_q4": f_ssm_h_q4,
        "f_ssm_v_q4": f_ssm_v_q4,
        "n_minus_s": n_eff - s_eff,
        "w_minus_e": w_eff - e_eff,
        "wv": wv,
        "wh": wh,
        "curv_q4": curv_q4,
        "lo_nb": lo_nb,
        "hi_nb": hi_nb,
        "is_bridge": is_bridge,
        "act_shift": np.array(
            [max(0, (int(kappa2).bit_length() - 8) // 2)], dtype=np.int32
        ),
    }


def _assemble_features_and_activity(
    base: dict[str, np.ndarray],
    sub_idx: int,
    msk: np.ndarray,
    decoded_odd: np.ndarray,
    coarse_resid_interp: np.ndarray | None,
    ch0_spatial: np.ndarray | None,
    ch0_resid: np.ndarray | None,
    ch1_resid: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build exact integer feature matrix ``Phi_q4`` and integer activity ``act_q4``."""
    h, w = msk.shape
    rr, cc = np.indices((h, w))
    feats: list[np.ndarray] = [
        base["f_sheaf_q4"][msk],
        base["f_cub_v_q4"][msk],
        base["f_cub_h_q4"][msk],
        base["f_ring5_q4"][msk],
        base["f_ssm_h_q4"][msk],
        base["f_ssm_v_q4"][msk],
    ]

    nb_mag_q4: np.ndarray | None = None
    if sub_idx == 1:
        rp = np.pad(decoded_odd, 2, mode="constant", constant_values=0)
        rv1 = rp[0:-4, 2:-2][msk]
        rv2 = rp[4:, 2:-2][msk]
        rh1 = rp[2:-2, 0:-4][msk]
        rh2 = rp[2:-2, 4:][msk]
        feats.append(4 * (rv1 + rv2 + rh1 + rh2))
        feats.append(8 * ((rv1 + rv2) - (rh1 + rh2)))
        nb_mag_q4 = 4 * (np.abs(rv1) + np.abs(rv2) + np.abs(rh1) + np.abs(rh2))
    elif sub_idx == 2:
        rp = np.pad(decoded_odd, 1, mode="constant", constant_values=0)
        rnw = rp[:-2, :-2][msk]
        rne = rp[:-2, 2:][msk]
        rsw = rp[2:, :-2][msk]
        rse = rp[2:, 2:][msk]
        r_sum4 = rnw + rne + rsw + rse
        wv_m = base["wv"][msk]
        wh_m = base["wh"][msk]
        feats.append(4 * r_sum4)
        feats.append(8 * ((rnw + rse) - (rne + rsw)))
        feats.append((4 * (wv_m - wh_m) * r_sum4) // (wv_m + wh_m))
        nb_mag_q4 = 4 * (np.abs(rnw) + np.abs(rne) + np.abs(rsw) + np.abs(rse))

    if coarse_resid_interp is not None:
        feats.append(8 * coarse_resid_interp[msk])

    if ch0_spatial is not None and ch0_resid is not None:
        # Parity-conditioned ISP / Bayer CFA / PhotoYCC 4:2:0 features (phi_acq):
        par_v = np.where((rr[msk] & 2) == 0, np.int64(1), np.int64(-1))
        par_h = np.where((cc[msk] & 2) == 0, np.int64(1), np.int64(-1))
        y0 = ch0_spatial.astype(np.int64, copy=False)
        y0p = _pad_mirror_2d(y0, 1)
        dy0 = (y0p[:-2, 1:-1] - y0p[2:, 1:-1])[msk]
        dx0 = (y0p[1:-1, :-2] - y0p[1:-1, 2:])[msk]
        lap0 = (
            4 * y0
            - y0p[:-2, 1:-1]
            - y0p[2:, 1:-1]
            - y0p[1:-1, :-2]
            - y0p[1:-1, 2:]
        )[msk]
        r0_m = ch0_resid.astype(np.int64, copy=False)[msk]
        feats.extend(
            [
                16 * r0_m,
                4 * lap0,
                8 * base["n_minus_s"][msk] * par_v,
                8 * base["w_minus_e"][msk] * par_h,
                8 * dy0 * par_v,
                8 * dx0 * par_h,
            ]
        )
        if ch1_resid is not None:
            feats.append(16 * ch1_resid.astype(np.int64, copy=False)[msk])

    phi_q4 = np.column_stack(feats).astype(np.int64, copy=False)

    curv_m = base["curv_q4"][msk]
    if nb_mag_q4 is not None:
        act_q4 = ((3 * curv_m) >> 3) + ((7 * nb_mag_q4) >> 3)
    elif coarse_resid_interp is not None:
        act_q4 = ((5 * curv_m) >> 3) + 6 * np.abs(coarse_resid_interp[msk])
    else:
        act_q4 = (curv_m * 5) >> 3

    if ch0_resid is not None:
        r0_abs = 16 * np.abs(ch0_resid.astype(np.int64, copy=False)[msk])
        act_q4 = ((5 * act_q4) >> 3) + ((2 * r0_abs) >> 3)

    return phi_q4, np.maximum(act_q4, 1)


def _bucket_from_act_q4(act_q4: np.ndarray, act_shift: int = 0) -> np.ndarray:
    scaled = np.maximum(act_q4 >> int(act_shift), 1)
    capped = np.minimum(scaled, np.int64(1) << 28)
    half_oct = ilog2_int64((capped * capped) >> 4)
    return np.clip(half_oct, 0, MAX_BUCKETS - 1).astype(np.int32)


def _coarse_resid_projection(
    prev_resid_packed: np.ndarray | None,
    info: PackInfo,
    valid: np.ndarray,
) -> np.ndarray | None:
    """Project coarser scale absolute/signed detail onto current grid's even sites, averaged onto odd."""
    if prev_resid_packed is None:
        return None
    unpacked = unpack_even(prev_resid_packed, info, valid).astype(np.int64)
    _, even = parity_masks(valid)
    up = np.pad(unpacked, 1, mode="constant", constant_values=0)
    vp = np.pad(even, 1, mode="constant", constant_values=False)
    cnt = (
        vp[:-2, 1:-1].astype(np.int64)
        + vp[2:, 1:-1].astype(np.int64)
        + vp[1:-1, :-2].astype(np.int64)
        + vp[1:-1, 2:].astype(np.int64)
    )
    acc_abs = (
        np.abs(up[:-2, 1:-1]) * vp[:-2, 1:-1]
        + np.abs(up[2:, 1:-1]) * vp[2:, 1:-1]
        + np.abs(up[1:-1, :-2]) * vp[1:-1, :-2]
        + np.abs(up[1:-1, 2:]) * vp[1:-1, 2:]
    )
    return np.where(cnt > 0, acc_abs // np.maximum(cnt, 1), 0)


def _split_mu_q4(mu_q4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split Q4 mean into ``(nearest_int, frac_idx in 0..15)``."""
    mu_int = (mu_q4 + 8) >> 4
    delta_q4 = mu_q4 - (mu_int << 4)  # in -8 .. +7
    frac_idx = np.clip(delta_q4 + 8, 0, NUM_FRACS - 1).astype(np.int32)
    return mu_int.astype(np.int64), frac_idx


def _estimate_subcoset_model(
    r_target: np.ndarray,
    mu0_q4: np.ndarray,
    phi_q4: np.ndarray,
    bkt: np.ndarray,
) -> tuple[bool, np.ndarray, int, np.ndarray]:
    """Fit MDL-gated Q8 Ridge weights ``w_q8``, Student-t ``nu_idx``, and bucket ``sig_idx``."""
    n = int(r_target.size)
    n_feat = int(phi_q4.shape[1])
    shifts_tbl, freqs_tbl, _cums, _tilt = get_parametric_tables()

    use_ridge = False
    w_q8 = np.zeros(n_feat, dtype=np.int64)
    mu_q4 = mu0_q4

    if n >= 128 and n_feat > 0:
        xf = phi_q4.astype(np.float64)
        yf = (16.0 * r_target.astype(np.float64)) - mu0_q4.astype(np.float64)
        # MDL Ridge penalty scales with feature norm:
        diag_scale = np.maximum(np.sum(xf * xf, axis=0) * 1e-4, 64.0)
        gram = xf.T @ xf + np.diag(diag_scale)
        try:
            w_flt = np.linalg.solve(gram, xf.T @ yf)
            cand_w = np.clip(
                np.round(w_flt * 256.0), -WEIGHT_MAX_Q8, WEIGHT_MAX_Q8
            ).astype(np.int64)
            if np.any(cand_w != 0):
                cand_mu_q4 = mu0_q4 + ((phi_q4 @ cand_w + 128) >> 8)
                # Compare L1 / log-scale MDL cost before committing weights:
                err0 = np.abs(16 * r_target.astype(np.int64) - mu0_q4)
                err1 = np.abs(16 * r_target.astype(np.int64) - cand_mu_q4)
                # Approximate header bit cost of signed Exp-Golomb weights:
                hdr_bits = sum(
                    2 * ((abs(int(v)) << 1) + 1).bit_length() for v in cand_w.tolist()
                )
                gain_bits = float(n) * float(
                    np.log2(float(err0.mean()) + 4.0) - np.log2(float(err1.mean()) + 4.0)
                )
                if gain_bits > hdr_bits + 8.0:
                    use_ridge = True
                    w_q8 = cand_w
                    mu_q4 = cand_mu_q4
        except np.linalg.LinAlgError:
            pass

    mu_int, frac_idx = _split_mu_q4(mu_q4)
    r_tilde = r_target.astype(np.int64) - mu_int

    # Initial sigma estimate per bucket from mean |r_tilde - delta_mu|:
    abs_err_q8 = np.abs((r_tilde << 4) - (frac_idx.astype(np.int64) - 8)) << 4
    sig_table = np.array(SIGMA_Q8, dtype=np.float64)

    if n < 64:
        mad_q8 = float(np.mean(abs_err_q8)) * 0.92 + 24.0
        s0 = int(np.argmin(np.abs(sig_table - mad_q8)))
        sig_map = np.full(MAX_BUCKETS, s0, dtype=np.int32)
        return use_ridge, w_q8, 1, sig_map

    sig_map = np.zeros(MAX_BUCKETS, dtype=np.int32)
    prev_s = 4
    for b_i in range(MAX_BUCKETS):
        bm = bkt == b_i
        cnt = int(bm.sum())
        if cnt < 8:
            sig_map[b_i] = prev_s
            continue
        est_q8 = float(np.mean(abs_err_q8[bm])) * 0.92 + 20.0
        s_guess = int(np.argmin(np.abs(sig_table - est_q8)))
        # Refine over local window [s_guess-2 .. s_guess+2] using exact Core66 bit cost at nu=3.5:
        rt_b = r_tilde[bm]
        fi_b = frac_idx[bm]
        best_s = s_guess
        best_cost = float("inf")
        for s_cand in range(max(0, s_guess - 2), min(NUM_SIGMAS, s_guess + 3)):
            bv = int(shifts_tbl[s_cand])
            if bv == 0:
                q_b = rt_b
            else:
                bias = 1 << (bv - 1)
                q_b = (rt_b + bias) >> bv
            abs_q = np.abs(q_b)
            esc = abs_q > 32
            sym = np.where(esc, 65, q_b + 32)
            f_vals = freqs_tbl[1, s_cand, fi_b, sym]
            cost = float(_NEG_LOG2_PROB[f_vals - 1].sum()) + float(cnt * bv)
            if np.any(esc):
                cost += float(np.sum(2.0 * np.log2(abs_q[esc] - 31.0) + 3.0))
            if cost < best_cost:
                best_cost = cost
                best_s = s_cand
        sig_map[b_i] = best_s
        prev_s = best_s

    # Pick optimal Student-t tail degree of freedom nu_idx in {0, 1, 2, 3}:
    si_all = sig_map[bkt]
    bv_all = shifts_tbl[si_all]
    pos_b = bv_all > 0
    bias_all = np.where(pos_b, np.int64(1) << np.maximum(bv_all - 1, 0), 0)
    q_all = np.where(pos_b, (r_tilde + bias_all) >> bv_all, r_tilde)
    sym_all = np.where(np.abs(q_all) > 32, 65, q_all + 32)

    best_nu = 1
    best_nu_cost = float("inf")
    for nu_cand in range(NUM_NUS):
        f_vals = freqs_tbl[nu_cand, si_all, frac_idx, sym_all]
        c_val = float(_NEG_LOG2_PROB[f_vals - 1].sum())
        if c_val < best_nu_cost:
            best_nu_cost = c_val
            best_nu = nu_cand

    return use_ridge, w_q8, best_nu, sig_map


def _write_subcoset_meta(
    writer: BitWriter,
    n_pts: int,
    use_ridge: bool,
    w_q8: np.ndarray,
    nu_idx: int,
    sig_map: np.ndarray,
    use_bridge: bool,
) -> None:
    writer.write_bits(1 if use_ridge else 0, 1)
    writer.write_bits(1 if use_bridge else 0, 1)
    writer.write_bits(int(nu_idx) & 3, 2)
    if use_ridge:
        for val in w_q8.tolist():
            writer.write_signed_golomb(int(val))
    if n_pts < 64:
        writer.write_bits(int(sig_map[0]) & 31, 5)
    else:
        first = int(sig_map[0]) & 31
        writer.write_bits(first, 5)
        prev = first
        for b_i in range(1, MAX_BUCKETS):
            cur = int(sig_map[b_i]) & 31
            writer.write_signed_golomb(cur - prev)
            prev = cur


def _read_subcoset_meta(
    reader: BitReader, n_pts: int, n_feat: int
) -> tuple[bool, np.ndarray, int, np.ndarray, bool]:
    use_ridge = reader.read_bits(1) == 1
    use_bridge = reader.read_bits(1) == 1
    nu_idx = reader.read_bits(2)
    if use_ridge:
        w_q8 = np.array(
            [reader.read_signed_golomb() for _ in range(n_feat)], dtype=np.int64
        )
    else:
        w_q8 = np.zeros(n_feat, dtype=np.int64)
    if n_pts < 64:
        s0 = reader.read_bits(5)
        sig_map = np.full(MAX_BUCKETS, s0, dtype=np.int32)
    else:
        sig_map = np.empty(MAX_BUCKETS, dtype=np.int32)
        prev = reader.read_bits(5)
        sig_map[0] = prev
        for b_i in range(1, MAX_BUCKETS):
            cur = prev + reader.read_signed_golomb()
            if cur < 0 or cur >= NUM_SIGMAS:
                raise CodecError("decoded sigma index out of range")
            sig_map[b_i] = cur
            prev = cur
    return use_ridge, w_q8, nu_idx, sig_map, use_bridge


def encode_slgf_tile(planes: list[np.ndarray], max_levels: int = 4) -> bytes:
    """Encode channel-lifted integer ``planes`` via unified Stage 1-3 SLGF."""
    h, w = planes[0].shape
    geom, coarse_valid = build_levels(h, w, max_levels)
    nlevels = len(geom)
    if nlevels == 0:
        blobs = []
        mask = np.ones((h, w), dtype=np.bool_)
        for p in planes:
            b = encode_raster(p, mask, kalman=True)
            blobs.append(len(b).to_bytes(4, "little") + b)
        return bytes([0]) + b"".join(blobs)

    # Forward linear RITC-safe quincunx towers for all channels:
    towers = [forward_tower(p, nlevels, sheaf=False) for p in planes]

    # Precompute channel 0 & 1 reconstructed spatial planes & odd residuals at each level
    # so chroma channels condition on exact decoded luma/opponent references:
    ch_recons: list[list[np.ndarray]] = []
    ch_resids: list[list[np.ndarray]] = []

    out_chunks: list[bytes] = [bytes([nlevels])]

    for ch_idx, (coarse, _cval, details, _g) in enumerate(towers):
        # 1. Encode coarse base band (size ~ 1/16 HW): compare MED-Raster vs Sheaf-Kalman
        b_med = encode_raster(coarse, coarse_valid, kalman=False)
        b_kal = encode_raster(coarse, coarse_valid, kalman=True)
        if len(b_kal) < len(b_med):
            c_flag = 1
            c_blob = b_kal
        else:
            c_flag = 0
            c_blob = b_med
        out_chunks.append(bytes([c_flag]) + len(c_blob).to_bytes(4, "little") + c_blob)

        current = coarse.copy()
        level_recons: list[np.ndarray] = [np.zeros((1, 1), np.int32)] * nlevels
        level_resids: list[np.ndarray] = [np.zeros((1, 1), np.int32)] * nlevels
        prev_resid_grid: np.ndarray | None = None

        # 2. Coarse-to-fine scales level = nlevels - 1 down to 0:
        for level in range(nlevels - 1, -1, -1):
            valid, info, _pmask = geom[level]
            lifted_true = details[level]
            # Construct exact decoder-side spatial_even (odd sites zeroed!):
            spatial_even = unpack_even(current, info, valid)
            odd, _even = parity_masks(valid)

            base = _level_geometry_features(spatial_even, valid)
            coarse_proj = _coarse_resid_projection(prev_resid_grid, info, valid)

            ch0_sp = ch_recons[0][level] if ch_idx >= 1 else None
            ch0_rs = ch_resids[0][level] if ch_idx >= 1 else None
            ch1_rs = ch_resids[1][level] if ch_idx >= 2 else None

            sub_masks = _subcoset_masks(valid)
            decoded_odd = np.zeros(spatial_even.shape, dtype=np.int64)
            meta_writer = BitWriter()
            sub_payloads: list[bytes] = []

            for sc_i, msk in enumerate(sub_masks):
                phi_q4, act_q4 = _assemble_features_and_activity(
                    base,
                    sc_i,
                    msk,
                    decoded_odd,
                    coarse_proj,
                    ch0_sp,
                    ch0_rs,
                    ch1_rs,
                )
                bkt = _bucket_from_act_q4(act_q4, int(base["act_shift"][0]))
                mu0_sub = base["mu0_q4"][msk]
                r_target = lifted_true[msk].astype(np.int32)

                use_ridge, w_q8, nu_idx, sig_map = _estimate_subcoset_model(
                    r_target, mu0_sub, phi_q4, bkt
                )
                mu_q4 = (
                    mu0_sub + ((phi_q4 @ w_q8 + 128) >> 8) if use_ridge else mu0_sub
                )
                mu_int, frac_idx = _split_mu_q4(mu_q4)
                r_tilde = (r_target.astype(np.int64) - mu_int).astype(np.int32)
                sig_idx = sig_map[bkt]

                # Fast MDL screen before evaluating bimodal bridge rANS:
                bm_cand = base["is_bridge"][msk] & (
                    get_parametric_tables()[0][sig_idx] == 0
                )
                use_bridge = False
                blo_cand = None
                bhi_cand = None
                if int(bm_cand.sum()) >= 24:
                    blo_arr = (base["lo_nb"][msk] - mu_int).astype(np.int32)
                    bhi_arr = (base["hi_nb"][msk] - mu_int).astype(np.int32)
                    # Only invoke bimodal table build when Residuals on bridge sites
                    # cluster closer to lo/hi endpoints than to 0:
                    rt_bm = r_tilde[bm_cand]
                    d_zero = np.abs(rt_bm)
                    d_modes = np.minimum(
                        np.abs(rt_bm - blo_arr[bm_cand]),
                        np.abs(rt_bm - bhi_arr[bm_cand]),
                    )
                    if float(d_modes.mean()) + 1.5 < float(d_zero.mean()):
                        blob_uni = encode_parametric_rans(
                            r_tilde, nu_idx, sig_idx, frac_idx
                        )
                        blob_bi = encode_parametric_rans(
                            r_tilde,
                            nu_idx,
                            sig_idx,
                            frac_idx,
                            bm_cand,
                            blo_arr,
                            bhi_arr,
                        )
                        if len(blob_bi) < len(blob_uni):
                            use_bridge = True
                            chosen_blob = blob_bi
                        else:
                            chosen_blob = blob_uni
                    else:
                        chosen_blob = encode_parametric_rans(
                            r_tilde, nu_idx, sig_idx, frac_idx
                        )
                else:
                    chosen_blob = encode_parametric_rans(
                        r_tilde, nu_idx, sig_idx, frac_idx
                    )

                _write_subcoset_meta(
                    meta_writer,
                    int(r_target.size),
                    use_ridge,
                    w_q8,
                    nu_idx,
                    sig_map,
                    use_bridge,
                )
                sub_payloads.append(
                    len(chosen_blob).to_bytes(4, "little") + chosen_blob
                )
                decoded_odd[msk] = r_target.astype(np.int64)

            meta_bytes = meta_writer.finish()
            level_blob = (
                len(meta_bytes).to_bytes(2, "little")
                + meta_bytes
                + b"".join(sub_payloads)
            )
            out_chunks.append(level_blob)

            # Reconstruct exact level plane identically to decoder:
            full_grid = spatial_even.copy()
            full_grid[odd] = decoded_odd[odd].astype(np.int32)
            resid_only = np.zeros_like(full_grid)
            resid_only[odd] = decoded_odd[odd].astype(np.int32)
            current = lift_inverse(full_grid, valid, sheaf=False)

            level_recons[level] = current
            level_resids[level] = resid_only
            prev_resid_grid = resid_only

        ch_recons.append(level_recons)
        ch_resids.append(level_resids)

    return b"".join(out_chunks)


def decode_slgf_tile(
    payload: bytes, height: int, width: int, channels: int
) -> list[np.ndarray]:
    """Exact pure-integer inverse of :func:`encode_slgf_tile`."""
    if len(payload) < 1:
        raise CodecError("truncated SLGF tile")
    nlevels = payload[0]
    pos = 1

    if nlevels == 0:
        planes: list[np.ndarray] = []
        mask = np.ones((height, width), dtype=np.bool_)
        for _ in range(channels):
            if pos + 4 > len(payload):
                raise CodecError("truncated level-0 SLGF plane")
            blen = int.from_bytes(payload[pos : pos + 4], "little")
            pos += 4
            if pos + blen > len(payload):
                raise CodecError("truncated level-0 SLGF data")
            planes.append(
                decode_raster(payload[pos : pos + blen], (height, width), mask, True)
            )
            pos += blen
        if pos != len(payload):
            raise CodecError("trailing bytes in level-0 SLGF tile")
        return planes

    geom, coarse_valid = build_levels(height, width, nlevels)
    if len(geom) != nlevels:
        raise CodecError("SLGF quincunx depth mismatch")

    ch_recons: list[list[np.ndarray]] = []
    ch_resids: list[list[np.ndarray]] = []
    out_planes: list[np.ndarray] = []

    for ch_idx in range(channels):
        if pos + 5 > len(payload):
            raise CodecError("truncated SLGF coarse header")
        c_flag = payload[pos]
        c_len = int.from_bytes(payload[pos + 1 : pos + 5], "little")
        pos += 5
        if pos + c_len > len(payload):
            raise CodecError("truncated SLGF coarse payload")
        current = decode_raster(
            payload[pos : pos + c_len],
            coarse_valid.shape,
            coarse_valid,
            bool(c_flag == 1),
        )
        pos += c_len

        level_recons = [np.zeros((1, 1), np.int32)] * nlevels
        level_resids = [np.zeros((1, 1), np.int32)] * nlevels
        prev_resid_grid: np.ndarray | None = None

        for level in range(nlevels - 1, -1, -1):
            valid, info, _pmask = geom[level]
            spatial_even = unpack_even(current, info, valid)
            odd, _even = parity_masks(valid)

            base = _level_geometry_features(spatial_even, valid)
            coarse_proj = _coarse_resid_projection(prev_resid_grid, info, valid)

            ch0_sp = ch_recons[0][level] if ch_idx >= 1 else None
            ch0_rs = ch_resids[0][level] if ch_idx >= 1 else None
            ch1_rs = ch_resids[1][level] if ch_idx >= 2 else None

            if pos + 2 > len(payload):
                raise CodecError("truncated SLGF level metadata length")
            meta_len = int.from_bytes(payload[pos : pos + 2], "little")
            pos += 2
            if pos + meta_len > len(payload):
                raise CodecError("truncated SLGF level metadata")
            meta_reader = BitReader(payload[pos : pos + meta_len])
            pos += meta_len

            sub_masks = _subcoset_masks(valid)
            decoded_odd = np.zeros(spatial_even.shape, dtype=np.int64)

            for sc_i, msk in enumerate(sub_masks):
                phi_q4, act_q4 = _assemble_features_and_activity(
                    base,
                    sc_i,
                    msk,
                    decoded_odd,
                    coarse_proj,
                    ch0_sp,
                    ch0_rs,
                    ch1_rs,
                )
                bkt = _bucket_from_act_q4(act_q4, int(base["act_shift"][0]))
                mu0_sub = base["mu0_q4"][msk]
                n_pts = int(mu0_sub.size)
                n_feat = int(phi_q4.shape[1])

                use_ridge, w_q8, nu_idx, sig_map, use_bridge = _read_subcoset_meta(
                    meta_reader, n_pts, n_feat
                )
                mu_q4 = (
                    mu0_sub + ((phi_q4 @ w_q8 + 128) >> 8) if use_ridge else mu0_sub
                )
                mu_int, frac_idx = _split_mu_q4(mu_q4)
                sig_idx = sig_map[bkt]

                if pos + 4 > len(payload):
                    raise CodecError("truncated SLGF sub-coset length")
                sub_len = int.from_bytes(payload[pos : pos + 4], "little")
                pos += 4
                if pos + sub_len > len(payload):
                    raise CodecError("truncated SLGF sub-coset stream")
                sub_blob = payload[pos : pos + sub_len]
                pos += sub_len

                if use_bridge:
                    bm_dec = base["is_bridge"][msk] & (
                        get_parametric_tables()[0][sig_idx] == 0
                    )
                    blo_dec = (base["lo_nb"][msk] - mu_int).astype(np.int32)
                    bhi_dec = (base["hi_nb"][msk] - mu_int).astype(np.int32)
                    r_tilde = decode_parametric_rans(
                        sub_blob,
                        n_pts,
                        nu_idx,
                        sig_idx,
                        frac_idx,
                        bm_dec,
                        blo_dec,
                        bhi_dec,
                    )
                else:
                    r_tilde = decode_parametric_rans(
                        sub_blob, n_pts, nu_idx, sig_idx, frac_idx
                    )

                r_rec = r_tilde.astype(np.int64) + mu_int
                decoded_odd[msk] = r_rec

            full_grid = spatial_even.copy()
            full_grid[odd] = decoded_odd[odd].astype(np.int32)
            resid_only = np.zeros_like(full_grid)
            resid_only[odd] = decoded_odd[odd].astype(np.int32)
            current = lift_inverse(full_grid, valid, sheaf=False)

            level_recons[level] = current
            level_resids[level] = resid_only
            prev_resid_grid = resid_only

        ch_recons.append(level_recons)
        ch_resids.append(level_resids)
        out_planes.append(current)

    if pos != len(payload):
        raise CodecError("trailing bytes in SLGF tile")
    return out_planes
