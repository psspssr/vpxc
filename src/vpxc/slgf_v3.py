"""Version-3 SLGF tile coder.

Workstream 1 replaces the coarse Rice coder with a causal predictor bank and
Dyadic-Core66 rANS. Workstream 2 adds directional Q8 regimes and an MDL
MANIAC tree with optional clustered histograms. Workstream 3 applies the
pre-gamma acquisition header inside the chroma mean and fuses causal D8
Sheaf-Kalman matches. Workstream 4, at effort 3, runs the packed INT8 stem
and tries S^3-BB-rANS. Decode of this payload is integer-only.
"""

from __future__ import annotations

import numpy as np

from vpxc.acquisition import PhiAcq, apply_phi, fit_phi, read_phi, write_phi
from vpxc.bitsback import decode_s3, encode_s3
from vpxc.coarse import decode_coarse, encode_coarse
from vpxc.entropy import (
    NUM_SIGMAS,
    BitReader,
    BitWriter,
    decode_parametric_rans,
    encode_parametric_rans,
    get_parametric_tables,
)
from vpxc.errors import CodecError
from vpxc.issm_int8 import context_map, project_head
from vpxc.lattice import (
    build_levels,
    forward_tower,
    lift_inverse,
    parity_masks,
    unpack_even,
)
from vpxc.matching import sheaf_kalman_delta
from vpxc.mdl import fit_ridge_q8, fit_sigma_nu
from vpxc.slgf_model import (
    MAX_BUCKETS,
    _assemble_features_and_activity,
    _bucket_from_act_q4,
    _coarse_resid_projection,
    _estimate_subcoset_model,
    _level_geometry_features,
    _split_mu_q4,
    _subcoset_masks,
)
from vpxc.spatial import decode_raster, encode_raster
from vpxc.trees import (
    apply_tree,
    build_sigma_tree,
    clustered_histograms,
    number_tree,
    quantize_props,
    read_histograms,
    read_tree,
    write_tree,
)

_D8 = 2
_HEAD = 4
_ISSM = 8


def _even_pad(arr: np.ndarray | None, axis_rows: bool) -> np.ndarray | None:
    if arr is None or arr.size == 0:
        return None
    out = arr
    if axis_rows and (out.shape[0] & 1):
        out = out[:-1]
    if (not axis_rows) and (out.shape[1] & 1):
        out = out[:, :-1]
    if out.size == 0:
        return None
    return out


def _features(spatial_even: np.ndarray, valid: np.ndarray, halo) -> dict[str, np.ndarray]:
    if halo is None:
        return _level_geometry_features(spatial_even, valid)
    top, left, corner = halo
    top = _even_pad(top, True)
    left = _even_pad(left, False)
    pt = 0 if top is None else int(top.shape[0])
    pl = 0 if left is None else int(left.shape[1])
    if pt == 0 and pl == 0:
        return _level_geometry_features(spatial_even, valid)
    height, width = spatial_even.shape
    big = np.zeros((height + pt, width + pl), dtype=np.int32)
    big_v = np.zeros(big.shape, dtype=np.bool_)
    if pt and top is not None:
        use_w = min(width, top.shape[1])
        big[:pt, pl : pl + use_w] = top[:, :use_w]
        big_v[:pt, pl : pl + use_w] = True
    if pl and left is not None:
        use_h = min(height, left.shape[0])
        big[pt : pt + use_h, :pl] = left[:use_h, :]
        big_v[pt : pt + use_h, :pl] = True
    if pt and pl and corner is not None:
        corner = corner[:pt, :pl]
        big[: corner.shape[0], : corner.shape[1]] = corner
        big_v[: corner.shape[0], : corner.shape[1]] = True
    big[pt:, pl:] = spatial_even
    big_v[pt:, pl:] = valid
    feat = _level_geometry_features(big, big_v)
    cropped: dict[str, np.ndarray] = {}
    for key, val in feat.items():
        if isinstance(val, np.ndarray) and val.ndim >= 2 and val.shape[:2] == big.shape:
            cropped[key] = val[pt:, pl:]
        else:
            cropped[key] = val
    return cropped


def _sectors(base: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    ns = base["n_minus_s"][mask]
    we = base["w_minus_e"][mask]
    ax = np.abs(we)
    ay = np.abs(ns)
    horiz = ax >= ay
    sx = we >= 0
    sy = ns >= 0
    sector = np.where(horiz, np.where(sx, 0, 4), np.where(sy, 2, 6))
    diag = (ax > 0) & (ay > 0) & (np.minimum(ax, ay) * 3 >= np.maximum(ax, ay))
    sector = np.where(
        diag,
        np.where(sx, np.where(sy, 1, 7), np.where(sy, 3, 5)),
        sector,
    )
    return sector.astype(np.int32)


def _product(decoded_odd: np.ndarray, mask: np.ndarray, sub_idx: int) -> np.ndarray | None:
    if sub_idx <= 0:
        return None
    if sub_idx == 1:
        padded = np.pad(decoded_odd, 2, mode="constant")
        left = padded[0:-4, 2:-2][mask]
        right = padded[4:, 2:-2][mask]
        return (left * right) >> 6
    padded = np.pad(decoded_odd, 1, mode="constant")
    return (padded[:-2, :-2][mask] * padded[2:, 2:][mask]) >> 6


def _props(
    base: dict[str, np.ndarray],
    mask: np.ndarray,
    decoded_odd: np.ndarray,
    coarse_proj: np.ndarray | None,
    ch0_resid: np.ndarray | None,
) -> np.ndarray:
    rr, cc = np.indices(mask.shape)
    ns = base["n_minus_s"][mask]
    we = base["w_minus_e"][mask]
    zeros = np.zeros(ns.shape, dtype=np.int64)
    prod = _product(decoded_odd, mask, 2)
    cols = [
        ns,
        we,
        np.abs(ns),
        np.abs(we),
        base["curv_q4"][mask],
        base["f_ring5_q4"][mask],
        prod if prod is not None else zeros,
        coarse_proj[mask] if coarse_proj is not None else zeros,
        ch0_resid[mask] if ch0_resid is not None else zeros,
        np.abs(ch0_resid[mask]) if ch0_resid is not None else zeros,
        ((rr[mask] & 1) << 1) | (cc[mask] & 1),
    ]
    return quantize_props(np.column_stack(cols))


def _apply_w(
    mu0: np.ndarray,
    phi: np.ndarray,
    sectors: np.ndarray,
    global_w: np.ndarray | None,
    overrides: dict[int, np.ndarray],
) -> np.ndarray:
    mu = mu0.astype(np.int64, copy=True)
    if global_w is not None and np.any(global_w):
        mu = mu0 + ((phi @ global_w.astype(np.int64) + 128) >> 8)
    for sector, weights in overrides.items():
        sel = sectors == sector
        if np.any(sel):
            mu[sel] = mu0[sel] + ((phi[sel] @ weights.astype(np.int64) + 128) >> 8)
    return mu


def _write_sig(writer: BitWriter, sig_map: np.ndarray, n_pts: int) -> None:
    if n_pts < 64:
        writer.write_bits(int(sig_map[0]) & 31, 5)
        return
    prev = int(sig_map[0]) & 31
    writer.write_bits(prev, 5)
    for b_i in range(1, MAX_BUCKETS):
        cur = int(sig_map[b_i]) & 31
        writer.write_signed_golomb(cur - prev)
        prev = cur


def _read_sig(reader: BitReader, n_pts: int) -> np.ndarray:
    if n_pts < 64:
        return np.full(MAX_BUCKETS, reader.read_bits(5), dtype=np.int32)
    sig = np.empty(MAX_BUCKETS, dtype=np.int32)
    prev = reader.read_bits(5)
    sig[0] = prev
    for b_i in range(1, MAX_BUCKETS):
        cur = prev + reader.read_signed_golomb()
        if cur < 0 or cur >= NUM_SIGMAS:
            raise CodecError("sigma index out of range")
        sig[b_i] = cur
        prev = cur
    return sig


def _l1(residual: np.ndarray, mu_q4: np.ndarray) -> int:
    return int(np.abs((residual.astype(np.int64) << 4) - mu_q4.astype(np.int64)).sum())


def _seed_mask(mask: np.ndarray) -> np.ndarray:
    rr, cc = np.indices(mask.shape)
    ys, xs = np.nonzero(mask)
    return (ys % 16 == 1) & (xs % 14 == 0)


def _bridge_blob(r_tilde, nu, sig_idx, frac, base, mask, mu_int):
    shifts = get_parametric_tables()[0]
    bridge = base["is_bridge"][mask] & (shifts[sig_idx] == 0)
    if int(bridge.sum()) < 24:
        return None
    lo = (base["lo_nb"][mask] - mu_int).astype(np.int32)
    hi = (base["hi_nb"][mask] - mu_int).astype(np.int32)
    rt = r_tilde[bridge]
    d0 = np.abs(rt)
    d1 = np.minimum(np.abs(rt - lo[bridge]), np.abs(rt - hi[bridge]))
    if float(d1.mean()) + 1.5 >= float(d0.mean()):
        return None
    plain = encode_parametric_rans(r_tilde, nu, sig_idx, frac)
    mixed = encode_parametric_rans(r_tilde, nu, sig_idx, frac, bridge, lo, hi)
    if len(mixed) < len(plain):
        return mixed, bridge, lo, hi
    return None


def encode_slgf_v3(
    planes: list[np.ndarray],
    max_levels: int = 4,
    effort: int = 2,
    halos: list | None = None,
    use_static: bool = True,
) -> bytes:
    """Encode lifted planes. ``halos`` entries are ``(top, left, corner)``."""
    height, width = planes[0].shape
    geom, coarse_valid = build_levels(height, width, max_levels)
    nlevels = len(geom)
    if nlevels == 0:
        blobs = []
        mask = np.ones((height, width), dtype=np.bool_)
        for plane in planes:
            blob = encode_raster(plane, mask, kalman=True)
            blobs.append(len(blob).to_bytes(4, "little") + blob)
        return bytes([0]) + b"".join(blobs)

    feature_mask = 0
    if effort >= 2 and min(height, width) >= 8:
        feature_mask |= _D8
        if use_static:
            feature_mask |= _HEAD
    if effort >= 3 and use_static:
        feature_mask |= _ISSM

    towers = [forward_tower(plane, nlevels, sheaf=False) for plane in planes]
    ch_recons: list[list[np.ndarray]] = []
    ch_resids: list[list[np.ndarray]] = []
    out = [bytes([nlevels])]
    coarse_ref: list[np.ndarray] = []

    for ch_idx, (coarse, _cval, details, _g) in enumerate(towers):
        ref = coarse_ref[0] if ch_idx > 0 and coarse_ref[0].shape == coarse.shape else None
        candidates = [
            (0, encode_raster(coarse, coarse_valid, kalman=False)),
            (1, encode_raster(coarse, coarse_valid, kalman=True)),
            (2, encode_coarse(coarse, coarse_valid, ref)),
        ]
        c_flag, c_blob = min(candidates, key=lambda item: len(item[1]))
        out.append(bytes([c_flag]) + len(c_blob).to_bytes(4, "little") + c_blob)
        coarse_ref.append(coarse)

        current = coarse.copy()
        level_recons = [np.zeros((1, 1), np.int32) for _ in range(nlevels)]
        level_resids = [np.zeros((1, 1), np.int32) for _ in range(nlevels)]
        prev_resid = None
        halo = None if halos is None else halos[ch_idx]

        for level in range(nlevels - 1, -1, -1):
            valid, info, _pmask = geom[level]
            lifted = details[level]
            spatial_even = unpack_even(current, info, valid)
            odd, _even = parity_masks(valid)
            base = _features(spatial_even, valid, halo if level == 0 else None)
            coarse_proj = _coarse_resid_projection(prev_resid, info, valid)
            ch0_sp = ch_recons[0][level] if ch_idx >= 1 else None
            ch0_rs = ch_resids[0][level] if ch_idx >= 1 else None
            ch1_rs = ch_resids[1][level] if ch_idx >= 2 else None

            phi_model: PhiAcq | None = None
            phi_pred = None
            if (
                effort >= 2
                and ch0_sp is not None
                and ch0_sp.shape == spatial_even.shape
                and int(odd.sum()) >= 2048
            ):
                true_plane = spatial_even.copy()
                true_plane[odd] = (
                    base["flat_pred"][odd] + lifted[odd].astype(np.int64)
                ).astype(np.int32)
                phi_model = fit_phi(ch0_sp, true_plane, base["flat_pred"], odd)
                if phi_model is not None:
                    phi_pred = apply_phi(ch0_sp, phi_model)

            issm_map = None
            if feature_mask & _ISSM:
                filled = spatial_even.astype(np.int64).copy()
                filled[odd] = base["flat_pred"][odd]
                issm_map = context_map(filled)

            sub_masks = _subcoset_masks(valid)
            decoded_odd = np.zeros(spatial_even.shape, dtype=np.int64)
            known = np.zeros(spatial_even.shape, dtype=np.bool_)
            meta = BitWriter()
            payloads: list[bytes] = []
            phi_used = False

            prepared = []
            for sub_idx, mask in enumerate(sub_masks):
                d8 = None
                if feature_mask & _D8:
                    samples = spatial_even.astype(np.int64).copy()
                    if np.any(known):
                        samples[known] = base["flat_pred"][known] + decoded_odd[known]
                    d8 = sheaf_kalman_delta(samples, known, mask, base["flat_pred"])
                base_phi, act = _assemble_features_and_activity(
                    base, sub_idx, mask, decoded_odd, coarse_proj, ch0_sp, ch0_rs, ch1_rs
                )
                phi = base_phi
                mu0 = base["mu0_q4"][mask].astype(np.int64).copy()
                target = lifted[mask].astype(np.int64)
                use_phi_mu = False
                if phi_pred is not None:
                    mu_phi = (phi_pred[mask] - base["flat_pred"][mask]) << 4
                    if _l1(target, mu_phi) + 64 < _l1(target, mu0):
                        mu0 = mu_phi
                        use_phi_mu = True
                        phi_used = True
                use_dir = False
                if effort >= 2:
                    mu_dir = mu0 + (base["f_sheaf_q4"][mask] >> 1)
                    if _l1(target, mu_dir) + 16 < _l1(target, mu0):
                        mu0 = mu_dir
                        use_dir = True
                use_d8 = False
                if d8 is not None and int(np.count_nonzero(d8[mask])):
                    mu_d8 = mu0 + (d8[mask].astype(np.int64) << 4)
                    if _l1(target, mu_d8) + 16 < _l1(target, mu0):
                        mu0 = mu_d8
                        use_d8 = True
                use_head = False
                if (feature_mask & _HEAD) and phi.shape[1] > 0:
                    head_q = project_head(phi)
                    mu_h = mu0 + head_q
                    if _l1(target, mu_h) + 16 < _l1(target, mu0):
                        mu0 = mu_h
                        use_head = True
                use_issm = False
                if issm_map is not None:
                    mu_i = mu0 + issm_map[mask].astype(np.int64)
                    if _l1(target, mu_i) + 16 < _l1(target, mu0):
                        mu0 = mu_i
                        use_issm = True
                sectors = _sectors(base, mask)
                buckets = _bucket_from_act_q4(act, int(base["act_shift"][0]))
                use_ridge, global_w, nu, sig_map = _estimate_subcoset_model(
                    target.astype(np.int32), mu0, phi, buckets
                )
                if not use_ridge:
                    global_w = None
                overrides: dict[int, np.ndarray] = {}
                if effort >= 2 and int(target.size) >= 200 and phi.shape[1] > 0:
                    target_q4 = (target << 4) - mu0
                    for sector in range(8):
                        sel = sectors == sector
                        if int(sel.sum()) < 64:
                            continue
                        local = fit_ridge_q8(phi[sel], target_q4[sel], min_count=64)
                        if local is not None:
                            overrides[sector] = local
                mu_q4 = _apply_w(mu0, phi, sectors, global_w, overrides)
                if overrides:
                    # Keep per-sector weights only when they beat the global model
                    # by more than their header on the actual Core66 coder.
                    mu_global = _apply_w(mu0, phi, sectors, global_w, {})
                    mu_i_g, frac_g = _split_mu_q4(mu_global)
                    resid_g = (target - mu_i_g).astype(np.int32)
                    nu_g, sig_g = fit_sigma_nu(resid_g, frac_g, buckets, MAX_BUCKETS)
                    sig_g_idx = sig_g[buckets]
                    if int(resid_g.size) < 64:
                        sig_g_idx = np.full(resid_g.shape, int(sig_g[0]), dtype=np.int32)
                    plain = encode_parametric_rans(resid_g, nu_g, sig_g_idx, frac_g)
                    mu_int, frac = _split_mu_q4(mu_q4)
                    residual = (target - mu_int).astype(np.int32)
                    nu, sig_map = fit_sigma_nu(residual, frac, buckets, MAX_BUCKETS)
                    sig_idx = sig_map[buckets]
                    if int(residual.size) < 64:
                        sig_idx = np.full(residual.shape, int(sig_map[0]), dtype=np.int32)
                    mixed = encode_parametric_rans(residual, nu, sig_idx, frac)
                    header_tax = 8 + 2 * sum(len(w) for w in overrides.values())
                    if len(mixed) + header_tax >= len(plain):
                        overrides = {}
                        mu_q4 = mu_global
                        mu_int, frac = mu_i_g, frac_g
                        residual = resid_g
                        nu, sig_map, sig_idx = nu_g, sig_g, sig_g_idx
                        use_ridge = global_w is not None
                else:
                    mu_int, frac = _split_mu_q4(mu_q4)
                    residual = (target - mu_int).astype(np.int32)
                    sig_idx = sig_map[buckets]
                    if int(residual.size) < 64:
                        sig_idx = np.full(residual.shape, int(sig_map[0]), dtype=np.int32)
                use_tree = False
                tree = None
                qprops = None
                if effort >= 2 and int(residual.size) >= 80:
                    qprops = _props(base, mask, decoded_odd, coarse_proj, ch0_rs)
                    tree = build_sigma_tree(qprops, residual)
                    if tree is not None:
                        sig_tree, _leaves = apply_tree(tree, qprops)
                        blob_b = encode_parametric_rans(residual, nu, sig_idx, frac)
                        blob_t = encode_parametric_rans(residual, nu, sig_tree, frac)
                        tree_hdr = BitWriter()
                        write_tree(tree_hdr, tree)
                        if len(blob_t) + len(tree_hdr.finish()) < len(blob_b):
                            use_tree = True
                            sig_idx = sig_tree
                use_hist = False
                hist_bytes = b""
                ext_f = None
                ext_c = None
                if use_tree and tree is not None and qprops is not None and int(residual.size) >= 400:
                    packed = clustered_histograms(qprops, tree, residual, nu)
                    if packed is not None:
                        ext_f, ext_c, hist_bytes = packed
                        blob_h = encode_parametric_rans(
                            residual, nu, sig_idx, frac, ext_freq=ext_f, ext_cum=ext_c
                        )
                        blob_p = encode_parametric_rans(residual, nu, sig_idx, frac)
                        if len(blob_h) + len(hist_bytes) < len(blob_p):
                            use_hist = True
                        else:
                            ext_f = ext_c = None
                            hist_bytes = b""
                use_bridge = False
                bridge_pack = None
                if not use_hist:
                    bridge_pack = _bridge_blob(residual, nu, sig_idx, frac, base, mask, mu_int)
                    if bridge_pack is not None:
                        use_bridge = True
                use_s3 = False
                s3_blob = None
                if (
                    effort >= 3
                    and level == 0
                    and min(height, width) >= 128
                    and not use_hist
                    and not use_bridge
                    and int(residual.size) >= 256
                ):
                    seed = _seed_mask(mask)
                    trial = encode_s3(residual, sig_idx, frac, nu, seed)
                    plain = encode_parametric_rans(residual, nu, sig_idx, frac)
                    if trial is not None and len(trial) < len(plain):
                        use_s3 = True
                        s3_blob = trial
                prepared.append(
                    (
                        mask,
                        residual,
                        nu,
                        sig_idx,
                        sig_map,
                        frac,
                        global_w,
                        overrides,
                        use_dir,
                        use_phi_mu,
                        use_d8,
                        use_head,
                        use_issm,
                        use_bridge,
                        bridge_pack,
                        use_tree,
                        tree,
                        use_hist,
                        hist_bytes,
                        ext_f,
                        ext_c,
                        use_s3,
                        s3_blob,
                        target,
                    )
                )
                decoded_odd[mask] = target
                known |= mask

            # Drop the acquisition header when no sub-coset used it.
            if not phi_used:
                phi_model = None
                for item in prepared:
                    if item[9]:
                        phi_used = True
                if not phi_used:
                    phi_model = None

            level_header = BitWriter()
            level_header.write_bits(feature_mask & 15, 4)
            if phi_model is None or not phi_used:
                level_header.write_bits(0, 1)
                phi_model = None
            else:
                level_header.write_bits(1, 1)
                write_phi(level_header, phi_model)
            header_bytes = level_header.finish()

            for item in prepared:
                (
                    mask,
                    residual,
                    nu,
                    sig_idx,
                    sig_map,
                    frac,
                    global_w,
                    overrides,
                    use_dir,
                    use_phi_mu,
                    use_d8,
                    use_head,
                    use_issm,
                    use_bridge,
                    bridge_pack,
                    use_tree,
                    tree,
                    use_hist,
                    hist_bytes,
                    ext_f,
                    ext_c,
                    use_s3,
                    s3_blob,
                    _target,
                ) = item
                if phi_model is None:
                    use_phi_mu = False
                meta.write_bits(1 if use_dir else 0, 1)
                meta.write_bits(1 if use_phi_mu else 0, 1)
                meta.write_bits(1 if use_d8 else 0, 1)
                meta.write_bits(1 if use_head else 0, 1)
                meta.write_bits(1 if use_issm else 0, 1)
                use_ridge = global_w is not None
                meta.write_bits(1 if use_ridge else 0, 1)
                meta.write_bits(1 if use_bridge else 0, 1)
                meta.write_bits(1 if overrides else 0, 1)
                meta.write_bits(int(nu) & 3, 2)
                if use_ridge:
                    for value in global_w.tolist():
                        meta.write_signed_golomb(int(value))
                if overrides:
                    bits = 0
                    for sector in overrides:
                        bits |= 1 << sector
                    meta.write_bits(bits, 8)
                    for sector in range(8):
                        if bits & (1 << sector):
                            for value in overrides[sector].tolist():
                                meta.write_signed_golomb(int(value))
                meta.write_bits(1 if use_tree else 0, 1)
                if use_tree and tree is not None:
                    write_tree(meta, tree)
                else:
                    _write_sig(meta, sig_map, int(residual.size))
                meta.write_bits(1 if use_hist else 0, 1)
                meta.write_bits(1 if use_s3 else 0, 1)
                if use_s3 and s3_blob is not None:
                    chosen = s3_blob
                elif use_bridge and bridge_pack is not None:
                    chosen = bridge_pack[0]
                elif use_hist and ext_f is not None and ext_c is not None:
                    rans = encode_parametric_rans(
                        residual, nu, sig_idx, frac, ext_freq=ext_f, ext_cum=ext_c
                    )
                    chosen = len(hist_bytes).to_bytes(2, "little") + hist_bytes + rans
                else:
                    chosen = encode_parametric_rans(residual, nu, sig_idx, frac)
                payloads.append(len(chosen).to_bytes(4, "little") + chosen)

            meta_bytes = meta.finish()
            level_blob = (
                len(header_bytes).to_bytes(2, "little")
                + header_bytes
                + len(meta_bytes).to_bytes(2, "little")
                + meta_bytes
                + b"".join(payloads)
            )
            out.append(level_blob)

            full = spatial_even.copy()
            full[odd] = decoded_odd[odd].astype(np.int32)
            resid_only = np.zeros_like(full)
            resid_only[odd] = decoded_odd[odd].astype(np.int32)
            current = lift_inverse(full, valid, sheaf=False)
            level_recons[level] = current
            level_resids[level] = resid_only
            prev_resid = resid_only

        ch_recons.append(level_recons)
        ch_resids.append(level_resids)

    return b"".join(out)


def _read_weights(reader: BitReader, n_feat: int) -> np.ndarray:
    return np.array([reader.read_signed_golomb() for _ in range(n_feat)], dtype=np.int64)


def decode_slgf_v3(
    payload: bytes,
    height: int,
    width: int,
    channels: int,
    halos: list | None = None,
) -> list[np.ndarray]:
    """Integer inverse of :func:`encode_slgf_v3`."""
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
            planes.append(decode_raster(payload[pos : pos + blen], (height, width), mask, True))
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
    coarse_ref: list[np.ndarray] = []

    for ch_idx in range(channels):
        if pos + 5 > len(payload):
            raise CodecError("truncated SLGF coarse header")
        c_flag = payload[pos]
        c_len = int.from_bytes(payload[pos + 1 : pos + 5], "little")
        pos += 5
        if pos + c_len > len(payload):
            raise CodecError("truncated SLGF coarse payload")
        blob = payload[pos : pos + c_len]
        pos += c_len
        ref = None
        if ch_idx > 0 and coarse_ref and coarse_ref[0].shape == coarse_valid.shape:
            ref = coarse_ref[0]
        if c_flag == 2:
            current = decode_coarse(blob, coarse_valid.shape, coarse_valid, ref)
        elif c_flag in (0, 1):
            current = decode_raster(blob, coarse_valid.shape, coarse_valid, c_flag == 1)
        else:
            raise CodecError("unknown coarse coder")
        coarse_ref.append(current)

        level_recons = [np.zeros((1, 1), np.int32) for _ in range(nlevels)]
        level_resids = [np.zeros((1, 1), np.int32) for _ in range(nlevels)]
        prev_resid = None
        halo = None if halos is None else halos[ch_idx]

        for level in range(nlevels - 1, -1, -1):
            valid, info, _pmask = geom[level]
            spatial_even = unpack_even(current, info, valid)
            odd, _even = parity_masks(valid)
            base = _features(spatial_even, valid, halo if level == 0 else None)
            coarse_proj = _coarse_resid_projection(prev_resid, info, valid)
            ch0_sp = ch_recons[0][level] if ch_idx >= 1 else None
            ch0_rs = ch_resids[0][level] if ch_idx >= 1 else None
            ch1_rs = ch_resids[1][level] if ch_idx >= 2 else None

            if pos + 2 > len(payload):
                raise CodecError("truncated level header")
            header_n = int.from_bytes(payload[pos : pos + 2], "little")
            pos += 2
            if pos + header_n > len(payload):
                raise CodecError("truncated level header")
            header = BitReader(payload[pos : pos + header_n])
            pos += header_n
            feature_mask = header.read_bits(4)
            phi_model = read_phi(header) if header.read_bits(1) == 1 else None
            phi_pred = None
            if phi_model is not None and ch0_sp is not None and ch0_sp.shape == spatial_even.shape:
                phi_pred = apply_phi(ch0_sp, phi_model)
            issm_map = None
            if feature_mask & _ISSM:
                filled = spatial_even.astype(np.int64).copy()
                filled[odd] = base["flat_pred"][odd]
                issm_map = context_map(filled)

            if pos + 2 > len(payload):
                raise CodecError("truncated sub-coset metadata")
            meta_n = int.from_bytes(payload[pos : pos + 2], "little")
            pos += 2
            if pos + meta_n > len(payload):
                raise CodecError("truncated sub-coset metadata")
            meta = BitReader(payload[pos : pos + meta_n])
            pos += meta_n

            sub_masks = _subcoset_masks(valid)
            decoded_odd = np.zeros(spatial_even.shape, dtype=np.int64)
            known = np.zeros(spatial_even.shape, dtype=np.bool_)
            for sub_idx, mask in enumerate(sub_masks):
                d8 = None
                if feature_mask & _D8:
                    samples = spatial_even.astype(np.int64).copy()
                    if np.any(known):
                        samples[known] = base["flat_pred"][known] + decoded_odd[known]
                    d8 = sheaf_kalman_delta(samples, known, mask, base["flat_pred"])
                base_phi, act = _assemble_features_and_activity(
                    base, sub_idx, mask, decoded_odd, coarse_proj, ch0_sp, ch0_rs, ch1_rs
                )
                phi = base_phi
                n_feat = int(phi.shape[1])
                mu0 = base["mu0_q4"][mask].astype(np.int64).copy()
                use_dir = meta.read_bits(1) == 1
                use_phi_mu = meta.read_bits(1) == 1
                use_d8 = meta.read_bits(1) == 1
                use_head = meta.read_bits(1) == 1
                use_issm = meta.read_bits(1) == 1
                use_ridge = meta.read_bits(1) == 1
                use_bridge = meta.read_bits(1) == 1
                use_regimes = meta.read_bits(1) == 1
                nu = meta.read_bits(2)
                if use_phi_mu and phi_pred is not None:
                    mu0 = (phi_pred[mask] - base["flat_pred"][mask]) << 4
                if use_dir:
                    mu0 = mu0 + (base["f_sheaf_q4"][mask] >> 1)
                if use_d8 and d8 is not None:
                    mu0 = mu0 + (d8[mask].astype(np.int64) << 4)
                if use_head and phi.shape[1] > 0:
                    mu0 = mu0 + project_head(phi)
                if use_issm and issm_map is not None:
                    mu0 = mu0 + issm_map[mask].astype(np.int64)
                global_w = _read_weights(meta, n_feat) if use_ridge else None
                overrides: dict[int, np.ndarray] = {}
                if use_regimes:
                    bits = meta.read_bits(8)
                    for sector in range(8):
                        if bits & (1 << sector):
                            overrides[sector] = _read_weights(meta, n_feat)
                sectors = _sectors(base, mask)
                mu_q4 = _apply_w(mu0, phi, sectors, global_w, overrides)
                mu_int, frac = _split_mu_q4(mu_q4)
                use_tree = meta.read_bits(1) == 1
                n_pts = int(mu0.size)
                if use_tree:
                    tree = read_tree(meta)
                    number_tree(tree)
                    qprops = _props(base, mask, decoded_odd, coarse_proj, ch0_rs)
                    sig_idx, leaf_ids = apply_tree(tree, qprops)
                    sig_map = None
                else:
                    tree = None
                    leaf_ids = None
                    sig_map = _read_sig(meta, n_pts)
                    buckets = _bucket_from_act_q4(act, int(base["act_shift"][0]))
                    sig_idx = sig_map[buckets]
                use_hist = meta.read_bits(1) == 1
                use_s3 = meta.read_bits(1) == 1
                if pos + 4 > len(payload):
                    raise CodecError("truncated sub-coset length")
                sub_n = int.from_bytes(payload[pos : pos + 4], "little")
                pos += 4
                if pos + sub_n > len(payload):
                    raise CodecError("truncated sub-coset stream")
                raw = payload[pos : pos + sub_n]
                pos += sub_n
                if use_s3:
                    seed = _seed_mask(mask)
                    residual = decode_s3(raw, n_pts, sig_idx, frac, nu, seed)
                elif use_hist:
                    if tree is None or leaf_ids is None:
                        raise CodecError("histogram without a context tree")
                    if len(raw) < 2:
                        raise CodecError("truncated histogram")
                    hist_n = int.from_bytes(raw[:2], "little")
                    if 2 + hist_n > len(raw):
                        raise CodecError("truncated histogram")
                    ext_f, ext_c = read_histograms(BitReader(raw[2 : 2 + hist_n]), leaf_ids, nu)
                    residual = decode_parametric_rans(
                        raw[2 + hist_n :],
                        n_pts,
                        nu,
                        sig_idx,
                        frac,
                        ext_freq=ext_f,
                        ext_cum=ext_c,
                    )
                elif use_bridge:
                    shifts = get_parametric_tables()[0]
                    bridge = base["is_bridge"][mask] & (shifts[sig_idx] == 0)
                    lo = (base["lo_nb"][mask] - mu_int).astype(np.int32)
                    hi = (base["hi_nb"][mask] - mu_int).astype(np.int32)
                    residual = decode_parametric_rans(
                        raw, n_pts, nu, sig_idx, frac, bridge, lo, hi
                    )
                else:
                    residual = decode_parametric_rans(raw, n_pts, nu, sig_idx, frac)
                decoded_odd[mask] = residual.astype(np.int64) + mu_int
                known |= mask

            full = spatial_even.copy()
            full[odd] = decoded_odd[odd].astype(np.int32)
            resid_only = np.zeros_like(full)
            resid_only[odd] = decoded_odd[odd].astype(np.int32)
            current = lift_inverse(full, valid, sheaf=False)
            level_recons[level] = current
            level_resids[level] = resid_only
            prev_resid = resid_only

        ch_recons.append(level_recons)
        ch_resids.append(level_resids)
        out_planes.append(current)

    if pos != len(payload):
        raise CodecError("trailing bytes in SLGF tile")
    return out_planes
