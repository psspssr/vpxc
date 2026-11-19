"""Comprehensive bijectivity, paper-conformance, and benchmark tests for vpxc."""

from __future__ import annotations

import unittest

import numpy as np

from vpxc.benchmark import benchmark_image, format_benchmark_table
from vpxc.codec import decode, encode
from vpxc.color import (
    MODE_SHEAR,
    MODE_SUB,
    MODE_WSIL,
    MODE_YCOCG,
    apply_color_forward,
    apply_color_inverse,
    choose_color,
    choose_color_transform,
    forward_channels,
    inverse_channels,
    ycocg_forward,
    ycocg_inverse,
)
from vpxc.entropy import (
    RANS_M,
    cumulative,
    decode_dyadic_rans,
    decode_parametric_rans,
    decode_rice_values,
    dyadic_shift,
    encode_dyadic_rans,
    encode_parametric_rans,
    encode_rice_values,
    eval_student_cdf_q24,
    join_dyadic,
    normalize_freqs,
    s3_bb_decode,
    s3_bb_encode,
    split_dyadic,
)
from vpxc.issm import scan_4way, sequential, tree, tree_rows
from vpxc.lattice import (
    build_levels,
    forward_tower,
    ilog2_int64,
    inverse_tower,
    lift_forward,
    lift_inverse,
    pack_even,
    predict,
    unpack_even,
)
from vpxc.spatial import decode_raster, encode_raster
from vpxc.wsil import (
    SL3Params,
    estimate_sl3_wsil,
    factor_sl3_matrix,
    q12,
    r12,
    sl3_forward,
    sl3_forward_arrays,
    sl3_inverse,
    sl3_inverse_arrays,
    wsil2_forward,
    wsil2_inverse,
)


class TestWSIL(unittest.TestCase):
    def test_s_equals_one_is_identity(self):
        one = 1 << 12
        rng = np.random.default_rng(0)
        xs = rng.integers(-1000, 1001, size=(5000, 2))
        err1 = []
        err2 = []
        for x1, x2 in xs:
            y1, y2 = wsil2_forward(int(x1), int(x2), one, one)
            back1, back2 = wsil2_inverse(y1, y2, one, one)
            self.assertEqual((back1, back2), (int(x1), int(x2)))
            err1.append(y1 - int(x1))
            err2.append(y2 - int(x2))
        self.assertEqual(float(np.var(err1)), 0.0)
        self.assertEqual(float(np.var(err2)), 0.0)

    def test_random_multipliers_are_bijective(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            s = int(rng.integers(1, 20000))
            s_inv = int(rng.integers(1, 20000))
            for _sample in range(20):
                x1 = int(rng.integers(-2000, 2000))
                x2 = int(rng.integers(-2000, 2000))
                y = wsil2_forward(x1, x2, s, s_inv)
                self.assertEqual(wsil2_inverse(*y, s, s_inv), (x1, x2))

    def test_rounding_matches_floor_half_up(self):
        self.assertEqual(r12(4096, 5), 5)
        self.assertEqual(r12(0, -7), 0)
        self.assertEqual(r12(-2048, 1), 0)
        self.assertEqual(r12(-2049, 1), -1)

    def test_sl3_scalar_and_array_match(self):
        rng = np.random.default_rng(2)
        perms = [(0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1)]
        for _ in range(40):
            raw_p = [int(rng.integers(-6000, 6000)) for _ in range(6)] + [
                int(rng.integers(512, 8000)) for _ in range(4)
            ]
            perm = perms[int(rng.integers(0, len(perms)))]
            sp = SL3Params(perm, *raw_p)
            arr = rng.integers(-2000, 2000, size=(3, 6, 5), dtype=np.int32)
            fwd = sl3_forward_arrays([arr[0], arr[1], arr[2]], sp)
            inv = sl3_inverse_arrays(fwd, sp)
            for c in range(3):
                self.assertTrue(np.array_equal(inv[c], arr[c]))
            # Match scalar Appendix A.1 implementation point-by-point:
            pt = [int(arr[0, 2, 3]), int(arr[1, 2, 3]), int(arr[2, 2, 3])]
            sy = sl3_forward(pt, perm, *raw_p)
            self.assertEqual(
                sy, (int(fwd[0][2, 3]), int(fwd[1][2, 3]), int(fwd[2][2, 3]))
            )
            self.assertEqual(list(sl3_inverse(sy, perm, *raw_p)), pt)

    def test_scale_125_rounding_stays_small(self):
        s = q12(1.25)
        s_inv = q12(1.0 / 1.25)
        rng = np.random.default_rng(3)
        e1 = []
        e2 = []
        for x1, x2 in rng.integers(-1000, 1001, size=(4000, 2)):
            y1, y2 = wsil2_forward(int(x1), int(x2), s, s_inv)
            self.assertEqual(wsil2_inverse(y1, y2, s, s_inv), (int(x1), int(x2)))
            e1.append(y1 - 1.25 * int(x1))
            e2.append(y2 - (1.0 / 1.25) * int(x2))
        self.assertLess(float(np.var(e1)), 1.0)
        self.assertLess(float(np.var(e2)), 1.0)

    def test_rotated_ccm_wsil_beats_ycocg(self):
        rng = np.random.default_rng(99)
        # Smooth opponent fields with stds (35, 4.5, 1.8) mixed through a rotated SL(3) CCM:
        h, w = 64, 64
        lat = np.stack(
            [
                rng.normal(0.0, 35.0, size=(h, w)),
                rng.normal(0.0, 4.5, size=(h, w)),
                rng.normal(0.0, 1.8, size=(h, w)),
            ],
            axis=-1,
        )
        a1, a2 = 0.72, -0.55
        r1 = np.array([[np.cos(a1), -np.sin(a1), 0], [np.sin(a1), np.cos(a1), 0], [0, 0, 1]])
        r2 = np.array([[1, 0, 0], [0, np.cos(a2), -np.sin(a2)], [0, np.sin(a2), np.cos(a2)]])
        ccm = r1 @ np.diag([1.3, 0.9, 1.0 / (1.3 * 0.9)]) @ r2
        rgb_arr = np.round(lat @ ccm.T + 500.0).astype(np.int32)
        chans = [rgb_arr[:, :, i] for i in range(3)]

        wsil_p = factor_sl3_matrix(np.linalg.inv(ccm))
        y_wsil = sl3_forward_arrays(chans, wsil_p)
        back = sl3_inverse_arrays(y_wsil, wsil_p)
        for c in range(3):
            self.assertTrue(np.array_equal(back[c], chans[c]))

        y_ycocg = ycocg_forward(*chans)
        std_wsil = sum(float(np.std(p)) for p in y_wsil)
        std_ycocg = sum(float(np.std(p)) for p in y_ycocg)
        self.assertLess(std_wsil, std_ycocg)


class TestColor(unittest.TestCase):
    def test_ycocg_roundtrip(self):
        rng = np.random.default_rng(4)
        r = rng.integers(0, 256, size=(40, 30), dtype=np.int32)
        g = rng.integers(0, 256, size=(40, 30), dtype=np.int32)
        b = rng.integers(0, 256, size=(40, 30), dtype=np.int32)
        back = ycocg_inverse(*ycocg_forward(r, g, b))
        self.assertTrue(np.array_equal(back[0], r))
        self.assertTrue(np.array_equal(back[1], g))
        self.assertTrue(np.array_equal(back[2], b))

    def test_choose_color_and_4channel_inverse(self):
        rng = np.random.default_rng(5)
        rgb = [rng.integers(0, 256, size=(32, 24), dtype=np.int32) for _ in range(3)]
        mode, perm, shear = choose_color(rgb)
        planes = forward_channels(rgb, mode, perm, shear)
        back = inverse_channels(planes, mode, perm, shear)
        for got, expect in zip(back, rgb):
            self.assertTrue(np.array_equal(got, expect))
        self.assertIn(mode, (0, MODE_YCOCG, MODE_SUB, MODE_SHEAR))

        # 4-channel correlated Bayer / RGBA transform:
        g2 = np.clip(rgb[1] + rng.integers(-3, 4, size=(32, 24), dtype=np.int32), 0, 255)
        rgba = rgb + [g2]
        tr = choose_color_transform(rgba)
        lifted4 = apply_color_forward(rgba, tr)
        restored4 = apply_color_inverse(lifted4, tr)
        for got, expect in zip(restored4, rgba):
            self.assertTrue(np.array_equal(got, expect))


class TestLattice(unittest.TestCase):
    def test_q_squared_equals_2i_dense_packing(self):
        # Verify the paper's Q = [[1, 1], [1, -1]] (Q^2 = 2I) packs even levels
        # onto 100%-dense (H/2, W/2) and (H/4, W/4) grids with zero dead rows.
        levels, coarse_valid = build_levels(256, 256, 4)
        self.assertEqual(len(levels), 4)
        self.assertEqual(levels[1][2].shape, (128, 128))
        self.assertTrue(bool(np.all(levels[1][2])))
        self.assertEqual(coarse_valid.shape, (64, 64))
        self.assertTrue(bool(np.all(coarse_valid)))

    def test_sheaf_breaks_step_edge_symmetry_and_uint16_scale(self):
        # Vertical contour: W=0, E=100, N=0, S=0 around odd site (2, 3):
        vert = np.zeros((6, 6), dtype=np.int32)
        vert[:, 4:] = 100
        valid = np.ones((6, 6), dtype=np.bool_)
        p_flat = predict(vert, valid, sheaf=False)
        p_sheaf = predict(vert, valid, sheaf=True)
        self.assertEqual(int(p_flat[2, 3]), 25)
        self.assertLessEqual(int(p_sheaf[2, 3]), 2)

        # 45-degree diagonal step edge where 2 neighbors are 0 and 2 are 100:
        diag = np.zeros((8, 8), dtype=np.int32)
        for r in range(8):
            for c in range(8):
                if r + c >= 7:
                    diag[r, c] = 100
        p_diag_sheaf = predict(diag, np.ones((8, 8), dtype=np.bool_), sheaf=True)
        # Must pick a coherent side (0 or 100), not the blurred midpoint 50:
        self.assertIn(int(p_diag_sheaf[3, 4]), (0, 100))

        # Integer log2 matches floor(log2(x)) across powers of two:
        vals = np.array([1, 2, 3, 4, 7, 8, 15, 16, 1023, 1024, 65536], dtype=np.int64)
        expect = np.array([0, 1, 1, 2, 2, 3, 3, 4, 9, 10, 16], dtype=np.int32)
        self.assertTrue(np.array_equal(ilog2_int64(vals), expect))

    def test_pack_is_bijective(self):
        rng = np.random.default_rng(6)
        for h, w in ((1, 1), (1, 5), (5, 1), (2, 2), (3, 5), (8, 8), (7, 6), (17, 13)):
            img = rng.integers(-50, 50, size=(h, w), dtype=np.int32)
            valid = np.ones((h, w), np.bool_)
            packed, mask, info = pack_even(img, valid)
            spatial = unpack_even(packed, info, valid)
            rr, cc = np.indices((h, w))
            even = ((rr + cc) & 1) == 0
            self.assertTrue(np.array_equal(spatial[even], img[even]))
            self.assertEqual(int(mask.sum()), int(even.sum()))

    def test_lift_and_tower_roundtrip(self):
        rng = np.random.default_rng(7)
        shapes = ((1, 1), (2, 2), (3, 3), (3, 5), (8, 8), (15, 17), (32, 31), (64, 40))
        for sheaf in (False, True):
            for h, w in shapes:
                img = rng.integers(0, 256, size=(h, w), dtype=np.int32)
                valid = np.ones((h, w), np.bool_)
                self.assertTrue(
                    np.array_equal(
                        lift_inverse(lift_forward(img, valid, sheaf), valid, sheaf), img
                    )
                )
                coarse, _cvalid, details, geom = forward_tower(img, 4, sheaf)
                back = inverse_tower(coarse, details, geom, sheaf)
                self.assertTrue(np.array_equal(back, img), f"sheaf={sheaf} shape={(h, w)}")
                geom2, _ = build_levels(h, w, len(geom))
                self.assertEqual(len(geom2), len(geom))


class TestEntropy(unittest.TestCase):
    def test_dyadic_shift_table(self):
        self.assertEqual(dyadic_shift(1), 0)
        self.assertEqual(dyadic_shift(2), 0)
        self.assertEqual(dyadic_shift(8), 2)
        self.assertEqual(dyadic_shift(25), 4)
        self.assertEqual(dyadic_shift(120), 6)
        self.assertLessEqual(dyadic_shift(10**6), 12)

    def test_split_join(self):
        for shift in range(0, 9):
            for residual in range(-80, 81):
                q, u = split_dyadic(residual, shift)
                if shift:
                    self.assertGreaterEqual(u, 0)
                    self.assertLess(u, 1 << shift)
                self.assertEqual(join_dyadic(q, u, shift), residual)

    def test_rice_masked_roundtrip(self):
        rng = np.random.default_rng(8)
        values = rng.integers(-40, 40, size=(20, 19), dtype=np.int32)
        valid = rng.random((20, 19)) > 0.3
        ctx = rng.integers(0, 8, size=values.shape, dtype=np.uint8)
        for mu in (False, True):
            blob = encode_rice_values(values, valid, ctx, mu)
            back = decode_rice_values(blob, values.shape, valid, ctx, mu)
            self.assertTrue(np.array_equal(back[valid], values[valid]))

    def test_rans_frequency_partition_and_pchip_monotonicity(self):
        freqs = normalize_freqs(np.zeros(66, dtype=np.int64))
        self.assertEqual(int(freqs.sum()), 65536)
        self.assertTrue(np.all(freqs >= 1))

        # Extreme Dirac spike where 65 bins clamp to 1 (deficit == -65):
        spike = np.zeros(66, dtype=np.int64)
        spike[32] = 1_000_000
        f_spike = normalize_freqs(spike)
        self.assertEqual(int(f_spike.sum()), 65536)
        self.assertEqual(int(f_spike[32]), 65536 - 65)

        z_grid = np.arange(-160000, 160001, 257, dtype=np.int64)
        for nu_idx in range(4):
            cdf = eval_student_cdf_q24(z_grid, nu_idx)
            self.assertTrue(np.all(np.diff(cdf) >= 0))
            self.assertEqual(int(eval_student_cdf_q24(np.array([0]), nu_idx)[0]), 1 << 23)

    def test_parametric_rans_with_tilted_remainders_and_bridges(self):
        rng = np.random.default_rng(91)
        n = 1500
        # Mix zero-shift, medium-shift, large 16-bit escapes, and bridge sites:
        r_tilde = rng.integers(-600, 600, size=n, dtype=np.int32)
        r_tilde[:700] = rng.integers(-6, 7, size=700, dtype=np.int32)
        sig_idx = rng.integers(0, 26, size=n, dtype=np.int32)
        sig_idx[:700] = rng.integers(0, 11, size=700, dtype=np.int32)
        frac_idx = rng.integers(0, 16, size=n, dtype=np.int32)
        bridge_mask = np.zeros(n, dtype=np.bool_)
        bridge_mask[50:150] = True
        bridge_lo = np.full(n, -12, dtype=np.int32)
        bridge_hi = np.full(n, 14, dtype=np.int32)

        for nu_idx in range(4):
            blob = encode_parametric_rans(
                r_tilde, nu_idx, sig_idx, frac_idx, bridge_mask, bridge_lo, bridge_hi
            )
            back = decode_parametric_rans(
                blob, n, nu_idx, sig_idx, frac_idx, bridge_mask, bridge_lo, bridge_hi
            )
            self.assertTrue(np.array_equal(back, r_tilde))

    def test_s3_bb_rans_zero_waste_roundtrip(self):
        rng = np.random.default_rng(77)
        # 256 isolated seed symbols in 0..15 (~1024 bits of seed entropy, > 384 bits)
        n_seed = 256
        n_main = 512
        seed_syms = rng.integers(0, 16, size=n_seed).tolist()
        main_syms = rng.integers(0, 8, size=n_main).tolist()

        q_seed = normalize_freqs(np.ones(16, dtype=np.int64), nsym=16, total=RANS_M)
        prior_z = normalize_freqs(np.array([5, 3, 2, 1], dtype=np.int64), nsym=4, total=RANS_M)
        post_table = [
            normalize_freqs(np.array([10, 4, 1, 1], dtype=np.int64), nsym=4, total=RANS_M),
            normalize_freqs(np.array([2, 9, 3, 2], dtype=np.int64), nsym=4, total=RANS_M),
        ]
        p_main_by_z = [
            normalize_freqs(np.arange(8, 0, -1, dtype=np.int64), nsym=8, total=RANS_M),
            normalize_freqs(np.ones(8, dtype=np.int64), nsym=8, total=RANS_M),
            normalize_freqs(np.arange(1, 9, dtype=np.int64), nsym=8, total=RANS_M),
            normalize_freqs(np.array([8, 1, 8, 1, 8, 1, 8, 1], dtype=np.int64), nsym=8, total=RANS_M),
        ]

        def seed_fn(_ctx):
            return q_seed

        def post_fn(ctx_parity: int):
            return post_table[ctx_parity & 1]

        ctx_val = sum(main_syms) & 1

        def enc_main_fn(z_val: int):
            f = p_main_by_z[z_val]
            c = cumulative(f)
            return main_syms, [int(c[s]) for s in main_syms], [int(f[s]) for s in main_syms]

        blob, z_enc = s3_bb_encode(
            seed_syms, seed_fn, post_fn, enc_main_fn, prior_z, ctx_val
        )

        def dec_main_fn(stack, z_val: int, count: int):
            f = p_main_by_z[z_val]
            c = cumulative(f)
            rec = [stack.pop(c, f) for _ in range(count)]
            return rec, (sum(rec) & 1)

        rec_main, rec_seed, z_dec = s3_bb_decode(
            blob, n_main, n_seed, seed_fn, post_fn, dec_main_fn, prior_z
        )
        self.assertEqual(z_dec, z_enc)
        self.assertEqual(rec_main, main_syms)
        self.assertEqual(rec_seed, seed_syms)

    def test_dyadic_rans_roundtrip(self):
        rng = np.random.default_rng(9)
        residuals = rng.integers(-200, 200, size=(30, 28), dtype=np.int32)
        valid = np.ones(residuals.shape, np.bool_)
        valid[::5, ::4] = False
        shifts = rng.integers(0, 6, size=residuals.shape, dtype=np.uint8)
        blob = encode_dyadic_rans(residuals, valid, shifts)
        back = decode_dyadic_rans(blob, residuals.shape, valid, shifts)
        self.assertTrue(np.array_equal(back[valid], residuals[valid]))


class TestSpatial(unittest.TestCase):
    def test_raster_and_d8_kalman(self):
        rng = np.random.default_rng(10)
        plane = rng.integers(-20, 280, size=(36, 25), dtype=np.int32)
        valid = np.ones(plane.shape, np.bool_)
        valid[:, 3] = False
        for kalman in (False, True):
            blob = encode_raster(plane, valid, kalman)
            back = decode_raster(blob, plane.shape, valid, kalman)
            self.assertTrue(np.array_equal(back[valid], plane[valid]))

        # Noiseless synthetic repeated pattern: Kalman should beat plain MED raster:
        motif = np.array([[10, 80, 200, 40], [80, 220, 30, 150], [200, 30, 90, 10], [40, 150, 10, 240]], dtype=np.int32)
        tiled = np.tile(motif, (12, 12))
        m_full = np.ones(tiled.shape, dtype=np.bool_)
        b_med = encode_raster(tiled, m_full, kalman=False)
        b_kal = encode_raster(tiled, m_full, kalman=True)
        self.assertTrue(np.array_equal(decode_raster(b_kal, tiled.shape, m_full, True), tiled))
        self.assertLess(len(b_kal), len(b_med))


class TestISSM(unittest.TestCase):
    def test_tree_tracks_sequential_and_batch_matches_scalar(self):
        for seed in (0, 11, 21):
            rng = np.random.default_rng(seed)
            decay = rng.integers(0, 65536 - 64, size=256, dtype=np.int64)
            bias = rng.integers(-500, 500, size=256, dtype=np.int64)
            gap = int(np.max(np.abs(sequential(decay, bias) - tree(decay, bias))))
            self.assertLessEqual(gap, 1)

        # Verify vectorized tree_rows is bit-for-bit identical (0 LSB) to scalar tree:
        rng = np.random.default_rng(33)
        for length in (1, 7, 16, 17, 48, 99):
            d2 = rng.integers(0, 65536 - 64, size=(6, length), dtype=np.int64)
            b2 = rng.integers(-2000, 2000, size=(6, length), dtype=np.int64)
            vec = tree_rows(d2, b2)
            for row_i in range(6):
                self.assertTrue(np.array_equal(vec[row_i], tree(d2[row_i], b2[row_i])))

        plane = rng.integers(0, 256, size=(20, 23), dtype=np.int32)
        h_e, h_w, h_s, h_n = scan_4way(plane)
        self.assertEqual(h_e.shape, plane.shape)
        self.assertEqual(h_n.dtype, np.int64)

    def test_tree_matches_left_associated_composition(self):
        from vpxc.issm import CHUNK, Q30_ONE, _chunk_map, _compose

        rng = np.random.default_rng(104)
        decay = rng.integers(65536 - 200, 65536 - 64, size=256, dtype=np.int64)
        bias = rng.integers(-50, 50, size=256, dtype=np.int64)
        n = len(decay)
        n_chunks = (n + CHUNK - 1) // CHUNK
        maps = [_chunk_map(decay, bias, c * CHUNK, CHUNK) for c in range(n_chunks)]
        acc = (Q30_ONE, 0)
        folded = np.empty(n, np.int64)
        for c, chunk_map in enumerate(maps):
            state = acc[1]
            for k in range(CHUNK):
                t = c * CHUNK + k
                state = ((int(decay[t]) * state + (1 << 15)) >> 16) + int(bias[t])
                folded[t] = state
            acc = _compose(chunk_map, acc)
        gap = int(np.max(np.abs(folded - tree(decay, bias))))
        self.assertLessEqual(gap, 4)


class TestCodecAndBenchmark(unittest.TestCase):
    def test_roundtrip_shapes_and_profiles(self):
        rng = np.random.default_rng(12)
        shapes = [(1, 1), (1, 9), (9, 1), (7, 5), (16, 16), (17, 13), (40, 31)]
        profiles = [None, "slgf", "raw", "raster", "kalman", "qx-rice", "qx-rans"]
        for h, w in shapes:
            image = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
            for profile in profiles:
                blob = encode(image, tile=32, effort=2, profile=profile)
                back = decode(blob)
                self.assertTrue(np.array_equal(back, image), f"{(h, w)} {profile}")
            gray = image[:, :, 1]
            self.assertTrue(np.array_equal(decode(encode(gray, tile=16, effort=1)), gray))

    def test_uint16_and_rgba(self):
        rng = np.random.default_rng(13)
        wide = rng.integers(0, 60000, size=(34, 30), dtype=np.uint16)
        self.assertTrue(np.array_equal(decode(encode(wide, tile=32, effort=2)), wide))
        wide_rgb = rng.integers(0, 65535, size=(24, 26, 3), dtype=np.uint16)
        self.assertTrue(np.array_equal(decode(encode(wide_rgb, tile=32, profile="slgf")), wide_rgb))
        rgba = rng.integers(0, 256, size=(20, 18, 4), dtype=np.uint8)
        self.assertTrue(np.array_equal(decode(encode(rgba, tile=16, profile="slgf")), rgba))

    def test_palette_grayscale_and_512_colors_and_expansion_bound(self):
        # Previously broken grayscale palette case:
        gray_ui = np.zeros((32, 32), np.uint8)
        gray_ui[::2, ::2] = 240
        gray_ui[4:20, 6:24] = 120
        b_gray = encode(gray_ui, tile=32, profile="palette")
        self.assertTrue(np.array_equal(decode(b_gray), gray_ui))

        # 400 unique colors (> 256, <= 512):
        vals = np.arange(400, dtype=np.int32)
        grid = vals[np.arange(32 * 32) % 400].reshape(32, 32)
        rgb_400 = np.stack([(grid & 0xFF), ((grid >> 2) & 0xFF), ((grid * 7) & 0xFF)], axis=-1).astype(np.uint8)
        b_400 = encode(rgb_400, tile=32, profile="palette")
        self.assertTrue(np.array_equal(decode(b_400), rgb_400))

        rng = np.random.default_rng(14)
        noise = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
        compressed = encode(noise, tile=32, effort=2)
        self.assertLessEqual(len(compressed), noise.size + 64)

        gradient = np.arange(64, dtype=np.uint8)
        smooth = np.broadcast_to(gradient, (64, 64)).copy()
        smooth = np.stack([smooth, smooth // 2, 255 - smooth], axis=-1)
        blob = encode(smooth, tile=64, effort=2)
        self.assertTrue(np.array_equal(decode(blob), smooth))
        self.assertLess(len(blob), smooth.size // 3)

    def test_benchmark_harness(self):
        gradient = np.arange(48, dtype=np.uint8)
        smooth = np.broadcast_to(gradient, (48, 48)).copy()
        rgb = np.stack([smooth, smooth // 2, 255 - smooth], axis=-1)
        res = benchmark_image(rgb, tile=64, effort=2, include_external=True)
        self.assertGreaterEqual(len(res), 1)
        self.assertTrue(any(r.codec.startswith("vpxc") for r in res))
        md = format_benchmark_table(res, markdown=True)
        txt = format_benchmark_table(res, markdown=False)
        self.assertIn("bpsp", md)
        self.assertIn("bpsp", txt)

    def test_corrupt_header_rejected(self):
        image = np.zeros((8, 8, 3), np.uint8)
        blob = bytearray(encode(image, tile=16, profile="raw"))
        blob[0] = ord("X")
        with self.assertRaises(Exception):
            decode(bytes(blob))
        blob = bytearray(encode(image, tile=16, profile="slgf"))
        blob[-1] ^= 0xFF
        with self.assertRaises(Exception):
            decode(bytes(blob))


if __name__ == "__main__":
    unittest.main()
