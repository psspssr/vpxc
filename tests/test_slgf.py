"""Byte-exact tests for the integer SLGF-Zero pipeline."""

from __future__ import annotations

import unittest

import numpy as np

from vpxc.codec import decode, encode
from vpxc.color import (
    MODE_SHEAR,
    MODE_SUB,
    MODE_YCOCG,
    choose_color,
    forward_channels,
    inverse_channels,
    ycocg_forward,
    ycocg_inverse,
)
from vpxc.entropy import (
    decode_dyadic_rans,
    decode_rice_values,
    dyadic_shift,
    encode_dyadic_rans,
    encode_rice_values,
    join_dyadic,
    normalize_freqs,
    split_dyadic,
)
from vpxc.issm import sequential, tree
from vpxc.lattice import (
    build_levels,
    forward_tower,
    inverse_tower,
    lift_forward,
    lift_inverse,
    pack_even,
    unpack_even,
)
from vpxc.spatial import decode_raster, encode_raster
from vpxc.wsil import q12, r12, sl3_forward, sl3_inverse, wsil2_forward, wsil2_inverse


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
        # -0.5 in Q12 rounds toward +infinity, i.e. to 0.
        self.assertEqual(r12(-2048, 1), 0)
        self.assertEqual(r12(-2049, 1), -1)

    def test_sl3_random_bijection(self):
        rng = np.random.default_rng(2)
        perms = [(0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1)]
        for _ in range(100):
            params = [int(rng.integers(-8000, 8000)) for _ in range(10)]
            perm = perms[int(rng.integers(0, len(perms)))]
            for _sample in range(5):
                x = [int(rng.integers(-3000, 3000)) for _ in range(3)]
                y = sl3_forward(x, perm, *params)
                back = sl3_inverse(y, perm, *params)
                self.assertEqual(list(back), x)

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

    def test_choose_color_inverse(self):
        rng = np.random.default_rng(5)
        rgb = [rng.integers(0, 256, size=(32, 24), dtype=np.int32) for _ in range(3)]
        # Give green a shifted copy of red so a shear is meaningful but still invertible.
        mode, perm, shear = choose_color(rgb)
        planes = forward_channels(rgb, mode, perm, shear)
        back = inverse_channels(planes, mode, perm, shear)
        for got, expect in zip(back, rgb):
            self.assertTrue(np.array_equal(got, expect))
        self.assertIn(mode, (0, MODE_YCOCG, MODE_SUB, MODE_SHEAR))


class TestLattice(unittest.TestCase):
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
                    np.array_equal(lift_inverse(lift_forward(img, valid, sheaf), valid, sheaf), img)
                )
                coarse, _cvalid, details, geom = forward_tower(img, 4, sheaf)
                back = inverse_tower(coarse, details, geom, sheaf)
                self.assertTrue(np.array_equal(back, img), f"sheaf={sheaf} shape={(h, w)}")
                # Geometry is a function of shape, so a second build matches the tower depth.
                geom2, _ = build_levels(h, w, len(geom))
                self.assertEqual(len(geom2), len(geom))


class TestEntropy(unittest.TestCase):
    def test_dyadic_shift_table(self):
        # Integers on the audit thresholds: 2→0 (2.5 boundary), 8→2, 25→4, 120→6.
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

    def test_rans_frequency_partition(self):
        freqs = normalize_freqs(np.zeros(66))
        self.assertEqual(int(freqs.sum()), 65536)
        self.assertTrue(np.all(freqs >= 1))
        skewed = np.zeros(66, np.int64)
        skewed[32] = 10000
        skewed[33] = 100
        freqs = normalize_freqs(skewed)
        self.assertEqual(int(freqs.sum()), 65536)

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
    def test_raster_and_kalman(self):
        rng = np.random.default_rng(10)
        plane = rng.integers(-20, 280, size=(36, 25), dtype=np.int32)
        valid = np.ones(plane.shape, np.bool_)
        valid[:, 3] = False
        for kalman in (False, True):
            blob = encode_raster(plane, valid, kalman)
            back = decode_raster(blob, plane.shape, valid, kalman)
            self.assertTrue(np.array_equal(back[valid], plane[valid]))


class TestISSM(unittest.TestCase):
    def test_tree_tracks_sequential(self):
        for seed in (0, 11, 21):
            rng = np.random.default_rng(seed)
            decay = rng.integers(0, 65536 - 64, size=256, dtype=np.int64)
            bias = rng.integers(-500, 500, size=256, dtype=np.int64)
            gap = int(np.max(np.abs(sequential(decay, bias) - tree(decay, bias))))
            self.assertLessEqual(gap, 1)
        # Partial final chunk and an empty input stay defined.
        decay = np.zeros(20, np.int64)
        bias = np.zeros(20, np.int64)
        self.assertEqual(tree(decay, bias).shape, (20,))
        self.assertEqual(tree(decay[:0], bias[:0]).shape, (0,))

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


class TestCodec(unittest.TestCase):
    def test_roundtrip_shapes_and_profiles(self):
        rng = np.random.default_rng(12)
        shapes = [(1, 1), (1, 9), (9, 1), (7, 5), (16, 16), (17, 13), (40, 31)]
        profiles = [None, "raw", "raster", "kalman", "qx-rice", "qx-rans"]
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
        wide = rng.integers(0, 5000, size=(18, 15), dtype=np.uint16)
        self.assertTrue(np.array_equal(decode(encode(wide, tile=16, effort=2)), wide))
        rgba = rng.integers(0, 256, size=(15, 14, 4), dtype=np.uint8)
        self.assertTrue(np.array_equal(decode(encode(rgba, tile=16, profile="qx-rice")), rgba))

    def test_palette_and_expansion_bound(self):
        tile = np.zeros((24, 24, 3), np.uint8)
        tile[:12, :, 0] = 200
        tile[12:, :, 1] = 80
        tile[:, ::2, 2] = 40
        blob = encode(tile, tile=32, profile="palette")
        self.assertTrue(np.array_equal(decode(blob), tile))
        rng = np.random.default_rng(14)
        noise = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
        compressed = encode(noise, tile=32, effort=2)
        self.assertLessEqual(len(compressed), noise.size + 64)
        gradient = np.arange(64, dtype=np.uint8)
        smooth = np.broadcast_to(gradient, (64, 64)).copy()
        smooth = np.stack([smooth, smooth // 2, 255 - smooth], axis=-1)
        blob = encode(smooth, tile=64, effort=2)
        self.assertTrue(np.array_equal(decode(blob), smooth))
        self.assertLess(len(blob), smooth.size // 2)

    def test_corrupt_header_rejected(self):
        image = np.zeros((8, 8, 3), np.uint8)
        blob = bytearray(encode(image, tile=16, profile="raw"))
        blob[0] = ord("X")
        with self.assertRaises(Exception):
            decode(bytes(blob))
        blob = bytearray(encode(image, tile=16, profile="raster"))
        blob[-1] ^= 0xFF
        with self.assertRaises(Exception):
            decode(bytes(blob))


if __name__ == "__main__":
    unittest.main()
