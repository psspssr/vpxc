"""Round-trips and invariants for the version-3 SLGF workstreams."""

from __future__ import annotations

import ast
import hashlib
import unittest

import numpy as np

from vpxc.acquisition import apply_phi, fit_phi, read_phi, write_phi
from vpxc.bitsback import decode_s3, encode_s3
from vpxc.coarse import decode_coarse, encode_coarse
from vpxc.codec import decode, encode
from vpxc.entropy import BitReader, BitWriter, cumulative, normalize_freqs
from vpxc.interleaved import decode_pairs, encode_symbols, symbol_lut
from vpxc.issm import sequential
from vpxc.issm_int8 import load_weights, pack_weights
from vpxc.lattice import parity_masks, predict
from vpxc.trees import apply_tree, build_sigma_tree, quantize_props, read_tree, write_tree


def _assert_no_true_division(func) -> None:
    tree = ast.parse(inspect_source(func))
    for node in ast.walk(tree):
        if isinstance(node, ast.Div):
            raise AssertionError(f"{func.__name__} uses true division")


def inspect_source(func) -> str:
    import inspect

    return inspect.getsource(func)


class TestCoarseAndContainer(unittest.TestCase):
    def test_coarse_roundtrip_with_reference(self):
        rng = np.random.default_rng(3)
        plane = rng.integers(0, 256, size=(36, 28), dtype=np.int32)
        ref = rng.integers(0, 256, size=plane.shape, dtype=np.int32)
        valid = np.ones(plane.shape, np.bool_)
        valid[:, 4] = False
        blob = encode_coarse(plane, valid, ref)
        back = decode_coarse(blob, plane.shape, valid, ref)
        self.assertTrue(np.array_equal(back[valid], plane[valid]))

    def test_version2_still_decodes_and_version3_multitile(self):
        rng = np.random.default_rng(4)
        image = rng.integers(0, 256, size=(70, 80, 3), dtype=np.uint8)
        legacy = encode(image, tile=32, effort=1, version=2, profile="slgf")
        self.assertEqual(legacy[4], 2)
        self.assertTrue(np.array_equal(decode(legacy), image))
        modern = encode(image, tile=32, effort=2, version=3)
        self.assertEqual(modern[4], 3)
        self.assertTrue(np.array_equal(decode(modern), image))
        self.assertLessEqual(len(encode(image, tile=32, effort=2, version=3, profile="slgf")), len(legacy) + 64)

    def test_noise_expansion_and_slgf_zero(self):
        rng = np.random.default_rng(5)
        noise = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
        blob = encode(noise, tile=32, effort=2)
        self.assertTrue(np.array_equal(decode(blob), noise))
        self.assertLessEqual(len(blob), noise.size + 64)
        forced = encode(noise, tile=32, effort=2, profile="slgf-zero")
        self.assertTrue(np.array_equal(decode(forced), noise))


class TestTreesPhiMatchingBits(unittest.TestCase):
    def test_maniac_tree_roundtrip(self):
        rng = np.random.default_rng(6)
        props = quantize_props(rng.integers(-40, 40, size=(300, 4)))
        props[:150, 0] = 0
        props[150:, 0] = 6
        resid = rng.integers(-8, 9, size=300)
        resid[:150] = rng.integers(-1, 2, size=150)
        tree = build_sigma_tree(props, resid)
        self.assertIsNotNone(tree)
        sigma, _leaves = apply_tree(tree, props)
        writer = BitWriter()
        write_tree(writer, tree)
        restored = read_tree(BitReader(writer.finish()))
        sigma2, _ = apply_tree(restored, props)
        self.assertTrue(np.array_equal(sigma, sigma2))

    def test_phi_header_and_integer_apply(self):
        yy, xx = np.mgrid[0:80, 0:80]
        luma = np.clip(30 + yy * 2 + xx, 0, 255).astype(np.int32)
        chroma = np.clip(luma // 2 + (yy & 1) * 18 + (xx & 1) * 8, 0, 255).astype(np.int32)
        odd, _even = parity_masks(np.ones(luma.shape, np.bool_))
        flat = predict(chroma, np.ones(chroma.shape, np.bool_), False).astype(np.int64)
        phi = fit_phi(luma, chroma, flat, odd)
        self.assertIsNotNone(phi)
        writer = BitWriter()
        write_phi(writer, phi)
        again = read_phi(BitReader(writer.finish()))
        self.assertTrue(np.array_equal(apply_phi(luma, phi), apply_phi(luma, again)))

    def test_s3_residual_roundtrip(self):
        rng = np.random.default_rng(7)
        resid = rng.integers(-20, 21, size=360).astype(np.int64)
        sig = np.full(360, 8, np.int32)
        frac = np.full(360, 8, np.int32)
        seed = np.zeros(360, np.bool_)
        seed[::10] = True
        blob = encode_s3(resid, sig, frac, 1, seed)
        self.assertIsNotNone(blob)
        back = decode_s3(blob, 360, sig, frac, 1, seed)
        self.assertTrue(np.array_equal(back, resid))

    def test_interleaved_rans_and_alias_table(self):
        rng = np.random.default_rng(8)
        symbols = rng.integers(0, 8, size=200).tolist()
        freqs = normalize_freqs(np.arange(1, 9, dtype=np.int64), nsym=8)
        cum = cumulative(freqs)
        blob = encode_symbols(symbols, freqs, cum)
        self.assertEqual(decode_pairs(blob, len(symbols), cum, freqs), symbols)
        lut = symbol_lut(cum)
        for cf in range(0, 65536, 97):
            sym = int(np.searchsorted(cum, cf, side="right") - 1)
            self.assertEqual(int(lut[cf]), sym)

    def test_weight_file_sha256(self):
        blob = pack_weights()
        digest = hashlib.sha256(blob[4:-32]).digest()
        self.assertEqual(blob[-32:], digest)
        head, dw, pw = load_weights(blob)
        self.assertEqual(head.shape, (16,))
        self.assertEqual(dw.shape, (8, 9))
        self.assertEqual(pw.shape, (8,))

    def test_decode_paths_are_integer(self):
        from vpxc.bitsback import decode_s3 as s3
        from vpxc.coarse import decode_coarse as coarse
        from vpxc.issm_int8 import context_map
        from vpxc.matching import sheaf_kalman_delta
        from vpxc.slgf_v3 import decode_slgf_v3

        for func in (coarse, s3, context_map, sheaf_kalman_delta, decode_slgf_v3, apply_phi):
            _assert_no_true_division(func)


class TestNativeKernel(unittest.TestCase):
    def test_optional_accel_matches_python(self):
        import sysconfig
        import subprocess
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        include = sysconfig.get_path("include")
        target = root / "src" / "vpxc" / f"_accel{suffix}"
        source = root / "csrc" / "vpxc_accel.c"
        built = subprocess.run(
            ["cc", "-O3", "-fPIC", "-shared", f"-I{include}", str(source), "-o", str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
        if built.returncode != 0:
            self.skipTest(built.stderr[-400:])
        sys.path.insert(0, str(root / "src"))
        import importlib

        accel = importlib.import_module("vpxc._accel")
        cum = cumulative(normalize_freqs(np.array([4, 2, 1, 1], dtype=np.int64), nsym=4))
        lut_py = symbol_lut(cum)
        lut_c = np.frombuffer(accel.symbol_lut(np.ascontiguousarray(cum, dtype=np.int32)), dtype=np.uint8)
        self.assertTrue(np.array_equal(lut_py, lut_c))

        symbols = [0, 1, 2, 3, 1, 0, 2] * 20
        freqs = normalize_freqs(np.array([4, 2, 1, 1], dtype=np.int64), nsym=4)
        pairs = [(int(cum[s]), int(freqs[s])) for s in symbols]
        py_blob = encode_symbols(symbols, freqs, cum)
        c_blob = accel.interleaved_encode(pairs)
        self.assertEqual(py_blob, c_blob)

        decay = np.arange(40, dtype=np.int64) * 1000 + 1000
        bias = np.arange(40, dtype=np.int64) - 5
        native = np.frombuffer(
            accel.sequential_scan(
                np.ascontiguousarray(decay).tobytes(),
                np.ascontiguousarray(bias).tobytes(),
            ),
            dtype=np.int64,
        )
        self.assertTrue(np.array_equal(native, sequential(decay, bias)))


if __name__ == "__main__":
    unittest.main()
