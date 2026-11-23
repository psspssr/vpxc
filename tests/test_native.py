"""Byte-identity, frontier/wp round-trips, float-free decode check, and fuzz tests for the C++ codec.

Skipped unless cpp/build/libvpxc_native.so exists. Build it with
``cmake -S cpp -B cpp/build -DCMAKE_CXX_COMPILER=g++ && cmake --build cpp/build``.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import unittest

import numpy as np

from vpxc.errors import CodecError

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "cpp" / "build" / "libvpxc_native.so"
FRONTIER_OBJ = ROOT / "cpp" / "build" / "CMakeFiles" / "vpxc.dir" / "src" / "frontier.cpp.o"


def _py_call(fn, *args, **kwargs):
    prev = os.environ.get("VPXC_PYTHON")
    os.environ["VPXC_PYTHON"] = "1"
    try:
        return fn(*args, **kwargs)
    finally:
        if prev is None:
            os.environ.pop("VPXC_PYTHON", None)
        else:
            os.environ["VPXC_PYTHON"] = prev


@unittest.skipUnless(LIB.is_file(), "build cpp/build/libvpxc_native.so")
class TestNativeCodec(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["VPXC_NATIVE_LIB"] = str(LIB)
        os.environ.pop("VPXC_PYTHON", None)
        import vpxc.native as native

        native._TRIED = False
        native._LIB = None
        self.native = native

    def test_random_profiles_match_python(self) -> None:
        from vpxc.codec import decode, encode

        rng = np.random.default_rng(19)
        images = [
            rng.integers(0, 256, size=(32, 24, 3), dtype=np.uint8),
            rng.integers(0, 5000, size=(20, 18), dtype=np.uint16),
            np.dstack(
                [
                    np.linspace(0, 255, 80, dtype=np.uint8)[None, :].repeat(64, 0),
                    np.full((64, 80), 40, np.uint8),
                    np.full((64, 80), 200, np.uint8),
                ]
            ),
        ]
        for image in images:
            for version in (2, 3):
                for effort in (1, 2):
                    for profile in ("slgf", "kalman", "qx-rans"):
                        py_blob = _py_call(
                            encode, image, tile=32, effort=effort, profile=profile, version=version
                        )
                        native_blob = self.native.encode_native(
                            image, tile=32, effort=effort, profile=profile, version=version
                        )
                        assert native_blob is not None
                        if version == 2 or profile in ("kalman", "qx-rans"):
                            self.assertEqual(native_blob, py_blob)
                        self.assertTrue(np.array_equal(self.native.decode_native(native_blob), image))
                        self.assertTrue(np.array_equal(self.native.decode_native(py_blob), image))
                        self.assertTrue(np.array_equal(_py_call(decode, native_blob), image))

    def test_frontier_and_wp_roundtrip_all_formats_and_efforts(self) -> None:
        from vpxc.codec import decode, encode

        rng = np.random.default_rng(42)
        cases = [
            np.array([[[17, 140, 223]]], dtype=np.uint8),
            rng.integers(0, 256, size=(1, 17, 3), dtype=np.uint8),
            rng.integers(0, 256, size=(17, 1, 3), dtype=np.uint8),
            rng.integers(0, 256, size=(3, 5, 3), dtype=np.uint8),
            rng.integers(0, 256, size=(33, 47), dtype=np.uint8),
            rng.integers(0, 256, size=(35, 41, 4), dtype=np.uint8),
            rng.integers(0, 65536, size=(29, 31), dtype=np.uint16),
            rng.integers(0, 65536, size=(27, 25, 3), dtype=np.uint16),
            rng.integers(0, 65536, size=(19, 23, 4), dtype=np.uint16),
        ]
        # Correlated photographic-like gradient + texture test image
        yy, xx = np.mgrid[0:96, 0:112]
        g = np.clip(40 + yy * 1.5 + xx * 0.9 + 12 * np.sin(xx / 5.0), 0, 255).astype(np.uint8)
        r = np.clip(g.astype(np.int32) + 18 - (yy // 4), 0, 255).astype(np.uint8)
        b = np.clip(g.astype(np.int32) - 14 + (xx // 5), 0, 255).astype(np.uint8)
        photo = np.dstack([r, g, b])
        cases.append(photo)

        for img in cases:
            for prof in ("frontier", "wp"):
                blob = encode(img, tile=32, effort=2, profile=prof, version=3)
                back = decode(blob)
                self.assertEqual(back.shape, img.shape)
                self.assertEqual(back.dtype, img.dtype)
                self.assertTrue(np.array_equal(back, img))

        b1 = encode(photo, tile=128, effort=1, profile="frontier", version=3)
        b2 = encode(photo, tile=128, effort=2, profile="frontier", version=3)
        b3 = encode(photo, tile=128, effort=3, profile="frontier", version=3)
        for b in (b1, b2, b3):
            self.assertTrue(np.array_equal(decode(b), photo))
        self.assertLessEqual(len(b3), len(b2))

    def test_adversarial_corrupt_frontier_streams_rejected(self) -> None:
        from vpxc.codec import decode, encode

        rng = np.random.default_rng(108)
        yy, xx = np.mgrid[0:48, 0:48]
        img = np.dstack(
            [
                ((yy * 3 + xx) & 0xFF).astype(np.uint8),
                ((yy * 2 + xx * 2) & 0xFF).astype(np.uint8),
                ((255 - yy * 2) & 0xFF).astype(np.uint8),
            ]
        )
        valid_blob = encode(img, tile=64, effort=2, profile="frontier", version=3)
        self.assertTrue(np.array_equal(decode(valid_blob), img))

        # Truncation at various points must raise CodecError cleanly
        for cut in (0, 5, 23, 24, 30, 34, len(valid_blob) // 2, len(valid_blob) - 1):
            with self.assertRaises(CodecError):
                decode(valid_blob[:cut])

        # Trailing garbage bytes must be rejected
        with self.assertRaises(CodecError):
            decode(valid_blob + b"\x00")

        # Random payload mutations must either fail parsing or fail CRC-32 with CodecError
        for _ in range(40):
            mut = bytearray(valid_blob)
            idx = int(rng.integers(24, len(mut)))
            mut[idx] ^= int(rng.integers(1, 256))
            with self.assertRaises(CodecError):
                decode(bytes(mut))

    @unittest.skipUnless(FRONTIER_OBJ.is_file(), "frontier.cpp.o not present")
    def test_frontier_decode_machine_code_has_zero_float_ops(self) -> None:
        out = subprocess.check_output(["objdump", "-d", "-C", str(FRONTIER_OBJ)], text=True)
        fp_pat = re.compile(r"\b(add|sub|mul|div|sqrt|min|max|ucomi|comi|cvt)[a-z]*(ss|sd|ps|pd)\b")
        decode_keywords = (
            "decode_",
            "MATree::read",
            "MATree::walk",
            "read_sector_ridge",
            "AdaptiveStreamDecoder",
            "AdaptiveCore66",
            "open_ma_stream",
            "RasterState",
            "PatchPredictor",
            "interp_features_exact",
            "color_inv_frontier",
            "pixel_bounds_q4",
        )
        checked = 0
        for block in out.split("\n\n"):
            lines = block.strip().splitlines()
            if not lines:
                continue
            header = lines[0]
            if any(k in header for k in decode_keywords):
                checked += 1
                for insn in lines[1:]:
                    self.assertIsNone(fp_pat.search(insn), f"FP op in {header}: {insn}")
        self.assertGreaterEqual(checked, 10)


if __name__ == "__main__":
    unittest.main()
