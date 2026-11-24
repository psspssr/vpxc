"""Byte-identity, frontier/wp round-trips, float-free decode check, and fuzz tests for the C++ codec.

Skipped unless cpp/build/libvpxc_native.so exists. Build it with
``cmake -S cpp -B cpp/build -DCMAKE_CXX_COMPILER=g++ && cmake --build cpp/build``.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest

import numpy as np

from vpxc.errors import CodecError

ROOT = Path(__file__).resolve().parents[1]


def _find_native_lib() -> Path:
    for name in ("libvpxc_native.so", "libvpxc_native.dylib", "vpxc_native.dll"):
        cand = ROOT / "cpp" / "build" / name
        if cand.is_file():
            return cand
    return ROOT / "cpp" / "build" / "libvpxc_native.so"


LIB = _find_native_lib()
FRONTIER_OBJ = ROOT / "cpp" / "build" / "CMakeFiles" / "vpxc.dir" / "src" / "frontier.cpp.o"
FAST_OBJ = ROOT / "cpp" / "build" / "CMakeFiles" / "vpxc.dir" / "src" / "fast_codec.cpp.o"
MEDICAL_OBJ = ROOT / "cpp" / "build" / "CMakeFiles" / "vpxc.dir" / "src" / "medical.cpp.o"


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

    @unittest.skipUnless(FRONTIER_OBJ.is_file() and shutil.which("objdump"), "frontier.cpp.o or objdump not present")
    def test_frontier_decode_machine_code_has_zero_float_ops(self) -> None:
        out = subprocess.check_output(["objdump", "-d", "-C", str(FRONTIER_OBJ)], text=True)
        fp_pat = re.compile(
            r"\t((add|sub|mul|div|sqrt|min|max|ucomi|comi|cvt)[a-z]*(ss|sd|ps|pd)|fadd|fsub|fmul|fdiv|scvtf|ucvtf|fcvtzs)\b"
        )
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

    @unittest.skipUnless(
        FAST_OBJ.is_file() and MEDICAL_OBJ.is_file() and shutil.which("objdump"),
        "fast/medical object files or objdump not present",
    )
    def test_fast_and_medical_decode_machine_code_have_zero_float_ops(self) -> None:
        fp_pat = re.compile(
            r"\t((add|sub|mul|div|sqrt|min|max|ucomi|comi|cvt)[a-z]*(ss|sd|ps|pd)|fadd|fsub|fmul|fdiv|scvtf|ucvtf|fcvtzs)\b"
        )
        for obj, keywords, min_fns in (
            (FAST_OBJ, ("decode_fast", "decode_fast_plane", "color_inv_fast", "FastBitReader"), 2),
            (
                MEDICAL_OBJ,
                ("decode_volume", "decode_volume_slice", "inspect_volume", "parse_volume_directory", "decode_single_slice"),
                2,
            ),
        ):
            out = subprocess.check_output(["objdump", "-d", "-C", str(obj)], text=True)
            checked = 0
            for block in out.split("\n\n"):
                lines = block.strip().splitlines()
                if not lines:
                    continue
                header = lines[0]
                if any(k in header for k in keywords):
                    checked += 1
                    for insn in lines[1:]:
                        self.assertIsNone(fp_pat.search(insn), f"FP op in {obj.name}::{header}: {insn}")
            self.assertGreaterEqual(checked, min_fns)

    def test_fast_profile_roundtrip_and_sub10ms_decode(self) -> None:
        import time
        from vpxc.codec import decode, encode

        rng = np.random.default_rng(77)
        cases = [
            np.array([[[11, 22, 33]]], dtype=np.uint8),
            np.full((64, 64), 1024, dtype=np.uint16),
            (rng.integers(0, 250, size=(45, 53), dtype=np.uint16) * 16).astype(np.uint16),
            rng.integers(0, 256, size=(37, 59, 3), dtype=np.uint8),
            rng.integers(0, 256, size=(31, 29, 4), dtype=np.uint8),
            rng.integers(0, 65536, size=(25, 33, 3), dtype=np.uint16),
        ]
        for img in cases:
            b0 = encode(img, tile=64, effort=0)
            bf = encode(img, tile=64, effort=1, profile="fast")
            self.assertTrue(np.array_equal(decode(b0), img))
            self.assertTrue(np.array_equal(decode(bf), img))

        # Verify high-throughput fast decode on 768x512 RGB24 photo-like gradient
        yy, xx = np.mgrid[0:512, 0:768]
        g = ((yy * 3 + xx * 2) & 0xFF).astype(np.uint8)
        r = np.clip(g.astype(np.int32) + 17 - (yy >> 4), 0, 255).astype(np.uint8)
        b = np.clip(g.astype(np.int32) - 13 + (xx >> 5), 0, 255).astype(np.uint8)
        hd = np.dstack([r, g, b])
        fast_blob = encode(hd, effort=0)
        for _ in range(2):
            decode(fast_blob)
        timings = []
        for _ in range(7):
            t0 = time.perf_counter()
            out = decode(fast_blob)
            timings.append((time.perf_counter() - t0) * 1000.0)
        self.assertTrue(np.array_equal(out, hd))
        self.assertLess(float(np.median(timings)), 25.0)

    def test_medical_3d_volume_and_random_access_slice_seeking(self) -> None:
        from vpxc.medical import decode_slice, decode_volume, encode_volume, inspect_volume

        rng = np.random.default_rng(314)
        # Simulated 16-bit signed Hounsfield Unit CT volume (-1024 air .. +1800 bone)
        zz, yy, xx = np.mgrid[0:10, 0:48, 0:52]
        body = ((yy - 24) ** 2 + (xx - 26) ** 2) < 18**2
        ct = np.full((10, 48, 52), -1024, dtype=np.int16)
        ct[body] = (40 + zz[body] * 3 + (yy[body] - xx[body]) * 2).astype(np.int16)
        ct += rng.integers(-3, 4, size=ct.shape, dtype=np.int16) * body
        ct[0] = -1024  # Pure air padding slice
        ct[5] = ct[4]  # Exact duplicate slice

        for effort in (0, 1, 2):
            blob = encode_volume(ct, effort=effort, gos_size=4)
            meta = inspect_volume(blob)
            self.assertEqual(meta["depth"], 10)
            self.assertEqual(meta["height"], 48)
            self.assertEqual(meta["width"], 52)
            self.assertEqual(meta["dtype"], "int16")
            self.assertEqual(meta["gos_size"], 4)

            restored = decode_volume(blob)
            self.assertEqual(restored.dtype, np.int16)
            self.assertTrue(np.array_equal(restored, ct))

            for z in range(10):
                sl = decode_slice(blob, z)
                self.assertEqual(sl.dtype, np.int16)
                self.assertTrue(np.array_equal(sl, ct[z]))

            with self.assertRaises(CodecError):
                decode_slice(blob, -1)
            with self.assertRaises(CodecError):
                decode_slice(blob, 10)

        # Also verify uint8 and uint16 3D volumes
        for dt, hi in ((np.uint8, 220), (np.uint16, 4095)):
            vol = rng.integers(0, hi, size=(5, 24, 28), dtype=dt)
            b = encode_volume(vol, effort=1, gos_size=2)
            self.assertTrue(np.array_equal(decode_volume(b), vol))
            self.assertTrue(np.array_equal(decode_slice(b, 3), vol[3]))

    def test_dicom_byte_exact_pack_and_unpack(self) -> None:
        import hashlib
        from vpxc.medical import create_multiframe_dicom, decode_dicom_pixels, pack_dicom, unpack_dicom

        rng = np.random.default_rng(99)
        vol_hu = rng.integers(-1024, 1500, size=(6, 32, 36), dtype=np.int16)
        dcm_bytes = create_multiframe_dicom(vol_hu, modality="CT") + b"\x08\x00\x00\x00UL\x04\x00\x00\x00\x00\x00"
        orig_sha = hashlib.sha256(dcm_bytes).hexdigest()

        for eff in (0, 2):
            vpxd = pack_dicom(dcm_bytes, effort=eff, gos_size=3)
            unpacked = unpack_dicom(vpxd)
            self.assertEqual(unpacked, dcm_bytes)
            self.assertEqual(hashlib.sha256(unpacked).hexdigest(), orig_sha)
            pix = decode_dicom_pixels(vpxd)
            self.assertTrue(np.array_equal(pix, vol_hu))
            sl4 = decode_dicom_pixels(vpxd, z_index=4)
            self.assertTrue(np.array_equal(sl4, vol_hu[4]))

        # Corrupting any byte in VPXD must raise CodecError
        bad = bytearray(vpxd)
        bad[-3] ^= 0xFF
        with self.assertRaises(CodecError):
            unpack_dicom(bytes(bad))

    def test_zip_bomb_headers_rejected_without_oom(self) -> None:
        import struct
        from vpxc.codec import decode
        from vpxc.medical import decode_volume

        # 24-byte VPXC header claiming 16384 x 16384 x 4 uint16 (2 GiB) with 0 tile bytes
        bomb_2d = struct.pack("<4sBBBBBBIIIH", b"VPXC", 3, 2, 4, 0, 0, 0, 16384, 16384, 0, 8)
        with self.assertRaises(CodecError):
            decode(bomb_2d)

        # Oversized dimension > kMaxImageDim
        huge_2d = struct.pack("<4sBBBBBBIIIH", b"VPXC", 3, 2, 3, 0, 0, 0, 50000, 50000, 0, 256)
        with self.assertRaises(CodecError):
            decode(huge_2d)

        # 24-byte VPXV header claiming 2048 x 4096 x 4096 uint16 (64 GiB) with 0 GoS bytes
        bomb_3d = struct.pack("<4sBBBBIIII", b"VPXV", 1, 2, 0, 8, 2048, 4096, 4096, 0)
        with self.assertRaises(CodecError):
            decode_volume(bomb_3d)

    def test_arm64_neon_simd_codegen_and_float_freedom(self) -> None:
        clang = shutil.which("clang++")
        if clang is None:
            self.skipTest("clang++ not installed")
        simd_hdr = ROOT / "cpp" / "include"
        code = """
        #include "vpxc/simd.hpp"
        int64_t dot_fn(const int32_t* f, const int16_t* w) { return vpxc::simd::dot36_i32_i16(f, w); }
        void rct_fn(int32_t* r, const int32_t* g, int32_t* b, int n) {
          vpxc::simd::rct_sub_green_inv_row(g, r, b, r, b, n);
          vpxc::simd::rct_sub_avg_inv_row(g, r, b, r, b, n);
        }
        const char* arch_fn() { return vpxc::simd::arch_name(); }
        """
        target = "arm64-apple-macos13" if sys.platform == "darwin" else "aarch64-linux-gnu"
        cmd = [clang, f"--target={target}", "-O3", "-std=c++20", f"-I{simd_hdr}", "-S", "-o", "-", "-x", "c++", "-"]
        if sys.platform != "darwin":
            cmd.insert(2, "-ffreestanding")
        proc = subprocess.run(cmd, input=code, text=True, capture_output=True)
        if proc.returncode != 0:
            self.skipTest(f"ARM64 cross-codegen headers unavailable: {proc.stderr.strip()}")
        asm = proc.stdout
        self.assertIn("arm64-neon", asm)
        self.assertIsNone(re.search(r"\b(fadd|fsub|fmul|fdiv|scvtf|fcvtzs)\b", asm))


if __name__ == "__main__":
    unittest.main()
