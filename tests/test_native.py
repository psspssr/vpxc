"""Byte-identity of the C++ codec against the Python reference.

Skipped unless cpp/build/libvpxc_native.so exists. Build it with
``cmake -S cpp -B cpp/build -DCMAKE_CXX_COMPILER=g++ && cmake --build cpp/build``.
"""

from __future__ import annotations

import os
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "cpp" / "build" / "libvpxc_native.so"


def _python_codec():
    os.environ["VPXC_PYTHON"] = "1"
    import vpxc.native as native
    from vpxc.codec import decode, encode

    native._TRIED = False
    native._LIB = None
    return encode, decode, native


@unittest.skipUnless(LIB.is_file(), "build cpp/build/libvpxc_native.so")
class TestNativeByteIdentity(unittest.TestCase):
    def test_random_profiles_match_python(self):
        os.environ["VPXC_NATIVE_LIB"] = str(LIB)
        py_encode, py_decode, native = _python_codec()
        os.environ.pop("VPXC_PYTHON", None)
        native._TRIED = False
        native._LIB = None
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
                    if version == 2 and effort > 2:
                        continue
                    for profile in (None, "slgf", "kalman", "qx-rans"):
                        blob = py_encode(image, tile=32, effort=effort, profile=profile, version=version)
                        native_blob = native.encode_native(
                            image, tile=32, effort=effort, profile=profile, version=version
                        )
                        # Version 3 effort 1–2 no longer runs the searches that measured as
                        # unused on Kodak, so those streams are not byte-identical to Python.
                        if version == 2 or profile in ("kalman", "qx-rans"):
                            self.assertEqual(native_blob, blob)
                        self.assertTrue(np.array_equal(native.decode_native(native_blob), image))
                        self.assertTrue(np.array_equal(native.decode_native(blob), image))
                        self.assertTrue(np.array_equal(py_decode(native_blob), image))


if __name__ == "__main__":
    unittest.main()
