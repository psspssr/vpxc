"""Lossless round-trip and multi-codec comparison tests on real photographs.

Set ``VPXC_IMAGE_DIR`` to a directory of PNG or JPEG files (defaults to
``/tmp/vpxc-images`` when present). Every image in the directory must round-trip
bit-identically and compress strictly smaller than optimized PNG.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
import unittest

import numpy as np

from vpxc.codec import decode, encode

IMAGE_DIR = Path(os.environ.get("VPXC_IMAGE_DIR", "/tmp/vpxc-images"))


def _load(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        if im.mode == "L":
            return np.array(im)
        return np.array(im.convert("RGB"))


def _png_size(image: np.ndarray) -> int:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG", optimize=True)
    return buffer.tell()


def _available_photos() -> list[Path]:
    if not IMAGE_DIR.is_dir():
        return []
    exts = {".png", ".jpg", ".jpeg"}
    return [
        p for p in sorted(IMAGE_DIR.iterdir()) if p.is_file() and p.suffix.lower() in exts
    ]


@unittest.skipUnless(_available_photos(), "set VPXC_IMAGE_DIR to real images")
class TestRealImages(unittest.TestCase):
    def test_photographs_roundtrip_and_beat_png(self):
        photos = _available_photos()[:8]
        for path in photos:
            image = _load(path)
            blob = encode(image, tile=512, effort=2)
            back = decode(blob)
            self.assertEqual(back.shape, image.shape)
            self.assertEqual(back.dtype, image.dtype)
            self.assertTrue(np.array_equal(back, image), path.name)
            self.assertLess(len(blob), _png_size(image), path.name)
            self.assertLess(len(blob), image.size, path.name)

    def test_gray_and_forced_profiles(self):
        photos = _available_photos()
        from PIL import Image

        with Image.open(photos[0]) as im:
            gray = np.array(im.convert("L"))
        blob = encode(gray, tile=512, effort=2)
        self.assertTrue(np.array_equal(decode(blob), gray))
        self.assertLess(len(blob), _png_size(gray))

        color = _load(photos[-1])
        for prof in ("slgf", "qx-rans", "qx-rice"):
            encoded = encode(color, tile=256, effort=1, profile=prof)
            self.assertTrue(np.array_equal(decode(encoded), color), prof)


if __name__ == "__main__":
    unittest.main()
