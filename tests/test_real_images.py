"""Lossless round trips on real photographs.

Set ``VPXC_IMAGE_DIR`` to a directory of PNG or JPEG files. The default
``/tmp/vpxc-images`` is used when that directory exists. Kodak PNGs are
lossless photographs. JPEG files are real photographs after a lossy decode;
the codec still has to restore those decoded samples exactly, and it is
compared with a PNG of the same samples.
"""

from __future__ import annotations

import io
import os
import unittest
from pathlib import Path

import numpy as np

from vpxc.codec import decode, encode

IMAGE_DIR = Path(os.environ.get("VPXC_IMAGE_DIR", "/tmp/vpxc-images"))
PHOTOS = (
    "kodim01.png",
    "kodim05.png",
    "kodim07.png",
    "kodim13.png",
    "kodim15.png",
    "kodim23.png",
    "monarch.jpg",
    "matterhorn.jpg",
)


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


@unittest.skipUnless(IMAGE_DIR.is_dir(), "set VPXC_IMAGE_DIR to real images")
class TestRealImages(unittest.TestCase):
    def test_photographs_roundtrip_and_beat_png(self):
        missing = [name for name in PHOTOS if not (IMAGE_DIR / name).is_file()]
        if missing:
            self.skipTest("missing " + ", ".join(missing))
        for name in PHOTOS:
            image = _load(IMAGE_DIR / name)
            blob = encode(image, tile=256, effort=2)
            back = decode(blob)
            self.assertEqual(back.shape, image.shape)
            self.assertEqual(back.dtype, image.dtype)
            self.assertTrue(np.array_equal(back, image), name)
            self.assertLess(len(blob), _png_size(image), name)
            self.assertLess(len(blob), image.size, name)

    def test_gray_kodak_and_forced_rans(self):
        color_path = IMAGE_DIR / "kodim07.png"
        rans_path = IMAGE_DIR / "kodim23.png"
        if not color_path.is_file() or not rans_path.is_file():
            self.skipTest("kodak files are not available")
        from PIL import Image

        with Image.open(color_path) as im:
            gray = np.array(im.convert("L"))
        blob = encode(gray, tile=256, effort=2)
        self.assertTrue(np.array_equal(decode(blob), gray))
        self.assertLess(len(blob), _png_size(gray))

        color = _load(rans_path)
        rans = encode(color, tile=256, effort=1, profile="qx-rans")
        self.assertTrue(np.array_equal(decode(rans), color))
        rice = encode(color, tile=256, effort=1, profile="qx-rice")
        self.assertTrue(np.array_equal(decode(rice), color))


if __name__ == "__main__":
    unittest.main()
