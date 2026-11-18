"""Command line for the SLGF-Zero codec."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from vpxc.codec import decode, encode


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vpxc",
        description="Lossless SLGF-Zero image codec (exact round trip).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    enc = sub.add_parser("encode", help="compress an image to a .vpxc bitstream")
    enc.add_argument("input")
    enc.add_argument("output")
    enc.add_argument("--tile", type=int, default=256)
    enc.add_argument("--effort", type=int, default=2, choices=(1, 2))
    enc.add_argument(
        "--profile",
        default=None,
        choices=("raw", "raster", "kalman", "qx-rice", "qx-rans", "palette"),
    )

    dec = sub.add_parser("decode", help="restore an image from a .vpxc bitstream")
    dec.add_argument("input")
    dec.add_argument("output")

    rnd = sub.add_parser("roundtrip", help="encode, decode, and report rate")
    rnd.add_argument("input")
    rnd.add_argument("--tile", type=int, default=256)
    rnd.add_argument("--effort", type=int, default=2, choices=(1, 2))
    rnd.add_argument(
        "--profile",
        default=None,
        choices=("raw", "raster", "kalman", "qx-rice", "qx-rans", "palette"),
    )

    args = parser.parse_args(argv)
    if args.cmd == "encode":
        image = _load(args.input)
        started = time.perf_counter()
        blob = encode(image, tile=args.tile, effort=args.effort, profile=args.profile)
        elapsed = time.perf_counter() - started
        with open(args.output, "wb") as handle:
            handle.write(blob)
        print(
            f"wrote {args.output}: {len(blob)} bytes, {_rate(image, len(blob))}, {elapsed:.2f}s"
        )
    elif args.cmd == "decode":
        with open(args.input, "rb") as handle:
            blob = handle.read()
        image = decode(blob)
        _save(args.output, image)
        print(f"wrote {args.output}: {image.shape} {image.dtype}")
    else:
        image = _load(args.input)
        started = time.perf_counter()
        blob = encode(image, tile=args.tile, effort=args.effort, profile=args.profile)
        encode_s = time.perf_counter() - started
        started = time.perf_counter()
        back = decode(blob)
        decode_s = time.perf_counter() - started
        if back.shape != image.shape or back.dtype != image.dtype or not np.array_equal(back, image):
            raise SystemExit("round trip mismatch")
        png = _png_size(image)
        rate = _rate(image, len(blob))
        print(
            f"{args.input}: {image.shape} {image.dtype}  "
            f"vpxc {len(blob)} bytes ({rate})  "
            f"png {png} bytes ({_rate(image, png)})  "
            f"encode {encode_s:.2f}s  decode {decode_s:.2f}s"
        )


def _rate(image: np.ndarray, nbytes: int) -> str:
    samples = int(image.size)
    pixels = int(np.prod(image.shape[:2]))
    bpsp = 8 * nbytes / samples
    bpp = 8 * nbytes / pixels
    return f"{bpsp:.4f} bpsp, {bpp:.4f} bpp"


def _load(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        if im.mode in ("I;16", "I;16B", "I;16L"):
            arr = np.array(im)
            if arr.dtype != np.uint16:
                arr = arr.astype(np.uint16)
            return arr
        if im.mode == "L":
            return np.array(im)
        if im.mode == "RGBA":
            return np.array(im)
        if im.mode == "LA":
            return np.array(im.convert("RGBA"))
        return np.array(im.convert("RGB"))


def _save(path: str, image: np.ndarray) -> None:
    from PIL import Image

    if image.dtype == np.uint8:
        Image.fromarray(image).save(path)
        return
    if image.dtype == np.uint16 and image.ndim == 2:
        Image.fromarray(image).save(path)
        return
    if image.dtype == np.uint16:
        # Pillow's 16-bit RGB writer is unreliable; scale a preview only when
        # asked to save a color uint16 image, and refuse rather than truncate.
        raise SystemExit("refusing to write a color uint16 image as an 8-bit preview")
    raise SystemExit(f"cannot save dtype {image.dtype}")


def _png_size(image: np.ndarray) -> int:
    import io

    from PIL import Image

    if image.dtype != np.uint8:
        return 0
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG", optimize=True)
    return buffer.tell()


if __name__ == "__main__":
    sys.exit(main())
