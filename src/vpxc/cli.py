"""Command-line interface for the vpxc lossless image codec."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

from vpxc.benchmark import CodecResult, benchmark_image, format_benchmark_table
from vpxc.codec import PROFILES, TILE, decode, encode


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vpxc",
        description="vpxc — exact lossless SLGF image compressor and benchmark tool.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    enc = sub.add_parser("encode", help="compress an image to a .vpxc file")
    enc.add_argument("input", help="input image path (PNG, TIFF, BMP, JPEG, etc.)")
    enc.add_argument("output", help="output .vpxc bitstream path")
    enc.add_argument("--tile", type=int, default=TILE, help="tile size in pixels (default: 768)")
    enc.add_argument("--effort", type=int, default=2, choices=(1, 2), help="search effort (1=fast, 2=best)")
    enc.add_argument("--profile", default=None, choices=PROFILES, help="force a specific coding profile")

    dec = sub.add_parser("decode", help="restore an exact image from a .vpxc file")
    dec.add_argument("input", help="input .vpxc bitstream path")
    dec.add_argument("output", help="output image path (e.g. restored.png)")

    rnd = sub.add_parser("roundtrip", help="encode, decode, verify bit-identity, and print rate")
    rnd.add_argument("input", help="input image path")
    rnd.add_argument("--tile", type=int, default=TILE, help="tile size in pixels (default: 768)")
    rnd.add_argument("--effort", type=int, default=2, choices=(1, 2))
    rnd.add_argument("--profile", default=None, choices=PROFILES)

    bench = sub.add_parser(
        "benchmark",
        help="compare vpxc lossless compression against PNG, QOI, JPEG-LS, WebP-L, JPEG 2000, and JPEG XL",
    )
    bench.add_argument(
        "inputs",
        nargs="+",
        help="one or more image files or directories of images",
    )
    bench.add_argument("--tile", type=int, default=TILE, help="tile size in pixels (default: 768)")
    bench.add_argument("--effort", type=int, default=2, choices=(1, 2))
    bench.add_argument("--profile", default=None, choices=PROFILES)
    bench.add_argument("--markdown", action="store_true", help="print summary table as Markdown")

    args = parser.parse_args(argv)
    if args.cmd == "encode":
        image = _load(args.input)
        started = time.perf_counter()
        blob = encode(image, tile=args.tile, effort=args.effort, profile=args.profile)
        elapsed = time.perf_counter() - started
        with open(args.output, "wb") as handle:
            handle.write(blob)
        print(
            f"wrote {args.output}: {len(blob):,} bytes ({_rate(image, len(blob))}) in {elapsed:.2f}s"
        )
    elif args.cmd == "decode":
        with open(args.input, "rb") as handle:
            blob = handle.read()
        started = time.perf_counter()
        image = decode(blob)
        elapsed = time.perf_counter() - started
        _save(args.output, image)
        print(
            f"wrote {args.output}: shape={image.shape} dtype={image.dtype} in {elapsed:.2f}s"
        )
    elif args.cmd == "roundtrip":
        image = _load(args.input)
        started = time.perf_counter()
        blob = encode(image, tile=args.tile, effort=args.effort, profile=args.profile)
        encode_s = time.perf_counter() - started
        started = time.perf_counter()
        back = decode(blob)
        decode_s = time.perf_counter() - started
        if (
            back.shape != image.shape
            or back.dtype != image.dtype
            or not np.array_equal(back, image)
        ):
            raise SystemExit("round trip mismatch")
        png = _png_size(image)
        rate = _rate(image, len(blob))
        png_info = f"  png {png:,} bytes ({_rate(image, png)})" if png else ""
        print(
            f"{args.input}: {image.shape} {image.dtype}  "
            f"vpxc {len(blob):,} bytes ({rate}){png_info}  "
            f"encode {encode_s:.2f}s  decode {decode_s:.2f}s"
        )
    else:
        paths = _collect_images(args.inputs)
        if not paths:
            raise SystemExit("no supported image files found")
        by_codec: dict[str, list[CodecResult]] = {}
        total_raw = 0
        total_samples = 0
        total_pixels = 0
        for p in paths:
            arr = _load(str(p))
            total_raw += int(arr.size * arr.dtype.itemsize)
            total_samples += int(arr.size)
            total_pixels += int(arr.shape[0] * arr.shape[1])
            res = benchmark_image(
                arr, tile=args.tile, effort=args.effort, profile=args.profile
            )
            for item in res:
                by_codec.setdefault(item.codec, []).append(item)

        n_imgs = len(paths)
        summary: list[CodecResult] = []
        for codec_name, items in by_codec.items():
            if len(items) != n_imgs:
                continue
            sum_bytes = sum(x.nbytes for x in items)
            summary.append(
                CodecResult(
                    codec=codec_name,
                    nbytes=sum_bytes,
                    bpsp=8.0 * sum_bytes / total_samples,
                    bpp=8.0 * sum_bytes / total_pixels,
                    ratio=total_raw / max(1, sum_bytes),
                    encode_ms=float(np.mean([x.encode_ms for x in items])),
                    decode_ms=float(np.mean([x.decode_ms for x in items])),
                )
            )
        summary.sort(key=lambda x: x.bpsp)
        if not args.markdown:
            print(f"Benchmarked {n_imgs} image(s) (bit-exact lossless verified):")
        print(format_benchmark_table(summary, markdown=args.markdown))


def _collect_images(inputs: list[str]) -> list[Path]:
    exts = {".png", ".tif", ".tiff", ".bmp", ".ppm", ".pgm", ".webp", ".jpg", ".jpeg"}
    found: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            for child in sorted(p.iterdir()):
                if child.is_file() and child.suffix.lower() in exts:
                    found.append(child)
        elif p.is_file():
            found.append(p)
    return found


def _rate(image: np.ndarray, nbytes: int) -> str:
    samples = int(image.size)
    pixels = int(np.prod(image.shape[:2]))
    bpsp = 8 * nbytes / samples
    bpp = 8 * nbytes / pixels
    return f"{bpsp:.4f} bpsp, {bpp:.4f} bpp"


def _load(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            arr = np.array(im)
            if arr.min() >= 0 and arr.max() <= 65535:
                return arr.astype(np.uint16)
            raise SystemExit(f"integer image {path} exceeds uint16 range")
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
        try:
            import imagecodecs as ic  # type: ignore[import-not-found]

            if path.lower().endswith(".png") and hasattr(ic, "png_encode"):
                Path(path).write_bytes(ic.png_encode(image, level=9))
                return
            if path.lower().endswith((".tif", ".tiff")) and hasattr(ic, "tiff_encode"):
                Path(path).write_bytes(ic.tiff_encode(image))
                return
        except Exception:
            pass
        raise SystemExit(
            "saving 16-bit multi-channel images requires imagecodecs (.png/.tiff) or NumPy .npy"
        )
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
