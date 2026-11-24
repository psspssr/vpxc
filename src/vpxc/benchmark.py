"""Lossless multi-codec comparison harness for vpxc.

Measures compressed size, ``bpsp`` (bits per sub-pixel / channel sample),
``bpp`` (bits per pixel), compression ratio, encode latency, and decode latency
for ``vpxc`` alongside standard lossless image codecs (``PNG``, ``QOI``,
``JPEG-LS``, ``WebP-L``, ``JPEG 2000``, and ``JPEG XL``).

Every external codec output is decoded and verified bit-for-bit identical to
the input array before its rate is accepted. Backends are discovered
automatically from ``imagecodecs``, ``Pillow``, and ``ffmpeg``.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Callable

import numpy as np

from vpxc.codec import TILE, decode, encode


@dataclass(frozen=True)
class CodecResult:
    codec: str
    nbytes: int
    bpsp: float
    bpp: float
    ratio: float
    encode_ms: float
    decode_ms: float


CodecRunner = Callable[[np.ndarray], tuple[bytes, np.ndarray, float, float] | None]


def _raw_bytes(image: np.ndarray) -> int:
    return int(image.size * image.dtype.itemsize)


def _run_vpxc(
    image: np.ndarray, *, tile: int = TILE, effort: int = 1, profile: str | None = None
) -> tuple[bytes, np.ndarray, float, float]:
    t0 = time.perf_counter()
    blob = encode(image, tile=tile, effort=effort, profile=profile)
    enc_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    back = decode(blob)
    dec_ms = (time.perf_counter() - t0) * 1000.0
    return blob, back, enc_ms, dec_ms


def _discover_external_runners() -> list[tuple[str, CodecRunner]]:
    runners: list[tuple[str, CodecRunner]] = []

    try:
        import imagecodecs as ic  # type: ignore[import-not-found]
    except Exception:
        ic = None

    try:
        from PIL import Image
    except Exception:
        Image = None

    ffmpeg_bin = shutil.which("ffmpeg")

    # 1. PNG (opt)
    if ic is not None and hasattr(ic, "png_encode"):

        def _png_ic(arr: np.ndarray):
            t0 = time.perf_counter()
            blob = ic.png_encode(arr, level=9)
            enc_ms = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            back = ic.png_decode(blob)
            dec_ms = (time.perf_counter() - t0) * 1000.0
            if back.ndim == 3 and back.shape[2] == 1 and arr.ndim == 2:
                back = back[:, :, 0]
            return blob, back, enc_ms, dec_ms

        runners.append(("PNG (opt)", _png_ic))
    elif Image is not None:

        def _png_pil(arr: np.ndarray):
            if arr.dtype != np.uint8 and not (arr.dtype == np.uint16 and arr.ndim == 2):
                return None
            buf = io.BytesIO()
            t0 = time.perf_counter()
            Image.fromarray(arr).save(buf, format="PNG", optimize=True)
            enc_ms = (time.perf_counter() - t0) * 1000.0
            blob = buf.getvalue()
            t0 = time.perf_counter()
            back = np.array(Image.open(io.BytesIO(blob)))
            dec_ms = (time.perf_counter() - t0) * 1000.0
            return blob, back, enc_ms, dec_ms

        runners.append(("PNG (opt)", _png_pil))

    # 2. QOI (8-bit RGB/RGBA)
    if ic is not None and hasattr(ic, "qoi_encode"):

        def _qoi_ic(arr: np.ndarray):
            if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] not in (3, 4):
                return None
            t0 = time.perf_counter()
            blob = ic.qoi_encode(arr)
            enc_ms = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            back = ic.qoi_decode(blob)
            dec_ms = (time.perf_counter() - t0) * 1000.0
            return blob, back, enc_ms, dec_ms

        runners.append(("QOI", _qoi_ic))
    elif ffmpeg_bin is not None:
        runners.append(("QOI", _make_ffmpeg_runner(ffmpeg_bin, "qoi", ".qoi", [])))

    # 3. JPEG-LS
    if ic is not None and hasattr(ic, "jpegls_encode"):

        def _jls_ic(arr: np.ndarray):
            t0 = time.perf_counter()
            blob = ic.jpegls_encode(arr)
            enc_ms = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            back = ic.jpegls_decode(blob)
            dec_ms = (time.perf_counter() - t0) * 1000.0
            if back.ndim == 3 and back.shape[2] == 1 and arr.ndim == 2:
                back = back[:, :, 0]
            return blob, back, enc_ms, dec_ms

        runners.append(("JPEG-LS", _jls_ic))
    elif ffmpeg_bin is not None:
        runners.append(("JPEG-LS", _make_ffmpeg_runner(ffmpeg_bin, "jpegls", ".jls", [])))

    # 4. WebP-L (8-bit RGB/RGBA)
    if ic is not None and hasattr(ic, "webp_encode"):

        def _webp_ic(arr: np.ndarray):
            if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] not in (3, 4):
                return None
            t0 = time.perf_counter()
            blob = ic.webp_encode(arr, lossless=True, level=9)
            enc_ms = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            back = ic.webp_decode(blob)
            dec_ms = (time.perf_counter() - t0) * 1000.0
            return blob, back, enc_ms, dec_ms

        runners.append(("WebP-L (-z 9)", _webp_ic))
    elif ffmpeg_bin is not None:
        runners.append(
            (
                "WebP-L",
                _make_ffmpeg_runner(
                    ffmpeg_bin,
                    "libwebp",
                    ".webp",
                    ["-lossless", "1", "-compression_level", "6", "-q:v", "100"],
                ),
            )
        )

    # 5. JPEG 2000 (reversible 5/3)
    if ic is not None and hasattr(ic, "jpeg2k_encode"):

        def _jp2_ic(arr: np.ndarray):
            t0 = time.perf_counter()
            blob = ic.jpeg2k_encode(arr, level=0)
            enc_ms = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            back = ic.jpeg2k_decode(blob)
            dec_ms = (time.perf_counter() - t0) * 1000.0
            if back.ndim == 3 and back.shape[2] == 1 and arr.ndim == 2:
                back = back[:, :, 0]
            return blob, back, enc_ms, dec_ms

        runners.append(("JPEG 2000 (5/3)", _jp2_ic))

    # 6. JPEG XL (-d 0 -e 3 / -e 7 / -e 9)
    if ic is not None and hasattr(ic, "jpegxl_encode"):
        for jxl_eff in (3, 7, 9):

            def _make_jxl_ic(eff_lvl: int):
                def _jxl_ic(arr: np.ndarray):
                    t0 = time.perf_counter()
                    blob = ic.jpegxl_encode(arr, lossless=True, effort=eff_lvl)
                    enc_ms = (time.perf_counter() - t0) * 1000.0
                    t0 = time.perf_counter()
                    back = ic.jpegxl_decode(blob)
                    dec_ms = (time.perf_counter() - t0) * 1000.0
                    if back.ndim == 3 and back.shape[2] == 1 and arr.ndim == 2:
                        back = back[:, :, 0]
                    return blob, back, enc_ms, dec_ms

                return _jxl_ic

            runners.append((f"JPEG XL (-e {jxl_eff})", _make_jxl_ic(jxl_eff)))
    elif ffmpeg_bin is not None:
        for jxl_eff in (3, 7, 9):
            runners.append(
                (
                    f"JPEG XL (-e {jxl_eff})",
                    _make_ffmpeg_runner(
                        ffmpeg_bin, "libjxl", ".jxl", ["-distance", "0", "-effort", str(jxl_eff)]
                    ),
                )
            )

    return runners


def _make_ffmpeg_runner(
    ffmpeg_bin: str, encoder: str, suffix: str, extra_args: list[str]
) -> CodecRunner:
    def _run(arr: np.ndarray):
        if arr.dtype != np.uint8:
            return None
        if arr.ndim == 2:
            pix_fmt = "gray"
        elif arr.ndim == 3 and arr.shape[2] == 3:
            pix_fmt = "rgb24"
        elif arr.ndim == 3 and arr.shape[2] == 4:
            pix_fmt = "rgba"
        else:
            return None
        h, w = arr.shape[:2]
        with tempfile.TemporaryDirectory() as td:
            raw_in = Path(td) / "in.raw"
            encoded = Path(td) / f"out{suffix}"
            raw_out = Path(td) / "out.raw"
            raw_in.write_bytes(np.ascontiguousarray(arr).tobytes())
            cmd_enc = [
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                pix_fmt,
                "-s",
                f"{w}x{h}",
                "-i",
                str(raw_in),
                "-c:v",
                encoder,
                *extra_args,
                str(encoded),
            ]
            t0 = time.perf_counter()
            r1 = subprocess.run(cmd_enc, capture_output=True)
            enc_ms = (time.perf_counter() - t0) * 1000.0
            if r1.returncode != 0 or not encoded.is_file():
                return None
            blob = encoded.read_bytes()
            cmd_dec = [
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(encoded),
                "-f",
                "rawvideo",
                "-pix_fmt",
                pix_fmt,
                str(raw_out),
            ]
            t0 = time.perf_counter()
            r2 = subprocess.run(cmd_dec, capture_output=True)
            dec_ms = (time.perf_counter() - t0) * 1000.0
            if r2.returncode != 0 or not raw_out.is_file():
                return None
            buf = raw_out.read_bytes()
            if len(buf) != arr.size:
                return None
            back = np.frombuffer(buf, dtype=np.uint8).reshape(arr.shape).copy()
            return blob, back, enc_ms, dec_ms

    return _run


def benchmark_image(
    image: np.ndarray,
    *,
    tile: int = TILE,
    effort: int = 1,
    profile: str | None = None,
    include_external: bool = True,
) -> list[CodecResult]:
    """Benchmark ``vpxc`` and available lossless codecs on a single array."""
    src = np.ascontiguousarray(image)
    samples = int(src.size)
    pixels = int(src.shape[0] * src.shape[1])
    raw_n = _raw_bytes(src)

    results: list[CodecResult] = []
    v_blob, v_back, v_enc, v_dec = _run_vpxc(
        src, tile=tile, effort=effort, profile=profile
    )
    if v_back.shape != src.shape or v_back.dtype != src.dtype or not np.array_equal(v_back, src):
        raise RuntimeError("vpxc lossless verification failed")
    label = f"vpxc ({profile})" if profile else f"vpxc (-e {effort})"
    results.append(
        CodecResult(
            codec=label,
            nbytes=len(v_blob),
            bpsp=8.0 * len(v_blob) / samples,
            bpp=8.0 * len(v_blob) / pixels,
            ratio=raw_n / max(1, len(v_blob)),
            encode_ms=v_enc,
            decode_ms=v_dec,
        )
    )

    if include_external:
        for name, runner in _discover_external_runners():
            try:
                out = runner(src)
            except Exception:
                continue
            if out is None:
                continue
            blob, back, enc_ms, dec_ms = out
            if back.shape != src.shape or back.dtype != src.dtype:
                continue
            if not np.array_equal(back, src):
                continue
            results.append(
                CodecResult(
                    codec=name,
                    nbytes=len(blob),
                    bpsp=8.0 * len(blob) / samples,
                    bpp=8.0 * len(blob) / pixels,
                    ratio=raw_n / max(1, len(blob)),
                    encode_ms=enc_ms,
                    decode_ms=dec_ms,
                )
            )

    results.sort(key=lambda item: item.bpsp)
    return results


def format_benchmark_table(
    results: list[CodecResult], *, markdown: bool = False
) -> str:
    """Format codec comparison results as a plain-text or Markdown table."""
    if markdown:
        lines = [
            "| Codec | Size (bytes) | Rate (`bpsp`) | Rate (`bpp`) | Ratio | Encode (ms) | Decode (ms) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in results:
            bold = "**" if r.codec.startswith("vpxc") else ""
            lines.append(
                f"| {bold}{r.codec}{bold} | {r.nbytes:,} | {bold}{r.bpsp:.4f}{bold} | "
                f"{r.bpp:.4f} | {r.ratio:.2f}x | {r.encode_ms:.1f} | {r.decode_ms:.1f} |"
            )
        return "\n".join(lines)

    header = (
        f"{'Codec':<18} {'Bytes':>11} {'bpsp':>9} {'bpp':>9} "
        f"{'Ratio':>8} {'Enc (ms)':>10} {'Dec (ms)':>10}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for r in results:
        lines.append(
            f"{r.codec:<18} {r.nbytes:>11,d} {r.bpsp:>9.4f} {r.bpp:>9.4f} "
            f"{r.ratio:>7.2f}x {r.encode_ms:>10.1f} {r.decode_ms:>10.1f}"
        )
    return "\n".join(lines)
