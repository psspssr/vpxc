"""vpxc — Lossless image compression via Symplectic Lattice-Gauge Flows (SLGF).

Provides exact, deterministic integer compression and decompression on 8-bit
and 16-bit grayscale, RGB, and RGBA arrays, plus built-in lossless codec
benchmarking.
"""

from vpxc.benchmark import CodecResult, benchmark_image, benchmark_volume, format_benchmark_table
from vpxc.codec import CodecError, decode, encode
from vpxc.medical import (
    create_multiframe_dicom,
    decode_dicom_pixels,
    decode_slice,
    decode_volume,
    encode_volume,
    inspect_volume,
    pack_dicom,
    unpack_dicom,
)

__all__ = [
    "CodecError",
    "CodecResult",
    "benchmark_image",
    "benchmark_volume",
    "create_multiframe_dicom",
    "decode",
    "decode_dicom_pixels",
    "decode_slice",
    "decode_volume",
    "encode",
    "encode_volume",
    "format_benchmark_table",
    "inspect_volume",
    "pack_dicom",
    "unpack_dicom",
]
__version__ = "0.4.0"
