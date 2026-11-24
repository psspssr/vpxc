"""vpxc — Deterministic lossless image, 3D medical volume, and DICOM codec.

Provides exact, cross-platform IEEE-754-free integer compression and
decompression for 8-bit/16-bit 2D images (.vpxc), 8/16-bit signed/unsigned 3D
CT & MRI stacks with O(1) Group-of-Slices random access (.vpxv), and SHA-256
byte-exact DICOM Part 10 archives (.vpxd).
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
