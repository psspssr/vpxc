"""vpxc — Lossless image compression via Symplectic Lattice-Gauge Flows (SLGF).

Provides exact, deterministic integer compression and decompression on 8-bit
and 16-bit grayscale, RGB, and RGBA arrays, plus built-in lossless codec
benchmarking.
"""

from vpxc.benchmark import CodecResult, benchmark_image, format_benchmark_table
from vpxc.codec import CodecError, decode, encode

__all__ = [
    "CodecError",
    "CodecResult",
    "benchmark_image",
    "decode",
    "encode",
    "format_benchmark_table",
]
__version__ = "0.3.0"
