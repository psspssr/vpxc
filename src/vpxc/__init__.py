"""SLGF-Zero lossless image codec.

Byte-exact implementation of the integer pipeline in
``docs/frontier-lossless-image-compression.md``: reversible color lifting,
one predict/update quincunx stage per scale, causal Sheaf-Kalman blending,
and dyadic quotient entropy coding. Research rate targets in the monograph
are not claimed as measured results.
"""

from vpxc.codec import CodecError, decode, encode

__all__ = ["CodecError", "decode", "encode"]
__version__ = "0.1.0"
