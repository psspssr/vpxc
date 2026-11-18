"""Shared codec errors."""


class CodecError(ValueError):
    """Bitstream is truncated, inconsistent, or not a vpxc payload."""
