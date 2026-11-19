"""ctypes bridge to the C++ codec.

``VPXC_PYTHON=1`` keeps the pure-Python reference. Otherwise encode and decode
use ``libvpxc_native`` when it has been built.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
import os

import numpy as np

from vpxc.errors import CodecError

_LIB: ctypes.CDLL | None = None
_TRIED = False


def _candidate_paths() -> list[Path]:
    env = os.environ.get("VPXC_NATIVE_LIB")
    paths = []
    if env:
        paths.append(Path(env))
    root = Path(__file__).resolve().parents[2]
    paths.extend(
        [
            root / "cpp" / "build" / "libvpxc_native.so",
            root / "cpp" / "build" / "libvpxc_native.dylib",
            Path(__file__).resolve().parent / "libvpxc_native.so",
        ]
    )
    return paths


def library() -> ctypes.CDLL | None:
    global _LIB, _TRIED
    if os.environ.get("VPXC_PYTHON") == "1":
        return None
    if _TRIED:
        return _LIB
    _TRIED = True
    for path in _candidate_paths():
        if not path.is_file():
            continue
        lib = ctypes.CDLL(str(path))
        lib.vpxc_encode.argtypes = [
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.vpxc_encode.restype = ctypes.c_int
        lib.vpxc_decode.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.vpxc_decode.restype = ctypes.c_int
        lib.vpxc_free.argtypes = [ctypes.c_void_p]
        lib.vpxc_free_samples.argtypes = [ctypes.c_void_p]
        lib.vpxc_last_error.restype = ctypes.c_char_p
        _LIB = lib
        return lib
    return None


def _raise_from(lib: ctypes.CDLL, code: int) -> None:
    raw = lib.vpxc_last_error()
    message = raw.decode() if raw else "native codec failed"
    if code == 1:
        raise CodecError(message)
    raise RuntimeError(message)


def _as_samples(image: np.ndarray) -> tuple[np.ndarray, int, int, int, int]:
    src = np.asarray(image)
    if src.dtype not in (np.uint8, np.uint16):
        raise CodecError("image dtype must be uint8 or uint16")
    if src.ndim == 3 and src.shape[2] == 1:
        src = src[:, :, 0]
    if src.ndim == 2:
        height, width = src.shape
        channels = 1
        flat = np.ascontiguousarray(src, dtype=np.uint16)
    elif src.ndim == 3 and src.shape[2] in (3, 4):
        height, width, channels = src.shape
        flat = np.ascontiguousarray(src, dtype=np.uint16).reshape(-1)
    else:
        raise CodecError("image must be HxW or HxWxC")
    return flat, int(height), int(width), int(channels), int(src.dtype == np.uint16)


def encode_native(
    image: np.ndarray,
    *,
    tile: int,
    effort: int,
    profile: str | None,
    version: int,
) -> bytes | None:
    lib = library()
    if lib is None:
        return None
    flat, height, width, channels, u16 = _as_samples(image)
    out = ctypes.POINTER(ctypes.c_uint8)()
    out_n = ctypes.c_size_t()
    prof = None if not profile else profile.encode()
    code = lib.vpxc_encode(
        flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        height,
        width,
        channels,
        u16,
        int(tile),
        int(effort),
        int(version),
        prof,
        ctypes.byref(out),
        ctypes.byref(out_n),
    )
    if code != 0:
        _raise_from(lib, code)
    try:
        return ctypes.string_at(out, out_n.value)
    finally:
        lib.vpxc_free(out)


def decode_native(data: bytes) -> np.ndarray | None:
    lib = library()
    if lib is None:
        return None
    buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    samples = ctypes.POINTER(ctypes.c_uint16)()
    height = ctypes.c_int()
    width = ctypes.c_int()
    channels = ctypes.c_int()
    u16 = ctypes.c_int()
    code = lib.vpxc_decode(
        buf,
        len(data),
        ctypes.byref(samples),
        ctypes.byref(height),
        ctypes.byref(width),
        ctypes.byref(channels),
        ctypes.byref(u16),
    )
    if code != 0:
        _raise_from(lib, code)
    count = int(height.value) * int(width.value) * int(channels.value)
    try:
        array = np.ctypeslib.as_array(samples, shape=(count,)).copy()
    finally:
        lib.vpxc_free_samples(samples)
    dtype = np.uint16 if u16.value else np.uint8
    array = array.astype(dtype, copy=False)
    if channels.value == 1:
        return array.reshape(int(height.value), int(width.value))
    return array.reshape(int(height.value), int(width.value), int(channels.value))
