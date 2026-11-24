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
    pkg = Path(__file__).resolve().parent
    paths.extend(
        [
            root / "cpp" / "build" / "libvpxc_native.so",
            root / "cpp" / "build" / "libvpxc_native.dylib",
            root / "cpp" / "build" / "vpxc_native.dll",
            root / "cpp" / "build" / "Release" / "vpxc_native.dll",
            pkg / "libvpxc_native.so",
            pkg / "libvpxc_native.dylib",
            pkg / "vpxc_native.dll",
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
            ctypes.c_void_p,
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
        lib.vpxc_arch_name.argtypes = []
        lib.vpxc_arch_name.restype = ctypes.c_char_p
        lib.vpxc_encode_volume.argtypes = [
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.vpxc_encode_volume.restype = ctypes.c_int
        lib.vpxc_decode_volume.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.vpxc_decode_volume.restype = ctypes.c_int
        lib.vpxc_decode_volume_slice.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.vpxc_decode_volume_slice.restype = ctypes.c_int
        lib.vpxc_inspect_volume.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.vpxc_inspect_volume.restype = ctypes.c_int
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


def _as_const_ptr(data: bytes | bytearray | memoryview) -> tuple[object, ctypes.c_void_p, int]:
    if isinstance(data, bytes):
        n = len(data)
        # Keep `data` alive alongside zero-copy PyBytes raw pointer
        ptr = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer_copy(b"\x00")) if n == 0 else ctypes.cast(data, ctypes.c_void_p).value)
        return data, ptr, n
    raw = bytes(data)
    n = len(raw)
    ptr = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer_copy(b"\x00")) if n == 0 else ctypes.cast(raw, ctypes.c_void_p).value)
    return raw, ptr, n


def decode_native(data: bytes) -> np.ndarray | None:
    lib = library()
    if lib is None:
        return None
    _keep, ptr, n = _as_const_ptr(data)
    samples = ctypes.POINTER(ctypes.c_uint16)()
    height = ctypes.c_int()
    width = ctypes.c_int()
    channels = ctypes.c_int()
    u16 = ctypes.c_int()
    code = lib.vpxc_decode(
        ptr,
        n,
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


def arch_name_native() -> str | None:
    lib = library()
    if lib is None:
        return None
    raw = lib.vpxc_arch_name()
    return raw.decode() if raw else None


def _as_volume_samples(volume: np.ndarray) -> tuple[np.ndarray, int, int, int, int]:
    src = np.asarray(volume)
    if src.dtype == np.uint8:
        dtype_code = 1
        flat = np.ascontiguousarray(src, dtype=np.uint16).reshape(-1)
    elif src.dtype == np.uint16:
        dtype_code = 2
        flat = np.ascontiguousarray(src, dtype=np.uint16).reshape(-1)
    elif src.dtype == np.int16:
        dtype_code = 3
        flat = np.ascontiguousarray(src).view(np.uint16).reshape(-1)
    else:
        raise CodecError("medical volume dtype must be uint8, uint16, or int16")

    if src.ndim == 2:
        depth = 1
        height, width = src.shape
    elif src.ndim == 3:
        depth, height, width = src.shape
    else:
        raise CodecError("medical volume must be ZxHxW (or HxW single slice)")
    return flat, int(depth), int(height), int(width), dtype_code


def _volume_from_raw(raw_u16: np.ndarray, shape: tuple[int, ...], dtype_code: int) -> np.ndarray:
    if dtype_code == 1:
        return raw_u16.astype(np.uint8, copy=False).reshape(shape)
    if dtype_code == 2:
        return raw_u16.reshape(shape)
    if dtype_code == 3:
        return raw_u16.view(np.int16).reshape(shape)
    raise CodecError(f"unknown medical volume dtype code {dtype_code}")


def encode_volume_native(volume: np.ndarray, *, effort: int = 0, gos_size: int = 8) -> bytes:
    lib = library()
    if lib is None:
        raise CodecError("native C++ library libvpxc_native.so is required for 3D medical volumes")
    flat, depth, height, width, dtype_code = _as_volume_samples(volume)
    out = ctypes.POINTER(ctypes.c_uint8)()
    out_n = ctypes.c_size_t()
    code = lib.vpxc_encode_volume(
        flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        depth,
        height,
        width,
        dtype_code,
        int(effort),
        int(gos_size),
        ctypes.byref(out),
        ctypes.byref(out_n),
    )
    if code != 0:
        _raise_from(lib, code)
    try:
        return ctypes.string_at(out, out_n.value)
    finally:
        lib.vpxc_free(out)


def decode_volume_native(data: bytes) -> np.ndarray:
    lib = library()
    if lib is None:
        raise CodecError("native C++ library libvpxc_native.so is required for 3D medical volumes")
    _keep, ptr, n = _as_const_ptr(data)
    samples = ctypes.POINTER(ctypes.c_uint16)()
    depth = ctypes.c_int()
    height = ctypes.c_int()
    width = ctypes.c_int()
    dtype_code = ctypes.c_int()
    code = lib.vpxc_decode_volume(
        ptr,
        n,
        ctypes.byref(samples),
        ctypes.byref(depth),
        ctypes.byref(height),
        ctypes.byref(width),
        ctypes.byref(dtype_code),
    )
    if code != 0:
        _raise_from(lib, code)
    total = int(depth.value) * int(height.value) * int(width.value)
    try:
        raw = np.ctypeslib.as_array(samples, shape=(total,)).copy()
    finally:
        lib.vpxc_free_samples(samples)
    return _volume_from_raw(raw, (int(depth.value), int(height.value), int(width.value)), int(dtype_code.value))


def decode_volume_slice_native(data: bytes, z_index: int) -> np.ndarray:
    lib = library()
    if lib is None:
        raise CodecError("native C++ library libvpxc_native.so is required for 3D medical volumes")
    _keep, ptr, n = _as_const_ptr(data)
    samples = ctypes.POINTER(ctypes.c_uint16)()
    height = ctypes.c_int()
    width = ctypes.c_int()
    dtype_code = ctypes.c_int()
    code = lib.vpxc_decode_volume_slice(
        ptr,
        n,
        int(z_index),
        ctypes.byref(samples),
        ctypes.byref(height),
        ctypes.byref(width),
        ctypes.byref(dtype_code),
    )
    if code != 0:
        _raise_from(lib, code)
    total = int(height.value) * int(width.value)
    try:
        raw = np.ctypeslib.as_array(samples, shape=(total,)).copy()
    finally:
        lib.vpxc_free_samples(samples)
    return _volume_from_raw(raw, (int(height.value), int(width.value)), int(dtype_code.value))


def inspect_volume_native(data: bytes) -> dict[str, int | str]:
    lib = library()
    if lib is None:
        raise CodecError("native C++ library libvpxc_native.so is required for 3D medical volumes")
    _keep, ptr, n = _as_const_ptr(data)
    depth = ctypes.c_int()
    height = ctypes.c_int()
    width = ctypes.c_int()
    dtype_code = ctypes.c_int()
    effort_mode = ctypes.c_int()
    gos_size = ctypes.c_int()
    code = lib.vpxc_inspect_volume(
        ptr,
        n,
        ctypes.byref(depth),
        ctypes.byref(height),
        ctypes.byref(width),
        ctypes.byref(dtype_code),
        ctypes.byref(effort_mode),
        ctypes.byref(gos_size),
    )
    if code != 0:
        _raise_from(lib, code)
    dtype_map = {1: "uint8", 2: "uint16", 3: "int16"}
    return {
        "depth": int(depth.value),
        "height": int(height.value),
        "width": int(width.value),
        "dtype": dtype_map.get(int(dtype_code.value), "unknown"),
        "effort": int(effort_mode.value),
        "gos_size": int(gos_size.value),
    }
