"""Tile container for the SLGF-Zero lossless codec.

Stage 0 picks a reversible channel lift (identity, YCoCg-R, subtract-base, or a
Q12 shear). Each tile then stores the smallest of:

- raw samples (worst-case expansion is one profile byte per tile)
- raster median prediction with adaptive Rice
- the same raster predictor with causal Sheaf-Kalman blending (M <= 4)
- a 4-level sheaf or flat quincunx tower, coarse band in Rice, detail bands in
  either Rice or Dyadic-Core66 rANS
- a palette plus Rice-coded indexes, when the tile uses at most 256 colors

The quincunx sheaf weights and the color-lift parameters are exactly invertible,
so a decoded tile matches the input samples before the CRC is checked.
"""

from __future__ import annotations

import struct
import zlib
from typing import Iterable

import numpy as np

from vpxc.color import (
    MODE_IDENTITY,
    choose_color,
    forward_channels,
    inverse_channels,
)
from vpxc.entropy import (
    decode_dyadic_rans,
    decode_rice_values,
    encode_dyadic_rans,
    encode_rice_values,
    shift_map_from_activity,
)
from vpxc.errors import CodecError
from vpxc.lattice import (
    activity_context,
    build_levels,
    forward_tower,
    lift_inverse,
    parity_masks,
    unpack_even,
)
from vpxc.spatial import decode_raster, encode_raster

MAGIC = b"VPXC"
VERSION = 1
TILE = 256
MAX_LEVELS = 4

PROFILE_RAW = 0
PROFILE_RASTER = 1
PROFILE_KALMAN = 2
PROFILE_QX_RICE = 3
PROFILE_QX_RANS = 4
PROFILE_PALETTE = 5

FLAG_SHEAF = 1

_HEADER = struct.Struct("<4sBBBBBhhhIIIH")


def encode(
    image: np.ndarray,
    *,
    tile: int = TILE,
    effort: int = 2,
    profile: str | None = None,
) -> bytes:
    """Compress ``image`` (``uint8`` or ``uint16``, ``HxW`` or ``HxWx{1,3,4}``).

    ``profile`` forces one tile coder (``raw``, ``raster``, ``kalman``,
    ``qx-rice``, ``qx-rans``, ``palette``). The default compares every coder
    enabled by ``effort`` and stores the smallest.
    """
    src, dtype = _normalize(image)
    if tile < 8 or tile > 4096:
        raise CodecError("tile size must be between 8 and 4096")
    height, width, channels = _dims(src)
    mode, perm, shear = _select_color(src)
    crc = zlib.crc32(_sample_bytes(src)) & 0xFFFFFFFF
    header = _HEADER.pack(
        MAGIC,
        VERSION,
        1 if dtype == np.uint8 else 2,
        channels,
        mode,
        perm,
        int(shear[0]),
        int(shear[1]),
        int(shear[2]),
        width,
        height,
        crc,
        tile,
    )
    parts = [header]
    for y0, y1, x0, x1 in _tiles(height, width, tile):
        block = src[y0:y1, x0:x1] if src.ndim == 3 else src[y0:y1, x0:x1]
        parts.append(_encode_tile(block, mode, perm, shear, effort, profile))
    return b"".join(parts)


def decode(data: bytes) -> np.ndarray:
    """Inverse of :func:`encode`. Raises :class:`CodecError` on a bad payload."""
    if len(data) < _HEADER.size:
        raise CodecError("truncated header")
    (
        magic,
        version,
        bps,
        channels,
        mode,
        perm,
        a,
        c,
        d,
        width,
        height,
        crc,
        tile,
    ) = _HEADER.unpack_from(data, 0)
    if magic != MAGIC or version != VERSION:
        raise CodecError("not a vpxc version-1 bitstream")
    if bps not in (1, 2) or channels not in (1, 3, 4):
        raise CodecError("unsupported sample format")
    if width <= 0 or height <= 0 or tile < 8:
        raise CodecError("invalid geometry")
    dtype = np.uint8 if bps == 1 else np.uint16
    if channels == 1:
        out = np.empty((height, width), dtype)
    else:
        out = np.empty((height, width, channels), dtype)
    pos = _HEADER.size
    shear = (int(a), int(c), int(d))
    for y0, y1, x0, x1 in _tiles(height, width, tile):
        block, pos = _decode_tile(
            data, pos, y1 - y0, x1 - x0, channels, dtype, mode, perm, shear
        )
        if channels == 1:
            out[y0:y1, x0:x1] = block
        else:
            out[y0:y1, x0:x1, :] = block
    if pos != len(data):
        raise CodecError("trailing bytes after the last tile")
    got = zlib.crc32(_sample_bytes(out)) & 0xFFFFFFFF
    if got != crc:
        raise CodecError("CRC mismatch")
    return out


def _normalize(image: np.ndarray) -> tuple[np.ndarray, np.dtype]:
    src = np.asarray(image)
    if src.dtype not in (np.uint8, np.uint16):
        raise CodecError("image dtype must be uint8 or uint16")
    if src.ndim == 3 and src.shape[2] == 1:
        src = src[:, :, 0]
    if src.ndim not in (2, 3):
        raise CodecError("image must be HxW or HxWxC")
    if src.size == 0:
        raise CodecError("empty image")
    _dims(src)
    return np.ascontiguousarray(src), src.dtype


def _dims(src: np.ndarray) -> tuple[int, int, int]:
    if src.ndim == 2:
        height, width = src.shape
        channels = 1
    else:
        height, width, channels = src.shape
        if channels not in (3, 4):
            raise CodecError("channel count must be 1, 3, or 4")
    if height <= 0 or width <= 0:
        raise CodecError("empty image")
    if height > 20000 or width > 20000:
        raise CodecError("image dimension exceeds 20000")
    return int(height), int(width), int(channels)


def _sample_bytes(src: np.ndarray) -> bytes:
    if src.dtype == np.uint8:
        return np.ascontiguousarray(src).tobytes()
    return np.ascontiguousarray(src).astype("<u2", copy=False).tobytes()


def _select_color(src: np.ndarray) -> tuple[int, int, tuple[int, int, int]]:
    if src.ndim != 3 or src.shape[2] < 3:
        return MODE_IDENTITY, 0, (0, 0, 0)
    rgb = [src[:, :, i].astype(np.int32) for i in range(3)]
    return choose_color(rgb)


def _tiles(height: int, width: int, tile: int) -> Iterable[tuple[int, int, int, int]]:
    for y0 in range(0, height, tile):
        for x0 in range(0, width, tile):
            yield y0, min(height, y0 + tile), x0, min(width, x0 + tile)


def _split(tile: np.ndarray) -> list[np.ndarray]:
    if tile.ndim == 2:
        return [tile.astype(np.int32, copy=False)]
    return [tile[:, :, i].astype(np.int32, copy=False) for i in range(tile.shape[2])]


def _encode_tile(
    tile: np.ndarray,
    mode: int,
    perm: int,
    shear: tuple[int, int, int],
    effort: int,
    profile: str | None,
) -> bytes:
    planes = forward_channels(_split(tile), mode, perm, shear)
    if profile is not None:
        forced = _forced_record(tile, planes, profile)
        if forced is None:
            raise CodecError(f"profile {profile} does not apply to this tile")
        return forced
    candidates = [_raw_record(tile), _raster_record(planes, kalman=False)]
    if effort >= 2:
        candidates.append(_raster_record(planes, kalman=True))
        palette = _palette_record(tile)
        if palette is not None:
            candidates.append(palette)
    candidates.extend(_quincunx_candidates(planes, effort))
    return min(candidates, key=len)


def _forced_record(
    tile: np.ndarray, planes: list[np.ndarray], profile: str
) -> bytes | None:
    if profile == "raw":
        return _raw_record(tile)
    if profile == "raster":
        return _raster_record(planes, kalman=False)
    if profile == "kalman":
        return _raster_record(planes, kalman=True)
    if profile == "palette":
        return _palette_record(tile)
    if profile in ("qx-rice", "qx-rans"):
        records = _quincunx_candidates(
            planes, effort=2, entropies=("rans" if profile == "qx-rans" else "rice",)
        )
        if not records:
            # Too small to lift: the raster coder is the quincunx coder with zero levels.
            return _raster_record(planes, kalman=False)
        return min(records, key=len)
    raise CodecError(f"unknown profile {profile}")


def _raw_record(tile: np.ndarray) -> bytes:
    return bytes([PROFILE_RAW]) + _sample_bytes(tile)


def _raster_record(planes: list[np.ndarray], kalman: bool) -> bytes:
    blobs = []
    mask = np.ones(planes[0].shape, np.bool_)
    for plane in planes:
        blobs.append(_len_prefix(encode_raster(plane, mask, kalman)))
    profile = PROFILE_KALMAN if kalman else PROFILE_RASTER
    return bytes([profile, 0, 0]) + _len_prefix(b"".join(blobs))


def _quincunx_candidates(
    planes: list[np.ndarray],
    effort: int,
    entropies: tuple[str, ...] | None = None,
) -> list[bytes]:
    shape = planes[0].shape
    geom, _coarse = build_levels(shape[0], shape[1], MAX_LEVELS)
    if not geom:
        return []
    analyses = []
    for sheaf in (False, True):
        towers = [forward_tower(plane, MAX_LEVELS, sheaf) for plane in planes]
        analyses.append((sheaf, towers, _tower_cost(towers)))
    analyses.sort(key=lambda item: item[2])
    # Keep the smoother tower, and the other one only when it is clearly close
    # enough that entropy coding (not the L1 proxy) might prefer it.
    chosen = [analyses[0]]
    if len(analyses) == 2 and analyses[1][2] <= analyses[0][2] * 1.02 + 0.05:
        chosen.append(analyses[1])
    records = []
    if entropies is None:
        entropies = ("rice", "rans") if effort >= 2 else ("rice",)
    for sheaf, towers, _cost in chosen:
        for entropy in entropies:
            records.append(_qx_record(towers, sheaf, entropy))
    return records


def _tower_cost(towers) -> float:
    total = 0
    count = 0
    for _coarse, _valid, details, geom in towers:
        for lifted, (valid, _info, _pmask) in zip(details, geom):
            odd, _even = parity_masks(valid)
            if np.any(odd):
                total += int(np.abs(lifted[odd], dtype=np.int64).sum())
                count += int(odd.sum())
    if count == 0:
        return 0.0
    return total / count


def _qx_record(towers, sheaf: bool, entropy: str) -> bytes:
    blobs = []
    nlevels = None
    for coarse, coarse_valid, details, geom in towers:
        if nlevels is None:
            nlevels = len(geom)
        elif nlevels != len(geom):
            raise CodecError("planes disagree on quincunx depth")
        blobs.append(_len_prefix(encode_raster(coarse, coarse_valid, kalman=False)))
        paired = list(zip(details, geom))
        for lifted, (valid, _info, _pmask) in reversed(paired):
            odd, _even = parity_masks(valid)
            ctx = activity_context(lifted, valid)
            if entropy == "rans":
                shifts = shift_map_from_activity(ctx)
                blob = encode_dyadic_rans(lifted, odd, shifts)
            else:
                blob = encode_rice_values(lifted, odd, ctx, causal_mu=False)
            blobs.append(_len_prefix(blob))
    flags = FLAG_SHEAF if sheaf else 0
    profile = PROFILE_QX_RANS if entropy == "rans" else PROFILE_QX_RICE
    return bytes([profile, flags, 0 if nlevels is None else nlevels]) + _len_prefix(
        b"".join(blobs)
    )


def _palette_record(tile: np.ndarray) -> bytes | None:
    if tile.ndim != 3:
        return None
    flat = np.ascontiguousarray(tile).reshape(-1, tile.shape[-1])
    if flat.shape[0] == 0:
        return None
    void_dt = np.dtype((np.void, flat.dtype.itemsize * flat.shape[1]))
    keys = np.ascontiguousarray(flat).view(void_dt).ravel()
    uniq = np.unique(keys)
    if uniq.size == 0 or uniq.size > 256:
        return None
    index = np.searchsorted(uniq, keys).astype(np.int32).reshape(tile.shape[:2])
    colors = np.ascontiguousarray(uniq.view(flat.dtype).reshape(-1, flat.shape[1]))
    payload = bytearray()
    payload += int(colors.shape[0]).to_bytes(2, "little")
    payload += _sample_bytes(colors)
    mask = np.ones(index.shape, np.bool_)
    payload += _len_prefix(encode_raster(index, mask, kalman=False))
    return bytes([PROFILE_PALETTE, 0, 0]) + _len_prefix(bytes(payload))


def _len_prefix(blob: bytes) -> bytes:
    if len(blob) >= 2**32:
        raise CodecError("tile payload exceeds 4 GiB")
    return len(blob).to_bytes(4, "little") + blob


def _decode_tile(
    data: bytes,
    pos: int,
    height: int,
    width: int,
    channels: int,
    dtype: np.dtype,
    mode: int,
    perm: int,
    shear: tuple[int, int, int],
) -> tuple[np.ndarray, int]:
    if pos >= len(data):
        raise CodecError("truncated tile")
    profile = data[pos]
    pos += 1
    if profile == PROFILE_RAW:
        nbytes = height * width * channels * (1 if dtype == np.uint8 else 2)
        if pos + nbytes > len(data):
            raise CodecError("truncated raw tile")
        raw = data[pos : pos + nbytes]
        pos += nbytes
        if dtype == np.uint8:
            arr = np.frombuffer(raw, dtype=np.uint8)
        else:
            arr = np.frombuffer(raw, dtype="<u2")
        arr = arr.reshape((height, width) if channels == 1 else (height, width, channels))
        return arr.copy(), pos
    if pos + 2 + 4 > len(data):
        raise CodecError("truncated tile header")
    flags = data[pos]
    nlevels = data[pos + 1]
    pos += 2
    length = int.from_bytes(data[pos : pos + 4], "little")
    pos += 4
    if pos + length > len(data):
        raise CodecError("truncated tile payload")
    payload = data[pos : pos + length]
    pos += length
    if profile in (PROFILE_RASTER, PROFILE_KALMAN):
        planes = _decode_raster_planes(
            payload, height, width, channels, profile == PROFILE_KALMAN
        )
        restored = inverse_channels(planes, mode, perm, shear)
        return _cast(restored, dtype, channels), pos
    if profile in (PROFILE_QX_RICE, PROFILE_QX_RANS):
        planes = _decode_qx_planes(
            payload,
            height,
            width,
            channels,
            nlevels,
            bool(flags & FLAG_SHEAF),
            profile == PROFILE_QX_RANS,
        )
        restored = inverse_channels(planes, mode, perm, shear)
        return _cast(restored, dtype, channels), pos
    if profile == PROFILE_PALETTE:
        return _decode_palette(payload, height, width, channels, dtype), pos
    raise CodecError(f"unknown tile profile {profile}")


def _read_prefixed(payload: bytes, cursor: int) -> tuple[bytes, int]:
    if cursor + 4 > len(payload):
        raise CodecError("truncated length prefix")
    length = int.from_bytes(payload[cursor : cursor + 4], "little")
    cursor += 4
    if cursor + length > len(payload):
        raise CodecError("truncated prefixed blob")
    return payload[cursor : cursor + length], cursor + length


def _decode_raster_planes(
    payload: bytes, height: int, width: int, channels: int, kalman: bool
) -> list[np.ndarray]:
    cursor = 0
    planes = []
    mask = np.ones((height, width), np.bool_)
    for _ in range(channels):
        blob, cursor = _read_prefixed(payload, cursor)
        planes.append(decode_raster(blob, (height, width), mask, kalman))
    if cursor != len(payload):
        raise CodecError("raster tile has trailing bytes")
    return planes


def _decode_qx_planes(
    payload: bytes,
    height: int,
    width: int,
    channels: int,
    nlevels: int,
    sheaf: bool,
    use_rans: bool,
) -> list[np.ndarray]:
    geom, _final = build_levels(height, width, nlevels)
    if len(geom) != nlevels:
        raise CodecError("quincunx depth does not match this tile")
    cursor = 0
    planes = []
    for _channel in range(channels):
        blob, cursor = _read_prefixed(payload, cursor)
        if nlevels == 0:
            mask = np.ones((height, width), np.bool_)
            planes.append(decode_raster(blob, (height, width), mask, False))
            continue
        coarse_valid = geom[-1][2]
        current = decode_raster(blob, coarse_valid.shape, coarse_valid, False)
        for level in range(nlevels - 1, -1, -1):
            valid, info, _pmask = geom[level]
            detail, cursor = _read_prefixed(payload, cursor)
            spatial = unpack_even(current, info, valid)
            odd, _even = parity_masks(valid)
            ctx = activity_context(spatial, valid)
            if use_rans:
                shifts = shift_map_from_activity(ctx)
                resid = decode_dyadic_rans(detail, spatial.shape, odd, shifts)
            else:
                resid = decode_rice_values(detail, spatial.shape, odd, ctx, False)
            spatial[odd] = resid[odd]
            current = lift_inverse(spatial, valid, sheaf)
        planes.append(current)
    if cursor != len(payload):
        raise CodecError("quincunx tile has trailing bytes")
    return planes


def _decode_palette(
    payload: bytes,
    height: int,
    width: int,
    channels: int,
    dtype: np.dtype,
) -> np.ndarray:
    if len(payload) < 2:
        raise CodecError("truncated palette")
    ncolors = int.from_bytes(payload[0:2], "little")
    if ncolors <= 0 or ncolors > 256:
        raise CodecError("palette size out of range")
    sample_bytes = 1 if dtype == np.uint8 else 2
    nbytes = ncolors * channels * sample_bytes
    if 2 + nbytes > len(payload):
        raise CodecError("truncated palette colors")
    raw = payload[2 : 2 + nbytes]
    if dtype == np.uint8:
        colors = np.frombuffer(raw, dtype=np.uint8).reshape(ncolors, channels)
    else:
        colors = np.frombuffer(raw, dtype="<u2").reshape(ncolors, channels)
    index_blob, cursor = _read_prefixed(payload, 2 + nbytes)
    if cursor != len(payload):
        raise CodecError("palette tile has trailing bytes")
    mask = np.ones((height, width), np.bool_)
    index = decode_raster(index_blob, (height, width), mask, False)
    if int(index.min()) < 0 or int(index.max()) >= ncolors:
        raise CodecError("palette index out of range")
    flat = colors[index.ravel()]
    if channels == 1:
        return flat.reshape(height, width).astype(dtype, copy=True)
    return flat.reshape(height, width, channels).astype(dtype, copy=True)


def _cast(planes: list[np.ndarray], dtype: np.dtype, channels: int) -> np.ndarray:
    if channels == 1:
        arr = planes[0]
    else:
        arr = np.stack(planes, axis=-1)
    info = np.iinfo(dtype)
    lo = int(arr.min())
    hi = int(arr.max())
    if lo < int(info.min) or hi > int(info.max):
        raise CodecError(f"decoded sample {lo}..{hi} outside {dtype.__name__}")
    return arr.astype(dtype, copy=False)

