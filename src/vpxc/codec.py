"""Tile container for the SLGF lossless image codec.

Stage 0 selects a reversible channel bijection over ``Z^C`` (identity, YCoCg-R,
subtract-base, 2-D-gradient Q12 shear, or 10-multiplier SL(3)-WSIL eigenframe,
plus optional 4th-channel coupling). Each tile then stores the smallest exact
representation among:

- ``slgf``: unified 4-level ``Q^2 = 2I`` linear quincunx tower + 3 causal
  sub-cosets + 4-way normative ``Q15.16``/``Q1.30`` ``I-SSM`` + closed-form
  Cholesky MDL Ridge head (``phi_head``) + CFA/4:2:0 parity cross-channel taps
  (``phi_acq``) + bimodal Student-t ``Dyadic-Core66 rANS``
- ``kalman``: causal Median Edge Detection with ``D_8``-equivariant Sheaf-Kalman
  patch fusion (collapsing to exact ``D_8`` copy when ``sigma_eta -> 0``)
- ``palette``: zero-rounding palette mode for tiles with ``<= 512`` unique colors
  (grayscale or multi-channel)
- ``qx-rice`` / ``qx-rans`` / ``raster``: standalone ablation profiles
- ``raw``: uncompressed tile bypass capping worst-case expansion on noise

All decoding paths are 100% IEEE-754-free integer operations verified by a
whole-image CRC-32 checksum.
"""

from __future__ import annotations

import struct
import zlib
from typing import Iterable

import numpy as np

from vpxc.color import (
    MODE_IDENTITY,
    MODE_WSIL,
    ColorTransform,
    apply_color_forward,
    apply_color_inverse,
    choose_color_transform,
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
from vpxc.slgf_model import decode_slgf_tile, encode_slgf_tile
from vpxc.slgf_v3 import decode_slgf_v3, encode_slgf_v3
from vpxc.spatial import decode_raster, encode_raster
from vpxc.wsil import SL3Params

MAGIC = b"VPXC"
VERSION = 3
VERSIONS = (2, 3)
TILE = 768
MAX_LEVELS = 4
MAX_PALETTE_COLORS = 512

PROFILE_RAW = 0
PROFILE_RASTER = 1
PROFILE_KALMAN = 2
PROFILE_QX_RICE = 3
PROFILE_QX_RANS = 4
PROFILE_PALETTE = 5
PROFILE_SLGF = 6

PROFILES = ("raw", "raster", "kalman", "qx-rice", "qx-rans", "palette", "slgf", "slgf-zero")

FLAG_SHEAF = 1
COLOR_FLAG_EXTRA = 1

_HEADER = struct.Struct("<4sBBBBBBIIIH")
_SHEAR_STRUCT = struct.Struct("<hhh")
_WSIL_STRUCT = struct.Struct("<BBBhhhhhhhhhh")


def encode(
    image: np.ndarray,
    *,
    tile: int = TILE,
    effort: int = 2,
    profile: str | None = None,
    version: int = VERSION,
) -> bytes:
    """Compress ``image`` (``uint8`` or ``uint16``, ``HxW`` or ``HxWx{1,3,4}``).

    ``profile`` forces one tile coder (``slgf``, ``slgf-zero``, ``kalman``,
    ``palette``, ``qx-rans``, ``qx-rice``, ``raster``, or ``raw``). By default
    ``encode`` evaluates the candidates enabled by ``effort`` (``1``, ``2``, or
    ``3``) and writes the shortest exact tile representation. ``version=2``
    writes the previous container.
    """
    from vpxc.native import encode_native

    native = encode_native(image, tile=tile, effort=effort, profile=profile, version=version)
    if native is not None:
        return native
    src, dtype = _normalize(image)
    if tile < 8 or tile > 4096:
        raise CodecError("tile size must be between 8 and 4096")
    if version not in VERSIONS:
        raise CodecError("unsupported bitstream version")
    if version == 2 and effort not in (1, 2):
        raise CodecError("version 2 effort must be 1 or 2")
    if version == 3 and effort not in (1, 2, 3):
        raise CodecError("effort must be 1, 2, or 3")
    height, width, channels = _dims(src)
    color_tr = _select_color(src)
    crc = zlib.crc32(_sample_bytes(src)) & 0xFFFFFFFF

    has_extra = channels == 4 and any(w != 0 for w in color_tr.extra_weights)
    cflags = COLOR_FLAG_EXTRA if has_extra else 0
    header = bytearray(
        _HEADER.pack(
            MAGIC,
            version,
            1 if dtype == np.uint8 else 2,
            channels,
            color_tr.mode,
            color_tr.perm,
            cflags,
            width,
            height,
            crc,
            tile,
        )
    )
    if color_tr.mode == MODE_WSIL:
        w = color_tr.wsil
        header += _WSIL_STRUCT.pack(
            w.perm[0],
            w.perm[1],
            w.perm[2],
            w.u01,
            w.u02,
            w.u12,
            w.l10,
            w.l20,
            w.l21,
            w.s1,
            w.s1_inv,
            w.s2,
            w.s2_inv,
        )
    elif color_tr.mode != MODE_IDENTITY:
        header += _SHEAR_STRUCT.pack(
            int(color_tr.shear[0]),
            int(color_tr.shear[1]),
            int(color_tr.shear[2]),
        )
    if has_extra:
        header += _SHEAR_STRUCT.pack(
            int(color_tr.extra_weights[0]),
            int(color_tr.extra_weights[1]),
            int(color_tr.extra_weights[2]),
        )

    lifted = None
    if version == 3:
        lifted = apply_color_forward(_split(src), color_tr)
    parts: list[bytes] = [bytes(header)]
    for y0, y1, x0, x1 in _tiles(height, width, tile):
        block = src[y0:y1, x0:x1]
        halos = None if lifted is None else _halos(lifted, y0, x0, y1, x1)
        parts.append(
            _encode_tile(
                block,
                color_tr,
                effort,
                profile,
                version=version,
                halos=halos,
            )
        )
    return b"".join(parts)


def decode(data: bytes) -> np.ndarray:
    """Exact integer inverse of :func:`encode`. Raises :class:`CodecError` on bad data."""
    from vpxc.native import decode_native

    native = decode_native(data)
    if native is not None:
        return native
    if len(data) < _HEADER.size:
        raise CodecError("truncated header")
    (
        magic,
        version,
        bps,
        channels,
        mode,
        perm,
        cflags,
        width,
        height,
        crc,
        tile,
    ) = _HEADER.unpack_from(data, 0)
    if magic != MAGIC or version not in VERSIONS:
        raise CodecError("not a readable vpxc bitstream")
    if bps not in (1, 2) or channels not in (1, 3, 4):
        raise CodecError("unsupported sample format")
    if width <= 0 or height <= 0 or tile < 8 or width > 20000 or height > 20000:
        raise CodecError("invalid geometry")

    pos = _HEADER.size
    shear = (0, 0, 0)
    wsil = SL3Params()
    if mode == MODE_WSIL:
        if pos + _WSIL_STRUCT.size > len(data):
            raise CodecError("truncated WSIL header")
        vals = _WSIL_STRUCT.unpack_from(data, pos)
        pos += _WSIL_STRUCT.size
        p_tup = (int(vals[0]), int(vals[1]), int(vals[2]))
        if sorted(p_tup) != [0, 1, 2]:
            raise CodecError("invalid WSIL channel permutation")
        wsil = SL3Params(
            perm=p_tup,
            u01=int(vals[3]),
            u02=int(vals[4]),
            u12=int(vals[5]),
            l10=int(vals[6]),
            l20=int(vals[7]),
            l21=int(vals[8]),
            s1=int(vals[9]),
            s1_inv=int(vals[10]),
            s2=int(vals[11]),
            s2_inv=int(vals[12]),
        )
    elif mode != MODE_IDENTITY:
        if pos + _SHEAR_STRUCT.size > len(data):
            raise CodecError("truncated color header")
        a, c, d = _SHEAR_STRUCT.unpack_from(data, pos)
        pos += _SHEAR_STRUCT.size
        shear = (int(a), int(c), int(d))

    extra_weights = (0, 0, 0)
    if cflags & COLOR_FLAG_EXTRA:
        if pos + _SHEAR_STRUCT.size > len(data):
            raise CodecError("truncated 4th-channel color header")
        ew0, ew1, ew2 = _SHEAR_STRUCT.unpack_from(data, pos)
        pos += _SHEAR_STRUCT.size
        extra_weights = (int(ew0), int(ew1), int(ew2))

    color_tr = ColorTransform(
        mode=int(mode),
        perm=int(perm),
        shear=shear,
        wsil=wsil,
        extra_weights=extra_weights,
    )

    dtype = np.uint8 if bps == 1 else np.uint16
    if channels == 1:
        out = np.empty((height, width), dtype)
    else:
        out = np.empty((height, width, channels), dtype)

    for y0, y1, x0, x1 in _tiles(height, width, tile):
        block, pos = _decode_tile(
            data,
            pos,
            y1 - y0,
            x1 - x0,
            channels,
            dtype,
            color_tr,
            version=version,
            dest=out,
            origin=(y0, x0),
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


def _select_color(src: np.ndarray) -> ColorTransform:
    if src.ndim != 3 or src.shape[2] < 3:
        return ColorTransform()
    chans = [src[:, :, i].astype(np.int32) for i in range(src.shape[2])]
    return choose_color_transform(chans)


def _tiles(
    height: int, width: int, tile: int
) -> Iterable[tuple[int, int, int, int]]:
    for y0 in range(0, height, tile):
        for x0 in range(0, width, tile):
            yield y0, min(height, y0 + tile), x0, min(width, x0 + tile)


def _split(tile: np.ndarray) -> list[np.ndarray]:
    if tile.ndim == 2:
        return [tile.astype(np.int32, copy=False)]
    return [tile[:, :, i].astype(np.int32, copy=False) for i in range(tile.shape[2])]


def _halos(
    lifted: list[np.ndarray], y0: int, x0: int, y1: int, x1: int, radius: int = 16
) -> list[tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]]:
    pt = min(radius, y0)
    pl = min(radius, x0)
    pt -= pt & 1
    pl -= pl & 1
    halos = []
    for plane in lifted:
        top = plane[y0 - pt : y0, x0:x1] if pt else None
        left = plane[y0:y1, x0 - pl : x0] if pl else None
        corner = plane[y0 - pt : y0, x0 - pl : x0] if pt and pl else None
        halos.append((top, left, corner))
    return halos


def _halos_from_decoded(
    dest: np.ndarray,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
    color_tr: ColorTransform,
) -> list[tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]]:
    pt = min(16, y0)
    pl = min(16, x0)
    pt -= pt & 1
    pl -= pl & 1

    def lift(y_a: int, y_b: int, x_a: int, x_b: int) -> list[np.ndarray]:
        return apply_color_forward(_split(dest[y_a:y_b, x_a:x_b]), color_tr)

    tops = lift(y0 - pt, y0, x0, x1) if pt else None
    lefts = lift(y0, y1, x0 - pl, x0) if pl else None
    corners = lift(y0 - pt, y0, x0 - pl, x0) if pt and pl else None
    if tops is None and lefts is None and corners is None:
        channels = 1 if dest.ndim == 2 else int(dest.shape[2])
        return [(None, None, None) for _ in range(channels)]
    count = len(tops or lefts or corners or [])
    halos = []
    for index in range(count):
        halos.append(
            (
                None if tops is None else tops[index],
                None if lefts is None else lefts[index],
                None if corners is None else corners[index],
            )
        )
    return halos


def _encode_tile(
    tile: np.ndarray,
    color_tr: ColorTransform,
    effort: int,
    profile: str | None,
    version: int = VERSION,
    halos: list | None = None,
) -> bytes:
    planes = apply_color_forward(_split(tile), color_tr)
    if profile is not None:
        forced = _forced_record(tile, planes, profile, effort, version, halos)
        if forced is None:
            raise CodecError(f"profile {profile} does not apply to this tile")
        return forced

    candidates = [_raw_record(tile), _slgf_record(planes, effort, version, halos, True)]
    palette = _palette_record(tile, effort=effort)
    if palette is not None:
        candidates.append(palette)
    if (effort >= 2 or min(tile.shape[:2]) < 32) and tile.shape[0] * tile.shape[1] <= 65536:
        candidates.append(_raster_record(planes, kalman=True))
    return min(candidates, key=len)


def _forced_record(
    tile: np.ndarray,
    planes: list[np.ndarray],
    profile: str,
    effort: int = 2,
    version: int = VERSION,
    halos: list | None = None,
) -> bytes | None:
    if profile == "raw":
        return _raw_record(tile)
    if profile == "raster":
        return _raster_record(planes, kalman=False)
    if profile == "kalman":
        return _raster_record(planes, kalman=True)
    if profile == "palette":
        return _palette_record(tile, effort=2)
    if profile == "slgf":
        return _slgf_record(planes, effort, version, halos, True)
    if profile == "slgf-zero":
        return _slgf_record(planes, min(effort, 2), version, halos, False)
    if profile in ("qx-rice", "qx-rans"):
        records = _quincunx_candidates(
            planes, effort=2, entropies=("rans" if profile == "qx-rans" else "rice",)
        )
        if not records:
            return _raster_record(planes, kalman=False)
        return min(records, key=len)
    raise CodecError(f"unknown profile {profile}")


def _raw_record(tile: np.ndarray) -> bytes:
    return bytes([PROFILE_RAW]) + _sample_bytes(tile)


def _slgf_record(
    planes: list[np.ndarray],
    effort: int = 2,
    version: int = VERSION,
    halos: list | None = None,
    use_static: bool = True,
) -> bytes:
    if version <= 2:
        payload = encode_slgf_tile(planes, max_levels=MAX_LEVELS)
    else:
        legacy = encode_slgf_tile(planes, max_levels=MAX_LEVELS)
        modern = encode_slgf_v3(
            planes,
            max_levels=MAX_LEVELS,
            effort=effort,
            halos=halos,
            use_static=use_static,
        )
        # Bit 7 of the depth byte selects the v3 payload. The legacy layout
        # stays available so a tile never grows just to carry an unused tool.
        if len(modern) < len(legacy) and modern:
            payload = bytes([modern[0] | 0x80]) + modern[1:]
        else:
            payload = legacy
    return bytes([PROFILE_SLGF, 0, 0]) + _len_prefix(payload)


def _raster_record(planes: list[np.ndarray], kalman: bool) -> bytes:
    blobs = []
    mask = np.ones(planes[0].shape, np.bool_)
    for plane in planes:
        blobs.append(_len_prefix(encode_raster(plane, mask, kalman)))
    prof = PROFILE_KALMAN if kalman else PROFILE_RASTER
    return bytes([prof, 0, 0]) + _len_prefix(b"".join(blobs))


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
    prof = PROFILE_QX_RANS if entropy == "rans" else PROFILE_QX_RICE
    return bytes([prof, flags, 0 if nlevels is None else nlevels]) + _len_prefix(
        b"".join(blobs)
    )


def _palette_med_residuals(index: np.ndarray) -> np.ndarray:
    """Vectorized causal MED prediction residuals on a 2-D palette index plane."""
    idx = index.astype(np.int32, copy=False)
    h, w = idx.shape
    out = np.empty((h, w), dtype=np.int32)
    out[0, 0] = idx[0, 0]
    if w > 1:
        out[0, 1:] = idx[0, 1:] - idx[0, :-1]
    if h > 1:
        out[1:, 0] = idx[1:, 0] - idx[:-1, 0]
    if h > 1 and w > 1:
        left = idx[1:, :-1]
        up = idx[:-1, 1:]
        ul = idx[:-1, :-1]
        hi = np.maximum(left, up)
        lo = np.minimum(left, up)
        pred = np.where(ul < lo, hi, np.where(ul > hi, lo, left + up - ul))
        out[1:, 1:] = idx[1:, 1:] - pred
    return out


def _palette_med_reconstruct(resid: np.ndarray) -> np.ndarray:
    h, w = resid.shape
    rows = resid.astype(np.int32, copy=False).tolist()
    out = [[0] * w for _ in range(h)]
    for y in range(h):
        row = out[y]
        rrow = rows[y]
        up_row = out[y - 1] if y > 0 else None
        for x in range(w):
            if y == 0 and x == 0:
                row[0] = rrow[0]
            elif y == 0:
                row[x] = row[x - 1] + rrow[x]
            elif x == 0 and up_row is not None:
                row[0] = up_row[0] + rrow[0]
            else:
                assert up_row is not None
                left = row[x - 1]
                up = up_row[x]
                ul = up_row[x - 1]
                hi = left if left > up else up
                lo = up if left > up else left
                if ul < lo:
                    pred = hi
                elif ul > hi:
                    pred = lo
                else:
                    pred = left + up - ul
                row[x] = pred + rrow[x]
    return np.asarray(out, dtype=np.int32)


def _palette_record(tile: np.ndarray, effort: int = 2) -> bytes | None:
    """Zero-rounding palette mode for grayscale and color tiles (<= 512 unique colors)."""
    if tile.ndim == 2:
        flat = np.ascontiguousarray(tile).reshape(-1, 1)
    elif tile.ndim == 3:
        flat = np.ascontiguousarray(tile).reshape(-1, tile.shape[-1])
    else:
        return None
    if flat.shape[0] == 0:
        return None

    void_dt = np.dtype((np.void, flat.dtype.itemsize * flat.shape[1]))
    keys = np.ascontiguousarray(flat).view(void_dt).ravel()
    uniq, inv = np.unique(keys, return_inverse=True)
    if uniq.size == 0 or uniq.size > MAX_PALETTE_COLORS:
        return None

    colors = np.ascontiguousarray(uniq.view(flat.dtype).reshape(-1, flat.shape[1]))
    lum_order = np.argsort(colors.astype(np.int64).sum(axis=1), kind="stable")
    rank = np.empty_like(lum_order)
    rank[lum_order] = np.arange(colors.shape[0])
    colors = np.ascontiguousarray(colors[lum_order])
    index = rank[inv].astype(np.int32).reshape(tile.shape[:2])

    mask = np.ones(index.shape, np.bool_)
    zeros_shift = np.zeros(index.shape, np.uint8)
    med_res = _palette_med_residuals(index)
    # Core66 rANS breaks the 1-bit-per-pixel floor of Rice on sparse UI index maps:
    b_rans = encode_dyadic_rans(med_res, mask, zeros_shift)
    mode_flag = 2
    chosen_idx_blob = b_rans

    b_med = encode_raster(index, mask, kalman=False)
    if len(b_med) < len(chosen_idx_blob):
        mode_flag = 0
        chosen_idx_blob = b_med
    if effort >= 2 and index.size <= 65536:
        b_kal = encode_raster(index, mask, kalman=True)
        if len(b_kal) < len(chosen_idx_blob):
            mode_flag = 1
            chosen_idx_blob = b_kal

    payload = bytearray()
    payload += int(colors.shape[0]).to_bytes(2, "little")
    payload += _sample_bytes(colors)
    payload += _len_prefix(chosen_idx_blob)
    return bytes([PROFILE_PALETTE, mode_flag, 0]) + _len_prefix(bytes(payload))


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
    color_tr: ColorTransform,
    version: int = VERSION,
    dest: np.ndarray | None = None,
    origin: tuple[int, int] = (0, 0),
) -> tuple[np.ndarray, int]:
    if pos >= len(data):
        raise CodecError("truncated tile")
    prof = data[pos]
    pos += 1
    if prof == PROFILE_RAW:
        nbytes = height * width * channels * (1 if dtype == np.uint8 else 2)
        if pos + nbytes > len(data):
            raise CodecError("truncated raw tile")
        raw = data[pos : pos + nbytes]
        pos += nbytes
        if dtype == np.uint8:
            arr = np.frombuffer(raw, dtype=np.uint8)
        else:
            arr = np.frombuffer(raw, dtype="<u2")
        arr = arr.reshape(
            (height, width) if channels == 1 else (height, width, channels)
        )
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

    if prof == PROFILE_SLGF:
        if version >= 3 and payload and (payload[0] & 0x80):
            y0, x0 = origin
            halos = None
            if dest is not None:
                halos = _halos_from_decoded(dest, y0, x0, y0 + height, x0 + width, color_tr)
            modern = bytes([payload[0] & 0x7F]) + payload[1:]
            planes = decode_slgf_v3(modern, height, width, channels, halos)
        else:
            planes = decode_slgf_tile(payload, height, width, channels)
        restored = apply_color_inverse(planes, color_tr)
        return _cast(restored, dtype, channels), pos
    if prof in (PROFILE_RASTER, PROFILE_KALMAN):
        planes = _decode_raster_planes(
            payload, height, width, channels, prof == PROFILE_KALMAN
        )
        restored = apply_color_inverse(planes, color_tr)
        return _cast(restored, dtype, channels), pos
    if prof in (PROFILE_QX_RICE, PROFILE_QX_RANS):
        planes = _decode_qx_planes(
            payload,
            height,
            width,
            channels,
            nlevels,
            bool(flags & FLAG_SHEAF),
            prof == PROFILE_QX_RANS,
        )
        restored = apply_color_inverse(planes, color_tr)
        return _cast(restored, dtype, channels), pos
    if prof == PROFILE_PALETTE:
        return (
            _decode_palette(payload, height, width, channels, dtype, int(flags)),
            pos,
        )
    raise CodecError(f"unknown tile profile {prof}")


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
    mode_flag: int,
) -> np.ndarray:
    if len(payload) < 2:
        raise CodecError("truncated palette")
    ncolors = int.from_bytes(payload[0:2], "little")
    if ncolors <= 0 or ncolors > MAX_PALETTE_COLORS:
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
    if mode_flag == 2:
        zeros_shift = np.zeros((height, width), np.uint8)
        resid = decode_dyadic_rans(index_blob, (height, width), mask, zeros_shift)
        index = _palette_med_reconstruct(resid)
    elif mode_flag in (0, 1):
        index = decode_raster(index_blob, (height, width), mask, mode_flag == 1)
    else:
        raise CodecError(f"unknown palette mode flag {mode_flag}")
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
