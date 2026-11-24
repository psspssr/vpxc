"""Medical 12/16-bit & 3D Volumetric CT/MRI Codec + Byte-Exact DICOM Wrapper.

Provides:
- ``encode_volume`` / ``decode_volume`` / ``decode_slice`` / ``inspect_volume``:
  3D volumetric lossless compression (``VPXV``) supporting ``uint8``, ``uint16``,
  and signed ``int16`` Hounsfield Units with Group-of-Slices (GoS) O(1) random
  access, per-slice + whole-volume CRC-32, and 100% IEEE-754-free decoding.
- ``pack_dicom`` / ``unpack_dicom`` / ``decode_dicom_pixels``:
  Zero-dependency DICOM Part 10 (Explicit & Implicit VR Little Endian) wrapper
  (``VPXD``) that losslessly compresses pixel payloads via ``vpxc`` while
  restoring 100% SHA-256 byte-for-byte identical ``.dcm`` files on unpack.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
import zlib

import numpy as np

from vpxc.codec import decode, encode
from vpxc.errors import CodecError
from vpxc.native import (
    decode_volume_native,
    decode_volume_slice_native,
    encode_volume_native,
    inspect_volume_native,
)

VPXD_MAGIC = b"VPXD"
VPXD_VERSION = 1
_VPXD_HEADER = struct.Struct("<4sBBBB32sIII")

# Explicit VR types whose 2-byte VR is followed by 2 reserved bytes + 4-byte uint32 length
_LONG_EXPLICIT_VRS = frozenset(
    {
        b"OB",
        b"OD",
        b"OF",
        b"OL",
        b"OV",
        b"OW",
        b"SQ",
        b"SV",
        b"UC",
        b"UN",
        b"UR",
        b"UT",
        b"UV",
    }
)

_UNCOMPRESSED_LE_UIDS = frozenset(
    {
        "1.2.840.10008.1.2",    # Implicit VR Little Endian
        "1.2.840.10008.1.2.1",  # Explicit VR Little Endian
    }
)


@dataclass(frozen=True)
class DicomPixelSpan:
    transfer_syntax_uid: str
    explicit_vr: bool
    rows: int
    cols: int
    frames: int
    samples_per_pixel: int
    bits_allocated: int
    pixel_representation: int  # 0 = unsigned, 1 = two's-complement signed
    planar_configuration: int  # 0 = chunky RGBRGB, 1 = planar R..G..B..
    pixel_offset: int
    pixel_nbytes: int


def encode_volume(volume: np.ndarray, *, effort: int = 0, gos_size: int = 8) -> bytes:
    """Compress a 3D medical volume ``(Z, H, W)`` in ``uint8``, ``uint16``, or ``int16``.

    ``effort=0`` selects the real-time PACS fast profile (4-way LUT rANS + 3D
    Delta-MED, multi-core GoS, millisecond random-access slice decode).
    ``effort=1..3`` selects the frontier 3D archival profile (bilateral + 30-tap
    spatial-Z predictor + MA context tree + Student-t Core66 rANS).
    """
    return encode_volume_native(volume, effort=effort, gos_size=gos_size)


def decode_volume(data: bytes) -> np.ndarray:
    """Decode a ``VPXV`` 3D medical volume bitstream to its exact ``(Z, H, W)`` NumPy array."""
    return decode_volume_native(data)


def decode_slice(data: bytes, z_index: int) -> np.ndarray:
    """Random-access decode a single axial slice ``z_index`` from a ``VPXV`` volume."""
    return decode_volume_slice_native(data, z_index)


def inspect_volume(data: bytes) -> dict[str, int | str]:
    """Read ``VPXV`` volume metadata without decoding voxel payloads."""
    return inspect_volume_native(data)


def _skip_undefined_length_value(buf: bytes, pos: int, end_tag: tuple[int, int]) -> int:
    n = len(buf)
    while pos + 8 <= n:
        grp, elem, length = struct.unpack_from("<HHI", buf, pos)
        pos += 8
        if (grp, elem) == end_tag:
            return pos
        if length == 0xFFFFFFFF:
            if (grp, elem) == (0xFFFE, 0xE000):
                pos = _skip_undefined_length_value(buf, pos, (0xFFFE, 0xE00D))
            else:
                pos = _skip_undefined_length_value(buf, pos, (0xFFFE, 0xE0DD))
        else:
            if pos + length > n:
                raise CodecError("truncated undefined-length DICOM item")
            pos += length
    raise CodecError("unterminated undefined-length DICOM sequence/item")


def _read_element(buf: bytes, pos: int, explicit_vr: bool) -> tuple[int, int, bytes, int, int, int]:
    """Return ``(group, elem, vr, value_offset, value_length, next_pos)``."""
    n = len(buf)
    if pos + 4 > n:
        raise CodecError("truncated DICOM tag")
    grp, elem = struct.unpack_from("<HH", buf, pos)
    if grp == 0xFFFE:
        if pos + 8 > n:
            raise CodecError("truncated DICOM item tag")
        length = struct.unpack_from("<I", buf, pos + 4)[0]
        val_off = pos + 8
        if length == 0xFFFFFFFF:
            end_tag = (0xFFFE, 0xE00D) if elem == 0xE000 else (0xFFFE, 0xE0DD)
            next_pos = _skip_undefined_length_value(buf, val_off, end_tag)
            return grp, elem, b"", val_off, next_pos - val_off, next_pos
        if val_off + length > n:
            raise CodecError("truncated DICOM item value")
        return grp, elem, b"", val_off, length, val_off + length

    use_explicit = True if grp == 0x0002 else explicit_vr
    if use_explicit:
        if pos + 8 > n:
            raise CodecError("truncated explicit-VR DICOM element")
        vr = buf[pos + 4 : pos + 6]
        if vr in _LONG_EXPLICIT_VRS:
            if pos + 12 > n:
                raise CodecError("truncated long explicit-VR DICOM element")
            length = struct.unpack_from("<I", buf, pos + 8)[0]
            val_off = pos + 12
        else:
            length = struct.unpack_from("<H", buf, pos + 6)[0]
            val_off = pos + 8
    else:
        if pos + 8 > n:
            raise CodecError("truncated implicit-VR DICOM element")
        vr = b""
        length = struct.unpack_from("<I", buf, pos + 4)[0]
        val_off = pos + 8

    if length == 0xFFFFFFFF:
        if (grp, elem) == (0x7FE0, 0x0010):
            raise CodecError("encapsulated/compressed DICOM PixelData is not raw uncompressed")
        next_pos = _skip_undefined_length_value(buf, val_off, (0xFFFE, 0xE0DD))
        return grp, elem, vr, val_off, next_pos - val_off, next_pos

    if val_off + length > n:
        raise CodecError("truncated DICOM element value")
    return grp, elem, vr, val_off, length, val_off + length


def parse_dicom_uncompressed(dcm_bytes: bytes) -> DicomPixelSpan:
    """Locate uncompressed Little-Endian PixelData and imaging tags in a DICOM Part 10 buffer."""
    if len(dcm_bytes) < 132 or dcm_bytes[128:132] != b"DICM":
        raise CodecError("not a DICOM Part 10 file (missing 128-byte preamble + DICM)")

    pos = 132
    n = len(dcm_bytes)
    ts_uid = "1.2.840.10008.1.2.1"

    # Parse Group 0002 File Meta Information (always Explicit VR Little Endian)
    while pos < n:
        grp, elem = struct.unpack_from("<HH", dcm_bytes, pos)
        if grp != 0x0002:
            break
        _, _, _, voff, vlen, pos = _read_element(dcm_bytes, pos, True)
        if (grp, elem) == (0x0002, 0x0010):
            ts_uid = dcm_bytes[voff : voff + vlen].rstrip(b"\x00 ").decode("ascii", errors="replace")

    if ts_uid not in _UNCOMPRESSED_LE_UIDS:
        raise CodecError(f"unsupported or already-compressed DICOM TransferSyntaxUID: {ts_uid}")

    explicit_vr = ts_uid != "1.2.840.10008.1.2"
    samples_per_pixel = 1
    frames = 1
    rows = 0
    cols = 0
    bits_allocated = 16
    pixel_rep = 0
    planar_cfg = 0

    while pos < n:
        grp, elem, _, voff, vlen, next_pos = _read_element(dcm_bytes, pos, explicit_vr)
        tag = (grp, elem)
        raw_val = dcm_bytes[voff : voff + vlen]
        if tag == (0x0028, 0x0002) and vlen >= 2:
            samples_per_pixel = struct.unpack_from("<H", raw_val, 0)[0]
        elif tag == (0x0028, 0x0006) and vlen >= 2:
            planar_cfg = struct.unpack_from("<H", raw_val, 0)[0]
        elif tag == (0x0028, 0x0008) and vlen > 0:
            txt = raw_val.rstrip(b"\x00 ").decode("ascii", errors="replace").strip()
            if txt:
                frames = int(txt)
        elif tag == (0x0028, 0x0010) and vlen >= 2:
            rows = struct.unpack_from("<H", raw_val, 0)[0]
        elif tag == (0x0028, 0x0011) and vlen >= 2:
            cols = struct.unpack_from("<H", raw_val, 0)[0]
        elif tag == (0x0028, 0x0100) and vlen >= 2:
            bits_allocated = struct.unpack_from("<H", raw_val, 0)[0]
        elif tag == (0x0028, 0x0103) and vlen >= 2:
            pixel_rep = struct.unpack_from("<H", raw_val, 0)[0]
        elif tag == (0x7FE0, 0x0010):
            if rows <= 0 or cols <= 0 or frames <= 0:
                raise CodecError("DICOM missing valid Rows/Columns/NumberOfFrames before PixelData")
            if bits_allocated not in (8, 16):
                raise CodecError(f"unsupported DICOM BitsAllocated={bits_allocated} (expected 8 or 16)")
            if samples_per_pixel not in (1, 3):
                raise CodecError(f"unsupported DICOM SamplesPerPixel={samples_per_pixel}")
            bytes_per_sample = bits_allocated // 8
            exact_nbytes = frames * rows * cols * samples_per_pixel * bytes_per_sample
            if vlen < exact_nbytes or voff + exact_nbytes > n:
                raise CodecError("DICOM PixelData shorter than Rows*Columns*Frames*BytesPerSample")
            return DicomPixelSpan(
                transfer_syntax_uid=ts_uid,
                explicit_vr=explicit_vr,
                rows=rows,
                cols=cols,
                frames=frames,
                samples_per_pixel=samples_per_pixel,
                bits_allocated=bits_allocated,
                pixel_representation=pixel_rep,
                planar_configuration=planar_cfg,
                pixel_offset=voff,
                pixel_nbytes=exact_nbytes,
            )
        pos = next_pos

    raise CodecError("DICOM dataset contains no (7FE0,0010) PixelData element")


def pack_dicom(dcm_bytes: bytes, *, effort: int = 1, gos_size: int = 8) -> bytes:
    """Losslessly pack an uncompressed ``.dcm`` file into a byte-exact ``VPXD`` archive."""
    span = parse_dicom_uncompressed(dcm_bytes)
    sha256 = hashlib.sha256(dcm_bytes).digest()

    prefix = dcm_bytes[: span.pixel_offset]
    raw_pixels = dcm_bytes[span.pixel_offset : span.pixel_offset + span.pixel_nbytes]
    suffix = dcm_bytes[span.pixel_offset + span.pixel_nbytes :]

    if span.bits_allocated == 8:
        dt = np.dtype(np.int8 if span.pixel_representation == 1 else np.uint8)
    else:
        dt = np.dtype("<i2" if span.pixel_representation == 1 else "<u2")

    arr = np.frombuffer(raw_pixels, dtype=dt)

    # Codec kind:
    # 0 = VPXV 3D grayscale volume (supports uint8, uint16, signed int16; 1 or N frames)
    # 1 = VPXC 2D multi-channel image (RGB uint8/uint16)
    flags = 0
    if span.samples_per_pixel == 1:
        if arr.dtype == np.int8:
            # Map signed int8 [-128..127] losslessly to uint8 via ^ 0x80
            vol = (arr.view(np.uint8) ^ np.uint8(0x80)).reshape(span.frames, span.rows, span.cols)
            flags |= 0x01  # restore int8 XOR on unpack
        else:
            vol = arr.reshape(span.frames, span.rows, span.cols)
        codec_kind = 0
        comp_pixels = encode_volume(vol, effort=effort, gos_size=gos_size)
    else:
        if span.frames != 1:
            raise CodecError("multi-frame RGB DICOM not yet supported; use grayscale 3D or single-frame RGB")
        if arr.dtype == np.int16:
            img = (arr.view(np.uint16) ^ np.uint16(0x8000)).reshape(span.rows, span.cols, 3)
            flags |= 0x02
        elif arr.dtype == np.int8:
            img = (arr.view(np.uint8) ^ np.uint8(0x80)).reshape(span.rows, span.cols, 3)
            flags |= 0x01
        else:
            img = arr.reshape(span.rows, span.cols, 3)
        if span.planar_configuration == 1:
            # Planar RRR..GGG..BBB.. -> HxWx3
            img = np.moveaxis(arr.reshape(3, span.rows, span.cols), 0, -1)
            if img.dtype == np.int16:
                img = img.view(np.uint16) ^ np.uint16(0x8000)
                flags |= 0x02
            flags |= 0x04
        codec_kind = 1
        comp_pixels = encode(img, effort=effort, profile="fast" if effort == 0 else "frontier")

    comp_prefix = zlib.compress(prefix, level=9)
    comp_suffix = zlib.compress(suffix, level=9)

    hdr = _VPXD_HEADER.pack(
        VPXD_MAGIC,
        VPXD_VERSION,
        codec_kind,
        flags,
        0,
        sha256,
        len(comp_prefix),
        len(comp_pixels),
        len(comp_suffix),
    )
    return hdr + comp_prefix + comp_pixels + comp_suffix


def unpack_dicom(vpxd_bytes: bytes) -> bytes:
    """Unpack a ``VPXD`` archive to a cryptographically verified byte-exact ``.dcm`` buffer."""
    if len(vpxd_bytes) < _VPXD_HEADER.size:
        raise CodecError("truncated VPXD header")
    magic, ver, codec_kind, flags, _rsv, expected_sha, pre_len, pix_len, suf_len = _VPXD_HEADER.unpack_from(
        vpxd_bytes, 0
    )
    if magic != VPXD_MAGIC or ver != VPXD_VERSION:
        raise CodecError("not a readable VPXD archive")
    total_need = _VPXD_HEADER.size + pre_len + pix_len + suf_len
    if total_need != len(vpxd_bytes):
        raise CodecError("corrupt VPXD section lengths")

    off = _VPXD_HEADER.size
    try:
        prefix = zlib.decompress(vpxd_bytes[off : off + pre_len])
        off += pre_len
        comp_pixels = vpxd_bytes[off : off + pix_len]
        off += pix_len
        suffix = zlib.decompress(vpxd_bytes[off : off + suf_len])
    except zlib.error as exc:
        raise CodecError(f"corrupt VPXD metadata stream: {exc}") from exc

    if codec_kind == 0:
        vol = decode_volume(comp_pixels)
        if flags & 0x01:
            raw_pixels = (vol ^ np.uint8(0x80)).tobytes()
        else:
            raw_pixels = np.ascontiguousarray(vol).astype(vol.dtype.newbyteorder("<"), copy=False).tobytes()
    elif codec_kind == 1:
        img = decode(comp_pixels)
        if flags & 0x04:
            img = np.moveaxis(img, -1, 0)
        if flags & 0x02:
            raw_pixels = np.ascontiguousarray(img ^ np.uint16(0x8000)).astype("<u2", copy=False).tobytes()
        elif flags & 0x01:
            raw_pixels = np.ascontiguousarray(img ^ np.uint8(0x80)).tobytes()
        else:
            dt = "<u2" if img.dtype == np.uint16 else "|u1"
            raw_pixels = np.ascontiguousarray(img).astype(dt, copy=False).tobytes()
    else:
        raise CodecError(f"unknown VPXD codec kind {codec_kind}")

    restored = prefix + raw_pixels + suffix
    if hashlib.sha256(restored).digest() != expected_sha:
        raise CodecError("VPXD SHA-256 verification failed")
    return restored


def decode_dicom_pixels(vpxd_bytes: bytes, z_index: int | None = None) -> np.ndarray:
    """Decode pixel array (or a single random-access slice ``z_index``) directly from ``VPXD``."""
    if len(vpxd_bytes) < _VPXD_HEADER.size:
        raise CodecError("truncated VPXD header")
    magic, ver, codec_kind, flags, _rsv, _sha, pre_len, pix_len, suf_len = _VPXD_HEADER.unpack_from(vpxd_bytes, 0)
    if magic != VPXD_MAGIC or ver != VPXD_VERSION:
        raise CodecError("not a readable VPXD archive")
    if _VPXD_HEADER.size + pre_len + pix_len + suf_len != len(vpxd_bytes):
        raise CodecError("corrupt VPXD section lengths")
    comp_pixels = vpxd_bytes[_VPXD_HEADER.size + pre_len : _VPXD_HEADER.size + pre_len + pix_len]
    if codec_kind == 0:
        if z_index is not None:
            sl = decode_slice(comp_pixels, z_index)
            return (sl ^ np.uint8(0x80)).view(np.int8) if (flags & 0x01) else sl
        vol = decode_volume(comp_pixels)
        if flags & 0x01:
            vol = (vol ^ np.uint8(0x80)).view(np.int8)
        return vol[0] if vol.shape[0] == 1 else vol
    if codec_kind == 1:
        if z_index not in (None, 0):
            raise CodecError("single-frame 2D VPXD only has slice 0")
        img = decode(comp_pixels)
        if flags & 0x02:
            return (img ^ np.uint16(0x8000)).view(np.int16)
        if flags & 0x01:
            return (img ^ np.uint8(0x80)).view(np.int8)
        return img
    raise CodecError(f"unknown VPXD codec kind {codec_kind}")


def create_multiframe_dicom(volume: np.ndarray, *, modality: str = "CT") -> bytes:
    """Build a minimal compliant Explicit VR Little Endian multi-frame DICOM Part 10 buffer."""
    arr = np.asarray(volume)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3 or arr.dtype not in (np.uint8, np.uint16, np.int16):
        raise CodecError("create_multiframe_dicom expects (Z, H, W) uint8, uint16, or int16")
    z, h, w = arr.shape
    bits = 8 if arr.dtype == np.uint8 else 16
    pix_rep = 1 if arr.dtype == np.int16 else 0

    def elem_short(grp: int, el: int, vr: bytes, val: bytes) -> bytes:
        if len(val) & 1:
            val = val + (b" " if vr in (b"UI", b"CS", b"IS", b"LO", b"SH") else b"\x00")
        if vr == b"UI" and val.endswith(b" "):
            val = val[:-1] + b"\x00"
        return struct.pack("<HH2sH", grp, el, vr, len(val)) + val

    def elem_long(grp: int, el: int, vr: bytes, val: bytes) -> bytes:
        if len(val) & 1:
            val = val + b"\x00"
        return struct.pack("<HH2sHI", grp, el, vr, 0, len(val)) + val

    ts_uid = b"1.2.840.10008.1.2.1"
    meta_body = (
        elem_long(0x0002, 0x0001, b"OB", b"\x00\x01")
        + elem_short(0x0002, 0x0002, b"UI", b"1.2.840.10008.5.1.4.1.1.2")
        + elem_short(0x0002, 0x0003, b"UI", b"1.2.826.0.1.3680043.10.1")
        + elem_short(0x0002, 0x0010, b"UI", ts_uid)
    )
    meta = elem_short(0x0002, 0x0000, b"UL", struct.pack("<I", len(meta_body))) + meta_body

    pix_bytes = np.ascontiguousarray(arr).astype(arr.dtype.newbyteorder("<"), copy=False).tobytes()
    vr_pix = b"OW" if bits == 16 else b"OB"
    dataset = (
        elem_short(0x0008, 0x0060, b"CS", modality.encode("ascii"))
        + elem_short(0x0028, 0x0002, b"US", struct.pack("<H", 1))
        + elem_short(0x0028, 0x0004, b"CS", b"MONOCHROME2")
        + elem_short(0x0028, 0x0008, b"IS", str(z).encode("ascii"))
        + elem_short(0x0028, 0x0010, b"US", struct.pack("<H", h))
        + elem_short(0x0028, 0x0011, b"US", struct.pack("<H", w))
        + elem_short(0x0028, 0x0100, b"US", struct.pack("<H", bits))
        + elem_short(0x0028, 0x0101, b"US", struct.pack("<H", bits))
        + elem_short(0x0028, 0x0102, b"US", struct.pack("<H", bits - 1))
        + elem_short(0x0028, 0x0103, b"US", struct.pack("<H", pix_rep))
        + elem_long(0x7FE0, 0x0010, vr_pix, pix_bytes)
    )
    return (b"\x00" * 128) + b"DICM" + meta + dataset
