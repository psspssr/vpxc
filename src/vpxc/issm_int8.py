"""Normative INT8 context stem and packed weight file.

The decode graph is integer: depthwise 3x3 filters, a pointwise mix, and one
Q15.16 row scan. Weights ship in ``weights/slgf_v1.bin`` (well under the
1.8 MB cap) and are checked with SHA-256. ``--profile slgf-zero`` never reads
them. Training lives in ``training/train_issm.py`` and rewrites this file.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from vpxc.errors import CodecError
from vpxc.issm import tree_rows

_MAGIC = b"SLG1"
_PATH = Path(__file__).with_name("weights") / "slgf_v1.bin"

# Analytic initialization used when the file is first created.
_DW = np.array(
    [
        [0, -8, 0, -8, 32, -8, 0, -8, 0],
        [-1, -2, -1, 0, 0, 0, 1, 2, 1],
        [-1, 0, 1, -2, 0, 2, -1, 0, 1],
        [0, -4, 4, -4, 0, 4, 4, -4, 0],
        [4, -4, 0, 4, 0, -4, 0, -4, 4],
        [-2, 0, 4, 0, -4, 0, 4, 0, -2],
        [0, 0, 0, 8, -16, 8, 0, 0, 0],
        [0, 8, 0, 0, -16, 0, 0, 8, 0],
    ],
    dtype=np.int8,
)
_PW = np.array([24, 8, 8, 4, 4, 4, 6, 6], dtype=np.int8)
_HEAD = np.array([16, -12, -12, 8, 8, -6, 10, -8, 4, -4, 6, -6, 0, 0, 0, 0], dtype=np.int8)


def pack_weights(
    head: np.ndarray | None = None,
    dw: np.ndarray | None = None,
    pw: np.ndarray | None = None,
) -> bytes:
    head_b = np.asarray(_HEAD if head is None else head, dtype=np.int8).reshape(16)
    dw_b = np.asarray(_DW if dw is None else dw, dtype=np.int8).reshape(8, 9)
    pw_b = np.asarray(_PW if pw is None else pw, dtype=np.int8).reshape(8)
    payload = head_b.tobytes() + dw_b.tobytes() + pw_b.tobytes()
    digest = hashlib.sha256(payload).digest()
    return _MAGIC + payload + digest


def load_weights(blob: bytes | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if blob is None:
        ensure_default_file()
        raw = _PATH.read_bytes()
    else:
        raw = blob
    if len(raw) != 4 + 16 + 72 + 8 + 32 or not raw.startswith(_MAGIC):
        raise CodecError("INT8 weight file is the wrong size or magic")
    payload, digest = raw[4:-32], raw[-32:]
    if hashlib.sha256(payload).digest() != digest:
        raise CodecError("INT8 weight file failed SHA-256")
    head = np.frombuffer(payload[:16], dtype=np.int8).copy()
    dw = np.frombuffer(payload[16:88], dtype=np.int8).copy().reshape(8, 9)
    pw = np.frombuffer(payload[88:96], dtype=np.int8).copy()
    return head, dw, pw


def ensure_default_file() -> Path:
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _PATH.exists():
        _PATH.write_bytes(pack_weights())
    return _PATH


def context_map(filled: np.ndarray, weights: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Integer context field. ``filled`` is the causal even lattice (odds already predicted)."""
    _head, dw, pw = load_weights() if weights is None else weights
    src = np.asarray(filled, dtype=np.int64)
    padded = np.pad(src, 1, mode="edge")
    height, width = src.shape
    mixed = np.zeros((height, width), dtype=np.int64)
    for k in range(8):
        kernel = dw[k].astype(np.int64)
        acc = np.zeros((height, width), dtype=np.int64)
        for dy in range(3):
            for dx in range(3):
                acc += kernel[dy * 3 + dx] * padded[dy : dy + height, dx : dx + width]
        mixed += int(pw[k]) * acc
    mixed = (mixed + 4) >> 3
    decay = np.full(src.shape, 60000, dtype=np.int64)
    bias = np.clip(mixed, -2000, 2000)
    state = tree_rows(decay, bias)
    return (state >> 4) + mixed


def project_head(phi_q4: np.ndarray, head: np.ndarray | None = None) -> np.ndarray:
    """Q8 dot of the leading phi columns with the packed INT8 head."""
    if head is None:
        head, _dw, _pw = load_weights()
    n = min(int(phi_q4.shape[1]), int(head.size))
    if n <= 0:
        return np.zeros(phi_q4.shape[0], dtype=np.int64)
    return (phi_q4[:, :n].astype(np.int64) @ head[:n].astype(np.int64) + 128) >> 8
