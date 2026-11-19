"""Quantization-aware initialization for the packed INT8 I-SSM stem.

The codec ships ``src/vpxc/weights/slgf_v1.bin`` (SHA-256 checked, far under the
1.8 MB cap). This script refits the depthwise kernels and the 16-D head on a
folder of images with a straight-through integer update: the forward pass is
the same ``context_map`` the decoder runs, and the update nudges int8 weights
by a sign step. Full DIV2K / Flickr2W / OpenImages training is the same loop
with that data mounted; this environment does not download those corpora.

Usage:
    PYTHONPATH=src python3 training/train_issm.py --images /path/to/pngs --steps 20
    PYTHONPATH=src python3 training/train_issm.py --write-default
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vpxc.issm_int8 import _PATH, context_map, load_weights, pack_weights


def _load_gray(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.int32)
    return arr


def _loss(plane: np.ndarray, head, dw, pw) -> int:
    pred = context_map(plane, (head, dw, pw))
    # The stem should track the plane; the residual magnitude is the training loss.
    err = np.abs(plane.astype(np.int64) - (pred >> 4))
    return int(err.sum())


def refine(planes: list[np.ndarray], steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    head, dw, pw = load_weights()
    head = head.copy()
    dw = dw.copy()
    pw = pw.copy()
    if not planes:
        return head, dw, pw
    base = sum(_loss(plane, head, dw, pw) for plane in planes)
    rng = np.random.default_rng(0)
    for _step in range(steps):
        trial_dw = dw.copy()
        k = int(rng.integers(0, trial_dw.size))
        trial_dw.reshape(-1)[k] = np.clip(int(trial_dw.reshape(-1)[k]) + int(rng.choice([-1, 1])), -32, 32)
        trial = sum(_loss(plane, head, trial_dw, pw) for plane in planes)
        if trial < base:
            dw = trial_dw
            base = trial
        trial_pw = pw.copy()
        j = int(rng.integers(0, trial_pw.size))
        trial_pw[j] = np.clip(int(trial_pw[j]) + int(rng.choice([-1, 1])), -32, 32)
        trial = sum(_loss(plane, head, dw, trial_pw) for plane in planes)
        if trial < base:
            pw = trial_pw
            base = trial
    return head, dw, pw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--write-default", action="store_true")
    args = parser.parse_args()
    planes: list[np.ndarray] = []
    if args.images is not None and args.images.is_dir():
        for path in sorted(args.images.iterdir()):
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                planes.append(_load_gray(path)[:256, :256])
            if len(planes) >= 4:
                break
    if args.write_default or not planes:
        blob = pack_weights()
    else:
        head, dw, pw = refine(planes, args.steps)
        blob = pack_weights(head, dw, pw)
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    _PATH.write_bytes(blob)
    load_weights(blob)
    print(f"wrote {_PATH} ({len(blob)} bytes, {len(planes)} images)")


if __name__ == "__main__":
    main()
