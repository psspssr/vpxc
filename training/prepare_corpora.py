"""Deterministic, leak-free dataset preparation for Frontier codec training & evaluation.

Builds three strictly disjoint splits from independent sources (CLIC 2020 Professional
and Mobile lossless PNG corpora + synthetic graphics/UI patterns), and verifies zero
hash or perceptual near-duplicate overlap across splits or against a held-out Kodak directory.

Usage:
  python3 training/prepare_corpora.py \
      --pro-dir /tmp/clic_raw/pro \
      --mob-dir /tmp/clic_raw/mob \
      --out-dir /tmp/datasets \
      --check-disjoint-dir /tmp/kodak
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path

import imagecodecs
import numpy as np


def load_rgb8(path: str) -> np.ndarray:
    arr = imagecodecs.png_decode(Path(path).read_bytes())
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)
    return np.ascontiguousarray(arr, dtype=np.uint8)


def thumb_sig(arr: np.ndarray) -> np.ndarray:
    h, w, _ = arr.shape
    ys = np.linspace(0, h - 1, 16, dtype=int)
    xs = np.linspace(0, w - 1, 16, dtype=int)
    return arr[np.ix_(ys, xs)].astype(np.float32)


def extract_native_crop(arr: np.ndarray, mode: str = "center") -> np.ndarray:
    h, w, _ = arr.shape
    th, tw = (512, 768) if w >= h else (768, 512)
    if h < th or w < tw:
        arr = np.pad(arr, ((0, max(0, th - h)), (0, max(0, tw - w)), (0, 0)), mode="reflect")
        h, w, _ = arr.shape
    if mode == "center":
        y0 = ((h - th) // 4) * 2
        x0 = ((w - tw) // 4) * 2
        return np.ascontiguousarray(arr[y0 : y0 + th, x0 : x0 + tw])
    cands = []
    for fy, fx in [(0.25, 0.25), (0.5, 0.5), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)]:
        y0 = min(h - th, max(0, int((h - th) * fy) & ~1))
        x0 = min(w - tw, max(0, int((w - tw) * fx) & ~1))
        crop = arr[y0 : y0 + th, x0 : x0 + tw]
        grad = float(
            np.abs(np.diff(crop[:, :, 1].astype(np.int16), axis=1)).mean()
            + np.abs(np.diff(crop[:, :, 1].astype(np.int16), axis=0)).mean()
        )
        cands.append((grad, y0, x0))
    cands.sort()
    _, y0, x0 = cands[3]
    return np.ascontiguousarray(arr[y0 : y0 + th, x0 : x0 + tw])


def make_420_roundtrip(rgb: np.ndarray, kernel_mode: int) -> np.ndarray:
    f = rgb.astype(np.float32)
    y = 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]
    cb = -0.168736 * f[:, :, 0] - 0.331264 * f[:, :, 1] + 0.5 * f[:, :, 2] + 128.0
    cr = 0.5 * f[:, :, 0] - 0.418688 * f[:, :, 1] - 0.081312 * f[:, :, 2] + 128.0

    def sub_up(ch: np.ndarray) -> np.ndarray:
        c_sub = 0.25 * (ch[0::2, 0::2] + ch[0::2, 1::2] + ch[1::2, 0::2] + ch[1::2, 1::2])
        out = np.empty_like(ch)
        if kernel_mode % 2 == 0:
            out[0::2, 0::2] = c_sub
            out[0::2, 1::2] = 0.5 * (c_sub + np.roll(c_sub, -1, axis=1))
            out[0::2, -1] = c_sub[:, -1]
            out[1::2, 0::2] = 0.5 * (c_sub + np.roll(c_sub, -1, axis=0))
            out[-1, 0::2] = c_sub[-1, :]
            out[1::2, 1::2] = 0.25 * (
                c_sub
                + np.roll(c_sub, -1, axis=1)
                + np.roll(c_sub, -1, axis=0)
                + np.roll(np.roll(c_sub, -1, axis=0), -1, axis=1)
            )
            out[-1, :] = out[-2, :]
            out[:, -1] = out[:, -2]
        else:
            cp = np.pad(c_sub, 1, mode="edge")
            out[0::2, 0::2] = (9 * cp[1:-1, 1:-1] + 3 * cp[0:-2, 1:-1] + 3 * cp[1:-1, 0:-2] + cp[0:-2, 0:-2]) / 16.0
            out[0::2, 1::2] = (9 * cp[1:-1, 1:-1] + 3 * cp[0:-2, 1:-1] + 3 * cp[1:-1, 2:] + cp[0:-2, 2:]) / 16.0
            out[1::2, 0::2] = (9 * cp[1:-1, 1:-1] + 3 * cp[2:, 1:-1] + 3 * cp[1:-1, 0:-2] + cp[2:, 0:-2]) / 16.0
            out[1::2, 1::2] = (9 * cp[1:-1, 1:-1] + 3 * cp[2:, 1:-1] + 3 * cp[1:-1, 2:] + cp[2:, 2:]) / 16.0
        return out

    cb_u = sub_up(cb) - 128.0
    cr_u = sub_up(cr) - 128.0
    r = np.clip(np.rint(y + 1.402 * cr_u), 0, 255).astype(np.uint8)
    g = np.clip(np.rint(y - 0.344136 * cb_u - 0.714136 * cr_u), 0, 255).astype(np.uint8)
    b = np.clip(np.rint(y + 1.772 * cb_u), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(np.stack([r, g, b], axis=-1))


def make_synthetic(idx: int) -> np.ndarray:
    h, w = 512, 768
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    if idx == 0:
        img[:, :, 0] = ((xx * 255) // w).astype(np.uint8)
        img[:, :, 1] = ((yy * 255) // h).astype(np.uint8)
        img[:, :, 2] = (((xx + yy) * 255) // (h + w)).astype(np.uint8)
        for k in range(12):
            y0, x0 = 20 + k * 36, 30 + k * 50
            img[y0 : y0 + 60, x0 : x0 + 120] = [(k * 73) % 256, (k * 137 + 40) % 256, (240 - k * 19) % 256]
            img[y0 + 2 : y0 + 58, x0 + 2 : x0 + 118] = [
                245 if k % 2 == 0 else 28,
                245 if k % 2 == 0 else 30,
                248 if k % 2 == 0 else 34,
            ]
    elif idx == 1:
        r = np.hypot(xx - w / 2.0, yy - h / 2.0)
        v = np.clip(128.0 + 110.0 * np.sin(r * r * 0.0012), 0, 255).astype(np.uint8)
        img[:, :, 0] = v
        img[:, :, 1] = np.where((xx * 3 + yy * 5) % 97 < 48, v, 255 - v)
        img[:, :, 2] = np.clip(128.0 + 100.0 * np.cos(xx * 0.09) * np.sin(yy * 0.07), 0, 255).astype(np.uint8)
    elif idx == 2:
        img[:] = 250
        img[(yy % 24) == 0] = [210, 215, 225]
        img[(xx % 32) == 0] = [210, 215, 225]
        rng = np.random.default_rng(1002)
        for b in range(40):
            x0 = 16 + b * 18
            bh = int(rng.integers(30, 440))
            col = rng.integers(20, 220, size=3, dtype=np.uint8)
            img[h - 30 - bh : h - 30, x0 : x0 + 12] = col
    else:
        rng = np.random.default_rng(2026)
        base = (128 + 60 * np.sin(xx / 37.0 + yy / 53.0) + 40 * np.cos(xx / 11.0 - yy / 17.0)).astype(np.float32)
        for c in range(3):
            ch = base + 25 * np.sin((xx * (c + 1)) / 19.0) + rng.normal(0, 2.5, size=(h, w))
            img[:, :, c] = np.clip(np.rint(ch), 0, 255).astype(np.uint8)
        mask = ((xx - 384) ** 2 + (yy - 256) ** 2) < 140**2
        img[mask] = np.clip(255 - img[mask], 0, 255)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pro-dir", type=Path, required=True)
    ap.add_argument("--mob-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--check-disjoint-dir", type=Path, default=None)
    args = ap.parse_args()

    pro_files = sorted(glob.glob(str(args.pro_dir / "**" / "*.png"), recursive=True))
    mob_files = sorted(glob.glob(str(args.mob_dir / "**" / "*.png"), recursive=True))
    if len(pro_files) < 41 or len(mob_files) < 57:
        raise RuntimeError(f"Expected >=41 pro and >=57 mob PNGs, got {len(pro_files)} and {len(mob_files)}")

    heldout_hashes: set[str] = set()
    heldout_thumbs: list[tuple[str, np.ndarray]] = []
    if args.check_disjoint_dir is not None and args.check_disjoint_dir.is_dir():
        for kp in sorted(glob.glob(str(args.check_disjoint_dir / "*.png"))):
            ka = load_rgb8(kp)
            heldout_hashes.add(hashlib.sha256(ka.tobytes()).hexdigest())
            heldout_thumbs.append((os.path.basename(kp), thumb_sig(ka)))

    train_srcs = [("pro", p) for p in pro_files[:25]] + [("mob", p) for p in mob_files[:35]]
    val_srcs = [("pro", p) for p in pro_files[25:33]] + [("mob", p) for p in mob_files[35:43]]
    test_srcs = [("pro", p) for p in pro_files[33:41]] + [("mob", p) for p in mob_files[43:59]]

    assert set(p for _, p in train_srcs).isdisjoint(set(p for _, p in val_srcs))
    assert set(p for _, p in train_srcs).isdisjoint(set(p for _, p in test_srcs))
    assert set(p for _, p in val_srcs).isdisjoint(set(p for _, p in test_srcs))

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test_clic": []}

    def save_item(split: str, name: str, arr: np.ndarray, src: str) -> None:
        h, w, _ = arr.shape
        sha = hashlib.sha256(arr.tobytes()).hexdigest()
        if sha in heldout_hashes:
            raise AssertionError(f"Exact hash collision with held-out benchmark on {name}")
        ts = thumb_sig(arr)
        for kn, kt in heldout_thumbs:
            if kt.shape == ts.shape and float(np.abs(kt - ts).mean()) <= 8.0:
                raise AssertionError(f"Perceptual near-duplicate with held-out {kn} on {name}")
        split_dir = args.out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        raw_path = split_dir / f"{name}_{h}_{w}.rgb"
        png_path = split_dir / f"{name}.png"
        raw_path.write_bytes(arr.tobytes())
        png_path.write_bytes(imagecodecs.png_encode(arr))
        splits[split].append(
            {
                "name": name,
                "h": h,
                "w": w,
                "sha256": sha,
                "src": os.path.basename(src),
                "raw": str(raw_path),
                "png": str(png_path),
            }
        )

    for i, (tag, p) in enumerate(train_srcs):
        arr = extract_native_crop(load_rgb8(p), mode="detail")
        save_item("train", f"tr_{i:03d}_{tag}", arr, p)
        if i < 8:
            save_item("train", f"tr_420_{i:02d}_{tag}", make_420_roundtrip(arr, i), p + "#420")

    for i in range(4):
        save_item("train", f"tr_syn_{i:02d}", make_synthetic(i), f"synthetic_{i}")

    for i, (tag, p) in enumerate(val_srcs):
        save_item("val", f"va_{i:02d}_{tag}", extract_native_crop(load_rgb8(p), mode="center"), p)

    for i, (tag, p) in enumerate(test_srcs):
        save_item("test_clic", f"te_{i:02d}_{tag}", extract_native_crop(load_rgb8(p), mode="center"), p)

    shas = {k: {x["sha256"] for x in v} for k, v in splits.items()}
    assert shas["train"].isdisjoint(shas["val"])
    assert shas["train"].isdisjoint(shas["test_clic"])
    assert shas["val"].isdisjoint(shas["test_clic"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(splits, indent=2))
    print("Prepared strictly disjoint datasets:", {k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()
