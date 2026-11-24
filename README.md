# `vpxc` — Fast, Deterministic Lossless Medical & Photographic Image Compression

`vpxc` is an open-source **lossless image and 3D clinical volume codec** (native C++20 engine with portable `x86_64 AVX2` & `ARM64 NEON` integer SIMD, Python bindings, and CLI) for 8-bit/16-bit grayscale, RGB, RGBA images, **3D signed/unsigned CT & MRI stacks (`VPXV`)**, and **byte-exact DICOM archives (`VPXD`)**.

Most high-ratio lossless compressors force a painful tradeoff: either fast coding with bloated files (`PNG`, `JPEG-LS`, `JPEG 2000` / `HTJ2K`), or multi-second encoding per slice/image to squeeze out the last few percent (`JPEG XL -e 9` or GPU neural codecs). **`vpxc` closes both sides of that gap on pure CPU:**

- **Medical 12/16-Bit & 3D Volumetric CT/MRI (`VPXV` + `VPXD`):** Compresses held-out 16-bit signed Hounsfield Unit clinical CT volumes **`24.9%` smaller than `JPEG 2000 (5/3)` (`HTJ2K`)**, **`16.6%` smaller than `JPEG-LS`**, and **`5.0%` smaller than `JPEG XL (-e 9)`** (`25x` faster encode), with **`0.5–3.2 ms` O(1) random-access single-slice decoding** and **cryptographic `SHA-256` byte-exact `.dcm` DICOM restoration**.
- **Sub-10ms Real-Time Fast Profile (`-e 0` / `--profile fast`):** 4-way lockstep 12-bit flat-LUT `rANS` + decoupled MED/3D-Delta-MED decodes `768×512` RGB24 photos in **`7.3 ms`** (`6.6 ms` encode) and full 672-slice 16-bit CT stacks at **`> 320 MB/s`**.
- **Smaller than `JPEG XL (-e 9)`, `WebP Lossless`, `JPEG 2000`, and `PNG`** on held-out 2D photographic benchmarks at **`~90–113 ms` default encode (`-e 1`)** (`11–12x` faster than `JPEG XL -e 9`).
- **100% IEEE-754-free & fuzz-hardened C++20 decoder:** Every decode path (`frontier`, `fast`, and `VPXV` 3D medical) compiles to **zero floating-point instructions** (verified via `objdump` disassembly in CI across `x86_64` and `ARM64`), enforces strict pre-allocation zip-bomb caps + per-slice/whole-volume `CRC-32`, and passes **350,000+ `libFuzzer + ASan + UBSan` iterations** with zero errors.

---

## Side-by-Side Comparison: Uncompressed vs. `vpxc` Lossless

<table>
  <tr>
    <th width="50%">Uncompressed Source (<code>1,179,648 bytes</code> raw RGB24)</th>
    <th width="50%">Compressed with <code>vpxc -e 1</code> (<code>451,715 bytes</code>) &amp; Restored</th>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="docs/assets/kawaguchiko_original.png" alt="Original uncompressed 768x512 RGB24 photograph of Lake Kawaguchi" width="100%" />
    </td>
    <td width="50%" align="center">
      <img src="docs/assets/kawaguchiko_vpxc_restored.png" alt="Restored 768x512 RGB24 photograph decoded from 451,715-byte vpxc stream" width="100%" />
    </td>
  </tr>
  <tr>
    <td valign="top">
      <ul>
        <li><b>Format:</b> Raw 8-bit RGB24 (<code>768 &times; 512</code>)</li>
        <li><b>Uncompressed size:</b> <code>1,179,648 bytes</code> (<code>8.00 bpsp</code>)</li>
        <li><b>Optimized PNG (<code>level 9</code>):</b> <code>626,407 bytes</code> (<code>174.3 ms</code> enc)</li>
        <li><b>Pixel SHA-256:</b> <code>e06e6c5a06cf09a5...</code></li>
      </ul>
    </td>
    <td valign="top">
      <ul>
        <li><b><code>.vpxc</code> size (<code>-e 1</code>):</b> <b><code>451,715 bytes</code></b> (<b><code>3.06 bpsp</code></b>, <b><code>2.61&times;</code></b> ratio)</li>
        <li><b>Savings:</b> <b><code>-61.7%</code></b> vs. raw &middot; <b><code>-27.9%</code></b> vs. PNG &middot; <b><code>-3.8%</code></b> vs. JPEG XL (<code>-e 9</code>)</li>
        <li><b>Speed:</b> <b><code>102.7 ms</code></b> encode &middot; <b><code>95.9 ms</code></b> decode</li>
        <li><b>Quality loss:</b> <b>Zero</b> (<code>Max Error = 0</code>, <code>PSNR = &infin;</code>, identical SHA-256 <code>e06e6c5a06cf09a5...</code>)</li>
      </ul>
    </td>
  </tr>
</table>

<details>
<summary><b>All codecs on this photograph (<code>Lake Kawaguchi</code>, <code>768&times;512</code> RGB24 — CC0 1.0 Public Domain)</b></summary>

<br />

| Codec | Compressed Size | Rate (`bpsp`) | Ratio vs. Raw | Encode (`ms`) | Decode (`ms`) | Exact Lossless |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| **`vpxc (-e 2)`** | **449,076 B** | **3.0455** | **2.63x** | **356.6** | **97.0** | Yes |
| **`vpxc (-e 3)`** | **449,966 B** | **3.0515** | **2.62x** | **849.8** | **86.7** | Yes |
| **`vpxc (-e 1, default)`** | **451,715 B** | **3.0634** | **2.61x** | **102.7** | **95.9** | Yes |
| `JPEG XL (-d 0 -e 9)` | 469,714 B | 3.1855 | 2.51x | 1,231.9 | 84.1 | Yes |
| `JPEG XL (-d 0 -e 7)` | 478,636 B | 3.2460 | 2.46x | 401.4 | 86.2 | Yes |
| `WebP-L (lossless)` | 486,162 B | 3.2970 | 2.43x | 154.6 | 6.8 | Yes |
| `JPEG XL (-d 0 -e 3)` | 494,497 B | 3.3535 | 2.39x | 47.1 | 32.5 | Yes |
| `JPEG 2000 (5/3)` | 510,727 B | 3.4636 | 2.31x | 80.3 | 70.3 | Yes |
| `JPEG-LS` | 528,110 B | 3.5815 | 2.23x | 19.2 | 15.2 | Yes |
| `PNG (level 9)` | 626,407 B | 4.2481 | 1.88x | 174.3 | 5.3 | Yes |

*Image credit: [Lake Kawaguchi jetties with Mount Setto in the background](https://commons.wikimedia.org/wiki/File:Kawaguchiko_rowing_center_jetties_on_Lake_Kawaguchi,_with_Mount_Setto_in_the_background,_Fujikawaguchiko,_2016.jpg) via Wikimedia Commons, dedicated to the public domain under [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).*

</details>

---

## Performance & Benchmark Suites

### Zero-Overfitting Evaluation Protocol

Universal `frontier` priors (`cpp/include/vpxc/frontier_priors.hpp`, ~5.5 KB of integer tables) were trained offline (`training/prepare_corpora.py`, `training/train_frontier.cpp`) using **8-fold image-grouped cross-validation** and **James-Stein $t$-statistic shrinkage** on an independent `CLIC` + synthetic training split. Every training and validation image was verified via exact SHA-256 and $16 \times 16$ perceptual distance to have **zero overlap** with both blind held-out test suites below.

All measurements below report **actual serialized container bytes** verified bit-for-bit (`np.array_equal(decoded, original)`), running in-process on the same CPU (`vpxc` C++20 native library vs. `libjxl`, `libwebp`, `OpenJPEG`, `CharLS`, and `libpng` via `imagecodecs`). Rates are shown in **`bpsp`** (bits per channel sample; raw 8-bit RGB = `8.00 bpsp`) and **`bpp`** (bits per pixel; raw 8-bit RGB = `24.00 bpp`).

### 1. Held-Out Modern `4:4:4` Photography (`TEST_CLIC`, 24 Disjoint Photographs, `768×512` RGB24)

On native `4:4:4` photographs, **every `vpxc` effort preset — including `~100 ms` `-e 1` — compresses smaller than `JPEG XL (-e 9)`**:

| Codec | Rate (`bpsp`) | Rate (`bpp`) | Ratio | Encode (`ms/img`) | Decode (`ms/img`) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **`vpxc (-e 3, tile=768)`** | **2.7906** | **8.3718** | **2.87x** | **1,193.4** | **241.6** |
| **`vpxc (-e 3, tile=384)`** | **2.7962** | **8.3886** | **2.86x** | **808.1** | **96.5** |
| **`vpxc (-e 2, tile=384)`** | **2.8070** | **8.4210** | **2.85x** | **331.0** | **99.9** |
| **`vpxc (-e 1, default 384)`** | **2.8215** | **8.4645** | **2.84x** | **112.9** | **99.6** |
| **`vpxc (-e 1, tile=256)`** | **2.8262** | **8.4786** | **2.83x** | **99.6** | **85.0** |
| `JPEG XL (-d 0 -e 9)` | 2.8651 | 8.5953 | 2.79x | 1,247.9 | 84.3 |
| `JPEG XL (-d 0 -e 7)` | 2.9040 | 8.7120 | 2.75x | 400.6 | 83.5 |
| `JPEG XL (-d 0 -e 5)` | 2.9555 | 8.8665 | 2.71x | 204.6 | 71.9 |
| `JPEG XL (-d 0 -e 3)` | 3.0388 | 9.1164 | 2.63x | 45.0 | 32.4 |
| `WebP-L (lossless)` | 3.2021 | 9.6063 | 2.50x | 157.5 | 7.1 |
| `JPEG 2000 (5/3)` | 3.2187 | 9.6561 | 2.49x | 75.1 | 66.0 |
| `JPEG-LS` | 4.2744 | 12.8232 | 1.87x | 19.5 | 15.1 |
| `PNG (level 9)` | 4.3926 | 13.1778 | 1.82x | 106.2 | 5.3 |

### 2. Held-Out Classic Kodak Suite (All 24 Photographs, `768×512` / `512×768` RGB24)

On classic `4:2:0`-origin PhotoCD scans, `vpxc (-e 1)` encodes in **`89–104 ms`** while beating `PNG` by **34%**, `WebP-L` and `JPEG 2000` by **9.3%**, and `JPEG XL (-e 5)` by **4.9%**; at `-e 3`, `vpxc` surpasses `JPEG XL (-e 9)` in both compression ratio and encode speed:

| Codec | Rate (`bpsp`) | Rate (`bpp`) | Ratio | Encode (`ms/img`) | Decode (`ms/img`) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **`vpxc (-e 3, tile=768)`** | **2.8227** | **8.4681** | **2.83x** | **1,139.0** | **225.0** |
| `JPEG XL (-d 0 -e 9)` | 2.8284 | 8.4851 | 2.83x | 1,318.5 | 94.5 |
| **`vpxc (-e 3, tile=384)`** | **2.8515** | **8.5545** | **2.81x** | **805.3** | **90.5** |
| `JPEG XL (-d 0 -e 7)` | 2.8630 | 8.5890 | 2.79x | 435.7 | 91.3 |
| **`vpxc (-e 2, tile=384)`** | **2.8794** | **8.6382** | **2.78x** | **322.3** | **98.2** |
| **`vpxc (-e 1, default 384)`** | **2.9046** | **8.7138** | **2.75x** | **104.4** | **94.1** |
| **`vpxc (-e 1, tile=256)`** | **2.9173** | **8.7519** | **2.74x** | **89.6** | **84.4** |
| `JPEG XL (-d 0 -e 5)` | 3.0534 | 9.1602 | 2.62x | 202.6 | 76.0 |
| **`vpxc (-e 0, fast)`** | **11,532,768 B total** | **3.2588** | **9.7764** | **2.45x** | **6.6** | **7.3** |
| `JPEG XL (-d 0 -e 3)` | 3.1461 | 9.4383 | 2.54x | 46.0 | 32.7 |
| `JPEG 2000 (5/3)` | 3.1921 | 9.5762 | 2.51x | 74.2 | 65.4 |
| `WebP-L (lossless)` | 3.2033 | 9.6099 | 2.50x | 168.2 | 6.9 |
| `JPEG-LS` | 4.3577 | 13.0731 | 1.84x | 19.4 | 15.0 |
| `PNG (level 9)` | 4.4047 | 13.2142 | 1.82x | 90.7 | 5.2 |

### 3. Held-Out Clinical Medical Imaging (`16-Bit Signed HU CT`, `3D Brain MRI` & `DICOM`)

`vpxc` includes a dedicated 3D volumetric clinical container (**`VPXV`**, `cpp/src/medical.cpp`) and zero-dependency **DICOM Part 10 byte-exact archive wrapper** (**`VPXD`**, `src/vpxc/medical.py`) tailored for PACS, teleradiology, and cloud health-imaging storage:

- **Native Signed `int16` Hounsfield Units (`-1024 .. +3071+` HU), `uint16`, & `uint8`:** Dynamic per-slice `[gmin, gmax, step_gcd]` range compaction + `O(1)` 5-byte air/padding slice bypass + adaptive `qsh` soft-tissue bilateral scaling.
- **3D Causal Inter-Slice Prediction with `O(1)` Random-Access Seeking:** Slices are partitioned into independent **Groups of Slices (`GoS`, default `8`)** with an explicit byte-offset directory and per-slice `CRC-32`. Archival modes (`-e 1, 2`) predict $\Delta_z S = S_z - S_{z-1}$ conditioned on $S_{z-1}$ spatial gradients ($\partial_x, \partial_y, \nabla^2$ compensating gantry tilt / respiratory drift) and causal $Z$-velocity ($S_{z-1} - S_{z-2}$), while `-e 0` uses 3D anatomy-switched Delta-MED + 4-way LUT `rANS` for **`0.5–3.2 ms` single-slice random-access PACS viewing**.
- **Cryptographic Byte-Exact `.dcm` Restoration (`VPXD`):** Compresses uncompressed Explicit/Implicit VR Little Endian DICOM pixel payloads via `vpxc` and non-pixel tags via `zlib`, verifying the restored `.dcm` file against an embedded **32-byte `SHA-256` digest**.

#### Aggregate Held-Out 16-Bit Clinical CT Benchmark (`57.09 MB` Raw, `672` Axial Slices Across 3 Clinical Scans: Abdominal CT, Head/Neck CT, High-DR Neurosurgical Electrode CT)

*(External 2D codecs are steelmanned with optimal unsigned bias shifting `-vol.min()` where `int16` is unsupported, and verified 100% bit-exact.)*

| Codec | Total Size | Bits/Voxel (`bpv`) | Ratio vs. Raw | Total Encode (`ms`) | Total Decode (`ms`) | Random-Access Slice Seek |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| **`vpxc 3D (-e 2, GoS=8)`** | **11,628,495 B** | **3.2588** | **4.91x** | **4,917** | **1,132** | **`~2.6–7.0 ms`** |
| **`vpxc 3D (-e 1, GoS=8)`** | **11,643,004 B** | **3.2629** | **4.90x** | **1,480** | **1,101** | **`~2.6–7.0 ms`** |
| `JPEG XL (-d 0 -e 9) (2D)` | 12,244,395 B | 3.4314 | 4.66x | 37,088 | 1,769 | Yes |
| `JPEG XL (-d 0 -e 7) (2D)` | 12,491,934 B | 3.5008 | 4.57x | 9,371 | 1,731 | Yes |
| **`vpxc 3D (-e 0, fast, GoS=8)`** | **12,627,015 B** | **3.5386** | **4.52x** | **201** | **178** | **`0.5–2.5 ms`** |
| **`vpxc 2D (-e 1)`** | **13,015,146 B** | **3.6474** | **4.39x** | **4,560** | **4,481** | Yes |
| `JPEG XL (-d 0 -e 3) (2D)` | 13,576,792 B | 3.8048 | 4.21x | 1,379 | 939 | Yes |
| **`vpxc 2D (-e 0, fast)`** | **13,933,562 B** | **3.9048** | **4.10x** | **314** | **381** | **`< 0.6 ms`** |
| `JPEG-LS (CharLS 2D)` | 13,943,945 B | 3.9077 | 4.09x | 378 | 356 | Yes |
| `JPEG 2000 (5/3, OpenJPEG 2D)` | 15,474,346 B | 4.3366 | 3.69x | 2,638 | 2,406 | Yes |
| `PNG (level 9, 2D)` | 19,568,255 B | 5.4839 | 2.92x | 2,632 | 256 | Yes |
| `NIfTI (.nii.gz, gzip)` | 20,633,262 B | 5.7824 | 2.77x | — | — | No (full stream) |

- **3D Structural T1 Brain MRI (`190 × 256 × 188`, `9.14 MB` raw):** `vpxc 3D (-e 2, GoS=8)` achieves **`2.5637 bpv` (`3.12x`)** and `-e 1` achieves **`2.5764 bpv` (`3.11x` in `543 ms`)**, beating `JPEG XL (-e 9)` (`2.7023 bpv` in `12,287 ms`), `JPEG-LS` (`2.9137 bpv`), `JPEG 2000 (5/3)` (`3.2054 bpv`), and `.nii.gz` (`4.18 MB` / `2.19x`).
- **UI, Diagrams & Screen Content:** Sub-millisecond 64-bit hash palette detection automatically switches sparse-color tiles ($\le 512$ unique colors) to zero-rounding MED-rANS palette coding (`0.0043 bpsp` / `1,872x` on `512×512` UI diagrams vs. `PNG` `442x`).

---

## Installation

### 1. Build the C++20 Native Engine (Recommended)

Building the C++20 shared library (`cpp/build/libvpxc_native.so`) enables the full-speed `frontier` multi-threaded encoder and integer decoder:

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
```

### 2. Install the Python Package & CLI

```bash
# Core NumPy API + CLI:
pip install -e .

# With standard image file I/O (PNG, TIFF, WebP, BMP, JPEG):
pip install -e ".[images]"

# With reference codecs (JPEG XL, WebP-L, JPEG 2000, JPEG-LS, PNG, QOI) for benchmarking:
pip install -e ".[bench]"
```

When `cpp/build/libvpxc_native.so` is present, `vpxc` automatically dispatches `encode` and `decode` to C++20 (`VPXC_PYTHON=1` forces the pure-Python reference path).

---

## Quickstart

### Command Line

```bash
# Real-time ultra-fast encode & decode (< 8 ms on 768x512 RGB24, effort 0):
vpxc encode photo.png photo_fast.vpxc --effort 0

# Fast lossless encode (~100 ms default, effort 1):
vpxc encode photo.png photo.vpxc

# Maximum archival compression (effort 3):
vpxc encode photo.png photo.vpxc --effort 3 --tile 768

# Losslessly pack & restore a byte-exact clinical DICOM (.dcm <-> .vpxd, SHA-256 verified):
vpxc pack-dicom scan.dcm scan.vpxd --effort 1 --gos 8
vpxc unpack-dicom scan.vpxd restored_scan.dcm

# Decode back to bit-exact pixels:
vpxc decode photo.vpxc restored.png

# Verify bit-identity and print compression ratio vs. PNG:
vpxc roundtrip photo.png

# Head-to-head benchmark against JPEG XL, WebP-L, JPEG 2000, JPEG-LS, PNG, and QOI:
vpxc benchmark photo.png --markdown
```

### Python API

```python
import numpy as np
from vpxc import (
    decode,
    encode,
    encode_volume,
    decode_volume,
    decode_slice,
    pack_dicom,
    unpack_dicom,
    benchmark_image,
    format_benchmark_table,
)

# 1. 2D Photographic / Grayscale Images (effort=0 for <10ms decode, effort=1..3 for frontier)
image = np.random.default_rng(0).integers(0, 256, size=(512, 768, 3), dtype=np.uint8)
blob = encode(image, effort=1)
assert np.array_equal(decode(blob), image)

# 2. 3D Volumetric 16-bit Signed Hounsfield Unit CT / MRI with O(1) Slice Seeking
ct_hu = np.random.default_rng(1).integers(-1024, 2000, size=(128, 256, 256), dtype=np.int16)
vol_blob = encode_volume(ct_hu, effort=0, gos_size=8)  # Real-time PACS fast mode
slice_73 = decode_slice(vol_blob, 73)                  # Millisecond random-access slice read
assert np.array_equal(slice_73, ct_hu[73])
assert np.array_equal(decode_volume(vol_blob), ct_hu)
```

### Effort Presets & Options

| Option | Values | Default | When to Use |
| :--- | :--- | :---: | :--- |
| `--effort 0` | Real-Time Fast | — | **`~6.6 ms` enc / `7.3 ms` dec** (`768×512` RGB24) or **`> 320 MB/s`** 3D CT decode. 4-way lockstep 12-bit flat-LUT `rANS` + MED / 3D Delta-MED. |
| `--effort 1` | Fast Archival | **Default** | **`~90–113 ms/img`**. Single-pass cross-validated prior + causal integer NLMS + pre-pruned Universal `MA` tree. Beats `JPEG XL -e 9` on `4:4:4` photos and 3D CT/MRI. |
| `--effort 2` | Balanced | — | **`~320 ms/img`**. Adds FPE-regularized subsampled MAP Ridge adaptation and `O(1)` histogram `MA` context-tree splitting. |
| `--effort 3` | Max | — | **`~805–1,190 ms/img`**. Full-tile MAP Ridge + exhaustive reversible color transform and $2\times 2$ chroma coset search. |
| `--tile` | `8 .. 4096` | `384` | Tile size in pixels. Controls multi-threaded tile parallelism and caps per-thread L1/L2 working memory (`768` maximizes single-tile context at `-e 3`). |

---

## How `vpxc` Works

Each tile in a `.vpxc` v3 container automatically selects the smallest representation among **`frontier`** (channel-causal photographic coder), **`palette`** ($\le 512$-color UI coder), and **`raw`** (incompressible bypass):

1. **Reversible Color Decorrelation (`cpp/src/frontier.cpp`, `cpp/src/color.cpp`):** Screens exact integer reversible color transforms (`Subtract-Green`, `YCoCg-R`, and luma-averaged permutations) in `0.2 ms` without rounding loss over $\mathbb{Z}^C$.
2. **Prior-Warmed Multi-Expert + Causal NLMS Prediction:** Combines an 8-expert Weighted Predictor (`WP8` with separate Luma/Chroma priors), a single-probe LRU-2 non-local `PatchPredictor`, cross-channel structural luma/residual guidance, and 36-tap directional sector predictors initialized from 8-fold cross-validated `W_prior` weights and adapted per pixel via division-free integer `NLMS`. Smooth/$4:2:0$-upsampled chroma channels automatically switch to a 4-stage $2\times 2$ bilateral coset interpolator.
3. **Anti-Overfitting `MA` Context Trees:** Routes residuals through either pre-pruned split-half validated Universal `MA` context trees (`-e 1`) or `O(1)` 32-bin histogram MDL trees (`-e 2, 3`) conditioned on local gradients, predictor spread, and cross-channel activity.
4. **Division-Free Student-$t$ `Core66 rANS` (`cpp/src/entropy.cpp`):** Encodes sub-pixel fractional Student-$t$ distributions and adaptive 66-symbol CDFs using a 65,537-entry 64-bit reciprocal table (zero hardware `divq` instructions in the inner loop) and decodes with **zero floating-point instructions**, verified by a whole-image CRC-32 checksum.

---

## Running Tests & Fuzzing

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release && cmake --build cpp/build -j
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The test suite verifies round-trip bit-identity across 2D (`uint8`/`uint16` grayscale/RGB/RGBA), 3D volumetric (`uint8`/`uint16`/signed `int16` Hounsfield Units + `O(1)` slice random access), and byte-exact `.dcm` DICOM (`SHA-256` verified), tests zip-bomb and corrupt-stream rejection, cross-compiles `simd.hpp` for `ARM64 NEON`, and disassembles `frontier.cpp.o`, `fast_codec.cpp.o`, and `medical.cpp.o` to enforce **zero floating-point instructions** across all decode paths. A `libFuzzer + AddressSanitizer + UndefinedBehaviorSanitizer` harness is provided in [`cpp/fuzz/fuzz_decode.cpp`](./cpp/fuzz/fuzz_decode.cpp) (`-DVPXC_BUILD_FUZZER=ON`).

---

## Technical Documentation & Research Notes

- [`training/`](./training/) — Leak-checked corpus builder (`prepare_corpora.py`) and 8-fold cross-validated offline prior trainer (`train_frontier.cpp`).
- [`docs/vpxc-slgf-specification.pdf`](./docs/vpxc-slgf-specification.pdf) — Mathematical specification, bijectivity proofs, and normative integer equations (`LaTeX` source in [`paper/`](./paper/)).
- [`docs/frontier-lossless-image-compression.md`](./docs/frontier-lossless-image-compression.md) — Architecture monograph and experimental roadmap.
- [`docs/slgf-improvement-plan.md`](./docs/slgf-improvement-plan.md) — Subband telemetry and engineering notes.
