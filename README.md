# `vpxc` — Lossless Image Compression with Symplectic Lattice-Gauge Flows

`vpxc` is an exact, deterministic **lossless image codec** for 8-bit and 16-bit images, implemented in pure Python and NumPy with zero pre-trained neural network weights and zero GPU dependencies.

Every compressed `.vpxc` stream decodes using **100% integer arithmetic** (zero IEEE-754 floating-point operations in the decoder) and verifies reconstruction integrity with a whole-image CRC-32 checksum. Across the full 24-image Kodak lossless photography suite, `vpxc` compresses smaller than **QOI**, **optimized PNG**, **JPEG-LS**, **WebP Lossless (`-z 9`)**, and **JPEG 2000 (reversible 5/3)**.

---

## Features

- **Bit-exact lossless round-trip** for `uint8` and `uint16` arrays in grayscale (`HxW`), RGB (`HxWx3`), and RGBA (`HxWx4`).
- **Cross-platform integer determinism:** decoder conditioning (`Q15.16`/`Q1.30` balanced-tree state-space scans, integer PCHIP Student-$t$ lookup tables, quincunx lifting, and `SL(3)` channel transforms) uses zero floating-point instructions, avoiding `rANS` drift across x86, ARM, and GPU targets.
- **Zero external model weights:** adapts to each image analytically in milliseconds via closed-form Cholesky Minimum Description Length (`MDL`) regression rather than per-image gradient descent.
- **Multi-domain adaptivity:** automatically selects among multiscale `SLGF` coding, zero-rounding palette mode ($\le 512$ colors) for UI/diagrams, $D_8$-equivariant Sheaf-Kalman prediction, and uncompressed raw tile bypass on incompressible noise.
- **Built-in multi-codec benchmark CLI:** compare `vpxc` directly against `JPEG XL`, `JPEG 2000`, `WebP-L`, `JPEG-LS`, `PNG`, and `QOI` on your own images with automatic bit-exact verification.

---

## Installation

Core array compression requires only Python $\ge 3.9$ and `numpy`:

```bash
pip install -e .
```

To load/save standard image files (`PNG`, `TIFF`, `BMP`, `JPEG`, `WebP`) from the CLI, install with `images`. To enable in-process head-to-head benchmarking against `JPEG XL`, `WebP-L`, `JPEG 2000`, `JPEG-LS`, `PNG`, and `QOI`, install with `bench`:

```bash
pip install -e ".[images]"
# or, with reference codecs for benchmarking:
pip install -e ".[bench]"
```

---

## Quickstart

### Command Line

```bash
# Compress an image to a .vpxc file
vpxc encode photo.png photo.vpxc

# Restore the exact original pixels
vpxc decode photo.vpxc restored.png

# Encode, decode, verify bit-identity, and compare against optimized PNG
vpxc roundtrip photo.png

# Benchmark vpxc against PNG, QOI, JPEG-LS, WebP-L, JPEG 2000, and JPEG XL
vpxc benchmark photo.png
vpxc benchmark ./my_images/ --markdown
```

### Python API

```python
import numpy as np
from vpxc import decode, encode, benchmark_image, format_benchmark_table

image = np.random.default_rng(0).integers(0, 256, size=(512, 768, 3), dtype=np.uint8)

blob = encode(image)          # bytes (.vpxc container)
restored = decode(blob)       # exact uint8 / uint16 ndarray
assert np.array_equal(restored, image)

# Compare against installed lossless codecs on any uint8 or uint16 array:
results = benchmark_image(image)
print(format_benchmark_table(results))
```

---

## Lossless Codec Benchmarks

All benchmarks below measure **actual serialized container sizes** verified for bit-exact lossless reconstruction (`np.array_equal(decoded, original)`). Rates are reported in **`bpsp`** (bits per sub-pixel / channel sample; raw 8-bit RGB = `8.00 bpsp`) and **`bpp`** (bits per pixel; raw 8-bit RGB = `24.00 bpp`). All codecs run in-process on the same CPU (`vpxc` C++20 native library `cpp/build/libvpxc_native.so`, and `libjxl`, `libwebp`, `OpenJPEG`, `CharLS`, `libpng` via `imagecodecs`).

Universal `frontier` priors (`cpp/include/vpxc/frontier_priors.hpp`) were trained offline (`training/prepare_corpora.py`, `training/train_frontier.cpp`) with 8-fold image-grouped cross-validation and James-Stein $t$-statistic stability shrinkage on an independent CLIC + synthetic training split, verified by SHA-256 and $16\times 16$ perceptual distance to have **zero overlap** with either held-out test set below.

### 1. Blind Held-Out Modern `4:4:4` Photography (`TEST_CLIC`, 24 Disjoint Photographs, `768×512` RGB24)

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

### 2. Blind Held-Out Kodak Lossless True Color Suite (All 24 Photographs, `768×512` / `512×768` RGB24)

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
| `JPEG XL (-d 0 -e 3)` | 3.1461 | 9.4383 | 2.54x | 46.0 | 32.7 |
| `JPEG 2000 (5/3)` | 3.1921 | 9.5762 | 2.51x | 74.2 | 65.4 |
| `WebP-L (lossless)` | 3.2033 | 9.6099 | 2.50x | 168.2 | 6.9 |
| `JPEG-LS` | 4.3577 | 13.0731 | 1.84x | 19.4 | 15.0 |
| `PNG (level 9)` | 4.4047 | 13.2142 | 1.82x | 90.7 | 5.2 |

### 3. 16-Bit High-Dynamic-Range / Medical CT Slice (`512×512 uint16`, Raw = `16.00 bpsp`)

| Codec | Size (bytes) | Rate (`bpsp`) | Compression Ratio |
| :--- | ---: | ---: | ---: |
| `JPEG XL (-d 0 -e 9)` | 227,002 | 6.9276 | 2.31x |
| `JPEG-LS` | 233,611 | 7.1292 | 2.24x |
| **`vpxc` (`SLGF`)** | **238,280** | **7.2717** | **2.20x** |
| `JPEG 2000 (5/3)` | 239,708 | 7.3153 | 2.19x |
| `PNG (opt)` | 325,966 | 9.9477 | 1.61x |

### 4. UI / Diagram Screen Content (`512×512` RGB24, Sparse Palette)

| Codec | Size (bytes) | Rate (`bpsp`) | Rate (`bpp`) | Compression Ratio |
| :--- | ---: | ---: | ---: | ---: |
| `WebP-L (-z 9)` | 244 | 0.0025 | 0.0074 | 3,223x |
| `JPEG XL (-d 0 -e 9)` | 343 | 0.0035 | 0.0105 | 2,293x |
| **`vpxc` (`SLGF`)** | **420** | **0.0043** | **0.0128** | **1,872x** |
| `PNG (opt)` | 1,781 | 0.0181 | 0.0544 | 442x |
| `JPEG-LS` | 6,800 | 0.0692 | 0.2075 | 116x |
| `QOI` | 11,257 | 0.1145 | 0.3435 | 70x |
| `JPEG 2000 (5/3)` | 17,485 | 0.1779 | 0.5336 | 45x |

---

## CLI Options & Coding Profiles

| Option | Values | Default | Description |
| :--- | :--- | :---: | :--- |
| `--tile` | `8 .. 4096` | `384` | Independent tile dimension in pixels (enables multi-threaded tile parallelism and caps RAM) |
| `--effort` | `1`, `2`, `3` | `1` | `1` = fast single-pass `frontier` prior + NLMS + Universal `MA` tree (`~100 ms`); `2` = subsampled MAP Ridge + `O(1)` histogram `MA` tree; `3` = full-tile MAP + exhaustive RCT/coset search |
| `--profile` | `frontier`, `wp`, `slgf`, `slgf-zero`, `kalman`, `palette`, `qx-rans`, `qx-rice`, `raster`, `raw` | Auto | Force a tile coder (`frontier` is the default native C++20 photography coder) |

---

## How `vpxc` Works

`vpxc` implements the zero-weight integer architecture of **Symplectic Lattice-Gauge Flows (`SLGF`)**:

1. **Stage 0 — Reversible Channel Lifting (`src/vpxc/color.py`, `src/vpxc/wsil.py`):** Maps multi-channel pixels bijectively over $\mathbb{Z}^C$ via `YCoCg-R`, base-channel subtraction, 2-D Laplacian `Q12` unitriangular shears, or 10-multiplier Whitehead–Symplectic Integer Lifting (`SL(3)-WSIL`), with optional 4th-channel coupling for RGBA / Bayer data. Sparse-color tiles ($\le 512$ colors) trigger zero-rounding palette coding.
2. **Stage 1 — `Q^2 = 2I` Shallow Quincunx Tower (`src/vpxc/lattice.py`):** Applies strictly one anti-aliased 4-neighbor integer predict/update step per scale across 4 quincunx levels ($Q = \begin{psmallmatrix}1 & 1\\1 & -1\end{psmallmatrix}$), avoiding Rounding-Induced Total Correlation (`RITC`) from deep stacked integer rounding while packing even levels onto 100%-dense Cartesian subgrids.
3. **Stage 2 — Causal Sub-Cosets, `I-SSM`, & Closed-Form MDL Head (`src/vpxc/slgf_model.py`, `src/vpxc/issm.py`, `src/vpxc/spatial.py`):** Splits each detail scale into 3 causal spatial sub-cosets $r_s^{(1,2,3)}$ conditioned on directional cellular-sheaf conductances, 4-way `Q15.16`/`Q1.30` balanced-tree integer state-space scans (`I-SSM`), `D_8`-equivariant Sheaf-Kalman coarse fusion, cross-channel / mod-4 chroma-phase taps ($\phi_{\mathrm{acq}}$), and closed-form `Q8` Cholesky Ridge weights ($\phi_{\mathrm{head}}$).
4. **Stage 3 — Normative Student-$t$ `Dyadic-Core66 rANS` (`src/vpxc/entropy.py`):** Evaluates sub-pixel fractional Student-$t$ distributions ($\nu \in \{2.5, 3.5, 6.0, \infty\}$) and bilateral jump bridges via a hardcoded $2\;\mathrm{KB}$ integer PCHIP lookup table into 66-symbol 16-bit `rANS` streams with tilted sub-bucket remainder trees (plus standalone `S^3-BB-rANS` zero-waste bits-back support).

> **Research specification vs. current package:** Container version 3 (see [`docs/slgf-improvement-plan.md`](./docs/slgf-improvement-plan.md)) adds the coarse-band Core66 predictor bank, MDL MANIAC trees, the pre-gamma `phi_acq` header, causal multiscale `D_8` Sheaf-Kalman matches, an optional packed INT8 stem (`src/vpxc/weights/slgf_v1.bin`, `--effort 3`), and tile-level `S^3-BB-rANS`. A version-3 file still decodes version-2 payloads, and each tile keeps the version-2 coding when that payload is smaller. The Kodak table above is the measured version-2 suite (`3.1861 bpsp`). The `2.40 ± 0.04 bpsp` flagship figure remains the specification target; this pure-Python build does not claim that rate.

---

## C++ codec

The integer codec also lives in `cpp/`. It encodes the same container as the Python reference and decodes either side's bytes. Build it with:

```bash
cmake -S cpp -B cpp/build -DCMAKE_CXX_COMPILER=g++
cmake --build cpp/build
```

When `cpp/build/libvpxc_native.so` is present, `encode` and `decode` call it. `VPXC_PYTHON=1` keeps the pure-Python path.

## Running Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

To include real-photograph round-trip tests, point `VPXC_IMAGE_DIR` at a folder of PNG or JPEG images:

```bash
VPXC_IMAGE_DIR=/path/to/images PYTHONPATH=src python3 -m unittest discover -s tests -v
```

---

## Specification & Research Papers

- [`docs/slgf-improvement-plan.md`](./docs/slgf-improvement-plan.md) — Engineering roadmap and measured subband telemetry for closing the `3.19 → 2.40 bpsp` gap past `JPEG XL (-e 9)` and `HPAC-FT`.
- [`docs/vpxc-slgf-specification.pdf`](./docs/vpxc-slgf-specification.pdf) — Typeset mathematical specification, bijectivity & `RITC` proofs, and normative integer equations (`LaTeX` source in [`paper/`](./paper/)).
- [`docs/frontier-lossless-image-compression.md`](./docs/frontier-lossless-image-compression.md) — Primary `SLGF` architecture monograph and 5-gate experimental roadmap.
- [`docs/slgf-derivations-and-numerical-checks.md`](./docs/slgf-derivations-and-numerical-checks.md) — Derivations and numerical rate audits (Part I).
- [`docs/vpxc-extended-ritc-wsil-rans-audits.md`](./docs/vpxc-extended-ritc-wsil-rans-audits.md) — Extended `RITC`, `SL(3)-WSIL`, `Dyadic-Core66 rANS`, and MDL Ridge numerical checks (Part II).
