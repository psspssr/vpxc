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

All benchmarks below measure **actual serialized container sizes** verified for bit-exact lossless reconstruction (`np.array_equal(decoded, original)`). Rates are reported in both **`bpsp`** (bits per sub-pixel / channel sample; raw 8-bit RGB = `8.00 bpsp`) and **`bpp`** (bits per pixel; raw 8-bit RGB = `24.00 bpp`). Reference codecs are C/C++ implementations invoked via `imagecodecs` (`libjxl`, `OpenJPEG`, `libwebp`, `CharLS`, `libpng`, `qoi`), while `vpxc` runs in pure Python + NumPy on CPU.

### 1. Kodak Lossless True Color Suite (All 24 Photographs, `768×512` / `512×768` RGB24)

| Codec | Total Size (bytes) | Rate (`bpsp`) | Rate (`bpp`) | Compression Ratio | Implementation |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `JPEG XL (-d 0 -e 9)` | 10,009,376 | 2.8284 | 8.4851 | 2.83x | C++ (`libjxl`, effort 9) |
| **`vpxc` (`SLGF`)** | **11,275,519** | **3.1861** | **9.5584** | **2.51x** | **Pure Python + NumPy (zero weights)** |
| `JPEG 2000 (5/3)` | 11,296,508 | 3.1921 | 9.5762 | 2.51x | C (`OpenJPEG` reversible DWT) |
| `WebP-L (-z 9)` | 11,425,502 | 3.2285 | 9.6855 | 2.48x | C (`libwebp` lossless level 9) |
| `JPEG-LS` | 15,421,702 | 4.3577 | 13.0731 | 1.84x | C++ (`CharLS`) |
| `PNG (opt)` | 15,588,128 | 4.4047 | 13.2142 | 1.82x | C (`libpng` level 9) |
| `QOI` | 16,501,520 | 4.6628 | 13.9885 | 1.72x | C (`qoi`) |

Sample per-image rates (`bpsp`) across representative Kodak photographs:

| Image | `JPEG XL (-e 9)` | **`vpxc` (`SLGF`)** | `JPEG 2000 (5/3)` | `WebP-L (-z 9)` | `PNG (opt)` | `QOI` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `kodim01` (stone wall) | 3.0840 | **3.5420** | 3.4618 | 3.4442 | 4.9099 | 5.0601 |
| `kodim05` (motocross) | 3.1642 | **3.6537** | 3.5750 | 3.5481 | 5.1481 | 5.4795 |
| `kodim07` (shuttered window) | 2.4550 | **2.8225** | 2.8669 | 2.8661 | 3.6881 | 4.2121 |
| `kodim13` (mountain stream) | 3.5786 | **3.9311** | 3.9682 | 4.1328 | 5.6672 | 5.7924 |
| `kodim15` (portrait) | 2.6656 | **2.9967** | 2.9808 | 3.1059 | 4.0085 | 4.3649 |
| `kodim23` (macaws) | 2.5245 | **2.8471** | 2.8352 | 2.9018 | 3.8985 | 4.5793 |

### 2. 16-Bit High-Dynamic-Range / Medical CT Slice (`512×512 uint16`, Raw = `16.00 bpsp`)

| Codec | Size (bytes) | Rate (`bpsp`) | Compression Ratio |
| :--- | ---: | ---: | ---: |
| `JPEG XL (-d 0 -e 9)` | 227,002 | 6.9276 | 2.31x |
| `JPEG-LS` | 233,611 | 7.1292 | 2.24x |
| **`vpxc` (`SLGF`)** | **238,280** | **7.2717** | **2.20x** |
| `JPEG 2000 (5/3)` | 239,708 | 7.3153 | 2.19x |
| `PNG (opt)` | 325,966 | 9.9477 | 1.61x |

### 3. UI / Diagram Screen Content (`512×512` RGB24, Sparse Palette)

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
| `--tile` | `8 .. 4096` | `768` | Independent tile dimension in pixels (caps peak working RAM) |
| `--effort` | `1`, `2`, `3` | `2` | `1` = fast selection; `2` = MANIAC / acquisition / D8 search; `3` = SLGF-Max (`S^3-BB-rANS` and the INT8 stem) |
| `--profile` | `slgf`, `slgf-zero`, `kalman`, `palette`, `qx-rans`, `qx-rice`, `raster`, `raw` | Auto | Force a tile coder. `slgf-zero` is the weight-free profile |

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
