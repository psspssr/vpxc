# `vpxc` Technical Architecture

`vpxc` is a standalone C++20 lossless compression engine (`libvpxc` static library + `libvpxc_native` shared C-ABI library) paired with zero-copy Python `ctypes` bindings (`src/vpxc/native.py`), a DICOM Part 10 byte-exact container (`src/vpxc/medical.py`), and an offline cross-validated prior trainer (`training/`).

---

## 1. Core Systems Invariants

1. **Zero-Float Cross-ISA Determinism:**
   In arithmetic / `rANS` entropy coding, a single 1-ULP IEEE-754 discrepancy (`FMA` contraction, x87/SSE rounding, or `libm` differences between `x86_64` and `ARM64`) shifts cumulative frequency thresholds and desynchronizes the entire remaining bitstream.
   - All decode translation units (`frontier.cpp`, `fast_codec.cpp`, `medical.cpp`, `entropy.cpp`) execute strictly in fixed-width `int16_t` / `int32_t` / `int64_t` / `uint64_t` integer arithmetic.
   - `tests/test_native.py` disassembles compiled object files via `objdump -d` in CI (`x86_64` and `ARM64`) and asserts **zero floating-point instructions** across all decode functions.
   - Encoders may use `double` solely to solve offline/per-tile `MAP Ridge` normal equations or score `MA` context-tree splits; the chosen quantized integer weights (`Q8`) and split thresholds are serialized into the tile header so every decoder executes identical integer operations.

2. **Hardware-Aware Throughput (`AVX2`, `ARM64 NEON`, & ILP Interleaving):**
   - `cpp/include/vpxc/simd.hpp` provides unified `x86_64 AVX2`, `ARM64 NEON`, and scalar fallbacks for 36-tap `Q8` inner products (`dot36_i32_i16`) and vectorized inverse reversible color transforms (`rct_sub_green_inv_row`, `rct_sub_avg_inv_row`).
   - `cpp/src/fast_codec.cpp` decouples serial `rANS` table-lookup latency by stepping **4 independent `rANS` states in lockstep** over a `16 KB` L1-resident 12-bit (`M = 4096`) lookup table (`sym | (freq << 8) | (bias << 20)` packed into a single `uint32_t` load per symbol) paired with 4-stripe row-decoupled `MED` / 3D `Delta-MED` prediction.
   - `cpp/src/entropy.cpp` replaces per-symbol 64-bit integer divisions in `Core66 rANS` encoding with precomputed `__uint128_t` reciprocal multiplications (`(x * rcp) >> 64` + single monotonic correction step).

3. **Untrusted-Input Security & Bounded Memory:**
   - Both 2D (`VPXC`) and 3D (`VPXV`) decoders validate header dimensions (`kMaxImageDim = 32768`, `kMaxImageSamples = 256M`, `kMaxVolumeSamples = 512M`) and enforce worst-case payload amplification bounds *before* allocating output buffers.
   - Verified with per-slice and whole-container `CRC-32` checksums plus continuous `libFuzzer + AddressSanitizer + UndefinedBehaviorSanitizer` fuzzing (`cpp/fuzz/fuzz_decode.cpp`).

---

## 2. Probabilistic Modeling & Entropy Engines

### 2.1 `frontier` High-Ratio Causal Autoregressive Coder (`-e 1 .. -e 3`)
Implemented in `cpp/src/frontier.cpp` for 8/16-bit 2D images and 3D volumetric slices:
- **Reversible Color Transforms (`RCT`):** Evaluates `Identity`, `Subtract-Green`, `Subtract-Average` (`R' = R - G`, `B' = B - ((R + G) >> 1)`), `YCoCg-R`, and optional $2\times 2$ chroma-coset phase conditioning.
- **Multi-Expert Causal Prediction + Online Test-Time Adaptation:**
  - **8-Expert Weighted Predictor (`WP8`):** Tracks exponentially weighted causal L1 errors of 8 directional/planar predictors per pixel and blends them with inverse-error integer weights.
  - **36-Tap `Q8` Directional Ridge + Causal `NLMS`:** Combines causal neighborhood gradients, cross-channel reconstructed luma derivatives ($\partial_x, \partial_y, \nabla^2$), and division-free shift-normalized `NLMS` online weight updates (fast test-time adaptation without per-image two-pass overhead at `-e 1`).
  - **LRU-2 `PatchPredictor`:** Exact hash-indexed causal template matching for lossless screen/diagram/repeating-structure bypass.
- **Anti-Overfitting `MA` Context Trees + `Dyadic-Core66 rANS`:**
  - Partitions causal gradient/error properties using hybrid linear-logarithmic histogram binning (`rm_bin_from_abs`, supporting full 16-bit dynamic range) with strict Rissanen MDL split penalties (`120` bits/split) and minimum leaf sample counts (`512`).
  - Models heavy-tailed prediction residuals with a quantized Student-$t$ family over 65 centered quotient bins + Exp-Golomb escape (`M = 65536` `rANS`).

### 2.2 Leak-Checked Offline Prior Training (`training/`)
- `training/prepare_corpora.py` constructs disjoint training and held-out evaluation sets verified via both **exact `SHA-256`** and **$16\times 16$ L1 perceptual distance** to prevent benchmark contamination.
- `training/train_frontier.cpp` fits universal `Q8` linear weights, `WP8` blend priors, and canonical `MA` tree topologies using **8-fold image-grouped cross-validation** and **James-Stein $t$-statistic shrinkage** (`cpp/include/vpxc/frontier_priors.hpp`, `~5.5 KB`).

### 2.3 `VPXV` 3D Volumetric Medical Container & `VPXD` Byte-Exact DICOM Wrapper
Implemented in `cpp/src/medical.cpp` and `src/vpxc/medical.py`:
- **Signed `int16` Hounsfield Unit Compaction:** Bijectively maps signed `int16` (`v ^ 0x8000`) and computes per-slice `[gmin, gmax, step_gcd]` dynamic-range compaction, 5-byte constant air/padding slice bypass, and 1-byte exact duplicate-slice bypass.
- **Group-of-Slices (`GoS`) Random Access:** Partitions `(Z, H, W)` volumes into parallel `GoS` blocks (default `8` slices) indexed by a byte-offset directory with per-slice `CRC-32`. Intra-GoS P-slices predict $\Delta_z S = S_z - S_{z-1}$ from $S_{z-1}$ spatial derivatives ($\partial_x, \partial_y, \nabla^2$) and causal $Z$-velocity ($S_{z-1} - S_{z-2}$), enabling **`0.5–3.2 ms` $O(1)$ single-slice random access** without decompressing the full volume stream.
- **Cryptographic DICOM Parity (`VPXD`):** Disaggregates Explicit/Implicit VR Little Endian DICOM Part 10 metadata from pixel payloads, compresses pixels with `VPXV`/`VPXC` and non-pixel tags with `zlib`, and verifies restored `.dcm` files against an embedded **32-byte `SHA-256` digest**.

---

## 3. Repository Map

```text
vpxc/
├── cpp/
│   ├── CMakeLists.txt
│   ├── fuzz/fuzz_decode.cpp          # libFuzzer + ASan + UBSan harness
│   ├── include/vpxc/
│   │   ├── frontier.hpp              # 2D & 3D high-ratio causal coder (-e 1..3)
│   │   ├── frontier_priors.hpp       # Cross-validated offline Q8 / MA / WP priors
│   │   ├── fast_codec.hpp            # Sub-10ms 4-way 12-bit LUT rANS coder (-e 0)
│   │   ├── medical.hpp               # VPXV 3D CT/MRI container & O(1) slice seek
│   │   ├── simd.hpp                  # x86_64 AVX2, ARM64 NEON, and scalar kernels
│   │   ├── entropy.hpp / bitio.hpp   # Reciprocal Student-t Core66 rANS & bit I/O
│   │   └── codec.hpp / image.hpp     # 2D container framing & multi-threaded tiling
│   └── src/                          # C++20 implementations & C ABI (c_api.cpp)
├── src/vpxc/
│   ├── codec.py / native.py          # Python API & zero-copy ctypes bridge
│   ├── medical.py                    # 3D volume wrappers & SHA-256 byte-exact DICOM
│   ├── benchmark.py                  # Multi-codec 2D & 3D bit-exact benchmark suite
│   └── cli.py                        # `vpxc` CLI entrypoint
├── training/
│   ├── prepare_corpora.py            # SHA-256 + perceptual-hash leak-checked splits
│   └── train_frontier.cpp            # 8-fold grouped CV offline prior trainer
└── tests/                            # 49 unit, differential, SIMD, & security tests
```
