# `vpxc` Engineering Improvement Plan: Closing the `3.19 → 2.40 bpsp` Gap

This document specifies the concrete technical roadmap to advance `vpxc` from its current zero-weight pure-Python baseline (**`3.1861 bpsp`** across the 24-image Kodak lossless suite — already outperforming `JPEG 2000`, `WebP-L -z 9`, `JPEG-LS`, `PNG`, and `QOI`) past **`JPEG XL (-d 0 -e 9)` (`2.8284 bpsp`)** and **`HPAC-FT` (`2.52 bpsp`)** toward the full specification target (**`2.40 ± 0.04 bpsp`**) while preserving 100% cross-ISA integer decode determinism.

---

## 1. Empirical Telemetry & Exact Bottleneck Attribution

Across all 24 Kodak lossless RGB24 photographs (`768×512` / `512×768`), `JPEG XL` and current `vpxc` measure as follows:

| Reference / Subband | Measured Kodak Rate (`bpsp`) | Share of `vpxc` Bitstream | Diagnostic Interpretation |
| :--- | :---: | :---: | :--- |
| `JPEG XL (-d 0 -e 1)` | `3.6418` | — | Fixed predictors + static contexts (`vpxc` beats by `-0.456 bpsp`) |
| `JPEG XL (-d 0 -e 3)` | `3.1461` | — | Light context adaptation (parity with current pure-Python `vpxc`) |
| `JPEG XL (-d 0 -e 7)` | `2.8630` | — | Learned per-image `MA` context trees + weighted predictor blending |
| `JPEG XL (-d 0 -e 9)` | `2.8284` | — | Exhaustive `MA` tree search + 2D `Patches` dictionary |
| **`vpxc` (`SLGF` v0.2 total)** | **`3.1861`** | **`100.0%`** | **`Y = 4.3966`, `Co = 2.5928`, `Cg = 2.5684` bits/sample** |
| ├─ `coarse` (`y_4`, `6.25%` of px) | `0.2974` | `9.3%` | Scalar Rice (`encode_raster`), no `Y → Co,Cg` regression (`4.76 bits/sample`) |
| ├─ `lvl3` (`6.25%` of px) | `0.2508` | `7.9%` | Global linear Ridge + 16 1-D activity buckets |
| ├─ `lvl2` (`12.5%` of px) | `0.4302` | `13.5%` | Global linear Ridge + 16 1-D activity buckets |
| ├─ `lvl1` (`25.0%` of px) | `0.8469` | `26.6%` | Missing fine-scale `D_8` patch search & tree-split contexts |
| ├─ `lvl0` (`50.0%` of px) | `1.3555` | `42.5%` | Post-gamma `PhotoYCC` 4:2:0 nonlinearity + single linear predictor |
| └─ `meta` (headers & weights) | `0.0051` | `0.2%` | Compact MDL-gated `Q8` Ridge weights + delta-coded `sig_idx` |

### Summary of Remaining Gap (`0.79 bpsp` to Flagship Target; `0.36 bpsp` to `JPEG XL -e 9`)

| Workstream | Subsystems Changed | Invasiveness | Expected Net Gain (Kodak) | Cumulative Target (`bpsp`) |
| :--- | :--- | :--- | :---: | :---: |
| **WS1: Coarse-Band `rANS` + Native SIMD Kernel** | `slgf_model.py`, `spatial.py`, new `csrc/` (`_vpxc_accel`) | Medium (new optional C++/Rust extension; minor format bump) | `-0.04` to `-0.06 bpsp` (`35–50×` speedup) | `≤ 3.13` |
| **WS2: MDL `MANIAC`/`MA` Trees + Predictor Bank** | `slgf_model.py`, `entropy.py`, new `trees.py` | High (replaces flat 16-bucket table with tree contexts & leaf histograms) | `-0.16` to `-0.22 bpsp` | `≤ 2.93` |
| **WS3: Full Pre-Gamma `φ_acq` + Multiscale `D_8` Kalman** | `color.py`, `spatial.py`, `codec.py`, new `acquisition.py` | Medium-High (adds Stage-0 inverse-gamma LUT, 96-tap parity solve, `16 px` tile halo) | `-0.14` to `-0.20 bpsp` | **`≤ 2.72`** *(beats `JPEG XL -e 9`)* |
| **WS4: Trained `1.8 MB` `INT8` `I-SSM` + `S^3-BB-rANS`** | `issm.py`, `slgf_model.py`, `entropy.py`, new `training/` | High (offline PyTorch QAT pipeline + packed `INT8` inference graph) | `-0.26` to `-0.32 bpsp` | **`2.40 ± 0.04`** *(beats `HPAC-FT`)* |

---

## 2. Workstream 1: Coarse-Band `Core66 rANS` & Native SIMD Acceleration

### 2.1 Upgrade Base Band `y_4` from Rice to Cross-Channel Ridge + `Core66 rANS`
- **Problem:** `encode_slgf_tile` (`src/vpxc/slgf_model.py`) currently delegates `coarse` (`6.25%` of samples) to `encode_raster` (`src/vpxc/spatial.py`), which uses scalar adaptive Rice coding without `Y → Co, Cg` cross-channel prediction and pays `4.76 bits/sample` (`0.2974 bpsp` of total image rate).
- **Design:**
  1. Predict `coarse` causally using a 4-predictor local bank (MED, GAP gradient-adjusted predictor, West, North) blended with causal `D_8` Sheaf-Kalman matches (`M ≤ 8`) and, for `c > 0`, co-located causal Laplacian and MED residual of already-decoded `y_4^(0)` via `Q8` Cholesky Ridge weights.
  2. Replace `BitWriter.write_rice` on `coarse` with `encode_parametric_rans` (`Dyadic-Core66 rANS` with fractional `Q4` mean offsets and 16 causal gradient/error-energy buckets).
- **Files:** `src/vpxc/slgf_model.py`, `src/vpxc/spatial.py`.

### 2.2 Native C++17 / Rust SIMD Extension (`_vpxc_accel`) with Pure-Python Fallback
- **Problem:** Pure Python spends `~2.8 s` encoding and `~2.0 s` decoding a `0.4 MP` image inside scalar `rANS` renormalization loops, which also prevents running 2-D non-local patch search and `MANIAC` tree splits at full resolution.
- **Design:**
  1. Introduce an optional C++17 (`pybind11` / C API) or Rust (`PyO3`) extension module `vpxc._accel` while keeping every pure-Python function in `src/vpxc/*.py` as a reference fallback and differential-test oracle.
  2. Implement **32-way SIMD-interleaved `Dyadic-Core66 rANS`** (`AVX2` on `x86_64`, `NEON` on `ARM64`) with a branchless `65536`-entry `uint8` symbol alias/lookup table (`64 KB`, fits in L1/L2 cache) eliminating `bisect_right` from the decode inner loop.
  3. Port `issm.tree_rows` (`Q15.16`/`Q1.30` Blelloch scan), `lattice.predict`/`lift_forward`/`lift_inverse`, and feature assembly into fused tiled C++/Rust kernels operating in `int32`/`int64`.
- **Risks & Mitigations:** Any divergence between `_accel` and the pure-Python reference breaks lossless decoding. Enforce bit-for-bit equality (`encode_py(X) == encode_native(X)`) in CI fuzz tests before enabling native dispatch by default.

---

## 3. Workstream 2: Per-Image `MANIAC` / `MA` Context Trees & Multi-Predictor Blending

### 3.1 Directional Multi-Predictor Bank per Sub-Coset
- **Problem:** Fitting a single linear weight vector `w_q8` across an entire tile (`_estimate_subcoset_model` in `src/vpxc/slgf_model.py`) averages across horizontal edges, vertical edges, diagonal edges, and textured interiors.
- **Design:**
  1. Partition sites deterministically by causal orientation regime (`8` directional sectors from `(dv, dh, d_nw, d_ne)` on `even` neighbors) and fit separate `Q8` Ridge predictors per regime whenever MDL bit savings exceed the Exp-Golomb weight header cost (`ΔL_MDL < 0`).
  2. Include causal **same-scale error-energy EMA** and **signed neighbor residual products** so oscillatory textures (foliage, water, fabric in `kodim13` and `kodim05`) adapt predictor damping locally.
- **Files:** `src/vpxc/slgf_model.py`.

### 3.2 MDL-Pruned `MANIAC` Decision Trees with Clustered `Core66` Histograms
- **Problem:** `JPEG XL` gains `~0.32 bpsp` between `-e 3` (`3.15 bpsp`) and `-e 9` (`2.83 bpsp`) primarily through per-image Meta-Adaptive (`MA`) context trees that split jointly on channel, scale, directional gradients, and neighbor residuals, paired with non-parametric leaf histograms. Current `vpxc` uses 16 fixed 1-D activity buckets and rigid Student-$t$ shapes.
- **Design:**
  1. Create `src/vpxc/trees.py` implementing greedy top-down binary context tree construction over quantized integer causal properties:
     - `prop_0`: channel & scale ID `(c, s, sub_idx)`
     - `prop_1..4`: signed/absolute axial disagreements `N - S`, `W - E`, `|N - S|`, `|W - E|`
     - `prop_5..6`: saddle curvature `|(N + S) - (W + E)|` and `sqrt(5)` ring Laplacian
     - `prop_7..9`: causal neighbor residual magnitudes & signs from `r_s^(<k)` and `r_{s+1}`
     - `prop_10..11`: co-located luminance residual `r_s^(0)` and `|r_s^(0)|` (for `c > 0`)
     - `prop_12`: CFA / `4:2:0` coset phase `(r mod 2) * 2 + (c mod 2)`
  2. Accept a split `(prop_j > threshold)` if and only if the exact empirical `Core66` KL reduction exceeds node serialization + histogram table cost (`ΔL_MDL < -16 bits`).
  3. Cluster leaf histograms via agglomerative Jensen–Shannon merging into `K ≤ 64` shared `Core66` distributions, and compress non-parametric deviations from the closest parametric Student-$t$ prior (`SIGMA_Q8`, `nu_idx`) using Golomb-coded frequency deltas (capturing skewed/non-Student-$t$ residual shapes at minimal header cost).
- **Dependencies:** Requires Workstream 1B native histogram accumulation for fast multi-threshold candidate sweeps at `--effort 2`.

---

## 4. Workstream 3: Pre-Gamma Acquisition Header `φ_acq` & Multiscale `D_8` Sheaf-Kalman

### 4.1 Inverse Tone-Curve LUT + `4 × 24` Parity-Conditioned Demosaic / `4:2:0` Taps (`φ_acq`)
- **Problem:** On Kodak, `Co` (`2.5928 bpsp`) and `Cg` (`2.5684 bpsp`) consume `54%` of all bits despite originating from `4:2:0 PhotoYCC` (`75%` interpolated chroma). Because `4:2:0` upsampling and camera Bayer demosaicing are linear in *pre-gamma* space, applying linear parity taps after nonlinear sRGB/PhotoYCC gamma leaves high-contrast edge chroma residuals uncancelled.
- **Design:**
  1. Implement `src/vpxc/acquisition.py` fitting an **8-knot monotone PCHIP inverse tone-curve LUT (`96 bits`)** jointly with the `SL(3)-WSIL` color matrix and **`4 × 24` `Q4.6` parity-indexed cross-channel taps** (`p_CFA in Z_2^2`).
  2. Maintain exact integer bijectivity without RITC inflation by evaluating the inverse-gamma lookup -> linear `4 × 24` `Q4.6` convolution -> forward-gamma lookup **purely inside the causal predictor** for `μ_Q4(v)` on channels `c ≥ 1` (conditioned on already-decoded `Y_0` and causal neighbors), using integer LUT interpolation so `decode()` remains 100% float-free.
  3. Gate the entire `φ_acq` block (~`145 bytes`) with a 1-bit MDL bypass flag so Lanczos-resized or synthetic images pay `1 bit` worst-case overhead.
- **Files:** `src/vpxc/acquisition.py`, `src/vpxc/color.py`, `src/vpxc/slgf_model.py`.

### 4.2 Multiscale 2-D Window `D_8` Sheaf-Kalman Fusion & `16 px` Causal Tile Halo
- **Problem:** Non-local self-similarity (repeating windows, bricks, text, spokes) predominantly lives in high-frequency detail scales `lvl1` and `lvl0` (`69%` of total bits), whereas `Sheaf-Kalman` currently runs only on `coarse` and small tiles. Additionally, `codec.py` slices tiles with zero halo, resetting spatial predictors and `I-SSM` states at tile boundaries.
- **Design:**
  1. Add a **`16 px` causal top/left halo** in `codec.py` / `slgf_model.py`: tiles are still encoded/decoded in raster tile order, passing the already-decoded `16 px` border from `(ty - 1, tx)` and `(ty, tx - 1)` into `forward_tower` / `_level_geometry_features` while coding only interior tile odd sites.
  2. Build a causal coarse-to-fine **`D_8 × Aff_1` patch hash index** over `5×5` quantized Luma patches in `c = 0`, retrieving up to `M = 8` causal non-local matches within a `128 px` search radius (plus global dictionary anchors for screen content).
  3. Apply $\chi^2_{0.99, d}$ Mahalanobis shot-noise gating (`||y_m - μ_loc||_2^2 ≤ τ · 2 σ_η^2`) and fuse accepted matches via Section 3.3 Eq. (4) into **both** the fractional mean `μ_Q4(v)` and the reduced Student-$t$ scale `σ_pred^2(v)` at `lvl2`, `lvl1`, and `lvl0`.
- **Dependencies:** Relies on Workstream 1B native patch-matching kernel to stay within `< 50 ms/MP`.

---

## 5. Workstream 4: Trained `1.8 MB` `INT8` `I-SSM` Backbone & `S^3-BB-rANS` Integration

### 5.1 Quantization-Aware Training (QAT) & Normative `INT8` Inference Graph
- **Problem:** Reaching `2.40 bpsp` on Kodak and `2.19 bpsp` on `CLIC.p` (surpassing `HPAC-FT`'s `2.52` / `2.34 bpsp` without `8.45 s` of test-time SGD) requires the specification's learned `1.8 MB` `INT8` multiscale backbone (`Gate G2`).
- **Design:**
  1. Add an offline training package `training/` (PyTorch + custom C++/CUDA `Q15.16`/`Q1.30` `I-SSM` straight-through-estimator autograd op) trained on `DIV2K-train`, `Flicker2W`, and `OpenImages` lossless crops.
  2. Architecture per scale `s in {0..3}`:
     - `INT8` depthwise-separable `3×3` diamond/axial conv stem (`INT32` accumulators, normative `requant_q16` shifts, integer ReLU6/hard-swish — zero floating-point ops).
     - 2 blocks of 4-directional **64-channel normative `Q15.16`/`Q1.30` `I-SSM`** (`A_t in [0, 65472]` predicted from causal integer gates) emitting context stalk `h_s(v) in Z^64`.
     - Shared parameter head emitting `(μ_+, μ_-, π_bridge, σ_q8, ν_logits)` plus the 64-D basis for the closed-form per-image **$64 × 64$ Cholesky MDL Ridge solve** `φ_head` (`< 3 ms` at encode time, pure `INT64` dot product at decode time).
  3. Pack quantized `INT8` weights (`≤ 1.8 MB`) as a versioned static resource (`vpxc/weights/slgf_v1.bin` with SHA-256 verification), while retaining the `0-byte` white-box profile (`--profile slgf-zero`) for weight-free embedded deployments.

### 5.2 End-to-End `S^3-BB-rANS` Tile Latent Coupling
- **Problem:** `s3_bb_encode` and `s3_bb_decode` are implemented and verified in `src/vpxc/entropy.py`, but not yet wired into tile-level latent coding in `slgf_model.py`.
- **Design:**
  1. On tiles `≥ 128 × 128`, isolate the deterministic `1%` non-adjacent seed sub-coset `x_seed ⊂ r_0^(3)` (`(r mod 16 == 1) & (c mod 14 == 0)`), whose 4 diamond even neighbors and 4 diagonal odd neighbors lie strictly inside `x_main`.
  2. Encode `x_seed` under `Q(· | x_main, φ̂)` into `RansBitsBackStack`, pop quantized per-block (`32×32`) texture/noise latent codes `z ~ q_Ψ(· | x_main)` (`-log2 q_Ψ ≤ 384 bits < |S_0|`), condition `x_main` on `z`, and push `z` under learned prior `P_Z`, achieving `0.000 bpsp` cold-start bit waste.
  3. Per Gate `G3` of the specification, retain `S^3-BB-rANS` in the default profile only if ablation on `DIV2K` and `CLIC.p` confirms `≥ 0.02 bpsp` net rate reduction; otherwise expose it exclusively under `--effort 3` (`SLGF-Max`).

---

## 6. Non-Negotiable Invariants & Falsifiable Acceptance Gates

Every pull request implementing the workstreams above must pass five strict invariants:

1. **Zero-Float Normative Decode:** Static AST inspection + runtime tracing confirming `decode()` executes **zero** IEEE-754 floating-point instructions across all profiles.
2. **Cross-ISA Bit-Identity (`Gate G0`):** Identical `SHA-256(decode(encode(X))) == SHA-256(X)` and identical decoded states across `x86_64` and `ARM64` (and `CUDA` once GPU kernels land) on `10^4` randomized 8-bit/12-bit/16-bit/RGBA/adversarial-noise tensors (`1×1` through `8192×8192`).
3. **Pigeonhole Expansion Cap:** Worst-case expansion on cryptographic white noise remains bounded by header + 1-byte raw tile bypass (`< +0.0001 bpsp` on `768×768` tiles).
4. **Bitstream Versioning:** Increment `VERSION` cleanly in `src/vpxc/codec.py` while keeping a v2 decoder path or explicit version dispatch so existing `.vpxc` payloads fail or decode deterministically without silent corruption.
5. **Measured Benchmark Gates:**

| Milestone | Scope | Kodak 24-Img Gate (`bpsp`) | `CLIC.p` Gate (`bpsp`) | `1 MP` CPU Decode Gate |
| :--- | :--- | :---: | :---: | :---: |
| **Current Baseline (`v0.2`)** | Zero-weight pure Python + NumPy | `3.186` *(measured)* | — | `~5,000 ms` *(Python)* |
| **Milestone M1 (`WS1`)** | Coarse `Core66 rANS` + native SIMD `_accel` | `≤ 3.13` | `≤ 2.90` | `≤ 60 ms` |
| **Milestone M2 (`WS2`)** | MDL `MANIAC` trees + multi-predictor bank | `≤ 2.93` | `≤ 2.68` | `≤ 75 ms` |
| **Milestone M3 (`WS3` / Gate `G1`)** | Pre-gamma `φ_acq` + multiscale `D_8` Sheaf-Kalman (`SLGF-Zero`) | **`≤ 2.72`** *(beats `JPEG XL -e 9`)* | **`≤ 2.51`** | `≤ 45 ms` (8-core SIMD) |
| **Milestone M4 (`WS4` / Gate `G2–G3`)** | Trained `1.8 MB` `INT8` `I-SSM` + `S^3-BB-rANS` (`SLGF`) | **`≤ 2.44`** *(beats `HPAC-FT`)* | **`≤ 2.24`** | `≤ 42 ms` CPU / `≤ 10 ms` GPU |
