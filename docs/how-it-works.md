# How `vpxc` Works (Plain-English Explainer)

At its core, lossless compression is a guessing game: **the better you predict the next pixel from pixels the decoder has already seen, the smaller the leftover error ("residual") you have to write to disk.** Because the decoder runs the exact same integer formula in the exact same order, it reconstructs `pixel = guess + residual` with zero loss.

`vpxc` applies that single idea across 2D photographs, ultra-fast `< 8 ms` decoding, and 16-bit 3D CT/MRI scans.

---

## 1. Remove Obvious Channel Redundancy (`Reversible Color Transforms`)

In a normal RGB photograph, the Red, Green, and Blue values of a pixel move up and down together (a bright pixel is bright in all three channels). Encoding them separately pays for the same brightness edge three times.

Before predicting pixels spatially, `vpxc` applies simple integer **Reversible Color Transforms** (such as `Subtract-Green`: keep `G`, store `R - G` and `B - G`, or `Subtract-Average`). Because every step uses integer addition and bit-shifts, the inverse transform restores the exact original RGB bytes with zero rounding loss.

---

## 2. Predict First, Store Only the Surprise (`Residuals`)

As `vpxc` walks an image top-to-bottom, left-to-right, neighboring pixels (above, left, upper-left, upper-right) and previously decoded color channels are already known to both encoder and decoder.

Instead of storing raw pixel values (`0..255` or `-1024..3071` in CT), `vpxc` computes an integer `guess` and stores only `residual = actual - guess`. On smooth surfaces, most residuals are `0`, `+1`, or `-1`, which take far fewer bits to encode than arbitrary 8-bit or 16-bit numbers.

`vpxc` ships two different prediction + entropy engines depending on whether you want **smallest file size** or **fastest speed**:

### Mode A: `frontier` (`--effort 1, 2, 3`) — Smallest Files (~100 ms)
Implemented in [`cpp/src/frontier.cpp`](../cpp/src/frontier.cpp):
- **8-Expert Weighted Predictor (`WP8`):** Runs 8 simple directional/planar predictors in parallel and weights each expert by how accurate it has been on recent nearby pixels.
- **36-Feature Linear Model + Online Adaptation (`NLMS`):** Combines a 36-tap integer linear model (`Q8` fixed-point weights accelerated with `AVX2` / `ARM64 NEON` SIMD) with lightweight per-pixel `NLMS` updates so weights adapt on the fly as the scan crosses different textures. At `-e 2` and `-e 3`, the encoder also fits regularized per-tile weights (`MAP Ridge`).
- **Context Decision Tree (`MA` Tree):** Flat sky has tiny errors (`±1`), while foliage or text edges have larger errors (`±15`). Mixing them into one probability table hurts both. A small decision tree groups pixels by local gradient/error activity so smooth regions use tight probability distributions and sharp edges use wider Student-$t$ distributions.
- **Exact Patch Match (`PatchPredictor`):** Keeps a tiny hash cache of recent pixel templates to bypass repeated UI/diagram patterns losslessly.

### Mode B: `fast` (`--effort 0`) — `< 8 ms` Real-Time Speed
Implemented in [`cpp/src/fast_codec.cpp`](../cpp/src/fast_codec.cpp):
- Replaces the 36-feature model and context tree with a tiny 3-neighbor median edge detector (`MED`), split into 4 row stripes.
- **Why single-stream entropy coding is slow:** Standard `rANS` decoding has a serial data dependency—you cannot look up symbol `i + 1` until symbol `i` updates the state integer.
- **4-Way Interleaving + `16 KB` L1 Lookup Table:** `fast` runs **4 independent `rANS` streams side-by-side** using a compact 12-bit (`4096`-entry, `16 KB`) lookup table that stays hot inside CPU L1 cache. The CPU pipelines 4 table lookups per loop iteration without stalling, decoding `768×512` RGB24 images in **`~7.3 ms`** and 3D CT stacks at **`> 320 MB/s`**.

---

## 3. 3D Medical CT/MRI Volumes (`VPXV`) & Byte-Exact DICOM (`VPXD`)

Medical CT and MRI scans have three properties 2D image formats handle poorly:
1. **Signed 16-bit Hounsfield Units (`int16`, e.g. `-1024` air to `+3000` bone/metal)** plus empty air padding slices and unused dynamic-range gaps.
2. **Strong 3D slice-to-slice similarity:** Slice `z` looks almost identical to slice `z - 1` a millimeter away.
3. **Random-access slice scrolling:** A `.nii.gz` (`gzip`) file compresses the whole scan as one stream, forcing you to decompress 50 MB just to view slice `#140`.

How `vpxc` handles all three ([`cpp/src/medical.cpp`](../cpp/src/medical.cpp), [`src/vpxc/medical.py`](../src/vpxc/medical.py)):
- **Per-Slice Compaction & Bypasses:** Bijectively maps signed `int16` to unsigned integers, compacts `[min, max, step_gcd]` per slice, encodes pure air/padding slices in **5 bytes**, and encodes exact duplicate slices in **1 byte**.
- **Group-of-Slices (`GoS`, default `8`) Random Access:** Splits the 3D volume into independent 8-slice chunks with a byte-offset table in the header. Slice 0 of a group is coded in 2D; slices `1..7` predict the 3D change `S[z] - S[z-1]` using spatial gradients from `S[z-1]` (handling slight gantry tilt or anatomy shift). Seeking any single slice only decodes at most a few slices inside one `GoS` (**`0.5–3.2 ms`**) instead of the whole volume.
- **`SHA-256` Byte-Exact DICOM (`VPXD`):** Splits a clinical `.dcm` file into its non-pixel DICOM tags (compressed with `zlib`) and pixel array (compressed with `vpxc`), storing the original file's 32-byte `SHA-256` hash so `unpack-dicom` is guaranteed to restore the **exact original `.dcm` byte-for-byte**.

---

## 4. Why "Zero Floats" Matters in Compression

`rANS` (range Asymmetric Numeral Systems) packs an entire tile of residuals into a single integer state using cumulative frequency tables.

If an encoder on an `x86_64` Linux server and a decoder on an `ARM64` laptop used floating-point math (`float` / `double`) while predicting pixels or computing probabilities, a single `0.0000001` hardware rounding difference (`FMA` fusion or SIMD instruction order) would shift one bin threshold by `1`—causing **every remaining pixel in the tile to decode into random garbage**.

To make cross-platform corruption impossible by construction:
- Every decode path in `vpxc` uses **pure fixed-width integer math** (`int16_t` / `int32_t` / `int64_t`).
- Automated CI ([`tests/test_native.py`](../tests/test_native.py)) disassembles the compiled C++ object files with `objdump -d` on both `x86_64` and `ARM64` to verify **zero floating-point CPU instructions** exist in the decoder, alongside `CRC-32` checksums and `libFuzzer + ASan + UBSan` testing on malformed inputs.

---

## 5. How Default Priors Were Trained Without Test-Set Leakage (`training/`)

At `--effort 1` (default ~100 ms mode), `frontier` avoids a slow two-pass per-image fit by starting from ~5.5 KB of pre-computed integer weights ([`cpp/include/vpxc/frontier_priors.hpp`](../cpp/include/vpxc/frontier_priors.hpp)) and adapting online from there:
- [`training/prepare_corpora.py`](../training/prepare_corpora.py) verifies zero overlap between training images and held-out benchmark suites (`TEST_CLIC`, `Kodak`) using both exact `SHA-256` hashes and $16 \times 16$ perceptual L1 distance.
- [`training/train_frontier.cpp`](../training/train_frontier.cpp) fits the default integer weights using 8-fold image-grouped cross-validation and shrinkage so weights generalize out-of-sample instead of memorizing training noise.

For low-level C++ module and C-ABI details, see [`docs/architecture.md`](./architecture.md).
