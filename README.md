# `vpxc` — Symplectic Lattice-Gauge Flows (`SLGF`) for Frontier Lossless Image Compression

`vpxc` explores next-generation **exact, deterministic, hardware-parallel lossless image compression** on $\mathbb{Z}_M^{H \times W \times C}$ engineered to surpass both classical SOTA (`JPEG XL Modular -e 9`, `FLIF/MANIAC`) and neural SOTA (`HPAC`, `HPAC-FT`, `CALLIC`, `ArIB-BPS`) without test-time backpropagation or floating-point non-determinism.

The Python package in `src/vpxc` is the **SLGF-Zero** codec: the integer, zero-weight part of that specification. It round-trips `uint8` and `uint16` images exactly. The rate and latency numbers in the projection table below are research targets for the full model, not measurements of this package. Measured rates are in [Measured SLGF-Zero rates](#measured-slgf-zero-rates).

```bash
PYTHONPATH=src python3 -m vpxc roundtrip photo.png
PYTHONPATH=src python3 -m vpxc encode photo.png photo.vpxc
PYTHONPATH=src python3 -m vpxc decode photo.vpxc photo.png
PYTHONPATH=src python3 -m unittest tests.test_slgf
```

`numpy` is required. Loading PNG and JPEG files from the command line also needs `pillow`.

---

## Core Mathematical Thesis

Existing lossless codecs leave **$0.35\text{--}0.55\;\mathrm{bpsp}$** ($1.05\text{--}1.65\;\mathrm{bpp}$ on RGB24) on the table due to six structural ceilings proved and audited in this repository:

1. **Non-Stationary Singularity Surprise vs. Szegő–Whittle Invariance:** While any causal half-plane scan achieves the exact block entropy rate on *stationary* Gaussian fields, natural jump-diffusion contours force causal raster predictors to pay contour-location + heavy-tail surprise at *every* scanline crossing (**$-1.6168\;\mathrm{bpsp}$** recovered on contour neighborhoods via coarse-to-fine bilateral sheaf line-process bridges).
2. **ISP / Bayer CFA & Chroma-Subsampling Submanifolds:** $50\%\text{--}66.7\%$ of rendered RGB24 sub-pixels lie within a $\pm 0.5\;\mathrm{LSB}$ tube of a camera/ISP acquisition submanifold (`PhotoYCC` 4:2:0 on Kodak; Bayer demosaicing on sRGB photos). `SLGF`'s closed-form two-part Rissanen MDL header $\phi_{\mathrm{acq}}$ ($\approx 1{,}154\;\mathrm{bits}$) + $\mathrm{SL}(C)$ Whitehead–Symplectic Integer Lifting (`WSIL`) recovers **$-0.6055\;\mathrm{bpsp}$** over `YCoCg-R` on rotated CCMs and up to **$-2.43\;\mathrm{bpsp}$** on 1:1 demosaiced green cosets in $<10\;\mathrm{ms}$.
3. **Shot-Noise Floor & Patch-Subtraction Penalty:** Hard block subtraction (`LZ77`, `IntraBC`, `JPEG XL Patches`) doubles sensor shot noise ($2\sigma_\eta^2$, **$+0.500\;\mathrm{bpsp}$** floor penalty). Precision-weighted $D_8$-equivariant **Sheaf-Kalman fusion** approaches the physical Poisson–Gaussian noise floor at rate $O(1/M)$ (**$-0.4175\;\mathrm{bpsp}$** vs. patch subtraction at $M=8$) while collapsing to $0\;\mathrm{bpsp}$ exact block copy when $\sigma_\eta \to 0$.
4. **Rounding-Induced Total Correlation (`RITC` Theorem):** Every bijection $T:\mathbb{Z}^N\to\mathbb{Z}^N$ preserves $H(T(X))=H(X)$, so stacking $M \gg 2$ nonlinear integer lifting layers (`iWave++`) converts rounding variance directly into cross-band total correlation (**$+0.5885\;\mathrm{bpsp}$** penalty at $M=12$). `SLGF` restricts spatial lifting to $L=1$ anti-aliased quincunx step per scale and places all deep nonlinear capacity inside fractional Student-$t$ CDF parameters.
5. **100% Cross-ISA Integer Determinism (`I-SSM` + `Dyadic-Core66 rANS`):** Normative `Q15.16`/`Q1.30` balanced-tree integer state-space scans + $132$-byte `Dyadic-Core66 rANS` tables eliminate IEEE-754 rANS desynchronization while holding discretization KL loss to **$< 0.0013\;\mathrm{bits/sample}$** across 8–16-bit images.
6. **Zero-Waste Single-Image Bits-Back (`S^3-BB-rANS`) & Closed-Form MDL Adaptation:** Seeding the `rANS` stack from an isolated $1\%$ fine-lattice sub-coset eliminates single-image bits-back cold-start overhead ($0\;\mathrm{bits}$ waste), while a single $<3\;\mathrm{ms}$ Cholesky solve ($\phi_{\mathrm{head}}$, $\approx 557\;\mathrm{bits}$) replaces $8.45\;\mathrm{s}$ of per-image LoRA SGD.

---

## Projected Performance Summary (`bpsp`, RGB24 = `8.00 bpsp` raw)

| Codec / Profile | Kodak | DIV2K | CLIC.p | CLIC.m | Doc24 | CT 16-bit | Encode ($1\;\mathrm{MP}$) | Decode ($1\;\mathrm{MP}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `PNG` (opt) | 4.35 | 4.23 | 3.93 | 3.88 | — | — | ~350 ms | ~8 ms CPU |
| `JPEG XL (-d 0 -e 9)` | 2.87 | 2.79 | 2.63 | 2.40 | 2.30 | 4.82 | ~900 ms | ~18 ms CPU |
| `HPAC` (2025, 0-shot) | 2.73 | 2.52 | 2.38 | — | 2.46 | 4.61 | ~45 ms GPU | ~45 ms GPU |
| `HPAC-FT` (2025, SGD) | 2.52 | 2.47 | 2.34 | — | 1.94 | 4.50 | ~8,450 ms GPU | ~45 ms GPU |
| **`vpxc` (`SLGF` Target)** | **2.40 ± 0.04** | **2.29 ± 0.04** | **2.19 ± 0.04** | **2.00 ± 0.04** | **1.81 ± 0.06** | **4.00 ± 0.08** | **25–65 ms** | **8 ms GPU / 42 ms CPU** |

---

## Measured SLGF-Zero rates

SLGF-Zero encodes each tile with the smallest of a raw bypass, a median-edge raster coder, a causal Sheaf-Kalman blend (`M ≤ 4`), a palette (at most 256 colors), and a 4-level quincunx tower. The tower uses one predict/update step per scale, either a flat neighbor mean or an integer sheaf weight, and codes detail bands with adaptive Rice or a 66-bin Dyadic-Core66 rANS table. Channel mixing is an exact lift: identity, YCoCg-R, subtract-base, or a Q12 shear.

On six Kodak lossless photographs (512×768 RGB), optimized PNG averaged about **4.76 bpsp** and SLGF-Zero averaged about **3.68 bpsp** (about 11.0 bits per pixel). Every image decoded to the same pixels. Encode took about 8–11 seconds and decode about 1.3–1.7 seconds per image in pure Python.

| Image | SLGF-Zero (bpsp) | PNG optimize (bpsp) |
| :--- | ---: | ---: |
| `kodim01` | 3.814 | 5.279 |
| `kodim05` | 4.081 | 5.471 |
| `kodim07` | 3.218 | 3.834 |
| `kodim13` | 4.357 | 6.032 |
| `kodim15` | 3.380 | 4.146 |
| `kodim23` | 3.227 | 3.777 |

Two further photographs were checked after a JPEG decode, so their samples are already quantized and are not comparable to the Kodak lossless numbers: a monarch butterfly (720×960, 2.793 vs 3.582 bpsp PNG) and the Matterhorn (640×960, 2.716 vs 3.746 bpsp PNG). Both round-tripped exactly, as did the luminance plane of `kodim07` (4.018 vs 4.138 bpsp).

Still specification-only, and covered by the falsification gates rather than this codec: trained `I-SSM` weights, a fractional Student-t mixture, split-lattice bits-back, the closed-form 64-dimensional residual head, and the ISP/CFA acquisition header. The integer `I-SSM` scan, the `SL(3)` WSIL map, and Dyadic-Core66 rANS are implemented and tested; they are not yet the full Stage-2/Stage-3 model those gates describe.

---

## Repository Contents

- `src/vpxc/` — **SLGF-Zero codec:** reversible channel lifts, quincunx sheaf lifting, causal Sheaf-Kalman, adaptive Rice, Dyadic-Core66 rANS, and the normative integer `I-SSM` scan. `tests/test_slgf.py` checks exact inversion. `tests/test_real_images.py` round-trips photographs when `VPXC_IMAGE_DIR` points at them.
- [`docs/frontier-lossless-image-compression.md`](./docs/frontier-lossless-image-compression.md) — **Primary Research Specification:** structural ceilings of classical/neural SOTA, six $\mathbb{Z}^d$ mathematical frameworks, flagship `SLGF` architecture, proofs of bijectivity & minimax MDL rate bounds, comparison against three alternative frontier paradigms (`Q-NLL-LoRA`, `H-BB-VDVAE`, `MG-MANIAC-OT`), failure modes, normative integer equations, and 5-gate falsification roadmap.
- [`docs/slgf-derivations-and-numerical-checks.md`](./docs/slgf-derivations-and-numerical-checks.md) — **Derivations & Numerical Rate Audits (Part I):** `bpsp`/`bpp` benchmark normalization, 2D `WSIL` zero-error identity, Szegő–Whittle vs. jump-diffusion singularity bracketing, Bayer/PhotoYCC ISP parity gains, Sheaf-Kalman fusion, and `S^3-BB-rANS` causality checks.
- [`docs/vpxc-extended-ritc-wsil-rans-audits.md`](./docs/vpxc-extended-ritc-wsil-rans-audits.md) — **Extended Numerical Audits (Part II):** `RITC` depth-vs-TC verification ($M=1\to 12$), $\mathrm{SL}(3)$-`WSIL` vs. `YCoCg-R` eigenframe test, `Dyadic-Core66` 16-bit `rANS` quantization bounds, chunked `Q16`/`Q30` `I-SSM` determinism, and closed-form Ridge MDL head verification.
