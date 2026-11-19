# C++ migration of `vpxc`

Python remains the normative reference for the container. The C++ library in `cpp/` is the implementation that should carry encode and decode. A stream written by either side must decode to the same pixels on the other side.

## Why C++

On the 24-image Kodak suite the pure-Python encoder spends about 32 seconds per photograph and the decoder about 6 seconds. The arithmetic is integer, branchy, and sequential inside each tile. That is a poor fit for a NumPy orchestration layer and a good fit for a straight C++ translation of the same integer rules.

## Layout

```
cpp/include/vpxc/   public and internal headers
cpp/src/            one translation unit per Python module
cpp/bindings/       pybind11 module vpxc._native
cpp/tests/          differential checks against the Python package
```

C++20. Errors are `vpxc::CodecError`. Planes are row-major `int32_t`. Decode paths do not use floating point. Encode may use `double` only for the same MDL ridge and tree-split decisions the Python encoder already makes. Those decisions are stored in the bitstream, so a small numeric difference changes which tool wins, not whether the other language can decode the result.

## Module map

| C++ | Python | Role |
| :--- | :--- | :--- |
| `bitio` | `entropy.BitWriter` / `BitReader` | MSB-first bit packing, Rice, Exp-Golomb |
| `entropy` | `entropy.py` | Dyadic split, rANS, Student-t Core66, S^3 stack |
| `wsil` / `color` | `wsil.py`, `color.py` | Reversible channel lifts |
| `lattice` | `lattice.py` | One-step quincunx lift and packing |
| `issm` | `issm.py` | Q15.16 / Q1.30 scan |
| `spatial` | `spatial.py` | MED raster and D8 Sheaf-Kalman |
| `coarse` | `coarse.py` | Base-band predictor bank + Core66 |
| `trees` | `trees.py` | MANIAC sigma tree and histogram clusters |
| `acquisition` | `acquisition.py` | 8-knot tone LUT and 4×24 taps |
| `matching` | `matching.py` | Causal odd-site D8 matches |
| `bitsback` | `bitsback.py` | Tile-level S^3-BB-rANS |
| `slgf` | `slgf_model.py` | Version-2 multiscale tile |
| `slgf_v3` | `slgf_v3.py` | Version-3 tile |
| `codec` | `codec.py` | Container, tiling, 16 px halo, profile choice |

## Public API

```cpp
std::vector<uint8_t> encode(const Image& image, const EncodeOptions& opt);
Image decode(std::span<const uint8_t> bytes);
```

`EncodeOptions` mirrors the Python arguments: `tile`, `effort` in {1, 2, 3}, `profile`, `version` in {2, 3}.

## Correctness bar

1. Every Python unit test still passes on the Python reference.
2. Python-encoded bytes decode in C++ to the same pixels, including version 2 and version 3, grayscale, RGB, RGBA, uint8, and uint16.
3. C++-encoded bytes decode in Python to the same pixels.
4. A forced `slgf` tile does not throw, and a random uint8 tile stays within the raw-size expansion cap.
5. Decode translation units contain no floating-point operations.

Byte-identical encodes are required for the integer coders (Rice, rANS, lifting, container framing). Ridge weights come from a double Gaussian elimination of the same normal equations NumPy solves. If a platform LAPACK disagrees in the last bit, the test records the mismatch and still requires cross-decode.

## What stays in Python

The benchmark harness, the training script, and the CLI stay in Python. The CLI calls the native module when it is built and the pure-Python codec otherwise. `VPXC_PYTHON=1` forces the reference.
