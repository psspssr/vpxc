# Extended Numerical Audits: RITC, $\mathrm{SL}(3)$ WSIL, Dyadic-Core66 rANS, and MDL Ridge Adaptation

Companion verification report extending [`slgf-derivations-and-numerical-checks.md`](./slgf-derivations-and-numerical-checks.md) for the `vpxc` primary research monograph ([`frontier-lossless-image-compression.md`](./frontier-lossless-image-compression.md)).

---

## 1. Numerical Verification of Theorem 1.2 (Rounding-Induced Total Correlation, `RITC`)

Setup: $N = 2 \times 10^5$ samples drawn from correlated discrete Gaussian $(X_1, X_2) = \lfloor \mathcal{N}(0, \Sigma) \rceil \in \mathbb{Z}^2$ with $\sigma = 18.0$, $\rho = 0.94$.
- Exact joint discrete entropy: $H(X_1, X_2) = 10.8667\;\mathrm{bits}$ ($5.4333\;\mathrm{bpsp}$).
- Uncorrelated marginal sum: $H(X_1) + H(X_2) = 12.4378\;\mathrm{bits}$ ($6.2189\;\mathrm{bpsp}$).

| Integer Transform Depth | Joint Entropy $H(Z_1, Z_2)$ (`bits`) | Factorized Rate $H(Z_1) + H(Z_2)$ (`bits`) | Induced Total Correlation $\mathrm{TC}(Z_1; Z_2)$ (`bits`) | Excess Rate Penalty (`bpsp`) |
| :--- | :---: | :---: | :---: | :---: |
| Shallow $L=1$ (1-predict, 1-update linear lifting) | $10.8667$ | $10.9073$ | $0.0406$ | $+0.0203$ |
| Stacked nonlinear lifting $M = 2$ | $10.8667$ | $11.2472$ | $0.3805$ | $+0.1903$ |
| Stacked nonlinear lifting $M = 4$ | $10.8667$ | $11.6469$ | $0.7802$ | $+0.3901$ |
| Stacked nonlinear lifting $M = 8$ | $10.8667$ | $11.8908$ | $1.0241$ | $+0.5120$ |
| Stacked nonlinear lifting $M = 12$ | $10.8667$ | $12.0436$ | $1.1769$ | $+0.5885$ |

**Takeaway:** Joint discrete entropy is strictly invariant to 4 decimal places across all depths $M$, proving that deep integer lifting (`iWave++`-style) degrades factorized coding purely by converting marginal rounding noise into deterministic cross-band total correlation.

---

## 2. Exact $\mathrm{SL}(3, \mathbb{R})$ `WSIL` Color Eigenframe vs. Fixed `YCoCg-R`

Setup: $N = 10^5$ samples with latent scene opponent standard deviations $(35.0, 4.5, 1.8)$ rendered through a unit-determinant non-standard camera CCM $A_{\mathrm{ccm}} \in \mathrm{SL}(3, \mathbb{R})$ and quantized to $\mathbb{Z}^3$. Inverse $W = A_{\mathrm{ccm}}^{-1} = L \operatorname{diag}(d_0, d_1, d_2) U$ factored into two unitriangular shears plus two 4-step 2D `WSIL` squeezes $\operatorname{diag}(1, d_0 d_1, (d_0 d_1)^{-1})$ and $\operatorname{diag}(d_0, d_0^{-1}, 1)$.

- Exact integer invertibility over $10^5$ $\mathbb{Z}^3$ triples: **$0$ reconstruction errors ($100\%$ bit-exact)**.
- True continuous-to-discrete latent eigenframe floor: **$4.7709\;\mathrm{bpsp}$**.
- Raw RGB factorized rate: **$6.9126\;\mathrm{bpsp}$**.
- Fixed `YCoCg-R` factorized rate: **$5.4495\;\mathrm{bpsp}$**.
- Image-adaptive $\mathrm{SL}(3)$-`WSIL` factorized rate: **$4.8440\;\mathrm{bpsp}$** (**$-0.6055\;\mathrm{bpsp}$** improvement over `YCoCg-R`, within $0.073\;\mathrm{bpsp}$ of the unmixed oracle eigenframe).

---

## 3. Dyadic Quotient-Remainder `Core-66` 16-Bit `rANS` Quantization Audit

Naive dense $K = 513$ `rANS` tables with $M = 2^{16}$ and Laplace safety floor $f_k \ge 1$ waste $513 / 65536 = 0.783\%$ of total probability mass on empty tail bins when $\sigma \lesssim 1$, causing $+0.0104\;\mathrm{bits/sample}$ quantization KL loss.

Using **Dyadic Quotient-Remainder `Core-66` `rANS`** — dyadic shift $b(\sigma) = \max(0, \lceil \log_2(\sigma / 2.5) \rceil)$, centered quotient $q = (r + 2^{b-1}) \gg b$ coded over $65$ core bins $[-32, 32]$ plus $1$ escape symbol ($M = 65536$, table size $= 132\;\mathrm{bytes}$), exact sub-bucket binary tree on $|q| \le 6$, raw $b$-bit bypass on $|q| > 6$, and Exp-Golomb tail on $|q| > 32$ — yields $< 0.0013\;\mathrm{bits/sample}$ total overhead across both 8-bit and 16-bit scales:

| Scale $\sigma$ (Student-$t$, $\nu = 3.5$) | Dyadic Shift $b(\sigma)$ | Exact Discrete Entropy $H(P)$ (`bits`) | `Dyadic-Core66` 16-Bit `rANS` Rate (`bits`) | Total Discretization + Escape Overhead (`bits`) |
| :---: | :---: | :---: | :---: | :---: |
| $0.8$ | $0$ | $2.23611$ | $2.23669$ | $+0.000579$ |
| $2.5$ | $0$ | $3.81291$ | $3.81342$ | $+0.000511$ |
| $8.0$ | $2$ | $5.48351$ | $5.48445$ | $+0.000939$ |
| $25.0$ | $4$ | $7.12664$ | $7.12759$ | $+0.000955$ |
| $120.0$ | $6$ | $9.38959$ | $9.39089$ | $+0.001293$ |

---

## 4. Canonical Chunked-Tree Integer State-Space Scan (`I-SSM`) & Closed-Form MDL Ridge Head

1. **Chunked `Q15.16` / `Q1.30` `I-SSM` Determinism:** For sequence length $L = 256$, chunk size $B = 16$, intra-chunk `Q15.16` sequential rollout bridged across chunks by canonical binary-tree `Q1.30` affine composition agrees with pure sequential `Q15.16` rollout to within $\max_t |h_t^{\mathrm{seq}} - h_t^{\mathrm{tree}}| = 1\;\mathrm{LSB}$ ($1.5 \times 10^{-5}$ in `Q16`), while reducing parallel critical-path depth from $L$ to $B + \log_2(L/B)$ and guaranteeing $100\%$ cross-platform integer determinism by defining the `Q1.30`-bridged chunked operator as the normative bitstream specification.
2. **Closed-Form MDL Linear Head Adaptation ($\phi_{\mathrm{head}}$):** On Kodak resolution ($N_{\mathrm{px}} = 393{,}216$, $D = 64$ features, Laplace domain shift $b_w = 1.8$, $\sigma_\eta = 3.2$), single-shot Cholesky Ridge regression quantized to $q = 2^{-5}$ requires **$557.2\;\mathrm{bits}$** of Laplace-coded header ($0.000472\;\mathrm{bpsp}$ across 3 channels) and reduces residual codelength from $4.0791$ to $3.7262\;\mathrm{bits/px}$ (**$-0.3528\;\mathrm{bits/px}$ net gain**) in a single $<3\;\mathrm{ms}$ matrix solve without SGD.
