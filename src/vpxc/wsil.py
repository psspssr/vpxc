"""Whitehead–symplectic integer lifting over SL(C, R).

Implements the scalar and vectorized 2-D hyperbolic squeeze and SL(3) integer
bijections from Section 3.1 and Appendix A.1 of the SLGF specification, plus
closed-form 2-D spatial-gradient eigenframe factorization ``W P^T = L D U``
into normative Q12 integer multipliers.

Every forward step updates one coordinate conditioned on frozen coordinates
using exact integer shift rounding ``(m * z + 2^11) >> 12``, guaranteeing a
bit-exact bijection on ``Z^C`` for arbitrary integer multipliers.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

Q12 = 12
Q12_ONE = 1 << Q12
Q12_BIAS = 1 << (Q12 - 1)


@dataclass(frozen=True)
class SL3Params:
    perm: tuple[int, int, int] = (0, 1, 2)
    u01: int = 0
    u02: int = 0
    u12: int = 0
    l10: int = 0
    l20: int = 0
    l21: int = 0
    s1: int = Q12_ONE
    s1_inv: int = Q12_ONE
    s2: int = Q12_ONE
    s2_inv: int = Q12_ONE

    def as_tuple(self) -> tuple[int, int, int, int, int, int, int, int, int, int]:
        return (
            self.u01,
            self.u02,
            self.u12,
            self.l10,
            self.l20,
            self.l21,
            self.s1,
            self.s1_inv,
            self.s2,
            self.s2_inv,
        )


def r12(m: int, z: int) -> int:
    """Round-to-nearest Q12 product, ties toward +infinity (``floor(x+1/2)``)."""
    return (int(m) * int(z) + Q12_BIAS) >> Q12


def r12_array(m: int, z: np.ndarray) -> np.ndarray:
    zz = z.astype(np.int64, copy=False)
    return ((np.int64(int(m)) * zz + Q12_BIAS) >> Q12).astype(np.int32)


def _r12_fused2(m1: int, z1: np.ndarray, m2: int, z2: np.ndarray) -> np.ndarray:
    a = z1.astype(np.int64, copy=False)
    b = z2.astype(np.int64, copy=False)
    return ((np.int64(int(m1)) * a + np.int64(int(m2)) * b + Q12_BIAS) >> Q12).astype(
        np.int32
    )


def wsil2_forward(x1: int, x2: int, s_q: int, s_inv_q: int) -> tuple[int, int]:
    """2-D WSIL squeeze. At ``s_q == s_inv_q == 4096`` this is the identity."""
    u2 = int(x2) - r12(s_q, x1)
    u1 = int(x1) + r12(int(s_inv_q) - Q12_ONE, u2)
    y2 = u2 + u1
    y1 = u1 + r12(int(s_q) - Q12_ONE, y2)
    return y1, y2


def wsil2_inverse(y1: int, y2: int, s_q: int, s_inv_q: int) -> tuple[int, int]:
    u1 = int(y1) - r12(int(s_q) - Q12_ONE, y2)
    u2 = int(y2) - u1
    x1 = u1 - r12(int(s_inv_q) - Q12_ONE, u2)
    x2 = u2 + r12(s_q, x1)
    return x1, x2


def wsil2_forward_array(
    x1: np.ndarray, x2: np.ndarray, s_q: int, s_inv_q: int
) -> tuple[np.ndarray, np.ndarray]:
    a1 = x1.astype(np.int32, copy=False)
    a2 = x2.astype(np.int32, copy=False)
    u2 = a2 - r12_array(s_q, a1)
    u1 = a1 + r12_array(int(s_inv_q) - Q12_ONE, u2)
    y2 = u2 + u1
    y1 = u1 + r12_array(int(s_q) - Q12_ONE, y2)
    return y1, y2


def wsil2_inverse_array(
    y1: np.ndarray, y2: np.ndarray, s_q: int, s_inv_q: int
) -> tuple[np.ndarray, np.ndarray]:
    b1 = y1.astype(np.int32, copy=False)
    b2 = y2.astype(np.int32, copy=False)
    u1 = b1 - r12_array(int(s_q) - Q12_ONE, b2)
    u2 = b2 - u1
    x1 = u1 - r12_array(int(s_inv_q) - Q12_ONE, u2)
    x2 = u2 + r12_array(s_q, x1)
    return x1, x2


def sl3_forward(
    x: tuple[int, int, int] | list[int],
    perm: tuple[int, int, int],
    u01: int,
    u02: int,
    u12: int,
    l10: int,
    l20: int,
    l21: int,
    s1: int,
    s1_inv: int,
    s2: int,
    s2_inv: int,
) -> tuple[int, int, int]:
    """Appendix A.1 forward map. Shear inner products use one fused rounding."""
    xp0 = int(x[perm[0]])
    xp1 = int(x[perm[1]])
    xp2 = int(x[perm[2]])
    c0 = xp0 + r12(int(u01) * xp1 + int(u02) * xp2, 1)
    c1 = xp1 + r12(u12, xp2)
    c2 = xp2
    u2 = c2 - r12(s2, c1)
    u1 = c1 + r12(int(s2_inv) - Q12_ONE, u2)
    c2p = u2 + u1
    c1p = u1 + r12(int(s2) - Q12_ONE, c2p)
    v2 = c1p - r12(s1, c0)
    v1 = c0 + r12(int(s1_inv) - Q12_ONE, v2)
    c1pp = v2 + v1
    c0pp = v1 + r12(int(s1) - Q12_ONE, c1pp)
    y0 = c0pp
    y1 = c1pp + r12(l10, c0pp)
    y2 = c2p + r12(int(l20) * c0pp + int(l21) * c1pp, 1)
    return y0, y1, y2


def sl3_inverse(
    y: tuple[int, int, int] | list[int],
    perm: tuple[int, int, int],
    u01: int,
    u02: int,
    u12: int,
    l10: int,
    l20: int,
    l21: int,
    s1: int,
    s1_inv: int,
    s2: int,
    s2_inv: int,
) -> tuple[int, int, int]:
    y0, y1, y2 = int(y[0]), int(y[1]), int(y[2])
    c0pp = y0
    c1pp = y1 - r12(l10, c0pp)
    c2p = y2 - r12(int(l20) * c0pp + int(l21) * c1pp, 1)
    v1 = c0pp - r12(int(s1) - Q12_ONE, c1pp)
    v2 = c1pp - v1
    c0 = v1 - r12(int(s1_inv) - Q12_ONE, v2)
    c1p = v2 + r12(s1, c0)
    u1 = c1p - r12(int(s2) - Q12_ONE, c2p)
    u2 = c2p - u1
    c1 = u1 - r12(int(s2_inv) - Q12_ONE, u2)
    c2 = u2 + r12(s2, c1)
    xp2 = c2
    xp1 = c1 - r12(u12, c2)
    xp0 = c0 - r12(int(u01) * xp1 + int(u02) * c2, 1)
    out = [0, 0, 0]
    out[perm[0]] = xp0
    out[perm[1]] = xp1
    out[perm[2]] = xp2
    return out[0], out[1], out[2]


def sl3_forward_arrays(
    channels: list[np.ndarray], params: SL3Params
) -> list[np.ndarray]:
    """Vectorized Appendix A.1 forward map on three ``int32`` channel planes."""
    xp0 = channels[params.perm[0]].astype(np.int32, copy=False)
    xp1 = channels[params.perm[1]].astype(np.int32, copy=False)
    xp2 = channels[params.perm[2]].astype(np.int32, copy=False)
    c0 = xp0 + _r12_fused2(params.u01, xp1, params.u02, xp2)
    c1 = xp1 + r12_array(params.u12, xp2)
    c2 = xp2
    c1p, c2p = wsil2_forward_array(c1, c2, params.s2, params.s2_inv)
    c0pp, c1pp = wsil2_forward_array(c0, c1p, params.s1, params.s1_inv)
    y0 = c0pp
    y1 = c1pp + r12_array(params.l10, c0pp)
    y2 = c2p + _r12_fused2(params.l20, c0pp, params.l21, c1pp)
    return [y0, y1, y2]


def sl3_inverse_arrays(
    planes: list[np.ndarray], params: SL3Params
) -> list[np.ndarray]:
    """Exact inverse of :func:`sl3_forward_arrays`."""
    y0 = planes[0].astype(np.int32, copy=False)
    y1 = planes[1].astype(np.int32, copy=False)
    y2 = planes[2].astype(np.int32, copy=False)
    c0pp = y0
    c1pp = y1 - r12_array(params.l10, c0pp)
    c2p = y2 - _r12_fused2(params.l20, c0pp, params.l21, c1pp)
    c0, c1p = wsil2_inverse_array(c0pp, c1pp, params.s1, params.s1_inv)
    c1, c2 = wsil2_inverse_array(c1p, c2p, params.s2, params.s2_inv)
    xp2 = c2
    xp1 = c1 - r12_array(params.u12, c2)
    xp0 = c0 - _r12_fused2(params.u01, xp1, params.u02, c2)
    out: list[np.ndarray | None] = [None, None, None]
    out[params.perm[0]] = xp0
    out[params.perm[1]] = xp1
    out[params.perm[2]] = xp2
    return [p for p in out if p is not None]


def q12(value: float) -> int:
    if not np.isfinite(value):
        return 0
    q = int(np.round(float(value) * Q12_ONE))
    if q > 32767:
        return 32767
    if q < -32768:
        return -32768
    return q


def factor_sl3_matrix(matrix: np.ndarray) -> SL3Params:
    """Factor ``matrix`` (normalized to ``det = +1``) into Appendix A.1 Q12 parameters.

    Searches column permutations ``P`` and sign-preserving row flips so the LDU
    pivots ``(d0, d1, d2)`` stay positive and as close to 1 as possible,
    minimizing WSIL rounding variance.
    """
    mat = np.asarray(matrix, dtype=np.float64).reshape(3, 3).copy()
    det = float(np.linalg.det(mat))
    if not np.isfinite(det) or abs(det) < 1e-8:
        return SL3Params()
    if det < 0:
        mat[2, :] = -mat[2, :]
        det = -det
    mat /= det ** (1.0 / 3.0)

    best_score = float("inf")
    best_params: SL3Params | None = None
    sign_flips = ((1.0, 1.0, 1.0), (1.0, -1.0, -1.0), (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0))

    for perm in permutations((0, 1, 2)):
        perm_sign = (
            1.0
            if (perm[0] - perm[1]) * (perm[0] - perm[2]) * (perm[1] - perm[2]) < 0
            else -1.0
        )
        # Check parity of permutation directly via inversion count:
        inv_cnt = int(perm[0] > perm[1]) + int(perm[0] > perm[2]) + int(perm[1] > perm[2])
        perm_sign = -1.0 if (inv_cnt & 1) else 1.0
        wp = mat[:, list(perm)].copy()
        if perm_sign < 0:
            wp[2, :] = -wp[2, :]
        for sf in sign_flips:
            m = wp * np.asarray(sf, dtype=np.float64)[:, None]
            d0 = float(m[0, 0])
            if d0 <= 0.15 or d0 >= 6.5:
                continue
            u01 = float(m[0, 1] / d0)
            u02 = float(m[0, 2] / d0)
            l10 = float(m[1, 0] / d0)
            l20 = float(m[2, 0] / d0)
            schur11 = float(m[1, 1] - l10 * d0 * u01)
            if schur11 <= 0.15 or schur11 >= 6.5:
                continue
            d1 = schur11
            u12 = float((m[1, 2] - l10 * d0 * u02) / d1)
            l21 = float((m[2, 1] - l20 * d0 * u01) / d1)
            s1 = d0
            s2 = d0 * d1
            if s2 <= 0.15 or s2 >= 6.5:
                continue
            shear_mag = max(abs(u01), abs(u02), abs(u12), abs(l10), abs(l20), abs(l21))
            if shear_mag > 7.5:
                continue
            # Prefer squeezes close to 1 (zero rounding error at s = 1) and small shears.
            score = abs(np.log(s1)) + abs(np.log(s2)) + 0.15 * shear_mag
            if score < best_score:
                best_score = score
                best_params = SL3Params(
                    perm=(int(perm[0]), int(perm[1]), int(perm[2])),
                    u01=q12(u01),
                    u02=q12(u02),
                    u12=q12(u12),
                    l10=q12(l10),
                    l20=q12(l20),
                    l21=q12(l21),
                    s1=max(1, q12(s1)),
                    s1_inv=max(1, q12(1.0 / s1)),
                    s2=max(1, q12(s2)),
                    s2_inv=max(1, q12(1.0 / s2)),
                )
    return best_params if best_params is not None else SL3Params()


def estimate_sl3_wsil(rgb: list[np.ndarray]) -> SL3Params:
    """Estimate an SL(3)-WSIL decorrelating eigenframe from 2-D diamond Laplacians."""
    assert len(rgb) == 3
    h, w = rgb[0].shape
    step_y = max(1, h // 256)
    step_x = max(1, w // 256)
    sub = [c[::step_y, ::step_x].astype(np.float64, copy=False) for c in rgb]
    if sub[0].shape[0] >= 3 and sub[0].shape[1] >= 3:
        g = np.stack(
            [
                (
                    c[1:-1, 1:-1]
                    - 0.25 * (c[:-2, 1:-1] + c[2:, 1:-1] + c[1:-1, :-2] + c[1:-1, 2:])
                ).ravel()
                for c in sub
            ],
            axis=1,
        )
    else:
        grads = []
        if sub[0].shape[1] >= 2:
            grads.append(np.stack([np.diff(c, axis=1).ravel() for c in sub], axis=1))
        if sub[0].shape[0] >= 2:
            grads.append(np.stack([np.diff(c, axis=0).ravel() for c in sub], axis=1))
        if grads:
            g = np.concatenate(grads, axis=0)
        else:
            g = np.stack([c.ravel() - float(c.mean()) for c in sub], axis=1)
    if g.shape[0] < 4:
        return SL3Params()
    cov = (g.T @ g) / float(g.shape[0]) + 1e-4 * np.eye(3)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    rot = eigvecs[:, order].T.copy()
    if np.linalg.det(rot) < 0:
        rot[2, :] = -rot[2, :]
    return factor_sl3_matrix(rot)
