"""Whitehead–symplectic integer lifting.

The 2-D squeeze and the SL(3) factorization follow Appendix A of
``docs/frontier-lossless-image-compression.md``. Any integer multipliers are
bijective: each step rewrites one coordinate from frozen coordinates, so the
inverse undoes those steps in reverse order. ``s`` and ``s^{-1}`` do not have
to multiply to 1 in Q12.
"""

from __future__ import annotations

import numpy as np

Q12 = 12
Q12_ONE = 1 << Q12
Q12_BIAS = 1 << (Q12 - 1)


def r12(m: int, z: int) -> int:
    """Round-to-nearest Q12 product, ties toward +infinity (``floor(x+1/2)``)."""
    return (int(m) * int(z) + Q12_BIAS) // Q12_ONE


def r12_array(m: int, z: np.ndarray) -> np.ndarray:
    zz = z.astype(np.int64, copy=False)
    return ((np.int64(m) * zz + Q12_BIAS) // Q12_ONE).astype(np.int32)


def wsil2_forward(x1: int, x2: int, s_q: int, s_inv_q: int) -> tuple[int, int]:
    """2-D WSIL squeeze. At ``s_q == s_inv_q == 4096`` this is the identity."""
    u2 = x2 - r12(s_q, x1)
    u1 = x1 + r12(s_inv_q - Q12_ONE, u2)
    y2 = u2 + u1
    y1 = u1 + r12(s_q - Q12_ONE, y2)
    return y1, y2


def wsil2_inverse(y1: int, y2: int, s_q: int, s_inv_q: int) -> tuple[int, int]:
    u1 = y1 - r12(s_q - Q12_ONE, y2)
    u2 = y2 - u1
    x1 = u1 - r12(s_inv_q - Q12_ONE, u2)
    x2 = u2 + r12(s_q, x1)
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
    c0 = xp0 + r12(u01 * xp1 + u02 * xp2, 1)
    c1 = xp1 + r12(u12, xp2)
    c2 = xp2
    u2 = c2 - r12(s2, c1)
    u1 = c1 + r12(s2_inv - Q12_ONE, u2)
    c2p = u2 + u1
    c1p = u1 + r12(s2 - Q12_ONE, c2p)
    v2 = c1p - r12(s1, c0)
    v1 = c0 + r12(s1_inv - Q12_ONE, v2)
    c1pp = v2 + v1
    c0pp = v1 + r12(s1 - Q12_ONE, c1pp)
    y0 = c0pp
    y1 = c1pp + r12(l10, c0pp)
    y2 = c2p + r12(l20 * c0pp + l21 * c1pp, 1)
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
    c2p = y2 - r12(l20 * c0pp + l21 * c1pp, 1)
    v1 = c0pp - r12(s1 - Q12_ONE, c1pp)
    v2 = c1pp - v1
    c0 = v1 - r12(s1_inv - Q12_ONE, v2)
    c1p = v2 + r12(s1, c0)
    u1 = c1p - r12(s2 - Q12_ONE, c2p)
    u2 = c2p - u1
    c1 = u1 - r12(s2_inv - Q12_ONE, u2)
    c2 = u2 + r12(s2, c1)
    xp2 = c2
    xp1 = c1 - r12(u12, c2)
    xp0 = c0 - r12(u01 * xp1 + u02 * c2, 1)
    out = [0, 0, 0]
    out[perm[0]] = xp0
    out[perm[1]] = xp1
    out[perm[2]] = xp2
    return out[0], out[1], out[2]


def q12(value: float) -> int:
    if not np.isfinite(value):
        return 0
    q = int(np.round(float(value) * Q12_ONE))
    if q > 32767:
        return 32767
    if q < -32768:
        return -32768
    return q
