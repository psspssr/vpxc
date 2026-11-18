"""Causal raster coder: median edge detection plus optional Sheaf-Kalman blend.

Predictions use only samples that are already decoded, so the residual stream
is an exact bijection. Sheaf-Kalman keeps up to four causal matches that share
a coarse context hash and precision-weights them against the local median
prediction, rejecting matches outside a gradient gate (the shot-noise gate in
the specification, with ``M <= 4``).
"""

from __future__ import annotations

import numpy as np

from vpxc.entropy import BitReader, BitWriter, rice_k, unzigzag, zigzag

MATCHES = 4


def _bucket(grad: int) -> int:
    if grad <= 0:
        return 0
    bit = grad.bit_length() - 1
    return bit if bit < 7 else 7


def _median(left: int, up: int, ul: int, have_l: bool, have_u: bool, have_ul: bool) -> int:
    if not have_l and not have_u:
        return 0
    if not have_l:
        return up
    if not have_u:
        return left
    if not have_ul:
        return (left + up) >> 1
    if left > up:
        hi, lo = left, up
    else:
        hi, lo = up, left
    pred = left + up - ul
    if ul < lo:
        return hi
    if ul > hi:
        return lo
    return pred


def _kalman(med: int, matches: list[int], grad: int) -> int:
    gate = max(8, 2 * grad + 2)
    accepted = [m for m in matches if abs(m - med) <= gate]
    if not accepted:
        return med
    s_loc2 = grad * grad + 4
    s_eta2 = 16
    acc = med * s_eta2
    weight = s_eta2
    for match in accepted:
        acc += match * s_loc2
        weight += s_loc2
    return int((2 * acc + weight) // (2 * weight))


def encode_raster(plane: np.ndarray, valid: np.ndarray, kalman: bool) -> bytes:
    rows = plane.astype(np.int32, copy=False).tolist()
    vrows = valid.tolist()
    height = len(rows)
    width = len(rows[0]) if height else 0
    writer = BitWriter()
    acc = [16] * 8
    count = [1] * 8
    table: dict[tuple[int, int, int], list[int]] = {}
    for y in range(height):
        row = rows[y]
        vrow = vrows[y]
        up_row = rows[y - 1] if y else None
        up_valid = vrows[y - 1] if y else None
        for x in range(width):
            if not vrow[x]:
                continue
            have_l = x > 0 and vrow[x - 1]
            have_u = up_valid is not None and up_valid[x]
            have_ul = up_valid is not None and x > 0 and up_valid[x - 1]
            left = row[x - 1] if have_l else 0
            up = up_row[x] if have_u and up_row is not None else 0
            ul = up_row[x - 1] if have_ul and up_row is not None else 0
            pred = _median(left, up, ul, have_l, have_u, have_ul)
            if have_l and have_u and have_ul:
                grad = abs(left - ul) + abs(up - ul)
            else:
                grad = abs(left - up)
            if kalman:
                key = (left >> 2, up >> 2, ul >> 2)
                pred = _kalman(pred, table.get(key, []), grad)
            err = row[x] - pred
            symbol = zigzag(err)
            ctx = _bucket(grad)
            k = rice_k(acc[ctx], count[ctx])
            writer.write_rice(symbol, k)
            acc[ctx] += symbol
            count[ctx] += 1
            if count[ctx] >= 64:
                acc[ctx] = max(1, acc[ctx] >> 1)
                count[ctx] >>= 1
            if kalman:
                key = (left >> 2, up >> 2, ul >> 2)
                bucket = table.get(key)
                if bucket is None:
                    table[key] = [row[x]]
                else:
                    if len(bucket) >= MATCHES:
                        del bucket[0]
                    bucket.append(row[x])
    return writer.finish()


def decode_raster(
    data: bytes, shape: tuple[int, int], valid: np.ndarray, kalman: bool
) -> np.ndarray:
    height, width = shape
    vrows = valid.tolist()
    rows = [[0] * width for _ in range(height)]
    reader = BitReader(data)
    acc = [16] * 8
    count = [1] * 8
    table: dict[tuple[int, int, int], list[int]] = {}
    for y in range(height):
        row = rows[y]
        vrow = vrows[y]
        up_row = rows[y - 1] if y else None
        up_valid = vrows[y - 1] if y else None
        for x in range(width):
            if not vrow[x]:
                continue
            have_l = x > 0 and vrow[x - 1]
            have_u = up_valid is not None and up_valid[x]
            have_ul = up_valid is not None and x > 0 and up_valid[x - 1]
            left = row[x - 1] if have_l else 0
            up = up_row[x] if have_u and up_row is not None else 0
            ul = up_row[x - 1] if have_ul and up_row is not None else 0
            pred = _median(left, up, ul, have_l, have_u, have_ul)
            if have_l and have_u and have_ul:
                grad = abs(left - ul) + abs(up - ul)
            else:
                grad = abs(left - up)
            if kalman:
                key = (left >> 2, up >> 2, ul >> 2)
                pred = _kalman(pred, table.get(key, []), grad)
            ctx = _bucket(grad)
            k = rice_k(acc[ctx], count[ctx])
            symbol = reader.read_rice(k)
            err = unzigzag(symbol)
            value = pred + err
            row[x] = value
            acc[ctx] += symbol
            count[ctx] += 1
            if count[ctx] >= 64:
                acc[ctx] = max(1, acc[ctx] >> 1)
                count[ctx] >>= 1
            if kalman:
                key = (left >> 2, up >> 2, ul >> 2)
                bucket = table.get(key)
                if bucket is None:
                    table[key] = [value]
                else:
                    if len(bucket) >= MATCHES:
                        del bucket[0]
                    bucket.append(value)
    return np.asarray(rows, dtype=np.int32)
