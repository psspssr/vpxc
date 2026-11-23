"""MDL-pruned MANIAC context trees and clustered Core66 leaf histograms.

The tree is built on encode (floating-point entropy estimates are allowed there)
and walked on decode with integer comparisons only. A split is kept only when
its description-length saving exceeds the node header. Leaf histograms, when
used, are Golomb-coded as frequency deltas from the closest Student-t prior.
"""

from __future__ import annotations

import math

import numpy as np

from vpxc.entropy import (
    NUM_SIGMAS,
    RANS_M,
    SIGMA_Q8,
    BitReader,
    BitWriter,
    cumulative,
    get_parametric_tables,
    normalize_freqs,
)
from vpxc.errors import CodecError
from vpxc.lattice import ilog2_int64

MAX_DEPTH = 4
MIN_LEAF = 40
MAX_LEAVES = 16
SPLIT_GAIN_BITS = 24


class TreeNode:
    __slots__ = ("sigma", "prop", "thr", "left", "right", "leaf_id")

    def __init__(
        self,
        sigma: int | None = None,
        prop: int | None = None,
        thr: int | None = None,
        left: TreeNode | None = None,
        right: TreeNode | None = None,
    ) -> None:
        self.sigma = sigma
        self.prop = prop
        self.thr = thr
        self.left = left
        self.right = right
        self.leaf_id = -1

    @property
    def is_leaf(self) -> bool:
        return self.left is None or self.right is None


def quantize_props(raw: np.ndarray) -> np.ndarray:
    """Signed integer log-magnitude, identical on encode and decode."""
    x = np.asarray(raw, dtype=np.int64)
    sign = np.where(x < 0, np.int64(-1), np.int64(1))
    sign = np.where(x == 0, np.int64(0), sign)
    mag = ilog2_int64(np.abs(x)).astype(np.int64)
    return (sign * mag).astype(np.int16)


def _mad_bits(abs_resid: np.ndarray) -> float:
    n = int(abs_resid.size)
    if n <= 0:
        return 0.0
    mad = float(abs_resid.sum()) / float(n) + 0.5
    return float(n) * math.log2(mad)


def _sigma_from_mad(abs_resid: np.ndarray) -> int:
    n = int(abs_resid.size)
    # E[|Student|] is a bit under sigma; map MAD in sample units onto Q8 scales.
    target = (int(abs_resid.sum()) * 320) // max(n, 1) + 48
    table = np.asarray(SIGMA_Q8, dtype=np.int64)
    return int(np.argmin(np.abs(table - target)))


def build_sigma_tree(qprops: np.ndarray, r_tilde: np.ndarray) -> TreeNode | None:
    """Greedy top-down tree. ``None`` when no split beats a single leaf."""
    props = np.asarray(qprops, dtype=np.int16)
    resid = np.abs(np.asarray(r_tilde, dtype=np.int64))
    n, n_prop = props.shape
    if n < MIN_LEAF * 2 or n_prop == 0:
        return None
    root_mask = np.ones(n, dtype=np.bool_)
    leaf_budget = [MAX_LEAVES]

    def rec(mask: np.ndarray, depth: int) -> TreeNode:
        subset = resid[mask]
        node = TreeNode(sigma=_sigma_from_mad(subset))
        if depth >= MAX_DEPTH or int(mask.sum()) < MIN_LEAF * 2 or leaf_budget[0] < 2:
            return node
        base = _mad_bits(subset)
        best: tuple[float, int, int, np.ndarray] | None = None
        for prop in range(n_prop):
            col = props[:, prop]
            vals = col[mask]
            lo = int(vals.min())
            hi = int(vals.max())
            if lo == hi:
                continue
            ordered = np.sort(vals)
            cands = []
            for frac in (1, 2, 3):
                cands.append(int(ordered[(ordered.size * frac) // 4]))
            seen = set()
            for thr in cands:
                if thr in seen or thr < lo or thr >= hi:
                    continue
                seen.add(thr)
                left = mask & (col <= thr)
                n_left = int(left.sum())
                n_right = int(mask.sum()) - n_left
                if n_left < MIN_LEAF or n_right < MIN_LEAF:
                    continue
                gain = base - _mad_bits(resid[left]) - _mad_bits(resid[mask & ~left])
                if best is None or gain > best[0]:
                    best = (gain, prop, thr, left)
        if best is None or best[0] < SPLIT_GAIN_BITS:
            return node
        _gain, prop, thr, left = best
        leaf_budget[0] -= 1
        right = mask & ~left
        node.sigma = None
        node.prop = prop
        node.thr = thr
        node.left = rec(left, depth + 1)
        node.right = rec(right, depth + 1)
        return node

    tree = rec(root_mask, 0)
    if tree.is_leaf:
        return None
    _number_leaves(tree, [])
    return tree


def _number_leaves(node: TreeNode, acc: list[TreeNode]) -> None:
    if node.is_leaf:
        node.leaf_id = len(acc)
        acc.append(node)
        return
    assert node.left is not None and node.right is not None
    _number_leaves(node.left, acc)
    _number_leaves(node.right, acc)


def apply_tree(node: TreeNode, qprops: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(sigma_index, leaf_id)`` for every row. Pure integer walk."""
    props = np.asarray(qprops)
    n = int(props.shape[0])
    sigma = np.empty(n, dtype=np.int32)
    leaves = np.empty(n, dtype=np.int32)

    def rec(cur: TreeNode, mask: np.ndarray) -> None:
        if cur.is_leaf or not np.any(mask):
            if np.any(mask):
                sigma[mask] = int(cur.sigma or 0)
                leaves[mask] = int(cur.leaf_id)
            return
        assert cur.left is not None and cur.right is not None and cur.prop is not None
        left = mask & (props[:, cur.prop] <= int(cur.thr or 0))
        rec(cur.left, left)
        rec(cur.right, mask & ~left)

    rec(node, np.ones(n, dtype=np.bool_))
    return sigma, leaves


def write_tree(writer: BitWriter, node: TreeNode) -> None:
    if node.is_leaf:
        writer.write_bits(1, 1)
        writer.write_bits(int(node.sigma or 0) & 31, 5)
        return
    writer.write_bits(0, 1)
    writer.write_bits(int(node.prop or 0) & 15, 4)
    writer.write_signed_golomb(int(node.thr or 0))
    assert node.left is not None and node.right is not None
    write_tree(writer, node.left)
    write_tree(writer, node.right)


def read_tree(reader: BitReader, depth: int = 0) -> TreeNode:
    if depth > 16:
        raise CodecError("MANIAC tree too deep")
    if reader.read_bits(1) == 1:
        node = TreeNode(sigma=reader.read_bits(5))
        return node
    prop = reader.read_bits(4)
    thr = reader.read_signed_golomb()
    left = read_tree(reader, depth + 1)
    right = read_tree(reader, depth + 1)
    node = TreeNode(prop=prop, thr=thr, left=left, right=right)
    return node


def number_tree(node: TreeNode) -> list[TreeNode]:
    acc: list[TreeNode] = []
    _number_leaves(node, acc)
    return acc


def _symbolize(r_tilde: np.ndarray, sig_idx: np.ndarray) -> np.ndarray:
    shifts, _freqs, _cums, _tilt = get_parametric_tables()
    rt = np.asarray(r_tilde, dtype=np.int64)
    bv = shifts[np.clip(sig_idx, 0, NUM_SIGMAS - 1)]
    pos = bv > 0
    bias = np.where(pos, np.int64(1) << np.maximum(bv - 1, 0), 0)
    quot = np.where(pos, (rt + bias) >> bv, rt)
    return np.where(np.abs(quot) > 32, 65, quot + 32).astype(np.int32)


def clustered_histograms(
    qprops: np.ndarray,
    tree: TreeNode,
    r_tilde: np.ndarray,
    nu_idx: int,
    max_clusters: int = 8,
) -> tuple[np.ndarray, np.ndarray, bytes] | None:
    """Agglomerative Jensen-Shannon-style merge of leaf Core66 histograms.

    Returns ``(freq_rows, cum_rows, header)`` or ``None`` when the header cannot
    pay for itself on this sample count.
    """
    _sigma, leaf_ids = apply_tree(tree, qprops)
    leaves = number_tree(tree)
    n_leaves = len(leaves)
    if n_leaves < 2:
        return None
    symbols = _symbolize(r_tilde, _sigma)
    counts = []
    sigmas = []
    for leaf in leaves:
        mask = leaf_ids == leaf.leaf_id
        if int(mask.sum()) == 0:
            counts.append(np.zeros(66, dtype=np.int64))
        else:
            counts.append(np.bincount(symbols[mask], minlength=66).astype(np.int64))
        sigmas.append(int(leaf.sigma or 0))
    # Merge the pair with the smallest count-weighted L1 until few clusters remain.
    active = list(range(n_leaves))
    parent = list(range(n_leaves))
    while len(active) > max_clusters:
        best = None
        for i, a in enumerate(active):
            for b in active[i + 1 :]:
                ca, cb = counts[a], counts[b]
                # Integer L1 on the normalized shape, scaled by 1024 to avoid floats.
                sa = int(ca.sum()) + 66
                sb = int(cb.sum()) + 66
                dist = int(np.abs((ca + 1) * sb - (cb + 1) * sa).sum() // max(sa, 1))
                if best is None or dist < best[0]:
                    best = (dist, a, b)
        assert best is not None
        _dist, a, b = best
        counts[a] = counts[a] + counts[b]
        parent[b] = a
        active.remove(b)
    # Path-compress leaf -> cluster representative.
    cluster_of = []
    reps = []
    rep_index = {}
    for leaf_i in range(n_leaves):
        root = leaf_i
        while parent[root] != root:
            root = parent[root]
        if root not in rep_index:
            rep_index[root] = len(reps)
            reps.append(root)
        cluster_of.append(rep_index[root])

    _shifts, freqs_tbl, _cums, _tilt = get_parametric_tables()
    freq_rows_c = []
    for root in reps:
        emp = normalize_freqs(counts[root])
        # Snap the prior sigma to the leaf sigma of the representative.
        freq_rows_c.append((emp, sigmas[root]))

    sample_freq = np.empty((symbols.size, 66), dtype=np.int32)
    for leaf_i, cluster in enumerate(cluster_of):
        mask = leaf_ids == leaf_i
        if np.any(mask):
            sample_freq[mask] = freq_rows_c[cluster][0]
    sample_cum = np.zeros((symbols.size, 67), dtype=np.int32)
    sample_cum[:, 1:] = np.cumsum(sample_freq, axis=1, dtype=np.int64).astype(np.int32)

    writer = BitWriter()
    writer.write_bits(len(reps), 6)
    writer.write_bits(n_leaves, 8)
    for cluster in cluster_of:
        writer.write_bits(cluster, 6)
    nu = int(nu_idx) & 3
    for emp, sigma in freq_rows_c:
        writer.write_bits(int(sigma) & 31, 5)
        prior = freqs_tbl[nu, int(sigma) & 31, 8]
        for bin_i in range(66):
            writer.write_signed_golomb(int(emp[bin_i]) - int(prior[bin_i]))
    header = writer.finish()
    # Cheap reject: header bigger than ~0.05 bit/sample is left to the caller,
    # but an empty or broken partition is refused here.
    if int(sample_freq.sum()) != symbols.size * RANS_M:
        return None
    return sample_freq, sample_cum, header


def read_histograms(
    reader: BitReader,
    leaf_ids: np.ndarray,
    nu_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of the histogram header written by :func:`clustered_histograms`."""
    n_clusters = reader.read_bits(6)
    n_leaves = reader.read_bits(8)
    if n_clusters <= 0 or n_clusters > 32 or n_leaves <= 0 or n_leaves > 64:
        raise CodecError("MANIAC histogram header is out of range")
    cluster_of = [reader.read_bits(6) for _ in range(n_leaves)]
    if any(c >= n_clusters for c in cluster_of):
        raise CodecError("MANIAC cluster index out of range")
    _shifts, freqs_tbl, _cums, _tilt = get_parametric_tables()
    clusters = []
    nu = int(nu_idx) & 3
    for _ in range(n_clusters):
        sigma = reader.read_bits(5)
        if sigma >= NUM_SIGMAS:
            raise CodecError("histogram prior sigma out of range")
        prior = freqs_tbl[nu, sigma, 8].astype(np.int32)
        emp = np.empty(66, dtype=np.int32)
        for bin_i in range(66):
            emp[bin_i] = int(prior[bin_i]) + reader.read_signed_golomb()
        if int(emp.min()) < 1 or int(emp.sum()) != RANS_M:
            raise CodecError("histogram row is not a Core66 partition")
        clusters.append(emp)
    freq = np.empty((leaf_ids.size, 66), dtype=np.int32)
    for leaf_i, cluster in enumerate(cluster_of):
        mask = leaf_ids == leaf_i
        if np.any(mask):
            freq[mask] = clusters[cluster]
    if np.any(leaf_ids < 0) or np.any(leaf_ids >= n_leaves):
        raise CodecError("MANIAC leaf id out of range")
    cum = np.zeros((leaf_ids.size, 67), dtype=np.int32)
    cum[:, 1:] = np.cumsum(freq, axis=1, dtype=np.int64).astype(np.int32)
    return freq, cum
