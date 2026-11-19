#include "vpxc/trees.hpp"

#include "vpxc/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace vpxc {
namespace {

constexpr int kMaxDepth = 4;
constexpr int kMinLeaf = 40;
constexpr int kMaxLeaves = 16;
constexpr double kSplitGain = 24.0;

int64_t iabs(int64_t v) {
  if (v >= 0) return v;
  if (v == INT64_MIN) return INT64_MAX;
  return -v;
}

int ilog2_abs(int64_t value) {
  uint64_t mag = 1;
  if (value > 0) mag = static_cast<uint64_t>(value);
  else if (value < 0 && value != INT64_MIN) mag = static_cast<uint64_t>(-value);
  else if (value == INT64_MIN) mag = 1ull << 63;
  return 63 - __builtin_clzll(mag);
}

const int32_t* freq_row(int nu, int sigma, int frac) { return get_parametric_tables().freqs[nu][sigma][frac]; }

double mad_bits(int64_t abs_sum, int n) {
  if (n <= 0) return 0.0;
  const double mad = static_cast<double>(abs_sum / n + 1);
  return static_cast<double>(n) * std::log2(mad);
}

int sigma_from_sum(int64_t abs_sum, int n) {
  const int64_t mad = abs_sum / std::max(n, 1);
  const int64_t target = mad * 320 + 48;
  int best = 0;
  int64_t best_d = iabs(static_cast<int64_t>(SIGMA_Q8[0]) - target);
  for (int i = 1; i < NUM_SIGMAS; ++i) {
    const int64_t gap = iabs(static_cast<int64_t>(SIGMA_Q8[i]) - target);
    if (gap < best_d) {
      best_d = gap;
      best = i;
    }
  }
  return best;
}

int clip_sigma(int sigma) {
  if (sigma < 0) return 0;
  if (sigma >= NUM_SIGMAS) return NUM_SIGMAS - 1;
  return sigma;
}

int symbol_of(int64_t residual, int shift) {
  __int128 quot = residual;
  if (shift > 0) {
    const __int128 bias = __int128{1} << (shift - 1);
    quot = (static_cast<__int128>(residual) + bias) >> shift;
  }
  const __int128 mag = quot < 0 ? -quot : quot;
  if (mag > 32) return 65;
  return static_cast<int>(quot + 32);
}

void number_leaves(TreeNode& node, std::vector<TreeNode*>& acc) {
  if (node.is_leaf()) {
    node.leaf_id = static_cast<int>(acc.size());
    acc.push_back(&node);
    return;
  }
  number_leaves(*node.left, acc);
  number_leaves(*node.right, acc);
}

}  // namespace

TreeNode::~TreeNode() = default;

std::vector<int16_t> quantize_props(const int64_t* raw, size_t count) {
  std::vector<int16_t> out(count);
  if (count > 0 && raw == nullptr) throw CodecError("property buffer is empty");
  for (size_t i = 0; i < count; ++i) {
    const int64_t value = raw[i];
    int sign = value < 0 ? -1 : 1;
    if (value == 0) sign = 0;
    out[i] = static_cast<int16_t>(sign * ilog2_abs(value));
  }
  return out;
}

std::unique_ptr<TreeNode> build_sigma_tree(const int16_t* qprops, int rows, int n_prop, const int64_t* residual) {
  if (rows < 0 || n_prop < 0) throw CodecError("MANIAC property shape is invalid");
  if (rows < kMinLeaf * 2 || n_prop == 0) return nullptr;
  if (qprops == nullptr || residual == nullptr) throw CodecError("MANIAC property buffer is empty");
  std::vector<int64_t> abs_r(static_cast<size_t>(rows));
  for (int i = 0; i < rows; ++i) abs_r[static_cast<size_t>(i)] = iabs(residual[i]);

  const auto rec = [&](auto&& self, const std::vector<int>& idx, int depth, int& budget) -> std::unique_ptr<TreeNode> {
    auto node = std::make_unique<TreeNode>();
    int64_t sum = 0;
    for (int i : idx) sum += abs_r[static_cast<size_t>(i)];
    node->sigma = sigma_from_sum(sum, static_cast<int>(idx.size()));
    if (depth >= kMaxDepth || idx.size() < static_cast<size_t>(kMinLeaf * 2) || budget < 2) return node;
    const double base = mad_bits(sum, static_cast<int>(idx.size()));
    bool have = false;
    double best_gain = 0.0;
    int best_prop = 0;
    int best_thr = 0;
    for (int prop = 0; prop < n_prop; ++prop) {
      std::vector<int16_t> vals;
      vals.reserve(idx.size());
      for (int i : idx) vals.push_back(qprops[static_cast<size_t>(i) * static_cast<size_t>(n_prop) + prop]);
      std::sort(vals.begin(), vals.end());
      const int lo = vals.front();
      const int hi = vals.back();
      if (lo == hi) continue;
      int seen[3];
      int nseen = 0;
      const int nv = static_cast<int>(vals.size());
      for (int frac = 1; frac <= 3; ++frac) {
        const int thr = vals[static_cast<size_t>((nv * frac) / 4)];
        bool dup = false;
        for (int s = 0; s < nseen; ++s) {
          if (seen[s] == thr) dup = true;
        }
        if (dup || thr < lo || thr >= hi) continue;
        seen[nseen++] = thr;
        int64_t sum_left = 0;
        int n_left = 0;
        for (int i : idx) {
          if (qprops[static_cast<size_t>(i) * static_cast<size_t>(n_prop) + prop] <= thr) {
            sum_left += abs_r[static_cast<size_t>(i)];
            ++n_left;
          }
        }
        const int n_right = static_cast<int>(idx.size()) - n_left;
        if (n_left < kMinLeaf || n_right < kMinLeaf) continue;
        const double gain = base - mad_bits(sum_left, n_left) - mad_bits(sum - sum_left, n_right);
        if (!have || gain > best_gain) {
          have = true;
          best_gain = gain;
          best_prop = prop;
          best_thr = thr;
        }
      }
    }
    if (!have || best_gain < kSplitGain) return node;
    --budget;
    std::vector<int> left;
    std::vector<int> right;
    left.reserve(idx.size());
    right.reserve(idx.size());
    for (int i : idx) {
      if (qprops[static_cast<size_t>(i) * static_cast<size_t>(n_prop) + best_prop] <= best_thr) left.push_back(i);
      else right.push_back(i);
    }
    node->prop = best_prop;
    node->thr = best_thr;
    node->left = self(self, left, depth + 1, budget);
    node->right = self(self, right, depth + 1, budget);
    return node;
  };

  std::vector<int> all(static_cast<size_t>(rows));
  std::iota(all.begin(), all.end(), 0);
  int budget = kMaxLeaves;
  auto tree = rec(rec, all, 0, budget);
  if (!tree || tree->is_leaf()) return nullptr;
  number_tree(*tree);
  return tree;
}

TreeMap apply_tree(const TreeNode& node, const int16_t* qprops, int rows, int n_prop) {
  if (rows < 0 || n_prop < 0) throw CodecError("MANIAC property shape is invalid");
  TreeMap out;
  out.sigma.assign(static_cast<size_t>(std::max(rows, 0)), 0);
  out.leaf.assign(static_cast<size_t>(std::max(rows, 0)), -1);
  if (rows == 0) return out;
  if (qprops == nullptr) throw CodecError("MANIAC property buffer is empty");
  const auto rec = [&](auto&& self, const TreeNode& cur, const std::vector<int>& idx) -> void {
    if (idx.empty()) return;
    if (cur.is_leaf()) {
      for (int i : idx) {
        out.sigma[static_cast<size_t>(i)] = cur.sigma;
        out.leaf[static_cast<size_t>(i)] = cur.leaf_id;
      }
      return;
    }
    if (cur.prop < 0 || cur.prop >= n_prop) throw CodecError("MANIAC property index out of range");
    std::vector<int> left;
    std::vector<int> right;
    for (int i : idx) {
      if (qprops[static_cast<size_t>(i) * static_cast<size_t>(n_prop) + cur.prop] <= cur.thr) left.push_back(i);
      else right.push_back(i);
    }
    self(self, *cur.left, left);
    self(self, *cur.right, right);
  };
  std::vector<int> all(static_cast<size_t>(rows));
  std::iota(all.begin(), all.end(), 0);
  rec(rec, node, all);
  return out;
}

void write_tree(BitWriter& writer, const TreeNode& node) {
  if (node.is_leaf()) {
    writer.write_bits(1, 1);
    writer.write_bits(static_cast<uint32_t>(node.sigma) & 31u, 5);
    return;
  }
  writer.write_bits(0, 1);
  writer.write_bits(static_cast<uint32_t>(node.prop) & 15u, 4);
  writer.write_signed_golomb(node.thr);
  write_tree(writer, *node.left);
  write_tree(writer, *node.right);
}

std::unique_ptr<TreeNode> read_tree(BitReader& reader, int depth) {
  if (depth > 16) throw CodecError("MANIAC tree too deep");
  auto node = std::make_unique<TreeNode>();
  if (reader.read_bits(1) == 1) {
    node->sigma = static_cast<int>(reader.read_bits(5));
    return node;
  }
  node->prop = static_cast<int>(reader.read_bits(4));
  node->thr = reader.read_signed_golomb();
  node->left = read_tree(reader, depth + 1);
  node->right = read_tree(reader, depth + 1);
  return node;
}

std::vector<TreeNode*> number_tree(TreeNode& node) {
  std::vector<TreeNode*> leaves;
  number_leaves(node, leaves);
  return leaves;
}

std::optional<ClusteredHist> clustered_histograms(const int16_t* qprops, int rows, int n_prop, TreeNode& tree,
                                                  const int64_t* residual, int nu_idx, int max_clusters) {
  if (rows < 0) throw CodecError("MANIAC property shape is invalid");
  TreeMap mapped = apply_tree(tree, qprops, rows, n_prop);
  const std::vector<TreeNode*> leaves = number_tree(tree);
  const int n_leaves = static_cast<int>(leaves.size());
  if (n_leaves < 2) return std::nullopt;
  if (residual == nullptr) throw CodecError("MANIAC residual buffer is empty");
  const int32_t* shifts = get_parametric_tables().shifts;
  std::vector<int> symbols(static_cast<size_t>(rows));
  for (int i = 0; i < rows; ++i) {
    const int sigma = clip_sigma(mapped.sigma[static_cast<size_t>(i)]);
    symbols[static_cast<size_t>(i)] = symbol_of(residual[i], shifts[sigma]);
  }
  std::vector<std::vector<int64_t>> counts(static_cast<size_t>(n_leaves), std::vector<int64_t>(66, 0));
  std::vector<int> sigmas(static_cast<size_t>(n_leaves), 0);
  for (int leaf = 0; leaf < n_leaves; ++leaf) sigmas[static_cast<size_t>(leaf)] = leaves[static_cast<size_t>(leaf)]->sigma;
  for (int i = 0; i < rows; ++i) {
    const int leaf = mapped.leaf[static_cast<size_t>(i)];
    if (leaf < 0 || leaf >= n_leaves) continue;
    const int sym = symbols[static_cast<size_t>(i)];
    if (sym >= 0 && sym < 66) ++counts[static_cast<size_t>(leaf)][static_cast<size_t>(sym)];
  }
  std::vector<int> active(static_cast<size_t>(n_leaves));
  std::iota(active.begin(), active.end(), 0);
  std::vector<int> parent(static_cast<size_t>(n_leaves));
  std::iota(parent.begin(), parent.end(), 0);
  while (static_cast<int>(active.size()) > max_clusters) {
    bool have = false;
    __int128 best_dist = 0;
    int best_a = 0;
    int best_b = 0;
    for (size_t ia = 0; ia < active.size(); ++ia) {
      for (size_t ib = ia + 1; ib < active.size(); ++ib) {
        const int a = active[ia];
        const int b = active[ib];
        int64_t sa = 66;
        int64_t sb = 66;
        for (int bin = 0; bin < 66; ++bin) {
          sa += counts[static_cast<size_t>(a)][static_cast<size_t>(bin)];
          sb += counts[static_cast<size_t>(b)][static_cast<size_t>(bin)];
        }
        __int128 dist = 0;
        for (int bin = 0; bin < 66; ++bin) {
          const __int128 term = (static_cast<__int128>(counts[static_cast<size_t>(a)][static_cast<size_t>(bin)]) + 1) * sb -
                                (static_cast<__int128>(counts[static_cast<size_t>(b)][static_cast<size_t>(bin)]) + 1) * sa;
          dist += term < 0 ? -term : term;
        }
        dist /= sa;
        if (!have || dist < best_dist) {
          have = true;
          best_dist = dist;
          best_a = a;
          best_b = b;
        }
      }
    }
    if (!have) throw CodecError("histogram clustering failed");
    for (int bin = 0; bin < 66; ++bin) {
      counts[static_cast<size_t>(best_a)][static_cast<size_t>(bin)] +=
          counts[static_cast<size_t>(best_b)][static_cast<size_t>(bin)];
    }
    parent[static_cast<size_t>(best_b)] = best_a;
    active.erase(std::find(active.begin(), active.end(), best_b));
  }

  std::vector<int> reps;
  std::vector<int> rep_of(static_cast<size_t>(n_leaves), -1);
  std::vector<int> cluster_of(static_cast<size_t>(n_leaves), 0);
  for (int leaf = 0; leaf < n_leaves; ++leaf) {
    int root = leaf;
    while (parent[static_cast<size_t>(root)] != root) root = parent[static_cast<size_t>(root)];
    if (rep_of[static_cast<size_t>(root)] < 0) {
      rep_of[static_cast<size_t>(root)] = static_cast<int>(reps.size());
      reps.push_back(root);
    }
    cluster_of[static_cast<size_t>(leaf)] = rep_of[static_cast<size_t>(root)];
  }

  std::vector<std::vector<int32_t>> emp(reps.size());
  std::vector<int> emp_sigma(reps.size(), 0);
  for (size_t c = 0; c < reps.size(); ++c) {
    emp[c] = normalize_freqs(std::span<const int64_t>(counts[static_cast<size_t>(reps[c])].data(), 66), RANS_M);
    emp_sigma[c] = sigmas[static_cast<size_t>(reps[c])];
  }
  ClusteredHist packed;
  packed.freq.assign(static_cast<size_t>(rows) * 66, 0);
  packed.cum.assign(static_cast<size_t>(rows) * 67, 0);
  for (int i = 0; i < rows; ++i) {
    const int leaf = mapped.leaf[static_cast<size_t>(i)];
    if (leaf < 0 || leaf >= n_leaves) continue;
    const int cluster = cluster_of[static_cast<size_t>(leaf)];
    int32_t run = 0;
    for (int bin = 0; bin < 66; ++bin) {
      const int32_t value = emp[static_cast<size_t>(cluster)][static_cast<size_t>(bin)];
      packed.freq[static_cast<size_t>(i) * 66 + bin] = value;
      run += value;
      packed.cum[static_cast<size_t>(i) * 67 + 1 + bin] = run;
    }
  }
  int64_t total = 0;
  for (int32_t value : packed.freq) total += value;
  if (total != static_cast<int64_t>(rows) * RANS_M) return std::nullopt;

  BitWriter writer;
  writer.write_bits(static_cast<uint32_t>(reps.size()), 6);
  writer.write_bits(static_cast<uint32_t>(n_leaves), 8);
  for (int cluster : cluster_of) writer.write_bits(static_cast<uint32_t>(cluster), 6);
  const int nu = nu_idx & 3;
  for (size_t c = 0; c < reps.size(); ++c) {
    const int sigma = emp_sigma[c] & 31;
    writer.write_bits(static_cast<uint32_t>(sigma), 5);
    const int32_t* prior = freq_row(nu, sigma, 8);
    for (int bin = 0; bin < 66; ++bin) {
      const int delta = emp[c][static_cast<size_t>(bin)] - prior[bin];
      writer.write_signed_golomb(delta);
    }
  }
  packed.header = writer.finish();
  return packed;
}

HistogramTables read_histograms(BitReader& reader, const int32_t* leaf_ids, int count, int nu_idx) {
  if (count < 0) throw CodecError("MANIAC histogram header is out of range");
  if (count > 0 && leaf_ids == nullptr) throw CodecError("MANIAC leaf id out of range");
  const int n_clusters = static_cast<int>(reader.read_bits(6));
  const int n_leaves = static_cast<int>(reader.read_bits(8));
  if (n_clusters <= 0 || n_clusters > 32 || n_leaves <= 0 || n_leaves > 64) {
    throw CodecError("MANIAC histogram header is out of range");
  }
  std::vector<int> cluster_of(static_cast<size_t>(n_leaves));
  for (int i = 0; i < n_leaves; ++i) {
    cluster_of[static_cast<size_t>(i)] = static_cast<int>(reader.read_bits(6));
    if (cluster_of[static_cast<size_t>(i)] >= n_clusters) throw CodecError("MANIAC cluster index out of range");
  }
  const int nu = nu_idx & 3;
  std::vector<std::vector<int32_t>> clusters(static_cast<size_t>(n_clusters));
  for (int c = 0; c < n_clusters; ++c) {
    const int sigma = static_cast<int>(reader.read_bits(5));
    if (sigma >= NUM_SIGMAS) throw CodecError("histogram prior sigma out of range");
    const int32_t* prior = freq_row(nu, sigma, 8);
    std::vector<int32_t> emp(66);
    int64_t sum = 0;
    int32_t lo = 0;
    for (int bin = 0; bin < 66; ++bin) {
      const int64_t value = static_cast<int64_t>(prior[bin]) + reader.read_signed_golomb();
      emp[static_cast<size_t>(bin)] = static_cast<int32_t>(value);
      sum += emp[static_cast<size_t>(bin)];
      if (bin == 0 || emp[static_cast<size_t>(bin)] < lo) lo = emp[static_cast<size_t>(bin)];
    }
    if (lo < 1 || sum != RANS_M) throw CodecError("histogram row is not a Core66 partition");
    clusters[static_cast<size_t>(c)] = std::move(emp);
  }
  HistogramTables out;
  out.freq.assign(static_cast<size_t>(count) * 66, 0);
  out.cum.assign(static_cast<size_t>(count) * 67, 0);
  for (int leaf = 0; leaf < n_leaves; ++leaf) {
    for (int i = 0; i < count; ++i) {
      if (leaf_ids[i] != leaf) continue;
      int32_t run = 0;
      for (int bin = 0; bin < 66; ++bin) {
        const int32_t value = clusters[static_cast<size_t>(cluster_of[static_cast<size_t>(leaf)])][static_cast<size_t>(bin)];
        out.freq[static_cast<size_t>(i) * 66 + bin] = value;
        run += value;
        out.cum[static_cast<size_t>(i) * 67 + 1 + bin] = run;
      }
    }
  }
  for (int i = 0; i < count; ++i) {
    if (leaf_ids[i] < 0 || leaf_ids[i] >= n_leaves) throw CodecError("MANIAC leaf id out of range");
  }
  return out;
}

}  // namespace vpxc
