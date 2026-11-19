#pragma once

#include "vpxc/bitio.hpp"
#include "vpxc/entropy.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace vpxc {

// MANIAC node. A node is a leaf when either child is absent.
// sigma is the leaf scale index. prop/thr are the integer split.
struct TreeNode {
  int sigma = 0;
  int prop = 0;
  int thr = 0;
  int leaf_id = -1;
  std::unique_ptr<TreeNode> left;
  std::unique_ptr<TreeNode> right;

  bool is_leaf() const { return !left || !right; }

  TreeNode() = default;
  TreeNode(const TreeNode&) = delete;
  TreeNode& operator=(const TreeNode&) = delete;
  TreeNode(TreeNode&&) noexcept = default;
  TreeNode& operator=(TreeNode&&) noexcept = default;
  ~TreeNode();
};

// Signed integer log-magnitude. raw has count elements; the result is the same length.
std::vector<int16_t> quantize_props(const int64_t* raw, size_t count);

// Greedy MDL tree. Null when no split beats one leaf.
// qprops is rows x n_prop, row-major. residual has rows elements.
std::unique_ptr<TreeNode> build_sigma_tree(const int16_t* qprops, int rows, int n_prop,
                                           const int64_t* residual);

struct TreeMap {
  std::vector<int32_t> sigma;
  std::vector<int32_t> leaf;
};

// Integer walk. leaf ids are whatever number_tree last wrote.
TreeMap apply_tree(const TreeNode& node, const int16_t* qprops, int rows, int n_prop);

void write_tree(BitWriter& writer, const TreeNode& node);
std::unique_ptr<TreeNode> read_tree(BitReader& reader, int depth = 0);

// Depth-first leaf numbering. Pointers remain valid while node is alive.
std::vector<TreeNode*> number_tree(TreeNode& node);

struct ClusteredHist {
  std::vector<int32_t> freq;  // rows * 66
  std::vector<int32_t> cum;   // rows * 67
  std::vector<uint8_t> header;
};

// Agglomerative merge of leaf Core66 histograms. Null when the partition cannot be used.
std::optional<ClusteredHist> clustered_histograms(const int16_t* qprops, int rows, int n_prop, TreeNode& tree,
                                                  const int64_t* residual, int nu_idx, int max_clusters = 8);

struct HistogramTables {
  std::vector<int32_t> freq;
  std::vector<int32_t> cum;
};

HistogramTables read_histograms(BitReader& reader, const int32_t* leaf_ids, int count, int nu_idx);

}  // namespace vpxc
