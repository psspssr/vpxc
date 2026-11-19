#pragma once

// Version-3 SLGF tile coder (src/vpxc/slgf_v3.py).

#include "vpxc/bitio.hpp"
#include "vpxc/slgf.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#if __has_include("vpxc/acquisition.hpp")
#include "vpxc/acquisition.hpp"
#define VPXC_HAS_ACQUISITION 1
#endif

#if __has_include("vpxc/matching.hpp")
#include "vpxc/matching.hpp"
#define VPXC_HAS_MATCHING 1
#endif

#if __has_include("vpxc/coarse.hpp")
#include "vpxc/coarse.hpp"
#define VPXC_HAS_COARSE 1
#endif

#if __has_include("vpxc/trees.hpp")
#include "vpxc/trees.hpp"
#define VPXC_HAS_TREES 1
#endif

#if __has_include("vpxc/bitsback.hpp")
#include "vpxc/bitsback.hpp"
#define VPXC_HAS_BITSBACK 1
#endif

#if __has_include("vpxc/issm_int8.hpp")
#include "vpxc/issm_int8.hpp"
#define VPXC_HAS_ISSM_INT8 1
#endif

namespace vpxc {

// Causal even halo around a tile: top strip, left strip, and the corner.
struct Halo {
  std::optional<Plane> top;
  std::optional<Plane> left;
  std::optional<Plane> corner;
};

#ifndef VPXC_HAS_ACQUISITION
struct PhiAcq {
  int shift = 0;
  int64_t knots[8] = {};
  int32_t taps[4][24] = {};
};
std::optional<PhiAcq> fit_phi(const Plane& luma, const Plane& chroma, const Plane& flat_pred,
                              const Mask& odd);
std::vector<int64_t> apply_phi(const Plane& luma, const PhiAcq& phi);
void write_phi(BitWriter& writer, const PhiAcq& phi);
PhiAcq read_phi(BitReader& reader);
#endif

#ifndef VPXC_HAS_MATCHING
// Full-grid correction (match - flat_pred). Zero where the gate fails.
std::vector<int64_t> sheaf_kalman_delta(const int64_t* samples, int height, int width,
                                        const Mask& known, const Mask& query,
                                        const int64_t* flat_pred, int radius = 128);
#endif

#ifndef VPXC_HAS_COARSE
std::vector<uint8_t> encode_coarse(const Plane& plane, const Mask& valid, const Plane* ref);
Plane decode_coarse(const uint8_t* data, size_t size, int height, int width, const Mask& valid,
                    const Plane* ref);
#endif

#ifndef VPXC_HAS_TREES
struct TreeNode {
  std::optional<int> sigma;
  std::optional<int> prop;
  std::optional<int> thr;
  std::unique_ptr<TreeNode> left;
  std::unique_ptr<TreeNode> right;
  int leaf_id = -1;
  bool is_leaf() const { return !left || !right; }
};

struct TreeMap {
  std::vector<int32_t> sigma;
  std::vector<int32_t> leaf;
};

struct ClusterHist {
  std::vector<int32_t> freq;  // rows * 66
  std::vector<int32_t> cum;   // rows * 67
  std::vector<uint8_t> header;
};

// raw is rows*cols int64. Result is rows*cols int16, row-major.
std::vector<int16_t> quantize_props(const int64_t* raw, int rows, int cols);
std::unique_ptr<TreeNode> build_sigma_tree(const int16_t* qprops, int rows, int cols,
                                           const int32_t* residual);
TreeMap apply_tree(const TreeNode& node, const int16_t* qprops, int rows, int cols);
void write_tree(BitWriter& writer, const TreeNode& node);
std::unique_ptr<TreeNode> read_tree(BitReader& reader);
void number_tree(TreeNode& node);
std::optional<ClusterHist> clustered_histograms(const int16_t* qprops, int rows, int cols,
                                                const TreeNode& tree, const int32_t* residual,
                                                int nu_idx);
std::pair<std::vector<int32_t>, std::vector<int32_t>> read_histograms(BitReader& reader,
                                                                      const int32_t* leaf_ids,
                                                                      size_t n, int nu_idx);
#endif

#ifndef VPXC_HAS_BITSBACK
std::optional<std::vector<uint8_t>> encode_s3(const int32_t* residual, size_t n, const int32_t* sig_idx,
                                              const int32_t* frac_idx, int nu_idx,
                                              const uint8_t* seed_mask);
std::vector<int32_t> decode_s3(const uint8_t* data, size_t size, int count, const int32_t* sig_idx,
                               const int32_t* frac_idx, int nu_idx, const uint8_t* seed_mask);
#endif

#ifndef VPXC_HAS_ISSM_INT8
std::vector<int64_t> context_map(const int64_t* filled, int height, int width);
std::vector<int64_t> project_head(const int64_t* phi_q4, int rows, int cols);
#endif

// sheaf_mode: 0 flat lift, 1 sheaf-weighted lift at every quincunx level.
std::vector<uint8_t> encode_slgf_v3(const std::vector<Plane>& planes, int max_levels = 4, int effort = 2,
                                    const std::vector<Halo>* halos = nullptr, bool use_static = true,
                                    int sheaf_mode = 0);
std::vector<Plane> decode_slgf_v3(const uint8_t* data, size_t size, int height, int width, int channels,
                                  const std::vector<Halo>* halos = nullptr);

}  // namespace vpxc
