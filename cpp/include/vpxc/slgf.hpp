#pragma once

// Version-2 SLGF tile coder (src/vpxc/slgf_model.py) plus the shared integer
// feature stage used by the version-3 tile coder.

#include "vpxc/error.hpp"
#include "vpxc/image.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#if __has_include("vpxc/lattice.hpp")
#include "vpxc/lattice.hpp"
#define VPXC_HAS_LATTICE 1
#endif

#if __has_include("vpxc/issm.hpp")
#include "vpxc/issm.hpp"
#define VPXC_HAS_ISSM 1
#endif

#if __has_include("vpxc/spatial.hpp")
#include "vpxc/spatial.hpp"
#define VPXC_HAS_SPATIAL 1
#endif

#if __has_include("vpxc/entropy.hpp")
#include "vpxc/entropy.hpp"
#define VPXC_HAS_ENTROPY 1
#endif

namespace vpxc {

inline constexpr int kMaxBuckets = 16;

#ifndef VPXC_HAS_LATTICE
// lattice.hpp is not in the tree yet. These are the calls slgf/codec make.
struct PackInfo {
  int u_shift = 0;
  int v_shift = 0;
  int parent_height = 0;
  int parent_width = 0;
  int packed_height = 0;
  int packed_width = 0;
};

struct LevelGeom {
  Mask valid;
  PackInfo info;
  Mask packed_valid;
};

struct LevelBuild {
  std::vector<LevelGeom> levels;
  Mask coarse_valid;
};

struct Tower {
  Plane coarse;
  Mask coarse_valid;
  std::vector<Plane> details;
  std::vector<LevelGeom> geom;
};

// odd, even. Both are subsets of valid: (r + c) odd / even.
std::pair<Mask, Mask> parity_masks(const Mask& valid);
Plane predict(const Plane& img, const Mask& valid, bool sheaf);
Plane lift_inverse(const Plane& lifted, const Mask& valid, bool sheaf);
Plane unpack_even(const Plane& packed, const PackInfo& info, const Mask& parent_valid);
LevelBuild build_levels(int height, int width, int max_levels);
Tower forward_tower(const Plane& plane, int max_levels, bool sheaf);
// Per-pixel activity bucket in 0..7, length height*width, row-major.
std::vector<uint8_t> activity_context(const Plane& lowpass, const Mask& valid);
#endif

#ifndef VPXC_HAS_ISSM
// scan_4way returns Q4 states (h_east, h_west, h_south, h_north).
struct Scan4 {
  std::vector<int64_t> east;
  std::vector<int64_t> west;
  std::vector<int64_t> south;
  std::vector<int64_t> north;
};
Scan4 scan_4way(const int64_t* samples, int height, int width, int base_decay, int edge_kappa);
#endif

#ifndef VPXC_HAS_SPATIAL
std::vector<uint8_t> encode_raster(const Plane& plane, const Mask& valid, bool kalman);
Plane decode_raster(const uint8_t* data, size_t size, int height, int width, const Mask& valid,
                    bool kalman);
#endif

#ifndef VPXC_HAS_ENTROPY
// Parametric Core66 tables. freqs is row-major [nu][sigma][frac][66],
// cums is [nu][sigma][frac][67], shifts is [32]. Pointers stay valid.
struct ParametricTables {
  const int32_t* shifts = nullptr;
  const int32_t* freqs = nullptr;
  const int32_t* cums = nullptr;
  const int32_t* tilt = nullptr;
};
ParametricTables get_parametric_tables();

// Bridge and external rows are optional (null means "absent", matching Python None).
// ext_freq is n*66, ext_cum is n*67. bridge_* are length n.
std::vector<uint8_t> encode_parametric_rans(const int32_t* r_tilde, size_t n, int nu_idx,
                                            const int32_t* sig_idx, const int32_t* frac_idx,
                                            const uint8_t* bridge_mask = nullptr,
                                            const int32_t* bridge_lo = nullptr,
                                            const int32_t* bridge_hi = nullptr,
                                            const int32_t* ext_freq = nullptr,
                                            const int32_t* ext_cum = nullptr);

std::vector<int32_t> decode_parametric_rans(const uint8_t* data, size_t size, int count, int nu_idx,
                                            const int32_t* sig_idx, const int32_t* frac_idx,
                                            const uint8_t* bridge_mask = nullptr,
                                            const int32_t* bridge_lo = nullptr,
                                            const int32_t* bridge_hi = nullptr,
                                            const int32_t* ext_freq = nullptr,
                                            const int32_t* ext_cum = nullptr);
#endif

// Full-grid Stage-1/2 features. Every vector is height*width except act_shift.
struct LevelFeatures {
  int height = 0;
  int width = 0;
  std::vector<int64_t> mu0_q4;
  std::vector<int64_t> f_sheaf_q4;
  std::vector<int64_t> f_cub_v_q4;
  std::vector<int64_t> f_cub_h_q4;
  std::vector<int64_t> f_ring5_q4;
  std::vector<int64_t> f_ssm_h_q4;
  std::vector<int64_t> f_ssm_v_q4;
  std::vector<int64_t> n_minus_s;
  std::vector<int64_t> w_minus_e;
  std::vector<int64_t> wv;
  std::vector<int64_t> wh;
  std::vector<int64_t> curv_q4;
  std::vector<int64_t> lo_nb;
  std::vector<int64_t> hi_nb;
  std::vector<uint8_t> is_bridge;
  std::vector<int64_t> flat_pred;
  int act_shift = 0;
};

struct FeatureRows {
  int rows = 0;
  int cols = 0;
  std::vector<int64_t> phi;  // row-major rows*cols, Q4
  std::vector<int64_t> act_q4;
};

struct SubcosetEstimate {
  bool use_ridge = false;
  std::vector<int64_t> w_q8;
  int nu_idx = 1;
  std::vector<int32_t> sig_map;  // kMaxBuckets
};

struct SigmaNuFit {
  int nu_idx = 1;
  std::vector<int32_t> sig_map;
};

inline std::vector<int> true_sites(const Mask& mask) {
  std::vector<int> sites;
  sites.reserve(mask.bits.size() / 2 + 1);
  for (int i = 0; i < static_cast<int>(mask.bits.size()); ++i) {
    if (mask.bits[static_cast<size_t>(i)]) sites.push_back(i);
  }
  return sites;
}

LevelFeatures level_geometry_features(const Plane& spatial_even, const Mask& valid);
std::vector<Mask> subcoset_masks(const Mask& valid);
FeatureRows assemble_features_and_activity(const LevelFeatures& base, int sub_idx,
                                           const std::vector<int>& sites,
                                           const std::vector<int64_t>& decoded_odd,
                                           const std::vector<int64_t>* coarse_proj,
                                           const Plane* ch0_spatial, const Plane* ch0_resid,
                                           const Plane* ch1_resid);
void split_mu_q4(const int64_t* mu_q4, size_t n, int64_t* mu_int, int32_t* frac_idx);
void bucket_from_act_q4(const int64_t* act_q4, size_t n, int act_shift, int32_t* buckets);
std::optional<std::vector<int64_t>> coarse_resid_projection(const Plane* prev_resid,
                                                            const PackInfo& info, const Mask& valid);
// Encode-only. early_below is 64 for the v2 head and 48 for fit_sigma_nu.
SigmaNuFit select_sigma_nu(const int64_t* r_tilde, const int32_t* frac_idx, const int32_t* buckets,
                           size_t n, int n_buckets, int early_below);
SubcosetEstimate estimate_subcoset_model(const int32_t* r_target, const int64_t* mu0_q4,
                                         const int64_t* phi, int rows, int cols,
                                         const int32_t* buckets);

std::vector<uint8_t> encode_slgf_tile(const std::vector<Plane>& planes, int max_levels = 4);
std::vector<Plane> decode_slgf_tile(const uint8_t* data, size_t size, int height, int width,
                                    int channels);

}  // namespace vpxc
