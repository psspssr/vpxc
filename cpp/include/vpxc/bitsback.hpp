#pragma once

#include "vpxc/entropy.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace vpxc {

// S^3 bits-back coder. nullopt when the seed cannot carry the latent.
// seed is nonzero on seed sites. Decode returns one int32 residual per site.
std::optional<std::vector<uint8_t>> encode_s3(const int64_t* residual, const int32_t* sig,
                                              const int32_t* frac, int nu, const uint8_t* seed,
                                              size_t count);

std::vector<int32_t> decode_s3(const uint8_t* data, size_t size, int count, const int32_t* sig,
                               const int32_t* frac, int nu, const uint8_t* seed);

}  // namespace vpxc
