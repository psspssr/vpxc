#pragma once

#include "vpxc/error.hpp"

#include <cstdint>
#include <vector>

namespace vpxc {

int zigzag(int err);
int unzigzag(int symbol);

class BitWriter {
 public:
  inline void write_bits(uint32_t value, int width) {
    if (width <= 0) {
      if (width < 0) throw CodecError("negative bit width");
      return;
    }
    const uint32_t mask = width == 32 ? 0xffffffffu : ((1u << width) - 1u);
    buf_ = (buf_ << width) | (value & mask);
    n_ += width;
    while (n_ >= 8) {
      n_ -= 8;
      out_.push_back(static_cast<uint8_t>((buf_ >> n_) & 0xff));
      if (n_) buf_ &= (1ull << n_) - 1ull;
      else buf_ = 0;
    }
  }
  void write_rice(int symbol, int k);
  void write_exp_golomb(uint32_t value);
  void write_signed_golomb(int value);
  std::vector<uint8_t> finish();

 private:
  uint64_t buf_ = 0;
  int n_ = 0;
  std::vector<uint8_t> out_;
};

class BitReader {
 public:
  explicit BitReader(const uint8_t* data, size_t size) : data_(data), size_(size) {}
  explicit BitReader(const std::vector<uint8_t>& data) : BitReader(data.data(), data.size()) {}

  inline uint32_t read_bits(int width) {
    if (width <= 0) {
      if (width < 0) throw CodecError("negative bit width");
      return 0;
    }
    while (n_ < width) {
      if (pos_ >= size_) throw CodecError("truncated bitstream");
      buf_ = (buf_ << 8) | data_[pos_++];
      n_ += 8;
    }
    n_ -= width;
    const uint32_t value = static_cast<uint32_t>((buf_ >> n_) & ((width == 32) ? 0xffffffffu : ((1u << width) - 1u)));
    if (n_) buf_ &= (1ull << n_) - 1ull;
    else buf_ = 0;
    return value;
  }
  int read_rice(int k);
  uint32_t read_exp_golomb();
  int read_signed_golomb();
  void check_zero_tail() const;

 private:
  const uint8_t* data_;
  size_t size_;
  size_t pos_ = 0;
  uint64_t buf_ = 0;
  int n_ = 0;
};

}  // namespace vpxc
