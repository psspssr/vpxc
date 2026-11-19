#pragma once

#include <cstdint>
#include <vector>

namespace vpxc {

int zigzag(int err);
int unzigzag(int symbol);

class BitWriter {
 public:
  void write_bits(uint32_t value, int width);
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
  explicit BitReader(const uint8_t* data, size_t size);
  explicit BitReader(const std::vector<uint8_t>& data) : BitReader(data.data(), data.size()) {}

  uint32_t read_bits(int width);
  int read_rice(int k);
  uint32_t read_exp_golomb();
  int read_signed_golomb();

 private:
  const uint8_t* data_;
  size_t size_;
  size_t pos_ = 0;
  uint64_t buf_ = 0;
  int n_ = 0;
};

}  // namespace vpxc
