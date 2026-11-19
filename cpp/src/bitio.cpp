#include "vpxc/bitio.hpp"

#include "vpxc/error.hpp"

namespace vpxc {
namespace {
constexpr int kRiceEscape = 32;
}

int zigzag(int err) {
  if (err < 0) return ((-err) << 1) - 1;
  return err << 1;
}

int unzigzag(int symbol) {
  if (symbol & 1) return -((symbol + 1) >> 1);
  return symbol >> 1;
}

void BitWriter::write_bits(uint32_t value, int width) {
  if (width < 0) throw CodecError("negative bit width");
  if (width == 0) return;
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

void BitWriter::write_exp_golomb(uint32_t value) {
  const uint32_t code = value + 1;
  int zeros = 0;
  for (uint32_t c = code; c > 1; c >>= 1) ++zeros;
  if (zeros) write_bits(0, zeros);
  write_bits(code, zeros + 1);
}

void BitWriter::write_signed_golomb(int value) { write_exp_golomb(static_cast<uint32_t>(zigzag(value))); }

void BitWriter::write_rice(int symbol, int k) {
  if (symbol < 0) throw CodecError("rice symbol must be non-negative");
  const int quotient = k ? (symbol >> k) : symbol;
  if (quotient < kRiceEscape) {
    if (quotient) write_bits((1u << quotient) - 1u, quotient);
    write_bits(0, 1);
    if (k) write_bits(static_cast<uint32_t>(symbol & ((1 << k) - 1)), k);
  } else {
    write_bits(0xffffffffu, kRiceEscape);
    write_bits(0, 1);
    write_exp_golomb(static_cast<uint32_t>(symbol));
  }
}

std::vector<uint8_t> BitWriter::finish() {
  if (n_) {
    out_.push_back(static_cast<uint8_t>((buf_ << (8 - n_)) & 0xff));
    n_ = 0;
    buf_ = 0;
  }
  return out_;
}

BitReader::BitReader(const uint8_t* data, size_t size) : data_(data), size_(size) {}

uint32_t BitReader::read_bits(int width) {
  if (width < 0) throw CodecError("negative bit width");
  if (width == 0) return 0;
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

uint32_t BitReader::read_exp_golomb() {
  int zeros = 0;
  while (read_bits(1) == 0) {
    ++zeros;
    if (zeros > 31) throw CodecError("exp-golomb overflow");
  }
  if (zeros == 0) return 0;
  const uint32_t rest = read_bits(zeros);
  return ((1u << zeros) | rest) - 1u;
}

int BitReader::read_signed_golomb() {
  // Zigzag symbols are unsigned. Casting a symbol above 2^31-1 to int wraps
  // and then overflows on symbol+1. The signed result of every uint32 zigzag
  // code still fits in int32.
  const uint32_t symbol = read_exp_golomb();
  const int64_t value = (symbol & 1u)
                            ? -static_cast<int64_t>((static_cast<uint64_t>(symbol) + 1ull) >> 1)
                            : static_cast<int64_t>(symbol >> 1);
  return static_cast<int>(value);
}

int BitReader::read_rice(int k) {
  int quotient = 0;
  while (quotient < kRiceEscape) {
    if (read_bits(1) == 0) {
      const uint32_t rem = k ? read_bits(k) : 0;
      return (quotient << k) | static_cast<int>(rem);
    }
    ++quotient;
  }
  if (read_bits(1) != 0) throw CodecError("bad rice escape");
  return static_cast<int>(read_exp_golomb());
}

}  // namespace vpxc
