#include "vpxc/bitsback.hpp"

#include "vpxc/bitio.hpp"
#include "vpxc/error.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace vpxc {
namespace {

int64_t iabs(int64_t v) {
  if (v >= 0) return v;
  if (v == INT64_MIN) return INT64_MAX;
  return -v;
}

const int32_t* freq_row(int nu, int sigma, int frac) { return get_parametric_tables().freqs[nu][sigma][frac]; }

const int32_t* cum_row(int nu, int sigma, int frac) { return get_parametric_tables().cums[nu][sigma][frac]; }

const std::vector<int32_t>& prior_freqs() {
  static const std::vector<int32_t> prior = [] {
    const int64_t counts[4] = {2, 8, 2, 1};
    return normalize_freqs(std::span<const int64_t>(counts, 4));
  }();
  return prior;
}

std::vector<int32_t> posterior_freqs(int64_t abs_sum, int count) {
  const int64_t avg = abs_sum / std::max(count, 1);
  int hat = 3;
  if (avg < 2) hat = 0;
  else if (avg < 6) hat = 1;
  else if (avg < 16) hat = 2;
  int64_t counts[4] = {1, 1, 1, 1};
  counts[hat] = 12;
  return normalize_freqs(std::span<const int64_t>(counts, 4));
}

int clip_sigma(int value) {
  if (value < 0) return 0;
  if (value >= NUM_SIGMAS) return NUM_SIGMAS - 1;
  return value;
}

struct Sym {
  int start = 0;
  int freq = 0;
  int shift = 0;
  int rem = 0;
  bool escaped = false;
  int64_t quot = 0;
};

void split_dyadic(int64_t residual, int shift, int64_t& quot, int64_t& rem) {
  if (shift <= 0) {
    quot = residual;
    rem = 0;
    return;
  }
  const __int128 bias = __int128{1} << (shift - 1);
  const __int128 q = (static_cast<__int128>(residual) + bias) >> shift;
  const __int128 r = static_cast<__int128>(residual) - q * ((__int128{1}) << shift) + bias;
  if (q < static_cast<__int128>(INT64_MIN) || q > static_cast<__int128>(INT64_MAX) ||
      r < static_cast<__int128>(INT64_MIN) || r > static_cast<__int128>(INT64_MAX)) {
    throw CodecError("bits-back quotient out of range");
  }
  quot = static_cast<int64_t>(q);
  rem = static_cast<int64_t>(r);
}

Sym make_sym(int64_t residual, int sigma, int frac, int nu) {
  if (sigma < 0 || sigma >= NUM_SIGMAS || frac < 0 || frac >= NUM_FRACS) {
    throw CodecError("bits-back context out of range");
  }
  Sym sym;
  sym.shift = get_parametric_tables().shifts[sigma];
  int64_t rem = 0;
  split_dyadic(residual, sym.shift, sym.quot, rem);
  if (rem < 0 || rem > static_cast<int64_t>(INT32_MAX)) throw CodecError("dyadic remainder out of range");
  sym.rem = static_cast<int>(rem);
  sym.escaped = iabs(sym.quot) > 32;
  const int bin = sym.escaped ? 65 : static_cast<int>(sym.quot + 32);
  sym.start = cum_row(nu, sigma, frac)[bin];
  sym.freq = freq_row(nu, sigma, frac)[bin];
  return sym;
}

std::vector<Sym> make_syms(const int64_t* residual, const int32_t* sig, const int32_t* frac,
                           const std::vector<int>& index, int nu) {
  std::vector<Sym> out;
  out.reserve(index.size());
  for (int i : index) out.push_back(make_sym(residual[i], sig[i], frac[i], nu));
  return out;
}

void push_reverse(RansBitsBackStack& stack, const std::vector<Sym>& syms) {
  for (int i = static_cast<int>(syms.size()) - 1; i >= 0; --i) stack.push(syms[i].start, syms[i].freq);
}

void write_extras(BitWriter& writer, const std::vector<Sym>& syms) {
  for (const Sym& sym : syms) {
    if (sym.escaped) {
      writer.write_bits(sym.quot < 0 ? 1u : 0u, 1);
      const int64_t mag = iabs(sym.quot) - 33;
      if (mag < 0 || mag > static_cast<int64_t>(UINT32_MAX)) throw CodecError("bits-back escape out of range");
      writer.write_exp_golomb(static_cast<uint32_t>(mag));
    }
    if (sym.shift > 0) writer.write_bits(static_cast<uint32_t>(sym.rem), sym.shift);
  }
}

int64_t read_residual(BitReader& reader, int sym, int shift) {
  int64_t quot;
  if (sym == 65) {
    const uint32_t sign = reader.read_bits(1);
    quot = static_cast<int64_t>(reader.read_exp_golomb()) + 33;
    if (sign) quot = -quot;
  } else {
    quot = static_cast<int64_t>(sym) - 32;
  }
  if (shift <= 0) return quot;
  const uint32_t rem = reader.read_bits(shift);
  const int64_t bias = int64_t{1} << (shift - 1);
  const int64_t shifted = static_cast<int64_t>(static_cast<uint64_t>(quot) << shift);
  return shifted - bias + static_cast<int64_t>(rem);
}

int64_t abs_quotients(const int64_t* residual, const int32_t* sig, const std::vector<int>& index) {
  const int32_t* shifts = get_parametric_tables().shifts;
  int64_t total = 0;
  for (int i : index) {
    const int sigma = sig[i];
    if (sigma < 0 || sigma >= NUM_SIGMAS) throw std::out_of_range("sigma");
    int64_t quot = 0;
    int64_t rem = 0;
    split_dyadic(residual[i], shifts[sigma], quot, rem);
    total += iabs(quot);
  }
  return total;
}

int64_t abs_quotients_main(const int64_t* residual, const int32_t* sig, size_t count) {
  const int32_t* shifts = get_parametric_tables().shifts;
  int64_t total = 0;
  for (size_t i = 0; i < count; ++i) {
    const int sigma = sig[i];
    if (sigma < 0 || sigma >= NUM_SIGMAS) throw CodecError("bits-back context out of range");
    int64_t quot = 0;
    int64_t rem = 0;
    split_dyadic(residual[i], shifts[sigma], quot, rem);
    total += iabs(quot);
  }
  return total;
}

std::vector<uint8_t> pack_stream(const RansBitsBackStack& stack, const std::vector<uint8_t>& extra) {
  const std::vector<uint8_t> body = stack.to_bytes();
  if (body.size() > 0xffffffffu) throw CodecError("bits-back body exceeds 32 bits");
  std::vector<uint8_t> out(4 + body.size() + extra.size());
  const uint32_t n = static_cast<uint32_t>(body.size());
  out[0] = static_cast<uint8_t>(n & 255u);
  out[1] = static_cast<uint8_t>((n >> 8) & 255u);
  out[2] = static_cast<uint8_t>((n >> 16) & 255u);
  out[3] = static_cast<uint8_t>((n >> 24) & 255u);
  std::copy(body.begin(), body.end(), out.begin() + 4);
  std::copy(extra.begin(), extra.end(), out.begin() + 4 + static_cast<std::ptrdiff_t>(body.size()));
  return out;
}

std::vector<uint8_t> finish_encode(RansBitsBackStack& stack, const std::vector<Sym>& main_syms,
                                   const std::vector<Sym>& seed_syms, int z) {
  const std::vector<int32_t>& prior = prior_freqs();
  const std::vector<int32_t> prior_c = cumulative(std::span<const int32_t>(prior));
  if (z < 0 || z >= static_cast<int>(prior.size())) throw CodecError("bits-back latent out of range");
  stack.push(prior_c[static_cast<size_t>(z)], prior[static_cast<size_t>(z)]);
  BitWriter writer;
  write_extras(writer, main_syms);
  write_extras(writer, seed_syms);
  return pack_stream(stack, writer.finish());
}

void split_sites(const uint8_t* seed, size_t count, std::vector<int>& seed_at, std::vector<int>& main_at) {
  for (size_t i = 0; i < count; ++i) {
    if (seed[i]) seed_at.push_back(static_cast<int>(i));
    else main_at.push_back(static_cast<int>(i));
  }
}

std::optional<std::vector<uint8_t>> encode_consistent(const int64_t* residual, const int32_t* sig,
                                                      const int32_t* frac, int nu, size_t count,
                                                      const std::vector<int>& seed_at,
                                                      const std::vector<int>& main_at,
                                                      const std::vector<int32_t>& post) {
  try {
    const std::vector<Sym> seed_syms = make_syms(residual, sig, frac, seed_at, nu);
    RansBitsBackStack stack;
    push_reverse(stack, seed_syms);
    const std::vector<int32_t> post_c = cumulative(std::span<const int32_t>(post));
    const int z = stack.pop(post_c, post);
    if (z < 0 || z >= static_cast<int>(post.size())) throw CodecError("bits-back latent out of range");
    std::vector<int32_t> shifted(sig, sig + count);
    for (int i : main_at) shifted[static_cast<size_t>(i)] = clip_sigma(sig[i] + z - 1);
    if (posterior_freqs(abs_quotients(residual, shifted.data(), main_at), static_cast<int>(main_at.size())) !=
        post) {
      return std::nullopt;
    }
    const std::vector<Sym> main_syms = make_syms(residual, shifted.data(), frac, main_at, nu);
    push_reverse(stack, main_syms);
    return finish_encode(stack, main_syms, seed_syms, z);
  } catch (const CodecError&) {
    return std::nullopt;
  }
}

}  // namespace

std::optional<std::vector<uint8_t>> encode_s3(const int64_t* residual, const int32_t* sig, const int32_t* frac,
                                              int nu, const uint8_t* seed, size_t count) {
  if (count > static_cast<size_t>(INT32_MAX)) throw CodecError("bits-back context length mismatch");
  std::vector<int> seed_at;
  std::vector<int> main_at;
  if (count > 0) {
    if (residual == nullptr || sig == nullptr || frac == nullptr || seed == nullptr) {
      throw CodecError("bits-back context length mismatch");
    }
    split_sites(seed, count, seed_at, main_at);
  }
  if (seed_at.size() < 8 || main_at.size() < 32) return std::nullopt;
  nu &= 3;
  try {
    const std::vector<Sym> seed_syms = make_syms(residual, sig, frac, seed_at, nu);
    RansBitsBackStack stack;
    push_reverse(stack, seed_syms);
    const std::vector<int32_t> post =
        posterior_freqs(abs_quotients(residual, sig, main_at), static_cast<int>(main_at.size()));
    const std::vector<int32_t> post_c = cumulative(std::span<const int32_t>(post));
    const int z = stack.pop(post_c, post);
    if (z < 0 || z >= static_cast<int>(post.size())) throw CodecError("bits-back latent out of range");
    std::vector<int32_t> shifted(sig, sig + count);
    for (int i : main_at) shifted[static_cast<size_t>(i)] = clip_sigma(sig[i] + z - 1);
    const std::vector<int32_t> post_z =
        posterior_freqs(abs_quotients(residual, shifted.data(), main_at), static_cast<int>(main_at.size()));
    if (post_z != post) return encode_consistent(residual, sig, frac, nu, count, seed_at, main_at, post_z);
    const std::vector<Sym> main_syms = make_syms(residual, shifted.data(), frac, main_at, nu);
    push_reverse(stack, main_syms);
    return finish_encode(stack, main_syms, seed_syms, z);
  } catch (const CodecError&) {
    return std::nullopt;
  }
}

std::vector<int32_t> decode_s3(const uint8_t* data, size_t size, int count, const int32_t* sig,
                               const int32_t* frac, int nu, const uint8_t* seed) {
  if (count < 0) throw CodecError("bits-back context length mismatch");
  if (size < 4) throw CodecError("truncated bits-back residual stream");
  const uint32_t body_n = static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8) |
                          (static_cast<uint32_t>(data[2]) << 16) | (static_cast<uint32_t>(data[3]) << 24);
  if (static_cast<uint64_t>(4) + body_n > size) throw CodecError("truncated bits-back body");
  if (count > 0 && (sig == nullptr || frac == nullptr || seed == nullptr)) {
    throw CodecError("bits-back context length mismatch");
  }
  RansBitsBackStack stack = RansBitsBackStack::from_bytes(std::span<const uint8_t>(data + 4, body_n));
  BitReader reader(data + 4 + body_n, size - 4 - body_n);
  std::vector<int> seed_at;
  std::vector<int> main_at;
  split_sites(seed, static_cast<size_t>(count), seed_at, main_at);
  nu &= 3;
  const std::vector<int32_t>& prior = prior_freqs();
  const std::vector<int32_t> prior_c = cumulative(std::span<const int32_t>(prior));
  const int z = stack.pop(prior_c, prior);
  if (z < 0 || z >= static_cast<int>(prior.size())) throw CodecError("bits-back latent out of range");

  const int32_t* shifts = get_parametric_tables().shifts;
  std::vector<int32_t> shifted(main_at.size());
  for (size_t i = 0; i < main_at.size(); ++i) shifted[i] = clip_sigma(sig[main_at[i]] + z - 1);

  std::vector<int64_t> main_out(main_at.size());
  for (size_t i = 0; i < main_at.size(); ++i) {
    const int s_v = shifted[i];
    const int f_v = frac[main_at[i]];
    if (s_v < 0 || s_v >= NUM_SIGMAS || f_v < 0 || f_v >= NUM_FRACS) {
      throw CodecError("bits-back context out of range");
    }
    const int sym = stack.pop(std::span<const int32_t>(cum_row(nu, s_v, f_v), CORE_BINS + 1),
                               std::span<const int32_t>(freq_row(nu, s_v, f_v), CORE_BINS));
    main_out[i] = read_residual(reader, sym, shifts[s_v]);
  }
  const std::vector<int32_t> post =
      posterior_freqs(abs_quotients_main(main_out.data(), shifted.data(), main_out.size()),
                      static_cast<int>(main_out.size()));
  const std::vector<int32_t> post_c = cumulative(std::span<const int32_t>(post));
  stack.push(post_c[static_cast<size_t>(z)], post[static_cast<size_t>(z)]);

  std::vector<int64_t> seed_out(seed_at.size());
  for (size_t i = 0; i < seed_at.size(); ++i) {
    const int s_v = sig[seed_at[i]];
    const int f_v = frac[seed_at[i]];
    if (s_v < 0 || s_v >= NUM_SIGMAS || f_v < 0 || f_v >= NUM_FRACS) {
      throw CodecError("bits-back context out of range");
    }
    const int sym = stack.pop(std::span<const int32_t>(cum_row(nu, s_v, f_v), CORE_BINS + 1),
                               std::span<const int32_t>(freq_row(nu, s_v, f_v), CORE_BINS));
    seed_out[i] = read_residual(reader, sym, shifts[s_v]);
  }
  if (stack.state != RANS64_L || !stack.words.empty()) {
    throw CodecError("bits-back final stack invariant violated");
  }
  std::vector<int32_t> out(static_cast<size_t>(count));
  size_t seed_i = 0;
  size_t main_i = 0;
  for (int i = 0; i < count; ++i) {
    const int64_t value = seed[i] ? seed_out[seed_i++] : main_out[main_i++];
    out[static_cast<size_t>(i)] = static_cast<int32_t>(value);
  }
  return out;
}

}  // namespace vpxc
