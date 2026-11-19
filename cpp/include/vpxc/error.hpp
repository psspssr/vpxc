#pragma once

#include <stdexcept>
#include <string>

namespace vpxc {

class CodecError : public std::runtime_error {
 public:
  explicit CodecError(const std::string& what) : std::runtime_error(what) {}
};

}  // namespace vpxc
