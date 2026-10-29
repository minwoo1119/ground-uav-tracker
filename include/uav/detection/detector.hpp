#pragma once

#include "uav/core/types.hpp"

#include <vector>

namespace uav {

class IDetector {
 public:
  virtual ~IDetector() = default;
  [[nodiscard]] virtual std::vector<Detection> detect(const Frame& frame) = 0;
};

}  // namespace uav
