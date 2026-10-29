#pragma once

#include "uav/core/types.hpp"

#include <optional>

namespace uav {

class IFrameSource {
 public:
  virtual ~IFrameSource() = default;
  [[nodiscard]] virtual std::optional<Frame> next() = 0;
};

}  // namespace uav
