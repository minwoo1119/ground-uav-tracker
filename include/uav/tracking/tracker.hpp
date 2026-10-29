#pragma once

#include "uav/core/types.hpp"

#include <span>
#include <vector>

namespace uav {

class ITracker {
 public:
  virtual ~ITracker() = default;
  [[nodiscard]] virtual std::vector<Track> update(
      const Frame& frame,
      std::span<const Detection> detections) = 0;
};

}  // namespace uav
