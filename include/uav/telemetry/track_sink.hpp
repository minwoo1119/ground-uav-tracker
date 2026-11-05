#pragma once

#include "uav/core/types.hpp"

#include <span>

namespace uav {

class ITrackSink {
 public:
  virtual ~ITrackSink() = default;
  virtual void write(const Frame& frame, std::span<const Track> tracks) = 0;
};

}  // namespace uav
