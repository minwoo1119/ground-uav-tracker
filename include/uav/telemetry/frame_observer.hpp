#pragma once

#include "uav/core/types.hpp"
#include "uav/telemetry/metrics_sink.hpp"

#include <span>

namespace uav {

class IFrameObserver {
 public:
  virtual ~IFrameObserver() = default;
  virtual void on_frame(const Frame& frame,
                        std::span<const Detection> detections,
                        std::span<const Track> tracks,
                        const FrameMetrics& metrics) = 0;
};

}  // namespace uav
