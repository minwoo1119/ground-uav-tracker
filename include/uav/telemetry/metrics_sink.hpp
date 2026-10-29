#pragma once

#include "uav/core/types.hpp"

#include <chrono>
#include <cstddef>

namespace uav {

struct FrameMetrics {
  FrameId frame_id{};
  Timestamp captured_at{};
  std::size_t detection_count{};
  std::size_t track_count{};
  std::chrono::microseconds processing_time{};
};

class IMetricsSink {
 public:
  virtual ~IMetricsSink() = default;
  virtual void write(const FrameMetrics& metrics) = 0;
};

}  // namespace uav
