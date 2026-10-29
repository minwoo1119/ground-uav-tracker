#pragma once

#include "uav/detection/detector.hpp"
#include "uav/telemetry/metrics_sink.hpp"
#include "uav/tracking/tracker.hpp"
#include "uav/video/frame_source.hpp"

#include <cstddef>
#include <optional>

namespace uav {

struct RunOptions {
  // Zero means run until the source is exhausted.
  std::size_t max_frames{};
};

struct RunSummary {
  std::size_t frames_processed{};
  std::size_t detections_produced{};
  std::size_t tracks_produced{};
};

class TrackingPipeline {
 public:
  TrackingPipeline(IFrameSource& source,
                   IDetector& detector,
                   ITracker& tracker,
                   IMetricsSink& metrics_sink) noexcept;
  [[nodiscard]] RunSummary run(const RunOptions& options = {});

 private:
  IFrameSource& source_;
  IDetector& detector_;
  ITracker& tracker_;
  IMetricsSink& metrics_sink_;
  std::optional<Timestamp> last_timestamp_{};
};

}  // namespace uav
