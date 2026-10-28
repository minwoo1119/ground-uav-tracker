#pragma once

#include "gut/core/interfaces.hpp"

#include <cstddef>

namespace gut {

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
                   IResultSink& sink) noexcept;
  [[nodiscard]] RunSummary run(const RunOptions& options = {});

 private:
  IFrameSource& source_;
  IDetector& detector_;
  ITracker& tracker_;
  IResultSink& sink_;
};

}  // namespace gut
