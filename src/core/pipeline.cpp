#include "uav/core/pipeline.hpp"

#include <chrono>
#include <stdexcept>

namespace uav {

TrackingPipeline::TrackingPipeline(IFrameSource& source,
                                   IDetector& detector,
                                   ITracker& tracker,
                                   IMetricsSink& metrics_sink) noexcept
    : source_(source),
      detector_(detector),
      tracker_(tracker),
      metrics_sink_(metrics_sink) {}

RunSummary TrackingPipeline::run(const RunOptions& options) {
  RunSummary summary{};
  while (options.max_frames == 0U ||
         summary.frames_processed < options.max_frames) {
    auto frame = source_.next();
    if (!frame.has_value()) {
      break;
    }
    if (!frame->is_valid()) {
      throw std::invalid_argument("frame source returned invalid image metadata");
    }
    if (last_timestamp_.has_value() &&
        frame->captured_at < last_timestamp_.value()) {
      throw std::invalid_argument(
          "frame source returned a timestamp older than the previous frame");
    }

    const auto started_at = std::chrono::steady_clock::now();
    auto detections = detector_.detect(*frame);
    auto tracks = tracker_.update(*frame, detections);
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - started_at);

    summary.detections_produced += detections.size();
    summary.tracks_produced += tracks.size();
    ++summary.frames_processed;
    last_timestamp_ = frame->captured_at;
    metrics_sink_.write(FrameMetrics{
        .frame_id = frame->id,
        .captured_at = frame->captured_at,
        .detection_count = detections.size(),
        .track_count = tracks.size(),
        .processing_time = elapsed,
    });
  }
  return summary;
}

}  // namespace uav
