#include "gut/core/pipeline.hpp"

#include <stdexcept>
#include <utility>

namespace gut {

TrackingPipeline::TrackingPipeline(IFrameSource& source,
                                   IDetector& detector,
                                   ITracker& tracker,
                                   IResultSink& sink) noexcept
    : source_(source), detector_(detector), tracker_(tracker), sink_(sink) {}

RunSummary TrackingPipeline::run(const RunOptions& options) {
  RunSummary summary{};
  while (options.max_frames == 0U ||
         summary.frames_processed < options.max_frames) {
    auto frame = source_.next();
    if (!frame.has_value()) {
      break;
    }
    if (!frame->is_valid()) {
      throw std::invalid_argument("frame source returned an invalid frame");
    }

    auto detections = detector_.detect(*frame);
    auto tracks = tracker_.update(frame->id, frame->captured_at, detections);
    summary.detections_produced += detections.size();
    summary.tracks_produced += tracks.size();
    ++summary.frames_processed;
    sink_.write(FrameResult{
        .frame_id = frame->id,
        .captured_at = frame->captured_at,
        .tracks = std::move(tracks),
    });
  }
  return summary;
}

}  // namespace gut
