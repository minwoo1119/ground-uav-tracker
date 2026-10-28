#include "gut/core/pipeline.hpp"
#include "gut/core/types.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

class TestFailure final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw TestFailure(std::string{message});
  }
}

class TwoFrameSource final : public gut::IFrameSource {
 public:
  std::optional<gut::FrameView> next() override {
    if (next_id_ > 2U) {
      return std::nullopt;
    }
    return gut::FrameView{
        .id = next_id_++,
        .captured_at = gut::Timestamp{100},
        .width = 2,
        .height = 2,
        .stride_bytes = 2,
        .pixel_format = gut::PixelFormat::gray8,
        .pixels = pixels_,
    };
  }

 private:
  std::array<std::byte, 4> pixels_{};
  gut::FrameId next_id_{1};
};

class StubDetector final : public gut::IDetector {
 public:
  std::vector<gut::Detection> detect(const gut::FrameView&) override {
    return {{.bounds = {0.0F, 0.0F, 1.0F, 1.0F},
             .confidence = 0.9F,
             .class_id = 0,
             .class_name = "uav"}};
  }
};

class StubTracker final : public gut::ITracker {
 public:
  std::vector<gut::Track> update(
      gut::FrameId,
      gut::Timestamp,
      std::span<const gut::Detection> detections) override {
    require(detections.size() == 1U, "tracker receives detector output");
    return {{.id = 7,
             .bounds = detections.front().bounds,
             .confidence = detections.front().confidence,
             .class_id = detections.front().class_id,
             .state = gut::TrackState::confirmed,
             .age_frames = 1,
             .missed_frames = 0}};
  }
};

class RecordingSink final : public gut::IResultSink {
 public:
  void write(const gut::FrameResult& result) override {
    frame_ids.push_back(result.frame_id);
  }
  std::vector<gut::FrameId> frame_ids{};
};

void test_bounding_boxes() {
  const gut::BoundingBox a{0.0F, 0.0F, 10.0F, 10.0F};
  const gut::BoundingBox b{5.0F, 5.0F, 10.0F, 10.0F};
  require(a.is_valid(), "positive bounding box is valid");
  require(!gut::BoundingBox{0.0F, 0.0F, 0.0F, 1.0F}.is_valid(),
          "zero-width bounding box is invalid");
  require(std::abs(gut::intersection_over_union(a, b) - (25.0F / 175.0F)) <
              0.0001F,
          "IoU is calculated correctly");
}

void test_frame_validation() {
  const std::array<std::byte, 12> pixels{};
  const gut::FrameView valid{.id = 1,
                             .captured_at = gut::Timestamp{0},
                             .width = 2,
                             .height = 2,
                             .stride_bytes = 6,
                             .pixel_format = gut::PixelFormat::rgb8,
                             .pixels = pixels};
  require(valid.is_valid(), "complete RGB frame is valid");
  auto truncated = valid;
  truncated.pixels = std::span<const std::byte>{pixels}.first(11);
  require(!truncated.is_valid(), "truncated frame is invalid");
}

void test_pipeline_contract() {
  TwoFrameSource source;
  StubDetector detector;
  StubTracker tracker;
  RecordingSink sink;
  gut::TrackingPipeline pipeline{source, detector, tracker, sink};
  const auto summary = pipeline.run({.max_frames = 1});
  require(summary.frames_processed == 1U, "frame limit is respected");
  require(summary.detections_produced == 1U, "detections are counted");
  require(summary.tracks_produced == 1U, "tracks are counted");
  require(sink.frame_ids == std::vector<gut::FrameId>{1},
          "result is written with the frame id");
}

}  // namespace

int main() {
  const std::array tests{
      std::pair{"bounding boxes", test_bounding_boxes},
      std::pair{"frame validation", test_frame_validation},
      std::pair{"pipeline contract", test_pipeline_contract},
  };
  std::size_t failures = 0;
  for (const auto& [name, test] : tests) {
    try {
      test();
      std::cout << "[PASS] " << name << '\n';
    } catch (const std::exception& error) {
      ++failures;
      std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
    }
  }
  std::cout << tests.size() - failures << '/' << tests.size()
            << " tests passed\n";
  return failures == 0U ? 0 : 1;
}
