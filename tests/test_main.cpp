#include "uav/core/cli.hpp"
#include "uav/core/pipeline.hpp"
#include "uav/core/types.hpp"

#include <array>
#include <cmath>
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

template <typename Callable>
void require_invalid_argument(Callable&& callable,
                              std::string_view expected_message) {
  try {
    callable();
  } catch (const std::invalid_argument& error) {
    require(std::string_view{error.what()}.find(expected_message) !=
                std::string_view::npos,
            "error message explains the invalid argument");
    return;
  }
  throw TestFailure("expected std::invalid_argument");
}

class TimestampSource final : public uav::IFrameSource {
 public:
  explicit TimestampSource(std::vector<uav::Timestamp> timestamps)
      : timestamps_(std::move(timestamps)) {}

  std::optional<uav::Frame> next() override {
    if (index_ == timestamps_.size()) {
      return std::nullopt;
    }
    const auto current = index_++;
    return uav::Frame{
        .id = static_cast<uav::FrameId>(current + 1U),
        .captured_at = timestamps_[current],
        .image_size = {.width = 640, .height = 480},
    };
  }

 private:
  std::vector<uav::Timestamp> timestamps_;
  std::size_t index_{};
};

class EmptyDetector final : public uav::IDetector {
 public:
  std::vector<uav::Detection> detect(const uav::Frame&) override {
    return {};
  }
};

class EmptyTracker final : public uav::ITracker {
 public:
  std::vector<uav::Track> update(
      const uav::Frame&,
      std::span<const uav::Detection>) override {
    return {};
  }
};

class RecordingMetricsSink final : public uav::IMetricsSink {
 public:
  void write(const uav::FrameMetrics& metrics) override {
    frame_ids.push_back(metrics.frame_id);
  }
  std::vector<uav::FrameId> frame_ids{};
};

void test_bounding_boxes() {
  const uav::BoundingBox first{0.0F, 0.0F, 10.0F, 10.0F};
  const uav::BoundingBox second{5.0F, 5.0F, 10.0F, 10.0F};
  require(first.is_valid(), "positive bounding box is valid");
  require(!uav::BoundingBox{0.0F, 0.0F, 0.0F, 1.0F}.is_valid(),
          "zero-width bounding box is invalid");
  require(std::abs(uav::intersection_over_union(first, second) -
                   (25.0F / 175.0F)) < 0.0001F,
          "IoU is calculated correctly");
}

void test_track_states() {
  const uav::Track removed{.id = 4, .state = uav::TrackState::removed};
  require(removed.state == uav::TrackState::removed,
          "removed track state is represented explicitly");
}

void test_cli_options() {
  const std::array valid{
      std::string_view{"--config"},
      std::string_view{"configs/example.yaml"},
      std::string_view{"--input"},
      std::string_view{"camera0"},
  };
  const auto options = uav::parse_cli(valid);
  require(options.config_path == "configs/example.yaml",
          "configuration path is parsed");
  require(options.input_source == "camera0", "input source is parsed");

  const std::array unknown{std::string_view{"--unknown"}};
  require_invalid_argument(
      [&] { static_cast<void>(uav::parse_cli(unknown)); }, "unknown option");
  const std::array missing_value{std::string_view{"--config"}};
  require_invalid_argument(
      [&] { static_cast<void>(uav::parse_cli(missing_value)); },
      "requires a value");
  const std::array missing_input{
      std::string_view{"--config"},
      std::string_view{"configs/example.yaml"},
  };
  require_invalid_argument(
      [&] { static_cast<void>(uav::parse_cli(missing_input)); },
      "missing required option: --input");
}

void test_timestamp_order() {
  TimestampSource source{{uav::Timestamp{20}, uav::Timestamp{10}}};
  EmptyDetector detector;
  EmptyTracker tracker;
  RecordingMetricsSink metrics_sink;
  uav::TrackingPipeline pipeline{source, detector, tracker, metrics_sink};
  require_invalid_argument(
      [&] { static_cast<void>(pipeline.run()); }, "timestamp older");
  require(metrics_sink.frame_ids == std::vector<uav::FrameId>{1},
          "only the ordered frame reaches the metrics sink");
}

void test_pipeline_summary() {
  TimestampSource source{{uav::Timestamp{10}, uav::Timestamp{20}}};
  EmptyDetector detector;
  EmptyTracker tracker;
  RecordingMetricsSink metrics_sink;
  uav::TrackingPipeline pipeline{source, detector, tracker, metrics_sink};
  const auto summary = pipeline.run();
  require(summary.frames_processed == 2U, "all ordered frames are processed");
  require(summary.detections_produced == 0U,
          "no synthetic detections are produced");
  require(summary.tracks_produced == 0U, "no synthetic tracks are produced");
}

}  // namespace

int main() {
  const std::array tests{
      std::pair{"bounding boxes", test_bounding_boxes},
      std::pair{"track states", test_track_states},
      std::pair{"CLI options", test_cli_options},
      std::pair{"timestamp order", test_timestamp_order},
      std::pair{"pipeline summary", test_pipeline_summary},
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
