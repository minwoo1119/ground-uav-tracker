#include "uav/core/cli.hpp"
#include "uav/core/pipeline.hpp"
#include "uav/detection/mot_replay_detector.hpp"
#include "uav/tracking/kalman_iou_tracker.hpp"
#include "uav/video/mot_sequence_source.hpp"

#include <array>
#include <filesystem>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#ifndef UAV_PROJECT_VERSION
#define UAV_PROJECT_VERSION "unknown"
#endif

namespace {

class SingleFrameSource final : public uav::IFrameSource {
 public:
  std::optional<uav::Frame> next() override {
    if (done_) {
      return std::nullopt;
    }
    done_ = true;
    return uav::Frame{
        .id = 1,
        .captured_at = uav::Timestamp{0},
        .image_size = {.width = 2, .height = 2},
    };
  }

 private:
  bool done_{};
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

class CountingMetricsSink final : public uav::IMetricsSink {
 public:
  void write(const uav::FrameMetrics& metrics) override {
    ++writes;
    total_processing_us += metrics.processing_time.count();
  }
  std::size_t writes{};
  std::int64_t total_processing_us{};
};

int run_self_check() {
  SingleFrameSource source;
  EmptyDetector detector;
  EmptyTracker tracker;
  CountingMetricsSink metrics_sink;
  uav::TrackingPipeline pipeline{source, detector, tracker, metrics_sink};
  const auto summary = pipeline.run();
  const bool ok = summary.frames_processed == 1U &&
                  summary.detections_produced == 0U &&
                  summary.tracks_produced == 0U &&
                  metrics_sink.writes == 1U;
  std::cout << (ok ? "self-check: OK" : "self-check: FAILED") << '\n'
            << "No detector model or tracker implementation was used.\n";
  return ok ? 0 : 1;
}

int run_mot_replay(const std::filesystem::path& sequence_dir) {
  uav::MotSequenceFrameSource source{sequence_dir};
  uav::MotReplayDetector detector{sequence_dir};
  uav::KalmanIoUTracker tracker{};
  CountingMetricsSink metrics_sink;

  uav::TrackingPipeline pipeline{source, detector, tracker, metrics_sink};
  const auto summary = pipeline.run();

  std::cout << "mot-replay: " << source.info().name << '\n'
            << "  frames_processed:    " << summary.frames_processed << '\n'
            << "  detections_produced: " << summary.detections_produced << '\n'
            << "  tracks_produced:     " << summary.tracks_produced << '\n'
            << "  total_processing_us: " << metrics_sink.total_processing_us
            << '\n';
  return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    std::vector<std::string_view> arguments;
    arguments.reserve(static_cast<std::size_t>(argc > 0 ? argc - 1 : 0));
    for (int index = 1; index < argc; ++index) {
      arguments.emplace_back(argv[index]);
    }

    const auto options = uav::parse_cli(arguments);
    switch (options.action) {
      case uav::CliAction::show_help:
        std::cout << uav::cli_usage();
        return 0;
      case uav::CliAction::show_version:
        std::cout << "uav-tracker " << UAV_PROJECT_VERSION << '\n';
        return 0;
      case uav::CliAction::self_check:
        return run_self_check();
      case uav::CliAction::replay_mot:
        return run_mot_replay(*options.replay_mot_dir);
      case uav::CliAction::run:
        break;
    }

    if (!std::filesystem::is_regular_file(*options.config_path)) {
      std::cerr << "error: configuration file not found: "
                << *options.config_path << '\n';
      return 2;
    }
    std::cerr
        << "error: live detector inference is not implemented yet; "
           "use --replay-mot <mot-sequence-dir> to run the tracking pipeline\n";
    return 3;
  } catch (const std::invalid_argument& error) {
    std::cerr << "error: " << error.what() << "\n\n" << uav::cli_usage();
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 3;
  }
}
