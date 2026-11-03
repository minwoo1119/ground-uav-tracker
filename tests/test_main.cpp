#include "uav/core/cli.hpp"
#include "uav/core/pipeline.hpp"
#include "uav/core/types.hpp"
#include "uav/detection/mot_replay_detector.hpp"
#include "uav/telemetry/mot_track_writer.hpp"
#include "uav/telemetry/track_sink.hpp"
#include "uav/tracking/kalman_iou_tracker.hpp"
#include "uav/video/mot_sequence_source.hpp"
#include "uav/video/opencv_frame_source.hpp"

#include <array>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
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

template <typename Callable>
void require_runtime_error(Callable&& callable,
                           std::string_view expected_message) {
  try {
    callable();
  } catch (const std::runtime_error& error) {
    require(std::string_view{error.what()}.find(expected_message) !=
                std::string_view::npos,
            "error message explains the runtime failure");
    return;
  }
  throw TestFailure("expected std::runtime_error");
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

class RecordingTrackSink final : public uav::ITrackSink {
 public:
  void write(const uav::Frame& frame,
             std::span<const uav::Track> tracks) override {
    frame_ids.push_back(frame.id);
    track_counts.push_back(tracks.size());
  }

  std::vector<uav::FrameId> frame_ids{};
  std::vector<std::size_t> track_counts{};
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

  const std::array replay{
      std::string_view{"--replay-mot"},
      std::string_view{"data/processed/mot/V_DRONE_001"},
      std::string_view{"--output-tracks"},
      std::string_view{"results/V_DRONE_001.txt"},
  };
  const auto replay_options = uav::parse_cli(replay);
  require(replay_options.action == uav::CliAction::replay_mot,
          "replay_mot action is parsed");
  require(replay_options.replay_mot_dir == "data/processed/mot/V_DRONE_001",
          "replay_mot_dir is parsed");
  require(replay_options.output_tracks_path == "results/V_DRONE_001.txt",
          "output_tracks_path is parsed");

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
  const std::array output_without_replay{
      std::string_view{"--output-tracks"},
      std::string_view{"results/tracks.txt"},
  };
  require_invalid_argument(
      [&] { static_cast<void>(uav::parse_cli(output_without_replay)); },
      "requires --replay-mot");
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
  RecordingTrackSink track_sink;
  uav::TrackingPipeline pipeline{
      source, detector, tracker, metrics_sink, &track_sink};
  const auto summary = pipeline.run();
  require(summary.frames_processed == 2U, "all ordered frames are processed");
  require(summary.detections_produced == 0U,
          "no synthetic detections are produced");
  require(summary.tracks_produced == 0U, "no synthetic tracks are produced");
  require(track_sink.frame_ids == std::vector<uav::FrameId>{1, 2},
          "track sink receives every ordered frame");
  require(track_sink.track_counts == std::vector<std::size_t>{0, 0},
          "track sink receives explicit empty track lists");
}

void test_mot_track_writer() {
  const auto temp_dir =
      std::filesystem::temp_directory_path() / "uav_mot_track_writer_test";
  std::filesystem::remove_all(temp_dir);
  std::filesystem::create_directories(temp_dir);
  const auto output_path = temp_dir / "tracks.txt";

  const uav::Frame frame{
      .id = 7,
      .captured_at = uav::Timestamp{200'000},
      .image_size = {.width = 640, .height = 512},
  };
  const std::array tracks{
      uav::Track{
          .id = 3,
          .bounds = {.x = 10.25F,
                     .y = 20.5F,
                     .width = 30.0F,
                     .height = 40.75F},
          .state = uav::TrackState::confirmed,
      },
      uav::Track{
          .id = 4,
          .bounds = {.x = 50.0F,
                     .y = 60.0F,
                     .width = 12.0F,
                     .height = 14.0F},
          .state = uav::TrackState::lost,
          .missed_frames = 1,
      },
      uav::Track{
          .id = 5,
          .bounds = {.x = 70.0F,
                     .y = 80.0F,
                     .width = 10.0F,
                     .height = 10.0F},
          .state = uav::TrackState::removed,
      },
  };

  {
    uav::MotTrackWriter writer{output_path};
    writer.write(frame, tracks);
    writer.write(
        uav::Frame{
            .id = 8,
            .captured_at = uav::Timestamp{233'333},
            .image_size = {.width = 640, .height = 512},
        },
        {});
  }

  std::ifstream result{output_path};
  const std::string contents{std::istreambuf_iterator<char>{result}, {}};
  require(
      contents ==
          "7,3,10.250000,20.500000,30.000000,40.750000,-1,-1,-1,-1\n"
          "7,4,50.000000,60.000000,12.000000,14.000000,-1,-1,-1,-1\n",
      "MOT writer records active tracks and skips removed tracks");
  result.close();

  require_invalid_argument(
      [&] { uav::MotTrackWriter writer{output_path}; }, "already exists");

  require_runtime_error(
      [&] {
        uav::MotTrackWriter writer{temp_dir / "missing" / "tracks.txt"};
      },
      "failed to open MOT track output");

  std::filesystem::create_directories(temp_dir / "gt");
  {
    std::ofstream sequence_info{temp_dir / "seqinfo.ini"};
    sequence_info << "[Sequence]\n";
    std::ofstream ground_truth{temp_dir / "gt" / "gt.txt"};
    ground_truth << "original labels\n";
  }
  require_invalid_argument(
      [&] {
        uav::validate_mot_track_output_path(temp_dir / "gt" / "gt.txt",
                                            temp_dir);
      },
      "must not overwrite sequence input");
  require_invalid_argument(
      [&] {
        uav::validate_mot_track_output_path(temp_dir / "seqinfo.ini",
                                            temp_dir);
      },
      "must not overwrite sequence input");

  std::filesystem::remove_all(temp_dir);
}

void test_kalman_fast_motion_and_velocity() {
  // Target is 20x20 px moving +25 px/frame horizontally.
  // Note: consecutive boxes are 25 px apart, so raw IoU between frame t-1 and t
  // is ZERO (20 < 25). Without Kalman velocity prediction, IoU matching fails!
  // On frame 1->2, initial velocity is 0, so we start with a 10 px step to
  // establish velocity, then accelerate to 25 px steps where raw IoU == 0.
  uav::KalmanIoUTracker tracker{uav::TrackerConfig{
      .iou_threshold = 0.15F,
      .min_hits_to_confirm = 3U,
      .max_missed_frames = 4U,
  }};

  const std::array<float, 6> x_positions{100.0F, 112.0F, 130.0F,
                                         152.0F, 175.0F, 198.0F};
  for (std::size_t idx = 0; idx < x_positions.size(); ++idx) {
    const uav::Frame frame{
        .id = static_cast<uav::FrameId>(idx + 1U),
        .captured_at = uav::Timestamp{static_cast<std::int64_t>(idx * 33'333)},
        .image_size = {.width = 640, .height = 512},
    };
    const std::array detections{uav::Detection{
        .class_id = 2,
        .confidence = 0.95F,
        .bounds = {.x = x_positions[idx],
                   .y = 200.0F,
                   .width = 20.0F,
                   .height = 20.0F},
        .class_name = "DRONE",
    }};
    const auto tracks = tracker.update(frame, detections);
    require(tracks.size() == 1U, "single track maintained during fast motion");
    require(tracks[0].id == 1U, "same track ID preserved across fast motion");
    if (idx < 2U) {
      require(tracks[0].state == uav::TrackState::tentative,
              "track starts tentative before min_hits_to_confirm");
    } else {
      require(tracks[0].state == uav::TrackState::confirmed,
              "track becomes confirmed at hit 3");
    }
  }
}

void test_kalman_occlusion_recovery_and_removal() {
  uav::KalmanIoUTracker tracker{uav::TrackerConfig{
      .iou_threshold = 0.15F,
      .min_hits_to_confirm = 2U,
      .max_missed_frames = 2U,
  }};

  auto make_frame = [](uav::FrameId id) {
    return uav::Frame{
        .id = id,
        .captured_at =
            uav::Timestamp{static_cast<std::int64_t>((id - 1U) * 33'333)},
        .image_size = {.width = 640, .height = 512},
    };
  };
  auto make_det = [](float x) {
    return std::vector<uav::Detection>{uav::Detection{
        .class_id = 2,
        .confidence = 0.9F,
        .bounds = {.x = x, .y = 100.0F, .width = 30.0F, .height = 30.0F},
        .class_name = "DRONE",
    }};
  };

  // Frame 1 & 2: detected moving +10 px/frame -> confirmed
  auto tracks = tracker.update(make_frame(1), make_det(100.0F));
  require(tracks.size() == 1U && tracks[0].state == uav::TrackState::tentative,
          "frame 1 tentative");
  tracks = tracker.update(make_frame(2), make_det(110.0F));
  require(tracks.size() == 1U && tracks[0].state == uav::TrackState::confirmed,
          "frame 2 confirmed");

  // Frame 3 & 4: missed detection (occlusion) -> state becomes lost, predicted
  // position advances along velocity (+10 px/frame)
  tracks = tracker.update(make_frame(3), {});
  require(tracks.size() == 1U && tracks[0].state == uav::TrackState::lost &&
              tracks[0].missed_frames == 1U,
          "frame 3 lost (missed=1)");
  tracks = tracker.update(make_frame(4), {});
  require(tracks.size() == 1U && tracks[0].state == uav::TrackState::lost &&
              tracks[0].missed_frames == 2U,
          "frame 4 lost (missed=2)");

  // Frame 5: target reappears near predicted x ~ 140 -> recovers same ID=1!
  tracks = tracker.update(make_frame(5), make_det(140.0F));
  require(tracks.size() == 1U && tracks[0].id == 1U &&
              tracks[0].state == uav::TrackState::confirmed &&
              tracks[0].missed_frames == 0U,
          "frame 5 recovered same track ID=1 after 2 missed frames");

  // Frame 6, 7, 8: 3 consecutive misses (> max_missed_frames=2) -> removed
  static_cast<void>(tracker.update(make_frame(6), {}));
  static_cast<void>(tracker.update(make_frame(7), {}));
  tracks = tracker.update(make_frame(8), {});
  require(tracks.empty(), "track is purged once missed_frames > max_missed");
}

void test_mot_replay_end_to_end() {
  const auto temp_dir =
      std::filesystem::temp_directory_path() / "uav_mot_replay_test";
  std::filesystem::remove_all(temp_dir);
  std::filesystem::create_directories(temp_dir / "gt");

  {
    std::ofstream ini{temp_dir / "seqinfo.ini"};
    ini << "[Sequence]\n"
        << "name=SYNTH_DRONE\n"
        << "frameRate=30\n"
        << "seqLength=4\n"
        << "imWidth=640\n"
        << "imHeight=512\n";
  }
  {
    std::ofstream gt{temp_dir / "gt" / "gt.txt"};
    gt << "1,1,100.0,150.0,30.0,25.0,1,-1,-1,-1\n"
       << "2,1,106.0,150.0,30.0,25.0,1,-1,-1,-1\n"
       << "3,1,112.0,150.0,30.0,25.0,1,-1,-1,-1\n"
       << "4,1,118.0,150.0,30.0,25.0,1,-1,-1,-1\n";
  }

  uav::MotSequenceFrameSource source{temp_dir};
  uav::MotReplayDetector detector{temp_dir};
  uav::KalmanIoUTracker tracker{};
  RecordingMetricsSink sink{};
  const auto result_path = temp_dir / "tracks.txt";
  uav::RunSummary summary{};
  {
    uav::MotTrackWriter track_writer{result_path};
    uav::TrackingPipeline pipeline{
        source, detector, tracker, sink, &track_writer};
    summary = pipeline.run();
  }

  require(summary.frames_processed == 4U, "4 MOT frames processed");
  require(summary.detections_produced == 4U, "4 MOT detections replayed");
  require(summary.tracks_produced == 4U, "4 tracks produced");
  require(sink.frame_ids.size() == 4U, "4 frame metrics recorded");

  std::ifstream result{result_path};
  std::size_t result_rows = 0U;
  for (std::string row; std::getline(result, row);) {
    if (!row.empty()) {
      ++result_rows;
    }
  }
  require(result_rows == 4U, "4 MOT track result rows written");
  result.close();

  std::filesystem::remove_all(temp_dir);
}

void test_opencv_frame_source_missing_file() {
  require_invalid_argument(
      [] {
        uav::OpenCvFrameSource src{"non_existent_video_12345.mp4"};
      },
      "video file not found");
}

}  // namespace

int main() {
  const std::array tests{
      std::pair{"bounding boxes", test_bounding_boxes},
      std::pair{"track states", test_track_states},
      std::pair{"CLI options", test_cli_options},
      std::pair{"timestamp order", test_timestamp_order},
      std::pair{"pipeline summary", test_pipeline_summary},
      std::pair{"MOT track writer", test_mot_track_writer},
      std::pair{"Kalman fast motion & velocity",
                test_kalman_fast_motion_and_velocity},
      std::pair{"Kalman occlusion recovery & removal",
                test_kalman_occlusion_recovery_and_removal},
      std::pair{"MOT replay end-to-end", test_mot_replay_end_to_end},
      std::pair{"OpenCvFrameSource error handling",
                test_opencv_frame_source_missing_file},
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
