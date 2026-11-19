#include "uav/core/cli.hpp"
#include "uav/core/pipeline.hpp"
#include "uav/core/runtime_config.hpp"
#include "uav/core/types.hpp"
#include "uav/detection/mot_replay_detector.hpp"
#include "uav/detection/onnx_yolox_detector.hpp"
#include "uav/detection/yolox_preprocessor.hpp"
#include "uav/detection/yolox_postprocessor.hpp"
#include "uav/telemetry/frame_observer.hpp"
#include "uav/telemetry/mot_track_writer.hpp"
#include "uav/telemetry/track_sink.hpp"
#include "uav/tracking/kalman_iou_tracker.hpp"
#include "uav/video/mot_sequence_source.hpp"
#include "uav/video/opencv_frame_source.hpp"
#include "uav/visualization/opencv_video_visualizer.hpp"
#include "uav/visualization/overlay.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
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

void test_frame_pixel_contract_and_yolox_preprocessing() {
  const uav::Frame metadata_only{
      .id = 1U,
      .captured_at = uav::Timestamp{0},
      .image_size = {.width = 2U, .height = 1U},
  };
  require(metadata_only.is_valid() && !metadata_only.has_pixels(),
          "metadata-only frames remain valid");

  const uav::Frame invalid_pixels{
      .id = 2U,
      .captured_at = uav::Timestamp{0},
      .image_size = {.width = 2U, .height = 1U},
      .pixel_format = uav::PixelFormat::bgr8,
      .row_stride_bytes = 5U,
      .pixel_data = std::vector<std::uint8_t>(5U),
  };
  require(!invalid_pixels.is_valid(),
          "BGR8 frames reject a stride smaller than width times three");

  uav::Frame pixels{
      .id = 3U,
      .captured_at = uav::Timestamp{0},
      .image_size = {.width = 2U, .height = 1U},
      .pixel_format = uav::PixelFormat::bgr8,
      .row_stride_bytes = 8U,
      .pixel_data = {10U, 20U, 30U, 10U, 20U, 30U, 99U, 99U},
  };
  require(pixels.is_valid() && pixels.has_pixels(),
          "stride-aware owned BGR8 frame is valid");

  const auto input =
      uav::preprocess_yolox_bgr(pixels, {.width = 4U, .height = 4U});
  require(input.tensor.size() == 48U && input.scale == 2.0F &&
              input.resized_size.width == 4U &&
              input.resized_size.height == 2U,
          "YOLOX preprocessing reports CHW tensor and letterbox geometry");
  require(input.tensor[0U] == 10.0F && input.tensor[16U] == 20.0F &&
              input.tensor[32U] == 30.0F,
          "YOLOX preprocessing preserves BGR channel order");
  require(input.tensor[8U] == 114.0F && input.tensor[24U] == 114.0F &&
              input.tensor[40U] == 114.0F,
          "YOLOX preprocessing pads below the top-left image with 114");

  const uav::Frame interpolation_fixture{
      .id = 4U,
      .captured_at = uav::Timestamp{0},
      .image_size = {.width = 2U, .height = 2U},
      .pixel_format = uav::PixelFormat::bgr8,
      .row_stride_bytes = 6U,
      .pixel_data = {
          0U, 0U, 0U, 100U, 0U, 0U,
          200U, 0U, 0U, 255U, 0U, 0U,
      },
  };
  const auto interpolated = uav::preprocess_yolox_bgr(
      interpolation_fixture, {.width = 4U, .height = 4U});
  const std::array expected_blue{
      0.0F,   25.0F,  75.0F,  100.0F,
      50.0F,  72.0F,  117.0F, 139.0F,
      150.0F, 167.0F, 200.0F, 216.0F,
      200.0F, 214.0F, 241.0F, 255.0F,
  };
  require(std::equal(expected_blue.begin(), expected_blue.end(),
                     interpolated.tensor.begin()),
          "bilinear resize matches the pinned OpenCV uint8 fixture");

  require_invalid_argument(
      [&] {
        static_cast<void>(uav::preprocess_yolox_bgr(
            metadata_only, {.width = 4U, .height = 4U}));
      },
      "CPU-owned BGR8");
}

uav::YoloXPostprocessorConfig make_yolox_config(
    bool class_agnostic_nms = true) {
  return uav::YoloXPostprocessorConfig{
      .input_size = {.width = 640U, .height = 640U},
      .strides = {8U, 16U, 32U},
      .class_names = {"AIRPLANE", "BIRD", "DRONE", "HELICOPTER"},
      .confidence_threshold = 0.5F,
      .nms_threshold = 0.45F,
      .class_agnostic_nms = class_agnostic_nms,
  };
}

void set_yolox_anchor(std::vector<float>& output,
                      std::size_t anchor,
                      float raw_x,
                      float raw_y,
                      float width,
                      float height,
                      float stride,
                      float objectness,
                      std::size_t class_index,
                      float class_probability) {
  constexpr std::size_t features = 9U;
  const auto offset = anchor * features;
  output[offset] = raw_x;
  output[offset + 1U] = raw_y;
  output[offset + 2U] = std::log(width / stride);
  output[offset + 3U] = std::log(height / stride);
  output[offset + 4U] = objectness;
  output[offset + 5U + class_index] = class_probability;
}

void test_yolox_decode_and_letterbox_restore() {
  const uav::YoloXPostprocessor postprocessor{make_yolox_config()};
  require(postprocessor.expected_anchor_count() == 8400U,
          "YOLOX-s 640 has 8,400 anchors");
  require(postprocessor.feature_count() == 9U,
          "four-class YOLOX output has nine features");

  std::vector<float> output(postprocessor.expected_anchor_count() *
                            postprocessor.feature_count());
  set_yolox_anchor(output, 0U, 12.5F, 6.25F, 100.0F, 50.0F, 8.0F,
                   0.8F, 2U, 0.75F);
  set_yolox_anchor(output, 6400U, 20.0F, 20.0F, 64.0F, 32.0F, 16.0F,
                   0.7F, 0U, 0.8F);
  set_yolox_anchor(output, 8000U, 15.0F, 3.0F, 64.0F, 64.0F, 32.0F,
                   0.6875F, 1U, 0.8F);
  const auto detections =
      postprocessor.process(output, {.width = 1280U, .height = 720U});

  require(detections.size() == 3U,
          "stride 8, 16, and 32 YOLOX candidates pass threshold");
  const auto& detection = detections.front();
  require(detection.class_id == 2 && detection.class_name == "DRONE",
          "YOLOX class ID and name mapping are preserved");
  require(std::abs(detection.confidence - 0.6F) < 0.0001F,
          "YOLOX confidence is objectness times class probability");
  require(std::abs(detection.bounds.x - 100.0F) < 0.001F &&
              std::abs(detection.bounds.y - 50.0F) < 0.001F &&
              std::abs(detection.bounds.width - 200.0F) < 0.001F &&
              std::abs(detection.bounds.height - 100.0F) < 0.001F,
          "top-left letterbox scale is restored to original coordinates");
  require(detections[1].class_id == 0 &&
              std::abs(detections[1].bounds.width - 128.0F) < 0.001F &&
              std::abs(detections[1].bounds.height - 64.0F) < 0.001F,
          "stride-16 anchor is decoded with its own grid scale");
  require(detections[2].class_id == 1 &&
              std::abs(detections[2].bounds.x - 896.0F) < 0.001F,
          "stride-32 anchor follows the concatenated YOLOX level order");
}

void test_yolox_clipping_and_nms_modes() {
  auto class_agnostic_config = make_yolox_config(true);
  class_agnostic_config.confidence_threshold = 0.2F;
  const uav::YoloXPostprocessor class_agnostic{class_agnostic_config};
  std::vector<float> output(class_agnostic.expected_anchor_count() *
                            class_agnostic.feature_count());

  set_yolox_anchor(output, 0U, 2.0F, 2.0F, 40.0F, 40.0F, 8.0F,
                   0.9F, 2U, 0.9F);
  set_yolox_anchor(output, 1U, 1.0F, 2.0F, 40.0F, 40.0F, 8.0F,
                   0.8F, 1U, 0.9F);
  const auto suppressed =
      class_agnostic.process(output, {.width = 640U, .height = 640U});
  require(suppressed.size() == 1U && suppressed.front().class_id == 2,
          "class-agnostic NMS keeps the strongest overlapping candidate");
  require(suppressed.front().bounds.x == 0.0F &&
              suppressed.front().bounds.y == 0.0F &&
              suppressed.front().bounds.width == 36.0F &&
              suppressed.front().bounds.height == 36.0F,
          "decoded boxes are clipped to original image bounds");

  const uav::YoloXPostprocessor class_aware{make_yolox_config(false)};
  const auto retained =
      class_aware.process(output, {.width = 640U, .height = 640U});
  require(retained.size() == 2U,
          "class-aware NMS retains overlapping different classes");
}

void test_yolox_invalid_contracts() {
  require_invalid_argument(
      [] {
        static_cast<void>(
            uav::YoloXPostprocessor{uav::YoloXPostprocessorConfig{}});
      },
      "confidence threshold");

  const uav::YoloXPostprocessor postprocessor{make_yolox_config()};
  std::vector<float> wrong_size(9U);
  require_invalid_argument(
      [&] {
        static_cast<void>(postprocessor.process(
            wrong_size, {.width = 640U, .height = 640U}));
      },
      "raw output length");

  std::vector<float> non_finite(postprocessor.expected_anchor_count() *
                                postprocessor.feature_count());
  non_finite[0] = std::numeric_limits<float>::quiet_NaN();
  require_invalid_argument(
      [&] {
        static_cast<void>(postprocessor.process(
            non_finite, {.width = 640U, .height = 640U}));
      },
      "NaN or Inf");

  std::vector<float> invalid_probability(
      postprocessor.expected_anchor_count() * postprocessor.feature_count());
  invalid_probability[4] = 1.1F;
  require_invalid_argument(
      [&] {
        static_cast<void>(postprocessor.process(
            invalid_probability, {.width = 640U, .height = 640U}));
      },
      "objectness");

  std::vector<float> decoded_overflow(
      postprocessor.expected_anchor_count() * postprocessor.feature_count());
  decoded_overflow[2] = 1000.0F;
  decoded_overflow[4] = 1.0F;
  decoded_overflow[5] = 1.0F;
  require_invalid_argument(
      [&] {
        static_cast<void>(postprocessor.process(
            decoded_overflow, {.width = 640U, .height = 640U}));
      },
      "decoded box");
}

void test_onnx_detector_build_contract() {
  require_invalid_argument(
      [] {
        static_cast<void>(uav::OnnxYoloXDetector{
            uav::OnnxYoloXDetectorConfig{
                .model_path = "non_existent_model_12345.onnx"}});
      },
      "ONNX model file not found");
#if defined(UAV_HAS_ONNXRUNTIME) && UAV_HAS_ONNXRUNTIME
  require(uav::OnnxYoloXDetector::is_compiled_with_onnxruntime(),
          "ONNX-enabled build reports runtime availability");
#else
  require(!uav::OnnxYoloXDetector::is_compiled_with_onnxruntime(),
          "default build reports ONNX Runtime as unavailable");
  const auto model_path =
      std::filesystem::temp_directory_path() / "uav_empty_model.onnx";
  {
    std::ofstream model{model_path, std::ios::binary};
    model << "not-an-onnx-model";
  }
  uav::OnnxYoloXDetector detector{
      uav::OnnxYoloXDetectorConfig{.model_path = model_path}};
  require_runtime_error(
      [&] { static_cast<void>(detector.detect(uav::Frame{})); },
      "built without ONNX Runtime support");
  std::filesystem::remove(model_path);
#endif
}

void test_track_states() {
  const uav::Track removed{.id = 4, .state = uav::TrackState::removed};
  require(removed.state == uav::TrackState::removed,
          "removed track state is represented explicitly");
}

[[nodiscard]] std::string valid_runtime_config_text() {
  return
      "runtime:\n"
      "  max_frames: 7\n"
      "detection:\n"
      "  model_path: ../models/test.onnx\n"
      "  input_width: 640\n"
      "  input_height: 640\n"
      "  strides: [8, 16, 32]\n"
      "  class_names: [AIRPLANE, BIRD, DRONE, HELICOPTER]\n"
      "  confidence_threshold: 0.25\n"
      "  nms_threshold: 0.45\n"
      "  class_agnostic_nms: true\n"
      "  intra_op_threads: 2\n"
      "tracking:\n"
      "  iou_threshold: 0.2\n"
      "  min_hits_to_confirm: 3\n"
      "  max_missed_frames: 5\n"
      "  default_dt_seconds: 0.033333333\n"
      "  position_process_noise: 4.0\n"
      "  size_process_noise: 1.0\n"
      "  velocity_process_noise: 2500.0\n"
      "  position_measurement_noise: 2.0\n"
      "  size_measurement_noise: 2.0\n";
}

void test_runtime_config() {
  const auto temp_dir =
      std::filesystem::temp_directory_path() / "uav_runtime_config_test";
  std::filesystem::remove_all(temp_dir);
  std::filesystem::create_directories(temp_dir / "configs");
  const auto config_path = temp_dir / "configs" / "runtime.yaml";
  {
    std::ofstream config_file{config_path};
    config_file << valid_runtime_config_text();
  }

  const auto config = uav::load_runtime_config(config_path);
  require(config.max_frames == 7U, "runtime max_frames is parsed");
  require(config.detector.model_path ==
              (temp_dir / "models" / "test.onnx").lexically_normal(),
          "relative model path is resolved from the config directory");
  require(config.detector.postprocessor.class_names.size() == 4U &&
              config.detector.postprocessor.class_names[2] == "DRONE",
          "detector class order is preserved");
  require(config.detector.postprocessor.strides ==
              std::vector<std::uint32_t>{8U, 16U, 32U},
          "detector strides are parsed");
  require(config.detector.intra_op_threads == 2U,
          "ONNX thread count is parsed");
  require(std::abs(config.tracker.iou_threshold - 0.2F) < 0.0001F &&
              config.tracker.min_hits_to_confirm == 3U &&
              config.tracker.max_missed_frames == 5U,
          "tracker configuration is parsed");

  const auto write_invalid = [&](std::string_view name,
                                 const std::string& contents) {
    const auto path = temp_dir / (std::string{name} + ".yaml");
    std::ofstream file{path};
    file << contents;
    file.close();
    return path;
  };

  const auto unknown = write_invalid(
      "unknown", valid_runtime_config_text() + "  unexpected: 1\n");
  require_invalid_argument(
      [&] { static_cast<void>(uav::load_runtime_config(unknown)); },
      "unknown key 'tracking.unexpected'");

  auto duplicate_text = valid_runtime_config_text();
  const auto detection_position = duplicate_text.find("detection:\n");
  duplicate_text.insert(detection_position + std::string{"detection:\n"}.size(),
                        "  input_width: 320\n");
  const auto duplicate = write_invalid("duplicate", duplicate_text);
  require_invalid_argument(
      [&] { static_cast<void>(uav::load_runtime_config(duplicate)); },
      "duplicate key 'detection.input_width'");

  auto missing_text = valid_runtime_config_text();
  const std::string required_line = "  nms_threshold: 0.45\n";
  missing_text.erase(missing_text.find(required_line), required_line.size());
  const auto missing = write_invalid("missing", missing_text);
  require_invalid_argument(
      [&] { static_cast<void>(uav::load_runtime_config(missing)); },
      "missing required key 'detection.nms_threshold'");

  auto invalid_bool_text = valid_runtime_config_text();
  const auto bool_position = invalid_bool_text.find("class_agnostic_nms: true");
  invalid_bool_text.replace(bool_position,
                            std::string{"class_agnostic_nms: true"}.size(),
                            "class_agnostic_nms: yes");
  const auto invalid_bool = write_invalid("invalid_bool", invalid_bool_text);
  require_invalid_argument(
      [&] { static_cast<void>(uav::load_runtime_config(invalid_bool)); },
      "must be true or false");

  auto invalid_list_text = valid_runtime_config_text();
  const auto list_position = invalid_list_text.find("strides: [8, 16, 32]");
  invalid_list_text.replace(list_position,
                            std::string{"strides: [8, 16, 32]"}.size(),
                            "strides: 8, 16, 32");
  const auto invalid_list = write_invalid("invalid_list", invalid_list_text);
  require_invalid_argument(
      [&] { static_cast<void>(uav::load_runtime_config(invalid_list)); },
      "must use [value, value] syntax");

  auto invalid_threads_text = valid_runtime_config_text();
  const auto threads_position =
      invalid_threads_text.find("intra_op_threads: 2");
  invalid_threads_text.replace(
      threads_position, std::string{"intra_op_threads: 2"}.size(),
      "intra_op_threads: 0");
  const auto invalid_threads =
      write_invalid("invalid_threads", invalid_threads_text);
  require_invalid_argument(
      [&] { static_cast<void>(uav::load_runtime_config(invalid_threads)); },
      "detection.intra_op_threads must be >= 1");

  auto invalid_range_text = valid_runtime_config_text();
  const auto range_position = invalid_range_text.find("iou_threshold: 0.2");
  invalid_range_text.replace(range_position,
                             std::string{"iou_threshold: 0.2"}.size(),
                             "iou_threshold: 1.5");
  const auto invalid_range =
      write_invalid("invalid_range", invalid_range_text);
  require_invalid_argument(
      [&] { static_cast<void>(uav::load_runtime_config(invalid_range)); },
      "tracking.iou_threshold must be in [0, 1]");

  const auto malformed = write_invalid("malformed", "runtime:\n   max_frames: 0\n");
  require_invalid_argument(
      [&] { static_cast<void>(uav::load_runtime_config(malformed)); },
      "only two-level keys");

  require_invalid_argument(
      [&] {
        static_cast<void>(uav::load_runtime_config(
            temp_dir / "missing_runtime_config.yaml"));
      },
      "runtime config file not found");

  std::filesystem::remove_all(temp_dir);
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

  const std::array video_with_output{
      std::string_view{"--config"},
      std::string_view{"configs/example.yaml"},
      std::string_view{"--input"},
      std::string_view{"video.mp4"},
      std::string_view{"--output-tracks"},
      std::string_view{"outputs/tracks.txt"},
  };
  const auto video_options = uav::parse_cli(video_with_output);
  require(video_options.action == uav::CliAction::run &&
              video_options.output_tracks_path == "outputs/tracks.txt",
          "video inference accepts optional MOT track output");

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
  const std::array mixed_modes{
      std::string_view{"--replay-mot"},
      std::string_view{"data/processed/mot/V_DRONE_001"},
      std::string_view{"--input"},
      std::string_view{"video.mp4"},
  };
  require_invalid_argument(
      [&] { static_cast<void>(uav::parse_cli(mixed_modes)); },
      "cannot be combined");

  const std::array video_run_args{
      std::string_view{"--config"},
      std::string_view{"configs/example.yaml"},
      std::string_view{"--input"},
      std::string_view{"video.mp4"},
      std::string_view{"--output-tracks"},
      std::string_view{"out/tracks.txt"},
      std::string_view{"--output-video"},
      std::string_view{"out/overlay.avi"},
  };
  const auto video_run = uav::parse_cli(video_run_args);
  require(video_run.action == uav::CliAction::run, "video run action parsed");
  require(video_run.output_video_path == "out/overlay.avi",
          "output_video_path is parsed");

  const std::array duplicate_video{
      std::string_view{"--config"},       std::string_view{"a.yaml"},
      std::string_view{"--input"},        std::string_view{"in.mp4"},
      std::string_view{"--output-video"}, std::string_view{"a.avi"},
      std::string_view{"--output-video"}, std::string_view{"b.avi"},
  };
  require_invalid_argument(
      [&] { static_cast<void>(uav::parse_cli(duplicate_video)); },
      "may only be specified once");

  const std::array replay_with_video{
      std::string_view{"--replay-mot"},
      std::string_view{"data/processed/mot/V_DRONE_001"},
      std::string_view{"--output-video"},
      std::string_view{"out/overlay.avi"},
  };
  require_invalid_argument(
      [&] { static_cast<void>(uav::parse_cli(replay_with_video)); },
      "cannot be combined");
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

class RecordingFrameObserver final : public uav::IFrameObserver {
 public:
  void on_frame(const uav::Frame& frame,
                std::span<const uav::Detection> detections,
                std::span<const uav::Track> tracks,
                const uav::FrameMetrics& metrics) override {
    frame_ids.push_back(frame.id);
    timestamps.push_back(frame.captured_at);
    detection_counts.push_back(detections.size());
    track_counts.push_back(tracks.size());
    metric_frame_ids.push_back(metrics.frame_id);
  }

  std::vector<uav::FrameId> frame_ids{};
  std::vector<uav::Timestamp> timestamps{};
  std::vector<std::size_t> detection_counts{};
  std::vector<std::size_t> track_counts{};
  std::vector<uav::FrameId> metric_frame_ids{};
};

void test_pipeline_frame_observer() {
  TimestampSource source{{uav::Timestamp{100}, uav::Timestamp{200}}};
  EmptyDetector detector;
  EmptyTracker tracker;
  RecordingMetricsSink metrics_sink;
  RecordingTrackSink track_sink;
  RecordingFrameObserver observer;
  uav::TrackingPipeline pipeline{
      source, detector, tracker, metrics_sink, &track_sink, &observer};
  const auto summary = pipeline.run();
  require(summary.frames_processed == 2U, "observer run processes 2 frames");
  require(observer.frame_ids == std::vector<uav::FrameId>{1, 2},
          "observer receives frame IDs in order");
  require(observer.timestamps ==
              std::vector<uav::Timestamp>{uav::Timestamp{100},
                                          uav::Timestamp{200}},
          "observer preserves frame timestamps");
  require(observer.metric_frame_ids == std::vector<uav::FrameId>{1, 2},
          "observer receives matching FrameMetrics");
  require(observer.detection_counts == std::vector<std::size_t>{0, 0} &&
              observer.track_counts == std::vector<std::size_t>{0, 0},
          "observer receives detection and track spans");
}

void test_overlay_formatting_and_path_validation() {
  const uav::ImageSize size{.width = 100U, .height = 80U};
  const auto inside = uav::clamp_bounding_box(
      {.x = 10.2F, .y = 20.6F, .width = 15.1F, .height = 10.0F}, size);
  require(inside.has_value() && inside->x == 10 && inside->y == 20 &&
              inside->width == 16 && inside->height == 11,
          "bounding box clamps and rounds to enclosing integer pixels");

  const auto clipped = uav::clamp_bounding_box(
      {.x = -5.0F, .y = 70.0F, .width = 20.0F, .height = 30.0F}, size);
  require(clipped.has_value() && clipped->x == 0 && clipped->y == 70 &&
              clipped->width == 15 && clipped->height == 10,
          "partially out-of-bounds box is clamped to image bounds");

  require(!uav::clamp_bounding_box(
               {.x = 120.0F, .y = 10.0F, .width = 10.0F, .height = 10.0F},
               size)
               .has_value(),
          "box outside image returns nullopt");
  require(!uav::clamp_bounding_box(
               {.x = std::numeric_limits<float>::quiet_NaN(),
                .y = 0.0F,
                .width = 10.0F,
                .height = 10.0F},
               size)
               .has_value(),
          "non-finite box returns nullopt");

  require(uav::format_detection_label(uav::Detection{
              .class_id = 2,
              .confidence = 0.876F,
              .class_name = "DRONE",
          }) == "DRONE 0.88",
          "detection label formats class name and confidence");
  require(uav::format_detection_label(uav::Detection{
              .class_id = 1,
              .confidence = 0.5F,
              .class_name = "",
          }) == "class_1 0.50",
          "empty class name falls back to class_<id>");
  require(uav::format_track_label(uav::Track{
              .id = 7, .state = uav::TrackState::confirmed}) ==
              "#7 [CONFIRMED]",
          "track label formats ID and state");
  require(uav::track_state_color(uav::TrackState::confirmed) !=
              uav::track_state_color(uav::TrackState::tentative) &&
              uav::track_state_color(uav::TrackState::confirmed) !=
                  uav::track_state_color(uav::TrackState::lost),
          "active track states use distinct colors");

  const std::string hud = uav::format_hud_summary(uav::FrameMetrics{
      .frame_id = 3U,
      .captured_at = uav::Timestamp{66'700},
      .detection_count = 2U,
      .track_count = 1U,
      .processing_time = std::chrono::microseconds{12'500},
  });
  require(hud == "Frame 3 | t=66.7ms | Det:2 | Trk:1 | Proc:12.5ms",
          "HUD summary formats measured metrics deterministically");

  require(uav::select_video_fourcc("out/sample.AVI") == "MJPG",
          "AVI selects MJPG fourcc case-insensitively");
  require(uav::select_video_fourcc("out/sample.mp4") == "mp4v",
          "MP4 selects mp4v fourcc");
  require_invalid_argument(
      [] { static_cast<void>(uav::select_video_fourcc("out/sample.mkv")); },
      "unsupported video output extension");

  const auto temp_dir =
      std::filesystem::temp_directory_path() / "uav_overlay_path_test";
  std::filesystem::remove_all(temp_dir);
  std::filesystem::create_directories(temp_dir);
  const auto existing_input = temp_dir / "input.mp4";
  {
    std::ofstream file{existing_input};
    file << "placeholder";
  }

  uav::validate_video_output_path(temp_dir / "overlay.avi", existing_input,
                                  temp_dir / "tracks.txt");
  require_invalid_argument(
      [&] {
        uav::validate_video_output_path(existing_input, existing_input,
                                        std::nullopt);
      },
      "already exists");
  require_invalid_argument(
      [&] {
        uav::validate_video_output_path(temp_dir / "missing" / "overlay.avi",
                                        existing_input, std::nullopt);
      },
      "parent directory does not exist");
  require_invalid_argument(
      [&] {
        uav::validate_video_output_path(temp_dir / "same.avi", existing_input,
                                        temp_dir / "same.avi");
      },
      "must not match --output-tracks");

  std::filesystem::remove_all(temp_dir);
}

void test_video_visualizer_contract_and_roundtrip() {
  const uav::Frame metadata_only{
      .id = 1U,
      .captured_at = uav::Timestamp{0},
      .image_size = {.width = 160U, .height = 120U},
  };
  const uav::FrameMetrics metrics{
      .frame_id = 1U,
      .captured_at = uav::Timestamp{0},
      .detection_count = 1U,
      .track_count = 1U,
      .processing_time = std::chrono::microseconds{4'200},
  };
  require_invalid_argument(
      [&] {
        static_cast<void>(
            uav::render_overlay_bgr(metadata_only, {}, {}, metrics));
      },
      "BGR8 frame with owned pixels");

  const auto temp_dir =
      std::filesystem::temp_directory_path() / "uav_video_visualizer_test";
  std::filesystem::remove_all(temp_dir);
  std::filesystem::create_directories(temp_dir);
  const auto avi_path = temp_dir / "rendered.avi";

  require_invalid_argument(
      [&] {
        uav::OpenCvVideoVisualizer visualizer{uav::VideoVisualizerConfig{
            .output_path = avi_path,
            .frame_size = {.width = 0U, .height = 120U},
            .fps = 25.0,
        }};
      },
      "positive frame_size");
  require_invalid_argument(
      [&] {
        uav::OpenCvVideoVisualizer visualizer{uav::VideoVisualizerConfig{
            .output_path = avi_path,
            .frame_size = {.width = 160U, .height = 120U},
            .fps = 0.0,
        }};
      },
      "positive finite fps");
  require_invalid_argument(
      [&] {
        uav::OpenCvVideoVisualizer visualizer{uav::VideoVisualizerConfig{
            .output_path = temp_dir / "bad.mkv",
            .frame_size = {.width = 160U, .height = 120U},
            .fps = 25.0,
        }};
      },
      "unsupported video output extension");

  constexpr std::uint32_t width = 160U;
  constexpr std::uint32_t height = 120U;
  constexpr std::size_t padded_stride = static_cast<std::size_t>(width) * 3U + 12U;
  uav::Frame padded_frame{
      .id = 1U,
      .captured_at = uav::Timestamp{0},
      .image_size = {.width = width, .height = height},
      .pixel_format = uav::PixelFormat::bgr8,
      .row_stride_bytes = padded_stride,
      .pixel_data = std::vector<std::uint8_t>(padded_stride * height, 0U),
  };

  const std::array detections{uav::Detection{
      .class_id = 2,
      .confidence = 0.91F,
      .bounds = {.x = 20.0F, .y = 40.0F, .width = 30.0F, .height = 24.0F},
      .class_name = "DRONE",
  }};
  const std::array tracks{
      uav::Track{
          .id = 1U,
          .bounds = {.x = 70.0F, .y = 50.0F, .width = 30.0F, .height = 24.0F},
          .state = uav::TrackState::confirmed,
      },
      uav::Track{
          .id = 2U,
          .bounds = {.x = 120.0F, .y = 85.0F, .width = 25.0F, .height = 20.0F},
          .state = uav::TrackState::removed,
      },
  };

  if (!uav::OpenCvVideoVisualizer::is_compiled_with_opencv()) {
    require_runtime_error(
        [&] {
          static_cast<void>(uav::render_overlay_bgr(
              padded_frame, detections, tracks, metrics));
        },
        "UAV_ENABLE_OPENCV=ON");
    require_runtime_error(
        [&] {
          uav::OpenCvVideoVisualizer visualizer{uav::VideoVisualizerConfig{
              .output_path = avi_path,
              .frame_size = {.width = width, .height = height},
              .fps = 25.0,
          }};
        },
        "UAV_ENABLE_OPENCV=ON");
    std::filesystem::remove_all(temp_dir);
    return;
  }

  const auto rendered =
      uav::render_overlay_bgr(padded_frame, detections, tracks, metrics);
  require(rendered.is_valid() && rendered.has_pixels() &&
              rendered.id == padded_frame.id &&
              rendered.captured_at == padded_frame.captured_at &&
              rendered.image_size.width == width &&
              rendered.image_size.height == height,
          "rendered frame preserves metadata and valid BGR8 pixels");
  require(std::all_of(padded_frame.pixel_data.begin(),
                      padded_frame.pixel_data.end(),
                      [](std::uint8_t value) { return value == 0U; }),
          "render_overlay_bgr does not mutate the input frame");

  const auto confirmed_color =
      uav::track_state_color(uav::TrackState::confirmed);
  const auto confirmed_offset =
      (50U * rendered.row_stride_bytes) + (70U * 3U);
  require(rendered.pixel_data[confirmed_offset] == confirmed_color.b &&
              rendered.pixel_data[confirmed_offset + 1U] == confirmed_color.g &&
              rendered.pixel_data[confirmed_offset + 2U] == confirmed_color.r,
          "confirmed track border is drawn with the confirmed state color");

  const auto removed_offset =
      (85U * rendered.row_stride_bytes) + (120U * 3U);
  require(rendered.pixel_data[removed_offset] == 0U &&
              rendered.pixel_data[removed_offset + 1U] == 0U &&
              rendered.pixel_data[removed_offset + 2U] == 0U,
          "removed track is not rendered");

  {
    uav::OpenCvVideoVisualizer visualizer{uav::VideoVisualizerConfig{
        .output_path = avi_path,
        .frame_size = {.width = width, .height = height},
        .fps = 25.0,
    }};
    require_invalid_argument(
        [&] { visualizer.on_frame(metadata_only, {}, {}, metrics); },
        "BGR8 frame with owned pixels");

    uav::Frame wrong_size{
        .id = 2U,
        .captured_at = uav::Timestamp{40'000},
        .image_size = {.width = 80U, .height = 60U},
        .pixel_format = uav::PixelFormat::bgr8,
        .row_stride_bytes = 80U * 3U,
        .pixel_data = std::vector<std::uint8_t>(80U * 3U * 60U, 0U),
    };
    require_invalid_argument(
        [&] { visualizer.on_frame(wrong_size, {}, {}, metrics); },
        "frame dimensions do not match");

    for (uav::FrameId id = 1U; id <= 3U; ++id) {
      padded_frame.id = id;
      padded_frame.captured_at =
          uav::Timestamp{static_cast<std::int64_t>((id - 1U) * 40'000)};
      uav::FrameMetrics frame_metrics = metrics;
      frame_metrics.frame_id = id;
      frame_metrics.captured_at = padded_frame.captured_at;
      visualizer.on_frame(padded_frame, detections, tracks, frame_metrics);
    }
    require(visualizer.frames_written() == 3U,
            "visualizer counts written frames");
    visualizer.close();
    require_runtime_error(
        [&] { visualizer.on_frame(padded_frame, detections, tracks, metrics); },
        "closed");
  }

  require_invalid_argument(
      [&] {
        uav::OpenCvVideoVisualizer duplicate{uav::VideoVisualizerConfig{
            .output_path = avi_path,
            .frame_size = {.width = width, .height = height},
            .fps = 25.0,
        }};
      },
      "already exists");

  {
    uav::OpenCvFrameSource reader{avi_path};
    require(reader.image_size().width == width &&
                reader.image_size().height == height,
            "written video preserves frame dimensions");
    require(std::abs(reader.fps() - 25.0) < 0.01,
            "written video preserves fps");
    std::size_t read_frames = 0U;
    bool observed_nonzero_pixel = false;
    while (auto decoded = reader.next()) {
      require(decoded->is_valid() && decoded->has_pixels(),
              "decoded output video frame is valid BGR8");
      if (std::any_of(decoded->pixel_data.begin(), decoded->pixel_data.end(),
                      [](std::uint8_t value) { return value > 10U; })) {
        observed_nonzero_pixel = true;
      }
      ++read_frames;
    }
    require(read_frames == 3U, "all 3 written video frames are readable");
    require(observed_nonzero_pixel,
            "decoded output video contains rendered overlay pixels");
  }

  std::filesystem::remove_all(temp_dir);
}

}  // namespace

int main() {
  const std::array tests{
      std::pair{"bounding boxes", test_bounding_boxes},
      std::pair{"Frame pixels and YOLOX preprocessing",
                test_frame_pixel_contract_and_yolox_preprocessing},
      std::pair{"YOLOX decode and letterbox restore",
                test_yolox_decode_and_letterbox_restore},
      std::pair{"YOLOX clipping and NMS modes",
                test_yolox_clipping_and_nms_modes},
      std::pair{"YOLOX invalid contracts", test_yolox_invalid_contracts},
      std::pair{"ONNX detector build contract",
                test_onnx_detector_build_contract},
      std::pair{"track states", test_track_states},
      std::pair{"runtime config", test_runtime_config},
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
      std::pair{"pipeline frame observer", test_pipeline_frame_observer},
      std::pair{"overlay formatting and path validation",
                test_overlay_formatting_and_path_validation},
      std::pair{"video visualizer contract and roundtrip",
                test_video_visualizer_contract_and_roundtrip},
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
