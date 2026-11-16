#include "uav/detection/onnx_yolox_detector.hpp"
#include "uav/video/opencv_frame_source.hpp"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
  try {
    if (argc != 2 && argc != 3) {
      throw std::invalid_argument(
          "usage: uav_onnx_tests <model.onnx> [video.mp4]");
    }
    if (!uav::OnnxYoloXDetector::is_compiled_with_onnxruntime()) {
      throw std::runtime_error("test binary lacks ONNX Runtime support");
    }
    uav::OnnxYoloXDetector detector{uav::OnnxYoloXDetectorConfig{
        .model_path = std::filesystem::path{argv[1]},
        .postprocessor = uav::YoloXPostprocessorConfig{
            .input_size = {.width = 640U, .height = 640U},
            .strides = {8U, 16U, 32U},
            .class_names = {"AIRPLANE", "BIRD", "DRONE", "HELICOPTER"},
            .confidence_threshold = 0.99F,
            .nms_threshold = 0.45F,
            .class_agnostic_nms = true,
        },
        .intra_op_threads = 1U,
    }};

    bool rejected_metadata_only = false;
    try {
      static_cast<void>(detector.detect(uav::Frame{
          .id = 1U,
          .captured_at = uav::Timestamp{0},
          .image_size = {.width = 64U, .height = 48U},
      }));
    } catch (const std::invalid_argument&) {
      rejected_metadata_only = true;
    }
    if (!rejected_metadata_only) {
      throw std::runtime_error("ONNX detector accepted a frame without pixels");
    }

    constexpr std::uint32_t width = 64U;
    constexpr std::uint32_t height = 48U;
    constexpr std::size_t stride = width * 3U;
    uav::Frame frame{
        .id = 1U,
        .captured_at = uav::Timestamp{0},
        .image_size = {.width = width, .height = height},
        .pixel_format = uav::PixelFormat::bgr8,
        .row_stride_bytes = stride,
        .pixel_data = std::vector<std::uint8_t>(stride * height, 0U),
    };
    const auto detections = detector.detect(frame);
    for (const auto& detection : detections) {
      if (!detection.bounds.is_valid() || detection.confidence < 0.99F ||
          detection.class_id < 0 || detection.class_id > 3) {
        throw std::runtime_error(
            "ONNX detector returned a detection outside its contract");
      }
    }
    std::cout << "ONNX detector smoke inference: OK (detections="
              << detections.size() << ")\n";

    if (argc == 3) {
      uav::OpenCvFrameSource source{std::filesystem::path{argv[2]}};
      auto video_frame = source.next();
      if (!video_frame.has_value() || !video_frame->is_valid() ||
          !video_frame->has_pixels() ||
          video_frame->pixel_format != uav::PixelFormat::bgr8) {
        throw std::runtime_error(
            "OpenCV source did not return an owned BGR8 frame");
      }
      const auto video_detections = detector.detect(*video_frame);
      std::cout << "OpenCV -> ONNX detector frame: OK (detections="
                << video_detections.size() << ")\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ONNX detector test failed: " << error.what() << '\n';
    return 1;
  }
}
