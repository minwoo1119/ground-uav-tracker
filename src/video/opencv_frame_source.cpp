#include "uav/video/opencv_frame_source.hpp"

#include <stdexcept>
#include <string>

#if defined(UAV_HAS_OPENCV) && UAV_HAS_OPENCV
#include <opencv2/videoio.hpp>
#endif

namespace uav {

#if defined(UAV_HAS_OPENCV) && UAV_HAS_OPENCV

struct OpenCvFrameSource::Impl {
  cv::VideoCapture capture{};
  FrameId next_id{1U};
  double fps{30.0};
  ImageSize image_size{};
};

OpenCvFrameSource::OpenCvFrameSource(const std::filesystem::path& video_path)
    : impl_(std::make_unique<Impl>()) {
  if (!std::filesystem::exists(video_path)) {
    throw std::invalid_argument("video file not found: " + video_path.string());
  }
  if (!impl_->capture.open(video_path.string())) {
    throw std::runtime_error("OpenCV failed to open video: " +
                             video_path.string());
  }

  const auto w =
      static_cast<std::uint32_t>(impl_->capture.get(cv::CAP_PROP_FRAME_WIDTH));
  const auto h =
      static_cast<std::uint32_t>(impl_->capture.get(cv::CAP_PROP_FRAME_HEIGHT));
  const double raw_fps = impl_->capture.get(cv::CAP_PROP_FPS);
  impl_->fps = raw_fps > 0.0 ? raw_fps : 30.0;
  impl_->image_size = ImageSize{.width = w, .height = h};
}

OpenCvFrameSource::~OpenCvFrameSource() = default;
OpenCvFrameSource::OpenCvFrameSource(OpenCvFrameSource&&) noexcept = default;
OpenCvFrameSource& OpenCvFrameSource::operator=(OpenCvFrameSource&&) noexcept =
    default;

std::optional<Frame> OpenCvFrameSource::next() {
  if (!impl_ || !impl_->capture.isOpened()) {
    return std::nullopt;
  }
  cv::Mat image;
  if (!impl_->capture.read(image) || image.empty()) {
    return std::nullopt;
  }

  const FrameId id = impl_->next_id++;
  const auto width = static_cast<std::uint32_t>(image.cols);
  const auto height = static_cast<std::uint32_t>(image.rows);
  const auto timestamp_us = static_cast<std::int64_t>(
      static_cast<double>(id - 1U) * 1'000'000.0 / impl_->fps);

  return Frame{
      .id = id,
      .captured_at = Timestamp{timestamp_us},
      .image_size = ImageSize{.width = width, .height = height},
  };
}

bool OpenCvFrameSource::is_compiled_with_opencv() noexcept { return true; }

#else

struct OpenCvFrameSource::Impl {};

OpenCvFrameSource::OpenCvFrameSource(const std::filesystem::path& video_path) {
  if (!std::filesystem::exists(video_path)) {
    throw std::invalid_argument("video file not found: " + video_path.string());
  }
  throw std::runtime_error(
      "OpenCvFrameSource requires building with OpenCV enabled "
      "(UAV_ENABLE_OPENCV=ON)");
}

OpenCvFrameSource::~OpenCvFrameSource() = default;
OpenCvFrameSource::OpenCvFrameSource(OpenCvFrameSource&&) noexcept = default;
OpenCvFrameSource& OpenCvFrameSource::operator=(OpenCvFrameSource&&) noexcept =
    default;

std::optional<Frame> OpenCvFrameSource::next() { return std::nullopt; }

bool OpenCvFrameSource::is_compiled_with_opencv() noexcept { return false; }

#endif

}  // namespace uav
