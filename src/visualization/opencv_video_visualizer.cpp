#include "uav/visualization/opencv_video_visualizer.hpp"

#include "uav/visualization/overlay.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(UAV_HAS_OPENCV) && UAV_HAS_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#endif

namespace uav {
namespace {

void validate_bgr8_frame(const Frame& frame) {
  if (!frame.is_valid()) {
    throw std::invalid_argument(
        "video visualizer requires a valid frame");
  }
  if (!frame.has_pixels() || frame.pixel_format != PixelFormat::bgr8) {
    throw std::invalid_argument(
        "video visualizer requires a BGR8 frame with owned pixels");
  }
  constexpr auto max_int =
      static_cast<std::uint32_t>(std::numeric_limits<int>::max());
  if (frame.image_size.width > max_int || frame.image_size.height > max_int) {
    throw std::invalid_argument(
        "frame dimensions exceed OpenCV integer limits");
  }
}

std::string validate_visualizer_config(const VideoVisualizerConfig& config) {
  if (!config.frame_size.is_valid()) {
    throw std::invalid_argument(
        "video visualizer requires a valid positive frame_size");
  }
  constexpr auto max_int =
      static_cast<std::uint32_t>(std::numeric_limits<int>::max());
  if (config.frame_size.width > max_int ||
      config.frame_size.height > max_int) {
    throw std::invalid_argument(
        "video visualizer frame_size exceeds OpenCV integer limits");
  }
  if (!std::isfinite(config.fps) || config.fps <= 0.0) {
    throw std::invalid_argument(
        "video visualizer requires a positive finite fps");
  }
  validate_video_output_path(config.output_path, std::filesystem::path{},
                             std::nullopt);
  return select_video_fourcc(config.output_path);
}

#if defined(UAV_HAS_OPENCV) && UAV_HAS_OPENCV

[[nodiscard]] cv::Scalar to_cv_scalar(const BgrColor& color) noexcept {
  return cv::Scalar(static_cast<double>(color.b), static_cast<double>(color.g),
                    static_cast<double>(color.r));
}

[[nodiscard]] cv::Mat copy_frame_to_mat(const Frame& frame) {
  const int width = static_cast<int>(frame.image_size.width);
  const int height = static_cast<int>(frame.image_size.height);
  const auto tight_stride = static_cast<std::size_t>(width) * 3U;
  cv::Mat canvas(height, width, CV_8UC3);
  for (int row = 0; row < height; ++row) {
    const auto* src = frame.pixel_data.data() +
                      static_cast<std::size_t>(row) * frame.row_stride_bytes;
    auto* dst = canvas.ptr<std::uint8_t>(row);
    std::copy_n(src, tight_stride, dst);
  }
  return canvas;
}

void draw_label_badge(cv::Mat& canvas,
                      const std::string& text,
                      int anchor_x,
                      int anchor_y,
                      const BgrColor& background_bgr,
                      const BgrColor& text_bgr) {
  constexpr int font_face = cv::FONT_HERSHEY_SIMPLEX;
  constexpr double font_scale = 0.45;
  constexpr int thickness = 1;
  constexpr int pad = 3;

  int baseline = 0;
  const cv::Size text_size =
      cv::getTextSize(text, font_face, font_scale, thickness, &baseline);
  const int box_w = text_size.width + pad * 2;
  const int box_h = text_size.height + baseline + pad * 2;

  const int max_x = std::max(0, canvas.cols - box_w);
  const int max_y = std::max(0, canvas.rows - box_h);
  const int rect_x = std::clamp(anchor_x, 0, max_x);
  const int rect_y = std::clamp(anchor_y, 0, max_y);

  const cv::Rect bg_rect(rect_x, rect_y, std::min(box_w, canvas.cols - rect_x),
                         std::min(box_h, canvas.rows - rect_y));
  if (bg_rect.width > 0 && bg_rect.height > 0) {
    cv::rectangle(canvas, bg_rect, to_cv_scalar(background_bgr), cv::FILLED);
    const cv::Point text_origin(
        rect_x + pad,
        std::min(canvas.rows - 1, rect_y + pad + text_size.height));
    cv::putText(canvas, text, text_origin, font_face, font_scale,
                to_cv_scalar(text_bgr), thickness, cv::LINE_AA);
  }
}

void draw_overlay_inplace(cv::Mat& canvas,
                          const ImageSize& image_size,
                          std::span<const Detection> detections,
                          std::span<const Track> tracks,
                          const FrameMetrics& metrics,
                          const OverlayOptions& options) {
  constexpr BgrColor dark_text{.b = 15U, .g = 15U, .r = 15U};
  constexpr BgrColor light_text{.b = 245U, .g = 245U, .r = 245U};
  constexpr BgrColor hud_bg{.b = 20U, .g = 20U, .r = 20U};

  if (options.draw_detections) {
    const auto det_color = detection_box_color();
    for (const auto& detection : detections) {
      const auto rect = clamp_bounding_box(detection.bounds, image_size);
      if (!rect.has_value()) {
        continue;
      }
      const cv::Rect cv_rect(rect->x, rect->y, rect->width, rect->height);
      cv::rectangle(canvas, cv_rect, to_cv_scalar(det_color), 1, cv::LINE_8);
      const int label_y =
          rect->y + rect->height + 2 < canvas.rows ? rect->y + rect->height + 2
                                                   : rect->y;
      draw_label_badge(canvas, format_detection_label(detection), rect->x,
                       label_y, det_color, dark_text);
    }
  }

  if (options.draw_tracks) {
    for (const auto& track : tracks) {
      if (track.state == TrackState::removed) {
        continue;
      }
      const auto rect = clamp_bounding_box(track.bounds, image_size);
      if (!rect.has_value()) {
        continue;
      }
      const auto color = track_state_color(track.state);
      const cv::Rect cv_rect(rect->x, rect->y, rect->width, rect->height);
      cv::rectangle(canvas, cv_rect, to_cv_scalar(color), 2, cv::LINE_8);
      const int preferred_y = rect->y >= 24 ? rect->y - 24 : rect->y;
      draw_label_badge(canvas, format_track_label(track), rect->x, preferred_y,
                       color, dark_text);
    }
  }

  if (options.draw_hud) {
    draw_label_badge(canvas, format_hud_summary(metrics), 6, 6, hud_bg,
                     light_text);
  }
}

#endif

}  // namespace

#if defined(UAV_HAS_OPENCV) && UAV_HAS_OPENCV

Frame render_overlay_bgr(const Frame& frame,
                         std::span<const Detection> detections,
                         std::span<const Track> tracks,
                         const FrameMetrics& metrics,
                         const OverlayOptions& options) {
  validate_bgr8_frame(frame);
  cv::Mat canvas = copy_frame_to_mat(frame);
  draw_overlay_inplace(canvas, frame.image_size, detections, tracks, metrics,
                       options);

  const auto tight_stride =
      static_cast<std::size_t>(frame.image_size.width) * 3U;
  std::vector<std::uint8_t> pixels(
      tight_stride * static_cast<std::size_t>(frame.image_size.height));
  for (std::uint32_t row = 0U; row < frame.image_size.height; ++row) {
    const auto* src = canvas.ptr<std::uint8_t>(static_cast<int>(row));
    std::copy_n(src, tight_stride,
                pixels.data() + static_cast<std::size_t>(row) * tight_stride);
  }

  return Frame{
      .id = frame.id,
      .captured_at = frame.captured_at,
      .image_size = frame.image_size,
      .pixel_format = PixelFormat::bgr8,
      .row_stride_bytes = tight_stride,
      .pixel_data = std::move(pixels),
  };
}

struct OpenCvVideoVisualizer::Impl {
  VideoVisualizerConfig config{};
  cv::VideoWriter writer{};
  std::size_t frames_written{};
};

OpenCvVideoVisualizer::OpenCvVideoVisualizer(VideoVisualizerConfig config)
    : impl_(std::make_unique<Impl>()) {
  const std::string fourcc_text = validate_visualizer_config(config);
  const int fourcc = cv::VideoWriter::fourcc(fourcc_text[0], fourcc_text[1],
                                             fourcc_text[2], fourcc_text[3]);
  const cv::Size size(static_cast<int>(config.frame_size.width),
                      static_cast<int>(config.frame_size.height));

  impl_->config = std::move(config);
  if (!impl_->writer.open(impl_->config.output_path.string(), fourcc,
                          impl_->config.fps, size, true)) {
    std::error_code remove_error;
    std::filesystem::remove(impl_->config.output_path, remove_error);
    throw std::runtime_error(
        "OpenCV failed to open video writer for " +
        impl_->config.output_path.string() + " (fourcc=" + fourcc_text + ")");
  }
}

OpenCvVideoVisualizer::~OpenCvVideoVisualizer() { close(); }

OpenCvVideoVisualizer::OpenCvVideoVisualizer(OpenCvVideoVisualizer&&) noexcept =
    default;
OpenCvVideoVisualizer& OpenCvVideoVisualizer::operator=(
    OpenCvVideoVisualizer&&) noexcept = default;

void OpenCvVideoVisualizer::on_frame(const Frame& frame,
                                     std::span<const Detection> detections,
                                     std::span<const Track> tracks,
                                     const FrameMetrics& metrics) {
  if (!impl_ || !impl_->writer.isOpened()) {
    throw std::runtime_error("OpenCvVideoVisualizer writer is closed");
  }
  validate_bgr8_frame(frame);
  if (frame.image_size.width != impl_->config.frame_size.width ||
      frame.image_size.height != impl_->config.frame_size.height) {
    throw std::invalid_argument(
        "frame dimensions do not match OpenCvVideoVisualizer configuration");
  }

  cv::Mat canvas = copy_frame_to_mat(frame);
  draw_overlay_inplace(canvas, frame.image_size, detections, tracks, metrics,
                       impl_->config.overlay);
  impl_->writer.write(canvas);
  ++impl_->frames_written;
}

void OpenCvVideoVisualizer::close() noexcept {
  if (impl_ && impl_->writer.isOpened()) {
    impl_->writer.release();
  }
}

const std::filesystem::path& OpenCvVideoVisualizer::output_path()
    const noexcept {
  static const std::filesystem::path empty_path{};
  return impl_ ? impl_->config.output_path : empty_path;
}

std::size_t OpenCvVideoVisualizer::frames_written() const noexcept {
  return impl_ ? impl_->frames_written : 0U;
}

bool OpenCvVideoVisualizer::is_compiled_with_opencv() noexcept { return true; }

#else

Frame render_overlay_bgr(const Frame& frame,
                         std::span<const Detection>,
                         std::span<const Track>,
                         const FrameMetrics&,
                         const OverlayOptions&) {
  validate_bgr8_frame(frame);
  throw std::runtime_error(
      "render_overlay_bgr requires building with OpenCV enabled "
      "(UAV_ENABLE_OPENCV=ON)");
}

struct OpenCvVideoVisualizer::Impl {
  std::filesystem::path output_path{};
};

OpenCvVideoVisualizer::OpenCvVideoVisualizer(VideoVisualizerConfig config) {
  static_cast<void>(validate_visualizer_config(config));
  throw std::runtime_error(
      "OpenCvVideoVisualizer requires building with OpenCV enabled "
      "(UAV_ENABLE_OPENCV=ON)");
}

OpenCvVideoVisualizer::~OpenCvVideoVisualizer() = default;
OpenCvVideoVisualizer::OpenCvVideoVisualizer(OpenCvVideoVisualizer&&) noexcept =
    default;
OpenCvVideoVisualizer& OpenCvVideoVisualizer::operator=(
    OpenCvVideoVisualizer&&) noexcept = default;

void OpenCvVideoVisualizer::on_frame(const Frame&,
                                     std::span<const Detection>,
                                     std::span<const Track>,
                                     const FrameMetrics&) {
  throw std::runtime_error(
      "OpenCvVideoVisualizer requires building with OpenCV enabled "
      "(UAV_ENABLE_OPENCV=ON)");
}

void OpenCvVideoVisualizer::close() noexcept {}

const std::filesystem::path& OpenCvVideoVisualizer::output_path()
    const noexcept {
  static const std::filesystem::path empty_path{};
  return impl_ ? impl_->output_path : empty_path;
}

std::size_t OpenCvVideoVisualizer::frames_written() const noexcept {
  return 0U;
}

bool OpenCvVideoVisualizer::is_compiled_with_opencv() noexcept { return false; }

#endif

}  // namespace uav
