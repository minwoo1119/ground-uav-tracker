#include "uav/detection/yolox_preprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace uav {
namespace {

[[nodiscard]] float sample_bgr(const Frame& frame,
                               std::uint32_t x,
                               std::uint32_t y,
                               std::size_t channel) noexcept {
  const auto offset = static_cast<std::size_t>(y) * frame.row_stride_bytes +
                      static_cast<std::size_t>(x) * 3U + channel;
  return static_cast<float>(frame.pixel_data[offset]);
}

}  // namespace

YoloXInput preprocess_yolox_bgr(const Frame& frame, ImageSize input_size) {
  if (!frame.is_valid() || !frame.has_pixels() ||
      frame.pixel_format != PixelFormat::bgr8) {
    throw std::invalid_argument(
        "YOLOX preprocessing requires a valid CPU-owned BGR8 frame");
  }
  if (!input_size.is_valid()) {
    throw std::invalid_argument("YOLOX input size must be positive");
  }
  if (input_size.width >
      std::numeric_limits<std::size_t>::max() / input_size.height) {
    throw std::invalid_argument("YOLOX input tensor dimensions overflow size_t");
  }

  const double scale = std::min(
      static_cast<double>(input_size.width) / frame.image_size.width,
      static_cast<double>(input_size.height) / frame.image_size.height);
  const auto resized_width = std::min(
      input_size.width,
      static_cast<std::uint32_t>(frame.image_size.width * scale));
  const auto resized_height = std::min(
      input_size.height,
      static_cast<std::uint32_t>(frame.image_size.height * scale));
  if (resized_width == 0U || resized_height == 0U) {
    throw std::invalid_argument(
        "YOLOX resize produced a zero-sized image for the requested input");
  }

  const auto plane_size =
      static_cast<std::size_t>(input_size.width) * input_size.height;
  if (plane_size > std::numeric_limits<std::size_t>::max() / 3U) {
    throw std::invalid_argument("YOLOX input tensor size overflows size_t");
  }
  std::vector<float> tensor(plane_size * 3U, 114.0F);
  const double source_per_output_x =
      static_cast<double>(frame.image_size.width) / resized_width;
  const double source_per_output_y =
      static_cast<double>(frame.image_size.height) / resized_height;

  for (std::uint32_t output_y = 0U; output_y < resized_height; ++output_y) {
    const double source_y =
        (static_cast<double>(output_y) + 0.5) * source_per_output_y - 0.5;
    const auto raw_y0 = static_cast<std::int64_t>(std::floor(source_y));
    const auto raw_y1 = raw_y0 + 1;
    const auto y0 = static_cast<std::uint32_t>(std::clamp<std::int64_t>(
        raw_y0, 0, static_cast<std::int64_t>(frame.image_size.height) - 1));
    const auto y1 = static_cast<std::uint32_t>(std::clamp<std::int64_t>(
        raw_y1, 0, static_cast<std::int64_t>(frame.image_size.height) - 1));
    const float y_weight = static_cast<float>(source_y - raw_y0);

    for (std::uint32_t output_x = 0U; output_x < resized_width; ++output_x) {
      const double source_x =
          (static_cast<double>(output_x) + 0.5) * source_per_output_x - 0.5;
      const auto raw_x0 = static_cast<std::int64_t>(std::floor(source_x));
      const auto raw_x1 = raw_x0 + 1;
      const auto x0 = static_cast<std::uint32_t>(std::clamp<std::int64_t>(
          raw_x0, 0, static_cast<std::int64_t>(frame.image_size.width) - 1));
      const auto x1 = static_cast<std::uint32_t>(std::clamp<std::int64_t>(
          raw_x1, 0, static_cast<std::int64_t>(frame.image_size.width) - 1));
      const float x_weight = static_cast<float>(source_x - raw_x0);

      for (std::size_t channel = 0U; channel < 3U; ++channel) {
        const float top =
            sample_bgr(frame, x0, y0, channel) * (1.0F - x_weight) +
            sample_bgr(frame, x1, y0, channel) * x_weight;
        const float bottom =
            sample_bgr(frame, x0, y1, channel) * (1.0F - x_weight) +
            sample_bgr(frame, x1, y1, channel) * x_weight;
        const float interpolated =
            top * (1.0F - y_weight) + bottom * y_weight;
        tensor[channel * plane_size +
               static_cast<std::size_t>(output_y) * input_size.width +
               output_x] =
            std::round(std::clamp(interpolated, 0.0F, 255.0F));
      }
    }
  }

  return YoloXInput{
      .tensor = std::move(tensor),
      .scale = static_cast<float>(scale),
      .resized_size = {.width = resized_width, .height = resized_height},
  };
}

}  // namespace uav
