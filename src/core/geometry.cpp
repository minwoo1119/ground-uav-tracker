#include "gut/core/types.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>

namespace gut {

float intersection_over_union(const BoundingBox& lhs,
                              const BoundingBox& rhs) noexcept {
  if (!lhs.is_valid() || !rhs.is_valid()) {
    return 0.0F;
  }
  const float left = std::max(lhs.x, rhs.x);
  const float top = std::max(lhs.y, rhs.y);
  const float right = std::min(lhs.x + lhs.width, rhs.x + rhs.width);
  const float bottom = std::min(lhs.y + lhs.height, rhs.y + rhs.height);
  const float intersection =
      std::max(0.0F, right - left) * std::max(0.0F, bottom - top);
  const float union_area = lhs.area() + rhs.area() - intersection;
  return union_area > 0.0F ? intersection / union_area : 0.0F;
}

bool FrameView::is_valid() const noexcept {
  if (width == 0U || height == 0U) {
    return false;
  }

  std::uint32_t channels = 0U;
  switch (pixel_format) {
    case PixelFormat::gray8:
      channels = 1U;
      break;
    case PixelFormat::rgb8:
    case PixelFormat::bgr8:
      channels = 3U;
      break;
  }

  const std::uint64_t minimum_stride =
      static_cast<std::uint64_t>(width) * channels;
  const std::uint64_t required_bytes =
      static_cast<std::uint64_t>(stride_bytes) * height;
  return stride_bytes >= minimum_stride &&
         required_bytes <= pixels.size() &&
         required_bytes <= std::numeric_limits<std::size_t>::max();
}

}  // namespace gut
