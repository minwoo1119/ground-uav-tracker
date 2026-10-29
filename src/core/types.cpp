#include "uav/core/types.hpp"

#include <algorithm>

namespace uav {

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

}  // namespace uav
