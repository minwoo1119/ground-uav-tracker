#pragma once

#include "uav/core/types.hpp"

#include <vector>

namespace uav {

struct YoloXInput {
  std::vector<float> tensor{};
  float scale{};
  ImageSize resized_size{};
};

// Produces an unnormalized BGR CHW float tensor using the YOLOX validation
// transform: aspect-preserving bilinear resize at the canvas top-left and 114
// padding. The batch dimension is implicit and always one.
[[nodiscard]] YoloXInput preprocess_yolox_bgr(
    const Frame& frame,
    ImageSize input_size = {.width = 640U, .height = 640U});

}  // namespace uav
