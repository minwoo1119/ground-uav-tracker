#pragma once

#include "uav/core/types.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace uav {

struct YoloXPostprocessorConfig {
  ImageSize input_size{.width = 640U, .height = 640U};
  std::vector<std::uint32_t> strides{8U, 16U, 32U};
  std::vector<std::string> class_names{
      "AIRPLANE", "BIRD", "DRONE", "HELICOPTER"};
  float confidence_threshold{-1.0F};
  float nms_threshold{-1.0F};
  bool class_agnostic_nms{true};
};

// Converts one YOLOX batch item's raw deploy tensor into project detections.
// The expected layout is [anchors, 5 + classes] with decode_in_inference=false.
class YoloXPostprocessor {
 public:
  explicit YoloXPostprocessor(YoloXPostprocessorConfig config);

  [[nodiscard]] std::vector<Detection> process(
      std::span<const float> raw_output,
      ImageSize original_size) const;
  [[nodiscard]] std::size_t expected_anchor_count() const noexcept;
  [[nodiscard]] std::size_t feature_count() const noexcept;

 private:
  YoloXPostprocessorConfig config_;
  std::size_t expected_anchor_count_{};
};

}  // namespace uav
