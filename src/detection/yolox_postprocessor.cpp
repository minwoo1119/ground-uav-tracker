#include "uav/detection/yolox_postprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>

namespace uav {
namespace {

struct DecodedBox {
  float left{};
  float top{};
  float right{};
  float bottom{};
};

struct Candidate {
  Detection detection{};
  DecodedBox nms_box{};
  std::size_t anchor_index{};
};

[[nodiscard]] bool is_probability(float value) noexcept {
  return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
}

[[nodiscard]] BoundingBox clip_box(float center_x,
                                   float center_y,
                                   float width,
                                   float height,
                                   float scale,
                                   ImageSize original_size) {
  const float left = std::clamp(
      (center_x - width * 0.5F) / scale, 0.0F,
      static_cast<float>(original_size.width));
  const float top = std::clamp(
      (center_y - height * 0.5F) / scale, 0.0F,
      static_cast<float>(original_size.height));
  const float right = std::clamp(
      (center_x + width * 0.5F) / scale, 0.0F,
      static_cast<float>(original_size.width));
  const float bottom = std::clamp(
      (center_y + height * 0.5F) / scale, 0.0F,
      static_cast<float>(original_size.height));
  return BoundingBox{
      .x = left,
      .y = top,
      .width = right - left,
      .height = bottom - top,
  };
}

[[nodiscard]] float decoded_iou(const DecodedBox& lhs,
                                const DecodedBox& rhs) noexcept {
  const float intersection_width =
      std::max(0.0F, std::min(lhs.right, rhs.right) -
                         std::max(lhs.left, rhs.left));
  const float intersection_height =
      std::max(0.0F, std::min(lhs.bottom, rhs.bottom) -
                         std::max(lhs.top, rhs.top));
  const float intersection = intersection_width * intersection_height;
  const float lhs_area =
      std::max(0.0F, lhs.right - lhs.left) *
      std::max(0.0F, lhs.bottom - lhs.top);
  const float rhs_area =
      std::max(0.0F, rhs.right - rhs.left) *
      std::max(0.0F, rhs.bottom - rhs.top);
  const float union_area = lhs_area + rhs_area - intersection;
  return union_area > 0.0F ? intersection / union_area : 0.0F;
}

}  // namespace

YoloXPostprocessor::YoloXPostprocessor(YoloXPostprocessorConfig config)
    : config_(std::move(config)) {
  if (!config_.input_size.is_valid()) {
    throw std::invalid_argument("YOLOX input size must be positive");
  }
  if (config_.strides.empty()) {
    throw std::invalid_argument("YOLOX strides must not be empty");
  }
  if (config_.class_names.empty()) {
    throw std::invalid_argument("YOLOX class names must not be empty");
  }
  if (config_.class_names.size() >
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
      std::any_of(config_.class_names.begin(), config_.class_names.end(),
                  [](const std::string& name) { return name.empty(); })) {
    throw std::invalid_argument(
        "YOLOX class names must be non-empty and fit int32 class IDs");
  }
  if (!std::isfinite(config_.confidence_threshold) ||
      config_.confidence_threshold < 0.0F ||
      config_.confidence_threshold > 1.0F) {
    throw std::invalid_argument(
        "YOLOX confidence threshold must be within [0, 1]");
  }
  if (!std::isfinite(config_.nms_threshold) ||
      config_.nms_threshold < 0.0F || config_.nms_threshold > 1.0F) {
    throw std::invalid_argument("YOLOX NMS threshold must be within [0, 1]");
  }

  for (const auto stride : config_.strides) {
    if (stride == 0U || config_.input_size.width % stride != 0U ||
        config_.input_size.height % stride != 0U) {
      throw std::invalid_argument(
          "YOLOX input dimensions must be divisible by every stride");
    }
    const auto grid_width = config_.input_size.width / stride;
    const auto grid_height = config_.input_size.height / stride;
    const auto level_count = static_cast<std::size_t>(grid_width) * grid_height;
    if (level_count >
        std::numeric_limits<std::size_t>::max() - expected_anchor_count_) {
      throw std::invalid_argument("YOLOX anchor count overflows size_t");
    }
    expected_anchor_count_ += level_count;
  }
}

std::vector<Detection> YoloXPostprocessor::process(
    std::span<const float> raw_output,
    ImageSize original_size) const {
  if (!original_size.is_valid()) {
    throw std::invalid_argument("original image size must be positive");
  }
  const auto features = feature_count();
  if (expected_anchor_count_ >
      std::numeric_limits<std::size_t>::max() / features ||
      raw_output.size() != expected_anchor_count_ * features) {
    throw std::invalid_argument(
        "YOLOX raw output length does not match anchors * features");
  }

  const float scale = std::min(
      static_cast<float>(config_.input_size.width) / original_size.width,
      static_cast<float>(config_.input_size.height) / original_size.height);
  std::vector<Candidate> candidates;
  std::size_t anchor_index = 0U;

  for (const auto stride : config_.strides) {
    const auto grid_width = config_.input_size.width / stride;
    const auto grid_height = config_.input_size.height / stride;
    for (std::uint32_t grid_y = 0U; grid_y < grid_height; ++grid_y) {
      for (std::uint32_t grid_x = 0U; grid_x < grid_width;
           ++grid_x, ++anchor_index) {
        const auto offset = anchor_index * features;
        const auto row = raw_output.subspan(offset, features);
        if (!std::all_of(row.begin(), row.end(), [](float value) {
              return std::isfinite(value);
            })) {
          throw std::invalid_argument("YOLOX raw output contains NaN or Inf");
        }
        if (!is_probability(row[4])) {
          throw std::invalid_argument(
              "YOLOX objectness must be a probability within [0, 1]");
        }

        std::size_t class_index = 0U;
        float best_class_probability = row[5];
        if (!is_probability(best_class_probability)) {
          throw std::invalid_argument(
              "YOLOX class score must be a probability within [0, 1]");
        }
        for (std::size_t index = 1U; index < config_.class_names.size();
             ++index) {
          const float probability = row[5U + index];
          if (!is_probability(probability)) {
            throw std::invalid_argument(
                "YOLOX class score must be a probability within [0, 1]");
          }
          if (probability > best_class_probability) {
            best_class_probability = probability;
            class_index = index;
          }
        }

        const float confidence = row[4] * best_class_probability;
        if (confidence < config_.confidence_threshold) {
          continue;
        }

        const float decoded_center_x =
            (row[0] + static_cast<float>(grid_x)) * stride;
        const float decoded_center_y =
            (row[1] + static_cast<float>(grid_y)) * stride;
        const float decoded_width = std::exp(row[2]) * stride;
        const float decoded_height = std::exp(row[3]) * stride;
        if (!std::isfinite(decoded_center_x) ||
            !std::isfinite(decoded_center_y) ||
            !std::isfinite(decoded_width) ||
            !std::isfinite(decoded_height)) {
          throw std::invalid_argument("YOLOX decoded box contains NaN or Inf");
        }
        const auto bounds = clip_box(decoded_center_x, decoded_center_y,
                                     decoded_width, decoded_height, scale,
                                     original_size);
        if (!bounds.is_valid()) {
          continue;
        }
        candidates.push_back(Candidate{
            .detection = Detection{
                .class_id = static_cast<std::int32_t>(class_index),
                .confidence = confidence,
                .bounds = bounds,
                .class_name = config_.class_names[class_index],
            },
            .nms_box = DecodedBox{
                .left = decoded_center_x - decoded_width * 0.5F,
                .top = decoded_center_y - decoded_height * 0.5F,
                .right = decoded_center_x + decoded_width * 0.5F,
                .bottom = decoded_center_y + decoded_height * 0.5F,
            },
            .anchor_index = anchor_index,
        });
      }
    }
  }

  std::stable_sort(candidates.begin(), candidates.end(),
                   [](const Candidate& lhs, const Candidate& rhs) {
                     if (lhs.detection.confidence != rhs.detection.confidence) {
                       return lhs.detection.confidence > rhs.detection.confidence;
                     }
                     return lhs.anchor_index < rhs.anchor_index;
                   });

  std::vector<Candidate> accepted_candidates;
  accepted_candidates.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    const bool suppressed = std::any_of(
        accepted_candidates.begin(), accepted_candidates.end(),
        [&](const Candidate& accepted) {
          const bool comparable =
              config_.class_agnostic_nms ||
              accepted.detection.class_id == candidate.detection.class_id;
          return comparable &&
                 decoded_iou(accepted.nms_box, candidate.nms_box) >
                     config_.nms_threshold;
        });
    if (!suppressed) {
      accepted_candidates.push_back(candidate);
    }
  }
  std::vector<Detection> detections;
  detections.reserve(accepted_candidates.size());
  std::transform(accepted_candidates.begin(), accepted_candidates.end(),
                 std::back_inserter(detections),
                 [](const Candidate& candidate) {
                   return candidate.detection;
                 });
  return detections;
}

std::size_t YoloXPostprocessor::expected_anchor_count() const noexcept {
  return expected_anchor_count_;
}

std::size_t YoloXPostprocessor::feature_count() const noexcept {
  return 5U + config_.class_names.size();
}

}  // namespace uav
