#pragma once

#include "uav/detection/onnx_yolox_detector.hpp"
#include "uav/tracking/kalman_iou_tracker.hpp"

#include <cstddef>
#include <filesystem>

namespace uav {

struct RuntimeConfig {
  std::size_t max_frames{};
  OnnxYoloXDetectorConfig detector{};
  TrackerConfig tracker{};
};

// Loads the project's strict two-level YAML subset. Relative model paths are
// resolved against the configuration file directory.
[[nodiscard]] RuntimeConfig load_runtime_config(
    const std::filesystem::path& config_path);

}  // namespace uav
