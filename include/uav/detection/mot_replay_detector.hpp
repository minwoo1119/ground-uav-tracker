#pragma once

#include "uav/detection/detector.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace uav {

class MotReplayDetector final : public IDetector {
 public:
  explicit MotReplayDetector(const std::filesystem::path& sequence_dir,
                             std::int32_t class_id = 2,
                             std::string class_name = "DRONE");

  [[nodiscard]] std::vector<Detection> detect(const Frame& frame) override;
  [[nodiscard]] std::size_t total_loaded_annotations() const noexcept {
    return total_annotations_;
  }

 private:
  std::unordered_map<FrameId, std::vector<Detection>> by_frame_{};
  std::size_t total_annotations_{};
};

}  // namespace uav
