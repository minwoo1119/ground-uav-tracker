#pragma once

#include "uav/detection/detector.hpp"
#include "uav/detection/yolox_postprocessor.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>

namespace uav {

struct OnnxYoloXDetectorConfig {
  std::filesystem::path model_path{};
  YoloXPostprocessorConfig postprocessor{};
  std::size_t intra_op_threads{1U};
};

class OnnxYoloXDetector final : public IDetector {
 public:
  explicit OnnxYoloXDetector(OnnxYoloXDetectorConfig config);
  ~OnnxYoloXDetector() override;

  OnnxYoloXDetector(const OnnxYoloXDetector&) = delete;
  OnnxYoloXDetector& operator=(const OnnxYoloXDetector&) = delete;
  OnnxYoloXDetector(OnnxYoloXDetector&&) noexcept;
  OnnxYoloXDetector& operator=(OnnxYoloXDetector&&) noexcept;

  [[nodiscard]] std::vector<Detection> detect(const Frame& frame) override;
  [[nodiscard]] static bool is_compiled_with_onnxruntime() noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_{};
};

}  // namespace uav
