#pragma once

#include "uav/video/frame_source.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace uav {

struct MotSequenceInfo {
  std::string name{};
  std::uint32_t frame_rate{30U};
  std::uint64_t seq_length{};
  ImageSize image_size{};
};

class MotSequenceFrameSource final : public IFrameSource {
 public:
  explicit MotSequenceFrameSource(const std::filesystem::path& sequence_dir);

  [[nodiscard]] std::optional<Frame> next() override;
  [[nodiscard]] const MotSequenceInfo& info() const noexcept { return info_; }

 private:
  MotSequenceInfo info_{};
  FrameId current_frame_{1U};
};

}  // namespace uav
