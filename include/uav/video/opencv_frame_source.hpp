#pragma once

#include "uav/video/frame_source.hpp"

#include <filesystem>
#include <memory>
#include <optional>

namespace uav {

class OpenCvFrameSource final : public IFrameSource {
 public:
  explicit OpenCvFrameSource(const std::filesystem::path& video_path);
  ~OpenCvFrameSource() override;

  OpenCvFrameSource(const OpenCvFrameSource&) = delete;
  OpenCvFrameSource& operator=(const OpenCvFrameSource&) = delete;
  OpenCvFrameSource(OpenCvFrameSource&&) noexcept;
  OpenCvFrameSource& operator=(OpenCvFrameSource&&) noexcept;

  [[nodiscard]] std::optional<Frame> next() override;
  [[nodiscard]] static bool is_compiled_with_opencv() noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_{};
};

}  // namespace uav
