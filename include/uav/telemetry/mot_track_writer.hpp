#pragma once

#include "uav/telemetry/track_sink.hpp"

#include <filesystem>
#include <fstream>
#include <span>

namespace uav {

void validate_mot_track_output_path(
    const std::filesystem::path& output_path,
    const std::filesystem::path& sequence_dir);

class MotTrackWriter final : public ITrackSink {
 public:
  explicit MotTrackWriter(const std::filesystem::path& output_path);

  void write(const Frame& frame, std::span<const Track> tracks) override;

  [[nodiscard]] const std::filesystem::path& output_path() const noexcept {
    return output_path_;
  }

 private:
  std::filesystem::path output_path_;
  std::ofstream stream_;
};

}  // namespace uav
