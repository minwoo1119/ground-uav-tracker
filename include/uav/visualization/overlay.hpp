#pragma once

#include "uav/core/types.hpp"
#include "uav/telemetry/metrics_sink.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace uav {

struct BgrColor {
  std::uint8_t b{};
  std::uint8_t g{};
  std::uint8_t r{};

  [[nodiscard]] constexpr bool operator==(const BgrColor&) const noexcept =
      default;
};

struct PixelRect {
  int x{};
  int y{};
  int width{};
  int height{};

  [[nodiscard]] constexpr bool operator==(const PixelRect&) const noexcept =
      default;
};

[[nodiscard]] std::optional<PixelRect> clamp_bounding_box(
    const BoundingBox& bounds,
    const ImageSize& image_size) noexcept;

[[nodiscard]] std::string_view track_state_tag(TrackState state) noexcept;
[[nodiscard]] BgrColor track_state_color(TrackState state) noexcept;
[[nodiscard]] BgrColor detection_box_color() noexcept;

[[nodiscard]] std::string format_detection_label(const Detection& detection);
[[nodiscard]] std::string format_track_label(const Track& track);
[[nodiscard]] std::string format_hud_summary(const FrameMetrics& metrics);

[[nodiscard]] std::string select_video_fourcc(
    const std::filesystem::path& output_path);

void validate_video_output_path(
    const std::filesystem::path& output_path,
    const std::filesystem::path& input_path,
    const std::optional<std::filesystem::path>& output_tracks_path =
        std::nullopt);

}  // namespace uav
