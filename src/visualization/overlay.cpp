#include "uav/visualization/overlay.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace uav {
namespace {

[[nodiscard]] std::string lowercase_ascii(std::string_view input) {
  std::string result;
  result.reserve(input.size());
  for (const char ch : input) {
    result.push_back(static_cast<char>(
        std::tolower(static_cast<unsigned char>(ch))));
  }
  return result;
}

[[nodiscard]] std::filesystem::path normalize_for_comparison(
    const std::filesystem::path& path) {
  std::error_code error;
  auto canonical = std::filesystem::weakly_canonical(path, error);
  if (!error) {
    return canonical.lexically_normal();
  }
  return path.lexically_normal();
}

[[nodiscard]] bool paths_refer_to_same_target(
    const std::filesystem::path& lhs,
    const std::filesystem::path& rhs) {
  std::error_code error;
  if (std::filesystem::equivalent(lhs, rhs, error) && !error) {
    return true;
  }
  const auto lhs_norm = normalize_for_comparison(lhs);
  const auto rhs_norm = normalize_for_comparison(rhs);
#ifdef _WIN32
  return lowercase_ascii(lhs_norm.string()) ==
         lowercase_ascii(rhs_norm.string());
#else
  return lhs_norm == rhs_norm;
#endif
}

}  // namespace

std::optional<PixelRect> clamp_bounding_box(
    const BoundingBox& bounds,
    const ImageSize& image_size) noexcept {
  if (!image_size.is_valid()) {
    return std::nullopt;
  }
  constexpr auto max_int =
      static_cast<std::uint32_t>(std::numeric_limits<int>::max());
  if (image_size.width > max_int || image_size.height > max_int) {
    return std::nullopt;
  }
  if (!std::isfinite(bounds.x) || !std::isfinite(bounds.y) ||
      !std::isfinite(bounds.width) || !std::isfinite(bounds.height)) {
    return std::nullopt;
  }
  if (bounds.width <= 0.0F || bounds.height <= 0.0F) {
    return std::nullopt;
  }

  const double max_w = static_cast<double>(image_size.width);
  const double max_h = static_cast<double>(image_size.height);
  const double raw_x1 = static_cast<double>(bounds.x);
  const double raw_y1 = static_cast<double>(bounds.y);
  const double raw_x2 = raw_x1 + static_cast<double>(bounds.width);
  const double raw_y2 = raw_y1 + static_cast<double>(bounds.height);

  const double clamped_x1 = std::clamp(std::floor(raw_x1), 0.0, max_w);
  const double clamped_y1 = std::clamp(std::floor(raw_y1), 0.0, max_h);
  const double clamped_x2 = std::clamp(std::ceil(raw_x2), 0.0, max_w);
  const double clamped_y2 = std::clamp(std::ceil(raw_y2), 0.0, max_h);

  const int x1 = static_cast<int>(clamped_x1);
  const int y1 = static_cast<int>(clamped_y1);
  const int x2 = static_cast<int>(clamped_x2);
  const int y2 = static_cast<int>(clamped_y2);

  if (x2 <= x1 || y2 <= y1) {
    return std::nullopt;
  }
  return PixelRect{
      .x = x1,
      .y = y1,
      .width = x2 - x1,
      .height = y2 - y1,
  };
}

std::string_view track_state_tag(TrackState state) noexcept {
  switch (state) {
    case TrackState::tentative:
      return "TENTATIVE";
    case TrackState::confirmed:
      return "CONFIRMED";
    case TrackState::lost:
      return "LOST";
    case TrackState::removed:
      return "REMOVED";
  }
  return "UNKNOWN";
}

BgrColor track_state_color(TrackState state) noexcept {
  switch (state) {
    case TrackState::confirmed:
      return BgrColor{.b = 40U, .g = 220U, .r = 40U};
    case TrackState::tentative:
      return BgrColor{.b = 0U, .g = 215U, .r = 255U};
    case TrackState::lost:
      return BgrColor{.b = 30U, .g = 80U, .r = 240U};
    case TrackState::removed:
      return BgrColor{.b = 140U, .g = 140U, .r = 140U};
  }
  return BgrColor{.b = 255U, .g = 255U, .r = 255U};
}

BgrColor detection_box_color() noexcept {
  return BgrColor{.b = 245U, .g = 210U, .r = 40U};
}

std::string format_detection_label(const Detection& detection) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  if (!detection.class_name.empty()) {
    out << detection.class_name;
  } else {
    out << "class_" << detection.class_id;
  }
  out << ' ';
  if (std::isfinite(detection.confidence)) {
    out << std::fixed << std::setprecision(2) << detection.confidence;
  } else {
    out << '?';
  }
  return out.str();
}

std::string format_track_label(const Track& track) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << '#' << track.id << " [" << track_state_tag(track.state) << ']';
  return out.str();
}

std::string format_hud_summary(const FrameMetrics& metrics) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  const double captured_ms =
      static_cast<double>(metrics.captured_at.count()) / 1000.0;
  const double proc_ms =
      static_cast<double>(metrics.processing_time.count()) / 1000.0;
  out << std::fixed << std::setprecision(1) << "Frame " << metrics.frame_id
      << " | t=" << captured_ms << "ms | Det:" << metrics.detection_count
      << " | Trk:" << metrics.track_count << " | Proc:" << proc_ms << "ms";
  return out.str();
}

std::string select_video_fourcc(const std::filesystem::path& output_path) {
  if (output_path.empty()) {
    throw std::invalid_argument("video output path must not be empty");
  }
  const std::string ext = lowercase_ascii(output_path.extension().string());
  if (ext == ".avi") {
    return "MJPG";
  }
  if (ext == ".mp4") {
    return "mp4v";
  }
  throw std::invalid_argument(
      "unsupported video output extension (supported: .avi, .mp4): " +
      output_path.string());
}

void validate_video_output_path(
    const std::filesystem::path& output_path,
    const std::filesystem::path& input_path,
    const std::optional<std::filesystem::path>& output_tracks_path) {
  if (output_path.empty()) {
    throw std::invalid_argument("video output path must not be empty");
  }
  static_cast<void>(select_video_fourcc(output_path));

  const auto parent = output_path.parent_path();
  if (!parent.empty()) {
    std::error_code error;
    if (!std::filesystem::exists(parent, error) || error ||
        !std::filesystem::is_directory(parent, error) || error) {
      throw std::invalid_argument(
          "video output parent directory does not exist: " + parent.string());
    }
  }

  if (std::filesystem::exists(output_path)) {
    throw std::invalid_argument("video output already exists: " +
                                output_path.string());
  }

  if (!input_path.empty() &&
      paths_refer_to_same_target(output_path, input_path)) {
    throw std::invalid_argument(
        "video output must not overwrite input video: " + input_path.string());
  }

  if (output_tracks_path.has_value() && !output_tracks_path->empty() &&
      paths_refer_to_same_target(output_path, *output_tracks_path)) {
    throw std::invalid_argument(
        "video output path must not match --output-tracks path: " +
        output_tracks_path->string());
  }
}

}  // namespace uav
