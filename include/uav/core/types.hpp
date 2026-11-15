#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace uav {

using FrameId = std::uint64_t;
using TrackId = std::uint64_t;
using Timestamp = std::chrono::microseconds;

struct ImageSize {
  std::uint32_t width{};
  std::uint32_t height{};

  [[nodiscard]] constexpr bool is_valid() const noexcept {
    return width > 0U && height > 0U;
  }
};

struct BoundingBox {
  float x{};
  float y{};
  float width{};
  float height{};

  [[nodiscard]] constexpr bool is_valid() const noexcept {
    return x >= 0.0F && y >= 0.0F && width > 0.0F && height > 0.0F;
  }
  [[nodiscard]] constexpr float area() const noexcept {
    return is_valid() ? width * height : 0.0F;
  }
};

[[nodiscard]] float intersection_over_union(const BoundingBox& lhs,
                                            const BoundingBox& rhs) noexcept;

enum class PixelFormat {
  none,
  bgr8,
};

// CPU pixels are owned by the Frame and remain valid for its lifetime. A
// metadata-only frame uses PixelFormat::none with an empty buffer and zero
// stride. GPU/zero-copy memory is intentionally a separate future contract.
struct Frame {
  FrameId id{};
  Timestamp captured_at{};
  ImageSize image_size{};
  PixelFormat pixel_format{PixelFormat::none};
  std::size_t row_stride_bytes{};
  std::vector<std::uint8_t> pixel_data{};

  [[nodiscard]] bool has_pixels() const noexcept {
    return pixel_format != PixelFormat::none;
  }

  [[nodiscard]] bool is_valid() const noexcept {
    if (!image_size.is_valid()) {
      return false;
    }
    if (pixel_format == PixelFormat::none) {
      return row_stride_bytes == 0U && pixel_data.empty();
    }
    if (pixel_format != PixelFormat::bgr8 ||
        image_size.width >
            std::numeric_limits<std::size_t>::max() / 3U) {
      return false;
    }
    const auto minimum_stride =
        static_cast<std::size_t>(image_size.width) * 3U;
    if (row_stride_bytes < minimum_stride ||
        image_size.height >
            std::numeric_limits<std::size_t>::max() / row_stride_bytes) {
      return false;
    }
    return pixel_data.size() ==
           row_stride_bytes * static_cast<std::size_t>(image_size.height);
  }
};

struct Detection {
  std::int32_t class_id{};
  float confidence{};
  BoundingBox bounds{};
  std::string class_name{};
};

enum class TrackState {
  tentative,
  confirmed,
  lost,
  removed,
};

struct Track {
  TrackId id{};
  BoundingBox bounds{};
  TrackState state{TrackState::tentative};
  std::uint32_t age_frames{};
  std::uint32_t missed_frames{};
};

struct FrameResult {
  FrameId frame_id{};
  Timestamp captured_at{};
  std::vector<Track> tracks{};
};

}  // namespace uav
