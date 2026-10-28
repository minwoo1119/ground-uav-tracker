#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace gut {

using FrameId = std::uint64_t;
using TrackId = std::uint64_t;
using Timestamp = std::chrono::microseconds;

enum class PixelFormat { gray8, rgb8, bgr8 };

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

struct FrameView {
  FrameId id{};
  Timestamp captured_at{};
  std::uint32_t width{};
  std::uint32_t height{};
  std::uint32_t stride_bytes{};
  PixelFormat pixel_format{PixelFormat::bgr8};
  std::span<const std::byte> pixels{};

  [[nodiscard]] bool is_valid() const noexcept;
};

struct Detection {
  BoundingBox bounds{};
  float confidence{};
  std::int32_t class_id{};
  std::string class_name{};
};

enum class TrackState { tentative, confirmed, lost };

struct Track {
  TrackId id{};
  BoundingBox bounds{};
  float confidence{};
  std::int32_t class_id{};
  TrackState state{TrackState::tentative};
  std::uint32_t age_frames{};
  std::uint32_t missed_frames{};
};

struct FrameResult {
  FrameId frame_id{};
  Timestamp captured_at{};
  std::vector<Track> tracks{};
};

}  // namespace gut
