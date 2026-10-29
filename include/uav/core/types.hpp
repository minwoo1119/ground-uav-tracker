#pragma once

#include <chrono>
#include <cstdint>
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

// Pixel storage, ownership, and CPU/GPU memory type remain intentionally TBD.
struct Frame {
  FrameId id{};
  Timestamp captured_at{};
  ImageSize image_size{};

  [[nodiscard]] constexpr bool is_valid() const noexcept {
    return image_size.is_valid();
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
