#pragma once

#include "uav/tracking/tracker.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace uav {

struct TrackerConfig {
  float iou_threshold{0.20F};
  std::uint32_t min_hits_to_confirm{3U};
  std::uint32_t max_missed_frames{5U};
  float default_dt_seconds{1.0F / 30.0F};
  float position_process_noise{4.0F};
  float size_process_noise{1.0F};
  float velocity_process_noise{2500.0F};
  float position_measurement_noise{2.0F};
  float size_measurement_noise{2.0F};
};

// Constant-velocity 2D bounding-box Kalman filter.
// State vector: [cx, cy, w, h, vx, vy]^T
// Measurement:  [cx, cy, w, h]^T
class KalmanBoxFilter {
 public:
  explicit KalmanBoxFilter(const BoundingBox& initial_bounds,
                           const TrackerConfig& config = {}) noexcept;

  void predict(float dt_seconds) noexcept;
  void update(const BoundingBox& measured_bounds) noexcept;

  [[nodiscard]] BoundingBox current_bounds() const noexcept;
  [[nodiscard]] float velocity_x() const noexcept { return state_[4]; }
  [[nodiscard]] float velocity_y() const noexcept { return state_[5]; }

 private:
  std::array<float, 6> state_{};
  std::array<std::array<float, 6>, 6> cov_{};
  TrackerConfig config_{};
};

class KalmanIoUTracker final : public ITracker {
 public:
  explicit KalmanIoUTracker(TrackerConfig config = {});

  [[nodiscard]] std::vector<Track> update(
      const Frame& frame,
      std::span<const Detection> detections) override;

  [[nodiscard]] const TrackerConfig& config() const noexcept { return config_; }

 private:
  struct TrackEntry {
    TrackId id{};
    KalmanBoxFilter filter;
    TrackState state{TrackState::tentative};
    std::uint32_t age_frames{1U};
    std::uint32_t hits{1U};
    std::uint32_t missed_frames{0U};
  };

  TrackerConfig config_{};
  TrackId next_track_id_{1U};
  std::optional<Timestamp> last_timestamp_{};
  std::vector<TrackEntry> entries_{};
};

}  // namespace uav
