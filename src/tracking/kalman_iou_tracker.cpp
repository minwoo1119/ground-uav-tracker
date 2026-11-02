#include "uav/tracking/kalman_iou_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace uav {
namespace {

struct CandidateMatch {
  std::size_t track_index{};
  std::size_t detection_index{};
  float iou{};
};

void update_axis_pos_vel(float& pos,
                         float& vel,
                         float& p_pos,
                         float& p_pos_vel,
                         float& p_vel,
                         float measured_pos,
                         float r_pos) noexcept {
  const float innovation = measured_pos - pos;
  const float s = p_pos + r_pos;
  if (s <= 1e-6F) {
    return;
  }
  const float k_pos = p_pos / s;
  const float k_vel = p_pos_vel / s;

  pos += k_pos * innovation;
  vel += k_vel * innovation;

  const float old_p_pos = p_pos;
  const float old_p_pos_vel = p_pos_vel;
  p_pos = (1.0F - k_pos) * old_p_pos;
  p_pos_vel = (1.0F - k_pos) * old_p_pos_vel;
  p_vel -= k_vel * old_p_pos_vel;
}

void update_axis_scalar(float& value,
                        float& variance,
                        float measured_value,
                        float r_var) noexcept {
  const float innovation = measured_value - value;
  const float s = variance + r_var;
  if (s <= 1e-6F) {
    return;
  }
  const float k = variance / s;
  value += k * innovation;
  variance = (1.0F - k) * variance;
}

}  // namespace

KalmanBoxFilter::KalmanBoxFilter(const BoundingBox& initial_bounds,
                                 const TrackerConfig& config) noexcept
    : config_(config) {
  const float cx = initial_bounds.x + 0.5F * initial_bounds.width;
  const float cy = initial_bounds.y + 0.5F * initial_bounds.height;
  state_ = {cx, cy, initial_bounds.width, initial_bounds.height, 0.0F, 0.0F};

  cov_[0][0] = 10.0F;
  cov_[1][1] = 10.0F;
  cov_[2][2] = 10.0F;
  cov_[3][3] = 10.0F;
  cov_[4][4] = 1.0e6F;
  cov_[5][5] = 1.0e6F;
}

void KalmanBoxFilter::predict(float dt_seconds) noexcept {
  const float dt = dt_seconds > 0.0F ? dt_seconds : config_.default_dt_seconds;

  // x = F * x
  state_[0] += state_[4] * dt;
  state_[1] += state_[5] * dt;
  state_[2] = std::max(1.0F, state_[2]);
  state_[3] = std::max(1.0F, state_[3]);

  // P = F * P * F^T + Q for coupled (cx, vx) and (cy, vy)
  for (std::size_t pos_idx = 0; pos_idx < 2U; ++pos_idx) {
    const std::size_t vel_idx = pos_idx + 4U;
    const float p_pp = cov_[pos_idx][pos_idx];
    const float p_pv = cov_[pos_idx][vel_idx];
    const float p_vv = cov_[vel_idx][vel_idx];

    cov_[pos_idx][pos_idx] =
        p_pp + 2.0F * dt * p_pv + dt * dt * p_vv + config_.position_process_noise;
    cov_[pos_idx][vel_idx] = p_pv + dt * p_vv;
    cov_[vel_idx][pos_idx] = cov_[pos_idx][vel_idx];
    cov_[vel_idx][vel_idx] = p_vv + config_.velocity_process_noise;
  }

  cov_[2][2] += config_.size_process_noise;
  cov_[3][3] += config_.size_process_noise;
}

void KalmanBoxFilter::update(const BoundingBox& measured_bounds) noexcept {
  const float meas_cx = measured_bounds.x + 0.5F * measured_bounds.width;
  const float meas_cy = measured_bounds.y + 0.5F * measured_bounds.height;

  update_axis_pos_vel(state_[0],
                      state_[4],
                      cov_[0][0],
                      cov_[0][4],
                      cov_[4][4],
                      meas_cx,
                      config_.position_measurement_noise);
  cov_[4][0] = cov_[0][4];

  update_axis_pos_vel(state_[1],
                      state_[5],
                      cov_[1][1],
                      cov_[1][5],
                      cov_[5][5],
                      meas_cy,
                      config_.position_measurement_noise);
  cov_[5][1] = cov_[1][5];

  update_axis_scalar(state_[2],
                     cov_[2][2],
                     measured_bounds.width,
                     config_.size_measurement_noise);
  update_axis_scalar(state_[3],
                     cov_[3][3],
                     measured_bounds.height,
                     config_.size_measurement_noise);

  state_[2] = std::max(1.0F, state_[2]);
  state_[3] = std::max(1.0F, state_[3]);
}

BoundingBox KalmanBoxFilter::current_bounds() const noexcept {
  const float w = std::max(1.0F, state_[2]);
  const float h = std::max(1.0F, state_[3]);
  const float x = std::max(0.0F, state_[0] - 0.5F * w);
  const float y = std::max(0.0F, state_[1] - 0.5F * h);
  return BoundingBox{.x = x, .y = y, .width = w, .height = h};
}

KalmanIoUTracker::KalmanIoUTracker(TrackerConfig config) : config_(config) {
  if (config_.iou_threshold < 0.0F || config_.iou_threshold > 1.0F) {
    throw std::invalid_argument("tracker iou_threshold must be in [0.0, 1.0]");
  }
  if (config_.min_hits_to_confirm == 0U) {
    throw std::invalid_argument("tracker min_hits_to_confirm must be >= 1");
  }
  if (config_.default_dt_seconds <= 0.0F) {
    throw std::invalid_argument("tracker default_dt_seconds must be > 0");
  }
}

std::vector<Track> KalmanIoUTracker::update(
    const Frame& frame,
    std::span<const Detection> detections) {
  float dt_seconds = config_.default_dt_seconds;
  if (last_timestamp_.has_value() && frame.captured_at > *last_timestamp_) {
    const auto delta_us = (frame.captured_at - *last_timestamp_).count();
    dt_seconds = static_cast<float>(delta_us) * 1e-6F;
  }
  last_timestamp_ = frame.captured_at;

  // 1. Predict all existing tracks forward by dt_seconds
  for (auto& entry : entries_) {
    entry.filter.predict(dt_seconds);
  }

  // 2. Build IoU candidate pairs between predicted tracks and valid detections
  std::vector<CandidateMatch> candidates;
  candidates.reserve(entries_.size() * detections.size());
  for (std::size_t t_idx = 0; t_idx < entries_.size(); ++t_idx) {
    const BoundingBox predicted = entries_[t_idx].filter.current_bounds();
    for (std::size_t d_idx = 0; d_idx < detections.size(); ++d_idx) {
      if (!detections[d_idx].bounds.is_valid()) {
        continue;
      }
      const float iou =
          intersection_over_union(predicted, detections[d_idx].bounds);
      if (iou >= config_.iou_threshold) {
        candidates.push_back(CandidateMatch{
            .track_index = t_idx,
            .detection_index = d_idx,
            .iou = iou,
        });
      }
    }
  }

  std::sort(candidates.begin(),
            candidates.end(),
            [](const CandidateMatch& lhs, const CandidateMatch& rhs) {
              if (lhs.iou != rhs.iou) {
                return lhs.iou > rhs.iou;
              }
              return lhs.track_index < rhs.track_index;
            });

  std::vector<bool> track_matched(entries_.size(), false);
  std::vector<bool> detection_matched(detections.size(), false);

  for (const auto& candidate : candidates) {
    if (track_matched[candidate.track_index] ||
        detection_matched[candidate.detection_index]) {
      continue;
    }
    track_matched[candidate.track_index] = true;
    detection_matched[candidate.detection_index] = true;

    auto& entry = entries_[candidate.track_index];
    entry.filter.update(detections[candidate.detection_index].bounds);
    ++entry.age_frames;
    ++entry.hits;
    entry.missed_frames = 0U;
    if (entry.hits >= config_.min_hits_to_confirm) {
      entry.state = TrackState::confirmed;
    }
  }

  // 3. Handle unmatched existing tracks
  for (std::size_t t_idx = 0; t_idx < entries_.size(); ++t_idx) {
    if (track_matched[t_idx]) {
      continue;
    }
    auto& entry = entries_[t_idx];
    ++entry.age_frames;
    ++entry.missed_frames;
    if (entry.missed_frames > config_.max_missed_frames) {
      entry.state = TrackState::removed;
    } else if (entry.state == TrackState::confirmed) {
      entry.state = TrackState::lost;
    }
  }

  // 4. Spawn tentative tracks for unmatched valid detections
  for (std::size_t d_idx = 0; d_idx < detections.size(); ++d_idx) {
    if (detection_matched[d_idx] || !detections[d_idx].bounds.is_valid()) {
      continue;
    }
    const TrackState initial_state = config_.min_hits_to_confirm <= 1U
                                         ? TrackState::confirmed
                                         : TrackState::tentative;
    entries_.push_back(TrackEntry{
        .id = next_track_id_++,
        .filter = KalmanBoxFilter{detections[d_idx].bounds, config_},
        .state = initial_state,
        .age_frames = 1U,
        .hits = 1U,
        .missed_frames = 0U,
    });
  }

  // 5. Collect output snapshot and purge removed tracks from active list
  std::vector<Track> result;
  result.reserve(entries_.size());
  for (const auto& entry : entries_) {
    if (entry.state == TrackState::removed) {
      continue;
    }
    result.push_back(Track{
        .id = entry.id,
        .bounds = entry.filter.current_bounds(),
        .state = entry.state,
        .age_frames = entry.age_frames,
        .missed_frames = entry.missed_frames,
    });
  }

  std::erase_if(entries_, [](const TrackEntry& entry) {
    return entry.state == TrackState::removed;
  });

  return result;
}

}  // namespace uav
