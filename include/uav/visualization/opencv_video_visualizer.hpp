#pragma once

#include "uav/core/types.hpp"
#include "uav/telemetry/frame_observer.hpp"
#include "uav/telemetry/metrics_sink.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>

namespace uav {

struct OverlayOptions {
  bool draw_detections{true};
  bool draw_tracks{true};
  bool draw_hud{true};
};

struct VideoVisualizerConfig {
  std::filesystem::path output_path{};
  ImageSize frame_size{};
  double fps{30.0};
  OverlayOptions overlay{};
};

[[nodiscard]] Frame render_overlay_bgr(
    const Frame& frame,
    std::span<const Detection> detections,
    std::span<const Track> tracks,
    const FrameMetrics& metrics,
    const OverlayOptions& options = {});

class OpenCvVideoVisualizer final : public IFrameObserver {
 public:
  explicit OpenCvVideoVisualizer(VideoVisualizerConfig config);
  ~OpenCvVideoVisualizer() override;

  OpenCvVideoVisualizer(const OpenCvVideoVisualizer&) = delete;
  OpenCvVideoVisualizer& operator=(const OpenCvVideoVisualizer&) = delete;
  OpenCvVideoVisualizer(OpenCvVideoVisualizer&&) noexcept;
  OpenCvVideoVisualizer& operator=(OpenCvVideoVisualizer&&) noexcept;

  void on_frame(const Frame& frame,
                std::span<const Detection> detections,
                std::span<const Track> tracks,
                const FrameMetrics& metrics) override;

  void close() noexcept;

  [[nodiscard]] const std::filesystem::path& output_path() const noexcept;
  [[nodiscard]] std::size_t frames_written() const noexcept;
  [[nodiscard]] static bool is_compiled_with_opencv() noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_{};
};

}  // namespace uav
