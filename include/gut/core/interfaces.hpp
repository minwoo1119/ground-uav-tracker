#pragma once

#include "gut/core/types.hpp"

#include <optional>
#include <vector>

namespace gut {

class IFrameSource {
 public:
  virtual ~IFrameSource() = default;
  [[nodiscard]] virtual std::optional<FrameView> next() = 0;
};

class IDetector {
 public:
  virtual ~IDetector() = default;
  [[nodiscard]] virtual std::vector<Detection> detect(const FrameView& frame) = 0;
};

class ITracker {
 public:
  virtual ~ITracker() = default;
  [[nodiscard]] virtual std::vector<Track> update(
      FrameId frame_id,
      Timestamp captured_at,
      std::span<const Detection> detections) = 0;
};

class IResultSink {
 public:
  virtual ~IResultSink() = default;
  virtual void write(const FrameResult& result) = 0;
};

}  // namespace gut
