#include "gut/core/pipeline.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace {

class SingleFrameSource final : public gut::IFrameSource {
 public:
  std::optional<gut::FrameView> next() override {
    if (done_) {
      return std::nullopt;
    }
    done_ = true;
    return gut::FrameView{
        .id = 1,
        .captured_at = gut::Timestamp{0},
        .width = 2,
        .height = 2,
        .stride_bytes = 2,
        .pixel_format = gut::PixelFormat::gray8,
        .pixels = pixels_,
    };
  }

 private:
  std::array<std::byte, 4> pixels_{};
  bool done_{};
};

class EmptyDetector final : public gut::IDetector {
 public:
  std::vector<gut::Detection> detect(const gut::FrameView&) override {
    return {};
  }
};

class EmptyTracker final : public gut::ITracker {
 public:
  std::vector<gut::Track> update(gut::FrameId,
                                 gut::Timestamp,
                                 std::span<const gut::Detection>) override {
    return {};
  }
};

class CountingSink final : public gut::IResultSink {
 public:
  void write(const gut::FrameResult&) override { ++writes; }
  std::size_t writes{};
};

void print_usage(std::ostream& output) {
  output << "Usage: ground-uav-tracker <command>\n"
            "\n"
            "Commands:\n"
            "  self-check  Run the model-free pipeline smoke check\n"
            "  --help      Show this help\n";
}

int run_self_check() {
  SingleFrameSource source;
  EmptyDetector detector;
  EmptyTracker tracker;
  CountingSink sink;
  gut::TrackingPipeline pipeline{source, detector, tracker, sink};
  const auto summary = pipeline.run();
  const bool ok = summary.frames_processed == 1U &&
                  summary.detections_produced == 0U &&
                  summary.tracks_produced == 0U && sink.writes == 1U;
  std::cout << (ok ? "self-check: OK" : "self-check: FAILED") << '\n'
            << "frames=" << summary.frames_processed
            << " detections=" << summary.detections_produced
            << " tracks=" << summary.tracks_produced << '\n';
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc == 2) {
    const std::string_view command{argv[1]};
    if (command == "--help" || command == "-h") {
      print_usage(std::cout);
      return 0;
    }
    if (command == "self-check") {
      return run_self_check();
    }
  }
  print_usage(std::cerr);
  return 2;
}
