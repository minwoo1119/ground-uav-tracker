#include "uav/telemetry/mot_track_writer.hpp"

#include <array>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>

namespace uav {

void validate_mot_track_output_path(
    const std::filesystem::path& output_path,
    const std::filesystem::path& sequence_dir) {
  const std::array protected_inputs{
      sequence_dir / "seqinfo.ini",
      sequence_dir / "gt" / "gt.txt",
  };

  for (const auto& input_path : protected_inputs) {
    std::error_code error;
    const bool same_file =
        std::filesystem::equivalent(output_path, input_path, error);
    if (!error && same_file) {
      throw std::invalid_argument(
          "MOT track output must not overwrite sequence input: " +
          input_path.string());
    }
  }
}

MotTrackWriter::MotTrackWriter(const std::filesystem::path& output_path)
    : output_path_(output_path) {
  if (output_path_.empty()) {
    throw std::invalid_argument("MOT track output path must not be empty");
  }
  if (std::filesystem::exists(output_path_)) {
    throw std::invalid_argument("MOT track output already exists: " +
                                output_path_.string());
  }

  stream_.open(output_path_, std::ios::out);
  if (!stream_.is_open()) {
    throw std::runtime_error("failed to open MOT track output: " +
                             output_path_.string());
  }
  stream_.imbue(std::locale::classic());
}

void MotTrackWriter::write(const Frame& frame,
                           std::span<const Track> tracks) {
  if (frame.id == 0U) {
    throw std::invalid_argument("MOT track output requires frame ID >= 1");
  }

  std::ostringstream frame_output;
  frame_output.imbue(std::locale::classic());
  frame_output << std::fixed << std::setprecision(6);

  for (const auto& track : tracks) {
    if (track.state == TrackState::removed) {
      continue;
    }
    if (track.id == 0U) {
      throw std::invalid_argument("MOT track output requires track ID >= 1");
    }
    if (!track.bounds.is_valid()) {
      throw std::invalid_argument(
          "MOT track output received an invalid bounding box");
    }

    // MOTChallenge result columns:
    // frame, id, bb_left, bb_top, bb_width, bb_height, confidence, x, y, z.
    // Track confidence and 3D coordinates are unavailable, so they remain -1.
    frame_output << frame.id << ',' << track.id << ',' << track.bounds.x << ','
                 << track.bounds.y << ',' << track.bounds.width << ','
                 << track.bounds.height << ",-1,-1,-1,-1\n";
  }

  stream_ << frame_output.str();
  stream_.flush();
  if (!stream_) {
    throw std::runtime_error("failed to write MOT track output: " +
                             output_path_.string());
  }
}

}  // namespace uav
