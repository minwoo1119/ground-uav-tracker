#include "uav/detection/mot_replay_detector.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace uav {
namespace {

std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> tokens;
  std::stringstream ss{line};
  std::string item;
  while (std::getline(ss, item, ',')) {
    tokens.push_back(item);
  }
  return tokens;
}

}  // namespace

MotReplayDetector::MotReplayDetector(const std::filesystem::path& sequence_dir,
                                     std::int32_t class_id,
                                     std::string class_name) {
  std::filesystem::path gt_path = sequence_dir;
  if (std::filesystem::is_directory(sequence_dir)) {
    gt_path = sequence_dir / "gt" / "gt.txt";
  }
  std::ifstream input{gt_path};
  if (!input.is_open()) {
    throw std::invalid_argument("MOT gt.txt not found: " + gt_path.string());
  }

  std::string line;
  std::size_t line_number = 0U;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty() || line.front() == '#') {
      continue;
    }
    const auto fields = split_csv_line(line);
    if (fields.size() < 6U) {
      throw std::invalid_argument("malformed MOT gt.txt at line " +
                                  std::to_string(line_number));
    }

    const FrameId frame_id = static_cast<FrameId>(std::stoull(fields[0]));
    const float x = std::stof(fields[2]);
    const float y = std::stof(fields[3]);
    const float w = std::stof(fields[4]);
    const float h = std::stof(fields[5]);
    const float conf = fields.size() >= 7U ? std::stof(fields[6]) : 1.0F;

    if (w <= 0.0F || h <= 0.0F) {
      continue;
    }

    by_frame_[frame_id].push_back(Detection{
        .class_id = class_id,
        .confidence = conf,
        .bounds = BoundingBox{.x = x, .y = y, .width = w, .height = h},
        .class_name = class_name,
    });
    ++total_annotations_;
  }
}

std::vector<Detection> MotReplayDetector::detect(const Frame& frame) {
  const auto it = by_frame_.find(frame.id);
  if (it == by_frame_.end()) {
    return {};
  }
  return it->second;
}

}  // namespace uav
