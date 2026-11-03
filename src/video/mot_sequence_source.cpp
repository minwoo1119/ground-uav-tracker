#include "uav/video/mot_sequence_source.hpp"

#include <charconv>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace uav {
namespace {

std::string_view trim_whitespace(std::string_view sv) noexcept {
  while (!sv.empty() &&
         (sv.front() == ' ' || sv.front() == '\t' || sv.front() == '\r')) {
    sv.remove_prefix(1U);
  }
  while (!sv.empty() &&
         (sv.back() == ' ' || sv.back() == '\t' || sv.back() == '\r')) {
    sv.remove_suffix(1U);
  }
  return sv;
}

template <typename IntType>
IntType parse_unsigned(std::string_view text, std::string_view key_name) {
  IntType value{};
  const auto [ptr, ec] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (ec != std::errc{} || ptr != text.data() + text.size()) {
    throw std::invalid_argument("invalid integer in seqinfo.ini for key: " +
                                std::string{key_name});
  }
  return value;
}

}  // namespace

MotSequenceFrameSource::MotSequenceFrameSource(
    const std::filesystem::path& sequence_dir) {
  const auto ini_path = sequence_dir / "seqinfo.ini";
  std::ifstream input{ini_path};
  if (!input.is_open()) {
    throw std::invalid_argument("seqinfo.ini not found in MOT directory: " +
                                ini_path.string());
  }

  std::string line;
  while (std::getline(input, line)) {
    const auto trimmed = trim_whitespace(line);
    if (trimmed.empty() || trimmed.front() == '[' || trimmed.front() == '#' ||
        trimmed.front() == ';') {
      continue;
    }
    const auto eq_pos = trimmed.find('=');
    if (eq_pos == std::string_view::npos) {
      continue;
    }
    const auto key = trim_whitespace(trimmed.substr(0U, eq_pos));
    const auto val = trim_whitespace(trimmed.substr(eq_pos + 1U));

    if (key == "name") {
      info_.name = std::string{val};
    } else if (key == "frameRate") {
      info_.frame_rate = parse_unsigned<std::uint32_t>(val, key);
    } else if (key == "seqLength") {
      info_.seq_length = parse_unsigned<std::uint64_t>(val, key);
    } else if (key == "imWidth") {
      info_.image_size.width = parse_unsigned<std::uint32_t>(val, key);
    } else if (key == "imHeight") {
      info_.image_size.height = parse_unsigned<std::uint32_t>(val, key);
    }
  }

  if (info_.seq_length == 0U || !info_.image_size.is_valid() ||
      info_.frame_rate == 0U) {
    throw std::invalid_argument(
        "seqinfo.ini is missing required valid fields (seqLength, frameRate, "
        "imWidth, imHeight)");
  }
}

std::optional<Frame> MotSequenceFrameSource::next() {
  if (current_frame_ > info_.seq_length) {
    return std::nullopt;
  }
  const FrameId id = current_frame_++;
  const std::int64_t timestamp_us = static_cast<std::int64_t>(
      ((id - 1U) * 1'000'000ULL) / info_.frame_rate);
  return Frame{
      .id = id,
      .captured_at = Timestamp{timestamp_us},
      .image_size = info_.image_size,
  };
}

}  // namespace uav
