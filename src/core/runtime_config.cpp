#include "uav/core/runtime_config.hpp"

#include "uav/detection/yolox_postprocessor.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace uav {
namespace {

struct ConfigValue {
  std::string text{};
  std::size_t line{};
};

using ConfigValues = std::map<std::string, ConfigValue>;

[[nodiscard]] std::string_view trim(std::string_view value) noexcept {
  const auto first = value.find_first_not_of(" \r\n");
  if (first == std::string_view::npos) {
    return {};
  }
  const auto last = value.find_last_not_of(" \r\n");
  return value.substr(first, last - first + 1U);
}

[[noreturn]] void config_error(const std::filesystem::path& path,
                               std::size_t line,
                               std::string_view message) {
  std::ostringstream output;
  output << "invalid runtime config " << path.string();
  if (line != 0U) {
    output << ':' << line;
  }
  output << ": " << message;
  throw std::invalid_argument(output.str());
}

[[nodiscard]] ConfigValues parse_values(
    const std::filesystem::path& config_path) {
  std::ifstream input{config_path};
  if (!input.is_open()) {
    throw std::invalid_argument("runtime config file not found: " +
                                config_path.string());
  }

  ConfigValues values;
  std::set<std::string> sections;
  std::string section;
  std::string line;
  std::size_t line_number = 0U;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.find('\t') != std::string::npos) {
      config_error(config_path, line_number,
                   "tabs are not allowed; use two spaces for keys");
    }
    const auto content = trim(line);
    if (content.empty() || content.starts_with('#')) {
      continue;
    }

    const bool indented = line.starts_with("  ");
    if (!indented) {
      if (line.starts_with(' ') || !content.ends_with(':') ||
          content.find(':') != content.size() - 1U) {
        config_error(config_path, line_number,
                     "expected an unindented 'section:' line");
      }
      section = std::string{trim(content.substr(0U, content.size() - 1U))};
      if (section.empty()) {
        config_error(config_path, line_number, "section name is empty");
      }
      static constexpr std::array known_sections{
          std::string_view{"runtime"},
          std::string_view{"detection"},
          std::string_view{"tracking"},
      };
      if (std::find(known_sections.begin(), known_sections.end(), section) ==
          known_sections.end()) {
        config_error(config_path, line_number,
                     "unknown section '" + section + "'");
      }
      if (!sections.insert(section).second) {
        config_error(config_path, line_number,
                     "duplicate section '" + section + "'");
      }
      continue;
    }

    if (line.size() > 2U && line[2] == ' ') {
      config_error(config_path, line_number,
                   "only two-level keys indented by two spaces are supported");
    }
    if (section.empty()) {
      config_error(config_path, line_number, "key appears before a section");
    }
    const auto key_value = std::string_view{line}.substr(2U);
    const auto separator = key_value.find(':');
    if (separator == std::string_view::npos) {
      config_error(config_path, line_number, "expected 'key: value'");
    }
    const auto key = trim(key_value.substr(0U, separator));
    const auto value = trim(key_value.substr(separator + 1U));
    if (key.empty() || value.empty()) {
      config_error(config_path, line_number, "key and value must not be empty");
    }

    const std::string full_key = section + '.' + std::string{key};
    if (!values.emplace(full_key,
                        ConfigValue{.text = std::string{value},
                                    .line = line_number})
             .second) {
      config_error(config_path, line_number,
                   "duplicate key '" + full_key + "'");
    }
  }

  if (!input.eof()) {
    throw std::runtime_error("failed to read runtime config: " +
                             config_path.string());
  }
  return values;
}

[[nodiscard]] const ConfigValue& require_value(
    const ConfigValues& values,
    const std::filesystem::path& path,
    std::string_view key) {
  const auto entry = values.find(std::string{key});
  if (entry == values.end()) {
    config_error(path, 0U, "missing required key '" + std::string{key} + "'");
  }
  return entry->second;
}

[[nodiscard]] std::string parse_string(const ConfigValue& value,
                                       const std::filesystem::path& path,
                                       std::string_view key) {
  auto text = std::string_view{value.text};
  if ((text.starts_with('"') && text.ends_with('"')) ||
      (text.starts_with('\'') && text.ends_with('\''))) {
    if (text.size() < 2U) {
      config_error(path, value.line,
                   "empty quoted value for '" + std::string{key} + "'");
    }
    text = text.substr(1U, text.size() - 2U);
  } else if (text.starts_with('"') || text.ends_with('"') ||
             text.starts_with('\'') || text.ends_with('\'')) {
    config_error(path, value.line,
                 "mismatched quotes for '" + std::string{key} + "'");
  }
  if (text.empty()) {
    config_error(path, value.line,
                 "empty value for '" + std::string{key} + "'");
  }
  return std::string{text};
}

template <typename Unsigned>
[[nodiscard]] Unsigned parse_unsigned(const ConfigValue& value,
                                      const std::filesystem::path& path,
                                      std::string_view key) {
  static_assert(std::is_unsigned_v<Unsigned>);
  std::uint64_t parsed{};
  const auto* begin = value.text.data();
  const auto* end = begin + value.text.size();
  const auto result = std::from_chars(begin, end, parsed);
  if (result.ec != std::errc{} || result.ptr != end ||
      parsed > static_cast<std::uint64_t>(
                   std::numeric_limits<Unsigned>::max())) {
    config_error(path, value.line,
                 "'" + std::string{key} + "' must be an unsigned integer");
  }
  return static_cast<Unsigned>(parsed);
}

[[nodiscard]] float parse_float(const ConfigValue& value,
                                const std::filesystem::path& path,
                                std::string_view key) {
  float parsed{};
  const auto* begin = value.text.data();
  const auto* end = begin + value.text.size();
  const auto result = std::from_chars(begin, end, parsed);
  if (result.ec != std::errc{} || result.ptr != end ||
      !std::isfinite(parsed)) {
    config_error(path, value.line,
                 "'" + std::string{key} + "' must be a finite number");
  }
  return parsed;
}

[[nodiscard]] bool parse_bool(const ConfigValue& value,
                              const std::filesystem::path& path,
                              std::string_view key) {
  if (value.text == "true") {
    return true;
  }
  if (value.text == "false") {
    return false;
  }
  config_error(path, value.line,
               "'" + std::string{key} + "' must be true or false");
}

[[nodiscard]] std::vector<std::string> parse_list(
    const ConfigValue& value,
    const std::filesystem::path& path,
    std::string_view key) {
  const auto text = std::string_view{value.text};
  if (text.size() < 2U || !text.starts_with('[') || !text.ends_with(']')) {
    config_error(path, value.line,
                 "'" + std::string{key} + "' must use [value, value] syntax");
  }

  std::vector<std::string> result;
  const auto body = text.substr(1U, text.size() - 2U);
  std::size_t start = 0U;
  while (start <= body.size()) {
    const auto separator = body.find(',', start);
    const auto item = trim(body.substr(
        start, separator == std::string_view::npos
                   ? std::string_view::npos
                   : separator - start));
    if (item.empty()) {
      config_error(path, value.line,
                   "'" + std::string{key} + "' contains an empty item");
    }
    result.emplace_back(item);
    if (separator == std::string_view::npos) {
      break;
    }
    start = separator + 1U;
  }
  return result;
}

[[nodiscard]] std::vector<std::uint32_t> parse_strides(
    const ConfigValue& value,
    const std::filesystem::path& path) {
  const auto items = parse_list(value, path, "detection.strides");
  std::vector<std::uint32_t> strides;
  strides.reserve(items.size());
  for (const auto& item : items) {
    strides.push_back(parse_unsigned<std::uint32_t>(
        ConfigValue{.text = item, .line = value.line}, path,
        "detection.strides"));
  }
  return strides;
}

void validate_known_keys(const ConfigValues& values,
                         const std::filesystem::path& path) {
  static const std::vector<std::string_view> known{
      "runtime.max_frames",
      "detection.model_path",
      "detection.input_width",
      "detection.input_height",
      "detection.strides",
      "detection.class_names",
      "detection.confidence_threshold",
      "detection.nms_threshold",
      "detection.class_agnostic_nms",
      "detection.intra_op_threads",
      "tracking.iou_threshold",
      "tracking.min_hits_to_confirm",
      "tracking.max_missed_frames",
      "tracking.default_dt_seconds",
      "tracking.position_process_noise",
      "tracking.size_process_noise",
      "tracking.velocity_process_noise",
      "tracking.position_measurement_noise",
      "tracking.size_measurement_noise",
  };
  for (const auto& [key, value] : values) {
    if (std::find(known.begin(), known.end(), key) == known.end()) {
      config_error(path, value.line, "unknown key '" + key + "'");
    }
  }
}

void validate_tracker_config(const TrackerConfig& config,
                             const std::filesystem::path& path) {
  if (!std::isfinite(config.iou_threshold) || config.iou_threshold < 0.0F ||
      config.iou_threshold > 1.0F) {
    config_error(path, 0U, "tracking.iou_threshold must be in [0, 1]");
  }
  if (config.min_hits_to_confirm == 0U) {
    config_error(path, 0U, "tracking.min_hits_to_confirm must be >= 1");
  }
  if (!std::isfinite(config.default_dt_seconds) ||
      config.default_dt_seconds <= 0.0F) {
    config_error(path, 0U, "tracking.default_dt_seconds must be > 0");
  }
  const std::vector<std::pair<std::string_view, float>> noises{
      {"tracking.position_process_noise", config.position_process_noise},
      {"tracking.size_process_noise", config.size_process_noise},
      {"tracking.velocity_process_noise", config.velocity_process_noise},
      {"tracking.position_measurement_noise",
       config.position_measurement_noise},
      {"tracking.size_measurement_noise", config.size_measurement_noise},
  };
  for (const auto& [name, noise] : noises) {
    if (!std::isfinite(noise) || noise < 0.0F) {
      config_error(path, 0U, std::string{name} + " must be >= 0");
    }
  }
}

}  // namespace

RuntimeConfig load_runtime_config(const std::filesystem::path& config_path) {
  if (!std::filesystem::is_regular_file(config_path)) {
    throw std::invalid_argument("runtime config file not found: " +
                                config_path.string());
  }
  const auto absolute_path = std::filesystem::absolute(config_path);
  const auto values = parse_values(absolute_path);
  validate_known_keys(values, absolute_path);

  RuntimeConfig config{};
  config.max_frames = parse_unsigned<std::size_t>(
      require_value(values, absolute_path, "runtime.max_frames"),
      absolute_path, "runtime.max_frames");

  auto model_path = std::filesystem::path{parse_string(
      require_value(values, absolute_path, "detection.model_path"),
      absolute_path, "detection.model_path")};
  if (model_path.is_relative()) {
    model_path = absolute_path.parent_path() / model_path;
  }
  config.detector.model_path = model_path.lexically_normal();
  config.detector.postprocessor.input_size = {
      .width = parse_unsigned<std::uint32_t>(
          require_value(values, absolute_path, "detection.input_width"),
          absolute_path, "detection.input_width"),
      .height = parse_unsigned<std::uint32_t>(
          require_value(values, absolute_path, "detection.input_height"),
          absolute_path, "detection.input_height"),
  };
  config.detector.postprocessor.strides = parse_strides(
      require_value(values, absolute_path, "detection.strides"),
      absolute_path);
  config.detector.postprocessor.class_names = parse_list(
      require_value(values, absolute_path, "detection.class_names"),
      absolute_path, "detection.class_names");
  config.detector.postprocessor.confidence_threshold = parse_float(
      require_value(values, absolute_path, "detection.confidence_threshold"),
      absolute_path, "detection.confidence_threshold");
  config.detector.postprocessor.nms_threshold = parse_float(
      require_value(values, absolute_path, "detection.nms_threshold"),
      absolute_path, "detection.nms_threshold");
  config.detector.postprocessor.class_agnostic_nms = parse_bool(
      require_value(values, absolute_path, "detection.class_agnostic_nms"),
      absolute_path, "detection.class_agnostic_nms");
  config.detector.intra_op_threads = parse_unsigned<std::size_t>(
      require_value(values, absolute_path, "detection.intra_op_threads"),
      absolute_path, "detection.intra_op_threads");
  if (config.detector.intra_op_threads == 0U) {
    config_error(absolute_path, 0U, "detection.intra_op_threads must be >= 1");
  }

  config.tracker = TrackerConfig{
      .iou_threshold = parse_float(
          require_value(values, absolute_path, "tracking.iou_threshold"),
          absolute_path, "tracking.iou_threshold"),
      .min_hits_to_confirm = parse_unsigned<std::uint32_t>(
          require_value(values, absolute_path,
                        "tracking.min_hits_to_confirm"),
          absolute_path, "tracking.min_hits_to_confirm"),
      .max_missed_frames = parse_unsigned<std::uint32_t>(
          require_value(values, absolute_path, "tracking.max_missed_frames"),
          absolute_path, "tracking.max_missed_frames"),
      .default_dt_seconds = parse_float(
          require_value(values, absolute_path, "tracking.default_dt_seconds"),
          absolute_path, "tracking.default_dt_seconds"),
      .position_process_noise = parse_float(
          require_value(values, absolute_path,
                        "tracking.position_process_noise"),
          absolute_path, "tracking.position_process_noise"),
      .size_process_noise = parse_float(
          require_value(values, absolute_path, "tracking.size_process_noise"),
          absolute_path, "tracking.size_process_noise"),
      .velocity_process_noise = parse_float(
          require_value(values, absolute_path,
                        "tracking.velocity_process_noise"),
          absolute_path, "tracking.velocity_process_noise"),
      .position_measurement_noise = parse_float(
          require_value(values, absolute_path,
                        "tracking.position_measurement_noise"),
          absolute_path, "tracking.position_measurement_noise"),
      .size_measurement_noise = parse_float(
          require_value(values, absolute_path,
                        "tracking.size_measurement_noise"),
          absolute_path, "tracking.size_measurement_noise"),
  };

  static_cast<void>(YoloXPostprocessor{config.detector.postprocessor});
  validate_tracker_config(config.tracker, absolute_path);
  return config;
}

}  // namespace uav
