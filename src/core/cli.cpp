#include "uav/core/cli.hpp"

#include <stdexcept>

namespace uav {
namespace {

[[nodiscard]] std::string require_value(
    std::span<const std::string_view> arguments,
    std::size_t& index,
    std::string_view option) {
  if (index + 1U >= arguments.size() || arguments[index + 1U].starts_with("-")) {
    throw std::invalid_argument(std::string{option} + " requires a value");
  }
  ++index;
  return std::string{arguments[index]};
}

}  // namespace

CliOptions parse_cli(std::span<const std::string_view> arguments) {
  if (arguments.empty()) {
    throw std::invalid_argument(
        "no arguments provided; use --help for usage");
  }

  if (arguments.size() == 1U) {
    if (arguments.front() == "--help" || arguments.front() == "-h") {
      return {.action = CliAction::show_help};
    }
    if (arguments.front() == "--version") {
      return {.action = CliAction::show_version};
    }
    if (arguments.front() == "self-check") {
      return {.action = CliAction::self_check};
    }
  }

  CliOptions options{};
  for (std::size_t index = 0; index < arguments.size(); ++index) {
    const auto argument = arguments[index];
    if (argument == "--config") {
      if (options.config_path.has_value()) {
        throw std::invalid_argument("--config may only be specified once");
      }
      options.config_path = require_value(arguments, index, argument);
    } else if (argument == "--input") {
      if (options.input_source.has_value()) {
        throw std::invalid_argument("--input may only be specified once");
      }
      options.input_source = require_value(arguments, index, argument);
    } else if (argument == "--replay-mot") {
      if (options.replay_mot_dir.has_value()) {
        throw std::invalid_argument("--replay-mot may only be specified once");
      }
      options.replay_mot_dir = require_value(arguments, index, argument);
      options.action = CliAction::replay_mot;
    } else if (argument == "--output-tracks") {
      if (options.output_tracks_path.has_value()) {
        throw std::invalid_argument(
            "--output-tracks may only be specified once");
      }
      options.output_tracks_path = require_value(arguments, index, argument);
    } else {
      throw std::invalid_argument("unknown option: " + std::string{argument});
    }
  }

  if (options.action == CliAction::replay_mot) {
    return options;
  }

  if (options.output_tracks_path.has_value()) {
    throw std::invalid_argument(
        "--output-tracks requires --replay-mot <mot-sequence-dir>");
  }

  if (!options.config_path.has_value()) {
    throw std::invalid_argument("missing required option: --config <path>");
  }
  if (!options.input_source.has_value()) {
    throw std::invalid_argument("missing required option: --input <source>");
  }
  return options;
}

std::string cli_usage() {
  return
      "Usage:\n"
      "  uav-tracker --help\n"
      "  uav-tracker --version\n"
      "  uav-tracker self-check\n"
      "  uav-tracker --replay-mot <mot-sequence-dir> "
      "[--output-tracks <path>]\n"
      "  uav-tracker --config <path> --input <source>\n"
      "\n"
      "Options:\n"
      "  --replay-mot <dir>  Replay a MOT Challenge sequence directory "
      "(seqinfo.ini + gt/gt.txt) through KalmanIoUTracker\n"
      "  --output-tracks <path>  Write replayed tracks as MOTChallenge "
      "result rows (requires --replay-mot)\n"
      "  --config <path>     Configuration file path\n"
      "  --input <source>    Video file, device, or stream source\n";
}

}  // namespace uav
