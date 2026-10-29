#pragma once

#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace uav {

enum class CliAction {
  run,
  show_help,
  show_version,
  self_check,
};

struct CliOptions {
  CliAction action{CliAction::run};
  std::optional<std::string> config_path{};
  std::optional<std::string> input_source{};
};

[[nodiscard]] CliOptions parse_cli(std::span<const std::string_view> arguments);
[[nodiscard]] std::string cli_usage();

}  // namespace uav
