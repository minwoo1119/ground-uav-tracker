cmake_minimum_required(VERSION 3.20)

project(
  ground_uav_tracker
  VERSION 0.1.0
  DESCRIPTION "Model-independent ground-based UAV tracking scaffold"
  LANGUAGES CXX
)

option(UAV_ENABLE_TENSORRT "Enable the future TensorRT detector adapter" OFF)
option(UAV_BUILD_TESTS "Build unit tests" ON)

if(UAV_ENABLE_TENSORRT)
  message(FATAL_ERROR
    "UAV_ENABLE_TENSORRT=ON is not implemented. "
    "Select and validate JetPack/TensorRT versions before enabling it.")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)

add_library(uav_core
  src/core/cli.cpp
  src/core/pipeline.cpp
  src/core/types.cpp
)
add_library(uav::core ALIAS uav_core)
target_compile_features(uav_core PUBLIC cxx_std_20)
target_include_directories(uav_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
uav_set_warnings(uav_core)

add_executable(uav-tracker apps/uav_tracker_main.cpp)
target_link_libraries(uav-tracker PRIVATE uav::core)
target_compile_definitions(uav-tracker PRIVATE
  UAV_PROJECT_VERSION="${PROJECT_VERSION}"
)
uav_set_warnings(uav-tracker)

if(UAV_BUILD_TESTS)
  include(CTest)
  add_executable(uav_tests tests/test_main.cpp)
  target_link_libraries(uav_tests PRIVATE uav::core)
  uav_set_warnings(uav_tests)
  add_test(NAME uav_tests COMMAND uav_tests)
endif()
