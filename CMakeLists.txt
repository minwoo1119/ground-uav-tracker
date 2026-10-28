cmake_minimum_required(VERSION 3.20)

project(
  ground_uav_tracker
  VERSION 0.1.0
  DESCRIPTION "Model-independent ground UAV tracking pipeline"
  LANGUAGES CXX
)

option(GUT_BUILD_TESTS "Build the ground UAV tracker tests" ON)

add_library(gut_core
  src/core/geometry.cpp
  src/core/pipeline.cpp
)
add_library(gut::core ALIAS gut_core)
target_compile_features(gut_core PUBLIC cxx_std_20)
target_include_directories(gut_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if(MSVC)
  target_compile_options(gut_core PRIVATE /W4 /permissive- /utf-8)
else()
  target_compile_options(gut_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(ground-uav-tracker src/app/main.cpp)
target_link_libraries(ground-uav-tracker PRIVATE gut::core)
if(MSVC)
  target_compile_options(ground-uav-tracker PRIVATE /W4 /permissive- /utf-8)
else()
  target_compile_options(ground-uav-tracker PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(GUT_BUILD_TESTS)
  include(CTest)
  add_executable(gut_tests tests/test_main.cpp)
  target_link_libraries(gut_tests PRIVATE gut::core)
  if(MSVC)
    target_compile_options(gut_tests PRIVATE /W4 /permissive- /utf-8)
  else()
    target_compile_options(gut_tests PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  add_test(NAME gut_tests COMMAND gut_tests)
endif()
