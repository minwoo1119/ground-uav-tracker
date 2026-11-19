cmake_minimum_required(VERSION 3.20)

project(
  ground_uav_tracker
  VERSION 0.1.0
  DESCRIPTION "Model-independent ground-based UAV tracking scaffold"
  LANGUAGES CXX
)

option(UAV_ENABLE_TENSORRT "Enable the future TensorRT detector adapter" OFF)
option(UAV_ENABLE_OPENCV "Enable OpenCV video input adapter when available" ON)
option(UAV_ENABLE_ONNXRUNTIME "Enable the ONNX Runtime CPU detector adapter" OFF)
option(UAV_BUILD_TESTS "Build unit tests" ON)
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to an extracted ONNX Runtime package")
set(UAV_TEST_ONNX_MODEL "" CACHE FILEPATH "Optional ONNX model for integration tests")
set(UAV_TEST_VIDEO "" CACHE FILEPATH "Optional video for OpenCV/ONNX integration tests")

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
  src/core/runtime_config.cpp
  src/core/types.cpp
  src/detection/mot_replay_detector.cpp
  src/detection/onnx_yolox_detector.cpp
  src/detection/yolox_preprocessor.cpp
  src/detection/yolox_postprocessor.cpp
  src/telemetry/mot_track_writer.cpp
  src/tracking/kalman_iou_tracker.cpp
  src/video/mot_sequence_source.cpp
  src/video/opencv_frame_source.cpp
)
add_library(uav::core ALIAS uav_core)
target_compile_features(uav_core PUBLIC cxx_std_20)
target_include_directories(uav_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
uav_set_warnings(uav_core)

if(UAV_ENABLE_ONNXRUNTIME)
  if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR
      "UAV_ENABLE_ONNXRUNTIME=ON requires ONNXRUNTIME_ROOT to point to an "
      "extracted official ONNX Runtime package")
  endif()
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS "${ONNXRUNTIME_ROOT}/include"
    NO_DEFAULT_PATH
  )
  find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
  )
  if(WIN32)
    find_file(ONNXRUNTIME_DLL NAMES onnxruntime.dll
      PATHS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/bin"
      NO_DEFAULT_PATH
    )
  endif()
  if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY OR
     (WIN32 AND NOT ONNXRUNTIME_DLL))
    message(FATAL_ERROR
      "ONNXRUNTIME_ROOT is missing include/onnxruntime_cxx_api.h, the runtime "
      "library, or the Windows DLL")
  endif()

  add_library(uav_onnxruntime SHARED IMPORTED)
  if(WIN32)
    set_target_properties(uav_onnxruntime PROPERTIES
      IMPORTED_IMPLIB "${ONNXRUNTIME_LIBRARY}"
      IMPORTED_LOCATION "${ONNXRUNTIME_DLL}"
      INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
    )
  else()
    set_target_properties(uav_onnxruntime PROPERTIES
      IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
    )
  endif()
  target_compile_definitions(uav_core PUBLIC UAV_HAS_ONNXRUNTIME=1)
  target_link_libraries(uav_core PUBLIC uav_onnxruntime)
else()
  target_compile_definitions(uav_core PUBLIC UAV_HAS_ONNXRUNTIME=0)
endif()

if(UAV_ENABLE_OPENCV)
  find_package(OpenCV QUIET COMPONENTS core videoio)
  if(OpenCV_FOUND)
    message(STATUS "OpenCV found (${OpenCV_VERSION}); enabling OpenCvFrameSource")
    target_compile_definitions(uav_core PUBLIC UAV_HAS_OPENCV=1)
    target_link_libraries(uav_core PUBLIC ${OpenCV_LIBS})
    target_include_directories(uav_core PUBLIC ${OpenCV_INCLUDE_DIRS})
  else()
    message(STATUS "OpenCV not found; building with UAV_HAS_OPENCV=0 (MOT replay and KalmanIoUTracker remain enabled)")
    target_compile_definitions(uav_core PUBLIC UAV_HAS_OPENCV=0)
  endif()
else()
  target_compile_definitions(uav_core PUBLIC UAV_HAS_OPENCV=0)
endif()

add_executable(uav-tracker apps/uav_tracker_main.cpp)
target_link_libraries(uav-tracker PRIVATE uav::core)
target_compile_definitions(uav-tracker PRIVATE
  UAV_PROJECT_VERSION="${PROJECT_VERSION}"
)
uav_set_warnings(uav-tracker)

function(uav_copy_onnxruntime_dll target_name)
  if(WIN32 AND UAV_ENABLE_ONNXRUNTIME)
    add_custom_command(TARGET ${target_name} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${ONNXRUNTIME_DLL}" "$<TARGET_FILE_DIR:${target_name}>"
      VERBATIM
    )
  endif()
endfunction()
uav_copy_onnxruntime_dll(uav-tracker)

if(UAV_BUILD_TESTS)
  include(CTest)
  add_executable(uav_tests tests/test_main.cpp)
  target_link_libraries(uav_tests PRIVATE uav::core)
  uav_set_warnings(uav_tests)
  add_test(NAME uav_tests COMMAND uav_tests)
  uav_copy_onnxruntime_dll(uav_tests)

  if(UAV_ENABLE_ONNXRUNTIME)
    add_executable(uav_onnx_tests tests/test_onnx_detector.cpp)
    target_link_libraries(uav_onnx_tests PRIVATE uav::core)
    uav_set_warnings(uav_onnx_tests)
    uav_copy_onnxruntime_dll(uav_onnx_tests)
    if(UAV_TEST_ONNX_MODEL)
      if(NOT EXISTS "${UAV_TEST_ONNX_MODEL}")
        message(FATAL_ERROR "UAV_TEST_ONNX_MODEL does not exist: ${UAV_TEST_ONNX_MODEL}")
      endif()
      set(UAV_ONNX_TEST_ARGUMENTS "${UAV_TEST_ONNX_MODEL}")
      if(UAV_TEST_VIDEO)
        if(NOT OpenCV_FOUND)
          message(FATAL_ERROR
            "UAV_TEST_VIDEO requires an OpenCV-enabled build")
        endif()
        if(NOT EXISTS "${UAV_TEST_VIDEO}")
          message(FATAL_ERROR "UAV_TEST_VIDEO does not exist: ${UAV_TEST_VIDEO}")
        endif()
        list(APPEND UAV_ONNX_TEST_ARGUMENTS "${UAV_TEST_VIDEO}")

        file(TO_CMAKE_PATH "${UAV_TEST_ONNX_MODEL}"
          UAV_TEST_ONNX_MODEL_CONFIG)
        set(UAV_CLI_TEST_CONFIG
          "${CMAKE_CURRENT_BINARY_DIR}/onnx-smoke.yaml")
        configure_file(
          "${CMAKE_CURRENT_SOURCE_DIR}/cmake/onnx-smoke.yaml.in"
          "${UAV_CLI_TEST_CONFIG}"
          @ONLY
        )
        add_test(NAME uav_cli_video
          COMMAND uav-tracker
                  --config "${UAV_CLI_TEST_CONFIG}"
                  --input "${UAV_TEST_VIDEO}")
      endif()
      add_test(NAME uav_onnx_tests
        COMMAND uav_onnx_tests ${UAV_ONNX_TEST_ARGUMENTS})
    else()
      message(STATUS
        "UAV_TEST_ONNX_MODEL is empty; building uav_onnx_tests without registering it in CTest")
    endif()
  endif()
endif()
