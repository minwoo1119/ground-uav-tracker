#include "uav/detection/onnx_yolox_detector.hpp"

#include "uav/detection/yolox_preprocessor.hpp"

#include <array>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(UAV_HAS_ONNXRUNTIME) && UAV_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace uav {
namespace {

void validate_config(const OnnxYoloXDetectorConfig& config) {
  if (!std::filesystem::is_regular_file(config.model_path)) {
    throw std::invalid_argument("ONNX model file not found: " +
                                config.model_path.string());
  }
  if (config.intra_op_threads == 0U ||
      config.intra_op_threads >
          static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument(
        "ONNX Runtime intra-op threads must fit a positive int");
  }
}

#if defined(UAV_HAS_ONNXRUNTIME) && UAV_HAS_ONNXRUNTIME

[[nodiscard]] bool matches_shape(std::span<const std::int64_t> actual,
                                 std::span<const std::int64_t> expected) {
  if (actual.size() != expected.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < actual.size(); ++index) {
    if (actual[index] != expected[index]) {
      return false;
    }
  }
  return true;
}

#endif

}  // namespace

#if defined(UAV_HAS_ONNXRUNTIME) && UAV_HAS_ONNXRUNTIME

struct OnnxYoloXDetector::Impl {
  explicit Impl(OnnxYoloXDetectorConfig config)
      : config_(std::move(config)),
        postprocessor_(config_.postprocessor),
        environment_(ORT_LOGGING_LEVEL_WARNING, "ground-uav-tracker"),
        session_(nullptr) {
    session_options_.SetIntraOpNumThreads(
        static_cast<int>(config_.intra_op_threads));
    session_options_.SetInterOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_ = Ort::Session(environment_, config_.model_path.c_str(),
                            session_options_);

    if (session_.GetInputCount() != 1U || session_.GetOutputCount() != 1U) {
      throw std::runtime_error(
          "YOLOX ONNX model must expose exactly one input and one output");
    }
    Ort::AllocatorWithDefaultOptions allocator;
    const auto input_name = session_.GetInputNameAllocated(0U, allocator);
    const auto output_name = session_.GetOutputNameAllocated(0U, allocator);
    input_name_ = input_name.get();
    output_name_ = output_name.get();

    const auto input_info =
        session_.GetInputTypeInfo(0U).GetTensorTypeAndShapeInfo();
    if (input_info.GetElementType() !=
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::runtime_error("YOLOX ONNX input must be float32");
    }
    const std::array<std::int64_t, 4U> expected_input{
        1,
        3,
        static_cast<std::int64_t>(config_.postprocessor.input_size.height),
        static_cast<std::int64_t>(config_.postprocessor.input_size.width),
    };
    const auto input_shape = input_info.GetShape();
    if (!matches_shape(input_shape, expected_input)) {
      throw std::runtime_error(
          "YOLOX ONNX input shape does not match [1,3,height,width]");
    }

    const auto output_info =
        session_.GetOutputTypeInfo(0U).GetTensorTypeAndShapeInfo();
    if (output_info.GetElementType() !=
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::runtime_error("YOLOX ONNX output must be float32");
    }
    const std::array<std::int64_t, 3U> expected_output{
        1,
        static_cast<std::int64_t>(postprocessor_.expected_anchor_count()),
        static_cast<std::int64_t>(postprocessor_.feature_count()),
    };
    const auto output_shape = output_info.GetShape();
    if (!matches_shape(output_shape, expected_output)) {
      throw std::runtime_error(
          "YOLOX ONNX output shape does not match [1,anchors,features]");
    }
  }

  OnnxYoloXDetectorConfig config_;
  YoloXPostprocessor postprocessor_;
  Ort::Env environment_;
  Ort::SessionOptions session_options_{};
  Ort::Session session_;
  std::string input_name_{};
  std::string output_name_{};
};

#else

struct OnnxYoloXDetector::Impl {};

#endif

OnnxYoloXDetector::OnnxYoloXDetector(OnnxYoloXDetectorConfig config) {
  validate_config(config);
#if defined(UAV_HAS_ONNXRUNTIME) && UAV_HAS_ONNXRUNTIME
  try {
    impl_ = std::make_unique<Impl>(std::move(config));
  } catch (const Ort::Exception& error) {
    throw std::runtime_error(std::string{"failed to initialize ONNX Runtime: "} +
                             error.what());
  }
#else
  static_cast<void>(config);
  impl_ = std::make_unique<Impl>();
#endif
}

OnnxYoloXDetector::~OnnxYoloXDetector() = default;
OnnxYoloXDetector::OnnxYoloXDetector(OnnxYoloXDetector&&) noexcept = default;
OnnxYoloXDetector& OnnxYoloXDetector::operator=(
    OnnxYoloXDetector&&) noexcept = default;

std::vector<Detection> OnnxYoloXDetector::detect(const Frame& frame) {
#if defined(UAV_HAS_ONNXRUNTIME) && UAV_HAS_ONNXRUNTIME
  if (!impl_) {
    throw std::runtime_error("ONNX Runtime detector is not initialized");
  }
  auto input =
      preprocess_yolox_bgr(frame, impl_->config_.postprocessor.input_size);
  try {
    const std::array<std::int64_t, 4U> input_shape{
        1,
        3,
        static_cast<std::int64_t>(
            impl_->config_.postprocessor.input_size.height),
        static_cast<std::int64_t>(
            impl_->config_.postprocessor.input_size.width),
    };
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input.tensor.data(), input.tensor.size(),
        input_shape.data(), input_shape.size());
    const std::array<const char*, 1U> input_names{impl_->input_name_.c_str()};
    const std::array<const char*, 1U> output_names{impl_->output_name_.c_str()};
    auto outputs = impl_->session_.Run(
        Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1U,
        output_names.data(), 1U);
    if (outputs.size() != 1U || !outputs.front().IsTensor()) {
      throw std::runtime_error(
          "YOLOX ONNX inference did not return one tensor output");
    }
    const auto output_info = outputs.front().GetTensorTypeAndShapeInfo();
    const std::array<std::int64_t, 3U> expected_output{
        1,
        static_cast<std::int64_t>(
            impl_->postprocessor_.expected_anchor_count()),
        static_cast<std::int64_t>(impl_->postprocessor_.feature_count()),
    };
    const auto output_shape = output_info.GetShape();
    if (output_info.GetElementType() !=
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        !matches_shape(output_shape, expected_output)) {
      throw std::runtime_error(
          "YOLOX ONNX runtime output contract changed unexpectedly");
    }
    const auto* raw_output = outputs.front().GetTensorData<float>();
    return impl_->postprocessor_.process(
        std::span<const float>{raw_output, output_info.GetElementCount()},
        frame.image_size);
  } catch (const Ort::Exception& error) {
    throw std::runtime_error(std::string{"ONNX Runtime inference failed: "} +
                             error.what());
  }
#else
  static_cast<void>(frame);
  throw std::runtime_error(
      "OnnxYoloXDetector was built without ONNX Runtime support");
#endif
}

bool OnnxYoloXDetector::is_compiled_with_onnxruntime() noexcept {
#if defined(UAV_HAS_ONNXRUNTIME) && UAV_HAS_ONNXRUNTIME
  return true;
#else
  return false;
#endif
}

}  // namespace uav
