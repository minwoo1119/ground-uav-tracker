# ONNX Runtime CPU reference detector

`OnnxYoloXDetector`는 Windows 개발 PC에서 YOLOX ONNX graph와 C++ 전처리·후처리 계약을 검증하기 위한 CPU reference adapter입니다. Jetson 배포 backend가 아니며 FPS, latency 또는 검출 정확도 기준선으로 사용하지 않습니다.

## 고정 의존성과 출처

- ONNX Runtime: `1.23.2`, CPU Windows x64 official release archive
- asset: `onnxruntime-win-x64-1.23.2.zip`
- SHA-256: `0b38df9af21834e41e73d602d90db5cb06dbd1ca618948b8f1d66d607ac9f3cd`
- 공식 C++ 안내: <https://onnxruntime.ai/docs/get-started/with-cpp.html>
- 공식 release: <https://github.com/microsoft/onnxruntime/releases/tag/v1.23.2>
- 실제 영상 통합 테스트(선택): [OpenCV `4.11.0` 공식 release](https://github.com/opencv/opencv/releases/tag/4.11.0)의 Windows package `opencv-4.11.0-windows.exe`; 이번 검증에서 계산한 SHA-256 `7c9d1c0b70db1b1952cc815252fced9a07f51267563cf3eaa1674d734c49b8e4`

archive, extracted library, ONNX model과 build output은 `artifacts/`, `models/`, `build-*`의 Git 제외 경로에만 둡니다. CMake가 네트워크에서 dependency를 자동 다운로드하지 않으므로 source build와 CI가 외부 상태에 따라 달라지지 않습니다.

## Frame CPU pixel 계약

`Frame`은 두 형태를 지원합니다.

- metadata-only: `PixelFormat::none`, stride 0, 빈 buffer. MOT replay에서 사용합니다.
- CPU image: `PixelFormat::bgr8`, `row_stride_bytes >= width × 3`, 정확히 `stride × height` bytes를 `std::vector<uint8_t>`로 소유합니다.

buffer는 `Frame` 수명 동안 유효하며 detector는 읽기만 합니다. `OpenCvFrameSource`는 `cv::VideoCapture`가 반환한 `CV_8UC3` 행을 tightly packed owned buffer로 복사합니다. GPU/zero-copy memory는 이 타입에 억지로 포함하지 않고 후속 별도 계약으로 남깁니다.

## 전처리와 inference 흐름

```text
Frame(BGR8 CPU bytes)
  -> aspect-ratio bilinear resize
  -> 640×640 canvas top-left placement, remaining pixels = 114
  -> unnormalized BGR CHW float `[1,3,640,640]`
  -> ONNX Runtime CPU session
  -> raw float `[1,8400,9]`
  -> YoloXPostprocessor
  -> Detection[]
```

전처리는 pinned YOLOX validation transform과 같이 RGB swap이나 0~1 normalization을 하지 않습니다. tensor와 model의 input/output 개수, float32 type, static shape가 다르면 session 생성 또는 inference에서 오류를 반환합니다.

## 빌드

기본 빌드는 ONNX Runtime 없이 유지됩니다.

```powershell
cmake -S . -B build-cpu `
  -DUAV_ENABLE_TENSORRT=OFF `
  -DUAV_ENABLE_ONNXRUNTIME=OFF `
  -DUAV_BUILD_TESTS=ON
cmake --build build-cpu --config Release
ctest --test-dir build-cpu -C Release --output-on-failure
```

공식 package를 사용한 reference build는 extracted root와 선택적 test model을 명시합니다.

```powershell
cmake -S . -B build-onnx `
  -DUAV_ENABLE_TENSORRT=OFF `
  -DUAV_ENABLE_OPENCV=ON `
  -DUAV_ENABLE_ONNXRUNTIME=ON `
  -DONNXRUNTIME_ROOT=C:\path\to\onnxruntime-win-x64-1.23.2 `
  -DOpenCV_DIR=C:\path\to\opencv\build\x64\vc16\lib `
  -DUAV_TEST_ONNX_MODEL=C:\path\to\yolox_s_uav.onnx `
  -DUAV_TEST_VIDEO=C:\path\to\sample.mp4 `
  -DUAV_BUILD_TESTS=ON
cmake --build build-onnx --config Release
$env:PATH="C:\path\to\opencv\build\x64\vc16\bin;$env:PATH"
ctest --test-dir build-onnx -C Release --output-on-failure
```

Windows build는 `onnxruntime.dll`을 각 executable 디렉터리에 복사합니다. `UAV_TEST_ONNX_MODEL`이 없으면 integration test executable은 빌드하지만 CTest에는 등록하지 않습니다. `UAV_TEST_VIDEO`를 함께 주면 OpenCV가 소유 BGR8 frame을 만들고 같은 detector가 실제 영상 첫 frame을 처리하는 경로도 검사합니다.

## 오류와 미구현 범위

- model 파일 부재, thread 수 오류, input/output count·type·shape 불일치는 명시적으로 거부합니다.
- pixel 없는 metadata frame이나 잘못된 BGR stride/size는 전처리에서 거부합니다.
- 기본 OFF build에서 `detect()`를 호출하면 ONNX Runtime 미포함 오류를 반환합니다.
- 현재 CLI/YAML은 이 adapter를 생성하지 않습니다. 직접 library API로 구성하거나 후속 CLI wiring이 필요합니다.
- local smoke ONNX는 구조 검증용 미학습 model입니다. detection 수가 0이어도 정확도 결과가 아닙니다.
- CUDA/DirectML/TensorRT provider, batching, asynchronous inference, GPU memory와 Jetson 검증은 포함하지 않습니다.
