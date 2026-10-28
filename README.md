# Ground UAV Tracker

지상 카메라 영상에서 UAV를 탐지하고 추적하기 위한 C++20 프로젝트의 초기 골격입니다. 현재 단계에는 모델, 데이터셋, 학습 코드 및 TensorRT 엔진이 포함되지 않습니다.

## 요구 사항

- CMake 3.20 이상
- C++20을 지원하는 컴파일러(MSVC, GCC 또는 Clang)

외부 런타임/테스트 라이브러리는 사용하지 않습니다.

## 빌드와 테스트

```sh
cmake -S . -B build -DGUT_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

모델 없이 최소 실행 경로를 확인합니다.

```sh
./build/ground-uav-tracker self-check
```

Visual Studio 다중 구성 생성기에서는 실행 파일이 `build/Release/` 아래에 생성될 수 있습니다.

## 현재 제공 범위

- 프레임, 탐지, 바운딩 박스, 트랙 및 결과 데이터 구조
- 프레임 소스, 탐지기, 추적기 및 결과 싱크 인터페이스
- 각 모듈을 연결하는 동기식 추적 파이프라인
- 모델 없이 동작하는 `self-check` CLI
- 외부 의존성이 없는 단위/계약 테스트

상세한 경계와 데이터 흐름은 [아키텍처 문서](docs/architecture.md), 구현 전에 결정할 항목은 [미결정 사항](docs/open-decisions.md)을 참고하세요.
