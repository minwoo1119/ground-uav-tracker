# 검출·추적 결과 시각화 및 영상 저장

`OpenCvVideoVisualizer`는 파이프라인의 프레임별 검출(`Detection[]`), 추적(`Track[]`) 및 측정 지표(`FrameMetrics`)를 입력 프레임의 복사본 위에 렌더링해 동영상 파일(`.avi`, `.mp4`)로 저장합니다.

실시간 GUI 창(`cv::imshow`)이나 디스플레이 세션에 의존하지 않으므로 CI, 원격 셸, 모니터가 없는 개발/임베디드 환경에서도 동일하게 동작합니다.

## 모듈 경계와 데이터 흐름

```text
TrackingPipeline
  -> detect(frame) -> detections
  -> update(frame, detections) -> tracks
  -> FrameMetrics (detect + update 처리 시간만 측정)
  -> ITrackSink::write(frame, tracks)           [선택: MotTrackWriter]
  -> IMetricsSink::write(metrics)
  -> IFrameObserver::on_frame(frame, detections, tracks, metrics)
       └─ OpenCvVideoVisualizer (선택, UAV_HAS_OPENCV=1)
            -> BGR8 복사 -> 오버레이 렌더링 -> cv::VideoWriter
```

- `IFrameObserver`(`include/uav/telemetry/frame_observer.hpp`)는 `ITrackSink`·`IMetricsSink`와 분리된 선택적 관측 포트입니다.
- `FrameMetrics::processing_time`은 `detect + update` 구간만 측정하며, 시각화 렌더링과 비디오 인코딩 시간은 포함하지 않습니다.
- `render_overlay_bgr`와 `OpenCvVideoVisualizer`는 전달받은 `const Frame&`의 픽셀 버퍼를 수정하지 않고 복사본에 렌더링합니다.

## 오버레이 표현 규칙

순수 C++ 보조 모듈(`include/uav/visualization/overlay.hpp`)이 좌표 클램핑, 라벨 포맷팅, 상태별 색상, 코덱 선택과 출력 경로 검증을 담당하며 OpenCV 없이도 단위 테스트됩니다.

| 요소 | 표시 내용 | 기본 색상 (BGR) | 비고 |
|---|---|---|---|
| Detection | `<class_name> <confidence>` (클래스명 없으면 `class_<id>`) | `(245, 210, 40)` (청록) | 1px 박스 |
| Track (`confirmed`) | `#<id> [CONFIRMED]` | `(40, 220, 40)` (녹색) | 2px 박스 |
| Track (`tentative`) | `#<id> [TENTATIVE]` | `(0, 215, 255)` (노랑) | 2px 박스 |
| Track (`lost`) | `#<id> [LOST]` | `(30, 80, 240)` (주황/빨강) | 2px 박스 |
| Track (`removed`) | 표시하지 않음 | - | 비활성 트랙 생략 |
| HUD (좌상단) | `Frame <id> \| t=<ms>ms \| Det:<n> \| Trk:<m> \| Proc:<ms>ms` | 어두운 배경 + 흰색 글자 | 측정값만 표시 |

- 모든 경계 상자는 `clamp_bounding_box`로 이미지 경계(`[0, width] × [0, height]`)에 클램핑되며, 유효 면적이 없거나 비유한(NaN/Inf) 좌표는 건너뜁니다.
- 숫자 포맷은 시스템 로캘의 영향을 받지 않도록 `std::locale::classic()`으로 고정합니다.
- HUD의 `Proc` 값은 해당 프레임의 `detect + update` 측정 시간(ms)이며, 전체 FPS나 실시간 성능 달성을 주장하지 않습니다.

## CLI 실행

OpenCV(`core`, `imgproc`, `videoio`)와 ONNX Runtime이 활성화된 빌드에서 `--output-video <path>` 옵션을 지정합니다.

```powershell
$env:PATH="C:\path\to\opencv\build\x64\vc16\bin;$env:PATH"
build-onnx\Release\uav-tracker.exe `
  --config configs\local-runtime.yaml `
  --input data\raw\sample.mp4 `
  --output-tracks outputs\sample-tracks.txt `
  --output-video outputs\sample-overlay.avi
```

- 지원 확장자와 코덱:
  - `.avi` → `MJPG`
  - `.mp4` → `mp4v`
- 출력 영상의 해상도와 FPS는 `OpenCvFrameSource::image_size()` 및 `OpenCvFrameSource::fps()`에서 읽어 입력 영상과 맞춥니다.

## 안전 규칙과 오류 경계

- **덮어쓰기 방지**: 출력 파일이 이미 존재하면 기존 파일을 수정하지 않고 오류로 종료합니다.
- **경로 충돌 방지**: `--output-video` 경로가 `--input` 또는 `--output-tracks`와 같으면 거부합니다.
- **부모 디렉터리 미자동 생성**: 부모 디렉터리가 없으면 오류를 반환합니다.
- **픽셀 계약 검증**: metadata-only 프레임(`PixelFormat::none`)이나 `Frame::image_size` 불일치 프레임은 거부합니다. 같은 이유로 `--replay-mot`와 `--output-video` 조합은 CLI 파싱 단계에서 거부합니다.
- **OpenCV 비활성 빌드(`UAV_HAS_OPENCV=0`)**: `render_overlay_bgr`와 `OpenCvVideoVisualizer`는 `UAV_ENABLE_OPENCV=ON`이 필요하다는 오류를 명시적으로 반환합니다.

## 미구현 범위

- 실시간 GUI 창 출력(`cv::imshow`) 및 인터랙티브 키 제어
- Jetson 하드웨어 인코더(NVENC/GStreamer) 연동
- 추적 궤적(trail) 히스토리 및 커스텀 설정 기반 시각화 스타일 전환
