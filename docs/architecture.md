# 아키텍처

## 현재 확정된 경계 및 구현체

입력, 탐지, 추적 및 관측 기록은 교체 가능한 포트(`IFrameSource`, `IDetector`, `ITracker`, `ITrackSink`, `IMetricsSink`)로 분리되어 있으며, 현재 구현된 구체 어댑터는 다음과 같습니다.

```text
IFrameSource ---------------------------------> Frame -> IDetector -----------------------> Detection[]
  ├─ MotSequenceFrameSource (metadata)                   ├─ MotReplayDetector (gt/gt.txt)       |
  └─ OpenCvFrameSource (owned BGR8 pixels, 선택)         └─ OnnxYoloXDetector (선택)             |
                                                               └─ preprocessor -> ORT -> postprocessor
                                                                                           v
                                                                                    ITracker -> Track[]
                                                                                      └─ KalmanIoUTracker
                                                                                           |
                                                                                 +---------+---------+
                                                                                 v                   v
                                                                            ITrackSink         IMetricsSink
                                                                              └─ MotTrackWriter
```

`TrackingPipeline`은 프레임 순서 검증(단조 증가 timestamp 보장), `detect` → `update` → 선택적 track sink → metrics sink 호출 순서 및 처리 개수/시간 집계를 담당합니다. `ITrackSink`를 전달하지 않으면 기존과 같이 추적 결과 파일을 생성하지 않습니다.

선택적 ONNX Runtime CPU reference adapter와 export된 YOLOX raw tensor 이후 경계가 구현되어 있습니다.

```text
OnnxYoloXDetector -> ONNX Runtime CPU -> raw float[8400][9]
  -> YoloXPostprocessor -> Detection[] -> ITracker
```

## 데이터 계약

- `Frame`: `FrameId`, 단조 증가 `Timestamp(microseconds)`, `ImageSize`와 선택적 CPU-owned BGR8 pixel buffer/row stride를 포함합니다. MOT source는 buffer가 없는 metadata-only frame을 사용합니다.
- `Detection`: `class_id`, `confidence`, `class_name` 및 픽셀 좌표 `BoundingBox(x, y, width, height)`입니다.
- `Track`: `TrackId`, `BoundingBox`, `age_frames`, `missed_frames`와 `tentative`, `confirmed`, `lost`, `removed` 상태를 가집니다.
- `ITrackSink`: 처리 중인 `Frame`과 그 프레임에서 반환된 `Track[]`을 함께 받아 frame ID와 timestamp의 연관을 유지합니다.
- `FrameMetrics`: `frame_id`, `captured_at`, `detection_count`, `track_count` 및 프레임 처리 시간(`processing_time`)을 전달합니다.

## 핵심 모듈 구현

### 1. `KalmanIoUTracker` (`include/uav/tracking/kalman_iou_tracker.hpp`)
- **상태 벡터 (`KalmanBoxFilter`)**: $[c_x, c_y, w, h, v_x, v_y]^T \in \mathbb{R}^6$ (2D 등속도 모델)
- **예측 단계 (`predict`)**: 프레임 간 타임스탬프 차분 $\Delta t$초만큼 중심 좌표를 등속 이동시키고 위치·속도·크기 공분산을 갱신합니다.
- **연관 및 갱신 (`update`)**: 예측된 트랙 박스와 현재 프레임 검출 박스 간 IoU를 계산하여 내림차순 탐욕적 매칭(`iou >= iou_threshold`)을 수행하고, 매칭된 트랙은 칼만 이득(Kalman Gain)으로 상태와 속도를 보정합니다.
- **수명주기 전이**:
  - 신규 검출 → `tentative` (`hits = 1`)
  - 누적 히트 `hits >= min_hits_to_confirm` → `confirmed`
  - `confirmed` 트랙 미검출(`1 <= missed_frames <= max_missed_frames`) → `lost` (등속 예측 유지)
  - `missed_frames > max_missed_frames` → `removed` (활성 목록에서 제거)

### 2. MOT 리플레이 및 비디오 입력 어댑터
- `MotSequenceFrameSource`: MOT Challenge 형식의 `seqinfo.ini`를 읽어 프레임 해상도, FPS, 시퀀스 길이를 파싱하고 단조 증가 타임스탬프를 가진 `Frame`을 공급합니다.
- `MotReplayDetector`: MOT Challenge 형식의 `gt/gt.txt`를 읽어 해당 `frame.id`의 바운딩 박스를 `Detection` 목록으로 반환합니다.
- `OpenCvFrameSource`: CMake `UAV_ENABLE_OPENCV=ON` 및 시스템에 OpenCV(`core`, `videoio`)가 설치된 경우 `cv::VideoCapture`로 영상을 읽고 `CV_8UC3` BGR pixel을 `Frame` 소유 buffer로 복사합니다. 미설치 환경에서도 `UAV_HAS_OPENCV=0` 폴백을 통해 코어 라이브러리와 MOT 리플레이가 외부 의존성 없이 동작합니다.

### 3. MOT 추적 결과 출력

- `MotTrackWriter`: 사용자가 지정한 파일을 새로 열어 각 활성 트랙을 MOTChallenge 결과 행 `frame,id,x,y,width,height,confidence,x3d,y3d,z3d`로 기록합니다.
- `tentative`, `confirmed`, `lost` 트랙은 현재 프레임의 출력 박스로 기록하고 `removed` 트랙은 제외합니다.
- 현재 `Track`에는 confidence와 3D 좌표가 없으므로 해당 네 열은 `-1`로 기록합니다. 이는 실제 confidence를 생성하거나 추정하지 않았음을 뜻합니다.
- 기존 결과 덮어쓰기를 방지하기 위해 출력 파일이 이미 있으면 거부합니다. 부모 디렉터리도 자동 생성하지 않으며, 열기 또는 쓰기 실패는 예외로 호출자에게 전달합니다.
- 원본 `seqinfo.ini` 또는 `gt/gt.txt`와 동일한 출력 경로는 입력 데이터 덮어쓰기를 막기 위해 거부합니다.

### 4. YOLOX raw 출력 후처리

- `YoloXPostprocessor`: 640 입력의 stride 8/16/32 raw tensor `[8400, 9]`를 grid decode하고, `objectness × class probability`, 원본 letterbox 좌표 복원, clipping 및 NMS를 거쳐 `Detection[]`을 반환합니다.
- 공식 demo와 같은 class-agnostic NMS를 기본 설정으로 사용할 수 있고, class-aware NMS도 호출자 설정으로 제공합니다. confidence와 NMS threshold는 호출자가 명시해야 하며 현재 저장소는 최적값을 주장하지 않습니다.
- tensor shape, class probability 범위, NaN/Inf 및 잘못된 설정은 명시적 예외로 반환합니다. 자세한 계약은 [YOLOX C++ 후처리 문서](yolox-cpp-postprocessing.md)에 기록합니다.

### 5. ONNX Runtime CPU reference detector

- `preprocess_yolox_bgr`: stride-aware BGR8 bytes를 bilinear resize하고 114 padding의 640×640 canvas에 top-left 배치한 뒤 unnormalized BGR CHW float tensor를 만듭니다.
- `OnnxYoloXDetector`: CMake `UAV_ENABLE_ONNXRUNTIME=ON`일 때 CPU session을 만들고 static `[1,3,640,640]` input과 `[1,8400,9]` float output을 검증한 뒤 `YoloXPostprocessor`에 전달합니다.
- `UAV_ENABLE_ONNXRUNTIME=OFF`가 기본값이므로 외부 runtime 없이 기존 build와 MOT 테스트가 유지됩니다. package 연결과 실제 smoke test 절차는 [ONNX Runtime detector 문서](onnx-runtime-detector.md)에 기록합니다.

## 실행 계층 (`apps/uav_tracker_main.cpp`)

- `uav-tracker self-check`: 빈 어댑터로 파이프라인 무결성 자가 진단
- `uav-tracker --replay-mot <mot-sequence-dir>`: `MotSequenceFrameSource` + `MotReplayDetector` + `KalmanIoUTracker`를 연결하여 실제 변환된 시퀀스를 End-to-End 리플레이하고 처리 프레임/검출/트랙 수 및 소요 시간을 출력
- `uav-tracker --replay-mot <mot-sequence-dir> --output-tracks <path>`: 동일한 리플레이 결과를 지정한 MOTChallenge 텍스트 파일에도 기록. 옵션이 없으면 파일을 생성하지 않음
- `OnnxYoloXDetector`는 아직 CLI/config factory에 연결되지 않았으며 library API와 integration test에서만 실행됩니다.

## 아직 확정하지 않은 설계

결과 시각화, 자동 보존/정리 정책, 추론 주기, 멀티스레딩/큐, ROI/tiling, GPU/zero-copy pixel 형식, CLI detector factory와 TensorRT adapter는 후속 단계에서 결정합니다.
