# 아키텍처

## 현재 확정된 경계 및 구현체

입력, 탐지, 추적 및 관측 기록은 교체 가능한 포트(`IFrameSource`, `IDetector`, `ITracker`, `IMetricsSink`)로 분리되어 있으며, 현재 구현된 구체 어댑터는 다음과 같습니다.

```text
IFrameSource ---------------------------------> Frame -> IDetector -----------------> Detection[]
  ├─ MotSequenceFrameSource (seqinfo.ini)                 └─ MotReplayDetector (gt/gt.txt) |
  └─ OpenCvFrameSource (cv::VideoCapture, 선택)                                            |
                                                                                           v
                                                                                    ITracker -> Track[]
                                                                                      └─ KalmanIoUTracker
                                                                                           |
                                                                                           v
                                                                                    IMetricsSink
```

`TrackingPipeline`은 프레임 순서 검증(단조 증가 timestamp 보장), `detect` → `update` → `write` 호출 순서 및 처리 개수/시간 집계를 담당합니다.

## 데이터 계약

- `Frame`: `FrameId`, 단조 증가 `Timestamp(microseconds)` 및 `ImageSize`를 포함합니다.
- `Detection`: `class_id`, `confidence`, `class_name` 및 픽셀 좌표 `BoundingBox(x, y, width, height)`입니다.
- `Track`: `TrackId`, `BoundingBox`, `age_frames`, `missed_frames`와 `tentative`, `confirmed`, `lost`, `removed` 상태를 가집니다.
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
- `OpenCvFrameSource`: CMake `UAV_ENABLE_OPENCV=ON` 및 시스템에 OpenCV(`core`, `videoio`)가 설치된 경우 `cv::VideoCapture`로 영상을 읽으며, 미설치 환경에서도 `UAV_HAS_OPENCV=0` 폴백을 통해 코어 라이브러리와 MOT 리플레이가 외부 의존성 없이 동작합니다.

## 실행 계층 (`apps/uav_tracker_main.cpp`)

- `uav-tracker self-check`: 빈 어댑터로 파이프라인 무결성 자가 진단
- `uav-tracker --replay-mot <mot-sequence-dir>`: `MotSequenceFrameSource` + `MotReplayDetector` + `KalmanIoUTracker`를 연결하여 실제 변환된 시퀀스를 End-to-End 리플레이하고 처리 프레임/검출/트랙 수 및 소요 시간을 출력

## 아직 확정하지 않은 설계

추론 주기, 멀티스레딩/큐, ROI/tiling, pixel buffer GPU 공유 형식 및 실제 딥러닝 검출기(TensorRT) 연동은 후속 단계에서 결정합니다.
