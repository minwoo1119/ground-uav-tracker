# 아키텍처

## 현재 확정된 경계

초기 구조는 입력, 탐지, 추적 및 관측 기록을 교체 가능한 포트로 분리합니다.

```text
IFrameSource -> Frame -> IDetector -> Detection[]
                             |
                             +-> ITracker -> Track[]
                                             |
                                             +-> IMetricsSink
```

`TrackingPipeline`은 프레임 순서, 호출 순서 및 처리 개수 집계만 담당합니다. 프레임 timestamp가 이전 프레임보다 오래되면 오류를 반환합니다.

## 데이터 계약

- `Frame`: frame ID, 단조 증가 timestamp 및 영상 크기만 포함합니다. pixel buffer의 소유권과 CPU/GPU 메모리 형식은 TBD입니다.
- `Detection`: 클래스 ID, confidence, 클래스 표시 이름 및 픽셀 좌표 바운딩 박스입니다.
- `Track`: track ID, 바운딩 박스, 나이/누락 프레임 수와 `Tentative`, `Confirmed`, `Lost`, `Removed` 상태를 가집니다.
- `FrameMetrics`: frame ID와 timestamp, 탐지/트랙 개수 및 파이프라인 처리 시간을 전달합니다. 성능 합격 기준은 포함하지 않습니다.

## 실행 계층

`apps/uav_tracker_main.cpp`가 CLI와 향후 구체 어댑터 조립을 담당합니다. 현재 self-check용 빈 구현은 탐지와 트랙을 생성하지 않습니다. 구성 파일 로더, 영상 디코더, 모델 런타임 및 추적 알고리즘은 없습니다.

## 아직 확정하지 않은 설계

추론 주기, 스레딩과 큐, 프레임 드롭, ROI/tiling, pixel buffer 형식, 추적 알고리즘 및 GPU 메모리 경계는 사용자 결정 전이므로 현재 아키텍처의 확정 사항이 아닙니다.
