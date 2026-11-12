# 의사결정 현황 및 미결정 사항

사용자 승인에 따라 확정된 항목과 아직 **TBD**인 항목을 구분하여 기록합니다.

## 1. 확정된 사항 (Approved)

| 결정 항목 | 확정 내용 | 근거 및 반영 위치 |
|---|---|---|
| 클래스 구성 | **4클래스** (`0: AIRPLANE`, `1: BIRD`, `2: DRONE`, `3: HELICOPTER`) | 조류/유인기 오탐 분리 목적, `scripts/mcos_parser.py` 및 `scripts/convert_dataset.py` 반영 |
| 데이터 정책 및 변환 | **DroneDetectionThesis (RGB 285개 우선)** + 자체 순수 Python MCOS 파서 + **마스터 JSON 기반 다중 변환(YOLO/MOT) 및 센서(`V`/`IR`) 분리 계층화 분할(70/15/15)** | [데이터셋 검토 보고서](datasets/DATASET_REVIEW.md), `scripts/` 파이프라인 구현 완료 |
| 비디오 처리 외부 라이브러리 | **OpenCV 도입 승인** (선택적 CMake 옵션으로 구성하여 미설치 환경에서도 코어/단위 테스트 빌드 유지) | `OpenCvFrameSource`로 구현 완료 |
| 추적 방식 | **Kalman Filter + IoU 매칭** 기반 추적기(`ITracker` 구현체) | `KalmanIoUTracker`로 구현 완료 |
| 첫 검출 baseline | **YOLOX-s, 4클래스, 640×640 full-frame** | Issue #19에서 사용자 승인. `training/yolox_s_uav.py`와 Windows 학습·ONNX 검증 도구에 반영 |
| 검출 학습 초기화 | **공식 YOLOX-s COCO 사전학습 가중치 사용** | YOLOX `0.1.1rc0` release의 `yolox_s.pth`; 80클래스 분류 head는 제외하고 호환 계층만 로드 |

## 2. 미결정 사항 (TBD)

| 결정 항목 | 주요 선택지 | 영향 |
|---|---|---|
| 입력 환경 세부 사양 | 실시간 카메라 모델, 화각, 해상도, RTSP 사용 여부 | 비디오 어댑터, 지연, 배포 의존성 |
| 후속 프레임 처리 | `1280 full-frame` vs `overlap tiling` | 첫 baseline은 640 full-frame으로 확정. 후속 방식은 실제 recall·처리량 측정 뒤 결정 |
| Jetson 스택 | JetPack/TensorRT/DeepStream 구체적 고정 버전 | 빌드 이미지, API, 배포 방식 |
| 성능 기준 | FPS, latency, 정확도와 로그 보존 기준 | 테스트 장비, 합격 기준, 메트릭 스키마 |
| 출력 연동 | 콘솔, 파일, 네트워크, UI | 결과 스키마와 운영 구조 |
| 저장소 라이선스 | 프로젝트 자체 LICENSE 선택 | 외부 Apache-2.0/AGPL 의존성 수용 범위와 공개 코드 재사용 조건. 현재 공개 저장소에는 LICENSE 없음 |

미결정(TBD) 항목은 사용자 결정 전까지 임의로 구현이나 버전을 고정하지 않습니다.
