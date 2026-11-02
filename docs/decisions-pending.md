# 의사결정 현황 및 미결정 사항

사용자 승인에 따라 확정된 항목과 아직 **TBD**인 항목을 구분하여 기록합니다.

## 1. 확정된 사항 (Approved)

| 결정 항목 | 확정 내용 | 근거 및 반영 위치 |
|---|---|---|
| 클래스 구성 | **4클래스** (`0: AIRPLANE`, `1: BIRD`, `2: DRONE`, `3: HELICOPTER`) | 조류/유인기 오탐 분리 목적, `scripts/mcos_parser.py` 및 `scripts/convert_dataset.py` 반영 |
| 데이터 정책 및 변환 | **DroneDetectionThesis (RGB 285개 우선)** + 자체 순수 Python MCOS 파서 + **마스터 JSON 기반 다중 변환(YOLO/MOT) 및 센서(`V`/`IR`) 분리 계층화 분할(70/15/15)** | [데이터셋 검토 보고서](datasets/DATASET_REVIEW.md), `scripts/` 파이프라인 구현 완료 |
| 비디오 처리 외부 라이브러리 | **OpenCV 도입 승인** (선택적 CMake 옵션으로 구성하여 미설치 환경에서도 코어/단위 테스트 빌드 유지) | Track B 이슈에서 반영 예정 |
| 추적 방식 | **Kalman Filter + IoU 매칭** 기반 추적기(`ITracker` 구현체) | Track B 이슈에서 반영 예정 |

## 2. 미결정 사항 (TBD)

| 결정 항목 | 주요 선택지 | 영향 |
|---|---|---|
| 검출 모델과 크기 | 후보 모델 계열(YOLO 계열 등), 라이선스(AGPL vs Apache-2.0/MIT), 입력 크기(`640` vs `1280`) | 라이선스, 소형 객체(61.85%가 `<32x32` px) Recall, 지연시간, TensorRT 호환성 |
| 입력 환경 세부 사양 | 실시간 카메라 모델, 화각, 해상도, RTSP 사용 여부 | 비디오 어댑터, 지연, 배포 의존성 |
| 프레임 처리 | full-frame, ROI, tiling과 추론 주기 | 소형 객체 recall, 처리량, 복잡도 |
| Jetson 스택 | JetPack/TensorRT/DeepStream 구체적 고정 버전 | 빌드 이미지, API, 배포 방식 |
| 성능 기준 | FPS, latency, 정확도와 로그 보존 기준 | 테스트 장비, 합격 기준, 메트릭 스키마 |
| 출력 연동 | 콘솔, 파일, 네트워크, UI | 결과 스키마와 운영 구조 |

미결정(TBD) 항목은 사용자 결정 전까지 임의로 구현이나 버전을 고정하지 않습니다.
