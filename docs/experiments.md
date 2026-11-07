# 실험 기록

아래 양식은 모델과 평가 기준을 결정한 뒤 사용합니다. 현재 기록된 수치는 GT detection replay 기반 tracker 진단이며 실제 검출 모델 성능이 아닙니다.

## 실험 정보

- 실험 ID:
- 날짜:
- 목적:
- 코드 리비전:
- 장비/소프트웨어 환경:
- 데이터셋 및 분할:
- 모델/추적 구성:
- 추적 결과 파일(MOTChallenge, 선택):

## 측정 항목

- 검출 정확도:
- 소형 객체 recall:
- 새/기타 객체 오탐:
- tracking success:
- FPS:
- latency:

## 결과 및 관찰

- 결과:
- 제한 사항:
- 후속 실험:

## 추적 결과 파일 주의사항

`--output-tracks <path>`로 생성한 파일은 재현 가능한 후속 평가 입력이며, 파일 생성 자체가 추적 성능 검증을 의미하지 않습니다. 현재 writer는 confidence와 3D 좌표를 `-1`로 기록하고 공식 MOTA/IDF1을 계산하지 않습니다. 평가 도구, 매칭 기준 및 합격 기준이 결정되기 전에는 이 파일로 성능을 주장하지 않습니다.

## TrackEval 기록 항목

- 평가 도구 및 버전: `trackeval==1.3.0`
- 평가 범위: GT detection replay 기반 tracking-only
- CLEAR/Identity IoU threshold: 기본 `0.5` 또는 실행값
- HOTA / DetA / AssA / LocA:
- MOTA / MOTP / IDSW / Frag:
- IDF1 / IDP / IDR:
- GT ID 출처: 원본 제공 / 변환기 파생(현재 데이터셋은 파생)

원본 라벨에 객체 ID가 없으므로 현재 MOT GT의 ID 기반 지표는 변환기의 IoU 연결 결과에 영향을 받습니다. 합격 기준은 여전히 TBD이며, 실제 검출 모델을 연결하기 전 결과는 end-to-end 탐지·추적 성능이 아닙니다.

## 2026-11-07 Tracking-only 샘플 기준선

- 목적: `KalmanIoUTracker`의 GT detection replay 결과와 TrackEval 어댑터 연결 검증
- 환경: Windows 11, MSVC Release, `trackeval==1.3.0`, Python 3.12
- 데이터: 로컬 변환 샘플(Git 제외), 결과 파일도 커밋하지 않음
- CLEAR/Identity IoU threshold: `0.5`
- HOTA: TrackEval alpha `0.05..0.95` 평균
- 합격 기준: 없음(TBD)

| 시퀀스 | GT boxes | Tracker rows | HOTA | MOTA | IDF1 | IDSW | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `V_DRONE_001` | 301 | 301 | 0.9780 | 1.0000 | 1.0000 | 0 | 0 | 0 |
| `V_AIRPLANE_007` | 300 | 305 | 0.7963 | 0.9800 | 0.8430 | 1 | 0 | 5 |
| `V_BIRD_009` | 1229 | 1254 | 0.9793 | 0.9797 | 0.9899 | 0 | 0 | 25 |
| `V_DRONE_108` | 520 | 525 | 0.9869 | 0.9904 | 0.9952 | 0 | 0 | 5 |
| `V_HELICOPTER_040` | 851 | 856 | 0.9942 | 0.9941 | 0.9971 | 0 | 0 | 5 |
| `IR_BIRD_001` | 384 | 394 | 0.7587 | 0.9714 | 0.7481 | 1 | 0 | 10 |

해석 제한: tracker는 실제 detector 출력이 아니라 GT box를 입력받았습니다. GT ID도 원본 제공값이 아니라 변환기의 탐욕적 IoU 연결로 파생됐습니다. 따라서 이 표는 평가 파이프라인과 연관 로직의 회귀 기준선이며 모델 정확도나 배포 성능 비교표가 아닙니다.
