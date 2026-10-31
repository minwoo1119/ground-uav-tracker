# Agent Instructions — Ground-Based Small UAV Detection & Tracking

## 0. 필수 GitHub 개발 워크플로

모든 신규 개발 작업과 새로운 에이전트 세션은 코드나 문서를 변경하기 전에 이 파일과 `docs/AGENT_WORKFLOW.md`를 전체 읽고 동일한 절차를 적용한다.

1. 작업 시작 전 Git 상태, 현재 브랜치, 원격 기본 브랜치, GitHub CLI 인증 상태를 확인하고 기존 Issue와 PR에서 중복 작업을 검색한다.
2. 적합한 Issue가 없으면 Background, Objective, Scope, Implementation Plan, Acceptance Criteria, Test Plan, Open Questions를 포함한 Issue를 먼저 생성한다.
3. 중요한 설계 결정이나 이 문서의 “사용자 승인 필요” 항목이 있으면 Issue에 대안과 영향을 기록하고 구현 전에 사용자에게 질문한다.
4. 신규 작업은 `issue/<번호>-<간단한-작업명>` 형식의 전용 브랜치에서 수행한다. 가능하면 `gh issue develop`로 Issue와 브랜치를 연결한다.
5. 하나의 Issue는 원칙적으로 하나의 작업 브랜치와 하나의 PR에 연결한다. 범위가 커지면 별도 Issue로 분리한다.
6. 기본 브랜치에서 직접 개발하거나 Push하지 않는다. 인증 부재, 미커밋 변경, 브랜치 충돌이 있으면 우회하지 말고 사용자에게 보고한다.
7. 구현 후 관련 테스트의 실제 결과와 미검증 항목을 확인하고, 변경된 파일만 선택적으로 Stage하여 작업 내용을 반영한 한글 커밋을 생성한다.
8. 작업 브랜치를 Push한 뒤 `gh pr create --draft`로 Draft PR을 생성하고 Issue를 `Closes #<번호>`로 연결한다.
9. PR 본문에는 목적, 파일별 역할, 입력→처리→출력→모듈 연결 흐름, 설계 결정과 대안, 실제 테스트 결과, 제약, 핵심 코드, 이해 확인 질문과 해설을 기록한다.
10. 사용자의 명시적 요청 없이 PR을 Ready for Review로 바꾸거나 Merge하지 않는다. 최종 검토와 Merge는 사용자가 수행한다.
11. 사용자가 수정한 파일과 커밋되지 않은 변경을 임의로 덮어쓰거나 삭제하지 않는다.

Issue 기반 개발 요청은 위 절차에 필요한 Issue 생성, 전용 브랜치 생성, 한글 커밋, 작업 브랜치 Push와 Draft PR 생성을 승인한 것으로 본다. 원격 설정 변경, 기본 브랜치 직접 Push, Ready 전환과 Merge는 포함하지 않는다.

## 1. 역할과 현재 작업 범위

당신은 C++/컴퓨터 비전/Jetson 임베디드 개발을 지원하는 코딩 에이전트다.
이 저장소는 **지상 고정 RGB 카메라에서 소형 드론을 탐지하고 영상 내 위치를 추적하는 비무장 관측용 포트폴리오 프로젝트**다. 장기적으로 NVIDIA Jetson Orin Nano에서 C++로 실행한다.

**이번 요청의 목표는 실행 가능한 초기 프로젝트 골격(scaffold)을 만드는 것**이다. 실제 탐지 모델의 종류, 학습 방법, 추적 알고리즘, 성능 목표 등은 아직 사용자 결정 전이다. 임의로 확정하지 말 것.

이번 단계에서는 데이터셋 다운로드, 모델 학습, 추론 엔진 생성, 실제 탐지 성능 주장, 드론 제어, 카메라 구동부 제어, 방해·요격·무기 연동을 수행하지 않는다.

## 2. 이미 합의된 제약

- 대상: 지상에서 촬영한 하늘의 소형 드론. **드론에 장착된 카메라가 지상을 보는 VisDrone 유형과 혼동하지 말 것.**
- 1차 범위: RGB 입력, 자동 드론 탐지, 단일 드론 영상 추적, 관측 화면과 성능 로깅.
- 추후 검토: 드론/새/항공기 구별, 복수 드론 추적, IR 입력, 움직임 기반 후보 영역 분석.
- 주 언어: C++20. 모델 학습·데이터 분석 보조 스크립트에 Python을 사용할 수 있다.
- 주 실행 환경: Jetson Orin Nano의 Linux/aarch64. 개발 PC의 Linux/x86_64에서도 최소 단위 테스트가 실행되도록 한다.
- 빌드: CMake 및 CTest. CI는 외부 서비스 연결 없이 로컬에서 검증 가능한 범위만 준비한다.
- 외부 프레임워크와 모델은 버전 호환성을 확인한 뒤 선택한다. **특정 YOLO 버전, Tracker, JetPack/DeepStream 버전을 아직 고정하지 말 것.**
- 모델/데이터가 없는 초기 상태에서는 실제 탐지나 추적이 수행된 것처럼 출력하지 않는다.
- 포트폴리오용 공개 소스에 비공개 업무 코드·데이터·경로·토큰을 넣지 않는다.

## 3. 초기 세팅에서 직접 실행할 작업

### A. 저장소 상태 확인

1. 현재 작업 디렉터리, 기존 파일, Git 상태, 최상위 `AGENTS.md`를 확인한다.
2. 기존 내용이 있다면 보존한다. 사용자 변경을 삭제하거나 덮어쓰지 않는다.
3. 이미 Git 저장소인 경우 기존 브랜치/리모트를 변경하지 않는다. Git이 없는 새 디렉터리에서는 필요하면 로컬 `git init`만 수행할 수 있다.

### B. 최소 프로젝트 구조 생성

아래를 출발점으로 생성하되 기존 구조가 있으면 충돌 없이 조정한다.

```text
ground-uav-tracker/
├── AGENTS.md
├── CMakeLists.txt
├── README.md
├── .gitignore
├── cmake/
├── apps/
│   └── uav_tracker_main.cpp
├── include/uav/
│   ├── core/
│   ├── video/
│   ├── detection/
│   ├── tracking/
│   └── telemetry/
├── src/
│   ├── core/
│   ├── video/
│   ├── detection/
│   ├── tracking/
│   └── telemetry/
├── configs/
│   └── example.yaml
├── scripts/
├── tests/
├── docs/
│   ├── architecture.md
│   ├── decisions-pending.md
│   └── experiments.md
└── data/
    └── README.md
```

폴더명과 소스 파일명은 적절히 조정 가능하지만, 역할별 모듈 경계와 파일 역할을 README에 기록한다.

### C. CPU-only 최소 빌드 준비

1. CMake에서 C++20과 적절한 경고 옵션을 설정한다.
2. Jetson/TensorRT 없이 실행 가능한 `UAV_ENABLE_TENSORRT=OFF` 구성과 최소 CLI 프로그램을 만든다.
3. `--help`, `--version`, `--config`와 입력 소스 옵션의 파싱 및 오류 메시지를 준비한다. 모델이 없는 실행에서 허위 Detection 결과를 생성하지 않는다.
4. 순수 데이터 타입과 인터페이스를 정의하되, 알고리즘은 결정 전까지 추상 인터페이스/명시적인 미구현 상태로 남긴다.
5. 외부 라이브러리를 다운로드·설치하지 않고도 가능한 최소 빌드부터 검증한다. OpenCV나 TensorRT의 부재는 명확히 보고한다.

### D. 인터페이스/테스트 골격

기존 구조가 없다면 다음을 기준으로 시작한다.

- `Frame`: frame ID, 단조증가 timestamp, 영상 크기 등 메타데이터. 실제 pixel buffer의 소유권/메모리 타입은 아직 결정하지 않는다.
- `Detection`: 클래스 ID, confidence, `x/y/width/height`의 경계 상자.
- `Track`: track ID, 경계 상자, 상태(`Tentative`, `Confirmed`, `Lost`, `Removed`).
- `IDetector`: 명시적 입력/출력을 갖는 검출기 인터페이스. 구현체가 없음을 숨기지 않는다.
- `ITracker`: 순서 있는 프레임과 검출 결과를 입력받는 추적기 인터페이스.
- `IFrameSource` 및 `IMetricsSink`: 입력과 관측 결과 기록의 추상화.

알고리즘 의존성이 없는 단위 테스트를 작성한다. 예: Bounding Box 형식 검증, 잘못된 옵션 처리, timestamp 순서, Track 상태 열거형 처리, 설정 누락 메시지. 임의의 FPS/정확도 수치가 나오는 가짜 테스트를 만들지 않는다.

### E. 데이터·모델 준비 문서화

1. `data/README.md`에 데이터 저장 규칙 및 원본 데이터 Git 제외 정책을 작성한다.
2. 사용 후보를 **참조 정보로만** 기록한다.
   - DroneDetectionThesis: https://github.com/DroneDetectionThesis/Drone-detection-dataset
   - Anti-UAV: https://github.com/ZhaoJ9014/Anti-UAV
   - Drone-vs-Bird: https://github.com/wosdetc/challenge
3. 데이터 구조를 실제로 확인하기 전 임의의 라벨 변환 성공을 주장하지 않는다. MATLAB `.mat` 레이블은 별도 검증이 필요하다고 표시한다.
4. 전처리/학습 스크립트는 사용자가 클래스, 데이터 분할, 모델을 결정한 뒤 구현한다. 초기에는 읽기 전용 데이터셋 검사 스크립트나 사용법 문서 정도만 준비할 수 있다.
5. 원본 영상, 라벨, 모델 가중치, TensorRT engine, 실험 출력은 `.gitignore`에서 제외한다. 재현에 필요한 설정/스크립트는 유지한다.

### F. 문서 및 작업 보고

1. README: 목적, 범위, 미구현 항목, 지원 플랫폼, 빌드/테스트 명령, 예상 향후 작업, 비무장 관측 용도.
2. `docs/architecture.md`: 현재 확정된 모듈 경계만 기술. 추론 주기, 스레딩 설계, Tracker 선택 등을 확정된 사실처럼 쓰지 않는다.
3. `docs/decisions-pending.md`: 사용자 결정이 필요한 선택지와 그에 따른 영향. `TBD` 상태 유지.
4. `docs/experiments.md`: 향후 비교 실험(검출 정확도, 소형 객체 Recall, 새 오탐, Tracking Success, FPS/Latency)을 위한 빈 실험 기록 양식. 숫자를 채우지 않는다.
5. 완료 보고에 **생성/변경 파일, 빌드·테스트 명령과 결과, 미검증 사항, 다음 사용자 결정 사항**을 구분하여 적는다.
6. 작업 단계 종료 시 변경에 적합한 **한글 Git 커밋 메시지**를 사용한다. Issue 기반 개발 작업은 0절에 따라 commit/push/Draft PR까지 수행하고, 단순 조사·설명처럼 Issue 기반 개발이 아닌 요청은 별도 지시가 없으면 commit/push하지 않는다.

## 4. 에이전트가 결정해도 되는 것 / 안 되는 것

**자율적으로 결정 가능:** 일반적인 폴더 세부명, CMake 타깃 이름, 헤더 가드 방식, 코드 스타일, 단위 테스트 프레임워크의 경량화 여부(추가 다운로드 없이 가능해야 함), 기본 CLI 도움말.

**사용자 승인 필요:**

- 검출 모델 계열/크기 및 라이선스 선택(예: YOLO11n vs 다른 모델)
- 2클래스(Drone/Bird) vs 4클래스(Drone/Bird/Airplane/Helicopter) 학습
- 카메라 모델, 화각, 해상도, RTSP 사용 여부
- 프레임 단위 Full-frame 추론 vs ROI/Tiling, 추론 주기
- 단일 객체 추적 알고리즘 선택(Kalman + association, Optical Flow, SOT 등)
- 다중 객체 추적(MOT)의 1차 포함 여부
- 카메라 이동/팬틸트, IR, 센서 융합, 외부 알림 통합
- JetPack/TensorRT/DeepStream의 구체적 고정 버전과 배포 방식
- FPS/Latency/정확도 합격 기준, 로그 보존 방식
- 데이터 다운로드·학습·가중치 취득·추가 패키지 설치 등 외부 리소스 사용

결정을 기다리는 동안 해당 기능을 끼워 넣지 말고, 의존성이 없는 초기 골격 작업을 먼저 끝낸다.

## 5. 품질 규칙

- 기존 변경 사항을 보존하고, 파괴적인 Git/파일 작업을 하지 않는다.
- 에러를 삼키지 않는다. 입력 파일, 모델, 디바이스가 없으면 구체적 오류를 반환한다.
- 비디오 처리 순서와 추적 결과 시각을 혼동하지 않도록 timestamp를 유지한다.
- C++ 런타임과 오프라인 Python 학습/평가 코드의 경계를 명확히 한다.
- 모델 성능을 검증하지 않고 `real-time`, `high accuracy`, `military-grade` 같은 성과 표현을 README에 쓰지 않는다.
- 외부 데이터·모델 라이선스와 재배포 조건을 확인하기 전 파일을 저장소에 커밋하지 않는다.
- 임의의 절대경로, 개발자 개인 컴퓨터 정보, 비밀 설정을 코드에 넣지 않는다.

## 6. 초기 단계 완료 기준(Definition of Done)

- [ ] 저장소 구조와 핵심 문서 생성
- [ ] 외부 AI 모델 없이 CMake configure/build 성공(가능한 환경에서)
- [ ] CLI `--help`와 잘못된 옵션의 오류 처리 확인
- [ ] CTest 단위 테스트 통과(가능한 환경에서)
- [ ] Jetson 전용 기능이 없는 개발 환경에서도 실패 원인이 명확함
- [ ] 데이터/모델/실험 출력이 Git에서 제외됨
- [ ] 사용자 결정 사항이 확정값 없이 별도 문서에 정리됨
- [ ] 변경 파일 및 실제 실행한 검증 명령/결과 보고
- [ ] 다음 단계로 넘어갈 때 필요한 질문과 한글 Git 커밋 메시지 제안

완료되지 않은 항목은 완료되었다고 표현하지 말고 원인과 다음 검증 방법을 적는다.
