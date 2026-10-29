
# Ground-Based Small UAV Detection & Tracking — Agent Instructions

## 1. Project Overview

본 프로젝트는 지상 고정 RGB 카메라를 이용하여 소형 UAV(드론)를 탐지하고, 영상 내 위치를 지속적으로 추적하는 실시간 Computer Vision 시스템을 개발하는 개인 포트폴리오 프로젝트다.

최종적으로 NVIDIA Jetson Orin Nano에서 C++ 기반 Detection 및 Tracking 파이프라인을 실행하고, 정확도와 실시간 처리 성능을 정량적으로 검증한다.

본 프로젝트는 비무장 관측 및 영상 분석을 목적으로 하며, 무기 제어, 요격, 공격용 표적 지정 등의 기능은 범위에 포함하지 않는다.

### 1.1 Project Objectives

- Ground-Based Small UAV Detection
- Drone / Bird / Aircraft Classification
- Real-Time Visual Object Tracking
- C++ 기반 영상처리 및 추론 파이프라인
- TensorRT 기반 Edge AI 추론
- Jetson Orin Nano 성능 최적화
- Detection / Tracking 정량 평가
- 재현 가능한 실험 및 기술 문서화

### 1.2 Current Project Status

- 초기 프로젝트 구조 및 기본 개발 환경 세팅 완료
- GitHub Issue 및 Pull Request 기반 개발 프로세스 도입
- 다음 작업 단계: 데이터셋 구조 분석 및 라벨 검증
- 모델 학습과 실제 추론 엔진 구현은 후속 단계에서 진행

기존 초기 세팅 작업을 불필요하게 반복하지 않는다.

현재 저장소의 실제 구현 상태를 먼저 확인하고, 해당 상태를 기준으로 작업한다.

---

## 2. Technical Stack

### 2.1 Primary Development Environment

- Language: C++20
- Build System: CMake
- Unit Testing: CTest
- Computer Vision: OpenCV
- Video Pipeline: GStreamer
- AI Inference: NVIDIA TensorRT
- GPU Acceleration: CUDA
- Embedded Platform: NVIDIA Jetson Orin Nano
- Operating System: Linux / aarch64
- Development Environment: Linux / x86_64 및 Windows 개발 PC

### 2.2 Supporting Environment

- Python: 데이터 전처리, 모델 학습 및 평가
- PyTorch: 모델 학습
- ONNX: 모델 변환
- Git / GitHub: 버전 관리, Issue, Pull Request
- NVIDIA Profiling Tools: 성능 분석

Python은 데이터 분석과 학습 등에 사용할 수 있지만, 최종 실시간 Detection 및 Tracking 애플리케이션의 핵심 런타임은 C++로 개발한다.

### 2.3 Model and Algorithm Candidates

현재 고려 중인 기본 방향은 다음과 같다.

- Detection: YOLO11n 기반 Baseline
- Classes: Drone, Bird, Airplane, Helicopter
- Tracking: Single-Object Tracking 우선
- Tracking Algorithm: Kalman Filter 기반 추적
- Inference: Full-frame Detection부터 시작
- Runtime: TensorRT C++ API
- Processing: 동기식 구현 후 비동기 파이프라인 확장

이는 현재 권장하는 설계 방향이며, 모든 세부 파라미터나 구현 방식을 최종 확정한 것은 아니다.

모델, 알고리즘, 데이터셋 구성 또는 시스템 아키텍처를 변경해야 한다면 사용자에게 근거와 대안을 제시한 후 결정한다.

---

## 3. Software Architecture

프로젝트는 기능별 모듈화를 기본 원칙으로 한다.

### 3.1 Core Modules

1. VideoSource
   - 영상 파일 입력
   - USB Camera 입력
   - RTSP 입력 확장
   - Frame ID 및 Timestamp 관리

2. FrameProcessor
   - Resize
   - Normalize
   - Color Conversion
   - Detection 입력 데이터 생성

3. DroneDetector
   - 객체 검출
   - Bounding Box 생성
   - Class 및 Confidence 반환
   - 추후 TensorRT 추론 엔진 연동

4. MotionAnalyzer
   - Frame Difference
   - Background Subtraction
   - Optical Flow
   - 이동 객체 후보 분석

5. TargetTracker
   - 위치 예측
   - Detection 결과와 추적 객체 연관
   - Tracking ID 및 상태 관리
   - 검출 실패와 재검출 처리

6. Visualizer
   - Bounding Box 표시
   - Class / Confidence 표시
   - Tracking ID 표시
   - FPS 및 추론 지연시간 표시

7. PerformanceMonitor
   - FPS
   - End-to-End Latency
   - CPU / GPU 사용량
   - Memory Usage
   - Power Consumption

### 3.2 Architecture Rules

- 모듈 간 의존성을 최소화한다.
- 영상 입력, Detection, Tracking, Visualization을 분리한다.
- 알고리즘 구현과 애플리케이션 제어 로직을 분리한다.
- 특정 AI 모델에 종속되지 않는 인터페이스를 우선 설계한다.
- 각 모듈의 입력과 출력 데이터 구조를 명확히 정의한다.
- Frame ID와 Timestamp를 유지한다.
- 실제 처리하지 않은 데이터에 대한 결과를 생성하지 않는다.
- 새로운 기능을 구현할 때 기존 인터페이스와의 호환성을 확인한다.

현재 저장소의 구조와 다른 부분이 있다면 기존 구조를 먼저 분석한다.

사용자 승인 없이 대규모 리팩터링하거나 아키텍처를 전면 변경하지 않는다.

---

## 4. GitHub Development Workflow

본 프로젝트는 Issue-Driven Development와 Pull Request 기반 검토 방식을 사용한다.

모든 주요 기능 개발, 실험, 버그 수정은 관련 GitHub Issue를 기준으로 진행한다.

### 4.1 Standard Workflow

1. Issue 생성
2. 요구사항 및 완료 기준 확인
3. Agent의 구현 계획 작성
4. 사용자 계획 검토 및 승인
5. 기능별 브랜치에서 구현
6. 테스트 및 검증
7. Draft Pull Request 작성
8. 사용자 코드 리뷰 및 이해 확인
9. 사용자 승인 후 Merge

에이전트는 이 절차를 임의로 생략하지 않는다.

단순 오탈자 수정이나 영향 범위가 매우 작은 문서 변경은 Issue 생성을 생략할 수 있으나, 변경 내용과 이유는 보고해야 한다.

### 4.2 Issue Types

다음 세 가지 Issue 유형을 사용한다.

#### Feature

새 기능 구현 또는 기존 기능 개선

필수 항목:
- 해결할 문제
- 구현 목적
- 기능 요구사항
- 완료 기준
- 관련 모듈
- 학습 및 이해 포인트

#### Experiment

모델, 알고리즘, 파라미터 또는 시스템 성능 비교 실험

필수 항목:
- 실험 목적
- 검증할 가설
- Baseline
- 변경 변수
- 데이터셋 및 평가 조건
- 평가 지표
- 결과 판단 기준

#### Bug

오류 수정 및 안정성 개선

필수 항목:
- 문제 설명
- 재현 절차
- 기대 결과
- 실제 결과
- 실행 환경
- 관련 로그
- 수정 후 검증 방법

### 4.3 Issue Scope

하나의 Issue에는 가능한 한 하나의 명확한 목표를 설정한다.

예를 들어 다음 작업은 별도 Issue로 관리하는 것이 좋다.

- 데이터셋 구조 분석
- MATLAB 라벨 변환
- Bounding Box 시각화
- 학습 데이터 분할
- Detection 모델 학습
- TensorRT 추론 구현
- Tracking 알고리즘 구현

불필요하게 여러 작업을 하나의 Issue로 묶지 않는다.

---

## 5. Implementation Planning Rules

에이전트는 주요 기능을 구현하기 전에 구현 계획을 설명해야 한다.

### 5.1 Required Implementation Plan

계획에는 다음 항목을 포함한다.

1. Problem
   - 어떤 문제를 해결하는가?
   - 현재 구현에서 무엇이 부족한가?

2. Proposed Solution
   - 어떤 방식으로 구현할 것인가?
   - 주요 알고리즘과 기술은 무엇인가?

3. Affected Files
   - 어떤 파일을 생성하거나 수정하는가?
   - 각 파일은 어떤 역할을 하는가?

4. Data Flow
   - 입력 데이터는 무엇인가?
   - 어떤 함수와 모듈을 통과하는가?
   - 최종 출력은 무엇인가?

5. Alternatives
   - 고려할 수 있는 다른 구현 방식은 무엇인가?
   - 현재 방식을 선택한 이유는 무엇인가?

6. Validation Plan
   - 어떤 테스트를 수행할 것인가?
   - 완료 여부를 어떻게 판단할 것인가?

7. Risks
   - 예상되는 문제나 제약은 무엇인가?
   - 기존 기능에 영향을 줄 가능성이 있는가?

### 5.2 Approval Rules

다음 작업은 구현 전에 사용자 승인이 필요하다.

- 프로젝트 아키텍처 변경
- 핵심 모듈의 인터페이스 변경
- Detection 모델 변경
- Tracking 알고리즘 변경
- 신규 외부 라이브러리 도입
- 데이터셋 라벨 정책 변경
- 학습 데이터 분할 정책 변경
- 성능 평가 기준 변경
- Jetson 소프트웨어 환경 변경
- 기존 데이터 또는 모델의 삭제·덮어쓰기
- 비용이 발생하는 클라우드 서비스 사용

사용자의 결정이 필요하지만 해당 결정 없이 다른 독립 작업을 진행할 수 있다면, 가능한 작업을 먼저 진행하고 결정이 필요한 항목을 명확히 보고한다.

---

## 6. Git Branch and Commit Rules

### 6.1 Branch Policy

기능별 브랜치를 사용한다.

권장 브랜치 이름:

- feat/issue-12-video-source
- feat/issue-15-label-converter
- experiment/issue-18-input-resolution
- fix/issue-21-frame-timestamp
- docs/issue-24-architecture

main 브랜치에 직접 기능 변경을 반영하지 않는다.

기존 브랜치와 작업 상태를 확인한 뒤 변경을 시작한다.

### 6.2 Commit Policy

Git 커밋 메시지는 한글로 작성한다.

예시:

- feat: 영상 파일 입력 모듈 구현
- feat: Detection 결과 데이터 구조 추가
- fix: 프레임 타임스탬프 처리 오류 수정
- test: Bounding Box 검증 테스트 추가
- docs: 객체 추적 아키텍처 문서 정리
- experiment: 입력 해상도별 추론 성능 평가 추가

하나의 커밋에는 논리적으로 관련된 변경사항을 포함한다.

기존 사용자 변경사항을 임의로 되돌리지 않는다.

별도 승인이 없다면 commit, push 또는 merge를 자동으로 수행하지 않는다.

사용자가 명시적으로 브랜치 푸시와 Draft PR 생성을 승인한 경우 해당 범위에서 수행할 수 있다.

---

## 7. Pull Request Rules

주요 작업이 완료되면 Pull Request 템플릿에 맞춰 변경 내용을 설명한다.

에이전트는 단순히 구현 완료 여부를 보고하는 것이 아니라, 사용자가 변경 내용을 이해할 수 있도록 설명해야 한다.

### 7.1 Required PR Sections

#### 1. Related Issue

- 관련 Issue 번호
- 해결하려는 문제
- 해당 Issue의 완료 기준

#### 2. Summary

- 변경 목적
- 새로 구현한 기능
- 변경 전후의 차이

#### 3. Changed Files

주요 변경 파일과 역할을 정리한다.

예시:

| File | Description |
| --- | --- |
| video_source.cpp | 영상 입력 및 프레임 획득 |
| frame_types.hpp | 프레임 데이터 구조 정의 |
| test_video_source.cpp | 영상 입력 기능 테스트 |

#### 4. Core Code Flow

변경된 코드의 핵심 실행 흐름을 설명한다.

예시:

VideoSource::read()
→ FrameProcessor::preprocess()
→ DroneDetector::infer()
→ TargetTracker::update()

각 함수에 대해 다음을 설명한다.

- 입력 데이터
- 수행하는 작업
- 출력 데이터
- 다음 함수와의 관계

#### 5. Design Decisions

- 이 구현 방식을 선택한 이유
- 고려한 대안
- Trade-off
- 향후 확장 가능성

#### 6. Testing and Validation

실제로 수행한 테스트를 기록한다.

- 실행 명령
- 사용한 환경
- 테스트 결과
- 실패 또는 미검증 항목

테스트를 실행하지 않았다면 통과했다고 표현하지 않는다.

#### 7. Performance Results

성능 관련 변경이라면 다음을 포함한다.

- Baseline
- 변경 후 결과
- 데이터셋
- 평가 조건
- 평가 지표
- 결과 해석

측정하지 않은 수치를 추정하여 결과로 작성하지 않는다.

#### 8. Risks and Limitations

- 알려진 문제
- 미구현 기능
- 성능 또는 호환성 제약
- 추가 검증이 필요한 부분

#### 9. Learning Review

사용자가 이해해야 할 기술 개념을 설명한다.

최소 두 가지의 이해 확인 질문을 작성하고, 질문에 대한 간단한 해설을 제공한다.

### 7.2 Draft PR Policy

- 구현 중인 작업은 Draft PR로 관리할 수 있다.
- 구현이 완료되지 않은 상태를 완료로 표시하지 않는다.
- 테스트 결과와 핵심 코드 설명을 작성한다.
- 사용자 승인 없이 PR을 병합하지 않는다.
- 사용자 피드백이 있다면 수정한 후 결과를 다시 보고한다.

---

## 8. Learning and Explanation Rules

본 프로젝트의 목적은 결과물을 완성하는 것뿐만 아니라 사용자가 C++, Computer Vision 및 Embedded AI 개발 역량을 습득하는 것이다.

따라서 에이전트는 중요한 구현 내용을 설명할 의무가 있다.

### 8.1 Explanation Requirements

새로운 알고리즘 또는 핵심 기능을 구현했다면 다음 내용을 설명한다.

1. What
   - 무엇을 구현했는가?

2. Why
   - 왜 이 기능이나 알고리즘이 필요한가?

3. How
   - 어떤 원리로 동작하는가?

4. Code
   - 실제 어떤 클래스와 함수에서 구현되는가?

5. Verification
   - 올바르게 동작하는지 어떻게 확인했는가?

설명은 한국어를 기본으로 하되 기술 용어, 코드 식별자 및 표준 알고리즘 이름은 원문을 사용할 수 있다.

### 8.2 Code Walkthrough

주요 함수에 대해 다음 내용을 설명한다.

- 함수의 목적
- 입력 및 출력
- 내부 처리 단계
- 다른 모듈과의 연결
- 오류 처리 방식
- 주의해야 할 조건

단순히 코드 내용을 줄별로 번역하는 방식은 피한다.

설계 의도와 데이터 흐름을 중심으로 설명한다.

### 8.3 Understanding Questions

PR에는 사용자가 직접 생각해볼 수 있는 질문 2~3개를 포함한다.

예시:

- Kalman Filter를 사용하는 이유는 무엇인가?
- Detection과 Tracking을 별도 모듈로 분리한 이유는 무엇인가?
- Frame Timestamp를 유지해야 하는 이유는 무엇인가?

질문의 해설을 함께 제공하되, 사용자가 먼저 답을 생각해볼 수 있도록 구분하여 작성한다.

### 8.4 Technical Decision Support

사용자가 기술 선택을 요청하면 다음 순서로 답한다.

1. 선택 가능한 방법
2. 각 방법의 장단점
3. 프로젝트 요구사항과의 적합성
4. 추천 방식
5. 추천 근거
6. 확인해야 할 불확실성

근거 없이 특정 알고리즘이나 라이브러리가 가장 우수하다고 단정하지 않는다.

---

## 9. Dataset Management Rules

### 9.1 Dataset Candidates

현재 검토 중인 공개 데이터셋은 다음과 같다.

1. Drone Detection Dataset
   - https://github.com/DroneDetectionThesis/Drone-detection-dataset

2. Anti-UAV
   - https://github.com/ZhaoJ9014/Anti-UAV

3. Drone-vs-Bird Detection Challenge
   - https://github.com/wosdetc/challenge

현재 우선 검토 대상은 Drone Detection Dataset의 RGB 영상이다.

데이터셋별 접근 조건, 라벨 형식, 클래스 구성 및 사용 허가를 확인해야 한다.

### 9.2 Data Preparation

데이터셋을 학습에 사용하기 전에 다음 내용을 확인한다.

- 영상 수
- 해상도
- FPS
- Class Distribution
- Bounding Box Distribution
- Object Size Distribution
- Missing Annotations
- Invalid Bounding Boxes
- Duplicate Data
- Train / Validation / Test Split

### 9.3 Label Conversion

원본 라벨을 변환할 때 다음 원칙을 따른다.

- 원본 데이터를 수정하지 않는다.
- 변환된 라벨은 별도 디렉터리에 저장한다.
- Bounding Box 좌표계와 정규화 방식을 검증한다.
- 원본과 변환된 라벨의 객체 수를 비교한다.
- 무작위 샘플을 시각화하여 좌표 오류를 확인한다.
- 변환 실패 또는 의심 항목을 로그로 남긴다.

MATLAB .mat 형식은 실제 파일 구조를 확인한 후 적절한 변환 도구를 사용한다.

### 9.4 Data Leakage Prevention

학습·검증·시험 데이터를 분리할 때 영상 시퀀스를 기준으로 분할한다.

동일 영상의 인접 프레임이 서로 다른 데이터 분할에 포함되지 않도록 한다.

분할 방식과 사용한 Random Seed를 기록한다.

### 9.5 Data Protection

다음 파일은 기본적으로 Git 저장소에 커밋하지 않는다.

- 원본 데이터셋
- 대용량 영상 파일
- 개인 또는 비공개 영상
- 모델 가중치
- TensorRT Engine 파일
- 대용량 추론 출력
- 임시 실험 결과

재현 가능한 학습을 위한 설정 파일과 스크립트는 별도로 관리한다.

외부 데이터와 모델의 라이선스를 확인하기 전 재배포하지 않는다.

---

## 10. Model Training Rules

모델 학습은 관련 Issue와 사용자 승인에 따라 진행한다.

### 10.1 Before Training

다음 항목을 확인한다.

- 모델 종류
- Pretrained Weight
- 클래스 구성
- 데이터셋 버전
- Train / Validation / Test Split
- 입력 해상도
- Batch Size
- Epoch
- Learning Rate
- Augmentation
- 학습 장비

### 10.2 Training Records

실험별로 다음 내용을 기록한다.

- Experiment ID
- Model Version
- Dataset Version
- Training Configuration
- Random Seed
- Hardware
- Training Duration
- Validation Metrics
- Model Checkpoint
- 알려진 문제

### 10.3 Evaluation

Detection 성능은 다음 지표를 우선 검토한다.

- Precision
- Recall
- mAP
- Class-wise Metrics
- Drone/Bird Confusion Matrix
- Small Object Recall

필요한 경우 객체 크기별 결과를 별도로 분석한다.

추적 성능은 해당 데이터셋의 정답 형식과 추적 과제에 적합한 지표로 평가한다.

단일 객체 추적과 다중 객체 추적의 평가 지표를 혼동하지 않는다.

---

## 11. C++ Development Rules

### 11.1 Coding Standards

- C++20을 기준으로 작성한다.
- 명확한 클래스 및 함수 이름을 사용한다.
- RAII 기반 리소스 관리를 우선한다.
- 스마트 포인터를 적절히 사용한다.
- 불필요한 전역 상태를 피한다.
- 함수의 역할을 명확히 분리한다.
- 오류 상황을 명시적으로 처리한다.
- 주요 인터페이스에는 필요한 설명을 작성한다.

### 11.2 Performance Considerations

실시간 영상처리 구현 시 다음 사항을 고려한다.

- Frame Copy 비용
- CPU / GPU Memory Transfer
- Inference Latency
- Frame Queue 관리
- Thread Synchronization
- Timestamp Consistency
- Dropped Frame Handling
- Memory Allocation

최적화가 필요하다는 이유만으로 복잡한 구조를 먼저 도입하지 않는다.

기본 동작을 검증한 후 실제 측정 결과를 근거로 최적화를 진행한다.

### 11.3 Testing

가능한 경우 다음 테스트를 수행한다.

- CMake Configure / Build
- CTest
- 모듈 단위 테스트
- 잘못된 입력 처리
- 영상 EOF 처리
- Timestamp 검증
- Bounding Box 검증
- 실제 샘플 영상 테스트

테스트 실행 결과를 변경 보고 또는 PR에 기록한다.

Jetson 하드웨어에서 실행하지 않은 테스트를 Jetson 검증 완료로 표시하지 않는다.

---

## 12. Jetson Deployment Rules

Jetson Orin Nano 배포 시 다음 내용을 확인한다.

- JetPack Version
- CUDA Version
- TensorRT Version
- OpenCV Version
- GStreamer Version
- Power Mode
- 모델 입력 크기 및 Precision
- 사용 가능한 메모리

설치된 환경을 확인하기 전 라이브러리 버전을 임의로 고정하거나 시스템을 업그레이드하지 않는다.

TensorRT Engine의 호환성을 확인하고 필요한 경우 대상 환경에서 다시 생성한다.

### 12.1 Performance Metrics

- Input FPS
- Processed FPS
- End-to-End Latency
- Inference Latency
- Memory Usage
- CPU / GPU Usage
- Power Consumption
- Temperature

측정 조건을 함께 기록한다.

서로 다른 조건에서 얻은 결과를 동일 조건의 성능 비교처럼 표현하지 않는다.

---

## 13. Documentation Rules

코드 구현과 함께 필요한 문서를 유지한다.

### 13.1 README.md

다음 내용을 포함한다.

- 프로젝트 목적
- 주요 기능
- 기술 스택
- 시스템 구조
- 빌드 및 실행 방법
- 현재 구현 상태
- 실험 결과
- 향후 계획

실제로 구현되지 않은 기능을 완성된 기능으로 표시하지 않는다.

### 13.2 Architecture Documentation

시스템 구조가 변경되면 관련 아키텍처 문서를 업데이트한다.

- 모듈 구성
- 데이터 흐름
- 클래스 관계
- 주요 인터페이스
- 설계 결정 및 근거

### 13.3 Experiment Documentation

실험 결과에는 다음을 포함한다.

- 목적
- Baseline
- 실험 조건
- 변경 변수
- 측정 결과
- 해석
- 한계

결과가 없거나 확인되지 않았다면 미측정 또는 미확인으로 명시한다.

---

## 14. Agent Reporting Format

각 작업이 완료되면 다음 순서로 보고한다.

### 1. 작업 결과

- 무엇을 완료했는가?
- 관련 Issue는 무엇인가?

### 2. 주요 변경사항

- 생성한 파일
- 수정한 파일
- 삭제한 파일
- 핵심 구현 내용

### 3. 기술 설명

- 주요 클래스와 함수
- 핵심 알고리즘
- 코드 실행 흐름
- 설계 근거

### 4. 테스트 및 검증

- 실행한 테스트
- 실제 결과
- 통과 여부
- 미검증 항목

### 5. 사용자 확인 사항

- 검토할 코드
- 중요한 기술적 결정
- 이해 확인 질문
- 남아 있는 문제

### 6. Git Information

- Branch Name
- Related Issue
- Commit Message
- PR 상태

커밋 메시지는 한글로 제안한다.

---

## 15. General Safety and Quality Rules

- 기존 사용자 코드를 임의로 삭제하거나 덮어쓰지 않는다.
- 관련 없는 파일을 수정하지 않는다.
- 사용자 승인 없이 대규모 구조 변경을 수행하지 않는다.
- 테스트 실패를 숨기지 않는다.
- 실제 측정하지 않은 성능 수치를 생성하지 않는다.
- 외부 모델이나 데이터셋의 라이선스를 확인한다.
- 비공개 코드, 경로, 토큰 및 인증 정보를 노출하지 않는다.
- 사용자 승인 없이 외부 서비스에 데이터를 업로드하지 않는다.
- 사용자 승인 없이 비용이 발생하는 서비스를 사용하지 않는다.
- 사용자 승인 없이 PR을 Merge하지 않는다.
- 구현 완료와 검증 완료를 구분한다.

---

## 16. Definition of Done

Issue를 완료했다고 판단하려면 다음 조건을 확인한다.

- [ ] Issue의 요구사항을 충족했다.
- [ ] 완료 기준을 검증했다.
- [ ] 관련 테스트를 실행하고 결과를 기록했다.
- [ ] 주요 변경 파일과 코드 흐름을 설명했다.
- [ ] 설계 근거와 Trade-off를 기록했다.
- [ ] 알려진 문제 및 미검증 사항을 명시했다.
- [ ] 필요한 문서를 업데이트했다.
- [ ] PR 템플릿을 작성했다.
- [ ] 사용자가 검토할 항목을 정리했다.
- [ ] 사용자에게 이해 확인 질문을 제공했다.
- [ ] Merge 여부는 사용자가 최종 결정한다.

모든 조건을 충족하지 못했다면 미완료 또는 부분 완료 상태로 보고한다.

---

## 17. Immediate Next Phase — Dataset Preparation

현재 프로젝트의 다음 개발 단계는 데이터셋 준비 및 라벨 검증이다.

다음 작업을 순서대로 추진한다.

1. 데이터셋 파일 및 디렉터리 구조 조사
2. 영상 및 라벨 형식 확인
3. 클래스 및 Bounding Box 분포 분석
4. 라벨 변환 도구 구현
5. 변환 결과 시각화 및 검증
6. 영상 단위 데이터 분할 계획 수립
7. 학습 데이터셋 구성 및 검증

이 단계에서는 사용자 승인 없이 본격적인 모델 학습을 시작하지 않는다.

각 작업은 별도의 Issue 또는 명확하게 정의된 작업 단위로 관리한다.

데이터셋 품질과 분포를 확인한 후 모델 학습 단계로 진행한다.

---

## 18. Final Principle

본 프로젝트에서 에이전트의 역할은 코드를 대신 작성하는 것에만 한정되지 않는다.

에이전트는 프로젝트 소유자가 시스템의 아키텍처, 알고리즘, 코드 실행 흐름 및 실험 결과를 이해하고 스스로 기술적 결정을 내릴 수 있도록 지원해야 한다.

모든 주요 작업은 다음 원칙을 따른다.

**Plan → Implement → Test → Explain → Review → Merge**

작업 속도보다 검증 가능성, 코드 품질, 재현성 및 사용자 이해를 우선한다.
