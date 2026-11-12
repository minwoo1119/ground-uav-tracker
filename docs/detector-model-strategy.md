# 검출 모델 후보와 Jetson 배포 전략

> 조사 기준일: 2026-11-12  
> 상태: **추천안 작성 완료 / 사용자 승인 대기(TBD)**

이 문서는 실제 모델을 다운로드하거나 성능을 측정한 결과가 아니다. 공식 저장소와 공식 문서에서 확인한 라이선스·배포 경로를 현재 데이터와 런타임 경계에 대입한 사전 의사결정 자료다. 모델 계열, 크기, 입력 방식과 외부 패키지 설치는 사용자 승인 전까지 확정하지 않는다.

## 1. 현재 프로젝트 조건

- 학습 클래스는 `AIRPLANE`, `BIRD`, `DRONE`, `HELICOPTER` 4개로 확정됐다.
- RGB 원본 285개 시퀀스는 `640x512`이며, 96,245개 BBox 중 54.78%가 COCO Small이다.
- DRONE BBox의 61.85%, BIRD BBox의 80.83%가 면적 `32x32 px` 미만이다.
- 변환기는 현재 YOLO 라벨과 MOT 라벨을 생성하지만 COCO JSON은 생성하지 않는다.
- C++ 런타임은 `Frame -> IDetector -> Detection[] -> ITracker` 경계를 이미 갖지만 실제 픽셀 버퍼와 TensorRT adapter는 미구현이다.
- 저장소는 공개 상태지만 프로젝트 자체 `LICENSE` 파일은 아직 없다. 공개 저장소라는 사실만으로 재사용 권한이나 AGPL 호환성이 자동 결정되지는 않는다.

## 2. 비교 기준

1. **라이선스 명확성**: 공개 포트폴리오와 향후 재사용에 미치는 영향
2. **데이터 연결 비용**: 현재 YOLO 라벨을 바로 사용하는지, COCO 변환이 필요한지
3. **학습 재현성**: 4클래스 custom dataset과 설정을 명시적으로 관리할 수 있는지
4. **배포 경로**: PyTorch -> ONNX -> TensorRT -> C++ 후처리가 공식적으로 설명되는지
5. **Jetson 부담**: 모델 크기뿐 아니라 전처리, 출력 decode, NMS, plugin 의존성까지 포함
6. **소형 객체 실험성**: 640 baseline 이후 입력 확대나 tiling을 독립 변수로 비교할 수 있는지

공식 COCO 수치와 데스크톱 GPU 속도는 이 프로젝트 데이터나 Jetson Orin Nano 성능을 보장하지 않으므로 선택 근거의 보조 정보로만 사용한다.

## 3. 후보 비교

| 후보 | 공식 라이선스 확인 | 현재 데이터 연결 | 공식 배포 근거 | 장점 | 주요 제약 |
|---|---|---|---|---|---|
| **YOLOX-s** | 공식 저장소 코드 `Apache-2.0` | 공식 custom-data 경로는 COCO/VOC 중심이므로 COCO exporter 추가 권장 | 공식 저장소가 ONNX와 TensorRT C++/Python 배포 예제를 제공 | 640 기준 모델이 있고, decode/NMS 흐름이 전통적인 detector라 C++ adapter 책임이 비교적 명확함 | 공식 README의 주요 업데이트 기록이 오래됐고 최신 PyTorch/ONNX/TensorRT 조합은 실제 설치 후 검증 필요 |
| **RT-DETRv2-S** | 공식 저장소 코드 `Apache-2.0` | COCO-style dataset 준비가 가장 자연스러워 COCO exporter 필요 | 공식 저장소가 ONNX Runtime, TensorRT, OpenVINO 지원과 TensorRT benchmark 도구를 안내 | 비교적 최근 계열이며 공식 저장소에 small-object용 sliced inference 지원이 기록됨 | 공식 표 기준 640에서 20M params/60 GFLOPs로 YOLOX-s(9M/26.8 GFLOPs)보다 무거움. Jetson C++ 출력 계약과 실제 parser 호환성을 별도 검증해야 함 |
| **Ultralytics YOLO26n** | `AGPL-3.0` 또는 유료 Enterprise. 공급자 설명상 자체 학습 모델도 적용 대상 | 현재 YOLO 라벨을 직접 사용 가능 | 공식 export가 ONNX와 TensorRT 형식을 지원 | 학습·검증·export 연결이 가장 짧고 문서화가 활발함 | 현재 저장소 전체의 라이선스 정책과 충돌 가능. AGPL 수용 또는 Enterprise 계약 전에는 기본안으로 고정하면 안 됨 |

### 가중치에 대한 별도 주의

- 위 표의 Apache-2.0은 각 공식 **코드 저장소**에서 확인한 값이다.
- YOLOX와 RT-DETR 사전학습 가중치는 취득 시점에 release asset의 별도 조건, 출처, checksum과 재배포 가능 여부를 다시 확인한다. 코드 라이선스만 보고 가중치 조건까지 같다고 단정하지 않는다.
- Ultralytics는 공식 라이선스 안내에서 코드, 모델, 학습 파이프라인과 fine-tuned model에 AGPL-3.0 또는 Enterprise 조건이 적용된다고 명시한다.
- 이 문서는 법률 자문이 아니며, 실제 상업 배포 전에는 적용 라이선스를 별도로 검토해야 한다.

## 4. 추천안

### 기본 추천: YOLOX-s + 640 letterbox

현재 목표가 공개 포트폴리오, Jetson C++ 연결, 라이선스 부담 최소화라면 첫 baseline으로 **YOLOX-s**를 추천한다.

- Apache-2.0 코드 기반이라 Ultralytics의 AGPL/Enterprise 선택을 먼저 강제하지 않는다.
- 공식 저장소에 ONNX export와 TensorRT C++ 배포 경로가 존재한다.
- YOLOX-Nano의 공식 기준 입력은 416이므로, `640x512` 원본의 작은 객체를 첫 실험부터 더 축소하지 않기 위해 640 기준인 `s`를 우선한다.
- 단점은 COCO exporter를 추가해야 하고 upstream/toolchain 호환성을 직접 검증해야 한다는 점이다.

이 추천은 성능 우승 주장이 아니다. **첫 재현 가능한 기준선**을 만들기 위한 엔지니어링 선택이다.

### 대안 A: RT-DETRv2-S

소형 객체 sliced inference와 비교적 최근 구조를 우선하면 두 번째 후보로 적합하다. 다만 첫 Jetson baseline부터 선택하면 모델 계산량과 export/postprocess 검증이 동시에 늘어난다. YOLOX-s baseline 이후 동일 COCO split에서 비교하는 편이 원인 분석에 유리하다.

### 대안 B: Ultralytics YOLO26n

사용자가 저장소와 산출물에 대한 AGPL-3.0 의무를 명시적으로 수용하거나 Enterprise 라이선스를 확보한다면 가장 짧은 학습·export 경로다. 그 결정 전에는 편의성만으로 선택하지 않는다.

## 5. 입력 크기와 tiling 전략

### 1단계: 640 full-frame letterbox

- `640x512` 원본을 비율 유지한 채 `640x640`에 padding한다.
- 원본 객체를 축소하지 않으면서 가장 단순한 baseline을 만든다.
- 모든 후속 실험은 동일 video-level split과 seed를 유지한다.

### 2단계: 1280 full-frame 비교

- 원본 영상을 확대하는 것이므로 새로운 광학 정보가 생기지는 않는다.
- feature-map에서 객체가 차지하는 셀 수는 늘 수 있지만 픽셀 수와 연산량은 640 정사각 입력 대비 약 4배가 된다.
- 따라서 640의 Small/Drone recall이 부족할 때만 독립 실험으로 실행한다.

### 3단계: overlap tiling 비교

- 예: 겹침을 둔 2x2 crop을 각 640 입력으로 추론하고 원본 좌표로 합친다.
- 작은 객체 확대 효과가 있지만 프레임당 추론 횟수, 경계 중복과 merge/NMS 복잡도가 증가한다.
- 첫 baseline에 포함하지 않고 1280 full-frame과 정확도·지연시간을 동일 조건에서 비교한다.

**추천 순서:** `640 full-frame -> 1280 full-frame -> overlap tiling`. 1280과 tiling을 처음부터 함께 넣으면 모델 차이와 입력 전략 차이를 분리하기 어렵다.

## 6. 권장 학습·배포 경계

```text
Windows 개발/학습 환경
  RGB 영상 + video-level split + 4-class labels
  -> 학습 프레임 추출 및 COCO annotation 생성
  -> 선택한 PyTorch detector 학습
  -> checkpoint (로컬 artifact)
  -> ONNX export
  -> ONNX Runtime 기준 출력 검증

Jetson Orin Nano 대상 환경
  ONNX + 고정된 JetPack/TensorRT 조합
  -> trtexec 또는 TensorRT builder로 FP16 engine 생성
  -> C++ TensorRTDetector가 전처리/inference/decode 수행
  -> Detection[class_id, confidence, bbox]
  -> 기존 KalmanIoUTracker
  -> MOT writer / metrics
```

ONNX를 교환 산출물로 두고 TensorRT engine은 대상 Jetson에서 생성한다. NVIDIA 문서상 engine은 기본적으로 TensorRT 버전, 플랫폼과 GPU 호환성의 영향을 받으며 JetPack에서는 일반 hardware-compatibility 모드가 지원되지 않기 때문이다. Windows에서 만든 engine을 Jetson으로 복사하는 경로를 기본값으로 삼지 않는다.

## 7. 첫 baseline 실험 계획

모델 승인 후 별도 Issue에서 다음 순서로 구현한다.

1. RGB split을 그대로 사용하는 COCO exporter와 schema 검증 추가
2. 선택 모델의 버전·라이선스·사전학습 가중치 출처를 lock file/실험 기록에 고정
3. `640 letterbox`, 4클래스, 고정 seed로 한 번의 smoke training 수행
4. 학습 중단/재개, 잘못된 dataset path와 class mapping 오류 검증
5. validation에서 class-wise Precision/Recall/mAP, Small-object recall, Drone/Bird confusion 기록
6. ONNX export 후 PyTorch와 출력 shape·좌표 복원·NMS 결과 비교
7. Jetson에서 FP16 engine을 생성하고 batch 1 latency/FPS/memory 측정
8. 기준선이 재현된 뒤에만 1280과 tiling 실험 분리

Epoch, batch size, augmentation, 합격 기준과 JetPack/TensorRT 버전은 장비와 모델 승인 후 결정한다. 이 문서에서는 임의의 성능 목표를 정하지 않는다.

## 8. 사용자 승인 요청

다음 구현으로 넘어가려면 아래 결정을 승인해야 한다.

1. **기본 추천:** YOLOX-s, Apache-2.0 코드, 640 full-frame baseline
2. **데이터 확장:** 기존 split을 보존하는 COCO exporter 추가
3. **외부 리소스:** 승인된 버전의 학습 패키지 설치와 사전학습 가중치 취득
4. **저장소 라이선스:** 프로젝트 자체 LICENSE를 별도 Issue에서 선택할지 여부

승인 전에는 모델 다운로드, 패키지 설치, 학습, ONNX/TensorRT 산출물 생성을 수행하지 않는다.

## 9. 공식 출처

- [YOLOX 공식 저장소 및 Apache-2.0 라이선스](https://github.com/Megvii-BaseDetection/YOLOX)
- [YOLOX 공식 TensorRT C++ 예제](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/TensorRT/cpp)
- [RT-DETR 공식 저장소 및 Apache-2.0 라이선스](https://github.com/lyuwenyu/RT-DETR)
- [RT-DETR 공식 TensorRT benchmark 안내](https://github.com/lyuwenyu/RT-DETR/blob/main/benchmark/trtexec.md)
- [Ultralytics 공식 라이선스 안내](https://www.ultralytics.com/license)
- [Ultralytics 공식 export 문서](https://docs.ultralytics.com/modes/export/)
- [NVIDIA TensorRT ONNX architecture 안내](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html#onnx)
- [NVIDIA TensorRT engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)

