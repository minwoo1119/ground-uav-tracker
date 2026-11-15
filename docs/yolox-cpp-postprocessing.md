# YOLOX C++ 후처리 계약

`YoloXPostprocessor`는 YOLOX graph 실행과 분리된 순수 C++20 변환 계층입니다. ONNX Runtime 또는 TensorRT adapter가 batch 1의 raw tensor를 만들면 이 계층이 프로젝트의 `Detection[]`으로 바꿉니다. 현재 구현은 실제 graph를 실행하지 않으며 모델 정확도나 처리 성능을 의미하지 않습니다.

## 입력 계약

Issue #19의 4클래스 640 baseline은 `decode_in_inference=false`로 export됩니다.

- shape: `[1, 8400, 9]`; postprocessor에는 batch 축을 제외한 연속 `float` span 전달
- anchor 순서: stride 8의 `80×80`, stride 16의 `40×40`, stride 32의 `20×20`을 순서대로 연결
- feature: `raw_x, raw_y, raw_log_w, raw_log_h, objectness, class_0..class_3`
- objectness와 class score: graph에서 sigmoid가 적용된 `[0,1]` 확률
- 클래스 순서: `AIRPLANE`, `BIRD`, `DRONE`, `HELICOPTER`

길이 불일치, 잘못된 설정, NaN/Inf, 확률 범위 위반은 `std::invalid_argument`로 거부합니다. threshold는 실험 결과 없이 고정하지 않도록 호출자가 명시해야 합니다.

## decode와 좌표 복원

각 feature level의 grid 좌표 `(gx, gy)`와 stride `s`에 대해 pinned YOLOX 0.3.0과 같은 식을 적용합니다.

```text
center_x = (raw_x + gx) * s
center_y = (raw_y + gy) * s
width    = exp(raw_log_w) * s
height   = exp(raw_log_h) * s
score    = objectness * max(class_probability)
```

YOLOX validation 전처리는 원본을 `ratio = min(640/original_width, 640/original_height)`로 축소하고 114 값의 canvas 왼쪽 위에 배치합니다. 중앙 정렬 padding offset은 없습니다. NMS는 공식 경로와 같이 clip 전 model 좌표에서 수행하고, 선택된 bbox를 원본 좌표로 돌릴 때 네 좌표를 `ratio`로 나눈 뒤 original image boundary로 clip합니다. clip 결과 면적이 0이면 결과에서 제외합니다.

## NMS

candidate는 confidence 내림차순, 동률이면 원래 anchor 순서로 정렬해 결과를 결정적으로 만듭니다.

- `class_agnostic_nms=true`: 공식 YOLOX demo와 동일하게 클래스와 무관하게 겹치는 낮은 confidence box를 제거
- `class_agnostic_nms=false`: 같은 클래스끼리만 제거
- IoU가 `nms_threshold`보다 큰 경우에만 억제

이 선택지는 코드 계약일 뿐 어느 정책의 정확도가 더 좋다는 결론이 아닙니다. confidence/NMS threshold와 정책 선택은 학습된 모델의 validation 실험에서 결정해야 합니다.

## 다음 adapter와의 경계

```text
Frame pixel buffer
  -> YOLOX top-left letterbox / CHW float
  -> ONNX Runtime 또는 TensorRT graph
  -> raw float[8400][9]
  -> YoloXPostprocessor
  -> Detection[]
  -> KalmanIoUTracker
```

아직 결정하지 않은 부분은 `Frame` pixel buffer 소유권, BGR/RGB 변환 책임, ONNX Runtime C++ 의존성 취득 방식, 실제 threshold 및 detector 실행 주기입니다. TensorRT/Jetson adapter도 이 postprocessor 계약을 재사용할 수 있지만 대상 장치에서 별도 검증해야 합니다.
