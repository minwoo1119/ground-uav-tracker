# 실제 영상 runtime 설정

`uav-tracker --config <path> --input <video>`는 `OpenCvFrameSource`, `OnnxYoloXDetector`, `KalmanIoUTracker`를 순서대로 연결합니다. 이 명령은 영상 파일 전용이며 camera index, RTSP와 GStreamer source는 아직 지원하지 않습니다.

## 실행

ONNX Runtime과 OpenCV를 활성화해 빌드한 뒤 다음과 같이 실행합니다.

```powershell
$env:PATH="C:\path\to\opencv\build\x64\vc16\bin;$env:PATH"
build-onnx\Release\uav-tracker.exe `
  --config configs\local-runtime.yaml `
  --input data\raw\sample.mp4 `
  --output-tracks outputs\sample-tracks.txt `
  --output-video outputs\sample-overlay.avi
```

`--output-tracks`와 `--output-video`는 모두 선택 사항입니다. 지정하면 기존 파일을 덮어쓰지 않고 각각 MOTChallenge 형식의 활성 track 행과 검출·추적 오버레이 영상(`.avi`/`.mp4`)을 기록합니다. 부모 디렉터리는 미리 만들어야 합니다. 시각화 세부 규칙은 [시각화 문서](visualization.md)를 참고합니다.

종료 요약의 `total_processing_us`는 detector와 tracker 처리 시간의 합이며 영상 decode, 파일 출력, 오버레이 렌더링과 영상 인코딩은 포함하지 않습니다. 따라서 이 값 하나를 FPS나 end-to-end latency 기준으로 해석하지 않습니다.

## 허용 설정 문법

[`configs/example.yaml`](../configs/example.yaml)은 프로젝트 전용 2단계 스키마입니다. 범용 YAML parser가 아니며 다음 문법만 허용합니다.

- 최상위 section은 `runtime`, `detection`, `tracking`만 허용
- key는 정확히 공백 두 칸으로 들여쓰기
- 빈 줄과 줄 전체가 `#`로 시작하는 comment 허용
- scalar는 `key: value`, list는 `[value, value]` 형식
- 알 수 없는 section/key, 중복, 누락, tab, 3단계 이상 중첩을 거부
- `detection.model_path`의 상대경로는 config 파일의 디렉터리를 기준으로 해석

필수 key는 다음과 같습니다.

| Section | Key | 의미 |
|---|---|---|
| `runtime` | `max_frames` | `0`이면 영상 끝까지, 양수면 해당 프레임 수까지만 처리 |
| `detection` | `model_path` | YOLOX ONNX 파일 경로 |
| `detection` | `input_width`, `input_height` | 모델의 고정 입력 크기 |
| `detection` | `strides` | YOLOX feature stride 목록 |
| `detection` | `class_names` | 학습/export와 동일한 클래스 순서 |
| `detection` | `confidence_threshold`, `nms_threshold` | 후처리 threshold |
| `detection` | `class_agnostic_nms` | class-agnostic NMS 사용 여부 |
| `detection` | `intra_op_threads` | ONNX Runtime CPU intra-op thread 수 |
| `tracking` | `iou_threshold` | 예측 track과 detection의 IoU 연관 기준 |
| `tracking` | `min_hits_to_confirm`, `max_missed_frames` | track 상태 전이 기준 |
| `tracking` | `default_dt_seconds` | timestamp 차분이 없을 때 사용할 양의 시간 간격 |
| `tracking` | `*_process_noise`, `*_measurement_noise` | Kalman filter의 0 이상 noise 값 |

예제의 threshold와 tracker 값은 실행 가능한 reference 값일 뿐 최적값이나 합격 기준이 아닙니다. 학습된 모델의 validation/test 평가로 별도 조정해야 합니다.

## 오류 경계

- config, video 또는 model 파일이 없으면 해당 경로를 포함한 오류를 반환합니다.
- OpenCV/ONNX Runtime이 빠진 빌드는 필요한 CMake option을 명시한 오류로 종료합니다.
- model의 input/output type과 shape가 config에서 계산한 계약과 다르면 session 생성 시 거부합니다.
- 출력 파일이 이미 있거나 열 수 없으면 기존 파일을 변경하지 않고 실패합니다.

로컬 model, 영상, 출력 결과와 dependency archive는 Git 제외 대상입니다.
