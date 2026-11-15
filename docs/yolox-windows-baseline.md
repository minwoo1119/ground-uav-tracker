# YOLOX-s 640 Windows baseline

이 문서는 Issue #19에서 승인된 4클래스 YOLOX-s 검출 baseline을 Windows 개발 PC에서 재현하는 절차입니다. Jetson Orin Nano, TensorRT engine 생성 및 실제 배포 검증은 포함하지 않습니다.

## 고정 입력과 출처

- 데이터: DroneDetectionThesis RGB 285개 영상, CC0-1.0
- 데이터 소스 commit: `442d0c6750708cc221636f7a1fd31cbd8cb0aeb8`
- 모델 코드: YOLOX 0.3.0, source commit `6ddff4824372906469a7fae2dc3206c7aa4bbaee`
- 초기 가중치: YOLOX `0.1.1rc0` release의 `yolox_s.pth`
- 확인한 가중치 크기: `72,089,125` bytes
- 확인한 SHA-256: `F55DED7181E1B0C13285C56E7790B8F0E8F8DB590FE4EDB37F0B7F345C913A30`
- 클래스: `AIRPLANE`, `BIRD`, `DRONE`, `HELICOPTER`
- 입력: 정적 `1×3×640×640`, full-frame

원본 영상, 변환 이미지, 가중치, checkpoint 및 ONNX 파일은 Git에 커밋하지 않습니다. 프로젝트 자체 라이선스는 별도 사용자 결정 사항이며, 외부 라이선스와 동일한 의미가 아닙니다.

## 환경 구성

Python 3.12 가상환경을 사용합니다. PyTorch는 [공식 이전 버전 설치표](https://pytorch.org/get-started/previous-versions/)의 조합 중 2.9.1/CUDA 12.8을 Windows 학습 기준으로 고정했습니다. NVIDIA driver가 CUDA 12.8 wheel을 지원하는지 먼저 확인해야 합니다. GPU가 없는 PC에서는 CPU requirements로 구조와 ONNX 경로만 검증합니다. YOLOX는 source 요구사항을 다시 해석하지 않도록 pinned package를 먼저 설치하고 `--no-deps`로 설치합니다.

```powershell
py -3.12 -m venv .venv-train
.\.venv-train\Scripts\python.exe -m pip install -r requirements\pytorch-cu128.txt
.\.venv-train\Scripts\python.exe -m pip install -r requirements\training-yolox.txt
.\.venv-train\Scripts\python.exe -m pip install --no-deps -r requirements\yolox-source.txt
```

CPU-only 검증 환경에서는 첫 설치 명령의 파일을 `requirements\pytorch-cpu.txt`로 바꿉니다. 설치 뒤 `torch.cuda.is_available()`가 `True`인지 확인하지 않고 학습을 시작하지 않습니다.

YOLOX source는 `onnx-simplifier`를 package dependency로 선언하지만 이 baseline의 exporter는 ONNX checker와 ONNX Runtime을 직접 사용하므로 설치 목록에서 의도적으로 제외했습니다. 따라서 `pip check`는 이 선택적 metadata dependency를 누락으로 표시할 수 있습니다. `opencv-python`을 포함한 학습·평가 runtime 의존성은 별도로 고정돼 있습니다.

## 데이터 취득과 COCO 변환

다운로더는 고정 commit의 Git tree에서 정확히 285개 RGB MP4를 요구하고, 파일 크기와 Git blob SHA-1을 검증한 뒤 원자적으로 게시합니다.

```powershell
.\.venv-train\Scripts\python.exe scripts\download_rgb_videos.py `
  --output-dir data\raw\Video_V --workers 8

.\.venv-train\Scripts\python.exe scripts\export_coco.py `
  --master data\processed\video_v\dataset_master.json `
  --splits data\processed\video_v\splits\splits.json `
  --videos-dir data\raw\Video_V `
  --output-dir data\processed\video_v\coco
```

COCO category ID는 표준 관례에 맞게 1부터 시작합니다. `1:AIRPLANE`, `2:BIRD`, `3:DRONE`, `4:HELICOPTER`이며 각 category의 `project_class_id`가 프로젝트의 0 기반 ID를 보존합니다. 이미지 경로는 `train2017`, `val2017`, `test2017`로 분리됩니다.

Issue #19 구현 시 전체 변환 결과는 train/val/test 순으로 이미지 `62,044 / 14,153 / 13,036`, annotation `65,857 / 15,076 / 15,312`, 시퀀스 `198 / 45 / 42`였습니다. split 간 시퀀스 중복과 bbox clipping은 없었습니다.

## 학습

공식 YOLOX trainer는 CUDA를 요구합니다. CUDA GPU가 있는 Windows 호스트에서만 아래 명령을 실행합니다.

```powershell
.\.venv-train\Scripts\python.exe scripts\train_yolox.py `
  --data-dir data\processed\video_v\coco `
  --pretrained models\yolox_s.pth `
  --output-dir outputs\yolox --epochs 50 --batch-size 8 --workers 4
```

현재 검증 PC에는 CUDA GPU가 없어 전체 학습과 detection 정확도 평가는 실행하지 않았습니다. CPU에서는 `scripts/yolox_smoke.py`로 모델 생성, 호환 가중치 로드, forward/backward 한 단계만 확인할 수 있습니다. 그 결과 checkpoint에는 `untrained_cpu_smoke_checkpoint`가 기록되며 학습 모델로 사용하면 안 됩니다.

학습이 끝난 뒤에는 사용하지 않은 test split을 공식 COCO AP와 클래스별 AP/AR로 평가합니다.

```powershell
.\.venv-train\Scripts\python.exe scripts\evaluate_yolox.py `
  --data-dir data\processed\video_v\coco `
  --checkpoint outputs\yolox\yolox_s_uav_640\best_ckpt.pth `
  --split test --batch-size 8 `
  --output outputs\yolox\test_metrics.json
```

평가 스크립트도 공식 YOLOX evaluator의 구현 제약 때문에 CUDA를 요구합니다. test metric으로 모델이나 threshold를 반복 선택하면 test leakage가 되므로 조정은 val split에서만 수행합니다.

## ONNX 검증

```powershell
.\.venv-train\Scripts\python.exe scripts\export_yolox_onnx.py `
  --checkpoint outputs\yolox\<experiment>\best_ckpt.pth `
  --data-dir data\processed\video_v\coco `
  --output artifacts\yolox_s_uav.onnx
```

exporter는 ONNX checker와 ONNX Runtime CPU 추론을 실행합니다. 4클래스 raw output의 기대 shape은 `[1, 8400, 9]`입니다. 후처리(NMS)는 ONNX 밖에 있고, TensorRT parser·정밀도·성능은 Jetson 단계에서 별도로 검증해야 합니다.

## 검증 경계

- 검증됨: 데이터 다운로드 무결성, 전체 COCO 변환, 4클래스 모델 생성, 공식 가중치의 호환 계층 로드, CPU forward/backward, ONNX export/check/runtime
- 미검증: CUDA 학습, AP/Recall, 새 오탐률, 실제 영상 detector→tracker 연결, TensorRT engine, Jetson FPS/latency/power
