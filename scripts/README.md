# 오프라인 데이터셋 전처리 및 변환 도구

이 디렉터리는 DroneDetectionThesis(Halmstad University) 등 항공체 관측 데이터셋의 MATLAB `.mat` 라벨을 디코딩하고, 4클래스 기준 마스터 JSON 및 파생 어노테이션(YOLO, MOT Challenge, 데이터 분할)으로 변환하는 오프라인 파이썬 도구를 제공합니다.

> [!NOTE]
> **외부 의존성 제로(Zero Dependencies)**
> 이 도구들은 파이썬 표준 라이브러리(`struct`, `zlib`, `json`, `pathlib`, `zipfile`, `xml.etree.ElementTree`)만을 사용하여 구현되었습니다. MATLAB 소프트웨어나 `scipy`, `pandas`, `openpyxl` 등의 추가 패키지 설치 없이 실행됩니다.

---

## 1. 지원 클래스 구성 (4-Class Mapping)

사용자 합의에 따라 지상 대공 관측의 4대 비행체 클래스를 지원합니다.

| Class ID | Class Name | 비고 |
|---|---|---|
| `0` | **AIRPLANE** | 고정익 항공기 |
| `1` | **BIRD** | 조류 (드론 오탐 방지용) |
| `2` | **DRONE** | 소형 무인기 / 쿼드콥터 |
| `3` | **HELICOPTER** | 회전익 헬리콥터 |

---

## 2. 제공 스크립트 목록

| 파일 | 주요 역할 |
|---|---|
| `mcos_parser.py` | MATLAB v5 `.mat` 내부 MCOS `groundTruth` 객체를 파싱하여 5열 타임테이블 스트라이드와 가변 BBox `(N x 4)`를 디코딩 |
| `metadata_parser.py` | `Video_dataset_description.xlsx`를 순수 파이썬으로 파싱하여 시퀀스별 메타데이터(거리 Bin, 기종, 촬영 출처) 추출 |
| `convert_dataset.py` | 마스터 JSON 중간 표현 생성, YOLO txt 변환, MOT Challenge 변환, 계층화 Train/Val/Test 분할 생성 파이프라인 |
| `video_probe.py` | 디코딩 없이 `.mp4` 헤더에서 코덱·해상도·프레임 수·FPS를 읽는 도구 |
| `test_converter.py` | 모의 MAT5/MP4 버퍼 기반 단위 테스트 + 실제 라벨 파일 회귀 테스트(선택) |

---

## 3. 사용 방법 (CLI Guide)

### 3.1 단위 테스트 실행

```sh
python scripts/test_converter.py -v

# 원본 *_LABELS.mat가 있는 로컬 폴더로 실제 파일 회귀 테스트까지 실행
UAV_DATASET_LABELS_DIR=data/raw/Video_V python scripts/test_converter.py -v
```

환경 변수가 없거나 해당 파일이 없으면 실제 파일 테스트는 skip된다. 검증 환경은 Python 3.14(`uv run python`)이며 다른 버전은 미확인이다.

### 3.2 개별 `.mat` 라벨 파일 검사

단일 MATLAB 라벨 파일의 프레임 수와 클래스별 BBox 개수를 빠르게 확인할 수 있습니다.

```sh
# 요약 출력
python scripts/mcos_parser.py data/raw/Video_V/V_DRONE_001_LABELS.mat

# 전체 JSON 덤프
python scripts/mcos_parser.py data/raw/Video_V/V_DRONE_001_LABELS.mat --json
```

### 3.3 전체 파이프라인 일괄 실행 (`all`)

라벨 디렉터리로부터 마스터 JSON, 계층화 분할, YOLO 라벨, MOT 시퀀스를 한 번에 생성합니다.

```sh
python scripts/convert_dataset.py all \
  --labels-dir data/raw/Video_V \
  --metadata-file data/raw/Video_dataset_description.xlsx \
  --videos-dir data/raw/Video_V \
  --output-dir data/processed
```

생성되는 디렉터리 구조:
```text
data/processed/
├── dataset_master.json       # 통합 마스터 어노테이션 JSON
├── splits/
│   ├── splits.json           # 분할 요약 메타데이터
│   ├── train_videos.txt      # 학습용 시퀀스 목록 (70%)
│   ├── val_videos.txt        # 검증용 시퀀스 목록 (15%)
│   └── test_videos.txt       # 평가용 시퀀스 목록 (15%)
├── yolo/
│   ├── data.yaml             # YOLO 학습 설정 파일 (nc: 4, names)
│   └── labels/               # 정규화된 frame별 txt 어노테이션
└── mot/
    └── <시퀀스명>/
        ├── seqinfo.ini       # MOT Challenge 시퀀스 메타데이터
        └── gt/
            └── gt.txt        # <frame>,<id>,<x>,<y>,<w>,<h>,1,-1,-1,-1
```

### 3.4 세부 단계별 실행

#### A. 마스터 JSON 생성 (`build-master`)
```sh
python scripts/convert_dataset.py build-master \
  --labels-dir data/raw/Video_V \
  --metadata-file data/raw/Video_dataset_description.xlsx \
  --output data/processed/dataset_master.json
```

#### B. 계층화 분할 생성 (`generate-splits`)
클래스 4종과 거리 구간(CLOSE/MEDIUM/DISTANT)을 균등 배분하여 70 : 15 : 15 분할을 생성합니다. 다중 드론 영상은 자동으로 test 세트에 배정됩니다.
```sh
python scripts/convert_dataset.py generate-splits \
  --master data/processed/dataset_master.json \
  --output-dir data/processed/splits \
  --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 --seed 42
```

#### C. YOLO 형식 익스포트 (`export-yolo`)
```sh
python scripts/convert_dataset.py export-yolo \
  --master data/processed/dataset_master.json \
  --output-dir data/processed/yolo \
  --splits-dir data/processed/splits
```
영상 크기는 마스터 JSON의 시퀀스별 값을 사용한다. 모든 시퀀스에 같은 크기를 강제하려면 `--image-width W --image-height H`를 추가한다.

#### D. MOT Challenge 형식 익스포트 (`export-mot`)
```sh
python scripts/convert_dataset.py export-mot \
  --master data/processed/dataset_master.json \
  --output-dir data/processed/mot
```

---

## 4. 마스터 JSON 스키마 구조

아래 `statistics` 값은 형식 설명용 자리표시자(`<int>`)이며 실제 통계가 아니다. `V_DRONE_001` 항목은 원본 라벨/영상 헤더에서 실제로 확인한 값이다.

```json
{
  "schema_version": "1.0",
  "created_at": "<ISO-8601 UTC>",
  "classes": {"0": "AIRPLANE", "1": "BIRD", "2": "DRONE", "3": "HELICOPTER"},
  "statistics": {
    "total_sequences": "<int>",
    "total_frames": "<int>",
    "class_box_counts": {"AIRPLANE": "<int>", "BIRD": "<int>", "DRONE": "<int>", "HELICOPTER": "<int>"}
  },
  "warnings": ["<영상 프레임 수와 라벨 프레임 수 불일치 등>"],
  "sequences": {
    "V_DRONE_001": {
      "sequence_name": "V_DRONE_001",
      "sensor": "V",
      "primary_class": "DRONE",
      "number": 1,
      "distance_bin": "CLOSE",
      "drone_type": "F450",
      "source": "author",
      "is_multi_drone": false,
      "label_file": "V_DRONE_001_LABELS.mat",
      "total_frames": 301,
      "fps": 30.0,
      "image_width": 640,
      "image_height": 512,
      "video_frame_count": 301,
      "video_info_source": "mp4_header",
      "class_box_counts": {"AIRPLANE": 0, "BIRD": 0, "DRONE": 301, "HELICOPTER": 0},
      "frames": [
        {
          "frame_index": 0,
          "timestamp_sec": 0.0,
          "annotations": [
            {"class_id": 2, "class_name": "DRONE", "track_id": 1, "bbox_xywh": [124.93, 22.94, 75.85, 39.2]}
          ]
        }
      ]
    }
  }
}
```

### 4.1 영상 크기·FPS 결정 규칙

| `video_info_source` | 의미 |
|---|---|
| `mp4_header` | `--videos-dir`의 같은 이름 `.mp4` 헤더(`stsd`, `mdhd`, `stts`)에서 읽은 값 |
| `sensor_default_unverified` | 영상이 없어 센서별 기본값(V 640x512, IR 320x256, 30 fps)을 쓴 값. 기본값은 `V_DRONE_001`, `IR_BIRD_001` 두 영상에서만 확인됨 |

YOLO 정규화는 시퀀스별 `image_width/height`를 사용하며, `--image-width/--image-height`는 명시적 덮어쓰기용이다.

### 4.2 Track ID에 대한 주의

원본 라벨에는 객체 ID가 없다. `track_id`는 클래스별 탐욕적 IoU(> 0.05) 연결로 **파생한 값**이며, 해당 클래스 박스가 없는 프레임이 끼면 새 ID가 시작된다. 가까이 교차하는 다수 객체(예: 새 무리)에서는 ID 교환이 생길 수 있어 추적 평가 GT로 쓰기 전 검수가 필요하다.

### 4.3 MATLAB 행렬 저장 순서

한 프레임에 같은 클래스 객체가 N개면 라벨은 N x 4 double 행렬이며 MATLAB은 이를 **열 우선**(`x1..xN, y1..yN, w1..wN, h1..hN`)으로 저장한다. 행 우선으로 읽으면 2개 이상 객체 프레임의 좌표가 뒤섞인다(PR #6 초기 구현의 결함, 실제 파일 검증으로 발견·수정).
