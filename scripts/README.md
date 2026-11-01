# 오프라인 데이터셋 전처리 및 변환 도구

이 디렉터리는 DroneDetectionThesis(Halmstad University) 등 항공체 관측 데이터셋의 MATLAB `.mat` 라벨을 디코딩하고, 4클래스 기준 마스터 JSON 및 파생 어노테이션(YOLO, MOT Challenge, 데이터 분할)으로 변환하는 오프라인 파이썬 도구를 제공합니다.

> [!NOTE]
> **외부 의존성 제로(Zero Dependencies)**
> 이 도구들은 파이썬 표준 라이브러리(`struct`, `zlib`, `json`, `pathlib`, `zipfile`, `xml.etree.ElementTree`)만을 사용하여 구현되었습니다. MATLAB 소프트웨어나 `scipy`, `pandas`, `openpyxl` 등의 추가 패키지 설치 없이 표준 Python 3.8+ 환경에서 즉시 실행 가능합니다.

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
| `test_converter.py` | 모의 바이너리 MAT5 버퍼 생성 및 파서·변환기 전수 단위 테스트 |

---

## 3. 사용 방법 (CLI Guide)

### 3.1 단위 테스트 실행

```sh
python scripts/test_converter.py -v
```

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
  --image-width 1920 --image-height 1080 \
  --splits-dir data/processed/splits
```

#### D. MOT Challenge 형식 익스포트 (`export-mot`)
```sh
python scripts/convert_dataset.py export-mot \
  --master data/processed/dataset_master.json \
  --output-dir data/processed/mot
```

---

## 4. 마스터 JSON 스키마 구조

```json
{
  "schema_version": "1.0",
  "created_at": "2026-11-01T07:30:00Z",
  "classes": {
    "0": "AIRPLANE",
    "1": "BIRD",
    "2": "DRONE",
    "3": "HELICOPTER"
  },
  "statistics": {
    "total_sequences": 285,
    "total_frames": 88000,
    "class_box_counts": {
      "AIRPLANE": 18200,
      "BIRD": 15400,
      "DRONE": 36800,
      "HELICOPTER": 17600
    }
  },
  "sequences": {
    "V_DRONE_001": {
      "sequence_name": "V_DRONE_001",
      "sensor": "V",
      "primary_class": "DRONE",
      "number": 1,
      "distance_bin": "MEDIUM",
      "drone_type": "F450",
      "source": "author",
      "is_multi_drone": false,
      "total_frames": 301,
      "fps": 30.0,
      "image_width": 1920,
      "image_height": 1080,
      "frames": [
        {
          "frame_index": 0,
          "timestamp_sec": 0.0,
          "annotations": [
            {
              "class_id": 2,
              "class_name": "DRONE",
              "track_id": 1,
              "bbox_xywh": [124.93, 22.94, 75.85, 39.2]
            }
          ]
        }
      ]
    }
  }
}
```
