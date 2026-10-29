# 데이터 보관 정책

실제 영상, 이미지, 라벨, 변환 결과 및 데이터셋 압축 파일은 이 디렉터리 아래에 로컬로 보관하고 Git에 추가하지 않습니다. 라이선스와 재배포 조건을 확인하기 전에는 외부 데이터를 저장소에 커밋하지 않습니다.

다음은 검토 후보 링크이며, 다운로드하거나 구조를 검증하지 않았습니다.

- [DroneDetectionThesis](https://github.com/DroneDetectionThesis/Drone-detection-dataset)
- [Anti-UAV](https://github.com/ZhaoJ9014/Anti-UAV)
- [Drone-vs-Bird](https://github.com/wosdetc/challenge)

각 후보가 지상 고정 카메라 관측 목표에 적합한지, 라이선스가 허용되는지, 라벨 구조가 무엇인지 별도로 확인해야 합니다. 특히 MATLAB `.mat` 라벨은 변환 가능 여부와 필드 구조를 실제 파일로 검증하기 전까지 지원된다고 간주하지 않습니다.

권장 로컬 배치 예시는 `data/raw/<dataset>`, `data/interim/<dataset>`, `data/processed/<dataset>`이지만 확정 스키마는 아닙니다.
