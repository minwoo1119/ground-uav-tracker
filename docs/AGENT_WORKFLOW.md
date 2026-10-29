# AI Coding Agent 작업 절차

이 문서는 모든 신규 개발 요청에서 AI Coding Agent가 따라야 하는 실행 절차입니다. 프로젝트 소유자가 목적, 설계 결정, 코드 흐름과 검증 결과를 이해한 뒤 직접 Merge하는 것을 목표로 합니다.

`AGENTS.md`가 최상위 규칙이며, 사람 중심의 리뷰 원칙과 예시는 `docs/DEVELOPMENT_WORKFLOW.md`를 함께 참고합니다.

## 1. 세션 시작 시 필수 확인

Agent는 새로운 세션과 새로운 작업 요청마다 다음을 먼저 수행합니다.

```sh
cat AGENTS.md
cat docs/AGENT_WORKFLOW.md
git status --short --branch
git branch --all --verbose
git remote -v
gh --version
gh auth status
gh repo view --json nameWithOwner,url,defaultBranchRef,viewerPermission
gh issue list --state all --limit 100
gh pr list --state all --limit 100
```

확인할 내용:

- 작업 트리에 사용자의 미커밋 변경이 있는가?
- 현재 브랜치가 기본 브랜치인가, 기존 작업 브랜치인가?
- 원격 기본 브랜치와 upstream이 정상인가?
- GitHub CLI 인증과 저장소 권한이 유효한가?
- 동일하거나 겹치는 Issue/PR가 이미 존재하는가?

미커밋 변경이나 브랜치 충돌이 있으면 파일을 덮어쓰거나 이동하지 않고 사용자에게 판단을 요청합니다. 인증이 없거나 권한이 부족하면 우회 토큰, 다른 계정, GitHub 웹 자동화를 사용하지 않고 `gh auth login` 또는 권한 확인을 안내합니다.

## 2. 작업 요청 분석

요청을 다음 항목으로 분해합니다.

1. 해결하려는 문제와 사용자 가치
2. 관찰 가능한 목표 동작
3. 포함 범위와 제외 범위
4. 영향받는 모듈과 데이터 흐름
5. 자동·수동 검증 방법
6. 사용자 승인이 필요한 설계 결정
7. 외부 리소스, 라이선스 또는 원격 변경 여부

특정 모델, 데이터, 카메라, 추적 알고리즘, Jetson 스택, 성능 기준처럼 `AGENTS.md`에서 승인을 요구하는 항목은 추천안만 제시하고 구현 전에 질문합니다.

## 3. 기존 Issue와 PR 검색

제목 검색과 열린 작업 목록을 함께 확인합니다.

```sh
gh issue list --state all --search "<핵심 검색어>"
gh pr list --state all --search "<핵심 검색어>"
gh issue view <번호>
gh pr view <번호>
```

적합한 Issue가 있으면 새 Issue를 만들지 않고 해당 번호를 사용합니다. 이미 같은 범위의 열린 PR가 있으면 중복 브랜치를 만들지 않고 사용자에게 기존 PR을 이어갈지 확인합니다.

## 4. Issue 생성

적합한 Issue가 없으면 구현 전에 `gh issue create`로 생성합니다. 저장소의 Feature, Experiment, Bug Issue Form 중 작업 성격에 맞는 구조를 사용합니다.

Issue 본문은 최소한 다음을 포함합니다.

```markdown
## Background
작업 배경과 현재 문제

## Objective
완료 후 기대하는 결과

## Scope
포함 범위와 제외 범위

## Implementation Plan
예상 파일, 호출/데이터 흐름, 설계 단계

## Acceptance Criteria
- [ ] 검증 가능한 완료 조건

## Test Plan
실행할 명령, 자동/수동 테스트와 오류 경로

## Open Questions
사용자 결정이 필요한 항목과 대안
```

아키텍처나 범위에 중대한 변경이 필요하면 Issue에 최소 두 가지 대안, 장단점과 추천안을 기록하고 사용자 승인 전에는 해당 결정을 코드로 고정하지 않습니다.

## 5. Issue 전용 브랜치

브랜치 이름은 다음 형식을 사용합니다.

```text
issue/<번호>-<간단한-kebab-case-작업명>
```

예:

```text
issue/12-dataset-review
issue/27-timestamp-validation
```

가능하면 Issue와 브랜치를 GitHub에서 연결합니다.

```sh
gh issue develop <번호> --name issue/<번호>-<작업명> --checkout
```

명령을 사용할 수 없으면 최신 원격 기본 브랜치에서 로컬 브랜치를 만들고, PR에서 Issue를 명시적으로 연결합니다. 하나의 Issue는 원칙적으로 하나의 브랜치와 하나의 PR을 사용합니다.

기본 브랜치에서는 파일을 수정하거나 직접 Push하지 않습니다. 새 원격 저장소의 최초 기본 브랜치 게시처럼 부트스트랩이 필요하면 사용자에게 예외 승인을 받습니다.

## 6. 구현 계획 기록

코드 작성 전에 Issue 댓글 또는 본문에 계획을 기록합니다.

- 현재 코드와 진입점
- 변경할 파일과 각 파일의 책임
- 입력 데이터, 단위, 소유권과 유효 기간
- 내부 검증·변환·상태 변화
- 출력, 부작용과 오류 전파
- 다음 모듈과의 연결 및 timestamp/frame ID 유지 방식
- 고려한 대안과 선택 이유
- 테스트와 실패 경로
- 범위 밖 항목과 미검증 항목

계획과 실제 구현이 달라지면 PR에서 차이를 설명하고, 범위가 실질적으로 커지면 Issue를 분리하거나 사용자 재승인을 받습니다.

## 7. 구현과 테스트

기존 사용자 변경을 보존하고 Issue 범위에 포함된 파일만 수정합니다. 완료 기준마다 대응하는 테스트를 실행합니다.

기본 C++ 검증 예:

```sh
cmake -S . -B build -DUAV_ENABLE_TENSORRT=OFF -DUAV_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

문서와 GitHub Actions 작업은 해당 형식, 스크립트 정상/실패 fixture, 변경 범위와 실제 원격 check를 검증합니다. 실행하지 못한 테스트는 통과로 표시하지 않고 사유와 재현 방법을 기록합니다.

## 8. 검토와 한글 커밋

커밋 전에 다음을 확인합니다.

```sh
git status --short
git diff --check
git diff --stat
git diff
```

Issue와 관계없는 파일은 Stage하지 않습니다. 생성 데이터, 모델, 가중치, 엔진, 비밀 설정과 개인 경로가 없는지 확인합니다.

작업 내용을 반영하는 한글 커밋을 생성합니다.

```sh
git add <변경된 파일만>
git commit -m "feat:한글 작업 내용"
```

작업 유형에 따라 `fix:`, `docs:`, `test:`, `chore:` 등 더 적합한 접두사를 사용할 수 있습니다.

## 9. Push와 Draft PR

작업 브랜치만 Push합니다.

```sh
git push -u origin issue/<번호>-<작업명>
gh pr create --draft --base <기본 브랜치> --head issue/<번호>-<작업명>
```

PR 본문에는 반드시 다음을 포함합니다.

- `Closes #<Issue 번호>`
- 작업 목적과 변경 전후 비교
- 변경 파일과 각 파일의 역할
- 입력 → 내부 처리 → 출력 → 다른 모듈의 데이터 흐름
- 핵심 구현 원리
- 설계 결정 근거와 고려한 대안
- 테스트 명령, 환경과 실제 결과
- 미해결 문제, 미검증 항목과 제한 사항
- 사용자가 직접 이해해야 할 핵심 코드·설정
- 이해 확인 질문 2~3개와 해설

Draft PR URL과 테스트 결과를 사용자에게 보고합니다. 사용자의 명시적인 요청 없이 Ready for Review로 전환하거나 Merge하지 않습니다.

## 10. PR Guard

`.github/workflows/pr-guard.yml`은 PR 생성·수정 시 다음 최소 형식을 검사합니다.

- `issue/<번호>-<작업명>` 브랜치 규칙
- `Closes #<번호>`, `Fixes #<번호>` 또는 `Resolves #<번호>` 연결
- 연결 번호가 실제 Issue인지 여부
- PR 템플릿의 필수 섹션 존재 여부

이 검사는 문자열과 GitHub 메타데이터의 존재만 확인합니다. 다음은 보증하지 못합니다.

- 설명과 코드의 기술적 정확성
- 테스트 결과의 진실성이나 충분성
- 설계 대안의 품질
- 사용자가 코드를 실제로 이해했는지 여부

따라서 PR Guard 통과는 사용자 리뷰와 이해 확인을 대체하지 않습니다.

## 11. 사용자 리뷰와 Merge

사용자는 Draft PR에서 변경 목적, 파일별 역할, 코드 흐름, 설계 대안, 테스트와 제약을 직접 검토합니다. 이해 확인 질문은 암기 평가가 아니라 설명의 빈틈을 찾기 위해 사용합니다.

Agent는 리뷰 의견에 따라 같은 Issue와 브랜치에서 수정·재검증·추가 커밋·Push를 수행할 수 있습니다. 사용자 요청 전에는 Draft를 해제하거나 Merge하지 않습니다.

## 12. 인증·권한·실패 처리

- `gh auth status` 실패: 작업을 중단하고 `gh auth login` 안내
- Issue/Push/PR 권한 부족: 오류를 그대로 보고하고 우회하지 않음
- 원격 기본 브랜치 없음: 사용자 승인 후 초기 브랜치 게시
- 작업 트리 dirty: 사용자 변경을 식별하고 충돌 시 판단 요청
- 기존 Issue/PR 발견: 중복 생성하지 않고 이어갈지 확인
- GitHub Actions 실패: 로그를 확인하고 같은 브랜치에서 수정

## 13. 브랜치 보호 권장 설정

이 저장소는 공개 저장소이고 현재 사용자는 관리자 권한을 보유하므로 GitHub Free 공개 저장소에서도 Branch Ruleset 또는 classic branch protection을 사용할 수 있습니다. 현재 `main`에는 ruleset이나 branch protection이 설정되어 있지 않습니다.

프로젝트 소유자가 GitHub 웹에서 직접 설정합니다.

1. **Settings → Rules → Rulesets → New branch ruleset**으로 이동
2. 대상 브랜치에 기본 브랜치 `main` 추가
3. **Require a pull request before merging** 활성화
4. 1인 프로젝트이므로 required approving reviews는 기본 `0` 유지
5. **Require status checks to pass** 활성화
6. 이 PR이 한 번 실행된 뒤 `PR Guard / Validate PR metadata` check 선택
7. **Block force pushes**와 **Restrict deletions** 활성화
8. Ruleset 상태와 bypass 권한을 검토한 뒤 활성화

Agent는 사용자 승인 없이 원격 ruleset이나 branch protection을 변경하지 않습니다.

## 14. 새 개발 요청 방법

사용자는 구현 목표와 제약을 자연어로 요청하면 됩니다. 예:

> 파일 기반 FrameSource 인터페이스 확장을 Issue 기반 워크플로로 진행해줘. OpenCV 도입 여부는 구현 전에 선택지를 설명하고 질문해줘.

Agent는 자동으로 기존 Issue/PR 검색, 필요한 Issue 생성, 전용 브랜치, 계획 기록, 구현, 테스트, 한글 커밋, Push와 Draft PR까지 수행합니다. 사용자 결정이 필요한 항목에서는 구현을 멈추고 질문합니다.
