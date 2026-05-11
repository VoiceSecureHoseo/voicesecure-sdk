# 팀원 Onboarding 가이드

VoiceSecure SDK 프로젝트에 오신 것을 환영합니다. 이 문서는 처음 합류한 팀원이 본인 컴퓨터에서 작업을 시작하기까지 필요한 모든 단계를 다룹니다. 순서대로 따라하면 됩니다.

---

## 0. 사전 준비물

다음이 본인 컴퓨터에 설치돼 있어야 합니다.

| 도구 | 확인 명령어 | 최소 버전 |
|---|---|---|
| **Git** | `git --version` | 2.30+ |
| **Python** | `python --version` | 3.10+ |
| **Git Bash** (Windows만) | 시작 메뉴에서 "Git Bash" 검색 | Git 설치 시 자동 |

설치 안 됐으면:
- Git: https://git-scm.com/download
- Python: https://www.python.org/downloads/ (설치 시 **"Add Python to PATH"** 반드시 체크)

설치 후에는 Git Bash를 **닫았다 다시 열어야** PATH가 적용됩니다.

---

## 1. GitHub 계정 + 조직 가입

### 1.1 GitHub 계정 준비

- 본인 GitHub 계정이 없으면 https://github.com/signup 에서 가입
- 호서대 이메일 또는 본인 메인 이메일 사용 권장

### 1.2 조직 초대 수락

조장(dPsk)이 `VoiceSecureHoseo` 조직에 초대를 보냈을 것입니다.

- 이메일 받은편지함 또는 GitHub 알림 확인
- **Accept invitation** 버튼 클릭
- 수락 후 https://github.com/VoiceSecureHoseo 페이지에 본인이 member로 표시되는지 확인

---

## 2. SSH 키 셋업 (인증)

토큰/비밀번호 없이 안전하게 GitHub와 통신하기 위해 SSH 키를 사용합니다. **한 번만 설정하면 영구적으로 동작합니다.**

### 2.1 SSH 키 생성

Git Bash에서:

```bash
ssh-keygen -t ed25519 -C "본인이메일@example.com"
```

세 가지 질문이 나오면 **모두 그냥 엔터** (기본값 사용, 비밀번호 빈 칸):

```
Enter file in which to save the key: [엔터]
Enter passphrase (empty for no passphrase): [엔터]
Enter same passphrase again: [엔터]
```

성공하면 다음 두 파일이 생깁니다:
- `~/.ssh/id_ed25519` (비밀키 — 절대 공유 금지)
- `~/.ssh/id_ed25519.pub` (공개키 — GitHub에 등록할 것)

### 2.2 공개키 복사

```bash
cat ~/.ssh/id_ed25519.pub
```

한 줄로 출력되는 `ssh-ed25519 AAAA...` 로 시작하는 전체 내용을 마우스로 드래그해서 복사합니다 (Git Bash는 드래그하면 자동 복사).

### 2.3 GitHub에 공개키 등록

브라우저에서:

1. github.com 로그인
2. 우상단 본인 프로필 → **Settings**
3. 좌측 메뉴 **SSH and GPG keys**
4. 우상단 초록색 **New SSH key**
5. 입력:
   - **Title**: `<본인이름> laptop` (예: `한승규 laptop`)
   - **Key type**: `Authentication Key`
   - **Key**: 위에서 복사한 `ssh-ed25519 AAAA...` 전체 붙여넣기
6. **Add SSH key** 클릭

### 2.4 연결 테스트

Git Bash에서:

```bash
ssh -T git@github.com
```

첫 시도 시 이런 메시지가 나옵니다:

```
The authenticity of host 'github.com (...)' can't be established.
ED25519 key fingerprint is SHA256:...
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

`yes` 입력 후 엔터.

성공하면:

```
Hi <본인username>! You've successfully authenticated, but GitHub does not provide shell access.
```

`Hi <username>!` 메시지가 나오면 SSH 인증 설정 완료입니다. 앞으로 push/pull 시 비밀번호나 토큰 입력 안 합니다.

---

## 3. Repository Clone

### 3.1 작업 폴더 만들기

본인이 원하는 위치에 작업 폴더를 만듭니다. 추천 위치:

- Windows: `C:\Users\<본인>\Desktop\voiceSecure`
- macOS/Linux: `~/projects`

Git Bash에서:

```bash
cd ~/Desktop
mkdir voiceSecure
cd voiceSecure
```

### 3.2 Clone

```bash
git clone git@github.com:VoiceSecureHoseo/voicesecure-sdk.git
cd voicesecure-sdk
```

⚠️ **주의**: URL이 `https://`가 아니라 `git@github.com:` 형식이어야 합니다. SSH 인증을 사용하기 위함입니다.

성공하면:

```
Cloning into 'voicesecure-sdk'...
remote: Enumerating objects: ..., done.
...
Resolving deltas: 100%, done.
```

### 3.3 확인

```bash
ls -la
```

다음 폴더와 파일이 보여야 합니다:

```
.git/  .github/  .gitignore  .pre-commit-config.yaml
README.md  configs/  docs/  pyproject.toml  scripts/  src/  tests/
```

---

## 4. 개발 환경 셋업

### 4.1 가상환경 생성

```bash
python -m venv .venv
```

(`python` 명령어가 안 되면 `python3` 또는 `py`로 시도)

### 4.2 가상환경 활성화

| 환경 | 명령어 |
|---|---|
| Windows + Git Bash | `source .venv/Scripts/activate` |
| Windows + cmd | `.venv\Scripts\activate.bat` |
| Windows + PowerShell | `.venv\Scripts\Activate.ps1` |
| macOS/Linux | `source .venv/bin/activate` |

성공하면 프롬프트 앞에 `(.venv)` 표시가 붙습니다:

```
(.venv) user@laptop ~/Desktop/voiceSecure/voicesecure-sdk
$
```

⚠️ **앞으로 작업할 때마다 가상환경을 활성화해야 합니다.** Git Bash 새로 열 때마다 `cd` 후 `source .venv/Scripts/activate`.

### 4.3 의존성 설치

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

5~15분 정도 걸립니다. PyTorch 같은 큰 라이브러리를 다운로드하기 때문입니다. 끝까지 기다리세요.

### 4.4 pre-commit hooks 설치

```bash
pre-commit install
```

결과:

```
pre-commit installed at .git/hooks/pre-commit
```

⚠️ **이 단계를 빠뜨리면 코드 포맷 검사가 자동으로 안 돌아갑니다. 반드시 실행하세요.**

### 4.5 동작 확인

```bash
pytest
```

`1 passed` 결과가 나오면 환경 셋업 완료입니다.

---

## 5. 작업 분담 (ARCHITECTURE.md 참조)

| 모듈 | 담당자 |
|---|---|
| `modulation/` | dPsk (조장) |
| `rl/` | 한승규 |
| `evaluators/` | 이도현 |
| `reward/`, `utils/` | 송준섭 |

본인 모듈 파일 위치:

```
src/voicesecure/
├── modulation/       # PsychoacousticMasker, Mixer, SafetyChecker
├── rl/               # StateExtractor, PolicyNetwork, RLAgent
├── evaluators/       # SpeakerEvaluator, TTSEvaluator, ASREvaluator
├── reward/           # RewardFunction
└── utils/            # FeatureCache, audio.py
```

테스트 파일은 같은 구조로 `tests/` 하위에 있습니다.

---

## 6. 코드 작성 전 필수 문서 읽기

순서대로 읽으세요:

1. **[ARCHITECTURE.md](ARCHITECTURE.md)** — 가장 중요. 모듈 간 입출력 계약, 데이터 형식, 명명 규칙. 본인 모듈 부분은 외워야 함.
2. **[PRD.md](PRD.md)** — 왜 만드는가, 목표값, 사용자
3. **[SRS.md](SRS.md)** — 정확한 기능 명세 (FR-1 ~ FR-13)
4. **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** — 일정·sprint·분담

특히 ARCHITECTURE.md의 본인 모듈 섹션(섹션 3)은 코드 짜기 전에 정확히 이해해야 합니다. PR 리뷰 시 "ARCHITECTURE.md 위반"이 reject 사유가 됩니다.

---

## 7. 일상 개발 워크플로

### 7.1 작업 시작 (매번)

```bash
# 1. 가상환경 활성화
cd ~/Desktop/voiceSecure/voicesecure-sdk
source .venv/Scripts/activate

# 2. main 최신 받기
git checkout main
git pull origin main

# 3. feature 브랜치 만들기
git checkout -b <type>/<module>-<short-desc>
```

브랜치 명명 규칙:

```
feat/<module>-<desc>     # 새 기능
fix/<module>-<desc>      # 버그 수정
docs/<desc>              # 문서
test/<module>-<desc>     # 테스트만
refactor/<module>-<desc> # 리팩터
chore/<desc>             # 빌드/CI 등
```

예시:
- `feat/modulation-add-masker`
- `fix/rl-policy-nan`
- `docs/update-onboarding`

### 7.2 코드 작성

본인 모듈의 파일을 만들거나 수정합니다. 작성 시 지킬 것:

- ARCHITECTURE.md의 시그니처 정확히 따르기
- docstring + type hint 모든 공개 메서드에 추가
- 새 기능에는 단위 테스트 함께 작성
- 모듈 의존성 단방향 규칙 위반 안 하기

### 7.3 commit

```bash
git add .
git commit -m "<type>(<module>): <subject>"
```

commit 메시지 형식 (Conventional Commits):

```
feat(modulation): add PsychoacousticMasker.compute_threshold
fix(rl): handle NaN in policy network output
docs(arch): clarify Action shape constraint
test(reward): add edge case for CER threshold
```

commit 실행 시 **pre-commit hooks가 자동 검사**합니다:

```
trim trailing whitespace.................................................Passed
fix end of files.........................................................Passed
black....................................................................Passed
isort....................................................................Passed
ruff.....................................................................Passed
```

#### 자동 수정이 발생한 경우

pre-commit이 파일을 자동으로 수정하면 commit이 일시 중단됩니다:

```
black....................................................................Failed
- hook id: black
- files were modified by this hook
```

이 경우 수정된 파일을 다시 스테이징하고 commit 재시도:

```bash
git add -u
git commit -m "<type>(<module>): <subject>"
```

이번엔 모두 Passed로 통과합니다.

### 7.4 push

```bash
git push origin <branch명>
```

성공 메시지:

```
remote: Create a pull request for '<branch>' on GitHub by visiting:
remote:      https://github.com/VoiceSecureHoseo/voicesecure-sdk/pull/new/<branch>
To github.com:VoiceSecureHoseo/voicesecure-sdk.git
 * [new branch]      <branch> -> <branch>
```

### 7.5 Pull Request (PR) 생성

push 메시지의 URL을 클릭하거나, 브라우저로 https://github.com/VoiceSecureHoseo/voicesecure-sdk 접속해서 노란 배너 **"Compare & pull request"** 클릭.

PR 페이지에서:

- **Title**: commit 메시지가 자동 입력됨
- **Description**: 다음 형식 따르기

```markdown
## What
<무엇을 변경했나>

## Why
<왜 필요한가, 어떤 issue/FR 해결>

## How (선택)
<주요 구현 방법>

## Closes
Closes #<issue번호>

## Checklist
- [ ] 단위 테스트 추가/수정
- [ ] Docstrings + type hints
- [ ] 새 외부 의존성 없음 (또는 합의됨)
- [ ] ARCHITECTURE.md 계약 준수
- [ ] 성능 영향 고려
```

- **Reviewers**: 본인 모듈의 부담당자 지정 (ARCHITECTURE.md 참조)
- **Labels**: `module:<모듈>`, `priority:p0/p1/p2`, `sprint:S<번호>`

**Create pull request** 클릭.

### 7.6 CI 통과 + 리뷰

PR 페이지 하단에서:

- 🟡 CI 진행 중 (3~5분)
- ✅ CI 통과
- ❌ CI 실패 → 로그 확인 + 수정

리뷰어가 코드 리뷰 후 **Approve** 또는 **Request changes** 결정.

### 7.7 머지

조건 모두 충족 시:

- ✅ CI 통과
- ✅ 리뷰어 1명 이상 approve
- ✅ Conversation 모두 resolved

**Merge pull request** 버튼 → **Squash and merge** 선택 → **Confirm**.

옵션: **Delete branch** 클릭 (권장 — 머지된 브랜치 자동 정리).

### 7.8 로컬 정리

```bash
git checkout main
git pull origin main
git branch -d <feature-branch명>
```

---

## 8. 막힘 해결 (자주 발생하는 문제)

### 8.1 pre-commit이 자꾸 막아요

정상입니다. 자동 수정이 일어났을 때:

```bash
git add -u
git commit -m "..."
```

### 8.2 CI가 실패해요

PR에서 빨간 X 클릭 → **Details** → 실패한 step 로그 확인.

로컬에서 미리 검사 가능:

```bash
black --check --line-length=100 src tests
isort --check --profile=black --line-length=100 src tests
ruff check src tests
pytest
```

자주 발생하는 실패:

```
# black 실패
black src tests   # 자동 수정 후 다시 commit

# isort 실패
isort --profile=black src tests

# ruff 실패
ruff check --fix src tests

# pytest 실패
pytest -v   # 어떤 테스트 실패했는지 확인 후 코드 수정
```

### 8.3 SSH 인증 실패

```bash
ssh -T git@github.com
```

`Hi <username>!` 안 나오면 SSH 키가 등록 안 됐거나 잘못된 것. 섹션 2 다시 진행.

### 8.4 가상환경 활성화 안 됨

프롬프트에 `(.venv)` 표시가 없으면:

```bash
source .venv/Scripts/activate    # Git Bash
# 또는
.venv\Scripts\activate.bat        # cmd
```

### 8.5 import voicesecure 실패

가상환경 활성화 확인 + 패키지 재설치:

```bash
pip install -e ".[dev]"
```

### 8.6 LF/CRLF 경고

```
warning: in the working copy of '...', LF will be replaced by CRLF
```

Windows ↔ Linux 줄바꿈 차이. **무시해도 됨**. 영원히 안 보고 싶으면:

```bash
git config --global core.autocrlf true
```

### 8.7 main에 직접 push 시도하면 거부됨

```
remote: error: GH013: Repository rule violations found for refs/heads/main.
! [remote rejected] main -> main (protected branch hook declined)
```

정상입니다. Branch Protection이 막아준 것. feature 브랜치 만들어서 PR 거치세요.

```bash
git checkout -b fix/quick-fix
git push origin fix/quick-fix
# GitHub에서 PR 생성
```

---

## 9. 커뮤니케이션 채널

| 채널 | 용도 |
|---|---|
| GitHub Issues | 작업 단위, 버그 보고, 토론 |
| GitHub PR | 코드 리뷰 |
| Slack/Discord `#standup` | 일일 비동기 standup (어제/오늘 작업, 막힘) |
| Slack/Discord `#alerts` | CI 실패, 성능 회귀 알림 |
| 정기 미팅 | 격주 월요일 sprint kickoff, 금요일 retrospective |

---

## 10. 정기 미팅

| 미팅 | 주기 | 길이 |
|---|---|---|
| Daily standup | 매일 (Slack 비동기) | 15분 |
| Sprint kickoff | 격주 월요일 | 1시간 |
| Sprint retrospective | 격주 금요일 | 1시간 |
| Architecture review | 월 1회 | 1시간 |
| Demo | 월 1회 | 30분 |

---

## 11. 도움 요청

막혀서 30분 이상 해결 안 되면 즉시 도움 요청하세요. 혼자 끙끙대지 마세요.

- Slack/Discord에 막힌 부분 + 시도한 것 + 에러 메시지 같이 올리기
- GitHub Issue로 토론할 만한 주제는 Issue 생성

좋은 도움 요청 예시:

```
PolicyNetwork.forward의 출력 shape이 ARCHITECTURE.md 3.2와 다르게 나옵니다.

기대: (batch, 257, 100)
실제: (batch, 257, 50)

n_fft=512, hop=160 사용 중. 1초 입력 (16000 샘플) 기준.

코드: <링크 또는 짧은 스니펫>
시도한 것: hop_length 조정, padding 변경 (둘 다 효과 없음)
```

---

## 12. 첫 작업 추천

환경 셋업 완료 후 본격 모듈 작업 전에, 워크플로 익히기 위한 작은 PR을 한 번 만들어보세요.

예시: 본인 이름을 README의 팀 섹션에 정확한 형식으로 수정:

```bash
git checkout -b docs/update-team-name
# README.md 본인 이름 부분 수정
git add README.md
git commit -m "docs: update team name format"
git push origin docs/update-team-name
# GitHub에서 PR 생성 → 머지
```

이 한 사이클 끝내고 나면 실제 모듈 작업 시작하면 됩니다.

---

## 환영합니다 🎉

질문, 막힘, 제안 모두 환영합니다. 모르는 건 부끄러운 게 아니라 자연스러운 거예요. 좋은 작업 함께 만들어요.

**문서 책임자**: dPsk
**문서 최종 갱신**: 2025-03-31
