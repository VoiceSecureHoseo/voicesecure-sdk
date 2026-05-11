# VoiceSecure SDK — 구현 계획서 v1.0

| 항목 | 내용 |
|---|---|
| **문서 상태** | v1.0 (확정) |
| **작성일** | 2025-03-31 |
| **참조 문서** | PRD v1.0, SRS v1.0, ARCHITECTURE.md v1.0 |
| **기간** | 2025-03-31 ~ 2025-09-15 (약 24주, 5.5개월) |
| **팀 구성** | 4명 (dPsk 조장, 한승규, 이도현, 송준섭) |

> **PRD가 "왜·무엇", SRS가 "어떻게 동작", 구현계획서가 "누가·언제·얼마나"다.**
> 이 문서는 일정·분담·리스크 관리·진행 추적을 다룬다.

---

## 1. 작업 분담 (Work Breakdown Structure)

### 1.1 모듈별 담당자

| 모듈 | 주담당 | 부담당 (리뷰어) | 산출물 |
|---|---|---|---|
| `modulation/` | dPsk | 송준섭 | Masker, Mixer, SafetyChecker |
| `rl/` | 한승규 | dPsk | StateExtractor, PolicyNetwork, RLAgent |
| `evaluators/` | 이도현 | 한승규 | SpeakerEval, TTSEval, ASREval |
| `reward/` | 송준섭 | 이도현 | RewardFunction |
| `utils/` | 송준섭 | 전원 | FeatureCache, audio.py |
| `scripts/train.py` | 한승규 | dPsk | 학습 진입점 |
| `scripts/infer.py` | dPsk | 한승규 | 추론 진입점 |
| **CI/Infra** | dPsk | 전원 | GitHub Actions, pre-commit |
| **문서** | dPsk (총괄) | 전원 (각자 모듈 docstring) | README, API docs |
| **Android SDK** (Phase 2) | dPsk + 한승규 | 이도현, 송준섭 | Android 패키지 |
| **논문 작성** | dPsk (1저자) | 전원 (공저자) | 학회 논문 |

### 1.2 역할 정의

- **주담당 (Owner)**: 해당 모듈의 설계·구현·테스트·문서 책임. 최종 PR 작성자.
- **부담당 (Reviewer)**: PR 리뷰 1차 책임. 주담당 부재 시 emergency fix 권한.
- **전원 (All)**: 큰 변경 시 4명 모두 review.

### 1.3 부하 균형 검증

| 멤버 | 주담당 모듈 수 | 예상 작업 시간 (주당) |
|---|---|---|
| dPsk | modulation + infer + CI + 문서총괄 + 논문 1저자 | 20시간 |
| 한승규 | rl + train | 18시간 |
| 이도현 | evaluators (3개 평가자) | 18시간 |
| 송준섭 | reward + utils | 14시간 (가장 가벼움 → 통합 테스트 추가 부담) |

**송준섭에게 통합 테스트 작성 추가 할당**으로 균형 맞춤.

---

## 2. 일정 (Timeline)

### 2.1 전체 마일스톤

```
3월 31일 ─── 4월 14일 ─── 4월 28일 ─── 5월 12일 ─── 6월 9일 ─── 7월 14일 ─── 7월 28일 ─── 9월 15일
   │            │            │            │            │            │            │            │
 문서 확정    개별 모듈     모듈 통합    Phase 1     Android      통합 테스트   Phase 2      논문 제출
              구현 완료     학습 동작     완료        통합 완료
```

### 2.2 Sprint 단위 (2주)

각 sprint 시작 시 월요일 동기화 미팅, 종료 시 금요일 회고.

| Sprint | 기간 | 목표 |
|---|---|---|
| **S0** | 03-31 ~ 04-13 | 인프라 + 기반 모듈 |
| **S1** | 04-14 ~ 04-27 | 핵심 모듈 구현 |
| **S2** | 04-28 ~ 05-11 | 모듈 통합 + 학습 |
| **S3** | 05-12 ~ 05-25 | Phase 1 데모 + 학습 안정화 |
| **S4** | 05-26 ~ 06-08 | Android 환경 셋업 |
| **S5** | 06-09 ~ 06-22 | Android 통합 |
| **S6** | 06-23 ~ 07-06 | 통합 테스트 + 청취 테스트 |
| **S7** | 07-07 ~ 07-20 | 성능 최적화 + 버그 수정 |
| **S8** | 07-21 ~ 08-03 | Phase 2 완료 + 패키징 |
| **S9** | 08-04 ~ 08-31 | 논문 작성 |
| **S10** | 09-01 ~ 09-15 | 논문 최종화 + 제출 |

### 2.3 Sprint별 상세 계획

#### S0 (3/31~4/13) — 인프라 + 기반

| 담당 | 작업 | 산출물 |
|---|---|---|
| dPsk | GitHub repo, CI/CD, pre-commit, 문서 | 동작하는 repo + 문서 3종 |
| 송준섭 | `utils/audio.py` 구현 | load/save/resample 함수 + 테스트 |
| 송준섭 | `utils/cache.py` 구현 | FeatureCache + 테스트 |
| 한승규 | PyTorch + Stable-Baselines3 환경 셋업, PolicyNetwork 스켈레톤 | dummy forward 동작 |
| 이도현 | 평가 모델 다운로드 + 추론 코드 prototype | 각 모델 동작 확인 |

**S0 완료 기준**:
- [ ] GitHub repo public, CI 통과
- [ ] `utils/` 모듈 단위 테스트 통과
- [ ] 평가 모델 5종 (WavLM-SV, CAM++, OpenVoice2, XTTS, wav2vec2) 추론 동작

#### S1 (4/14~4/27) — 핵심 모듈

| 담당 | 작업 |
|---|---|
| dPsk | `modulation/masker.py` (PsychoacousticMasker) |
| dPsk | `modulation/mixer.py` (Mixer) |
| dPsk | `modulation/safety.py` (SafetyChecker) |
| 한승규 | `rl/state.py` (StateExtractor) |
| 한승규 | `rl/policy.py` (PolicyNetwork) |
| 한승규 | `rl/agent.py` (RLAgent — 학습/추론 인터페이스) |
| 이도현 | `evaluators/base.py` (Evaluator 추상 클래스) |
| 이도현 | `evaluators/speaker.py` (SpeakerEvaluator) |
| 송준섭 | `reward/function.py` (RewardFunction) |

**S1 완료 기준**:
- [ ] 위 모듈 모두 단위 테스트 통과
- [ ] 모듈별 PR 리뷰 완료, main 머지

#### S2 (4/28~5/11) — 통합 + 학습

| 담당 | 작업 |
|---|---|
| 이도현 | `evaluators/tts.py`, `evaluators/asr.py` (나머지 평가자) |
| 한승규 | `scripts/train.py` (학습 진입점) |
| 한승규 | dummy 데이터로 학습 1 epoch 성공 |
| dPsk | `scripts/infer.py` (추론 진입점) |
| 송준섭 | 통합 테스트 작성 (end-to-end) |
| 전원 | KsponSpeech 다운로드 + preprocessing |

**S2 완료 기준**:
- [ ] end-to-end 파이프라인 동작 (audio in → modified audio out)
- [ ] PPO 학습 1 epoch 완료, 체크포인트 저장
- [ ] 통합 테스트 모두 통과

#### S3 (5/12~5/25) — Phase 1 데모

| 담당 | 작업 |
|---|---|
| 한승규 | 본격 학습 (수렴까지 며칠 소요) |
| 한승규 | 학습 안정화 (NaN, divergence 디버깅) |
| 이도현 | 매 epoch마다 평가 + 결과 기록 |
| dPsk | Raspberry Pi 4B 환경 셋업 + ONNX 변환 |
| dPsk | Phase 1 데모 영상 제작 |
| 송준섭 | latency/메모리 프로파일링 |

**S3 완료 기준**:
- [ ] PRD G1~G5 목표값 달성 (또는 근접)
- [ ] Raspberry Pi에서 200ms 이내 latency 달성
- [ ] 데모 영상 + Phase 1 보고서

#### S4~S5 (5/26~6/22) — Android 통합

| 담당 | 작업 |
|---|---|
| dPsk + 한승규 | Android Studio 프로젝트, JNI/Kotlin 인터페이스 |
| dPsk | ONNX Runtime Android 통합 |
| 한승규 | Native AudioRecord/AudioTrack 연결 |
| 이도현 | Sample app 화면 (간단한 토글 UI) |
| 송준섭 | Android 단위 테스트 (AndroidJUnitRunner) |

**S5 완료 기준**:
- [ ] Pixel 6 등 실기기에서 동작
- [ ] 300ms 이내 latency
- [ ] Sample app으로 보호 효과 시연 가능

#### S6~S7 (6/23~7/20) — 테스트 + 최적화

| 담당 | 작업 |
|---|---|
| 송준섭 | 24시간 연속 부하 테스트 |
| 이도현 | ABX 청취 테스트 (외부 5명 섭외) |
| dPsk | 성능 병목 프로파일링 + 최적화 |
| 한승규 | Hyperparameter 튜닝 (재학습) |
| 전원 | 발견된 버그 수정 |

**S6~S7 완료 기준**:
- [ ] PRD 모든 목표값 달성
- [ ] 24시간 부하 테스트 메모리 누수 없음
- [ ] ABX 청취 테스트 결과 60% 이하

#### S8 (7/21~8/3) — Phase 2 완료

| 담당 | 작업 |
|---|---|
| dPsk | README, API 문서, 통합 가이드 |
| 한승규 | Sample app 최종 정리 |
| 이도현 | 평가 결과 정리 (논문용 표·그래프) |
| 송준섭 | CI 최종 점검, 릴리스 자동화 |
| 전원 | v1.0 태그, GitHub release |

#### S9~S10 (8/4~9/15) — 논문

| 담당 | 작업 |
|---|---|
| dPsk (1저자) | 논문 초고 작성 |
| 한승규 | RL 섹션 (방법론) |
| 이도현 | Experiments 섹션 (평가 결과) |
| 송준섭 | Implementation 섹션 (시스템 설계) |
| 전원 | Reviewer comment 대응 |

**S10 완료 기준**:
- [ ] 학회 논문 1편 제출

---

## 3. 의존성 그래프 (Dependency Graph)

작업 간 선후관계. 화살표가 있는 작업이 끝나야 다음 작업 시작.

```
[S0: 인프라]
    │
    ├──→ [S0: utils/audio]
    │       │
    │       ├──→ [S1: modulation/masker]
    │       └──→ [S1: rl/state]
    │
    ├──→ [S0: utils/cache]
    │       │
    │       └──→ [S1: evaluators/speaker]  (cache 의존)
    │
    └──→ [S0: 평가 모델 다운로드]
            │
            └──→ [S1: evaluators/*]


[S1 모든 모듈]
    │
    └──→ [S2: scripts/train.py]   (모든 모듈 통합)
            │
            └──→ [S3: 학습/평가]
                    │
                    └──→ [S4: ONNX 변환]
                            │
                            └──→ [S5: Android 통합]


[S6: 평가 결과] ───→ [S9: 논문 작성]
[S8: 패키지 완료]
```

### 3.1 임계 경로 (Critical Path)

이 경로의 작업이 늦으면 전체 일정이 밀린다:

```
S0 인프라 → S1 RL 모듈 → S2 학습 통합 → S3 학습 수렴 → S4-5 Android → S8 패키지
```

**가장 위험한 단계**: S3 (학습 수렴). RL 학습이 안 수렴하면 일정 전체가 영향. → 백업 계획 필요 (4번 항목).

---

## 4. 리스크 관리

PRD에서 식별한 리스크에 대한 **구체적 대응 계획**.

| 리스크 | 발생 시점 | 조기 경고 신호 | 대응 |
|---|---|---|---|
| RL 학습 안 수렴 | S3 | 100 epoch 후도 reward 평탄 | 백업안: action space를 5-discrete으로 단순화, 보상 항목 수 줄이기 |
| Raspberry Pi latency 초과 | S3 후반 | 프로토타입 latency > 300ms | 모델 레이어 축소, FP16 quantization, 청크 크기 축소 |
| 평가 모델 메모리 초과 | S1~S2 | OOM 발생 | Lazy loading, 평가자 sharding |
| Android 통합 어려움 | S4~S5 | JNI 빌드 실패 지속 | Phase 2 기능을 Python wrapper로 일단 시연, native는 v1.1로 연기 |
| 팀원 일정 차질 | 상시 | 매주 보고에서 다음 sprint 작업 미완료 예상 | 부담당이 백업, 또는 다음 sprint로 이월 (PRD 비목표 추가 검토) |
| 학회 마감 변경 | S9 | CFP 변경 공지 | 백업 학회 (KSC 등) 미리 식별, 노트북 paper로 대체 |
| 데이터셋 라이선스 문제 | S0 후반 | 다운로드/사용 조건 발견 | 자체 수집으로 부분 대체, 영어만 사용 |
| pre-commit/CI 학습 곡선 | S0 | PR 머지 지연 | dPsk가 1:1 onboarding, FAQ 문서 작성 |

### 4.1 리스크 모니터링

매 sprint 종료 회고에서 리스크 상태 점검:
- 신규 리스크 발견되면 위 표에 추가
- 발생한 리스크는 별도 incident 문서로 기록
- 임계 리스크 발생 시 즉시 4명 긴급 회의

---

## 5. 작업 추적 (Tracking)

### 5.1 GitHub Issues + Projects

- 모든 작업은 GitHub Issue로 생성
- 라벨: `module:modulation`, `module:rl`, ... (모듈별), `priority:p0/p1/p2`, `sprint:S1` 등
- GitHub Projects의 칸반 보드 사용 (Todo / In Progress / Review / Done)
- 각 Issue는 SRS의 FR-X와 연결 (Issue 본문에 `Implements: FR-3`)

### 5.2 PR 규칙

- PR 제목: `<type>(<module>): <subject>` (Conventional Commits)
- PR 본문 필수 항목:
  - `Closes #<issue>`
  - 변경 모듈
  - 테스트 추가 여부
  - ARCHITECTURE.md 변경 여부
- 머지 조건: CI 통과 + 부담당 1명 이상 approve

### 5.3 진행률 측정

매 sprint 종료 시:

```
이번 sprint 계획 작업 수: N
완료: M
완료율: M/N

PR 머지 수: K
평균 리뷰 시간: X시간
```

연속 2 sprint 완료율 < 70%이면 일정 재계획.

### 5.4 정기 미팅

| 미팅 | 주기 | 길이 | 의제 |
|---|---|---|---|
| Daily standup | 매일 (Slack 비동기) | 15분 | 어제/오늘 작업, 막힘 |
| Sprint kickoff | 격주 월요일 | 1시간 | 이번 sprint 계획 |
| Sprint retrospective | 격주 금요일 | 1시간 | 회고, 개선점, 리스크 |
| Architecture review | 월 1회 | 1시간 | ARCHITECTURE.md 변경 검토 |
| Demo | 월 1회 | 30분 | 진행 상황 시연, 지도교수 보고 |

---

## 6. 품질 관리

### 6.1 코드 품질

- pre-commit hooks 통과 강제 (black, isort, ruff)
- 단위 테스트 커버리지 ≥ 70% (CI에서 측정)
- 모든 공개 메서드에 docstring + type hint
- ARCHITECTURE.md 계약 준수 (PR 리뷰에서 확인)

### 6.2 테스트 전략

- **Unit**: 모듈별 (`tests/test_<module>/`), 빠르게 실행 (외부 모델 의존 X)
- **Integration**: end-to-end 파이프라인 (`tests/integration/`)
- **Slow**: 실제 평가 모델 사용 (`@pytest.mark.slow`), nightly만 실행
- **Performance**: latency·메모리 (`tests/perf/`), Raspberry Pi 실기 측정

### 6.3 PR 리뷰 가이드

리뷰어가 확인할 것:

1. ARCHITECTURE.md 계약 준수 여부
2. 단위 테스트 추가됐는지 (계약 검증 포함)
3. type hint, docstring 완비
4. 새 의존성 추가 시 합의 여부
5. 성능 영향 (특히 임계 경로)

---

## 7. 커뮤니케이션 채널

| 채널 | 용도 |
|---|---|
| GitHub Issues | 작업 단위, 버그, 토론 |
| GitHub PR | 코드 리뷰 |
| Slack/Discord `#general` | 일상 대화 |
| Slack/Discord `#standup` | 일일 비동기 standup |
| Slack/Discord `#alerts` | CI 실패, 성능 회귀 알림 |
| Notion or Wiki | 회의록, 결정 기록 |
| Email | 지도교수 공식 보고 (월 1회) |

---

## 8. 산출물 체크리스트 (Definition of Done)

각 phase 완료 시 다음 항목 모두 충족.

### Phase 1 (5월 중순)

- [ ] `src/voicesecure/` 모든 모듈 구현 완료
- [ ] 단위 테스트 커버리지 ≥ 70%
- [ ] 통합 테스트 통과
- [ ] PRD G1~G5 목표값 달성
- [ ] Raspberry Pi 4B 데모 영상
- [ ] Phase 1 보고서 (지도교수 제출)

### Phase 2 (7월 말)

- [ ] Android Studio 프로젝트 + Sample app
- [ ] PRD G6~G9 목표값 달성
- [ ] ABX 청취 테스트 완료
- [ ] 24시간 부하 테스트 통과
- [ ] README + API 문서 + 통합 가이드 완성
- [ ] GitHub v1.0 release

### 학술 산출물 (9월 중순)

- [ ] 논문 초고 (최소 8페이지)
- [ ] 모든 실험 결과 표/그래프
- [ ] 학회 제출 완료

---

## 9. 변경 절차

이 구현계획서는 sprint 종료 회고에서 갱신될 수 있다. 절차:

1. 변경 제안자가 회고 미팅에서 발의
2. 영향 평가 (다른 작업/일정/품질 영향)
3. 4명 합의 시 PR로 문서 업데이트
4. 변경 이력에 기록

---

## 10. 변경 이력

| 버전 | 날짜 | 변경 사항 | 작성자 |
|---|---|---|---|
| v1.0 | 2025-03-31 | 초안 확정 | dPsk |

---

**문서 책임자**: dPsk
**문의**: GitHub Issues (label: `plan`)
