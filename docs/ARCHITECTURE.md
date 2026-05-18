# VoiceSecure SDK — 아키텍처 문서 v1.0

**문서 상태**: v1.0 (3월 말 기준 확정)
**대상 독자**: 팀원 4명 (한승규 · 이도현 · 송준섭 · dPsk)
**적용 범위**: `src/voicesecure/` 하위 전체 모듈
**목적**: 모든 인원이 동일한 입출력 계약·데이터 형식·명명 규칙으로 구현하도록 강제

> 이 문서는 **합의된 계약(contract)** 이다. 코드와 문서가 어긋나면 코드를 고치거나, 충분한 사유로 PR에서 문서를 갱신한다 (둘 다 같은 PR에서). 임의로 시그니처를 바꾸지 않는다.

---

## 0. 한눈에 보는 시스템

```
원본 음성 ──┬──→ State Extractor ──→ RL Agent (πθ) ──→ raw noise ──→ Psychoacoustic Masker
            │                                                                  │
            │                                                                  ↓ safe noise
            └─────────────────────────────────────────────→ Mixer ←────────────┘
                                                              │
                                                              ↓
                                                       Safety Check
                                                              │
                                                              ↓ 변형 음성
                              ┌──────────────────────────────┼──────────────────────────────┐
                              ↓                              ↓                              ↓
                      SV Evaluator                  TTS Evaluator                  ASR Evaluator
                  (WavLM-SV + CAM++)            (OpenVoice2 + XTTS)        (wav2vec2-xlsr-korean)
                              │                              │                              │
                              └──────────────────────────────┼──────────────────────────────┘
                                                              ↓
                                                       Reward Function
                                                              │
                                                              ↓ scalar reward
                                                       PPO Update → πθ
```

---

## 1. 디렉터리 구조 (정식)

```
voicesecure-sdk/
├── src/voicesecure/
│   ├── __init__.py
│   ├── types.py                      # 공통 타입 정의 (★ 모든 모듈이 import)
│   ├── modulation/                   # 담당: dPsk
│   │   ├── __init__.py
│   │   ├── masker.py                 # PsychoacousticMasker
│   │   ├── mixer.py                  # Mixer (원본 + 안전 노이즈 합성)
│   │   └── safety.py                 # SafetyChecker
│   ├── rl/                           # 담당: 한승규
│   │   ├── __init__.py
│   │   ├── state.py                  # StateExtractor
│   │   ├── policy.py                 # PolicyNetwork (PPO Actor-Critic)
│   │   └── agent.py                  # RLAgent (학습/추론 인터페이스)
│   ├── evaluators/                   # 담당: 이도현
│   │   ├── __init__.py
│   │   ├── base.py                   # Evaluator 추상 클래스 (★ 모든 평가자 상속)
│   │   ├── speaker.py                # SpeakerEvaluator (WavLM-SV + CAM++)
│   │   ├── tts.py                    # TTSEvaluator (OpenVoice2 + XTTS)
│   │   └── asr.py                    # ASREvaluator (wav2vec2-xlsr-korean)
│   ├── reward/                       # 담당: 송준섭
│   │   ├── __init__.py
│   │   └── function.py               # RewardFunction
│   └── utils/                        # 담당: 송준섭
│       ├── __init__.py
│       ├── cache.py                  # FeatureCache
│       └── audio.py                  # 공통 오디오 유틸 (load/save/resample)
├── tests/                            # 모듈 구조 미러링
│   ├── test_modulation/
│   ├── test_rl/
│   ├── test_evaluators/
│   ├── test_reward/
│   └── test_utils/
├── scripts/
│   ├── train.py                      # 학습 진입점
│   └── infer.py                      # 추론 진입점
├── configs/
│   └── default.yaml                  # 하이퍼파라미터·경로 일괄 관리
└── docs/
    └── ARCHITECTURE.md               # 이 문서
```

**규칙**:
- 모듈 간 의존성은 **단방향**: `utils ← modulation ← rl ← reward`, `utils ← evaluators ← reward`. evaluators와 modulation/rl 사이 직접 import 금지.
- 모듈 내부 파일은 자유롭게 서로 import 가능. 모듈 외부에서는 `__init__.py`에 export된 클래스/함수만 사용.
- 새 파일 추가 시 PR 설명에 어느 모듈 소속인지 명시.

---

## 2. 공통 타입 (`src/voicesecure/types.py`)

**모든 모듈이 이 파일의 타입을 사용한다.** 새 타입이 필요하면 이 파일에 추가하고 PR로 합의.

```python
"""공통 타입 정의. 모든 모듈은 여기 정의된 타입만 입출력으로 사용한다."""
from dataclasses import dataclass
from typing import TypedDict
import numpy as np
import numpy.typing as npt
import torch

# ===== 오디오 =====
# 오디오는 항상 mono float32, 정규화 범위 [-1.0, 1.0]
# shape: (num_samples,) — 1차원 numpy array
# sample_rate: 16000 Hz 고정 (v1.0)
AudioArray = npt.NDArray[np.float32]

SAMPLE_RATE = 16000
AUDIO_DTYPE = np.float32
AUDIO_RANGE = (-1.0, 1.0)

# ===== State (RL) =====
# RL Agent의 입력 state. 16-dim 벡터.
# torch.Tensor로 통일 (PPO 학습 시 GPU 이동 편하게)
State = torch.Tensor  # shape: (16,) or (batch, 16), dtype=float32
STATE_DIM = 16

# ===== Action (RL) =====
# RL Agent의 출력. 노이즈 spectrogram (주파수 × 시간)
# v1.0: float32 tensor, shape (n_freq_bins, n_time_frames)
Action = torch.Tensor

# ===== Reward =====
Reward = float  # scalar

# ===== Embedding =====
# 평가 모델 출력 (화자 embedding 등)
Embedding = npt.NDArray[np.float32]  # shape: (embedding_dim,)


# ===== 평가 결과 =====
@dataclass(frozen=True)
class EvaluatorOutput:
    """모든 Evaluator의 표준 출력 형식."""
    score: float                # 정규화된 단일 점수 (높을수록 방어/명료성 성공)
    raw_metric: float           # 정규화 전 원본 지표 (cosine sim, CER 등)
    metadata: dict              # 디버깅/로깅용 부가 정보


# ===== Reward 컴포넌트 =====
class RewardComponents(TypedDict):
    """Reward 함수가 받는 평가 결과 묶음."""
    sv_score: float             # Speaker Verification 거리 점수
    tts_score: float            # TTS 복제 실패 점수
    asr_cer: float              # ASR Character Error Rate (낮을수록 좋음)


# ===== 학습 한 step의 transition =====
@dataclass
class Transition:
    """PPO 학습용 한 step의 (s, a, r, s', done)."""
    state: State
    action: Action
    reward: Reward
    next_state: State
    done: bool
    log_prob: torch.Tensor       # 정책의 로그확률 (PPO ratio 계산용)
    value: torch.Tensor          # Critic의 V(s)
```

---

## 3. 모듈별 입출력 계약

각 모듈의 **공개 인터페이스**(외부에서 호출하는 클래스/메서드)만 명시한다. 내부 구현은 자유.

### 3.1 `modulation/` (담당: dPsk)

#### `PsychoacousticMasker`

```python
class PsychoacousticMasker:
    """RL이 만든 raw noise를 사람 청각 임계치 이내로 clamp한다."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        ...

    def compute_threshold(self, audio: AudioArray) -> AudioArray:
        """
        원본 음성 기반 청각 마스킹 임계치 계산.

        Args:
            audio: shape (num_samples,), float32, [-1, 1]
        Returns:
            threshold: shape (n_freq_bins, n_time_frames), float32
                각 (주파수, 시간) bin에서 사람이 인지 못 하는 노이즈 최대 크기
        """

    def clamp(self, audio: AudioArray, raw_noise: Action) -> Action:
        """
        raw noise를 청각 임계치 이내로 잘라 안전 노이즈 반환.

        Args:
            audio: 원본 음성, shape (num_samples,)
            raw_noise: RL Agent 출력, shape (n_freq_bins, n_time_frames)
        Returns:
            safe_noise: shape (n_freq_bins, n_time_frames)
                |safe_noise| ≤ threshold 보장
        """
```

#### `Mixer`

```python
class Mixer:
    """원본 음성과 안전 노이즈를 합성해 변형 음성을 만든다."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        ...

    def mix(self, audio: AudioArray, safe_noise: Action) -> AudioArray:
        """
        Args:
            audio: 원본 음성, shape (num_samples,)
            safe_noise: 주파수 영역 노이즈, shape (n_freq_bins, n_time_frames)
        Returns:
            modified_audio: shape (num_samples,), [-1, 1] 보장 (clipping)
        """
```

#### `SafetyChecker`

```python
class SafetyChecker:
    """변형 음성에 대한 코드 레벨 안전장치."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        ...

    def check(self, original: AudioArray, modified: AudioArray) -> AudioArray:
        """
        변형 음성을 검증하고 필요시 보정.
        - NaN/Inf 검출 → 0으로 대체
        - clipping 한계 [-1, 1] 재확인
        - RMS 비율이 위험 수준이면 원본 비중 늘려 fallback

        Args:
            original: 원본 음성
            modified: Mixer 출력 변형 음성
        Returns:
            safe_modified: 안전한 최종 변형 음성
        Raises:
            SafetyError: 복구 불가능한 경우 (테스트로 검증)
        """
```

### 3.2 `rl/` (담당: 한승규)

#### `StateExtractor`

```python
class StateExtractor:
    """원본 음성에서 RL Agent의 16-dim state 벡터 추출."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        ...

    def extract(self, audio: AudioArray) -> State:
        """
        Args:
            audio: 원본 음성, shape (num_samples,)
        Returns:
            state: shape (STATE_DIM,) = (16,), float32 torch.Tensor
                구성: [MFCC_mean(13), F0_mean(1), duration(1), rms(1)]
                (정확한 구성은 별도 RFC로 합의)
        """
```

#### `PolicyNetwork`

```python
class PolicyNetwork(torch.nn.Module):
    """PPO Actor-Critic 정책 네트워크."""

    def __init__(self, state_dim: int = STATE_DIM, action_shape: tuple[int, int] = ...) -> None:
        ...

    def forward(self, state: State) -> tuple[Action, torch.Tensor, torch.Tensor]:
        """
        Args:
            state: shape (batch, 16) or (16,)
        Returns:
            action: 노이즈 spectrogram, shape (batch, n_freq, n_time)
            log_prob: shape (batch,) — 행동의 로그확률
            value: shape (batch,) — Critic의 V(s)
        """
```

#### `RLAgent`

```python
class RLAgent:
    """학습/추론 통합 인터페이스. scripts/train.py가 호출.

    내부에 StateExtractor와 PolicyNetwork를 보유하고,
    audio 입력을 받아 state 추출부터 action 생성까지 일괄 처리한다.
    """

    def __init__(self, config: dict | None = None) -> None:
        """
        Args:
            config: 하이퍼파라미터 dict. 없으면 기본값 사용.
                예: {"lr": 3e-4, "ppo_epsilon": 0.2, "gamma": 0.99}
        """
        ...

    def act(
        self,
        audio: AudioArray,
        deterministic: bool = False,
    ) -> tuple[Action, torch.Tensor, torch.Tensor]:
        """음성 입력을 받아 노이즈(action)를 생성.

        내부적으로 StateExtractor로 state 추출 후 PolicyNetwork forward 호출.
        추론 시 deterministic=True (평균 action),
        학습 시 deterministic=False (분포에서 샘플링).

        Args:
            audio: 원본 음성, shape (num_samples,), float32
            deterministic: True면 평균 action, False면 sampling

        Returns:
            action:   노이즈 spectrogram, shape (n_freq, n_time) = (257, 100)
            log_prob: 이 action의 로그확률 (PPO 학습용; 추론 시 무시 가능)
            value:    Critic의 V(s) (PPO 학습용; 추론 시 무시 가능)
        """
        ...

    def update(self, transitions: list[Transition]) -> dict:
        """PPO 업데이트 1회. 학습 시 호출.

        Returns:
            metrics: {"policy_loss": float, "value_loss": float, "entropy": float}
        """
        ...

    def save(self, path: str) -> None:
        """학습된 PolicyNetwork + optimizer 상태 체크포인트로 저장."""
        ...

    def load(self, path: str) -> None:
        """저장된 체크포인트 불러오기."""
        ...
```

**v1.0 설계 결정 (act 시그니처)**

`act`는 `State`가 아닌 `AudioArray`를 입력으로 받는다. 이유:
- 호출자가 매번 `StateExtractor.extract()`를 외부에서 호출하는 부담을 없앰
- audio → state → action 파이프라인이 RLAgent 내부에 캡슐화

`act`는 `(action, log_prob, value)` tuple을 반환한다. 이유:
- 학습 루프에서 Transition을 만들 때 log_prob, value가 필요
- 추론 시에는 `action, _, _ = agent.act(audio)` 형태로 무시 가능

### 3.3 `evaluators/` (담당: 이도현)

#### `Evaluator` (추상 베이스)

```python
class Evaluator(abc.ABC):
    """모든 평가자가 상속해야 하는 베이스 클래스."""

    @abc.abstractmethod
    def evaluate(
        self,
        original: AudioArray,
        modified: AudioArray,
    ) -> EvaluatorOutput:
        """
        Args:
            original: 원본 음성
            modified: 최종 변형 음성 (Safety Check 통과)
        Returns:
            EvaluatorOutput
        """

    @abc.abstractmethod
    def precompute(self, original: AudioArray) -> dict:
        """
        Feature 캐싱용. 원본 features를 미리 계산해 dict로 반환.
        FeatureCache가 호출.
        """
```

#### 구체 평가자

```python
class SpeakerEvaluator(Evaluator):
    """WavLM-SV + CAM++로 화자 embedding 거리 측정."""

    # score = (wavlm_distance * 0.5 + camplus_distance * 0.5)  # 정규화 [0,1]
    # raw_metric = cosine_distance


class TTSEvaluator(Evaluator):
    """OpenVoice2와 XTTS로 변형 음성 복제 시도, 복제 실패 정도 점수화."""

    # score = clone 실패율 [0,1]
    # raw_metric = 복제 음성과 원본의 cosine distance


class ASREvaluator(Evaluator):
    """wav2vec2-xlsr-korean으로 CER 측정.

    ASR은 base Evaluator와 달리 evaluate에 `original_text` 인자를 추가로 받는다.
    SRS FR-9에 명시된 입력 사양 반영.

    Note: precompute는 audio feature 캐싱이 불필요 (text label 비교만 함).
    빈 dict 반환하며 audio validation만 수행.
    """

    def evaluate(
        self,
        original: AudioArray,
        modified: AudioArray,
        original_text: str,
    ) -> EvaluatorOutput:
        """
        Args:
            original: 원본 음성
            modified: 최종 변형 음성
            original_text: 원본 정답 텍스트 (학습 데이터에서 제공)
        Returns:
            EvaluatorOutput
        """

    # score = max(0, 1 - cer)  # 명료성 점수 [0,1]
    # raw_metric = cer

### 3.4 `reward/` (담당: 송준섭)

```python
class RewardFunction:
    """평가 결과를 단일 reward 스칼라로 변환."""

    def __init__(
        self,
        alpha: float = 0.6,   # SV 가중치
        beta: float = 0.4,    # TTS 가중치
        lambda_asr: float = 1.0,  # ASR 페널티 가중치
        cer_threshold: float = 0.3,  # 이 값 이상이면 페널티
    ) -> None:
        ...

    def compute(self, components: RewardComponents) -> Reward:
        """
        R = α·sv_score + β·tts_score − λ·max(0, cer − cer_threshold)

        Args:
            components: SV/TTS/ASR 점수 묶음
        Returns:
            scalar reward (float)
        """
```

### 3.5 `utils/` (담당: 송준섭)

#### `FeatureCache`

```python
class FeatureCache:
    """원본 음성의 evaluator features를 캐싱해 학습 속도 가속."""

    def __init__(self) -> None:
        ...

    def get_or_compute(
        self,
        audio_id: str,
        evaluator: Evaluator,
        audio: AudioArray,
    ) -> dict:
        """
        캐시 hit이면 저장된 features 반환, miss이면 evaluator.precompute() 호출 후 저장.

        Args:
            audio_id: 원본 음성 고유 ID (해시 또는 파일명)
            evaluator: 어느 평가자 features인지
            audio: 원본 음성 (cache miss 시만 사용)
        """

    def clear(self) -> None: ...
```

#### `audio.py`

```python
def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> AudioArray: ...
def save_audio(audio: AudioArray, path: str, sample_rate: int = SAMPLE_RATE) -> None: ...
def resample(audio: AudioArray, src_sr: int, dst_sr: int) -> AudioArray: ...
```

---

## 4. 데이터 형식 (요약)

| 데이터 | 타입 | shape | dtype | 범위 | 단위 |
|--------|------|-------|-------|------|------|
| 오디오 | `np.ndarray` | `(num_samples,)` | `float32` | `[-1, 1]` | mono, 16kHz |
| State | `torch.Tensor` | `(16,)` 또는 `(B, 16)` | `float32` | 정규화됨 | — |
| Action (raw/safe noise) | `torch.Tensor` | `(n_freq, n_time)` 또는 `(B, n_freq, n_time)` | `float32` | — | 주파수 영역 |
| Reward | `float` | scalar | — | 약 `[-2, 2]` | — |
| Embedding | `np.ndarray` | `(emb_dim,)` | `float32` | — | — |

**STFT 파라미터 (v1.0 고정)**: `n_fft=512`, `hop_length=160`, `win_length=400`, `window="hann"`. 모든 모듈이 동일하게 사용.

**오디오 변환 규칙**:
- 외부에서 들어온 오디오는 `utils.audio.load_audio()`를 거쳐서 정규화 후 사용 (16kHz mono float32 보장).
- 모듈 간 전달 시 변환 금지 — 이미 표준 형식이라고 가정.
- numpy ↔ torch 변환은 **호출자가** 책임. 모듈은 시그니처에 명시된 타입만 받는다.

---

## 5. 명명 규칙

### 5.1 Python 코드

| 대상 | 규칙 | 예시 |
|------|------|------|
| 모듈/파일명 | `snake_case`, 단수형 | `masker.py`, `policy.py` |
| 클래스명 | `PascalCase`, 명사 | `PsychoacousticMasker`, `RLAgent` |
| 함수/메서드 | `snake_case`, 동사로 시작 | `compute_threshold`, `extract`, `evaluate` |
| 변수 | `snake_case` | `safe_noise`, `audio_id` |
| 상수 | `UPPER_SNAKE_CASE` | `SAMPLE_RATE`, `STATE_DIM` |
| 비공개 | `_` 접두 | `_compute_mfcc` |
| 타입 별칭 | `PascalCase` (typing) | `AudioArray`, `Action` |

### 5.2 도메인 용어 (한글 ↔ 영어 매핑)

코드에서는 **항상 영어** 용어 사용. 주석·docstring에서는 한글 병기 가능.

| 한글 | 영어 (코드) | 변수명 |
|------|-------------|--------|
| 원본 음성 | original audio | `original`, `audio` |
| 변형 음성 | modified audio | `modified` |
| 안전 노이즈 | safe noise | `safe_noise` |
| 학습/추론 | training/inference | `is_training` 플래그 |
| 화자 | speaker | `speaker_id`, `speaker_embedding` |
| 임베딩 | embedding | `embedding`, `emb` |
| 보상 | reward | `reward`, `r` |
| 정책 | policy | `policy`, `pi` |
| 평가자 | evaluator | `evaluator`, `eval_*` |

### 5.3 파일 명명

- **오디오 데이터**: `{speaker_id}_{utterance_id}.wav` (예: `spk001_utt042.wav`)
- **체크포인트**: `policy_{run_name}_{step}.pt` (예: `policy_baseline_50000.pt`)
- **로그**: `{module}_{YYYYMMDD}.log`
- **config**: `configs/{purpose}.yaml` (예: `configs/default.yaml`, `configs/ablation_no_tts.yaml`)

### 5.4 Git 브랜치/커밋

**브랜치**: `<type>/<module>-<short-desc>`
- 예: `feat/modulation-add-masker`, `fix/rl-policy-nan`, `docs/architecture-v1`

**커밋 메시지** (Conventional Commits):
```
<type>(<module>): <subject>

[body 선택]
```
- type: `feat` `fix` `docs` `test` `refactor` `chore`
- 예: `feat(modulation): add PsychoacousticMasker.compute_threshold`
- 예: `fix(rl): handle NaN in policy network output`

### 5.5 PR 제목

브랜치명과 동일 형식. 본문에 다음 포함:
- 변경 모듈
- 입출력 계약 변경 여부 (Yes면 이 문서도 업데이트 필요)
- 새 의존성 추가 여부
- 테스트 커버리지

---

## 6. 의존성 정책

- **외부 라이브러리 추가**: PR로 제안 → 팀 합의 → `pyproject.toml` 업데이트.
- **버전 고정**: 핵심 라이브러리(torch, transformers, librosa)는 minor 버전까지 명시. 예: `torch>=2.0,<2.3`.
- **모델 weights**: 절대 repo에 commit 금지. `data/` 또는 `checkpoints/` 폴더는 `.gitignore`에 포함, 다운로드 스크립트로 관리.

---

## 7. 테스트 정책

- 모든 공개 메서드는 **최소 1개 이상의 단위 테스트** 필수.
- 테스트 파일은 모듈 구조 미러링: `src/voicesecure/modulation/masker.py` → `tests/test_modulation/test_masker.py`.
- **외부 모델 의존 테스트는 `@pytest.mark.slow`** 마커 부착, 기본 CI에서 제외, 별도 nightly 잡으로 실행.
- 입출력 계약 위반(shape, dtype, 범위)을 검증하는 테스트를 **반드시** 포함:

```python
def test_masker_output_shape(masker, audio):
    raw_noise = torch.randn(257, 100)
    safe = masker.clamp(audio, raw_noise)
    assert safe.shape == raw_noise.shape          # shape 계약
    assert safe.dtype == torch.float32             # dtype 계약
    assert torch.all(torch.abs(safe) <= threshold) # 범위 계약
```

---

## 8. 합의 절차 (이 문서 변경 시)

1. 변경 제안자가 PR 생성 (`docs/architecture-vX-Y`).
2. 본문에 **변경 동기**, **영향받는 모듈**, **마이그레이션 비용** 명시.
3. **팀원 4명 모두 approve** 필요 (브랜치 보호 규칙으로 강제).
4. 머지 시 문서 상단의 버전·날짜 업데이트.
5. 변경된 계약을 따르도록 각 모듈 코드도 같은 PR 또는 follow-up PR로 갱신.

**예외**: 오타·문장 다듬기는 1명 approve로 머지 가능 (라벨 `docs:typo`).

---

## 9. v1.0 → v1.1 예정 변경 사항 (참고)

다음 항목들은 v1.0에서 **잠정 결정**, 추후 변경 가능:
- STATE_DIM = 16의 정확한 구성 (`[mfcc_mean, f0_mean, duration, rms]`)
- Action의 `n_freq_bins`, `n_time_frames` 동적 vs 고정
- TTS Evaluator의 점수 정규화 방식
- FeatureCache의 영속화 (메모리 → 디스크)

이 항목들에 의존하는 코드는 **외부에서 직접 접근하지 말고** 모듈 인터페이스를 거쳐서 사용.

---
| v1.1 | 2025-05-18 | RLAgent.act 시그니처를 코드(agent.py)와 정합화 | dPsk |
| v1.2 | 2025-05-18 | ASREvaluator 시그니처 명시 (original_text 추가), precompute no-op 명시 | dPsk |
**v1.0 확정일**: 2025-03-31
**다음 리뷰 예정**: v1 모듈 통합 직후 (5월 중순 예상)
**문서 책임자**: dPsk (조장)
