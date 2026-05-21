"""공통 타입 정의. 모든 모듈은 여기 정의된 타입만 입출력으로 사용한다."""

from dataclasses import dataclass
from typing import Any, TypedDict

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
# RL Agent의 입력 state. 36-dim 벡터.
# torch.Tensor로 통일 (PPO 학습 시 GPU 이동 편하게)
State = torch.Tensor  # shape: (36,) or (batch, 36), dtype=float32
STATE_DIM = 36

# ===== Action (RL) =====
# RL Agent의 출력. 노이즈 방향 spectrogram.
# shape: (n_freq_bins, n_time_frames)
#   값 범위 [-1, 1]: 부호가 노이즈 방향, 심리음향 임계치로 scale
Action = torch.Tensor

# n_freq_bins = n_fft/2 + 1 = 257 (STFT 파라미터 고정값)
# n_time_frames = 100 (1초 청크, hop_length=160 기준: 16000/160=100)
ACTION_N_FREQ = 257
ACTION_N_TIME = 100

# ===== Reward =====
Reward = float  # scalar

# ===== Embedding =====
# 평가 모델 출력 (화자 embedding 등)
Embedding = npt.NDArray[np.float32]  # shape: (embedding_dim,)


# ===== 평가 결과 =====
@dataclass(frozen=True)
class EvaluatorOutput:
    """모든 Evaluator의 표준 출력 형식."""

    score: float  # 정규화된 단일 점수 (높을수록 방어/명료성 성공)
    raw_metric: float  # 정규화 전 원본 지표 (cosine sim, CER 등)
    metadata: dict[str, Any]  # 디버깅/로깅용 부가 정보


# ===== Reward 컴포넌트 =====
class RewardComponents(TypedDict):
    """Reward 함수가 받는 평가 결과 묶음."""

    sv_score: float  # Speaker Verification 거리 점수
    tts_score: float  # TTS 복제 실패 점수
    asr_cer: float  # ASR Character Error Rate (낮을수록 좋음)


# ===== FR-1 예외 =====
class InsufficientAudioError(ValueError):
    """입력 길이가 100ms(1600 샘플) 미만일 때."""


class ChunkSizeError(ValueError):
    """입력 길이가 16000 샘플(1초)을 초과할 때. 호출자가 청크 분할 후 재호출 필요."""


# ===== 학습 한 step의 transition =====
@dataclass
class Transition:
    """PPO 학습용 한 step의 (s, a, r, s', done)."""

    state: State
    action: Action
    reward: Reward
    next_state: State
    done: bool
    log_prob: torch.Tensor  # 정책의 로그확률 (PPO ratio 계산용)
    value: torch.Tensor  # Critic의 V(s)
    freq_pattern: torch.Tensor  # tanh(freq_raw), shape (n_freq,) — evaluate_actions 역산용
    time_gate: torch.Tensor  # sigmoid(time_raw), shape (n_time,) — evaluate_actions 역산용
