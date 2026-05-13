"""Shared public types for VoiceSecure modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt

try:  # PyTorch is required by the full SDK, but evaluators can be unit-tested without it.
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal environments
    torch = None  # type: ignore[assignment]


AudioArray = npt.NDArray[np.float32]

SAMPLE_RATE = 16000
AUDIO_DTYPE = np.float32
AUDIO_RANGE = (-1.0, 1.0)

State = Any if torch is None else torch.Tensor
STATE_DIM = 16
Action = Any if torch is None else torch.Tensor

Reward = float
Embedding = npt.NDArray[np.float32]


@dataclass(frozen=True)
class EvaluatorOutput:
    """Standard output returned by every evaluator."""

    score: float
    raw_metric: float
    metadata: dict[str, Any]


class RewardComponents(TypedDict):
    """Evaluation values consumed by the reward function."""

    sv_score: float
    tts_score: float
    asr_cer: float


@dataclass
class Transition:
    """One PPO transition."""

    state: State
    action: Action
    reward: Reward
    next_state: State
    done: bool
    log_prob: State
    value: State

