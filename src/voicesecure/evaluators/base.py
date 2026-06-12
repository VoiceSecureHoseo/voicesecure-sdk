"""Base contract and helpers for training-time evaluators.

═══════════════════════════════════════════════════════════════════════════
[V1] 원본 대비 수정사항
═══════════════════════════════════════════════════════════════════════════
1. synthesize_clone()에 text 파라미터 추가 + 어댑터로 전파.
   - 왜: 기존엔 model.clone(reference_audio) 단일 인자 호출만 있어서,
     CosyVoiceAdapter가 기본값 "안녕하세요"로 모든 클론을 합성함
     (tts_text와 prompt_text 양쪽 모두). CosyVoice zero-shot의 prompt_text는
     "레퍼런스 오디오의 실제 전사"여야 하므로, 다른 문장을 말하는 KSS 오디오에
     "안녕하세요"를 프롬프트로 주면 클론 자체가 망가져 reward가 perturbation과
     무관해짐. 설계 의도(원본과 클론이 같은 텍스트를 말하게 해서 텍스트 변수를
     통제하고 화자 임베딩만 비교 — train.py/adapter docstring에 명시돼 있었으나
     코드 경로에 미구현)를 실제로 구현.
   - XTTS 호환: XTTSAdapter.clone(reference_audio)은 text 인자가 없고 고정
     clone_text를 사용하므로, inspect.signature로 어댑터가 text 파라미터를
     지원하는 경우에만 전달 (XTTS 경로는 기존 동작 그대로 유지).
═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import abc
import inspect  # [V1 추가] clone 어댑터의 text 파라미터 지원 여부 검사용
from collections.abc import Callable
from typing import Any

import numpy as np

from voicesecure.types import AUDIO_DTYPE, AUDIO_RANGE, AudioArray, Embedding, EvaluatorOutput


class EvaluatorError(RuntimeError):
    """Raised when an evaluator cannot produce a valid result."""


class ModelNotConfiguredError(EvaluatorError):
    """Raised when a required external model adapter was not provided."""


class Evaluator(abc.ABC):
    """Base class that every evaluator must inherit."""

    @abc.abstractmethod
    def evaluate(self, original: AudioArray, modified: AudioArray) -> EvaluatorOutput:
        """Evaluate a final modified waveform against its original waveform."""

    @abc.abstractmethod
    def precompute(self, original: AudioArray) -> dict[str, Any]:
        """Precompute features for cache use."""

    @staticmethod
    def validate_audio(audio: AudioArray, *, name: str) -> AudioArray:
        """Validate and normalize an audio input to the public AudioArray contract."""

        array = np.asarray(audio)
        if array.ndim != 1:
            raise ValueError(f"{name} must be a 1-D mono waveform, got shape {array.shape}.")
        if array.size == 0:
            raise ValueError(f"{name} must not be empty.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or Inf.")

        array = array.astype(AUDIO_DTYPE, copy=False)
        lo, hi = AUDIO_RANGE
        if np.any(array < lo) or np.any(array > hi):
            raise ValueError(f"{name} must be normalized to [{lo}, {hi}].")
        return array

    @staticmethod
    def extract_embedding(model: Any, audio: AudioArray, *, model_name: str) -> Embedding:
        """Run a model adapter and coerce the result to a 1-D float32 embedding."""

        if model is None:
            raise ModelNotConfiguredError(f"{model_name} model adapter is required.")

        if hasattr(model, "extract_embedding"):
            result = model.extract_embedding(audio)
        elif hasattr(model, "encode"):
            result = model.encode(audio)
        elif callable(model):
            result = model(audio)
        else:
            raise TypeError(
                f"{model_name} must be callable or expose extract_embedding(audio)/encode(audio)."
            )

        embedding = np.asarray(result, dtype=np.float32)
        if embedding.ndim != 1:
            raise ValueError(f"{model_name} embedding must be 1-D, got shape {embedding.shape}.")
        if embedding.size == 0:
            raise ValueError(f"{model_name} embedding must not be empty.")
        if not np.all(np.isfinite(embedding)):
            raise ValueError(f"{model_name} embedding contains NaN or Inf.")
        return embedding

    @staticmethod
    def synthesize_clone(
        model: Any,
        reference_audio: AudioArray,
        *,
        model_name: str,
        text: str | None = None,  # [V1 추가] 클론이 말할 텍스트 (= 레퍼런스 오디오의 전사)
    ) -> AudioArray:
        """Run a TTS clone adapter and coerce the output to AudioArray.

        [V1] text가 주어지고 어댑터의 clone()이 text 파라미터를 지원하면 전달한다.
        CosyVoice 계열은 prompt_text(레퍼런스 전사)가 실제 오디오 내용과 일치해야
        클로닝 품질이 보장되므로 필수. XTTSAdapter처럼 text를 받지 않는 어댑터는
        기존과 동일하게 단일 인자로 호출된다.
        """

        if model is None:
            raise ModelNotConfiguredError(f"{model_name} model adapter is required.")

        if hasattr(model, "clone"):
            # [V1 수정] 기존: result = model.clone(reference_audio) 고정
            #          → CosyVoiceAdapter가 항상 기본값 "안녕하세요"로 합성되는 버그.
            # 수정: 어댑터 시그니처에 text가 있고 호출자가 text를 줬을 때만 전달.
            if text is not None and "text" in inspect.signature(model.clone).parameters:
                result = model.clone(reference_audio, text=text)
            else:
                result = model.clone(reference_audio)
        elif hasattr(model, "synthesize"):
            result = model.synthesize(reference_audio)
        elif callable(model):
            result = model(reference_audio)
        else:
            raise TypeError(
                f"{model_name} must be callable or expose clone(audio)/synthesize(audio)."
            )

        return Evaluator.validate_audio(result, name=f"{model_name} clone")

    @staticmethod
    def transcribe(model: Any, audio: AudioArray, *, model_name: str) -> str:
        """Run an ASR adapter and coerce the result to text."""

        if model is None:
            raise ModelNotConfiguredError(f"{model_name} model adapter is required.")

        if hasattr(model, "transcribe"):
            result = model.transcribe(audio)
        elif hasattr(model, "predict"):
            result = model.predict(audio)
        elif callable(model):
            result = model(audio)
        else:
            raise TypeError(
                f"{model_name} must be callable or expose transcribe(audio)/predict(audio)."
            )

        if not isinstance(result, str):
            raise TypeError(f"{model_name} transcription must be str, got {type(result).__name__}.")
        return result


def cosine_distance(left: Embedding, right: Embedding) -> float:
    """Return 1 - cosine similarity for two embeddings."""

    lhs = np.asarray(left, dtype=np.float32)
    rhs = np.asarray(right, dtype=np.float32)
    if lhs.shape != rhs.shape:
        raise ValueError(f"Embedding shapes must match, got {lhs.shape} and {rhs.shape}.")

    denominator = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    if denominator == 0.0:
        raise ValueError("Cosine distance is undefined for zero-vector embeddings.")

    similarity = float(np.dot(lhs, rhs) / denominator)
    return 1.0 - float(np.clip(similarity, -1.0, 1.0))


def clip_unit_interval(value: float) -> float:
    """Clip a score to the normalized [0, 1] range."""

    return float(np.clip(value, 0.0, 1.0))


Normalizer = Callable[[float], float]
