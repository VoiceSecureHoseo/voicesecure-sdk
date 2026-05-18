"""Base contract and helpers for training-time evaluators."""

from __future__ import annotations

import abc
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
    def synthesize_clone(model: Any, reference_audio: AudioArray, *, model_name: str) -> AudioArray:
        """Run a TTS clone adapter and coerce the output to AudioArray."""

        if model is None:
            raise ModelNotConfiguredError(f"{model_name} model adapter is required.")

        if hasattr(model, "clone"):
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
