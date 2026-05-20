"""External model adapters for training-time evaluators."""

from __future__ import annotations


def __getattr__(name: str):
    if name == "CAMPlusAdapter":
        from voicesecure.evaluators.adapters.campplus import CAMPlusAdapter

        return CAMPlusAdapter
    if name == "Wav2Vec2KoreanAdapter":
        from voicesecure.evaluators.adapters.wav2vec2_asr import Wav2Vec2KoreanAdapter

        return Wav2Vec2KoreanAdapter
    if name == "WavLMSVAdapter":
        from voicesecure.evaluators.adapters.wavlm_sv import WavLMSVAdapter

        return WavLMSVAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CAMPlusAdapter",
    "Wav2Vec2KoreanAdapter",
    "WavLMSVAdapter",
]
