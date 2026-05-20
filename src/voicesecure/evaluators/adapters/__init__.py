"""External model adapters for training-time evaluators."""

from voicesecure.evaluators.adapters.campplus import CAMPlusAdapter
from voicesecure.evaluators.adapters.wav2vec2_asr import Wav2Vec2KoreanAdapter
from voicesecure.evaluators.adapters.wavlm_sv import WavLMSVAdapter

__all__ = [
    "CAMPlusAdapter",
    "Wav2Vec2KoreanAdapter",
    "WavLMSVAdapter",
]
