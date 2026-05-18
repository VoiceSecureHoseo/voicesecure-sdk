"""Audio utility functions.

프로젝트 전체에서 사용하는 오디오 입출력 표준화 함수.
오디오는 mono float32, [-1, 1], 16kHz 기준을 따른다.
"""

import logging

import librosa
import numpy as np
import soundfile as sf

from voicesecure.types import SAMPLE_RATE, AudioArray

logger = logging.getLogger(__name__)


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> AudioArray:
    """오디오 파일을 로드하고 표준 형식으로 변환한다."""
    _validate_sample_rate(target_sr)

    audio, _ = librosa.load(path, sr=target_sr, mono=True)

    return _to_audio_array(audio)


def save_audio(
    audio: AudioArray,
    path: str,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """오디오 배열을 WAV 등 파일로 저장한다."""
    _validate_sample_rate(sample_rate)

    safe_audio = _to_audio_array(audio)

    sf.write(path, safe_audio, sample_rate)


def resample(
    audio: AudioArray,
    src_sr: int,
    dst_sr: int,
) -> AudioArray:
    """오디오를 새로운 sample rate로 변환한다."""
    _validate_sample_rate(src_sr)
    _validate_sample_rate(dst_sr)

    safe_audio = _to_audio_array(audio)

    if src_sr == dst_sr:
        return safe_audio

    resampled = librosa.resample(
        safe_audio,
        orig_sr=src_sr,
        target_sr=dst_sr,
    )

    return _to_audio_array(resampled)


def _to_audio_array(audio: AudioArray) -> AudioArray:
    """입력 오디오를 프로젝트 표준 AudioArray로 변환한다."""
    array = np.asarray(audio, dtype=np.float32)

    if array.size == 0:
        raise ValueError("audio must not be empty")

    if array.ndim != 1:
        raise ValueError(f"audio must be 1-dimensional mono array, got shape {array.shape}")

    if not np.isfinite(array).all():
        raise ValueError("audio must not contain NaN or Inf")

    out_of_range = int(np.sum(np.abs(array) > 1.0))

    if out_of_range > 0:
        fraction = out_of_range / len(array)

        if fraction > 0.001:
            logger.warning(
                "audio: %.2f%% samples out of [-1, 1] -> clipped",
                fraction * 100,
            )

    return np.clip(array, -1.0, 1.0).astype(np.float32)


def _validate_sample_rate(sample_rate: int) -> None:
    """sample rate가 양의 정수인지 검증한다."""
    if not isinstance(sample_rate, int):
        raise TypeError(f"sample_rate must be int, got {type(sample_rate).__name__}")

    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
