"""Audio utility functions.

프로젝트 전체에서 사용하는 오디오 입출력 표준화 함수.
오디오는 mono float32, [-1, 1], 16kHz 기준을 따른다.
"""

import logging

import librosa
import numpy as np
import soundfile as sf

from voicesecure.types import SAMPLE_RATE, AudioArray, ChunkSizeError, InsufficientAudioError

# FR-1 상수
_CHUNK_SAMPLES = SAMPLE_RATE          # 16000 — 정확히 1초
_MIN_SAMPLES = int(SAMPLE_RATE * 0.1)  # 1600 — 100ms
_SILENCE_RMS_THRESHOLD = 0.001

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


def prepare_chunk(audio: AudioArray) -> tuple[AudioArray, bool]:
    """FR-1: 1초 청크 단위 입력을 검증하고 표준 형식으로 준비한다.

    동작 순서:
        1. 기본 검증 (_to_audio_array)
        2. 길이 검증 — 너무 짧으면 InsufficientAudioError, 너무 길면 ChunkSizeError
        3. zero-padding — 1600~15999 샘플이면 16000 샘플로 채움
        4. 무음 감지 — RMS < 0.001이면 is_silent=True 반환 (호출자가 pass-through 처리)

    Args:
        audio: 원본 음성 청크, shape (num_samples,), float32

    Returns:
        chunk:     shape (16000,), float32, [-1, 1] — 패딩 완료된 청크
        is_silent: True이면 무음 구간 → 호출자가 변형 없이 원본 반환 권장

    Raises:
        InsufficientAudioError: 입력이 100ms(1600 샘플) 미만
        ChunkSizeError:         입력이 1초(16000 샘플) 초과
    """
    array = _to_audio_array(audio)

    if len(array) < _MIN_SAMPLES:
        raise InsufficientAudioError(
            f"audio too short: {len(array)} samples "
            f"(minimum {_MIN_SAMPLES} = 100ms at {SAMPLE_RATE}Hz)"
        )

    if len(array) > _CHUNK_SAMPLES:
        raise ChunkSizeError(
            f"audio too long: {len(array)} samples "
            f"(maximum {_CHUNK_SAMPLES} = 1s at {SAMPLE_RATE}Hz). "
            "Split into 1-second chunks before calling."
        )

    # zero-padding: 1초보다 짧으면 뒤를 0으로 채움
    if len(array) < _CHUNK_SAMPLES:
        pad_width = _CHUNK_SAMPLES - len(array)
        array = np.pad(array, (0, pad_width), mode="constant", constant_values=0.0)

    # 무음 감지
    rms = float(np.sqrt(np.mean(array ** 2)))
    is_silent = rms < _SILENCE_RMS_THRESHOLD

    if is_silent:
        logger.debug("prepare_chunk: silent audio detected (RMS=%.6f), pass-through recommended", rms)

    return array, is_silent


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
