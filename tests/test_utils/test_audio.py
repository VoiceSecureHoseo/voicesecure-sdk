"""audio.py unit tests."""

from pathlib import Path

import numpy as np
import pytest

from voicesecure.types import SAMPLE_RATE, ChunkSizeError, InsufficientAudioError
from voicesecure.utils.audio import load_audio, prepare_chunk, resample, save_audio


def test_resample_same_sample_rate():
    """같은 sample rate이면 원본 유지."""
    audio = np.random.randn(SAMPLE_RATE).astype(np.float32)

    output = resample(
        audio,
        src_sr=SAMPLE_RATE,
        dst_sr=SAMPLE_RATE,
    )

    assert output.shape == audio.shape
    assert output.dtype == np.float32


@pytest.mark.slow
def test_resample_different_sample_rate():
    """sample rate 변경 확인."""
    audio = np.random.randn(8000).astype(np.float32)

    output = resample(
        audio,
        src_sr=8000,
        dst_sr=16000,
    )

    assert output.dtype == np.float32
    assert len(output) > len(audio)


@pytest.mark.slow
def test_save_and_load_audio(tmp_path: Path):
    """오디오 저장 후 다시 로드."""
    audio = np.random.randn(SAMPLE_RATE).astype(np.float32)

    save_path = tmp_path / "test.wav"

    save_audio(
        audio,
        str(save_path),
    )

    loaded = load_audio(
        str(save_path),
    )

    assert loaded.dtype == np.float32
    assert loaded.ndim == 1


def test_resample_clips_audio_range():
    """오디오 값이 [-1, 1] 범위를 벗어나면 clipping되어야 한다."""
    audio = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)

    output = resample(
        audio,
        src_sr=SAMPLE_RATE,
        dst_sr=SAMPLE_RATE,
    )

    assert output.min() >= -1.0
    assert output.max() <= 1.0


def test_empty_audio_raises_value_error():
    """빈 audio는 ValueError가 발생해야 한다."""
    audio = np.array([], dtype=np.float32)

    with pytest.raises(ValueError):
        resample(
            audio,
            src_sr=SAMPLE_RATE,
            dst_sr=SAMPLE_RATE,
        )


def test_stereo_audio_raises_value_error():
    """2차원 stereo audio는 표준 AudioArray가 아니므로 ValueError가 발생해야 한다."""
    audio = np.zeros((2, SAMPLE_RATE), dtype=np.float32)

    with pytest.raises(ValueError):
        resample(
            audio,
            src_sr=SAMPLE_RATE,
            dst_sr=SAMPLE_RATE,
        )


def test_invalid_sample_rate_raises_value_error():
    """sample rate가 0 이하이면 ValueError가 발생해야 한다."""
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)

    with pytest.raises(ValueError):
        resample(
            audio,
            src_sr=0,
            dst_sr=SAMPLE_RATE,
        )


def test_non_integer_sample_rate_raises_type_error():
    """sample rate가 int가 아니면 TypeError가 발생해야 한다."""
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)

    with pytest.raises(TypeError):
        resample(
            audio,
            src_sr=16000.0,
            dst_sr=SAMPLE_RATE,
        )


def test_non_integer_dst_sample_rate_raises_type_error():
    """dst sample rate가 int가 아니면 TypeError가 발생해야 한다."""
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)

    with pytest.raises(TypeError):
        resample(
            audio,
            src_sr=SAMPLE_RATE,
            dst_sr=16000.0,
        )


def test_audio_with_nan_or_inf_raises_value_error():
    """audio에 NaN 또는 Inf가 있으면 ValueError가 발생해야 한다."""
    audio = np.array([0.0, np.nan, np.inf], dtype=np.float32)

    with pytest.raises(ValueError):
        resample(
            audio,
            src_sr=SAMPLE_RATE,
            dst_sr=SAMPLE_RATE,
        )


# ── FR-1: prepare_chunk ──────────────────────────────────────────────────────


def test_prepare_chunk_exact_length():
    """정확히 1초(16000 샘플)이면 그대로 반환."""
    audio = np.random.uniform(-0.5, 0.5, SAMPLE_RATE).astype(np.float32)
    chunk, is_silent = prepare_chunk(audio)
    assert chunk.shape == (SAMPLE_RATE,)
    assert chunk.dtype == np.float32
    assert not is_silent


def test_prepare_chunk_short_audio_is_padded():
    """100ms~1초 미만 입력은 16000 샘플로 zero-padding."""
    audio = np.random.uniform(-0.5, 0.5, 8000).astype(np.float32)  # 0.5초
    chunk, _ = prepare_chunk(audio)
    assert chunk.shape == (SAMPLE_RATE,)
    assert np.all(chunk[8000:] == 0.0)  # 뒷부분이 0으로 채워졌는지


def test_prepare_chunk_too_short_raises():
    """100ms(1600 샘플) 미만은 InsufficientAudioError."""
    audio = np.zeros(1599, dtype=np.float32)
    with pytest.raises(InsufficientAudioError):
        prepare_chunk(audio)


def test_prepare_chunk_minimum_length_passes():
    """정확히 1600 샘플(100ms)은 통과."""
    audio = np.random.uniform(-0.5, 0.5, 1600).astype(np.float32)
    chunk, _ = prepare_chunk(audio)
    assert chunk.shape == (SAMPLE_RATE,)


def test_prepare_chunk_too_long_raises():
    """1초(16000 샘플) 초과는 ChunkSizeError."""
    audio = np.zeros(16001, dtype=np.float32)
    with pytest.raises(ChunkSizeError):
        prepare_chunk(audio)


def test_prepare_chunk_silent_audio_detected():
    """RMS < 0.001인 무음은 is_silent=True."""
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    _, is_silent = prepare_chunk(audio)
    assert is_silent


def test_prepare_chunk_non_silent_audio():
    """충분한 에너지가 있는 음성은 is_silent=False."""
    audio = np.random.uniform(-0.5, 0.5, SAMPLE_RATE).astype(np.float32)
    _, is_silent = prepare_chunk(audio)
    assert not is_silent


def test_prepare_chunk_output_range():
    """출력이 [-1, 1] 범위 내에 있어야 한다."""
    audio = np.random.uniform(-0.9, 0.9, SAMPLE_RATE).astype(np.float32)
    chunk, _ = prepare_chunk(audio)
    assert chunk.min() >= -1.0
    assert chunk.max() <= 1.0
