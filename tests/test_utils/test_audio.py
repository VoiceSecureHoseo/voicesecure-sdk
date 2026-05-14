"""audio.py unit tests."""

from pathlib import Path

import numpy as np
import pytest

from voicesecure.types import SAMPLE_RATE
from voicesecure.utils.audio import load_audio, resample, save_audio


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
