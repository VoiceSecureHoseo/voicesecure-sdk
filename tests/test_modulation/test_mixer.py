"""Tests for Mixer.

See Also
--------
- src/voicesecure/modulation/mixer.py
- ARCHITECTURE.md section 3.1
- SRS.md FR-5
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from voicesecure.modulation.mixer import Mixer, MixerConfig


@pytest.fixture
def mixer() -> Mixer:
    return Mixer()


@pytest.fixture
def audio() -> np.ndarray:
    """1초 분량 더미 음성."""
    rng = np.random.default_rng(seed=42)
    return (rng.standard_normal(16000) * 0.1).astype(np.float32)


@pytest.fixture
def safe_noise() -> torch.Tensor:
    """안전한 노이즈 spectrogram (작은 값)."""
    return torch.randn(257, 100) * 0.01


class TestInitialization:
    def test_default_config(self) -> None:
        mixer = Mixer()
        assert mixer.config.sample_rate == 16000
        assert mixer.config.n_fft == 512

    def test_custom_config(self) -> None:
        config = MixerConfig(sample_rate=22050, n_fft=1024)
        mixer = Mixer(config)
        assert mixer.config.sample_rate == 22050
        assert mixer.config.n_fft == 1024


class TestMix:
    def test_output_shape_matches_input(
        self, mixer: Mixer, audio: np.ndarray, safe_noise: torch.Tensor
    ) -> None:
        """출력 길이는 입력과 동일."""
        modified = mixer.mix(audio, safe_noise)
        assert modified.shape == audio.shape

    def test_output_dtype(self, mixer: Mixer, audio: np.ndarray, safe_noise: torch.Tensor) -> None:
        """출력 dtype은 float32."""
        modified = mixer.mix(audio, safe_noise)
        assert modified.dtype == np.float32

    def test_output_range_clipped(self, mixer: Mixer, audio: np.ndarray) -> None:
        """과한 노이즈를 더해도 출력은 [-1, 1] 범위."""
        huge_noise = torch.randn(257, 100) * 100.0
        modified = mixer.mix(audio, huge_noise)
        assert modified.min() >= -1.0
        assert modified.max() <= 1.0

    def test_zero_noise_preserves_audio(self, mixer: Mixer, audio: np.ndarray) -> None:
        """노이즈가 0이면 출력은 원본과 거의 동일 (STFT round-trip 오차 내)."""
        zero_noise = torch.zeros(257, 100)
        modified = mixer.mix(audio, zero_noise)
        # STFT/iSTFT는 완벽한 round-trip이 아니라서 약간의 오차 허용
        # 양쪽 끝부분은 STFT padding 효과로 더 큰 오차 가능 → 중간 부분만 검증
        center = audio[1000:-1000]
        center_modified = modified[1000:-1000]
        assert np.allclose(center_modified, center, atol=0.05)


class TestNoiseAlignment:
    """safe_noise shape이 spectrogram과 안 맞을 때 처리."""

    def test_shorter_noise_padded(self, mixer: Mixer, audio: np.ndarray) -> None:
        """noise가 짧으면 padding되어 동작."""
        short_noise = torch.randn(257, 50) * 0.01  # 절반 길이
        modified = mixer.mix(audio, short_noise)
        assert modified.shape == audio.shape  # 정상 동작

    def test_longer_noise_truncated(self, mixer: Mixer, audio: np.ndarray) -> None:
        """noise가 길면 잘려서 동작."""
        long_noise = torch.randn(257, 200) * 0.01  # 두 배 길이
        modified = mixer.mix(audio, long_noise)
        assert modified.shape == audio.shape  # 정상 동작

    def test_wrong_freq_dim_raises(self, mixer: Mixer, audio: np.ndarray) -> None:
        """주파수 차원 불일치 시 ValueError."""
        wrong_noise = torch.randn(128, 100) * 0.01  # 잘못된 n_freq
        with pytest.raises(ValueError, match="n_freq mismatch"):
            mixer.mix(audio, wrong_noise)
