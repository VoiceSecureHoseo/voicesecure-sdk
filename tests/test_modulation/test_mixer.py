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

from voicesecure.modulation.mixer import Mixer


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


class TestMix:
    @pytest.mark.skip(reason="Mixer.mix not yet implemented")
    def test_output_shape_matches_input(
        self, mixer: Mixer, audio: np.ndarray, safe_noise: torch.Tensor
    ) -> None:
        """출력 길이는 입력과 동일."""
        modified = mixer.mix(audio, safe_noise)
        assert modified.shape == audio.shape

    @pytest.mark.skip(reason="Mixer.mix not yet implemented")
    def test_output_dtype(self, mixer: Mixer, audio: np.ndarray, safe_noise: torch.Tensor) -> None:
        """출력 dtype은 float32."""
        modified = mixer.mix(audio, safe_noise)
        assert modified.dtype == np.float32

    @pytest.mark.skip(reason="Mixer.mix not yet implemented")
    def test_output_range_clipped(self, mixer: Mixer, audio: np.ndarray) -> None:
        """과한 노이즈를 더해도 출력은 [-1, 1] 범위."""
        huge_noise = torch.randn(257, 100) * 100.0
        modified = mixer.mix(audio, huge_noise)
        assert modified.min() >= -1.0
        assert modified.max() <= 1.0

    @pytest.mark.skip(reason="Mixer.mix not yet implemented")
    def test_zero_noise_preserves_audio(self, mixer: Mixer, audio: np.ndarray) -> None:
        """노이즈가 0이면 출력은 원본과 거의 동일 (STFT round-trip 오차 내)."""
        zero_noise = torch.zeros(257, 100)
        modified = mixer.mix(audio, zero_noise)
        # STFT/iSTFT는 완벽한 round-trip이 아니라서 약간의 오차 허용
        assert np.allclose(modified, audio, atol=0.01)
