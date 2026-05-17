"""Tests for PsychoacousticMasker.

See Also
--------
- src/voicesecure/modulation/masker.py
- ARCHITECTURE.md section 3.1
- SRS.md FR-4
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from voicesecure.modulation.masker import MaskerConfig, PsychoacousticMasker


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def masker() -> PsychoacousticMasker:
    """기본 설정 Masker 인스턴스."""
    return PsychoacousticMasker()


@pytest.fixture
def audio_white_noise() -> np.ndarray:
    """1초 분량 백색 잡음."""
    rng = np.random.default_rng(seed=42)
    return (rng.standard_normal(16000) * 0.1).astype(np.float32)


@pytest.fixture
def audio_sine_1khz() -> np.ndarray:
    """1초 분량 1 kHz 사인파 (80 dB SPL 가정)."""
    t = np.arange(16000, dtype=np.float32) / 16000.0
    return (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)


@pytest.fixture
def audio_silence() -> np.ndarray:
    """1초 분량 무음."""
    return np.zeros(16000, dtype=np.float32)


# ----------------------------------------------------------------------
# Initialization tests
# ----------------------------------------------------------------------
class TestInitialization:
    def test_default_config(self) -> None:
        """기본 config로 초기화 정상."""
        masker = PsychoacousticMasker()
        assert masker.config.sample_rate == 16000
        assert masker.config.n_fft == 512
        assert masker.config.hop_length == 160
        assert masker.config.win_length == 400
        assert masker.config.n_bark_bands == 24

    def test_custom_config(self) -> None:
        """custom config 적용 확인."""
        config = MaskerConfig(sample_rate=22050, n_bark_bands=20)
        masker = PsychoacousticMasker(config)
        assert masker.config.sample_rate == 22050
        assert masker.config.n_bark_bands == 20


# ----------------------------------------------------------------------
# Stage 1: Power Spectrum
# ----------------------------------------------------------------------
class TestPowerSpectrum:
    @pytest.mark.skip(reason="Stage 1 not yet implemented")
    def test_power_spectrum_shape(
        self, masker: PsychoacousticMasker, audio_white_noise: np.ndarray
    ) -> None:
        """STFT 출력 shape 검증.

        n_freq = n_fft // 2 + 1 = 257
        n_time = (16000 - win_length) // hop_length + 1
        """
        power = masker.compute_power_spectrum(audio_white_noise)
        assert power.ndim == 2
        assert power.shape[0] == 257  # n_fft // 2 + 1
        assert power.dtype == np.float32

    @pytest.mark.skip(reason="Stage 1 not yet implemented")
    def test_power_spectrum_db_scale(
        self, masker: PsychoacousticMasker, audio_sine_1khz: np.ndarray
    ) -> None:
        """1 kHz 사인파의 파워가 1 kHz bin에서 peak."""
        power = masker.compute_power_spectrum(audio_sine_1khz)
        peak_bin = int(np.argmax(power[:, 50]))  # mid-frame
        # 1 kHz at sr=16000, n_fft=512 → bin ≈ 32
        assert 30 <= peak_bin <= 34


# ----------------------------------------------------------------------
# Stage 2: Bark Scale
# ----------------------------------------------------------------------
class TestBarkScale:
    @pytest.mark.skip(reason="Stage 2 not yet implemented")
    def test_hz_to_bark_known_values(self) -> None:
        """Zwicker 공식 알려진 값 검증.

        - 0 Hz → ~0 Bark
        - 500 Hz → ~5 Bark
        - 1 kHz → ~8.5 Bark
        - 4 kHz → ~17.5 Bark
        - 8 kHz → ~22 Bark
        """
        freqs = np.array([0, 500, 1000, 4000, 8000], dtype=np.float32)
        barks = PsychoacousticMasker.hz_to_bark(freqs)
        expected = np.array([0.0, 5.0, 8.5, 17.5, 22.0], dtype=np.float32)
        assert np.allclose(barks, expected, atol=1.0)

    @pytest.mark.skip(reason="Stage 2 not yet implemented")
    def test_hz_to_bark_to_hz_roundtrip(self) -> None:
        """Hz → Bark → Hz 변환 라운드트립."""
        freqs_orig = np.array([100, 1000, 4000, 8000], dtype=np.float32)
        barks = PsychoacousticMasker.hz_to_bark(freqs_orig)
        freqs_back = PsychoacousticMasker.bark_to_hz(barks)
        assert np.allclose(freqs_orig, freqs_back, rtol=0.05)

    @pytest.mark.skip(reason="Stage 2 not yet implemented")
    def test_power_to_bark_shape(
        self, masker: PsychoacousticMasker, audio_white_noise: np.ndarray
    ) -> None:
        """Bark 변환 후 shape (24, n_time)."""
        power = masker.compute_power_spectrum(audio_white_noise)
        bark_power = masker.power_to_bark(power)
        assert bark_power.shape[0] == 24
        assert bark_power.shape[1] == power.shape[1]


# ----------------------------------------------------------------------
# Stage 3: ATH
# ----------------------------------------------------------------------
class TestAbsoluteThreshold:
    @pytest.mark.skip(reason="Stage 3 not yet implemented")
    def test_ath_minimum_at_3khz(self) -> None:
        """ATH는 3-4 kHz 근처에서 최소 (사람 청각이 가장 민감)."""
        freqs = np.linspace(100, 8000, 200, dtype=np.float32)
        ath = PsychoacousticMasker.absolute_threshold(freqs)
        min_idx = int(np.argmin(ath))
        min_freq = float(freqs[min_idx])
        assert 2500 <= min_freq <= 4500

    @pytest.mark.skip(reason="Stage 3 not yet implemented")
    def test_ath_1khz_near_zero(self) -> None:
        """1 kHz에서 ATH는 약 0 dB SPL."""
        ath = PsychoacousticMasker.absolute_threshold(np.array([1000.0], dtype=np.float32))
        assert -5 <= ath[0] <= 10

    @pytest.mark.skip(reason="Stage 3 not yet implemented")
    def test_ath_cache(self, masker: PsychoacousticMasker) -> None:
        """ATH cache는 최초 호출 후 재사용."""
        ath1 = masker._build_ath_cache()
        ath2 = masker._build_ath_cache()
        assert ath1 is ath2  # 같은 객체 (캐싱됨)


# ----------------------------------------------------------------------
# Stage 4: Simultaneous Masking
# ----------------------------------------------------------------------
class TestSimultaneousMasking:
    @pytest.mark.skip(reason="Stage 4 not yet implemented")
    def test_find_maskers_sine_peak(
        self, masker: PsychoacousticMasker, audio_sine_1khz: np.ndarray
    ) -> None:
        """1 kHz 사인파에서 마스커 1개 검출 (또는 그 근처)."""
        power = masker.compute_power_spectrum(audio_sine_1khz)
        peaks, _levels = masker.find_maskers(power[:, 50])
        assert len(peaks) >= 1
        # 1 kHz bin (≈32) 근처에 peak 있는지
        assert any(30 <= p <= 34 for p in peaks)

    @pytest.mark.skip(reason="Stage 4 not yet implemented")
    def test_classify_tonal_sine(
        self, masker: PsychoacousticMasker, audio_sine_1khz: np.ndarray
    ) -> None:
        """사인파의 peak은 tonal로 분류."""
        power = masker.compute_power_spectrum(audio_sine_1khz)
        peaks, _ = masker.find_maskers(power[:, 50])
        is_tonal = masker.classify_tonal(power[:, 50], peaks)
        assert is_tonal.any()


# ----------------------------------------------------------------------
# Stage 5: Final Threshold M(f, t)
# ----------------------------------------------------------------------
class TestFinalThreshold:
    @pytest.mark.skip(reason="Stage 5 not yet implemented")
    def test_threshold_shape(
        self, masker: PsychoacousticMasker, audio_white_noise: np.ndarray
    ) -> None:
        """M(f, t) shape (n_freq=257, n_time)."""
        threshold = masker.compute_threshold(audio_white_noise)
        assert threshold.shape[0] == 257
        assert threshold.dtype == np.float32

    @pytest.mark.skip(reason="Stage 5 not yet implemented")
    def test_threshold_silence_is_ath(
        self, masker: PsychoacousticMasker, audio_silence: np.ndarray
    ) -> None:
        """무음 입력은 마스커가 없으므로 threshold ≈ ATH."""
        threshold = masker.compute_threshold(audio_silence)
        ath = masker._build_ath_cache()
        # 모든 시간 프레임에서 ATH와 거의 일치
        assert np.allclose(threshold[:, 0], ath, atol=1.0)

    @pytest.mark.skip(reason="Stage 5 not yet implemented")
    def test_threshold_louder_audio_raises_mask(
        self,
        masker: PsychoacousticMasker,
        audio_sine_1khz: np.ndarray,
        audio_silence: np.ndarray,
    ) -> None:
        """큰 음성 입력 시 마스킹 임계치가 ATH보다 높아야 함."""
        thr_loud = masker.compute_threshold(audio_sine_1khz)
        thr_silent = masker.compute_threshold(audio_silence)
        # 1 kHz 근처 bin (≈32)에서 loud > silent
        assert thr_loud[32].mean() > thr_silent[32].mean()


# ----------------------------------------------------------------------
# Clamp interface (the main public API used by RL pipeline)
# ----------------------------------------------------------------------
class TestClamp:
    @pytest.mark.skip(reason="clamp not yet implemented")
    def test_clamp_within_threshold(
        self, masker: PsychoacousticMasker, audio_white_noise: np.ndarray
    ) -> None:
        """clamp 결과의 모든 element가 threshold 이내."""
        # raw_noise는 의도적으로 큰 값
        n_freq = masker.config.n_fft // 2 + 1
        raw_noise = torch.randn(n_freq, 100) * 10.0

        safe = masker.clamp(audio_white_noise, raw_noise)

        threshold_db = masker.compute_threshold(audio_white_noise)
        # threshold_db shape이 raw_noise와 맞을 때만 비교 가능
        threshold_linear = torch.from_numpy((10 ** (threshold_db / 20)).astype(np.float32))
        # 시간 차원 align 필요 — 일단 첫 N 프레임만
        n_time = min(safe.shape[1], threshold_linear.shape[1])
        assert torch.all(torch.abs(safe[:, :n_time]) <= threshold_linear[:, :n_time] + 1e-5)

    @pytest.mark.skip(reason="clamp not yet implemented")
    def test_clamp_preserves_shape(
        self, masker: PsychoacousticMasker, audio_white_noise: np.ndarray
    ) -> None:
        """clamp 후 shape 유지."""
        raw_noise = torch.randn(257, 100)
        safe = masker.clamp(audio_white_noise, raw_noise)
        assert safe.shape == raw_noise.shape

    @pytest.mark.skip(reason="clamp not yet implemented")
    def test_clamp_preserves_sign(
        self, masker: PsychoacousticMasker, audio_white_noise: np.ndarray
    ) -> None:
        """clamp가 부호를 바꾸지 않음 (magnitude만 줄임)."""
        raw_noise = torch.randn(257, 100)
        safe = masker.clamp(audio_white_noise, raw_noise)
        # 부호 동일성: nonzero 위치에서
        nonzero_mask = raw_noise.abs() > 1e-10
        assert torch.all(torch.sign(safe[nonzero_mask]) == torch.sign(raw_noise[nonzero_mask]))
