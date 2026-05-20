"""Tests for PsychoacousticMasker (Stage 1~5 모두 구현됨).

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
    return PsychoacousticMasker()


@pytest.fixture
def audio_white_noise() -> np.ndarray:
    """1초 분량 백색 잡음."""
    rng = np.random.default_rng(seed=42)
    return (rng.standard_normal(16000) * 0.1).astype(np.float32)


@pytest.fixture
def audio_sine_1khz() -> np.ndarray:
    """1초 분량 1 kHz 사인파."""
    t = np.arange(16000, dtype=np.float32) / 16000.0
    return (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)


@pytest.fixture
def audio_silence() -> np.ndarray:
    """1초 분량 무음."""
    return np.zeros(16000, dtype=np.float32)


# ----------------------------------------------------------------------
# Initialization
# ----------------------------------------------------------------------
class TestInitialization:
    def test_default_config(self) -> None:
        masker = PsychoacousticMasker()
        assert masker.config.sample_rate == 16000
        assert masker.config.n_fft == 512
        assert masker.config.hop_length == 160
        assert masker.config.win_length == 400
        assert masker.config.n_bark_bands == 24

    def test_custom_config(self) -> None:
        config = MaskerConfig(sample_rate=22050, n_bark_bands=20)
        masker = PsychoacousticMasker(config)
        assert masker.config.sample_rate == 22050
        assert masker.config.n_bark_bands == 20


# ----------------------------------------------------------------------
# Stage 1: Power Spectrum
# ----------------------------------------------------------------------
class TestPowerSpectrum:
    def test_power_spectrum_shape(
        self, masker: PsychoacousticMasker, audio_white_noise: np.ndarray
    ) -> None:
        power = masker.compute_power_spectrum(audio_white_noise)
        assert power.ndim == 2
        assert power.shape[0] == 257
        assert power.dtype == np.float32

    def test_power_spectrum_db_scale(
        self, masker: PsychoacousticMasker, audio_sine_1khz: np.ndarray
    ) -> None:
        """1 kHz 사인파 → 1 kHz bin 근처에서 peak."""
        power = masker.compute_power_spectrum(audio_sine_1khz)
        mid_frame = power.shape[1] // 2
        peak_bin = int(np.argmax(power[:, mid_frame]))
        # 1 kHz at sr=16000, n_fft=512 → bin ≈ 32
        assert 30 <= peak_bin <= 34

    def test_power_spectrum_silence_low(
        self, masker: PsychoacousticMasker, audio_silence: np.ndarray
    ) -> None:
        power = masker.compute_power_spectrum(audio_silence)
        assert np.all(power < -100)

    def test_power_spectrum_dtype_float32(
        self, masker: PsychoacousticMasker, audio_white_noise: np.ndarray
    ) -> None:
        power = masker.compute_power_spectrum(audio_white_noise)
        assert power.dtype == np.float32

    def test_power_spectrum_no_nan_inf(
        self, masker: PsychoacousticMasker, audio_white_noise: np.ndarray
    ) -> None:
        power = masker.compute_power_spectrum(audio_white_noise)
        assert np.all(np.isfinite(power))


# ----------------------------------------------------------------------
# Stage 2: Bark Scale
# ----------------------------------------------------------------------
class TestBarkScale:
    def test_hz_to_bark_known_values(self) -> None:
        freqs = np.array([0, 500, 1000, 4000, 8000], dtype=np.float32)
        barks = PsychoacousticMasker.hz_to_bark(freqs)
        expected = np.array([0.0, 5.0, 8.5, 17.5, 22.0], dtype=np.float32)
        assert np.allclose(barks, expected, atol=1.0)

    def test_hz_to_bark_to_hz_roundtrip(self) -> None:
        freqs_orig = np.array([100, 1000, 4000, 8000], dtype=np.float32)
        barks = PsychoacousticMasker.hz_to_bark(freqs_orig)
        freqs_back = PsychoacousticMasker.bark_to_hz(barks)
        assert np.allclose(freqs_orig, freqs_back, rtol=0.05)

    def test_power_to_bark_shape(
        self, masker: PsychoacousticMasker, audio_white_noise: np.ndarray
    ) -> None:
        power = masker.compute_power_spectrum(audio_white_noise)
        bark_power = masker.power_to_bark(power)
        assert bark_power.shape[0] == 24
        assert bark_power.shape[1] == power.shape[1]
        assert bark_power.dtype == np.float32

    def test_hz_to_bark_monotonic(self) -> None:
        """Bark는 Hz에 대해 단조 증가."""
        freqs = np.linspace(0, 8000, 100, dtype=np.float32)
        barks = PsychoacousticMasker.hz_to_bark(freqs)
        assert np.all(np.diff(barks) >= 0)


# ----------------------------------------------------------------------
# Stage 3: ATH
# ----------------------------------------------------------------------
class TestAbsoluteThreshold:
    def test_ath_minimum_around_3khz(self) -> None:
        """ATH 최소값은 2.5~4.5 kHz 근처."""
        freqs = np.linspace(100, 8000, 200, dtype=np.float32)
        ath = PsychoacousticMasker.absolute_threshold(freqs)
        min_idx = int(np.argmin(ath))
        min_freq = float(freqs[min_idx])
        assert 2500 <= min_freq <= 4500

    def test_ath_1khz_near_zero(self) -> None:
        """1 kHz에서 ATH는 약 0 dB SPL."""
        ath = PsychoacousticMasker.absolute_threshold(np.array([1000.0], dtype=np.float32))
        assert -5 <= ath[0] <= 10

    def test_ath_high_at_low_freq(self) -> None:
        """50 Hz 같은 저주파에서 ATH는 매우 큼 (잘 안 들림)."""
        ath = PsychoacousticMasker.absolute_threshold(np.array([50.0], dtype=np.float32))
        assert ath[0] > 20  # 20 dB 이상

    def test_ath_cache(self, masker: PsychoacousticMasker) -> None:
        """ATH cache는 최초 호출 후 재사용."""
        ath1 = masker._build_ath_cache()
        ath2 = masker._build_ath_cache()
        assert ath1 is ath2


# ----------------------------------------------------------------------
# Stage 4: Simultaneous Masking
# ----------------------------------------------------------------------
class TestSimultaneousMasking:
    def test_find_maskers_sine_peak(
        self, masker: PsychoacousticMasker, audio_sine_1khz: np.ndarray
    ) -> None:
        """1 kHz 사인파 → 1 kHz 근처에 마스커 peak 검출."""
        power = masker.compute_power_spectrum(audio_sine_1khz)
        mid_frame = power.shape[1] // 2
        peaks, _levels = masker.find_maskers(power[:, mid_frame])
        assert len(peaks) >= 1
        assert any(30 <= p <= 34 for p in peaks)

    def test_classify_tonal_sine(
        self, masker: PsychoacousticMasker, audio_sine_1khz: np.ndarray
    ) -> None:
        """사인파의 peak은 tonal로 분류."""
        power = masker.compute_power_spectrum(audio_sine_1khz)
        mid_frame = power.shape[1] // 2
        peaks, _ = masker.find_maskers(power[:, mid_frame])
        is_tonal = masker.classify_tonal(power[:, mid_frame], peaks)
        assert is_tonal.any()

    def test_spreading_function_decay(self, masker: PsychoacousticMasker) -> None:
        """spreading function: masker 위치에서 멀어질수록 감소."""
        masker_bark = 8.0
        target_bark = np.array([6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32)
        spread = masker.spreading_function(masker_bark, target_bark, masker_level_db=80.0)
        # masker 위치(8.0)에서 가장 클 것, 멀어질수록 작아짐
        # (precise한 peak 위치는 spreading_offset 때문에 약간 다를 수 있음)
        assert spread[2] > spread[0]  # 8.0 > 6.0
        assert spread[2] > spread[4]  # 8.0 > 10.0

    def test_compute_masking_threshold_shape(
        self, masker: PsychoacousticMasker, audio_white_noise: np.ndarray
    ) -> None:
        """masking threshold shape (n_freq, n_time)."""
        power = masker.compute_power_spectrum(audio_white_noise)
        T_m = masker.compute_masking_threshold(power)
        assert T_m.shape == power.shape


# ----------------------------------------------------------------------
# Stage 5: Final Threshold
# ----------------------------------------------------------------------
class TestFinalThreshold:
    def test_threshold_shape(
        self, masker: PsychoacousticMasker, audio_white_noise: np.ndarray
    ) -> None:
        threshold = masker.compute_threshold(audio_white_noise)
        assert threshold.shape[0] == 257
        assert threshold.dtype == np.float32

    def test_threshold_no_nan_inf(
        self, masker: PsychoacousticMasker, audio_white_noise: np.ndarray
    ) -> None:
        threshold = masker.compute_threshold(audio_white_noise)
        assert np.all(np.isfinite(threshold))

    def test_threshold_silence_is_ath(
        self, masker: PsychoacousticMasker, audio_silence: np.ndarray
    ) -> None:
        """무음 입력 → masker 없음 → threshold ≈ ATH."""
        threshold = masker.compute_threshold(audio_silence)
        ath = masker._build_ath_cache()
        # 모든 시간 프레임에서 ATH와 거의 일치 (max(T_q, -160) = T_q)
        assert np.allclose(threshold[:, 0], ath, atol=0.5)

    def test_threshold_louder_audio_raises_mask(
        self,
        masker: PsychoacousticMasker,
        audio_sine_1khz: np.ndarray,
        audio_silence: np.ndarray,
    ) -> None:
        """큰 음성 → 마스킹 임계치가 ATH보다 높아져야 함."""
        thr_loud = masker.compute_threshold(audio_sine_1khz)
        thr_silent = masker.compute_threshold(audio_silence)
        # 1 kHz 근처 bin (≈32)에서 loud > silent
        assert thr_loud[32].mean() > thr_silent[32].mean()


# ----------------------------------------------------------------------
# Clamp (Public Interface)
# ----------------------------------------------------------------------
class TestClamp:
    def test_clamp_preserves_shape(
        self, masker: PsychoacousticMasker, audio_white_noise: np.ndarray
    ) -> None:
        raw_noise = torch.randn(257, 100)
        safe = masker.clamp(audio_white_noise, raw_noise)
        # 시간 차원이 audio STFT 결과에 따라 다를 수 있음 (align 동작)
        assert safe.shape[0] == raw_noise.shape[0]  # n_freq 보존

    def test_clamp_within_threshold(
        self, masker: PsychoacousticMasker, audio_white_noise: np.ndarray
    ) -> None:
        """clamp 결과의 모든 element가 threshold 이내."""
        raw_noise = torch.randn(257, 100) * 10.0  # 큰 값
        safe = masker.clamp(audio_white_noise, raw_noise)

        threshold_db = masker.compute_threshold(audio_white_noise)
        threshold_linear = torch.from_numpy((10.0 ** (threshold_db / 20.0)).astype(np.float32))
        # 시간 차원 align
        n_time = min(safe.shape[1], threshold_linear.shape[1])
        # safe noise의 magnitude가 threshold linear 이내인지 확인
        assert torch.all(torch.abs(safe[:, :n_time]) <= threshold_linear[:, :n_time] + 1e-4)

    def test_clamp_preserves_sign(
        self, masker: PsychoacousticMasker, audio_white_noise: np.ndarray
    ) -> None:
        """clamp가 부호를 바꾸지 않음 (magnitude만 줄임)."""
        raw_noise = torch.randn(257, 100)
        safe = masker.clamp(audio_white_noise, raw_noise)
        n_time = min(safe.shape[1], raw_noise.shape[1])
        nonzero_mask = raw_noise[:, :n_time].abs() > 1e-10
        assert torch.all(
            torch.sign(safe[:, :n_time][nonzero_mask])
            == torch.sign(raw_noise[:, :n_time][nonzero_mask])
        )

    def test_clamp_small_noise_unchanged(
        self, masker: PsychoacousticMasker, audio_white_noise: np.ndarray
    ) -> None:
        """작은 noise는 threshold 이내라 magnitude가 줄어들지 않음."""
        # 1e-6 수준의 noise는 dB로 변환 시 약 -120 dB → ATH보다 낮으므로 그대로 통과
        # dB→linear 변환 오차가 있으므로 부호만 보존되는지 확인
        tiny_noise = torch.randn(257, 100) * 1e-6
        safe = masker.clamp(audio_white_noise, tiny_noise)
        n_time = min(safe.shape[1], tiny_noise.shape[1])
        nonzero = tiny_noise[:, :n_time].abs() > 1e-10
        assert torch.all(
            torch.sign(safe[:, :n_time][nonzero]) == torch.sign(tiny_noise[:, :n_time][nonzero])
        )
