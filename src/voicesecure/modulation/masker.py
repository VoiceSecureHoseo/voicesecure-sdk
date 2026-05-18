"""Psychoacoustic masking module.

심리음향 마스킹 임계치 M(f, t)를 계산해서 RL Agent가 만든 raw noise를
사람 청각으로 안 들리는 범위 내로 clamp한다.

References
----------
- Zwicker & Fastl, "Psychoacoustics: Facts and Models" (1990)
- Terhardt et al., "Algorithm for extraction of pitch and pitch salience" (1982)
- ISO/IEC 11172-3 (MPEG-1 Audio) psychoacoustic model 1
- TUIlmenauAMS/Python-Audio-Coder (psyacmodel.py)

See Also
--------
- ARCHITECTURE.md section 3.1: Masker contract
- SRS.md FR-4: Psychoacoustic masking spec
- team_task_assignment.html: 한승규 담당 5-stage 구현
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from voicesecure.types import SAMPLE_RATE, AudioArray

# STFT parameters (ARCHITECTURE.md section 4)
N_FFT: int = 512
HOP_LENGTH: int = 160
WIN_LENGTH: int = 400


@dataclass
class MaskerConfig:
    """Psychoacoustic Masker hyperparameters."""

    sample_rate: int = SAMPLE_RATE
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    win_length: int = WIN_LENGTH
    n_bark_bands: int = 24
    spreading_slope_low: float = 27.0  # dB/Bark (upward masking)
    spreading_slope_high: float = -15.0  # dB/Bark (downward masking)


class PsychoacousticMasker:
    """RL Agent가 만든 raw noise를 사람 청각 임계치 이내로 clamp한다.

    Pipeline (team_task_assignment.html, 한승규 담당):
        Stage 1: 파워 스펙트럼 — STFT, P(f) [dB]
        Stage 2: Bark 변환 — Zwicker 공식, 24 임계대역 보간
        Stage 3: ATH (Absolute Threshold of Hearing) — Terhardt T_q(f)
        Stage 4: 동시 마스킹 — peak-finding, tonal/non-tonal, Schroeder spreading
        Stage 5: 최종 임계치 — M(f, t) = 10·log₁₀(10^(T_q/10) + 10^(T_m/10))

    Examples
    --------
    >>> masker = PsychoacousticMasker()
    >>> audio = np.random.randn(16000).astype(np.float32) * 0.1
    >>> threshold = masker.compute_threshold(audio)
    >>> raw_noise = torch.randn(257, 100)
    >>> safe_noise = masker.clamp(audio, raw_noise)
    >>> assert torch.all(torch.abs(safe_noise) <= torch.from_numpy(threshold))
    """

    def __init__(self, config: MaskerConfig | None = None) -> None:
        self.config = config or MaskerConfig()
        self._ath_cache: npt.NDArray[np.float32] | None = None
        self._bark_grid: npt.NDArray[np.float32] | None = None

    # ------------------------------------------------------------------
    # Stage 1: Power Spectrum
    # ------------------------------------------------------------------
    def compute_power_spectrum(self, audio: AudioArray) -> npt.NDArray[np.float32]:
        """STFT로 파워 스펙트럼 P(f, t) [dB] 계산.

        Parameters
        ----------
        audio
            shape (num_samples,), float32, normalized to [-1, 1]

        Returns
        -------
        P_db
            shape (n_freq, n_time), float32, dB scale
            n_freq = n_fft // 2 + 1 = 257
        """
        # TODO (한승규): scipy.signal.stft 또는 직접 구현
        # - Hann window, win_length=400, hop_length=160
        # - magnitude → power → dB (20 * log10)
        raise NotImplementedError("Stage 1: compute_power_spectrum not yet implemented")

    # ------------------------------------------------------------------
    # Stage 2: Bark Scale Conversion
    # ------------------------------------------------------------------
    @staticmethod
    def hz_to_bark(freq_hz: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Zwicker 공식으로 Hz → Bark 변환.

        bark = 13 * arctan(0.00076 * f) + 3.5 * arctan((f / 7500)²)
        """
        # TODO (한승규): Zwicker 공식 구현
        raise NotImplementedError("Stage 2: hz_to_bark not yet implemented")

    @staticmethod
    def bark_to_hz(bark: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Bark → Hz 역변환 (수치해석 또는 근사식)."""
        # TODO (한승규): 역변환 (lookup table 또는 inverse interpolation)
        raise NotImplementedError("Stage 2: bark_to_hz not yet implemented")

    def power_to_bark(self, power_db: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Hz 축의 파워 스펙트럼을 Bark 축 24개 대역으로 보간.

        Parameters
        ----------
        power_db
            shape (n_freq, n_time), dB

        Returns
        -------
        power_bark
            shape (n_bark_bands, n_time), dB
        """
        # TODO (한승규): scipy.interpolate로 보간
        raise NotImplementedError("Stage 2: power_to_bark not yet implemented")

    # ------------------------------------------------------------------
    # Stage 3: Absolute Threshold of Hearing (ATH)
    # ------------------------------------------------------------------
    @staticmethod
    def absolute_threshold(freq_hz: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Terhardt ATH 공식.

        T_q(f) = 3.64·(f/1000)^(-0.8)
               - 6.5·exp(-0.6·(f/1000 - 3.3)²)
               + 10^(-3)·(f/1000)^4   [dB SPL]
        """
        # TODO (한승규): Terhardt 공식 구현
        # 1 kHz에서 약 0 dB가 나와야 함
        raise NotImplementedError("Stage 3: absolute_threshold not yet implemented")

    def _build_ath_cache(self) -> npt.NDArray[np.float32]:
        """전 주파수 범위 ATH lookup table 캐싱."""
        if self._ath_cache is None:
            n_freq = self.config.n_fft // 2 + 1
            freqs = np.linspace(0, self.config.sample_rate / 2, n_freq, dtype=np.float32)
            self._ath_cache = self.absolute_threshold(freqs)
        return self._ath_cache

    # ------------------------------------------------------------------
    # Stage 4: Simultaneous Masking
    # ------------------------------------------------------------------
    def find_maskers(
        self, power_db: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
        """파워 스펙트럼에서 마스커(peak) 찾기.

        Returns
        -------
        peak_indices
            shape (n_peaks,)
        peak_levels
            shape (n_peaks,), dB
        """
        # TODO (한승규): scipy.signal.find_peaks 활용
        raise NotImplementedError("Stage 4: find_maskers not yet implemented")

    def classify_tonal(
        self, power_db: npt.NDArray[np.float32], peak_indices: npt.NDArray[np.int32]
    ) -> npt.NDArray[np.bool_]:
        """Peak prominence 기준 tonal/non-tonal 분류.

        ISO/IEC 11172-3 기준: 주변 ±2 bin 대비 7 dB 이상 높으면 tonal.
        """
        # TODO (한승규): tonal/non-tonal 분류
        raise NotImplementedError("Stage 4: classify_tonal not yet implemented")

    def spreading_function(
        self,
        masker_bark: float,
        target_bark: npt.NDArray[np.float32],
        masker_level_db: float,
    ) -> npt.NDArray[np.float32]:
        """Schroeder spreading function.

        Δz = target_bark - masker_bark
        spread(Δz) = 15.81 + 7.5·(Δz + 0.474) - 17.5·sqrt(1 + (Δz + 0.474)²)
        """
        # TODO (한승규): Schroeder spreading function 구현
        raise NotImplementedError("Stage 4: spreading_function not yet implemented")

    def compute_masking_threshold(
        self, power_db: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """동시 마스킹으로 마스킹 임계치 T_m(f, t) 계산.

        Stage 4 전체:
            1. find_maskers
            2. classify_tonal
            3. spreading_function 적용
            4. 마스커들 에너지 합산
        """
        # TODO (한승규): Stage 4 통합
        raise NotImplementedError("Stage 4: compute_masking_threshold not yet implemented")

    # ------------------------------------------------------------------
    # Stage 5: Final Threshold M(f, t)
    # ------------------------------------------------------------------
    def compute_threshold(self, audio: AudioArray) -> npt.NDArray[np.float32]:
        """최종 청각 임계치 M(f, t) 계산.

        M(f, t) = 10·log₁₀(10^(T_q(f)/10) + 10^(T_m(f, t)/10))

        Parameters
        ----------
        audio
            shape (num_samples,), float32, [-1, 1]

        Returns
        -------
        threshold
            shape (n_freq, n_time), float32, dB
            이 값보다 작은 노이즈는 사람에게 안 들림 (이론상)
        """
        # TODO (한승규): Stage 1~5 통합
        # power_db = self.compute_power_spectrum(audio)
        # T_q = self._build_ath_cache()  # shape (n_freq,)
        # T_m = self.compute_masking_threshold(power_db)  # shape (n_freq, n_time)
        # threshold = 10 * np.log10(
        #     10 ** (T_q[:, None] / 10) + 10 ** (T_m / 10)
        # )
        # return threshold.astype(np.float32)
        raise NotImplementedError("compute_threshold not yet implemented")

    # ------------------------------------------------------------------
    # Public clamp interface (used by RL pipeline)
    # ------------------------------------------------------------------
    def clamp(
        self,
        audio: AudioArray,
        raw_noise: torch.Tensor,
    ) -> torch.Tensor:
        """RL이 만든 raw noise를 청각 임계치 이내로 element-wise clamp.

        safe_noise[f, t] = sign(raw_noise[f, t]) * min(|raw_noise[f, t]|, M[f, t])

        Parameters
        ----------
        audio
            원본 음성, shape (num_samples,), float32
        raw_noise
            RL Agent 출력, shape (n_freq, n_time), float32

        Returns
        -------
        safe_noise
            shape (n_freq, n_time), 모든 element가 M(f, t) 이내 보장
        """
        # TODO (한승규): clamp 구현
        # threshold_db = self.compute_threshold(audio)
        # threshold_linear = 10 ** (threshold_db / 20)  # dB → linear
        # threshold_torch = torch.from_numpy(threshold_linear).to(raw_noise.device)
        # safe = torch.sign(raw_noise) * torch.minimum(
        #     torch.abs(raw_noise), threshold_torch
        # )
        # return safe
        raise NotImplementedError("clamp not yet implemented")
