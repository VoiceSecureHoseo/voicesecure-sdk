"""Psychoacoustic masking module.

심리음향 마스킹 임계치 M(f, t)를 계산해서 RL Agent가 만든 raw noise를
사람 청각으로 안 들리는 범위 내로 clamp한다.

5-stage 신호처리 (ISO/IEC 11172-3 Psychoacoustic Model 1 기반):
    Stage 1: Power Spectrum (STFT → dB)
    Stage 2: Bark Scale conversion (Zwicker 공식)
    Stage 3: ATH — Absolute Threshold of Hearing (Terhardt 공식)
    Stage 4: Simultaneous Masking (peak finding + Schroeder spreading)
    Stage 5: Final Threshold M(f, t) = max(T_q, T_m)

References
----------
- Zwicker & Fastl, "Psychoacoustics: Facts and Models" (1990)
- Terhardt et al., "Algorithm for extraction of pitch and pitch salience" (1982)
- ISO/IEC 11172-3 (MPEG-1 Audio) psychoacoustic model 1
- Schroeder et al., "Optimizing digital speech coders by exploiting masking" (1979)

See Also
--------
- ARCHITECTURE.md section 3.1: Masker contract
- SRS.md FR-4: Psychoacoustic masking spec
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from scipy.signal import find_peaks

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
    # Peak detection
    peak_min_height_db: float = -40.0  # 이 dB 이하 peak은 무시
    peak_min_distance_bins: int = 3  # 최소 peak 간격 (bin)
    # Schroeder spreading function 계수
    spreading_offset: float = 0.474  # Schroeder 표준 offset


class PsychoacousticMasker:
    """RL Agent가 만든 raw noise를 사람 청각 임계치 이내로 clamp한다.

    Pipeline (5-stage):
        Stage 1: 파워 스펙트럼 — STFT, P(f, t) [dB]
        Stage 2: Bark 변환 — Zwicker 공식
        Stage 3: ATH — Terhardt T_q(f)
        Stage 4: 동시 마스킹 — Schroeder spreading
        Stage 5: 최종 임계치 — M(f, t) = max(T_q, T_m)

    Examples
    --------
    >>> masker = PsychoacousticMasker()
    >>> audio = np.random.randn(16000).astype(np.float32) * 0.1
    >>> threshold = masker.compute_threshold(audio)
    >>> assert threshold.shape[0] == 257
    >>> raw_noise = torch.randn(257, 100) * 10.0
    >>> safe_noise = masker.clamp(audio, raw_noise)
    """

    def __init__(self, config: MaskerConfig | None = None) -> None:
        self.config = config or MaskerConfig()
        self._ath_cache: npt.NDArray[np.float32] | None = None
        # Hann window 캐싱
        self._window = torch.hann_window(self.config.win_length)

    # ------------------------------------------------------------------
    # Stage 1: Power Spectrum
    # ------------------------------------------------------------------
    def compute_power_spectrum(self, audio: AudioArray) -> npt.NDArray[np.float32]:
        """STFT로 파워 스펙트럼 P(f, t) [dB] 계산.

        Pipeline:
            1. STFT (Hann window, n_fft=512, hop=160, win=400)
            2. |spec|² → power
            3. 10·log₁₀(power + epsilon) → dB

        Parameters
        ----------
        audio
            shape (num_samples,), float32, [-1, 1]

        Returns
        -------
        P_db
            shape (n_freq, n_time), float32, dB scale.
            n_freq = n_fft // 2 + 1 = 257.
        """
        audio_torch = torch.from_numpy(audio).float()
        spec = torch.stft(
            audio_torch,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self._window,
            return_complex=True,
        )
        magnitude = torch.abs(spec)
        power = magnitude**2
        # log(0) 방지 epsilon (-160 dB)
        epsilon = 1e-16
        power_db = 10.0 * torch.log10(power + epsilon)
        return power_db.numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Stage 2: Bark Scale Conversion
    # ------------------------------------------------------------------
    @staticmethod
    def hz_to_bark(freq_hz: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Zwicker 공식: Hz → Bark.

        bark = 13·arctan(0.00076·f) + 3.5·arctan((f/7500)²)
        """
        freq_hz = np.asarray(freq_hz, dtype=np.float32)
        bark = 13.0 * np.arctan(0.00076 * freq_hz) + 3.5 * np.arctan((freq_hz / 7500.0) ** 2)
        return bark.astype(np.float32)

    @staticmethod
    def bark_to_hz(bark: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Bark → Hz 역변환 (numerical, via lookup interpolation)."""
        bark = np.asarray(bark, dtype=np.float32)
        hz_grid = np.linspace(0, 20000, 40001, dtype=np.float32)
        bark_grid = PsychoacousticMasker.hz_to_bark(hz_grid)
        freq_hz = np.interp(bark, bark_grid, hz_grid)
        return freq_hz.astype(np.float32)

    def power_to_bark(self, power_db: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Hz 축 power spectrum을 Bark 24개 대역으로 그룹화 (dB 평균)."""
        n_freq, n_time = power_db.shape
        n_bands = self.config.n_bark_bands

        freqs_hz = np.linspace(0, self.config.sample_rate / 2, n_freq, dtype=np.float32)
        freqs_bark = self.hz_to_bark(freqs_hz)
        max_bark = freqs_bark[-1]
        band_edges = np.linspace(0, max_bark, n_bands + 1, dtype=np.float32)

        power_bark = np.zeros((n_bands, n_time), dtype=np.float32)
        for band_idx in range(n_bands):
            lo, hi = band_edges[band_idx], band_edges[band_idx + 1]
            mask = (freqs_bark >= lo) & (freqs_bark < hi)
            if band_idx == n_bands - 1:
                mask = (freqs_bark >= lo) & (freqs_bark <= hi)
            if not mask.any():
                power_bark[band_idx, :] = -160.0
                continue
            power_bark[band_idx, :] = power_db[mask, :].mean(axis=0)
        return power_bark

    # ------------------------------------------------------------------
    # Stage 3: Absolute Threshold of Hearing (ATH)
    # ------------------------------------------------------------------
    @staticmethod
    def absolute_threshold(
        freq_hz: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Terhardt ATH 공식.

        T_q(f) = 3.64·(f/1000)^(-0.8)
               - 6.5·exp(-0.6·(f/1000 - 3.3)²)
               + 10^(-3)·(f/1000)^4   [dB SPL]

        - 3 kHz 근처에서 최소 (사람 청각이 가장 민감)
        - 0 Hz / 8 kHz 부근에서 매우 큼 (잘 안 들림)
        """
        freq_hz = np.asarray(freq_hz, dtype=np.float32)
        # 0 Hz 처리 (pow(-0.8) → inf 방지)
        freq_safe = np.maximum(freq_hz, 20.0)  # 사람 청각 한계 ~20 Hz
        f_khz = freq_safe / 1000.0
        T_q = 3.64 * f_khz**-0.8 - 6.5 * np.exp(-0.6 * (f_khz - 3.3) ** 2) + 1e-3 * f_khz**4
        return T_q.astype(np.float32)

    def _build_ath_cache(self) -> npt.NDArray[np.float32]:
        """전 주파수 ATH lookup table 캐싱."""
        if self._ath_cache is None:
            n_freq = self.config.n_fft // 2 + 1
            freqs = np.linspace(0, self.config.sample_rate / 2, n_freq, dtype=np.float32)
            self._ath_cache = self.absolute_threshold(freqs)
        return self._ath_cache

    # ------------------------------------------------------------------
    # Stage 4: Simultaneous Masking
    # ------------------------------------------------------------------
    def find_maskers(
        self, power_db_frame: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
        """한 시간 프레임의 파워 스펙트럼에서 peak (masker) 찾기.

        Parameters
        ----------
        power_db_frame
            shape (n_freq,), dB

        Returns
        -------
        peak_indices
            shape (n_peaks,), int32
        peak_levels
            shape (n_peaks,), float32, dB
        """
        peaks, _props = find_peaks(
            power_db_frame,
            height=self.config.peak_min_height_db,
            distance=self.config.peak_min_distance_bins,
        )
        peaks = peaks.astype(np.int32)
        levels = power_db_frame[peaks].astype(np.float32)
        return peaks, levels

    def classify_tonal(
        self,
        power_db_frame: npt.NDArray[np.float32],
        peak_indices: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.bool_]:
        """Tonal/non-tonal 분류 (ISO/IEC 11172-3 단순화 버전).

        Tonal 기준: 주변 ±5 bin (STFT 윈도우 leakage 고려) 대비 5 dB 이상 높음.

        ISO 표준의 ±2 bin / 7 dB 기준은 PSD 기반이고, 우리는 STFT 사용이라
        Hann window leakage 때문에 사인파 peak도 인접 bin이 비슷한 dB.
        그래서 더 넓은 범위 (±5)를 보고 임계값도 완화 (5 dB).
        """
        is_tonal = np.zeros(len(peak_indices), dtype=np.bool_)
        n_freq = len(power_db_frame)
        window_radius = 5  # ±5 bin (window leakage 고려)
        tonal_threshold_db = 5.0  # 주변 대비 차이 임계
        for i, peak in enumerate(peak_indices):
            lo = max(0, peak - window_radius)
            hi = min(n_freq, peak + window_radius + 1)
            # peak 자체 제외한 주변 bin
            neighbors = np.concatenate([power_db_frame[lo:peak], power_db_frame[peak + 1 : hi]])
            if len(neighbors) == 0:
                continue
            # 가장 큰 주변 값 대비 차이 (leakage 영향 받는 인접 bin들 중 max로 비교)
            # 실제로는 인접 ±2 bin은 거의 같으니, 더 멀리 (±3~5) 대비 차이를 봐야 함
            # 그래서 평균 대비 차이로 변경 (또는 멀리 있는 bin과 비교)
            if power_db_frame[peak] - neighbors.mean() >= tonal_threshold_db:
                is_tonal[i] = True
        return is_tonal

    def spreading_function(
        self,
        masker_bark: float,
        target_bark: npt.NDArray[np.float32],
        masker_level_db: float,
    ) -> npt.NDArray[np.float32]:
        """Schroeder spreading function.

        Δz = target - masker (Bark 차이)
        spread(Δz) = 15.81 + 7.5·(Δz + 0.474) - 17.5·sqrt(1 + (Δz + 0.474)²)   [dB]

        masker 위치에서 멀어질수록 마스킹 효과 감소.
        """
        target_bark = np.asarray(target_bark, dtype=np.float32)
        delta_z = target_bark - masker_bark + self.config.spreading_offset
        spread = 15.81 + 7.5 * delta_z - 17.5 * np.sqrt(1.0 + delta_z**2)
        # masker level만큼 더해서 절대 masking threshold 반환
        return (masker_level_db + spread).astype(np.float32)

    def compute_masking_threshold(
        self, power_db: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """동시 마스킹 임계치 T_m(f, t) 계산.

        매 시간 프레임마다:
            1. find peaks (maskers)
            2. classify tonal/non-tonal (tonal은 그대로, non-tonal은 -5 dB penalty)
            3. 각 masker의 spreading function 적용
            4. 모든 masker 효과를 dB 도메인 max로 합성

        Parameters
        ----------
        power_db
            shape (n_freq, n_time), dB

        Returns
        -------
        T_m
            shape (n_freq, n_time), dB. 마스킹 임계치.
            마스커 없으면 매우 낮음 (-160 dB).
        """
        n_freq, n_time = power_db.shape

        # Hz bin 중심 주파수를 Bark로 변환 (미리 계산)
        freqs_hz = np.linspace(0, self.config.sample_rate / 2, n_freq, dtype=np.float32)
        freqs_bark = self.hz_to_bark(freqs_hz)

        T_m = np.full((n_freq, n_time), -160.0, dtype=np.float32)

        for t in range(n_time):
            frame = power_db[:, t]
            peaks, levels = self.find_maskers(frame)
            if len(peaks) == 0:
                continue
            is_tonal = self.classify_tonal(frame, peaks)
            for peak_idx, level, tonal in zip(peaks, levels, is_tonal, strict=True):
                # Non-tonal은 5 dB penalty (덜 효과적 masker)
                effective_level = level if tonal else level - 5.0
                masker_bark = freqs_bark[peak_idx]
                spread = self.spreading_function(masker_bark, freqs_bark, effective_level)
                # 모든 masker 효과의 max (dB 도메인 — 약간 보수적이지만 단순)
                T_m[:, t] = np.maximum(T_m[:, t], spread)

        return T_m

    # ------------------------------------------------------------------
    # Stage 5: Final Threshold M(f, t)
    # ------------------------------------------------------------------
    def compute_threshold(self, audio: AudioArray) -> npt.NDArray[np.float32]:
        """최종 청각 임계치 M(f, t) [dB] 계산.

        M(f, t) = max(T_q(f), T_m(f, t))

        - T_q(f): ATH (절대 청각 임계치, 시간 무관)
        - T_m(f, t): 동시 마스킹 임계치 (시간 의존)

        이 값보다 작은 노이즈는 사람에게 안 들림 (이론상).

        Parameters
        ----------
        audio
            shape (num_samples,), float32, [-1, 1]

        Returns
        -------
        threshold
            shape (n_freq, n_time), float32, dB
        """
        # Stage 1: power spectrum
        power_db = self.compute_power_spectrum(audio)
        # Stage 3: ATH (시간 무관) — broadcasting을 위해 (n_freq, 1)
        T_q = self._build_ath_cache()[:, None]  # (n_freq, 1)
        # Stage 4: simultaneous masking (시간 의존)
        T_m = self.compute_masking_threshold(power_db)  # (n_freq, n_time)
        # Stage 5: 최종 = max(T_q, T_m)
        threshold = np.maximum(T_q, T_m)
        return threshold.astype(np.float32)

    # ------------------------------------------------------------------
    # Public clamp interface
    # ------------------------------------------------------------------
    def clamp(
        self,
        audio: AudioArray,
        raw_noise: torch.Tensor,
    ) -> torch.Tensor:
        """RL이 만든 raw noise를 청각 임계치 이내로 element-wise clamp.

        safe_noise[f, t] = sign(raw_noise[f, t]) * min(|raw_noise[f, t]|, threshold[f, t])

        Parameters
        ----------
        audio
            원본 음성, shape (num_samples,)
        raw_noise
            RL Agent 출력, shape (n_freq, n_time), float32

        Returns
        -------
        safe_noise
            shape (n_freq, n_time), 모든 element가 threshold 이내 보장
        """
        # dB threshold 계산
        threshold_db = self.compute_threshold(audio)  # (n_freq, n_time_audio)
        # dB → linear amplitude (10^(dB/20))
        threshold_linear = (10.0 ** (threshold_db / 20.0)).astype(np.float32)
        threshold_torch = torch.from_numpy(threshold_linear)

        # raw_noise와 threshold shape align (시간 차원)
        n_freq_noise, n_time_noise = raw_noise.shape
        n_freq_thr, n_time_thr = threshold_torch.shape
        if n_freq_noise != n_freq_thr:
            raise ValueError(f"n_freq mismatch: noise={n_freq_noise}, threshold={n_freq_thr}")
        # 시간 align (mixer와 동일 전략)
        if n_time_noise > n_time_thr:
            raw_noise = raw_noise[:, :n_time_thr]
        elif n_time_noise < n_time_thr:
            threshold_torch = threshold_torch[:, :n_time_noise]

        # element-wise clamp (부호 유지)
        sign = torch.sign(raw_noise)
        magnitude = torch.abs(raw_noise)
        safe_magnitude = torch.minimum(magnitude, threshold_torch)
        safe_noise = sign * safe_magnitude
        return safe_noise
