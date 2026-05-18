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
- team_task_assignment.html: 5-stage 구현 가이드
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

    Pipeline (5-stage 신호처리):
        Stage 1: 파워 스펙트럼 — STFT, P(f, t) [dB]   ✅ 구현됨
        Stage 2: Bark 변환 — Zwicker 공식, 24 임계대역 보간
        Stage 3: ATH (Absolute Threshold of Hearing) — Terhardt T_q(f)
        Stage 4: 동시 마스킹 — peak-finding, tonal/non-tonal, Schroeder spreading
        Stage 5: 최종 임계치 — M(f, t) = 10·log₁₀(10^(T_q/10) + 10^(T_m/10))

    Examples
    --------
    >>> masker = PsychoacousticMasker()
    >>> audio = np.random.randn(16000).astype(np.float32) * 0.1
    >>> power_db = masker.compute_power_spectrum(audio)
    >>> assert power_db.shape[0] == 257
    """

    def __init__(self, config: MaskerConfig | None = None) -> None:
        self.config = config or MaskerConfig()
        self._ath_cache: npt.NDArray[np.float32] | None = None
        self._bark_grid: npt.NDArray[np.float32] | None = None
        # Hann window 캐싱 (Mixer와 동일 패턴)
        self._window = torch.hann_window(self.config.win_length)

    # ------------------------------------------------------------------
    # Stage 1: Power Spectrum (✅ 구현됨)
    # ------------------------------------------------------------------
    def compute_power_spectrum(self, audio: AudioArray) -> npt.NDArray[np.float32]:
        """STFT로 파워 스펙트럼 P(f, t) [dB] 계산.

        Pipeline:
            1. STFT (Hann window, n_fft=512, hop=160, win=400)
            2. |spec|² → 파워 (에너지)
            3. 10·log₁₀(power) → dB scale

        Parameters
        ----------
        audio
            shape (num_samples,), float32, normalized to [-1, 1]

        Returns
        -------
        P_db
            shape (n_freq, n_time), float32, dB scale.
            n_freq = n_fft // 2 + 1 = 257.
        """
        # ── Step 1: STFT ──────────────────────────────────────────────
        audio_torch = torch.from_numpy(audio).float()

        spec = torch.stft(
            audio_torch,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self._window,
            return_complex=True,
        )
        # spec shape: (n_freq, n_time), complex64

        # ── Step 2: magnitude squared = power ─────────────────────────
        magnitude = torch.abs(spec)
        power = magnitude**2

        # ── Step 3: dB 변환 ───────────────────────────────────────────
        # log(0) 방지를 위한 작은 epsilon (-160 dB 정도 = 사실상 무음)
        epsilon = 1e-16
        power_db = 10.0 * torch.log10(power + epsilon)

        # numpy 변환
        return power_db.numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Stage 2: Bark Scale Conversion (TODO)
    # ------------------------------------------------------------------
    @staticmethod
    def hz_to_bark(freq_hz: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Zwicker 공식으로 Hz → Bark 변환.

        bark = 13 * arctan(0.00076 * f) + 3.5 * arctan((f / 7500)²)
        """
        raise NotImplementedError("Stage 2: hz_to_bark not yet implemented")

    @staticmethod
    def bark_to_hz(bark: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Bark → Hz 역변환."""
        raise NotImplementedError("Stage 2: bark_to_hz not yet implemented")

    def power_to_bark(self, power_db: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Hz 축의 파워 스펙트럼을 Bark 축 24개 대역으로 보간."""
        raise NotImplementedError("Stage 2: power_to_bark not yet implemented")

    # ------------------------------------------------------------------
    # Stage 3: Absolute Threshold of Hearing (TODO)
    # ------------------------------------------------------------------
    @staticmethod
    def absolute_threshold(freq_hz: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Terhardt ATH 공식."""
        raise NotImplementedError("Stage 3: absolute_threshold not yet implemented")

    def _build_ath_cache(self) -> npt.NDArray[np.float32]:
        """전 주파수 범위 ATH lookup table 캐싱."""
        if self._ath_cache is None:
            n_freq = self.config.n_fft // 2 + 1
            freqs = np.linspace(0, self.config.sample_rate / 2, n_freq, dtype=np.float32)
            self._ath_cache = self.absolute_threshold(freqs)
        return self._ath_cache

    # ------------------------------------------------------------------
    # Stage 4: Simultaneous Masking (TODO)
    # ------------------------------------------------------------------
    def find_maskers(
        self, power_db: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
        """파워 스펙트럼에서 마스커(peak) 찾기."""
        raise NotImplementedError("Stage 4: find_maskers not yet implemented")

    def classify_tonal(
        self, power_db: npt.NDArray[np.float32], peak_indices: npt.NDArray[np.int32]
    ) -> npt.NDArray[np.bool_]:
        """Peak prominence 기준 tonal/non-tonal 분류."""
        raise NotImplementedError("Stage 4: classify_tonal not yet implemented")

    def spreading_function(
        self,
        masker_bark: float,
        target_bark: npt.NDArray[np.float32],
        masker_level_db: float,
    ) -> npt.NDArray[np.float32]:
        """Schroeder spreading function."""
        raise NotImplementedError("Stage 4: spreading_function not yet implemented")

    def compute_masking_threshold(
        self, power_db: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """동시 마스킹으로 마스킹 임계치 T_m(f, t) 계산."""
        raise NotImplementedError("Stage 4: compute_masking_threshold not yet implemented")

    # ------------------------------------------------------------------
    # Stage 5: Final Threshold M(f, t) (TODO)
    # ------------------------------------------------------------------
    def compute_threshold(self, audio: AudioArray) -> npt.NDArray[np.float32]:
        """최종 청각 임계치 M(f, t) 계산.

        M(f, t) = 10·log₁₀(10^(T_q(f)/10) + 10^(T_m(f, t)/10))
        """
        raise NotImplementedError("compute_threshold not yet implemented")

    # ------------------------------------------------------------------
    # Public clamp interface (TODO)
    # ------------------------------------------------------------------
    def clamp(
        self,
        audio: AudioArray,
        raw_noise: torch.Tensor,
    ) -> torch.Tensor:
        """RL이 만든 raw noise를 청각 임계치 이내로 element-wise clamp."""
        raise NotImplementedError("clamp not yet implemented")
