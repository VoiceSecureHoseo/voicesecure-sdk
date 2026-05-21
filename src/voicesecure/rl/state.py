"""StateExtractor: 원본 음성에서 RL Agent의 36-dim state 벡터를 추출한다.

state 구성 (36-dim):
    - 스펙트럼 대역 에너지 mean+std × 3대역 (저/중/고역) = 6-dim
    - 스펙트럼 시간 변화 (프레임별 에너지 std) = 1-dim
    - MFCC mean×13 + std×13 = 26-dim
    - F0 mean + std = 2-dim
    - RMS = 1-dim
    합계 = 36-dim

기존 16-dim 대비 개선:
    - MFCC std 추가: 발화 내 음색 변화 정도
    - 스펙트럼 대역별 에너지: 어느 주파수 대역이 강한지 (노이즈 방향 결정에 직접 연관)
    - 스펙트럼 시간 변화: 유성음/무성음 비율, 에너지 변동성
    - F0 std: 억양 변화 정도

librosa 대신 scipy/numpy만 사용 — Colab에서 speechbrain k2 lazy-loader와의
충돌을 피하기 위함.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from scipy.fft import dct
from scipy.signal import lfilter

from voicesecure.types import SAMPLE_RATE, STATE_DIM, AudioArray, State

logger = logging.getLogger(__name__)

_N_FFT = 512
_HOP_LENGTH = 160
_N_MFCC = 13
_N_MELS = 40
_F0_FMIN = 50.0
_F0_FMAX = 500.0

# 스펙트럼 대역 경계 (Hz) — 저/중/고역
_BAND_EDGES_HZ = [0.0, 800.0, 3200.0, 8000.0]


class StateExtractor:
    """원본 음성에서 RL Agent의 36-dim state 벡터를 추출한다.

    외부 딥러닝 모델 의존성 없음. scipy/numpy만 사용.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._mel_fb = self._build_mel_filterbank()
        self._band_masks = self._build_band_masks()

    def extract(self, audio: AudioArray) -> State:
        """음성에서 36-dim state 벡터를 추출한다.

        Returns:
            state: shape (36,), dtype=float32
                [band_energy_mean×3, band_energy_std×3,  (6)
                 spectral_flux,                           (1)
                 mfcc_mean×13, mfcc_std×13,              (26)
                 f0_mean, f0_std,                        (2)
                 rms]                                    (1)
        """
        mag_frames = self._compute_stft_magnitude(audio)  # (n_frames, n_freq)

        band_mean, band_std = self._compute_band_energy(mag_frames)  # (3,), (3,)
        spectral_flux = self._compute_spectral_flux(mag_frames)       # (1,)
        mfcc_mean, mfcc_std = self._compute_mfcc(audio)               # (13,), (13,)
        f0_mean, f0_std = self._compute_f0(audio)                     # (1,), (1,)
        rms = self._compute_rms(mag_frames)                           # (1,)

        state_np = np.concatenate([
            band_mean, band_std,   # 6
            spectral_flux,         # 1
            mfcc_mean, mfcc_std,   # 26
            f0_mean, f0_std,       # 2
            rms,                   # 1
        ])  # 총 36-dim

        assert state_np.shape == (STATE_DIM,), f"state shape 오류: {state_np.shape}"
        return torch.from_numpy(state_np).float()

    # ── 내부 메서드 ───────────────────────────────────────────────────────────

    def _compute_stft_magnitude(self, audio: AudioArray) -> np.ndarray:
        """STFT magnitude 프레임 행렬 계산.

        Returns:
            mag: shape (n_frames, n_freq), float32
        """
        emphasized = lfilter([1, -0.97], [1], audio).astype(np.float32)
        frame_len = _N_FFT
        hop = _HOP_LENGTH
        window = np.hanning(frame_len).astype(np.float32)

        n_frames = max(1, (len(emphasized) - frame_len) // hop + 1)
        frames = []
        for i in range(n_frames):
            start = i * hop
            end = start + frame_len
            if end > len(emphasized):
                break
            frames.append(emphasized[start:end] * window)

        if not frames:
            frames = [np.zeros(frame_len, dtype=np.float32)]

        frames_np = np.stack(frames)  # (n_frames, frame_len)
        mag = np.abs(np.fft.rfft(frames_np, n=_N_FFT)).astype(np.float32)  # (n_frames, n_freq)
        return mag

    def _build_band_masks(self) -> list[np.ndarray]:
        """저/중/고역 주파수 bin 마스크 생성."""
        n_freq = _N_FFT // 2 + 1
        freqs = np.linspace(0, self.sample_rate / 2, n_freq)
        masks = []
        for i in range(len(_BAND_EDGES_HZ) - 1):
            lo, hi = _BAND_EDGES_HZ[i], _BAND_EDGES_HZ[i + 1]
            mask = (freqs >= lo) & (freqs < hi)
            if not mask.any():
                mask[0] = True  # 최소 1개 bin 보장
            masks.append(mask)
        return masks

    def _compute_band_energy(
        self, mag_frames: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """저/중/고역 대역별 에너지 mean + std.

        Returns:
            band_mean: (3,) — 각 대역의 프레임 평균 에너지
            band_std:  (3,) — 각 대역의 프레임간 에너지 변동성
        """
        power = mag_frames ** 2  # (n_frames, n_freq)
        means, stds = [], []
        for mask in self._band_masks:
            band_energy = power[:, mask].mean(axis=1)  # (n_frames,)
            means.append(float(band_energy.mean()))
            stds.append(float(band_energy.std() + 1e-10))

        # log scale + 정규화
        means_np = np.log(np.array(means, dtype=np.float32) + 1e-10)
        stds_np = np.log(np.array(stds, dtype=np.float32) + 1e-10)

        # [-5, 5] 범위로 클리핑
        means_np = np.clip(means_np, -5.0, 5.0)
        stds_np = np.clip(stds_np, -5.0, 5.0)
        return means_np, stds_np

    def _compute_spectral_flux(self, mag_frames: np.ndarray) -> np.ndarray:
        """스펙트럼 시간 변화량 (프레임 간 magnitude 차이 평균).

        유성음/무성음 전환이 많을수록 큰 값 → 노이즈 시간 게이팅에 유용.
        """
        if mag_frames.shape[0] < 2:
            return np.array([0.0], dtype=np.float32)
        diff = np.diff(mag_frames, axis=0)  # (n_frames-1, n_freq)
        flux = float(np.sqrt((diff ** 2).mean()))
        # 정규화: 일반적 범위 [0, 0.1] → [0, 1]
        return np.array([np.clip(flux * 10.0, 0.0, 1.0)], dtype=np.float32)

    def _build_mel_filterbank(self) -> np.ndarray:
        """Mel filterbank (n_mels × n_freq) 생성."""
        n_freq = _N_FFT // 2 + 1
        freqs = np.linspace(0, self.sample_rate / 2, n_freq)

        def hz_to_mel(f):
            return 2595.0 * np.log10(1.0 + f / 700.0)

        def mel_to_hz(m):
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        mel_min = hz_to_mel(0.0)
        mel_max = hz_to_mel(self.sample_rate / 2)
        mel_points = np.linspace(mel_min, mel_max, _N_MELS + 2)
        hz_points = mel_to_hz(mel_points)

        fb = np.zeros((_N_MELS, n_freq), dtype=np.float32)
        for m in range(1, _N_MELS + 1):
            f_left = hz_points[m - 1]
            f_center = hz_points[m]
            f_right = hz_points[m + 1]
            for k, f in enumerate(freqs):
                if f_left <= f <= f_center:
                    fb[m - 1, k] = (f - f_left) / (f_center - f_left + 1e-8)
                elif f_center < f <= f_right:
                    fb[m - 1, k] = (f_right - f) / (f_right - f_center + 1e-8)
        return fb

    def _compute_mfcc(self, audio: AudioArray) -> tuple[np.ndarray, np.ndarray]:
        """MFCC 13차원 시간 평균 + std.

        Returns:
            mfcc_mean: (13,)
            mfcc_std:  (13,)
        """
        emphasized = lfilter([1, -0.97], [1], audio).astype(np.float32)
        frame_len = _N_FFT
        hop = _HOP_LENGTH
        n_frames = max(1, (len(emphasized) - frame_len) // hop + 1)
        window = np.hanning(frame_len).astype(np.float32)

        frames = []
        for i in range(n_frames):
            start = i * hop
            end = start + frame_len
            if end > len(emphasized):
                break
            frames.append(emphasized[start:end] * window)

        if not frames:
            frames = [np.zeros(frame_len, dtype=np.float32)]

        frames_np = np.stack(frames)
        spec = np.abs(np.fft.rfft(frames_np, n=_N_FFT)) ** 2
        mel_spec = spec @ self._mel_fb.T
        log_mel = np.log(mel_spec + 1e-10)
        mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :_N_MFCC]  # (n_frames, 13)

        return (
            mfcc.mean(axis=0).astype(np.float32),
            (mfcc.std(axis=0) + 1e-10).astype(np.float32),
        )

    def _compute_f0(self, audio: AudioArray) -> tuple[np.ndarray, np.ndarray]:
        """F0 추정 (자기상관 기반). mean + std 반환, [0,1] 정규화.

        Returns:
            f0_mean: (1,)
            f0_std:  (1,)
        """
        hop = _HOP_LENGTH
        frame_len = _N_FFT
        min_lag = int(self.sample_rate / _F0_FMAX)
        max_lag = int(self.sample_rate / _F0_FMIN)

        f0_list = []
        for start in range(0, len(audio) - frame_len, hop):
            frame = audio[start: start + frame_len].astype(np.float64)
            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(corr) // 2:]
            search = corr[min_lag: max_lag + 1]
            if search.size == 0 or corr[0] < 1e-10:
                continue
            peak_lag = int(np.argmax(search)) + min_lag
            if corr[peak_lag] / (corr[0] + 1e-10) > 0.3:
                f0_list.append(self.sample_rate / peak_lag)

        if not f0_list:
            return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)

        f0_arr = np.array(f0_list, dtype=np.float32) / _F0_FMAX
        f0_mean = np.clip(float(f0_arr.mean()), 0.0, 1.0)
        f0_std = np.clip(float(f0_arr.std() + 1e-10), 0.0, 1.0)
        return np.array([f0_mean], dtype=np.float32), np.array([f0_std], dtype=np.float32)

    def _compute_rms(self, mag_frames: np.ndarray) -> np.ndarray:
        """전체 RMS energy (magnitude frames 기반)."""
        rms = float(np.sqrt((mag_frames ** 2).mean()))
        # log scale 정규화
        rms_log = float(np.log(rms + 1e-10))
        return np.array([np.clip(rms_log / 5.0, -1.0, 1.0)], dtype=np.float32)
