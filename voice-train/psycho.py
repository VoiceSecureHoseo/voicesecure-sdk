"""
사람 귀에 최대한 안 들리게 노이즈를 숨기는 부분 (심리음향 마스킹)

사람 귀는 신기하게도, 큰 소리 옆에 있는 작은 소리는 잘 못 듣는다
(예: 시끄러운 곳에서 작은 소리가 묻히는 것과 비슷한 원리).
이 원리를 "심리음향 마스킹"이라고 부르는데, MP3 압축에도 쓰이는 기술이다.

여기서는 이 원리를 거꾸로 이용한다. 원본 소리에서 "이 정도 크기까지는
사람이 못 알아챈다"는 한계선을 미리 계산해두고, 우리가 만든 노이즈가
그 한계선을 넘지 않도록 깎아준다. 그러면 노이즈는 존재하지만 사람 귀에는
최대한 안 들리게 된다.

- compute_masking_threshold: 이 "안 들리는 한계선"을 계산하는 함수
- soft_psycho_loss: 학습 중에 쓰는 버전 (한계선 넘으면 페널티만 줌, 미분 가능해야 해서)
- hard_project: 실제 추론할 때 쓰는 버전 (한계선 넘는 부분을 아예 강제로 잘라냄)
"""

import sys
import os
import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F

SPL_REF    = 96.0
ATH_OFFSET = -60.0


# -- 청각 임계값 / Bark 스케일 유틸 -------------------------------

def _ath_db(freqs: np.ndarray) -> np.ndarray:
    f = np.maximum(freqs, 20.0) / 1000.0
    ath = (
        3.64 * f ** (-0.8)
        - 6.5  * np.exp(-0.6 * (f - 3.3) ** 2)
        + 1e-3 * f ** 4
    )
    return np.clip(ath + ATH_OFFSET, -10.0, 96.0)


def _bark(freqs: np.ndarray) -> np.ndarray:
    return 13.0 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan((freqs / 7500.0) ** 2)


# -- 마스킹 임계값 계산 -------------------------------------------

def compute_masking_threshold(audio: np.ndarray) -> np.ndarray:
    window = np.hanning(WIN_LENGTH).astype(np.float32)
    _, _, stft = scipy.signal.stft(
        audio, fs=SAMPLE_RATE, window=window,
        nperseg=WIN_LENGTH, noverlap=WIN_LENGTH - HOP_LENGTH,
        nfft=N_FFT, padded=True,
    )
    mag = np.abs(stft).astype(np.float32)
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SAMPLE_RATE)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-10)) + SPL_REF
    ath_db = _ath_db(freqs)
    bark = _bark(freqs)
    n_freq = len(freqs)
    spreading = np.zeros((n_freq, n_freq), dtype=np.float32)
    for i in range(n_freq):
        dz = bark - bark[i]
        spreading[i] = np.where(dz >= 0,
                                10.0 ** (-27.0 * dz / 10.0),
                                10.0 ** (  6.0 * dz / 10.0))
    power = 10.0 ** (mag_db / 10.0)
    masked_power = spreading.T @ power
    masking_db = 10.0 * np.log10(np.maximum(masked_power, 1e-30))
    threshold_db = np.maximum(ath_db[:, None], masking_db)
    threshold = 10.0 ** ((threshold_db - SPL_REF) / 20.0)
    return threshold.astype(np.float32)


# -- hard projection (STFT 도메인) ---------------------------------

def _project_psychoacoustic(delta_np: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    window = np.hanning(WIN_LENGTH).astype(np.float32)
    _, _, delta_stft = scipy.signal.stft(
        delta_np, fs=SAMPLE_RATE, window=window,
        nperseg=WIN_LENGTH, noverlap=WIN_LENGTH - HOP_LENGTH,
        nfft=N_FFT, padded=True,
    )
    n_frames_stft   = delta_stft.shape[1]
    n_frames_thresh = threshold.shape[1]
    if n_frames_thresh < n_frames_stft:
        pad = n_frames_stft - n_frames_thresh
        threshold = np.concatenate([threshold, np.tile(threshold[:, -1:], (1, pad))], axis=1)
    elif n_frames_thresh > n_frames_stft:
        threshold = threshold[:, :n_frames_stft]
    mag = np.abs(delta_stft)
    phase = np.angle(delta_stft)
    mag_clipped = np.minimum(mag, threshold)
    delta_stft_proj = mag_clipped * np.exp(1j * phase)
    _, delta_proj = scipy.signal.istft(
        delta_stft_proj, fs=SAMPLE_RATE, window=window,
        nperseg=WIN_LENGTH, noverlap=WIN_LENGTH - HOP_LENGTH, nfft=N_FFT,
    )
    n = len(delta_np)
    if len(delta_proj) >= n:
        delta_proj = delta_proj[:n]
    else:
        delta_proj = np.pad(delta_proj, (0, n - len(delta_proj)))
    return delta_proj.astype(np.float32)

SAMPLE_RATE = 16_000
N_FFT       = 512
HOP_LENGTH  = 160
WIN_LENGTH  = 400


# -- 학습용 soft constraint / 추론용 hard projection -------------

def soft_psycho_loss(delta: torch.Tensor, threshold_np: np.ndarray) -> torch.Tensor:
    """
    학습용 버전. 노이즈(delta)가 "안 들리는 한계선"을 넘는 만큼 벌점을
    준다. 학습 중 역전파 신호가 계속 흐르도록 계산하며, 한계선을 안
    넘었으면 벌점은 0이다.
    """
    device = delta.device

    window = torch.hann_window(WIN_LENGTH, device=device)

    # torch.stft: (T,) -> (N_FREQ, n_frames) complex
    stft = torch.stft(
        delta,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        center=False,   # scipy compute_masking_threshold와 프레임 수 일치
        return_complex=True,
    )  # (N_FREQ, n_frames) complex

    # magnitude (복소수 절댓값, gradient 유지)
    mag = stft.abs() + 1e-12  # (N_FREQ, n_frames)

    # threshold를 tensor로 변환 + 크기 맞춤
    thresh = torch.from_numpy(threshold_np).to(device)  # (N_FREQ, n_frames_thresh)

    n_frames_stft  = mag.shape[1]
    n_frames_thresh = thresh.shape[1]

    if n_frames_thresh < n_frames_stft:
        pad = n_frames_stft - n_frames_thresh
        thresh = torch.cat([thresh, thresh[:, -1:].expand(-1, pad)], dim=1)
    elif n_frames_thresh > n_frames_stft:
        thresh = thresh[:, :n_frames_stft]

    # 초과분에만 페널티 (ReLU)
    exceed = F.relu(mag - thresh)  # (N_FREQ, n_frames)
    return exceed.mean()


def hard_project(delta_np: np.ndarray, threshold_np: np.ndarray) -> np.ndarray:
    """
    실제 사용(추론) 시 쓰는 버전. 노이즈가 한계선을 넘는 부분을 페널티만
    주는 게 아니라 아예 강제로 깎아낸다.
    """
    return _project_psychoacoustic(delta_np, threshold_np)


def compute_threshold(audio_np: np.ndarray) -> np.ndarray:
    """compute_masking_threshold를 간단히 호출할 수 있게 감싼 함수."""
    return compute_masking_threshold(audio_np)
