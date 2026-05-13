"""StateExtractor: 원본 음성에서 RL Agent의 16-dim state 벡터를 추출한다.

state 구성: [MFCC_mean(13), F0_mean(1), duration(1), rms(1)] = 16-dim
데이터셋 구조: LibriSpeech 형식
    Original/{speaker_id}/{chapter_id}/{speaker_id}-{chapter_id}-{utterance_id}.wav
"""

import logging
import warnings

import librosa
import numpy as np
import torch

from voicesecure.types import SAMPLE_RATE, STATE_DIM, AudioArray, State

logger = logging.getLogger(__name__)

# STFT 파라미터 — ARCHITECTURE.md v1.0 고정값
_N_FFT = 512
_HOP_LENGTH = 160
_N_MFCC = 13  # state 앞 13차원

# F0 추정 파라미터 (librosa.yin)
_F0_FMIN = 50.0  # Hz
_F0_FMAX = 500.0  # Hz


class StateExtractor:
    """원본 음성에서 RL Agent의 16-dim state 벡터를 추출한다.

    추론 시에도 매 음성 청크마다 호출되므로 경량 신호처리만 사용한다.
    외부 딥러닝 모델 의존성 없음.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate

    def extract(self, audio: AudioArray) -> State:
        """음성에서 16-dim state 벡터를 추출한다.

        Args:
            audio: 원본 음성, shape (num_samples,), float32, [-1, 1], 16kHz mono

        Returns:
            state: shape (STATE_DIM,) = (16,), dtype=float32 torch.Tensor
                   [MFCC_mean×13, F0_mean×1, duration×1, rms×1]
        """
        mfcc_mean = self._compute_mfcc(audio)  # (13,)
        f0_mean = self._compute_f0(audio)  # (1,)
        duration = self._compute_duration(audio)  # (1,)
        rms = self._compute_rms(audio)  # (1,)

        state_np = np.concatenate([mfcc_mean, f0_mean, duration, rms])  # (16,)
        assert state_np.shape == (STATE_DIM,), f"state shape 오류: {state_np.shape}"

        return torch.from_numpy(state_np).float()

    # ------------------------------------------------------------------ #
    # 내부 특징 추출 메서드                                                 #
    # ------------------------------------------------------------------ #

    def _compute_mfcc(self, audio: AudioArray) -> np.ndarray:
        """MFCC 계산 후 시간축 평균 → 13-dim 벡터 반환."""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=_N_MFCC,
            n_fft=_N_FFT,
            hop_length=_HOP_LENGTH,
        )
        # mfcc shape: (13, n_frames) → 시간축 평균 → (13,)
        return mfcc.mean(axis=1).astype(np.float32)

    def _compute_f0(self, audio: AudioArray) -> np.ndarray:
        """F0(기본 주파수) 추정 후 평균 → 1-dim 벡터 반환.

        librosa.yin은 무음 구간에서 0을 반환하므로,
        0이 아닌 값만 평균내어 유성음 구간의 피치를 반영한다.
        추정 실패 시(전체 무음) 0으로 채우고 경고 로그를 남긴다.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # librosa yin 내부 경고 억제
            f0 = librosa.yin(
                audio,
                fmin=_F0_FMIN,
                fmax=_F0_FMAX,
                sr=self.sample_rate,
                hop_length=_HOP_LENGTH,
            )

        voiced = f0[f0 > _F0_FMIN]  # fmin(50Hz) 이하는 추정 실패로 간주
        if voiced.size == 0:
            # 유성음 구간 없음 (무음 또는 F0 추정 실패) — SRS FR-2 예외 처리
            # librosa.yin은 무음에서도 fmin 값을 반환할 수 있어 0 보장 안 됨
            logger.warning("F0 추정 실패: 유성음 구간 없음. 0으로 대체.")
            f0_mean = 0.0
        else:
            # sample_rate로 나눠서 정규화 (0~1 범위로 스케일)
            f0_mean = float(voiced.mean()) / self.sample_rate

        return np.array([f0_mean], dtype=np.float32)

    def _compute_duration(self, audio: AudioArray) -> np.ndarray:
        """음성 길이(초)를 30초로 나눠 정규화 → 1-dim 벡터 반환.

        SRS FR-1에서 최대 입력 길이를 30초로 정의하므로 30으로 나눠 [0, 1] 범위로 정규화.
        """
        duration = len(audio) / self.sample_rate / 30.0
        return np.array([duration], dtype=np.float32)

    def _compute_rms(self, audio: AudioArray) -> np.ndarray:
        """RMS energy 계산 후 평균 → 1-dim 벡터 반환."""
        rms = librosa.feature.rms(
            y=audio,
            frame_length=_N_FFT,
            hop_length=_HOP_LENGTH,
        )
        # rms shape: (1, n_frames) → 평균 → scalar
        return np.array([rms.mean()], dtype=np.float32)
