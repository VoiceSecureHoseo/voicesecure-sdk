"""Audio mixing module.

원본 음성과 안전한 노이즈(masker로 clamp된)를 합쳐서 변형 음성 출력.

Pipeline (SRS.md FR-5):
    1. audio STFT → complex spectrogram (magnitude + phase)
    2. magnitude + safe_noise (phase는 원본 유지)
    3. iSTFT → 시간 영역 음성
    4. clipping [-1, 1]
    5. 원본 길이로 truncate

See Also
--------
- ARCHITECTURE.md section 3.1: Mixer contract
- SRS.md FR-5: Audio synthesis (mixing)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from voicesecure.types import SAMPLE_RATE, AudioArray


@dataclass
class MixerConfig:
    """Mixer hyperparameters (ARCHITECTURE.md section 4 STFT 고정값)."""

    sample_rate: int = SAMPLE_RATE
    n_fft: int = 512
    hop_length: int = 160
    win_length: int = 400


class Mixer:
    """원본 음성 + safe_noise → 변형 음성.

    Pipeline:
        1. audio STFT → complex spectrogram (magnitude + phase)
        2. magnitude + safe_noise (phase는 원본 유지)
        3. iSTFT → 시간 영역 음성
        4. clipping [-1, 1]
        5. 원본 길이로 truncate

    Examples
    --------
    >>> mixer = Mixer()
    >>> audio = np.random.randn(16000).astype(np.float32) * 0.1
    >>> safe_noise = torch.randn(257, 100) * 0.01
    >>> modified = mixer.mix(audio, safe_noise)
    >>> assert modified.shape == audio.shape
    >>> assert -1.0 <= modified.min() and modified.max() <= 1.0
    """

    def __init__(self, config: MixerConfig | None = None) -> None:
        self.config = config or MixerConfig()
        # Hann window 캐싱 — 매 호출마다 새로 만들 필요 없음
        self._window = torch.hann_window(self.config.win_length)

    def mix(
        self,
        audio: AudioArray,
        safe_noise: torch.Tensor,
    ) -> AudioArray:
        """원본 음성에 안전 노이즈를 더해 변조 음성 생성.

        Parameters
        ----------
        audio
            원본 음성, shape (num_samples,), float32, [-1, 1]
        safe_noise
            마스킹 임계치 이내로 clamp된 노이즈 spectrogram.
            shape (n_freq, n_time), float32.

        Returns
        -------
        modified_audio
            변형 음성, shape (num_samples,), float32, [-1, 1].
            원본과 동일 길이 보장.
        """
        original_length = len(audio)

        audio_torch = torch.from_numpy(audio).float()
        spec = torch.stft(
            audio_torch,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self._window,
            return_complex=True,
        )  # (n_freq, n_time_actual)

        magnitude = torch.abs(spec)
        phase = torch.angle(spec)

        safe_noise = self._align_noise_shape(safe_noise, target_shape=magnitude.shape)
        new_magnitude = torch.clamp(magnitude + safe_noise, min=0.0)
        new_spec = torch.polar(new_magnitude, phase)

        modified = torch.istft(
            new_spec,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self._window,
        )

        modified = torch.clamp(modified, -1.0, 1.0)
        modified = modified[:original_length]
        if len(modified) < original_length:
            modified = torch.nn.functional.pad(modified, (0, original_length - len(modified)))

        return modified.detach().numpy().astype(np.float32)

    def _align_noise_shape(
        self,
        safe_noise: torch.Tensor,
        target_shape: torch.Size,
    ) -> torch.Tensor:
        """safe_noise를 target spectrogram shape에 맞춤.

        safe_noise: (n_freq, n_time_noise)
        target:     (n_freq, n_time_spec)

        n_time_noise와 n_time_spec이 다르면 잘라내거나 padding.
        """
        target_n_freq, target_n_time = target_shape
        noise_n_freq, noise_n_time = safe_noise.shape

        # 주파수 차원 검증 (이게 다르면 심각한 버그)
        if noise_n_freq != target_n_freq:
            raise ValueError(
                f"safe_noise n_freq mismatch: got {noise_n_freq}, expected {target_n_freq}"
            )

        # 시간 차원 align
        if noise_n_time == target_n_time:
            return safe_noise

        if noise_n_time > target_n_time:
            # noise가 더 길면 잘라냄
            return safe_noise[:, :target_n_time]

        # noise가 더 짧으면 마지막 프레임 반복으로 padding
        pad_amount = target_n_time - noise_n_time
        last_frame = safe_noise[:, -1:].expand(-1, pad_amount)
        return torch.cat([safe_noise, last_frame], dim=1)
