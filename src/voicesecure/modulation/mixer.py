"""Audio mixing module.

원본 음성과 안전한 노이즈(masker로 clamp된)를 합쳐서 변형 음성 출력.

See Also
--------
- ARCHITECTURE.md section 3.1: Mixer contract
- SRS.md FR-5: Audio synthesis (mixing)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from voicesecure.types import SAMPLE_RATE, AudioArray


@dataclass
class MixerConfig:
    """Mixer hyperparameters."""

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
    >>> safe_noise = torch.randn(257, 100)
    >>> modified = mixer.mix(audio, safe_noise)
    >>> assert modified.shape == audio.shape
    >>> assert -1.0 <= modified.min() and modified.max() <= 1.0
    """

    def __init__(self, config: MixerConfig | None = None) -> None:
        self.config = config or MixerConfig()

    def mix(
        self,
        audio: AudioArray,
        safe_noise: torch.Tensor,
    ) -> AudioArray:
        """원본 음성에 안전 노이즈를 spectral domain에서 더한 후 시간 영역으로 복원.

        Parameters
        ----------
        audio
            원본 음성, shape (num_samples,), float32, [-1, 1]
        safe_noise
            마스킹 임계치 이내로 clamp된 노이즈 spectrogram
            shape (n_freq, n_time), float32

        Returns
        -------
        modified_audio
            변형 음성, shape (num_samples,), float32, [-1, 1]
            원본과 동일 길이 보장
        """
        # TODO: 구현
        # 1. STFT
        #    spec = torch.stft(
        #        torch.from_numpy(audio),
        #        n_fft=self.config.n_fft,
        #        hop_length=self.config.hop_length,
        #        win_length=self.config.win_length,
        #        window=torch.hann_window(self.config.win_length),
        #        return_complex=True,
        #    )
        # 2. magnitude + noise, phase 유지
        #    magnitude = torch.abs(spec)
        #    phase = torch.angle(spec)
        #    # safe_noise shape이 spec과 동일하다고 가정
        #    new_magnitude = magnitude + safe_noise
        #    new_spec = new_magnitude * torch.exp(1j * phase)
        # 3. iSTFT
        #    modified = torch.istft(
        #        new_spec,
        #        n_fft=self.config.n_fft,
        #        hop_length=self.config.hop_length,
        #        win_length=self.config.win_length,
        #        window=torch.hann_window(self.config.win_length),
        #    )
        # 4. clipping
        #    modified = torch.clamp(modified, -1.0, 1.0)
        # 5. truncate to original length
        #    modified = modified[: len(audio)]
        # 6. numpy 변환
        #    return modified.numpy().astype(np.float32)
        raise NotImplementedError("Mixer.mix not yet implemented")
