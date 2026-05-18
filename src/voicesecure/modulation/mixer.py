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
        """원본 음성에 안전 노이즈를 spectral domain에서 더한 후 시간 영역으로 복원.

        Parameters
        ----------
        audio
            원본 음성, shape (num_samples,), float32, [-1, 1]
        safe_noise
            마스킹 임계치 이내로 clamp된 노이즈 spectrogram.
            shape (n_freq, n_time), float32.
            n_freq = n_fft // 2 + 1 = 257.

        Returns
        -------
        modified_audio
            변형 음성, shape (num_samples,), float32, [-1, 1].
            원본과 동일 길이 보장.
        """
        original_length = len(audio)

        # ── Step 1: STFT (실수 → 복소수 spectrogram) ──────────────────
        audio_torch = torch.from_numpy(audio).float()

        spec = torch.stft(
            audio_torch,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self._window,
            return_complex=True,
        )
        # spec shape: (n_freq, n_time_actual), complex64

        # ── Step 2: magnitude + safe_noise, phase 유지 ───────────────
        # 사람 청각은 phase 변화에 둔감하고 magnitude 변화에 민감
        # → magnitude에만 노이즈 더하고 phase는 그대로 유지
        magnitude = torch.abs(spec)  # (n_freq, n_time_actual)
        phase = torch.angle(spec)  # (n_freq, n_time_actual)

        # safe_noise shape align (입력 길이에 따라 spec의 n_time이 달라질 수 있음)
        safe_noise = self._align_noise_shape(safe_noise, target_shape=spec.shape)

        # 마그니튜드에 노이즈 더하기
        new_magnitude = magnitude + safe_noise

        # magnitude는 항상 ≥ 0이어야 하므로 음수 방지
        new_magnitude = torch.clamp(new_magnitude, min=0.0)

        # 새 complex spectrogram = new_magnitude * e^(i*phase)
        new_spec = torch.polar(new_magnitude, phase)

        # ── Step 3: iSTFT (복소수 → 시간 영역) ────────────────────────
        modified = torch.istft(
            new_spec,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self._window,
        )

        # ── Step 4: clipping [-1, 1] ─────────────────────────────────
        modified = torch.clamp(modified, -1.0, 1.0)

        # ── Step 5: 원본 길이로 truncate (STFT padding 제거) ──────────
        modified = modified[:original_length]

        # 원본보다 짧으면 zero-padding으로 길이 맞춤
        if len(modified) < original_length:
            modified = torch.nn.functional.pad(modified, (0, original_length - len(modified)))

        # ── Step 6: numpy 변환 + dtype 확정 ──────────────────────────
        return modified.numpy().astype(np.float32)

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
