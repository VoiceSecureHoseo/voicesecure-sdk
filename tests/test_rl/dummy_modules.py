"""더미 모듈 — 실제 모듈 완성 전 파이프라인 테스트용."""

import numpy as np
import torch

from voicesecure.types import AudioArray, Reward


class DummyMasker:
    """더미 PsychoacousticMasker. ±0.01로 단순 clamp."""

    def clamp(self, audio: AudioArray, raw_noise: torch.Tensor) -> torch.Tensor:
        return raw_noise.clamp(-0.01, 0.01)


class DummyMixer:
    """더미 Mixer. safe_noise 시간축 평균 → 원본에 덧셈."""

    def mix(self, audio: AudioArray, safe_noise: torch.Tensor) -> AudioArray:
        noise_1d = safe_noise.mean(dim=0).detach().numpy()
        n = len(audio)
        if len(noise_1d) >= n:
            noise_1d = noise_1d[:n]
        else:
            repeats = (n // len(noise_1d)) + 1
            noise_1d = np.tile(noise_1d, repeats)[:n]
        return np.clip(audio + noise_1d.astype(np.float32), -1.0, 1.0)


class DummyReward:
    """더미 Reward. 랜덤 0~1 반환."""

    def compute(self, modified_audio: AudioArray) -> Reward:
        return float(np.random.uniform(0.0, 1.0))
