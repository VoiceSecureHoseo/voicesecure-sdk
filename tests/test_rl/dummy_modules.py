"""더미 모듈 모음 — 실제 모듈 완성 전 파이프라인 테스트용.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
실제 모듈로 교체하는 방법 (팀원 모듈 완성 후):

    # test_agent.py 상단에서 아래처럼 교체

    # [교체 전 - 더미]
    from dummy_modules import DummyMasker, DummyMixer, DummyReward
    masker    = DummyMasker()
    mixer     = DummyMixer()
    reward_fn = DummyReward()

    # [교체 후 - 실제]
    from voicesecure.modulation.masker import PsychoacousticMasker
    from voicesecure.modulation.mixer import Mixer
    from voicesecure.evaluators.speaker import SpeakerEvaluator
    masker    = PsychoacousticMasker()
    mixer     = Mixer()
    reward_fn = SpeakerEvaluator()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import torch

from voicesecure.types import ACTION_N_FREQ, ACTION_N_TIME, AudioArray, Reward


class DummyMasker:
    """더미 PsychoacousticMasker (dPsk 담당 모듈 완성 전까지 사용).

    ── 실제 모듈에서 필요한 것 ──────────────────────────────────────
    인터페이스:
        clamp(audio: AudioArray, raw_noise: torch.Tensor) -> torch.Tensor

    입력:
        audio     : 원본 음성, shape (num_samples,), float32, [-1, 1]
        raw_noise : PolicyNetwork가 생성한 노이즈 spectrogram,
                    shape (n_freq, n_time) = (257, 100), float32

    출력:
        safe_noise: 청각 임계치 이하로 clamp된 노이즈 spectrogram,
                    shape (n_freq, n_time) = (257, 100), float32
                    → 이 값이 Mixer로 전달됨

    실제 동작:
        1. 원본 음성으로 STFT → 주파수별 에너지 계산
        2. 심리음향 마스킹 임계치(masking threshold) 계산
        3. raw_noise를 임계치 이하로 clamp → 청각적으로 감지 불가능한 노이즈
    ──────────────────────────────────────────────────────────────────
    더미 동작: 단순히 ±0.01로 clamp (심리음향 계산 없음)
    """

    def clamp(self, audio: AudioArray, raw_noise: torch.Tensor) -> torch.Tensor:
        # TODO: 실제 PsychoacousticMasker.clamp()로 교체
        #       실제는 audio 기반 마스킹 임계치 계산 후 주파수별 clamp
        return raw_noise.clamp(-0.01, 0.01)


class DummyMixer:
    """더미 Mixer (실제 STFT 기반 Mixer 완성 전까지 사용).

    ── 실제 모듈에서 필요한 것 ──────────────────────────────────────
    인터페이스:
        mix(audio: AudioArray, safe_noise: torch.Tensor) -> AudioArray

    입력:
        audio      : 원본 음성, shape (num_samples,), float32, [-1, 1]
        safe_noise : Masker가 clamp한 노이즈 spectrogram,
                     shape (n_freq, n_time) = (257, 100), float32

    출력:
        modified_audio: 노이즈가 섞인 변형 음성,
                        shape (num_samples,), float32, [-1, 1]
                        → 원본과 동일한 길이여야 함 (SRS FR-3)

    실제 동작:
        1. 원본 음성 STFT → spectrogram
        2. spectrogram + safe_noise (주파수 도메인 합산)
        3. iSTFT → 시간 도메인으로 복원
        4. 원본 길이로 trim/pad → 반환
        파라미터: n_fft=512, hop_length=160, win_length=400 (ARCHITECTURE.md 고정값)
    ──────────────────────────────────────────────────────────────────
    더미 동작: safe_noise 시간축 평균 → 1D로 압축 후 원본에 단순 덧셈
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate

    def mix(self, audio: AudioArray, safe_noise: torch.Tensor) -> AudioArray:
        # TODO: 실제 Mixer.mix()로 교체
        #       실제는 STFT 도메인에서 합산 후 iSTFT로 복원
        noise_1d = safe_noise.mean(dim=0).detach().numpy()  # (n_time,) — 더미 근사

        # 원본 음성 길이에 맞게 noise를 반복하거나 잘라서 맞춤
        n = len(audio)
        if len(noise_1d) >= n:
            noise_1d = noise_1d[:n]
        else:
            repeats = (n // len(noise_1d)) + 1
            noise_1d = np.tile(noise_1d, repeats)[:n]

        modified = audio + noise_1d.astype(np.float32)
        return np.clip(modified, -1.0, 1.0)


class DummyReward:
    """더미 RewardFunction (실제 Evaluator 완성 전까지 사용).

    ── 실제 모듈에서 필요한 것 ──────────────────────────────────────
    인터페이스:
        compute(modified_audio: AudioArray) -> Reward (float)

    입력:
        modified_audio: 노이즈가 섞인 변형 음성,
                        shape (num_samples,), float32, [-1, 1]

    출력:
        reward: float, 높을수록 deepfake AI를 잘 속인 것
                범위: [0.0, 1.0] 권장 (PPO 학습 안정성)

    실제 동작 (evaluator 종류에 따라):
        SV  (Speaker Verification) : 화자 인식 AI가 다른 사람으로 판별 → reward 높음
        TTS (Text-To-Speech)       : TTS 모델이 다른 화자로 복제 시도 → reward 높음
        ASR (Automatic Speech Rec) : 음성 인식 정확도 유지 여부 → reward 보조 지표
        통합 reward = SV_score * w1 + TTS_score * w2 - ASR_degradation * w3
    ──────────────────────────────────────────────────────────────────
    더미 동작: 랜덤 0~1 반환 (PPO 로직 자체가 동작하는지만 검증)
    """

    def compute(self, modified_audio: AudioArray) -> Reward:
        # TODO: 실제 SpeakerEvaluator.compute()로 교체
        #       실제는 SV/TTS/ASR evaluator로 변형 음성 평가 후 reward 계산
        return float(np.random.uniform(0.0, 1.0))
