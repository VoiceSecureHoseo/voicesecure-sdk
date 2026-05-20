"""실제 모듈을 연결한 통합 파이프라인 테스트.

dummy_modules.py 없이 실제 PsychoacousticMasker / Mixer / SafetyChecker /
RewardFunction을 사용해서 학습 파이프라인 한 step이 끝까지 동작하는지 검증.

기존 test_rl/test_agent.py (더미 사용)와는 별도로 유지된다.
한승규의 더미 테스트는 RL 단독 검증용으로 그대로 활용 가능.

전체 흐름:
    audio → agent.act() → masker.clamp() → mixer.mix() → safety.check()
          → reward (dict 직접) → PPO update

evaluator 어댑터가 아직 없으므로 reward 입력은 더미 dict로 직접 전달.
실제 학습 시에는 SV/TTS/ASR evaluator가 이 dict를 채워준다.

See Also
--------
- ARCHITECTURE.md section 5: 학습/추론 파이프라인
- tests/test_rl/test_agent.py: 한승규의 더미 기반 RL 단독 테스트
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from voicesecure.modulation.masker import PsychoacousticMasker
from voicesecure.modulation.mixer import Mixer
from voicesecure.modulation.safety import SafetyChecker
from voicesecure.reward.function import RewardFunction
from voicesecure.rl.agent import RLAgent
from voicesecure.types import ACTION_N_FREQ, ACTION_N_TIME, SAMPLE_RATE, Transition


# ----------------------------------------------------------------------
# Fixtures: 실제 음성 대신 합성 음성 사용 (LibriSpeech 파일 의존성 제거)
# ----------------------------------------------------------------------
@pytest.fixture
def audio() -> np.ndarray:
    """1초 분량 합성 음성 (사인 + 약간의 노이즈, 음성과 유사한 특성)."""
    rng = np.random.default_rng(seed=42)
    t = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    # 200 Hz + 800 Hz + 1.5 kHz 혼합 (사람 목소리 대략 흉내)
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 800 * t)
        + 0.1 * np.sin(2 * np.pi * 1500 * t)
        + 0.02 * rng.standard_normal(SAMPLE_RATE)
    )
    return audio.astype(np.float32)


@pytest.fixture
def agent() -> RLAgent:
    return RLAgent()


@pytest.fixture
def masker() -> PsychoacousticMasker:
    return PsychoacousticMasker()


@pytest.fixture
def mixer() -> Mixer:
    return Mixer()


@pytest.fixture
def safety() -> SafetyChecker:
    return SafetyChecker()


@pytest.fixture
def reward_fn() -> RewardFunction:
    return RewardFunction()


def _dummy_reward_components() -> dict:
    """evaluator 어댑터 없을 때 임시 더미 점수."""
    rng = np.random.default_rng()
    return {
        "sv_score": float(rng.uniform(0.4, 0.9)),
        "tts_score": float(rng.uniform(0.3, 0.8)),
        "asr_cer": float(rng.uniform(0.05, 0.25)),
    }


# ----------------------------------------------------------------------
# Step별 단독 검증
# ----------------------------------------------------------------------
class TestPipelineSteps:
    """파이프라인 각 step이 개별적으로 잘 동작하는지 확인."""

    def test_act_produces_valid_action(self, agent: RLAgent, audio: np.ndarray) -> None:
        """RLAgent.act가 정상 action 생성."""
        action, log_prob, value = agent.act(audio)
        assert action.shape == (ACTION_N_FREQ, ACTION_N_TIME)
        assert not torch.isnan(action).any()
        assert not torch.isnan(log_prob)
        assert not torch.isnan(value)

    def test_masker_clamps_action(
        self,
        agent: RLAgent,
        masker: PsychoacousticMasker,
        audio: np.ndarray,
    ) -> None:
        """Masker가 action을 safe_noise로 clamp."""
        action, _, _ = agent.act(audio)
        safe_noise = masker.clamp(audio, action)
        # shape 보존 (시간 차원은 audio STFT 결과에 따라 다를 수 있음)
        assert safe_noise.shape[0] == action.shape[0]
        assert not torch.isnan(safe_noise).any()

    def test_mixer_produces_audio(
        self,
        agent: RLAgent,
        masker: PsychoacousticMasker,
        mixer: Mixer,
        audio: np.ndarray,
    ) -> None:
        """Mixer가 변형 음성 생성."""
        action, _, _ = agent.act(audio)
        safe_noise = masker.clamp(audio, action)
        modified = mixer.mix(audio, safe_noise)
        assert modified.shape == audio.shape
        assert modified.dtype == np.float32
        assert modified.min() >= -1.0
        assert modified.max() <= 1.0

    def test_safety_passes_clean_audio(
        self,
        agent: RLAgent,
        masker: PsychoacousticMasker,
        mixer: Mixer,
        safety: SafetyChecker,
        audio: np.ndarray,
    ) -> None:
        """SafetyChecker가 mixer 출력 통과."""
        action, _, _ = agent.act(audio)
        safe_noise = masker.clamp(audio, action)
        modified = mixer.mix(audio, safe_noise)
        final = safety.check(audio, modified)
        assert final.shape == audio.shape
        assert not np.any(np.isnan(final))
        assert final.min() >= -1.0
        assert final.max() <= 1.0

    def test_reward_function_computes(self, reward_fn: RewardFunction) -> None:
        """RewardFunction이 더미 점수로 reward 계산."""
        reward = reward_fn.compute(_dummy_reward_components())
        assert isinstance(reward, float)
        assert not np.isnan(reward)


# ----------------------------------------------------------------------
# 전체 파이프라인 통합
# ----------------------------------------------------------------------
class TestFullPipeline:
    """모든 모듈을 연결한 통합 검증."""

    def test_one_step_pipeline(
        self,
        agent: RLAgent,
        masker: PsychoacousticMasker,
        mixer: Mixer,
        safety: SafetyChecker,
        reward_fn: RewardFunction,
        audio: np.ndarray,
    ) -> None:
        """한 step 풀 파이프라인: audio → action → safe_noise → mixed → safe → reward."""
        # Step 1: action
        action, log_prob, value = agent.act(audio)

        # Step 2: 청각 임계치 이내로 clamp
        safe_noise = masker.clamp(audio, action)

        # Step 3: spectral domain mixing
        modified = mixer.mix(audio, safe_noise)

        # Step 4: safety check
        final = safety.check(audio, modified)

        # Step 5: reward (더미 evaluator 결과 사용)
        reward = reward_fn.compute(_dummy_reward_components())

        # 모든 단계 정상
        assert final.shape == audio.shape
        assert isinstance(reward, float)
        assert not np.isnan(reward)

        # 학습 데이터 (Transition) 만들 수 있는지 확인
        next_state = agent.extractor.extract(final)
        state = agent.extractor.extract(audio)
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=False,
            log_prob=log_prob,
            value=value,
        )
        assert transition.action.shape == (ACTION_N_FREQ, ACTION_N_TIME)

    def test_multi_step_with_ppo_update(
        self,
        agent: RLAgent,
        masker: PsychoacousticMasker,
        mixer: Mixer,
        safety: SafetyChecker,
        reward_fn: RewardFunction,
        audio: np.ndarray,
    ) -> None:
        """여러 step 수집 후 PPO 업데이트 1회 동작."""
        n_steps = 8
        transitions = []

        for _ in range(n_steps):
            action, log_prob, value = agent.act(audio, deterministic=False)
            safe_noise = masker.clamp(audio, action)
            modified = mixer.mix(audio, safe_noise)
            final = safety.check(audio, modified)
            reward = reward_fn.compute(_dummy_reward_components())

            state = agent.extractor.extract(audio)
            next_state = agent.extractor.extract(final)

            transitions.append(
                Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=False,
                    log_prob=log_prob,
                    value=value,
                )
            )

        # PPO 업데이트
        metrics = agent.update(transitions)

        # metrics 정상
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert not np.isnan(metrics["policy_loss"])
        assert not np.isnan(metrics["value_loss"])
        assert not np.isnan(metrics["entropy"])

    def test_pipeline_repeatable(
        self,
        agent: RLAgent,
        masker: PsychoacousticMasker,
        mixer: Mixer,
        safety: SafetyChecker,
        reward_fn: RewardFunction,
        audio: np.ndarray,
    ) -> None:
        """같은 입력 + deterministic 모드 → 같은 출력 (재현성)."""
        # 1차 실행
        action1, _, _ = agent.act(audio, deterministic=True)
        safe_noise1 = masker.clamp(audio, action1)
        modified1 = mixer.mix(audio, safe_noise1)
        final1 = safety.check(audio, modified1)

        # 2차 실행
        action2, _, _ = agent.act(audio, deterministic=True)
        safe_noise2 = masker.clamp(audio, action2)
        modified2 = mixer.mix(audio, safe_noise2)
        final2 = safety.check(audio, modified2)

        # 동일 결과 (PPO update 안 하면 weight 변경 없으므로)
        assert torch.allclose(action1, action2)
        assert torch.allclose(safe_noise1, safe_noise2)
        assert np.allclose(modified1, modified2)
        assert np.allclose(final1, final2)


# ----------------------------------------------------------------------
# 수동 실행: python tests/test_integration/test_full_pipeline.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("실제 모듈 통합 파이프라인 테스트")
    print("=" * 60)

    rng = np.random.default_rng(seed=42)
    t = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 800 * t)
        + 0.1 * np.sin(2 * np.pi * 1500 * t)
        + 0.02 * rng.standard_normal(SAMPLE_RATE)
    ).astype(np.float32)

    agent = RLAgent()
    masker = PsychoacousticMasker()
    mixer = Mixer()
    safety = SafetyChecker()
    reward_fn = RewardFunction()

    print(f"\n[입력] 합성 음성 ({len(audio)/SAMPLE_RATE:.1f}초, max={np.abs(audio).max():.3f})")

    # 한 step 풀 파이프라인
    print("\n[Step 1] RLAgent.act")
    action, log_prob, value = agent.act(audio)
    print(f"  action.shape    = {action.shape}")
    print(f"  action min/max  = {action.min():.4f} / {action.max():.4f}")
    print(f"  log_prob        = {log_prob.item():.4f}")
    print(f"  value           = {value.item():.4f}")

    print("\n[Step 2] PsychoacousticMasker.clamp")
    safe_noise = masker.clamp(audio, action)
    print(f"  safe_noise.shape = {safe_noise.shape}")
    print(f"  safe_noise max  = {safe_noise.abs().max():.6f}")

    print("\n[Step 3] Mixer.mix")
    modified = mixer.mix(audio, safe_noise)
    print(f"  modified.shape  = {modified.shape}")
    print(f"  modified max    = {np.abs(modified).max():.4f}")

    print("\n[Step 4] SafetyChecker.check")
    final = safety.check(audio, modified)
    print(f"  final.shape     = {final.shape}")
    print(f"  final max       = {np.abs(final).max():.4f}")

    print("\n[Step 5] RewardFunction (더미 evaluator 결과)")
    components = _dummy_reward_components()
    reward = reward_fn.compute(components)
    print(f"  components      = {components}")
    print(f"  reward          = {reward:.4f}")

    # PPO update 검증
    print("\n[Step 6] PPO update (8 transitions)")
    transitions = []
    for _ in range(8):
        a, lp, v = agent.act(audio, deterministic=False)
        sn = masker.clamp(audio, a)
        m = mixer.mix(audio, sn)
        f = safety.check(audio, m)
        r = reward_fn.compute(_dummy_reward_components())
        transitions.append(
            Transition(
                state=agent.extractor.extract(audio),
                action=a,
                reward=r,
                next_state=agent.extractor.extract(f),
                done=False,
                log_prob=lp,
                value=v,
            )
        )
    metrics = agent.update(transitions)
    print(f"  policy_loss     = {metrics['policy_loss']:.4f}")
    print(f"  value_loss      = {metrics['value_loss']:.4f}")
    print(f"  entropy         = {metrics['entropy']:.4f}")

    print("\n" + "=" * 60)
    print("통합 파이프라인 정상 동작 ✓")
    print("=" * 60)
    sys.exit(0)
