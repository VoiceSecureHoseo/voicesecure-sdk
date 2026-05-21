"""실제 모듈을 연결한 통합 파이프라인 테스트.

전체 흐름:
    audio → agent.act() → masker.clamp() → mixer.mix() → Transition → PPO update
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from voicesecure.modulation.masker import PsychoacousticMasker
from voicesecure.modulation.mixer import Mixer
from voicesecure.rl.agent import RLAgent
from voicesecure.types import ACTION_N_FREQ, ACTION_N_TIME, SAMPLE_RATE, Transition


@pytest.fixture
def audio() -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    t = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
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


class TestPipelineSteps:
    def test_act_produces_valid_action(self, agent: RLAgent, audio: np.ndarray) -> None:
        state, action, log_prob, value, freq_pattern, time_gate = agent.act(audio)
        assert action.shape == (ACTION_N_FREQ, ACTION_N_TIME)
        assert not torch.isnan(action).any()
        assert not torch.isnan(log_prob)
        assert not torch.isnan(value)

    def test_masker_clamps_action(
        self, agent: RLAgent, masker: PsychoacousticMasker, audio: np.ndarray
    ) -> None:
        _, action, _, _, _, _ = agent.act(audio)
        safe_noise = masker.clamp(audio, action)
        assert safe_noise.shape[0] == action.shape[0]
        assert not torch.isnan(safe_noise).any()

    def test_mixer_produces_audio(
        self, agent: RLAgent, masker: PsychoacousticMasker, mixer: Mixer, audio: np.ndarray
    ) -> None:
        _, action, _, _, _, _ = agent.act(audio)
        safe_noise = masker.clamp(audio, action)
        modified = mixer.mix(audio, safe_noise)
        assert modified.shape == audio.shape
        assert modified.dtype == np.float32
        assert modified.min() >= -1.0
        assert modified.max() <= 1.0


class TestFullPipeline:
    def test_one_step_pipeline(
        self, agent: RLAgent, masker: PsychoacousticMasker, mixer: Mixer, audio: np.ndarray
    ) -> None:
        state, action, log_prob, value, freq_pattern, time_gate = agent.act(audio)
        safe_noise = masker.clamp(audio, action)
        modified = mixer.mix(audio, safe_noise)

        assert modified.shape == audio.shape
        assert modified.min() >= -1.0
        assert modified.max() <= 1.0

        next_state = agent.extractor.extract(modified)
        transition = Transition(
            state=agent.extractor.extract(audio),
            action=action,
            reward=0.5,
            next_state=next_state,
            done=False,
            log_prob=log_prob,
            value=value,
            freq_pattern=freq_pattern,
            time_gate=time_gate,
        )
        assert transition.action.shape == (ACTION_N_FREQ, ACTION_N_TIME)

    def test_multi_step_with_ppo_update(
        self, agent: RLAgent, masker: PsychoacousticMasker, mixer: Mixer, audio: np.ndarray
    ) -> None:
        transitions = []
        for _ in range(8):
            state, action, log_prob, value, freq_pattern, time_gate = agent.act(
                audio, deterministic=False
            )
            safe_noise = masker.clamp(audio, action)
            modified = mixer.mix(audio, safe_noise)
            transitions.append(
                Transition(
                    state=agent.extractor.extract(audio),
                    action=action,
                    reward=float(np.random.uniform(0.3, 0.9)),
                    next_state=agent.extractor.extract(modified),
                    done=False,
                    log_prob=log_prob,
                    value=value,
                    freq_pattern=freq_pattern,
                    time_gate=time_gate,
                )
            )

        metrics = agent.update(transitions)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert not np.isnan(metrics["policy_loss"])
        assert not np.isnan(metrics["value_loss"])

    def test_pipeline_repeatable(
        self, agent: RLAgent, masker: PsychoacousticMasker, mixer: Mixer, audio: np.ndarray
    ) -> None:
        _, action1, _, _, _, _ = agent.act(audio, deterministic=True)
        safe_noise1 = masker.clamp(audio, action1)
        modified1 = mixer.mix(audio, safe_noise1)

        _, action2, _, _, _, _ = agent.act(audio, deterministic=True)
        safe_noise2 = masker.clamp(audio, action2)
        modified2 = mixer.mix(audio, safe_noise2)

        assert torch.allclose(action1, action2)
        assert torch.allclose(safe_noise1, safe_noise2)
        assert np.allclose(modified1, modified2)
