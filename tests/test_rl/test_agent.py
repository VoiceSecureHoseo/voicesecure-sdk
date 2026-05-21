"""RLAgent 테스트."""

import os
import sys

import numpy as np
import pytest
import soundfile as sf
import torch
from math import gcd
from scipy.signal import resample_poly

from voicesecure.rl.agent import RLAgent
from voicesecure.types import ACTION_N_FREQ, ACTION_N_TIME, SAMPLE_RATE, Transition

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from dummy_modules import DummyMasker, DummyMixer, DummyReward

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEST_FILE = os.path.join(_ROOT, "Original", "100", "121669", "100-121669-0000.wav")


def _load_audio():
    data, sr = sf.read(TEST_FILE, dtype="float32", always_2d=True)
    audio = data.mean(axis=1).astype(np.float32)
    if sr != SAMPLE_RATE:
        g = gcd(SAMPLE_RATE, sr)
        audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return audio


@pytest.mark.slow
def test_act_output_shape():
    audio = _load_audio()
    agent = RLAgent()
    state, action, log_prob, value, freq_pattern, time_gate = agent.act(audio)

    assert action.shape == (ACTION_N_FREQ, ACTION_N_TIME)
    assert log_prob.shape == ()
    assert value.shape == ()


@pytest.mark.slow
def test_act_no_nan():
    audio = _load_audio()
    agent = RLAgent()
    state, action, log_prob, value, freq_pattern, time_gate = agent.act(audio)

    assert not torch.isnan(action).any()
    assert not torch.isnan(log_prob)
    assert not torch.isnan(value)


@pytest.mark.slow
def test_full_pipeline_one_step():
    audio = _load_audio()
    agent = RLAgent()
    masker = DummyMasker()
    mixer = DummyMixer()
    reward_fn = DummyReward()

    state, action, log_prob, value, freq_pattern, time_gate = agent.act(audio, deterministic=False)
    safe_noise = masker.clamp(audio, action)
    modified_audio = mixer.mix(audio, safe_noise)
    reward = reward_fn.compute(modified_audio)

    assert modified_audio.shape == audio.shape
    assert np.all(np.abs(modified_audio) <= 1.0)
    assert isinstance(reward, float)


@pytest.mark.slow
def test_ppo_update_one_step():
    audio = _load_audio()
    agent = RLAgent()
    masker = DummyMasker()
    mixer = DummyMixer()
    reward_fn = DummyReward()

    transitions = []
    for _ in range(8):
        state, action, log_prob, value, freq_pattern, time_gate = agent.act(audio, deterministic=False)
        safe_noise = masker.clamp(audio, action)
        modified = mixer.mix(audio, safe_noise)
        reward = reward_fn.compute(modified)
        transitions.append(
            Transition(
                state=agent.extractor.extract(audio),
                action=action,
                reward=reward,
                next_state=agent.extractor.extract(audio),
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
    assert not np.isnan(metrics["policy_loss"])
    assert not np.isnan(metrics["value_loss"])


@pytest.mark.slow
def test_save_and_load(tmp_path):
    audio = _load_audio()
    agent = RLAgent()

    _, action_before, _, _, _, _ = agent.act(audio, deterministic=True)
    path = str(tmp_path / "test_checkpoint.pt")
    agent.save(path)
    agent.load(path)
    _, action_after, _, _, _, _ = agent.act(audio, deterministic=True)

    assert torch.allclose(action_before, action_after)
