"""StateExtractor → PolicyNetwork 파이프라인 테스트."""

import os
from math import gcd

import numpy as np
import pytest
import soundfile as sf
import torch
from scipy.signal import resample_poly

from voicesecure.rl.policy import PolicyNetwork
from voicesecure.rl.state import StateExtractor
from voicesecure.types import ACTION_N_FREQ, ACTION_N_TIME, SAMPLE_RATE, STATE_DIM

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEST_FILE = os.path.join(_ROOT, "Original", "100", "121669", "100-121669-0000.wav")


def _load(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = data.mean(axis=1).astype(np.float32)
    if sr != SAMPLE_RATE:
        g = gcd(SAMPLE_RATE, sr)
        audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return audio


@pytest.mark.slow
def test_pipeline_shape():
    audio = _load(TEST_FILE)
    extractor = StateExtractor()
    policy = PolicyNetwork()

    state = extractor.extract(audio)
    action, log_prob, value = policy(state)

    assert state.shape == (STATE_DIM,)
    assert action.shape == (ACTION_N_FREQ, ACTION_N_TIME)


@pytest.mark.slow
def test_pipeline_no_nan():
    audio = _load(TEST_FILE)
    extractor = StateExtractor()
    policy = PolicyNetwork()

    state = extractor.extract(audio)
    action, log_prob, value = policy(state)

    assert not torch.isnan(action).any()
    assert not torch.isnan(log_prob).any()
    assert not torch.isnan(value).any()


@pytest.mark.slow
def test_different_audio_different_action():
    file2 = os.path.join(_ROOT, "Original", "1001", "134707", "1001-134707-0000.wav")
    if not os.path.exists(file2):
        pytest.skip("두 번째 테스트 파일 없음")

    extractor = StateExtractor()
    policy = PolicyNetwork()

    audio1 = _load(TEST_FILE)
    audio2 = _load(file2)

    state1 = extractor.extract(audio1)
    state2 = extractor.extract(audio2)

    action1, _, _ = policy(state1, deterministic=True)
    action2, _, _ = policy(state2, deterministic=True)

    assert not torch.allclose(state1, state2)
    assert not torch.allclose(action1, action2)
