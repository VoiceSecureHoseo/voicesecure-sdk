"""StateExtractor 테스트 (36-dim state)."""

import numpy as np
import pytest
import torch

from voicesecure.rl.state import StateExtractor
from voicesecure.types import SAMPLE_RATE, STATE_DIM

LABELS = (
    ["band_lo_mean", "band_mid_mean", "band_hi_mean"]
    + ["band_lo_std", "band_mid_std", "band_hi_std"]
    + ["spectral_flux"]
    + [f"mfcc_mean_{i+1:02d}" for i in range(13)]
    + [f"mfcc_std_{i+1:02d}" for i in range(13)]
    + ["f0_mean", "f0_std"]
    + ["rms"]
)
assert len(LABELS) == STATE_DIM


def _synth_audio(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 800 * t)
        + 0.1 * np.sin(2 * np.pi * 3000 * t)
        + 0.02 * rng.standard_normal(SAMPLE_RATE).astype(np.float32)
    )
    return audio.astype(np.float32)


def test_output_shape():
    state = StateExtractor().extract(_synth_audio())
    assert state.shape == (STATE_DIM,), f"shape 오류: {state.shape}"


def test_output_dtype():
    state = StateExtractor().extract(_synth_audio())
    assert state.dtype == torch.float32


def test_no_nan_or_inf():
    state = StateExtractor().extract(_synth_audio())
    assert not torch.isnan(state).any(), "NaN 검출"
    assert not torch.isinf(state).any(), "Inf 검출"


def test_silent_audio():
    """무음 입력 — NaN/Inf 없고 shape 유지."""
    silent = np.zeros(SAMPLE_RATE, dtype=np.float32)
    state = StateExtractor().extract(silent)
    assert state.shape == (STATE_DIM,)
    assert not torch.isnan(state).any()
    assert not torch.isinf(state).any()


def test_different_audio_different_state():
    """다른 음성 → 다른 state."""
    ext = StateExtractor()
    s1 = ext.extract(_synth_audio(seed=42))
    s2 = ext.extract(_synth_audio(seed=99))
    assert not torch.allclose(s1, s2), "다른 음성인데 state가 동일"


def test_band_energy_ordering():
    """저역이 강한 신호 → band_lo_mean > band_hi_mean."""
    t = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    low_audio = (0.8 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    state = StateExtractor().extract(low_audio)
    band_lo_mean = state[0].item()
    band_hi_mean = state[2].item()
    assert band_lo_mean > band_hi_mean, (
        f"저역 강한 신호인데 band_lo({band_lo_mean:.3f}) <= band_hi({band_hi_mean:.3f})"
    )


def test_feature_values(capsys):
    """각 특징값 출력 — 값 범위 확인."""
    state = StateExtractor().extract(_synth_audio())
    vals = state.tolist()

    with capsys.disabled():
        print(f"\n  {'특징':<20} {'값':>10}")
        print(f"  {'-'*32}")
        for label, val in zip(LABELS, vals):
            print(f"  {label:<20} {val:>10.4f}")

    # 스펙트럼 flux: [0, 1] 범위
    assert 0.0 <= state[6].item() <= 1.0, f"spectral_flux 범위 오류: {state[6].item()}"
    # F0: [0, 1] 정규화
    assert 0.0 <= state[-3].item() <= 1.0, f"f0_mean 범위 오류: {state[-3].item()}"
    assert 0.0 <= state[-2].item() <= 1.0, f"f0_std 범위 오류: {state[-2].item()}"
    # rms: [-1, 1] log 정규화
    assert -1.0 <= state[-1].item() <= 1.0, f"rms 범위 오류: {state[-1].item()}"
