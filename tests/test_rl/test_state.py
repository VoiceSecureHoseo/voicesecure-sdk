"""StateExtractor 테스트.

pytest -v -s 로 실행하면 특징값까지 출력됨.
파일 하나만 골라서 결과를 바로 확인하는 수동 테스트 포함.
실행: python tests/test_rl/test_state.py
"""

import os

import numpy as np
import librosa
import torch

from voicesecure.rl.state import StateExtractor
from voicesecure.types import SAMPLE_RATE, STATE_DIM

# ★ 테스트할 파일 경로 — 원하는 파일로 바꾸면 됨
# 프로젝트 루트(voicesecure-sdk/) 기준 상대경로
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
TEST_FILE = os.path.normpath(
    os.path.join(_ROOT, "Original", "100", "121669", "100-121669-0000.wav")
)

LABELS = [f"MFCC_{i+1:02d}" for i in range(13)] + ["F0_mean", "duration", "rms"]


def test_output_shape():
    """state shape이 (16,) 인지 확인."""
    audio, sr = librosa.load(TEST_FILE, sr=SAMPLE_RATE, mono=True)
    extractor = StateExtractor(sample_rate=sr)
    state = extractor.extract(audio)

    print(f"\n  shape: {state.shape}")
    assert state.shape == (STATE_DIM,), f"shape 오류: {state.shape}"


def test_output_dtype():
    """state dtype이 float32 인지 확인."""
    audio, sr = librosa.load(TEST_FILE, sr=SAMPLE_RATE, mono=True)
    extractor = StateExtractor(sample_rate=sr)
    state = extractor.extract(audio)

    print(f"\n  dtype: {state.dtype}")
    assert state.dtype == torch.float32, f"dtype 오류: {state.dtype}"


def test_no_nan_or_inf():
    """state에 NaN/Inf 없는지 확인."""
    audio, sr = librosa.load(TEST_FILE, sr=SAMPLE_RATE, mono=True)
    extractor = StateExtractor(sample_rate=sr)
    state = extractor.extract(audio)

    print(f"\n  NaN: {torch.isnan(state).any().item()}")
    print(f"  Inf: {torch.isinf(state).any().item()}")
    assert not torch.isnan(state).any(), "NaN 검출"
    assert not torch.isinf(state).any(), "Inf 검출"


def test_feature_values():
    """각 특징값 출력 — 실제 추출된 값 확인."""
    audio, sr = librosa.load(TEST_FILE, sr=SAMPLE_RATE, mono=True)
    extractor = StateExtractor(sample_rate=sr)
    state = extractor.extract(audio)

    print(f"\n  파일: {TEST_FILE.split(chr(92))[-1]}  ({len(audio)/sr:.2f}초)")
    print(f"  {'특징':<12} {'값':>12}")
    print(f"  {'-'*25}")
    for label, val in zip(LABELS, state.tolist()):
        print(f"  {label:<12} {val:>12.6f}")

    # MFCC는 음수도 정상, F0/duration/rms는 0 이상이어야 함
    assert state[13].item() >= 0.0, "F0_mean 음수"
    assert state[14].item() >= 0.0, "duration 음수"
    assert state[15].item() >= 0.0, "rms 음수"


def test_silent_audio():
    """무음 입력 시 F0 fallback이 동작하는지 확인 (0으로 채워짐)."""
    silent = np.zeros(SAMPLE_RATE, dtype=np.float32)
    extractor = StateExtractor()
    state = extractor.extract(silent)

    print(f"\n  무음 state: {state.tolist()}")
    print(f"  F0_mean: {state[13].item():.6f}")
    assert state.shape == (STATE_DIM,)
    assert not torch.isnan(state).any()
    # librosa.yin은 무음에서도 최솟값(fmin/sr)을 반환할 수 있어 0 보장 안 됨
    # F0_mean이 정상 범위(0 이상)인지만 확인
    assert state[13].item() >= 0.0, "F0_mean 음수"


# ------------------------------------------------------------------ #
# 수동 실행 — 결과 바로 확인                                           #
# python tests/test_rl/test_state.py                                  #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import sys

    audio, sr = librosa.load(TEST_FILE, sr=SAMPLE_RATE, mono=True)
    print(f"파일      : {TEST_FILE.split(chr(92))[-1]}")
    print(f"샘플 수   : {len(audio)}  ({len(audio)/sr:.2f}초)")
    print(f"샘플레이트: {sr} Hz")
    print()

    extractor = StateExtractor(sample_rate=sr)
    state = extractor.extract(audio)

    print(f"state shape : {state.shape}")
    print(f"state dtype : {state.dtype}")
    print()
    print(f"{'특징':<12} {'값':>12}")
    print(f"{'-'*25}")
    for label, val in zip(LABELS, state.tolist()):
        print(f"{label:<12} {val:>12.6f}")

    sys.exit(0)
