"""StateExtractor → PolicyNetwork 파이프라인 테스트.

실제 음성 파일에서 state를 뽑고, 그 state로 노이즈를 생성하는
전체 흐름을 확인한다.

실행: python tests/test_rl/test_pipeline.py
"""

import os

import librosa
import torch

from voicesecure.rl.policy import PolicyNetwork
from voicesecure.rl.state import StateExtractor
from voicesecure.types import ACTION_N_FREQ, ACTION_N_TIME, SAMPLE_RATE, STATE_DIM

# ★ 테스트할 파일 경로 — 원하는 파일로 바꾸면 됨
# 프로젝트 루트(voicesecure-sdk/) 기준 상대경로
_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEST_FILE = os.path.join(_ROOT, "Original", "100", "121669", "100-121669-0000.wav")


def test_pipeline_shape():
    """음성 → state → action 전체 흐름에서 shape 확인."""
    audio, sr = librosa.load(TEST_FILE, sr=SAMPLE_RATE, mono=True)

    extractor = StateExtractor(sample_rate=sr)
    policy = PolicyNetwork()

    state = extractor.extract(audio)
    action, log_prob, value = policy(state)

    assert state.shape == (STATE_DIM,)
    assert action.shape == (ACTION_N_FREQ, ACTION_N_TIME)


def test_pipeline_no_nan():
    """파이프라인 전체에서 NaN/Inf 없는지 확인."""
    audio, sr = librosa.load(TEST_FILE, sr=SAMPLE_RATE, mono=True)

    extractor = StateExtractor(sample_rate=sr)
    policy = PolicyNetwork()

    state = extractor.extract(audio)
    action, log_prob, value = policy(state)

    assert not torch.isnan(action).any()
    assert not torch.isnan(log_prob).any()
    assert not torch.isnan(value).any()


def test_different_audio_different_action():
    """다른 음성이면 다른 state → 다른 action이 나오는지 확인."""
    file1 = os.path.join(_ROOT, "Original", "100", "121669", "100-121669-0000.wav")
    file2 = os.path.join(_ROOT, "Original", "1001", "134707", "1001-134707-0000.wav")

    extractor = StateExtractor()
    policy = PolicyNetwork()

    audio1, _ = librosa.load(file1, sr=SAMPLE_RATE, mono=True)
    audio2, _ = librosa.load(file2, sr=SAMPLE_RATE, mono=True)

    state1 = extractor.extract(audio1)
    state2 = extractor.extract(audio2)

    action1, _, _ = policy(state1, deterministic=True)
    action2, _, _ = policy(state2, deterministic=True)

    # 다른 음성 → 다른 state → 다른 action
    assert not torch.allclose(state1, state2), "두 음성의 state가 동일함"
    assert not torch.allclose(action1, action2), "다른 state인데 action이 동일함"


# ------------------------------------------------------------------ #
# 수동 실행 — 결과 바로 확인                                           #
# python tests/test_rl/test_pipeline.py                               #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import sys

    print("=" * 50)
    print("음성 → StateExtractor → PolicyNetwork 파이프라인")
    print("=" * 50)

    # 오디오 로드
    audio, sr = librosa.load(TEST_FILE, sr=SAMPLE_RATE, mono=True)
    print(f"\n[입력 음성]")
    print(f"  파일    : {TEST_FILE.split(chr(92))[-1]}")
    print(f"  길이    : {len(audio)/sr:.2f}초")

    extractor = StateExtractor(sample_rate=sr)
    policy = PolicyNetwork()

    # Step 1: 음성 → state
    state = extractor.extract(audio)
    print(f"\n[Step 1] StateExtractor")
    print(f"  state shape : {state.shape}")
    print(f"  state 값    : {[f'{v:.3f}' for v in state.tolist()]}")

    # Step 2: state → action
    action, log_prob, value = policy(state, deterministic=True)
    print(f"\n[Step 2] PolicyNetwork (deterministic)")
    print(f"  action shape   : {action.shape}")
    print(f"  action min/max : {action.min():.4f} / {action.max():.4f}")
    print(f"  log_prob       : {log_prob.item():.4f}")
    print(f"  value          : {value.item():.4f}")

    # 다른 음성과 비교
    file2 = os.path.join(_ROOT, "Original", "100", "121669", "100-121669-0001.wav")
    audio2, _ = librosa.load(file2, sr=SAMPLE_RATE, mono=True)
    state2 = extractor.extract(audio2)
    action2, _, _ = policy(state2, deterministic=True)

    print(f"\n[비교] 다른 음성 파일")
    print(f"  파일    : {file2.split(chr(92))[-1]}")
    print(f"  state 값: {[f'{v:.3f}' for v in state2.tolist()]}")
    print(f"  action min/max : {action2.min():.4f} / {action2.max():.4f}")
    print(f"\n  → state 동일? {torch.allclose(state, state2)}")
    print(f"  → action 동일? {torch.allclose(action, action2)}")

    sys.exit(0)
