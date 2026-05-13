"""RLAgent 전체 파이프라인 테스트 (더미 모듈 사용).

실제 모듈(Masker, Mixer, Evaluator)이 완성되면
dummy_modules.py의 더미를 실제로 교체하면 된다.

실행: python tests/test_rl/test_agent.py
"""

import os
import sys

import librosa
import numpy as np
import pytest
import torch

from voicesecure.rl.agent import RLAgent
from voicesecure.types import ACTION_N_FREQ, ACTION_N_TIME, SAMPLE_RATE, Transition

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from dummy_modules import DummyMasker, DummyMixer, DummyReward

# ★ 테스트할 파일 경로 — 원하는 파일로 바꾸면 됨
# 프로젝트 루트(voicesecure-sdk/) 기준 상대경로
_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEST_FILE = os.path.join(_ROOT, "Original", "100", "121669", "100-121669-0000.wav")


def _load_audio():
    audio, _ = librosa.load(TEST_FILE, sr=SAMPLE_RATE, mono=True)
    return audio


@pytest.mark.slow
def test_act_output_shape():
    """act() 출력 shape 확인."""
    audio = _load_audio()
    agent = RLAgent()
    action, log_prob, value = agent.act(audio)

    print(f"\n  action shape  : {action.shape}")
    print(f"  log_prob shape: {log_prob.shape}")
    print(f"  value shape   : {value.shape}")

    assert action.shape == (ACTION_N_FREQ, ACTION_N_TIME)
    assert log_prob.shape == ()
    assert value.shape == ()


@pytest.mark.slow
def test_act_no_nan():
    """act() 출력에 NaN/Inf 없는지 확인."""
    audio = _load_audio()
    agent = RLAgent()
    action, log_prob, value = agent.act(audio)

    print(
        f"\n  action   NaN:{torch.isnan(action).any().item()}  Inf:{torch.isinf(action).any().item()}"
    )
    print(f"  log_prob NaN:{torch.isnan(log_prob).item()}  값:{log_prob.item():.4f}")
    print(f"  value    NaN:{torch.isnan(value).item()}  값:{value.item():.4f}")
    print(f"  action min/max: {action.min():.4f} / {action.max():.4f}")

    assert not torch.isnan(action).any()
    assert not torch.isnan(log_prob)
    assert not torch.isnan(value)


@pytest.mark.slow
def test_full_pipeline_one_step():
    """음성 → act → 더미 Masker/Mixer → 더미 reward → 전체 1스텝 동작 확인."""
    audio = _load_audio()
    agent = RLAgent()

    # ── 더미 모듈 초기화 ──────────────────────────────────────────────
    # 실제 모듈로 교체 시 아래 줄만 바꾸면 됨
    masker = DummyMasker()
    mixer = DummyMixer()
    reward_fn = DummyReward()

    # Step 1: 음성 → state → action (노이즈)
    action, log_prob, value = agent.act(audio, deterministic=False)

    # Step 2: 더미 Masker → safe noise
    safe_noise = masker.clamp(audio, action)

    # Step 3: 더미 Mixer → 변형 음성
    modified_audio = mixer.mix(audio, safe_noise)

    # Step 4: 더미 reward
    reward = reward_fn.compute(modified_audio)

    print(f"\n  원본 audio    : shape={audio.shape}  max={np.abs(audio).max():.4f}")
    print(f"  action        : shape={action.shape}  max={action.abs().max():.4f}")
    print(
        f"  safe_noise    : shape={safe_noise.shape}  max={safe_noise.abs().max():.6f}  (DummyMasker clamp±0.01)"
    )
    print(f"  modified_audio: shape={modified_audio.shape}  max={np.abs(modified_audio).max():.4f}")
    print(f"  reward        : {reward:.4f}  (DummyReward: 랜덤 0~1)")

    # 검증
    assert modified_audio.shape == audio.shape, "변형 음성 길이가 원본과 다름"
    assert np.all(np.abs(modified_audio) <= 1.0), "변형 음성이 [-1,1] 범위 초과"
    assert isinstance(reward, float), "reward가 float이 아님"


@pytest.mark.slow
def test_ppo_update_one_step():
    """더미 Transition으로 PPO 업데이트 1회 동작 확인."""
    audio = _load_audio()
    agent = RLAgent()
    masker = DummyMasker()
    mixer = DummyMixer()
    reward_fn = DummyReward()

    # Transition 여러 개 수집
    transitions = []
    rewards = []
    n_steps = 8
    print(f"\n  [에피소드 수집] 총 {n_steps}스텝")
    for i in range(n_steps):
        action, log_prob, value = agent.act(audio, deterministic=False)
        safe_noise = masker.clamp(audio, action)
        modified = mixer.mix(audio, safe_noise)
        reward = reward_fn.compute(modified)
        rewards.append(reward)
        print(f"  step {i+1}/{n_steps}: reward={reward:.4f}  action_max={action.abs().max():.4f}")

        # 다음 state (더미: 같은 음성 재사용)
        next_action, _, next_value = agent.act(audio, deterministic=False)

        transitions.append(
            Transition(
                state=agent.extractor.extract(audio),
                action=action,
                reward=reward,
                next_state=agent.extractor.extract(audio),
                done=False,
                log_prob=log_prob,
                value=value,
            )
        )

    # PPO 업데이트 3회
    n_updates = 3
    print(f"\n  [PPO 업데이트] 총 {n_updates}회")
    for update_i in range(n_updates):
        metrics = agent.update(transitions)
        print(
            f"  update {update_i+1}/{n_updates}: policy_loss={metrics['policy_loss']:.4f}  value_loss={metrics['value_loss']:.4f}  entropy={metrics['entropy']:.4f}"
        )

    print(f"\n  수집 스텝      : {len(transitions)}개")
    print(
        f"  reward 평균    : {np.mean(rewards):.4f}  (min={min(rewards):.4f}  max={max(rewards):.4f})"
    )
    print(
        f"  NaN 없음       : policy={not np.isnan(metrics['policy_loss'])}  value={not np.isnan(metrics['value_loss'])}"
    )

    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert not np.isnan(metrics["policy_loss"])
    assert not np.isnan(metrics["value_loss"])


@pytest.mark.slow
def test_save_and_load(tmp_path):
    """save → load 후 동일한 결과 확인."""
    audio = _load_audio()
    agent = RLAgent()

    # 저장 전 결과
    action_before, _, _ = agent.act(audio, deterministic=True)

    # 저장 후 불러오기
    path = str(tmp_path / "test_checkpoint.pt")
    agent.save(path)
    agent.load(path)

    # 불러온 후 결과
    action_after, _, _ = agent.act(audio, deterministic=True)

    match = torch.allclose(action_before, action_after)
    max_diff = (action_before - action_after).abs().max().item()
    print(f"\n  체크포인트 경로 : {path}")
    print(f"  저장 전 action_max : {action_before.abs().max():.6f}")
    print(f"  불러온 후 action_max: {action_after.abs().max():.6f}")
    print(f"  최대 차이         : {max_diff:.2e}")
    print(f"  일치 여부          : {match}")

    assert match, "save/load 후 결과가 달라짐"


# ------------------------------------------------------------------ #
# 수동 실행 — 전체 파이프라인 결과 바로 확인                            #
# python tests/test_rl/test_agent.py                                  #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("전체 파이프라인 테스트 (더미 Masker/Mixer/Reward 사용)")
    print("=" * 60)

    audio, sr = librosa.load(TEST_FILE, sr=SAMPLE_RATE, mono=True)
    agent = RLAgent()
    masker = DummyMasker()
    mixer = DummyMixer()
    reward_fn = DummyReward()

    print(f"\n[입력 음성] {TEST_FILE.split(chr(92))[-1]}  ({len(audio)/sr:.2f}초)")

    # 에피소드 5개 돌려보기
    transitions = []
    print("\n[에피소드 실행]")
    for ep in range(5):
        action, log_prob, value = agent.act(audio, deterministic=False)
        safe_noise = masker.clamp(audio, action)
        modified = mixer.mix(audio, safe_noise)
        reward = reward_fn.compute(modified)

        transitions.append(
            Transition(
                state=agent.extractor.extract(audio),
                action=action,
                reward=reward,
                next_state=agent.extractor.extract(audio),
                done=(ep == 4),
                log_prob=log_prob,
                value=value,
            )
        )
        print(f"  ep {ep+1}: reward={reward:.4f}  action_max={action.abs().max():.4f}")

    # PPO 업데이트
    print("\n[PPO 업데이트]")
    metrics = agent.update(transitions)
    print(f"  policy_loss : {metrics['policy_loss']:.4f}")
    print(f"  value_loss  : {metrics['value_loss']:.4f}")
    print(f"  entropy     : {metrics['entropy']:.4f}")

    # 업데이트 후 action 변화 확인
    action_after, _, _ = agent.act(audio, deterministic=True)
    print(f"\n[업데이트 후] action_max={action_after.abs().max():.6f}")

    sys.exit(0)
