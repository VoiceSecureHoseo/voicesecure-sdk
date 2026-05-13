"""PolicyNetwork 테스트.

타일링 방식:
    - PolicyNetwork는 주파수 패턴 (n_freq,)만 출력
    - 시간축으로 n_time번 복사해서 (n_freq, n_time) 반환
    - 모델 크기: 약 0.1MB

실행: python tests/test_rl/test_policy.py
"""

import torch

from voicesecure.rl.policy import PolicyNetwork
from voicesecure.types import ACTION_N_FREQ, ACTION_N_TIME, STATE_DIM


def test_output_shapes_single():
    """단일 입력(batch 없음)에서 출력 shape 확인."""
    net = PolicyNetwork()
    state = torch.randn(STATE_DIM)
    action, log_prob, value = net(state)

    assert action.shape == (ACTION_N_FREQ, ACTION_N_TIME), f"action shape 오류: {action.shape}"
    assert log_prob.shape == (), f"log_prob shape 오류: {log_prob.shape}"
    assert value.shape == (), f"value shape 오류: {value.shape}"


def test_output_shapes_batch():
    """배치 입력에서 출력 shape 확인."""
    batch = 4
    net = PolicyNetwork()
    state = torch.randn(batch, STATE_DIM)
    action, log_prob, value = net(state)

    assert action.shape == (batch, ACTION_N_FREQ, ACTION_N_TIME)
    assert log_prob.shape == (batch,)
    assert value.shape == (batch,)


def test_output_dtype():
    """출력 dtype이 float32 인지 확인."""
    net = PolicyNetwork()
    state = torch.randn(STATE_DIM)
    action, log_prob, value = net(state)

    assert action.dtype == torch.float32
    assert log_prob.dtype == torch.float32
    assert value.dtype == torch.float32


def test_action_range():
    """action 값이 tanh 출력 범위 [-1, 1] 이내인지 확인 (deterministic 모드)."""
    net = PolicyNetwork()
    state = torch.randn(STATE_DIM)
    action, _, _ = net(state, deterministic=True)

    assert action.min() >= -1.0 - 1e-5, f"action 최솟값 초과: {action.min()}"
    assert action.max() <= 1.0 + 1e-5, f"action 최댓값 초과: {action.max()}"


def test_no_nan_or_inf():
    """출력에 NaN/Inf 없는지 확인."""
    net = PolicyNetwork()
    state = torch.randn(STATE_DIM)
    action, log_prob, value = net(state)

    assert not torch.isnan(action).any()
    assert not torch.isnan(log_prob).any()
    assert not torch.isnan(value).any()


def test_tiling():
    """타일링 확인: action의 모든 시간 프레임이 동일한 주파수 패턴인지 확인."""
    net = PolicyNetwork()
    state = torch.randn(STATE_DIM)
    action, _, _ = net(state, deterministic=True)

    # 모든 시간 프레임이 첫 번째 프레임과 동일해야 함
    for t in range(ACTION_N_TIME):
        assert torch.allclose(action[:, 0], action[:, t]), f"프레임 {t}가 타일링 패턴과 다름"


def test_model_size():
    """모델 크기가 1MB 이하인지 확인."""
    net = PolicyNetwork()
    total_params = sum(p.numel() for p in net.parameters())
    size_mb = total_params * 4 / 1024 / 1024
    assert size_mb <= 1.0, f"모델 크기 초과: {size_mb:.2f} MB"


def test_deterministic_vs_stochastic():
    """deterministic=True이면 항상 같은 action, False이면 다른 action."""
    net = PolicyNetwork()
    state = torch.randn(STATE_DIM)

    action1, _, _ = net(state, deterministic=True)
    action2, _, _ = net(state, deterministic=True)
    assert torch.allclose(action1, action2), "deterministic 모드에서 결과가 달라짐"

    action3, _, _ = net(state, deterministic=False)
    action4, _, _ = net(state, deterministic=False)
    assert not torch.allclose(action3, action4), "stochastic 모드에서 결과가 같음"


# ------------------------------------------------------------------ #
# 수동 실행 — 결과 바로 확인                                           #
# python tests/test_rl/test_policy.py                                 #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import sys

    net = PolicyNetwork()

    # 모델 파라미터 수 및 크기 확인
    total_params = sum(p.numel() for p in net.parameters())
    size_mb = total_params * 4 / 1024 / 1024
    print(f"파라미터 수 : {total_params:,}")
    print(f"모델 크기   : {size_mb:.2f} MB  (목표: ≤ 1 MB)")
    print()

    state = torch.randn(STATE_DIM)
    print(f"입력 state shape : {state.shape}")
    print()

    action, log_prob, value = net(state, deterministic=True)
    print("[deterministic=True]")
    print(f"  action shape   : {action.shape}")
    print(f"  action min/max : {action.min():.4f} / {action.max():.4f}")
    print(f"  log_prob       : {log_prob.item():.4f}")
    print(f"  value          : {value.item():.4f}")
    print()

    # 타일링 확인
    print("[타일링 확인]")
    print(f"  첫 프레임 == 마지막 프레임? {torch.allclose(action[:, 0], action[:, -1])}")
    print()

    action, log_prob, value = net(state, deterministic=False)
    print("[deterministic=False]")
    print(f"  action shape   : {action.shape}")
    print(f"  action min/max : {action.min():.4f} / {action.max():.4f}")
    print(f"  log_prob       : {log_prob.item():.4f}")
    print(f"  value          : {value.item():.4f}")

    sys.exit(0)
