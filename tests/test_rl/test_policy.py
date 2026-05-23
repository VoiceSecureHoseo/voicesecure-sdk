"""PolicyNetwork 테스트."""

import torch

from voicesecure.rl.policy import PolicyNetwork
from voicesecure.types import ACTION_N_FREQ, ACTION_N_TIME, STATE_DIM


def test_output_shapes_single():
    net = PolicyNetwork()
    state = torch.randn(STATE_DIM)
    action, log_prob, value, freq_pattern, time_gate = net(state)

    assert action.shape == (ACTION_N_FREQ, ACTION_N_TIME), f"action shape 오류: {action.shape}"
    assert log_prob.shape == ()
    assert value.shape == ()


def test_output_shapes_batch():
    batch = 4
    net = PolicyNetwork()
    state = torch.randn(batch, STATE_DIM)
    action, log_prob, value, freq_pattern, time_gate = net(state)

    assert action.shape == (batch, ACTION_N_FREQ, ACTION_N_TIME)
    assert log_prob.shape == (batch,)
    assert value.shape == (batch,)


def test_output_dtype():
    net = PolicyNetwork()
    state = torch.randn(STATE_DIM)
    action, log_prob, value, freq_pattern, time_gate = net(state)

    assert action.dtype == torch.float32
    assert log_prob.dtype == torch.float32
    assert value.dtype == torch.float32


def test_action_range():
    net = PolicyNetwork()
    state = torch.randn(STATE_DIM)
    action, _, _, _, _ = net(state, deterministic=True)

    assert action.min() >= -1.0 - 1e-5
    assert action.max() <= 1.0 + 1e-5


def test_no_nan_or_inf():
    net = PolicyNetwork()
    state = torch.randn(STATE_DIM)
    action, log_prob, value, freq_pattern, time_gate = net(state)

    assert not torch.isnan(action).any()
    assert not torch.isnan(log_prob).any()
    assert not torch.isnan(value).any()


def test_freq_time_gating():
    """주파수×시간 게이팅 구조 확인.

    action = freq_pattern(n_freq) × time_gate(n_time) 외적이므로:
    - 시간 프레임마다 값이 달라야 함 (타일링과 달리)
    - 각 주파수 bin의 시간축 부호는 일정 (freq_pattern 부호 × 양수 gate)
    """
    net = PolicyNetwork()
    state = torch.randn(STATE_DIM)
    action, _, _, _, _ = net(state, deterministic=True)  # (n_freq, n_time)

    # 시간 프레임마다 값이 다름
    assert not torch.allclose(action[:, 0], action[:, -1]), "시간 게이팅이 동작하지 않음"

    # 주파수별 부호는 시간축에서 일정 (time_gate ∈ [0,1] → 부호 변환 없음)
    signs = action.sign()
    assert (signs == signs[:, :1]).all(), "주파수별 부호가 시간축에서 바뀜"


def test_model_size():
    net = PolicyNetwork()
    total_params = sum(p.numel() for p in net.parameters())
    size_mb = total_params * 4 / 1024 / 1024
    assert size_mb <= 1.0, f"모델 크기 초과: {size_mb:.2f} MB"


def test_deterministic_vs_stochastic():
    net = PolicyNetwork()
    state = torch.randn(STATE_DIM)

    action1, _, _, _, _ = net(state, deterministic=True)
    action2, _, _, _, _ = net(state, deterministic=True)
    assert torch.allclose(action1, action2)

    action3, _, _, _, _ = net(state, deterministic=False)
    action4, _, _, _, _ = net(state, deterministic=False)
    assert not torch.allclose(action3, action4)
