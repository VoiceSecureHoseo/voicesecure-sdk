"""PolicyNetwork: PPO Actor-Critic 정책 네트워크.

역할:
    - Actor: state(36-dim)를 받아 노이즈 spectrogram(action)을 출력
    - Critic: state를 받아 현재 상태의 가치 V(s)를 출력

action 설계 (주파수 패턴 × 시간 게이팅):
    - freq_pattern: state → (n_freq,) — 주파수별 노이즈 방향/강도
    - time_gate:    state → (n_time,) — 시간별 노이즈 강도 (유성음 구간에 집중 등)
    - action[f, t] = tanh(freq_raw[f]) * sigmoid(time_raw[t])
      → 주파수 패턴이 시간 게이트로 변조됨
      → freq: [-1, 1] (방향 + 강도), time: [0, 1] (게이팅 비율)

타일링 대비 장점:
    - 시간 변화 반영 가능 (유성음/무성음 구간 구분)
    - 파라미터 수: hidden×(n_freq + n_time) = 64×357 (타일링 64×257과 비슷)
    - 외적(outer product)이라 n_freq × n_time 직접 출력(64×25700)보다 훨씬 작음
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

from voicesecure.types import ACTION_N_FREQ, ACTION_N_TIME, STATE_DIM, Action, State

_HIDDEN_DIM = 64
_LOG_STD_INIT = -0.5
_LOG_STD_MIN = -3.0
_LOG_STD_MAX = 1.0


class PolicyNetwork(nn.Module):
    """PPO Actor-Critic 정책 네트워크 (주파수×시간 게이팅 방식).

    freq_pattern × time_gate 외적으로 action (n_freq, n_time)을 생성.
    state가 36-dim 스펙트럼 요약 벡터이므로 시간 게이팅이 의미 있음.

    Args:
        state_dim: state 벡터 차원수. 기본값 STATE_DIM=36.
        n_freq: 노이즈 주파수 bin 수. 기본값 257.
        n_time: 노이즈 시간 프레임 수. 기본값 100.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        n_freq: int = ACTION_N_FREQ,
        n_time: int = ACTION_N_TIME,
    ) -> None:
        super().__init__()

        self.n_freq = n_freq
        self.n_time = n_time

        # 공유 trunk: state → 공통 표현
        self.shared = nn.Sequential(
            nn.Linear(state_dim, _HIDDEN_DIM),
            nn.LayerNorm(_HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(_HIDDEN_DIM, _HIDDEN_DIM),
            nn.Tanh(),
        )

        # Actor freq: 공통 표현 → 주파수 패턴 pre-activation (n_freq,)
        # tanh squash → [-1, 1]: 방향(±) + 강도(크기)
        self.actor_freq = nn.Linear(_HIDDEN_DIM, n_freq)
        self.log_std_freq = nn.Parameter(torch.full((n_freq,), _LOG_STD_INIT))

        # Actor time: 공통 표현 → 시간 게이트 pre-activation (n_time,)
        # sigmoid squash → [0, 1]: 각 프레임의 노이즈 강도 비율
        self.actor_time = nn.Linear(_HIDDEN_DIM, n_time)
        self.log_std_time = nn.Parameter(torch.full((n_time,), _LOG_STD_INIT))

        # Critic: 공통 표현 → V(s) 스칼라
        self.critic = nn.Linear(_HIDDEN_DIM, 1)

        self._init_weights()

    def forward(
        self, state: State, deterministic: bool = False
    ) -> tuple[Action, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """state → action, log_prob, value, freq_pattern, time_gate.

        action[f, t] = tanh(freq_raw[f]) * sigmoid(time_raw[t])

        Args:
            state: shape (36,) 또는 (batch, 36)
            deterministic: True이면 평균 action, False이면 샘플링

        Returns:
            action:       shape (n_freq, n_time) 또는 (batch, n_freq, n_time)
            log_prob:     shape () 또는 (batch,)
            value:        shape () 또는 (batch,)
            freq_pattern: shape (n_freq,) 또는 (batch, n_freq) — evaluate_actions 역산용
            time_gate:    shape (n_time,) 또는 (batch, n_time) — evaluate_actions 역산용
        """
        squeezed = state.dim() == 1
        if squeezed:
            state = state.unsqueeze(0)  # (36,) → (1, 36)

        shared_out = self.shared(state)  # (batch, hidden)

        # ── 주파수 패턴 ──────────────────────────────────────────────────
        freq_mean = self.actor_freq(shared_out)  # (batch, n_freq)
        log_std_f = self.log_std_freq.clamp(_LOG_STD_MIN, _LOG_STD_MAX)
        std_f = log_std_f.exp()
        dist_f = Normal(freq_mean, std_f)
        freq_raw = freq_mean if deterministic else dist_f.rsample()  # (batch, n_freq)
        freq_pattern = torch.tanh(freq_raw)  # (batch, n_freq) ∈ [-1,1]

        # tanh change-of-variables log_prob 보정
        log_prob_f = dist_f.log_prob(freq_raw).sum(dim=-1) - torch.log(
            1 - freq_pattern**2 + 1e-6
        ).sum(
            dim=-1
        )  # (batch,)

        # ── 시간 게이트 ──────────────────────────────────────────────────
        time_mean = self.actor_time(shared_out)  # (batch, n_time)
        log_std_t = self.log_std_time.clamp(_LOG_STD_MIN, _LOG_STD_MAX)
        std_t = log_std_t.exp()
        dist_t = Normal(time_mean, std_t)
        time_raw = time_mean if deterministic else dist_t.rsample()  # (batch, n_time)
        time_gate = torch.sigmoid(time_raw)  # (batch, n_time) ∈ [0,1]

        # sigmoid change-of-variables log_prob 보정
        # sigmoid(x) = s, log|ds/dx| = log(s*(1-s))
        log_prob_t = dist_t.log_prob(time_raw).sum(dim=-1) - torch.log(
            time_gate * (1 - time_gate) + 1e-6
        ).sum(
            dim=-1
        )  # (batch,)

        # ── 외적으로 action 조합 ─────────────────────────────────────────
        # action[b, f, t] = freq_pattern[b, f] * time_gate[b, t]
        action = torch.bmm(
            freq_pattern.unsqueeze(2),  # (batch, n_freq, 1)
            time_gate.unsqueeze(1),  # (batch, 1, n_time)
        )  # (batch, n_freq, n_time)

        log_prob = log_prob_f + log_prob_t  # 독립 분포 합산
        value = self.critic(shared_out).squeeze(-1)  # (batch,)

        if squeezed:
            action = action.squeeze(0)  # (n_freq, n_time)
            log_prob = log_prob.squeeze(0)
            value = value.squeeze(0)
            freq_pattern = freq_pattern.squeeze(0)  # (n_freq,)
            time_gate = time_gate.squeeze(0)  # (n_time,)

        return action, log_prob, value, freq_pattern, time_gate

    def evaluate_actions(
        self,
        states: State,
        actions: Action,
        freq_patterns: torch.Tensor,
        time_gates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PPO 업데이트 시 기존 action에 대한 log_prob, value, entropy를 재계산한다.

        forward()에서 저장한 freq_pattern(tanh), time_gate(sigmoid)를 받아
        각각 atanh/logit으로 역산한다 (SB3 TanhBijector.inverse 방식).
        외적 구조에서 action 자체로는 분해 불가능하므로 두 값을 별도 저장하는 것이 필수.

        Args:
            states:        shape (batch, 36)
            actions:       shape (batch, n_freq, n_time)  — 미사용, 호환성 유지용
            freq_patterns: shape (batch, n_freq) — forward()에서 반환한 tanh(freq_raw)
            time_gates:    shape (batch, n_time) — forward()에서 반환한 sigmoid(time_raw)

        Returns:
            log_probs: shape (batch,)
            values:    shape (batch,)
            entropy:   scalar
        """
        shared_out = self.shared(states)

        # ── 주파수 패턴 역산 (SB3 TanhBijector.inverse 방식) ──────────────
        freq_mean = self.actor_freq(shared_out)  # (batch, n_freq)
        log_std_f = self.log_std_freq.clamp(_LOG_STD_MIN, _LOG_STD_MAX)
        dist_f = Normal(freq_mean, log_std_f.exp())

        fp = freq_patterns.clamp(-1 + 1e-6, 1 - 1e-6)  # (batch, n_freq)
        freq_raw = torch.atanh(fp)  # 정확한 역산
        log_prob_f = dist_f.log_prob(freq_raw).sum(dim=-1) - torch.log(1 - fp**2 + 1e-6).sum(
            dim=-1
        )  # (batch,)

        # ── 시간 게이트 역산 (sigmoid 역함수 = logit) ─────────────────────
        time_mean = self.actor_time(shared_out)  # (batch, n_time)
        log_std_t = self.log_std_time.clamp(_LOG_STD_MIN, _LOG_STD_MAX)
        dist_t = Normal(time_mean, log_std_t.exp())

        tg = time_gates.clamp(1e-6, 1 - 1e-6)  # (batch, n_time)
        time_raw = torch.logit(tg)  # 정확한 역산
        log_prob_t = dist_t.log_prob(time_raw).sum(dim=-1) - torch.log(tg * (1 - tg) + 1e-6).sum(
            dim=-1
        )  # (batch,)

        # entropy: Normal 분포의 해석적 entropy (= 0.5 + 0.5*log(2π) + log_std)
        entropy_f = dist_f.entropy().sum(dim=-1).mean()
        entropy_t = dist_t.entropy().sum(dim=-1).mean()
        entropy = entropy_f + entropy_t

        log_probs = log_prob_f + log_prob_t
        values = self.critic(shared_out).squeeze(-1)
        return log_probs, values, entropy

    def _init_weights(self) -> None:
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

        # freq/time head 모두 작은 초기값 — 초기 action이 중간 범위에서 시작
        for linear in (self.actor_freq, self.actor_time):
            nn.init.orthogonal_(linear.weight, gain=0.01)
            nn.init.zeros_(linear.bias)

        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)
