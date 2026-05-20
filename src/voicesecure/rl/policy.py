"""PolicyNetwork: PPO Actor-Critic 정책 네트워크.

역할:
    - Actor: state(16-dim)를 받아 노이즈 spectrogram(action)을 출력
    - Critic: state를 받아 현재 상태의 가치 V(s)를 출력

이 클래스는 순수 신경망이다. RL 로직(PPO 업데이트)은 RLAgent에서 담당한다.

action 설계 (타일링 방식):
    - PolicyNetwork는 주파수 패턴 (n_freq,) = (257,) 만 출력
    - 이 패턴을 시간축으로 n_time(100)번 복사해서 (257, 100) 만들기
    - 이유: 16-dim state로 시간 변화를 반영하는 건 불가능하므로
            주파수별 패턴만 결정하고 시간축은 동일하게 유지
    - 장점: 파라미터 수 대폭 감소 (300×25700 → 300×257), log_prob NaN 방지
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

from voicesecure.types import ACTION_N_FREQ, ACTION_N_TIME, STATE_DIM, Action, State

# Actor/Critic 공유 hidden layer 크기
# 타일링 방식이라 마지막 레이어가 hidden → 257 dim으로 작음
# H=64로도 충분, 모델 크기 << 1MB
_HIDDEN_DIM = 64

# 정책 분포의 log_std 초기값 및 범위
_LOG_STD_INIT = -0.5
_LOG_STD_MIN = -3.0
_LOG_STD_MAX = 1.0


class PolicyNetwork(nn.Module):
    """PPO Actor-Critic 정책 네트워크 (타일링 방식).

    Actor는 주파수 패턴 (n_freq,)만 출력하고,
    forward에서 시간축으로 타일링해 (n_freq, n_time)으로 확장한다.

    Args:
        state_dim: state 벡터 차원수. 기본값 STATE_DIM=16.
        n_freq: 노이즈 주파수 bin 수. 기본값 257 (n_fft=512 기준).
        n_time: 노이즈 시간 프레임 수. 기본값 100 (1초 청크 기준).
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

        # ── 공유 trunk ──────────────────────────────────────────────
        # state(16-dim)에서 공통 표현을 추출
        # Actor와 Critic이 공유 → 파라미터 절감
        self.shared = nn.Sequential(
            nn.Linear(state_dim, _HIDDEN_DIM),
            nn.LayerNorm(_HIDDEN_DIM),  # MFCC(-537)와 rms(0.04) 스케일 차이 정규화
            nn.Tanh(),
            nn.Linear(_HIDDEN_DIM, _HIDDEN_DIM),
            nn.Tanh(),
        )

        # ── Actor head ───────────────────────────────────────────────
        # 공유 표현 → 주파수 패턴 (n_freq,) 출력
        # 타일링으로 (n_freq, n_time) 확장하므로 출력 차원이 작음
        # [개선 고려] 현재는 시간축이 동일한 패턴 반복 (타일링).
        #            RNN/Transformer 도입 시 시간에 따른 동적 변조 가능.
        # TODO: 출력을 (n_freq×2,)로 확장해서 노이즈 방향 + 스펙트럼 워핑 방향을
        #       동시에 학습. Mixer에서 워핑 적용 로직 추가 필요.
        #       변경 범위: policy.py, types.py, masker.py, mixer.py, agent.py, train.py
        self.actor_mean = nn.Sequential(
            nn.Linear(_HIDDEN_DIM, n_freq),
            nn.Tanh(),  # [-1, 1] 범위 제한 (Masker clamp 전 안정화)
        )

        # action 분포의 log 표준편차 (학습 가능한 파라미터)
        # n_freq 차원 — 주파수별로 다른 탐색 정도를 가질 수 있음
        self.log_std = nn.Parameter(torch.full((n_freq,), _LOG_STD_INIT))

        # ── Critic head ──────────────────────────────────────────────
        # 공유 표현 → V(s) 스칼라
        # 학습 시 PPO advantage 계산에만 사용, 추론 시 불필요
        self.critic = nn.Linear(_HIDDEN_DIM, 1)

        self._init_weights()

    def forward(
        self, state: State, deterministic: bool = False
    ) -> tuple[Action, torch.Tensor, torch.Tensor]:
        """state를 받아 action, log_prob, value를 반환한다.

        Args:
            state: shape (16,) 또는 (batch, 16), float32
            deterministic: True이면 평균 action (추론 시),
                           False이면 분포에서 샘플링 (학습 시 탐색).

        Returns:
            action:   shape (n_freq, n_time) 또는 (batch, n_freq, n_time)
                      주파수 패턴을 시간축으로 타일링한 노이즈 spectrogram.
            log_prob: shape () 또는 (batch,)
            value:    shape () 또는 (batch,)
        """
        squeezed = state.dim() == 1
        if squeezed:
            state = state.unsqueeze(0)  # (16,) → (1, 16)

        batch = state.shape[0]

        # ── 공유 trunk ───────────────────────────────────────────────
        shared_out = self.shared(state)  # (batch, hidden)

        # ── Actor: 주파수 패턴 생성 ──────────────────────────────────
        mean = self.actor_mean(shared_out)  # (batch, n_freq)

        log_std = self.log_std.clamp(_LOG_STD_MIN, _LOG_STD_MAX)
        std = log_std.exp()  # (n_freq,)

        dist = Normal(mean, std)

        if deterministic:
            freq_pattern = mean  # (batch, n_freq)
        else:
            freq_pattern = dist.rsample()  # (batch, n_freq)

        # log_prob: n_freq 차원 평균 → 스칼라
        log_prob = dist.log_prob(freq_pattern).mean(dim=-1)  # (batch,)

        # 주파수 패턴을 시간축으로 타일링: (batch, n_freq) → (batch, n_freq, n_time)
        # unsqueeze로 시간 차원 추가 후 n_time번 반복
        action = freq_pattern.unsqueeze(-1).expand(batch, self.n_freq, self.n_time)
        # expand는 메모리 공유라 contiguous()로 복사
        action = action.contiguous()

        # ── Critic ───────────────────────────────────────────────────
        value = self.critic(shared_out).squeeze(-1)  # (batch,)

        if squeezed:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            value = value.squeeze(0)

        return action, log_prob, value

    def evaluate_actions(self, states: State, actions: Action) -> tuple[torch.Tensor, torch.Tensor]:
        """PPO 업데이트 시 기존 action에 대한 log_prob와 value를 재계산한다.

        타일링된 action (batch, n_freq, n_time)에서
        주파수 패턴 (batch, n_freq)만 추출해서 log_prob 계산.

        Args:
            states:  shape (batch, 16)
            actions: shape (batch, n_freq, n_time)

        Returns:
            log_probs: shape (batch,)
            values:    shape (batch,)
        """
        shared_out = self.shared(states)  # (batch, hidden)
        mean = self.actor_mean(shared_out)  # (batch, n_freq)

        log_std = self.log_std.clamp(_LOG_STD_MIN, _LOG_STD_MAX)
        std = log_std.exp()

        dist = Normal(mean, std)

        # 타일링된 action에서 시간 첫 프레임만 추출 (어느 프레임이든 동일)
        freq_pattern = actions[:, :, 0]  # (batch, n_freq)
        log_probs = dist.log_prob(freq_pattern).mean(dim=-1)  # (batch,)

        values = self.critic(shared_out).squeeze(-1)
        return log_probs, values

    def _init_weights(self) -> None:
        """가중치 초기화.

        Actor 마지막 레이어를 작은 값으로 초기화.
        이유: 초반 action이 크면 Masker clamp 후 유효 노이즈가 거의 없어지므로
              작은 노이즈부터 시작해서 점진적으로 키우는 게 유리.
        """
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

        # Actor 마지막 Linear: 작은 초기값
        last_actor = self.actor_mean[0]
        nn.init.orthogonal_(last_actor.weight, gain=0.01)
        nn.init.zeros_(last_actor.bias)

        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)
