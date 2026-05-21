"""RLAgent: PolicyNetwork를 들고 있으면서 PPO로 학습시키는 관리자.

역할:
    1. act()   — 음성 입력 → state 추출 → action(노이즈) 반환 (추론)
    2. update() — 쌓인 Transition들로 PPO 업데이트 (학습)
    3. save() / load() — 체크포인트 저장/불러오기

PPO 핵심 로직:
    ratio = exp(새 log_prob - 이전 log_prob)
    loss  = -min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage)
    → ratio가 1±ε 범위를 벗어나면 잘라서 한 번에 너무 많이 바뀌지 않게 제약
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from voicesecure.rl.policy import PolicyNetwork
from voicesecure.rl.state import StateExtractor
from voicesecure.types import Action, AudioArray, Transition

logger = logging.getLogger(__name__)

# PPO 하이퍼파라미터 기본값
_PPO_EPSILON = 0.2  # clipping 범위 (ratio를 1±ε 안으로 제한)
_PPO_EPOCHS = 4  # 같은 transition으로 몇 번 업데이트할지
_GAMMA = 0.99  # 미래 reward 할인율
_GAE_LAMBDA = 0.95  # GAE(Generalized Advantage Estimation) lambda
_LR = 3e-4  # learning rate
_VALUE_COEF = 0.5  # value loss 가중치
_ENTROPY_COEF = 0.01  # entropy bonus 가중치 (탐색 장려)


class RLAgent:
    """PPO 기반 RL Agent.

    StateExtractor와 PolicyNetwork를 들고 있으면서
    추론(act)과 학습(update)을 담당한다.

    Args:
        config: 하이퍼파라미터 dict. 없으면 기본값 사용.
                예: {"lr": 3e-4, "ppo_epsilon": 0.2, "gamma": 0.99}
    """

    def __init__(self, config: dict | None = None) -> None:
        config = config or {}

        self.extractor = StateExtractor()
        self.policy = PolicyNetwork()

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.get("lr", _LR),
        )

        # PPO 하이퍼파라미터 (config로 덮어쓰기 가능)
        self.epsilon = config.get("ppo_epsilon", _PPO_EPSILON)
        self.ppo_epochs = config.get("ppo_epochs", _PPO_EPOCHS)
        self.gamma = config.get("gamma", _GAMMA)
        self.gae_lambda = config.get("gae_lambda", _GAE_LAMBDA)
        self.value_coef = config.get("value_coef", _VALUE_COEF)
        self.entropy_coef = config.get("entropy_coef", _ENTROPY_COEF)

    def act(
        self, audio: AudioArray, deterministic: bool = False
    ) -> tuple[Action, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """음성 입력을 받아 노이즈(action)를 생성한다.

        추론 시에는 deterministic=True로 호출.
        학습 시에는 deterministic=False로 호출해 탐색(exploration) 허용.

        Args:
            audio: 원본 음성, shape (num_samples,), float32
            deterministic: True면 평균 action, False면 분포에서 샘플링

        Returns:
            state:        추출된 state 벡터, shape (STATE_DIM,)
            action:       노이즈 spectrogram, shape (n_freq, n_time)
            log_prob:     이 action의 로그확률 (PPO 학습용)
            value:        Critic의 V(s) (PPO 학습용)
            freq_pattern: tanh(freq_raw), shape (n_freq,) — evaluate_actions 역산용
            time_gate:    sigmoid(time_raw), shape (n_time,) — evaluate_actions 역산용
        """
        with torch.no_grad():
            state = self.extractor.extract(audio)
            action, log_prob, value, freq_pattern, time_gate = self.policy(
                state, deterministic=deterministic
            )
        return state, action, log_prob, value, freq_pattern, time_gate

    def update(self, transitions: list[Transition]) -> dict:
        """쌓인 Transition들로 PPO 업데이트를 수행한다.

        PPO 로직:
            1. GAE로 advantage 계산
            2. ppo_epochs번 반복하며 policy/value loss 계산 및 업데이트

        Args:
            transitions: 학습에 사용할 Transition 리스트
                         각 Transition: (state, action, reward, next_state, done, log_prob, value)

        Returns:
            metrics: {"policy_loss": float, "value_loss": float, "entropy": float}
        """
        # ── Transition 리스트 → 텐서로 변환 ──────────────────────────
        states = torch.stack([t.state for t in transitions])               # (N, STATE_DIM)
        actions = torch.stack([t.action for t in transitions])             # (N, n_freq, n_time)
        old_log_probs = torch.stack([t.log_prob.detach() for t in transitions])  # (N,)
        old_values = torch.stack([t.value.detach() for t in transitions])        # (N,)
        freq_patterns = torch.stack([t.freq_pattern.detach() for t in transitions])  # (N, n_freq)
        time_gates = torch.stack([t.time_gate.detach() for t in transitions])        # (N, n_time)

        # rewards는 old_values와 같은 device로 생성 (GPU 환경 대비)
        device = old_values.device
        rewards = torch.tensor(
            [t.reward for t in transitions], dtype=torch.float32, device=device
        )

        # ── advantage 계산 ────────────────────────────────────────────
        # 1 step = 1 episode (done=True 고정) 구조이므로 next_value는 항상 0.
        # advantage = reward - V(s) 로 단순화 (GAE next_value forward pass 불필요)
        advantages = self._compute_advantage(rewards, old_values)
        returns = rewards  # target = 즉각 reward (done=True → 미래 가치 없음)

        # advantage 정규화 (학습 안정화)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── PPO 업데이트 (같은 데이터로 ppo_epochs번 반복) ────────────
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.ppo_epochs):
            # 현재 정책으로 다시 forward — 기존 action에 대한 log_prob 재계산
            # 새로 샘플링하는 게 아니라 이전에 선택한 action을 현재 정책으로 평가
            new_log_probs, new_values, entropy = self.policy.evaluate_actions(
                states, actions, freq_patterns, time_gates
            )

            # ratio = 새 정책 확률 / 이전 정책 확률
            ratio = (new_log_probs - old_log_probs).exp()

            # PPO clipped loss
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss: Critic이 예측한 V(s)와 실제 return의 차이
            value_loss = nn.functional.mse_loss(new_values, returns)

            # 전체 loss (entropy는 Normal 분포의 해석적 entropy)
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            # gradient clipping: 너무 큰 gradient가 가중치를 망가뜨리는 것 방지
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        metrics = {
            "policy_loss": total_policy_loss / self.ppo_epochs,
            "value_loss": total_value_loss / self.ppo_epochs,
            "entropy": total_entropy / self.ppo_epochs,
        }
        logger.debug("PPO update: %s", metrics)
        return metrics

    def save(self, path: str) -> None:
        """학습된 PolicyNetwork 가중치를 저장한다."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        logger.info("체크포인트 저장: %s", path)

    def load(self, path: str) -> None:
        """저장된 가중치를 불러온다."""
        checkpoint = torch.load(path, map_location="cpu")
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("체크포인트 로드: %s", path)

    def _compute_advantage(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """advantage 계산.

        1 step = 1 episode (done=True 고정) 구조이므로 next_value = 0.
        advantage = reward - V(s)
        multi-step 구조로 바뀌면 GAE로 교체 필요.
        """
        return rewards - values
