"""train.py 훈련 루프 통합 테스트.

로컬(Windows) 환경 기준:
    - ECAPA-TDNN: 실제 모델 사용
    - CAM++ / CosyVoice3: 더미 모듈 (Colab 전용)

전체 train() 대신 내부 로직만 직접 실행:
    act → clamp → mix → reward 계산 → transition → PPO update
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from voicesecure.evaluators.adapters.ecapa_tdnn import ECAPATDNNAdapter
from voicesecure.evaluators.base import cosine_distance
from voicesecure.modulation.masker import PsychoacousticMasker
from voicesecure.modulation.mixer import Mixer
from voicesecure.rl.agent import RLAgent
from voicesecure.types import SAMPLE_RATE, AudioArray, Embedding, Transition

pytestmark = pytest.mark.slow

# ── 더미 어댑터 ───────────────────────────────────────────────────────────────


class _DummyCosyVoiceAdapter:
    """CAM++ + CosyVoice3 클로닝 더미 (Colab 전용 모듈 대체)."""

    EMB_DIM = 192

    def extract_embedding(self, audio: AudioArray) -> Embedding:
        rng = np.random.default_rng(seed=int(np.abs(audio).sum() * 1e4) % (2**32))
        emb = rng.standard_normal(self.EMB_DIM).astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-8
        return emb

    def clone(self, reference_audio: AudioArray, text: str = "") -> AudioArray:
        # 약간 왜곡된 오디오를 반환 (클로닝 시뮬레이션)
        noise = (
            np.random.default_rng(seed=0).standard_normal(len(reference_audio)).astype(np.float32)
        )
        cloned = reference_audio + 0.05 * noise
        return np.clip(cloned, -1.0, 1.0).astype(np.float32)


# ── 픽스처 ────────────────────────────────────────────────────────────────────
# 주의: RLAgent() → torch.optim.Adam → torch._dynamo → inspect.getframeinfo →
#       speechbrain k2 LazyModule 충돌 방지를 위해 agent를 ecapa보다 먼저 초기화.
#       module scope 픽스처는 선언 순서대로 초기화되므로 agent를 위에 선언.


@pytest.fixture(scope="module")
def agent() -> RLAgent:
    return RLAgent()


@pytest.fixture(scope="module")
def masker() -> PsychoacousticMasker:
    return PsychoacousticMasker()


@pytest.fixture(scope="module")
def mixer() -> Mixer:
    return Mixer()


@pytest.fixture(scope="module")
def ecapa() -> ECAPATDNNAdapter:
    return ECAPATDNNAdapter(device="cpu")


@pytest.fixture(scope="module")
def cosy() -> _DummyCosyVoiceAdapter:
    return _DummyCosyVoiceAdapter()


@pytest.fixture(scope="module")
def sample_audio() -> AudioArray:
    rng = np.random.default_rng(seed=42)
    t = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 800 * t)
        + 0.1 * np.sin(2 * np.pi * 1500 * t)
        + 0.02 * rng.standard_normal(SAMPLE_RATE).astype(np.float32)
    )
    return audio.astype(np.float32)


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

REWARD_WEIGHT_ECAPA = 0.5
REWARD_WEIGHT_CAM = 0.5
TTS_EVAL_INTERVAL = 10
PPO_BUFFER_SIZE = 8  # 테스트용: 빠르게 PPO update 트리거


def _run_episode(
    episode_idx: int,
    original: AudioArray,
    orig_ecapa_emb: Embedding,
    orig_cam_emb: Embedding,
    agent: RLAgent,
    masker: PsychoacousticMasker,
    mixer: Mixer,
    ecapa: ECAPATDNNAdapter,
    cosy: _DummyCosyVoiceAdapter,
    text: str = "안녕하세요",
) -> tuple[Transition, float, float, float]:
    """train.py 에피소드 로직 한 스텝."""
    state, action, log_prob, value, freq_pattern, time_gate = agent.act(original)
    safe_noise = masker.clamp(original, action)
    modified = mixer.mix(original, safe_noise)

    mod_ecapa_emb = ecapa.extract_embedding(modified)
    mod_cam_emb = cosy.extract_embedding(modified)

    ecapa_dist = cosine_distance(orig_ecapa_emb, mod_ecapa_emb)
    cam_dist = cosine_distance(orig_cam_emb, mod_cam_emb)
    emb_dist_reward = float(REWARD_WEIGHT_ECAPA * ecapa_dist + REWARD_WEIGHT_CAM * cam_dist)

    reward = emb_dist_reward
    tts_defense = None

    if (episode_idx + 1) % TTS_EVAL_INTERVAL == 0:
        cloned = cosy.clone(modified, text=text)
        clone_ecapa_emb = ecapa.extract_embedding(cloned)
        clone_cam_emb = cosy.extract_embedding(cloned)
        tts_ecapa_dist = cosine_distance(orig_ecapa_emb, clone_ecapa_emb)
        tts_cam_dist = cosine_distance(orig_cam_emb, clone_cam_emb)
        tts_defense = float((tts_ecapa_dist + tts_cam_dist) / 2.0)
        reward = tts_defense

    next_state = agent.extractor.extract(modified)
    transition = Transition(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=True,
        log_prob=log_prob,
        value=value,
        freq_pattern=freq_pattern,
        time_gate=time_gate,
    )
    return transition, reward, ecapa_dist, cam_dist


# ── 테스트 ─────────────────────────────────────────────────────────────────────


class TestSingleEpisode:
    def test_reward_is_scalar_in_range(
        self, sample_audio, agent, masker, mixer, ecapa, cosy
    ) -> None:
        orig_ecapa = ecapa.extract_embedding(sample_audio)
        orig_cam = cosy.extract_embedding(sample_audio)
        transition, reward, ecapa_dist, cam_dist = _run_episode(
            episode_idx=0,
            original=sample_audio,
            orig_ecapa_emb=orig_ecapa,
            orig_cam_emb=orig_cam,
            agent=agent,
            masker=masker,
            mixer=mixer,
            ecapa=ecapa,
            cosy=cosy,
        )
        assert isinstance(reward, float)
        assert not np.isnan(reward)
        # cosine distance ∈ [0, 2]
        assert 0.0 <= reward <= 2.0

    def test_transition_fields(self, sample_audio, agent, masker, mixer, ecapa, cosy) -> None:
        orig_ecapa = ecapa.extract_embedding(sample_audio)
        orig_cam = cosy.extract_embedding(sample_audio)
        transition, reward, _, _ = _run_episode(
            episode_idx=0,
            original=sample_audio,
            orig_ecapa_emb=orig_ecapa,
            orig_cam_emb=orig_cam,
            agent=agent,
            masker=masker,
            mixer=mixer,
            ecapa=ecapa,
            cosy=cosy,
        )
        from voicesecure.types import ACTION_N_FREQ, ACTION_N_TIME

        assert transition.action.shape == (ACTION_N_FREQ, ACTION_N_TIME)
        assert not torch.isnan(transition.log_prob)
        assert not torch.isnan(transition.value)
        assert transition.done is True


class TestTTSEvalEpisode:
    def test_tts_eval_triggers_clone_reward(
        self, sample_audio, agent, masker, mixer, ecapa, cosy
    ) -> None:
        """TTS_EVAL_INTERVAL 번째 에피소드에서 clone_dist로 reward 교체 확인."""
        orig_ecapa = ecapa.extract_embedding(sample_audio)
        orig_cam = cosy.extract_embedding(sample_audio)

        # episode_idx=9 → (9+1) % 10 == 0 → TTS 평가 트리거
        transition, reward, ecapa_dist, cam_dist = _run_episode(
            episode_idx=9,
            original=sample_audio,
            orig_ecapa_emb=orig_ecapa,
            orig_cam_emb=orig_cam,
            agent=agent,
            masker=masker,
            mixer=mixer,
            ecapa=ecapa,
            cosy=cosy,
        )
        # TTS 평가 에피소드 reward는 clone_dist이므로 emb_dist_reward와 다를 수 있음
        assert isinstance(reward, float)
        assert not np.isnan(reward)
        assert 0.0 <= reward <= 2.0

    def test_non_tts_episode_uses_emb_dist(
        self, sample_audio, agent, masker, mixer, ecapa, cosy
    ) -> None:
        """비TTS 에피소드에서 reward == emb_dist_reward 확인."""
        orig_ecapa = ecapa.extract_embedding(sample_audio)
        orig_cam = cosy.extract_embedding(sample_audio)

        transition, reward, ecapa_dist, cam_dist = _run_episode(
            episode_idx=0,  # 1 % 10 != 0 → TTS 미트리거
            original=sample_audio,
            orig_ecapa_emb=orig_ecapa,
            orig_cam_emb=orig_cam,
            agent=agent,
            masker=masker,
            mixer=mixer,
            ecapa=ecapa,
            cosy=cosy,
        )
        expected = float(REWARD_WEIGHT_ECAPA * ecapa_dist + REWARD_WEIGHT_CAM * cam_dist)
        assert abs(reward - expected) < 1e-6


class TestPPOUpdateLoop:
    def test_multi_episode_ppo_update(
        self, sample_audio, agent, masker, mixer, ecapa, cosy
    ) -> None:
        """PPO_BUFFER_SIZE 에피소드 쌓인 후 PPO 업데이트 성공 확인."""
        orig_ecapa = ecapa.extract_embedding(sample_audio)
        orig_cam = cosy.extract_embedding(sample_audio)

        buffer: list[Transition] = []
        rewards: list[float] = []

        for i in range(PPO_BUFFER_SIZE):
            transition, reward, _, _ = _run_episode(
                episode_idx=i,
                original=sample_audio,
                orig_ecapa_emb=orig_ecapa,
                orig_cam_emb=orig_cam,
                agent=agent,
                masker=masker,
                mixer=mixer,
                ecapa=ecapa,
                cosy=cosy,
            )
            buffer.append(transition)
            rewards.append(reward)

        assert len(buffer) == PPO_BUFFER_SIZE

        metrics = agent.update(buffer)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert not np.isnan(metrics["policy_loss"])
        assert not np.isnan(metrics["value_loss"])
        assert not np.isnan(metrics["entropy"])

    def test_rewards_are_positive(self, sample_audio, agent, masker, mixer, ecapa, cosy) -> None:
        """cosine distance는 항상 양수 (reward ≥ 0)."""
        orig_ecapa = ecapa.extract_embedding(sample_audio)
        orig_cam = cosy.extract_embedding(sample_audio)

        for i in range(5):
            _, reward, _, _ = _run_episode(
                episode_idx=i,
                original=sample_audio,
                orig_ecapa_emb=orig_ecapa,
                orig_cam_emb=orig_cam,
                agent=agent,
                masker=masker,
                mixer=mixer,
                ecapa=ecapa,
                cosy=cosy,
            )
            assert reward >= 0.0, f"episode {i}: reward={reward} < 0"

    def test_embeddings_shape(self, sample_audio, agent, ecapa, cosy) -> None:
        """agent 초기화 후 ECAPA + 더미 CAM++ 임베딩 shape 확인."""
        emb_e = ecapa.extract_embedding(sample_audio)
        emb_c = cosy.extract_embedding(sample_audio)
        assert emb_e.shape == (192,)
        assert emb_c.shape == (192,)
        assert not np.isnan(emb_e).any()
        assert not np.isnan(emb_c).any()
