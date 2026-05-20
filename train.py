"""VoiceSecure 훈련 스크립트.

훈련 흐름:
    1. 전체 데이터셋 로드 + 원본 임베딩 캐싱 (WavLM + CosyVoice3 CAM++)
    2. 에폭 루프 (1 에폭 = 전체 파일 수만큼 에피소드, 매 에폭 셔플)
    3. 에피소드마다:
        - RLAgent.act() → raw_noise
        - Masker.clamp() → safe_noise (심리음향 마스킹)
        - Mixer.mix() → 변조 음성
        - WavLM + CAM++ 임베딩 거리 평균 → reward
        - Transition 버퍼에 추가
        - 32개 쌓이면 PPO 업데이트 (버퍼 크기: PPO_BUFFER_SIZE)
    4. 매 TTS_EVAL_INTERVAL 에피소드마다:
        - CosyVoice3 실제 클로닝 → 클론 임베딩 vs 원본 임베딩 → 방어 점수 로깅
    5. 매 CHECKPOINT_INTERVAL 에피소드마다 체크포인트 저장

사용법 (Colab):
    python train.py --data_dir /path/to/kss --epochs 3
    python train.py --data_dir /path/to/kss --resume checkpoints/episode_1000.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from voicesecure.evaluators.adapters.cosyvoice import CosyVoiceAdapter
from voicesecure.evaluators.adapters.wavlm_sv import WavLMSVAdapter
from voicesecure.evaluators.base import cosine_distance
from voicesecure.modulation.masker import PsychoacousticMasker
from voicesecure.modulation.mixer import Mixer
from voicesecure.rl.agent import RLAgent
from voicesecure.types import AudioArray, Transition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")

# ── 하이퍼파라미터 상수 ────────────────────────────────────────────────────────

# PPO 업데이트 버퍼 크기.
# 에피소드마다 1개의 Transition이 쌓이고, 이 숫자만큼 쌓이면 PPO 업데이트 1회.
# 값이 작으면 업데이트가 잦아 불안정, 크면 안정적이나 느림. 32가 적절한 균형점.
PPO_BUFFER_SIZE = 32

# CosyVoice3 실제 클로닝 평가 주기 (에피소드 단위).
# 클로닝은 무거운 연산이라 매 에피소드마다 하면 너무 느림.
# 너무 느리면 값을 키워서 조정 (예: 20, 50).
TTS_EVAL_INTERVAL = 10

# reward 가중치: WavLM(0.4) + CAM++(0.4) + TTS 클로닝 방어(0.2)
# TTS는 매 TTS_EVAL_INTERVAL마다만 계산되므로 가중치를 낮게 설정.
# 나머지 에피소드는 직전 tts_defense 값을 캐싱해서 사용.
REWARD_WEIGHT_WAVLM = 0.4
REWARD_WEIGHT_CAM = 0.4
REWARD_WEIGHT_TTS = 0.2

# 체크포인트 저장 주기 (에피소드 단위).
# 기본 1000 에피소드마다 저장. Colab 런타임 제한 고려해서 설정.
CHECKPOINT_INTERVAL = 1000

# 오디오 로딩 샘플레이트 (16kHz 고정)
SAMPLE_RATE = 16000


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────


def load_dataset(data_dir: str) -> list[dict]:
    """KSS 데이터셋을 로드한다.

    구조:
        data_dir/
            1/  2/  3/  4/   ← 화자 폴더 (구분 없이 전체 섞어서 사용)
            Labels.txt       ← "파일명(확장자없음) 텍스트" 형식

    Returns:
        [{"path": Path, "text": str, "file_id": str}, ...]
        전체 파일을 화자 구분 없이 섞은 리스트.
    """
    data_dir = Path(data_dir)
    labels_path = data_dir / "Labels.txt"

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels.txt not found: {labels_path}")

    # Labels.txt 파싱: "1_0000 그는 괜찮은 척하려고 애쓰는 것 같았다."
    labels: dict[str, str] = {}
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                file_id, text = parts
                labels[file_id] = text

    # 화자 폴더(숫자 이름) 안의 wav 파일 수집
    samples = []
    for speaker_dir in sorted(data_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        for wav_path in sorted(speaker_dir.glob("*.wav")):
            file_id = wav_path.stem  # "1_0000"
            text = labels.get(file_id, "")
            if not text:
                logger.warning("라벨 없음, 건너뜀: %s", file_id)
                continue
            samples.append({"path": wav_path, "text": text, "file_id": file_id})

    logger.info("데이터셋 로드 완료: %d개 파일", len(samples))
    return samples


def load_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> AudioArray:
    """wav 파일을 16kHz mono float32로 로드한다."""
    import librosa

    audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    audio = audio.astype(np.float32)
    # [-1, 1] 범위 보장
    max_val = np.abs(audio).max()
    if max_val > 1.0:
        audio = audio / max_val
    return audio


# ── 임베딩 캐시 ───────────────────────────────────────────────────────────────


def build_embedding_cache(
    samples: list[dict],
    wavlm: WavLMSVAdapter,
    cosy: CosyVoiceAdapter,
    cache_path: Path,
) -> dict[str, dict]:
    """전체 원본 음성의 WavLM + CosyVoice3 CAM++ 임베딩을 미리 추출해 캐싱한다.

    훈련 중 매 에피소드마다 원본 임베딩을 재계산하면 느리므로,
    훈련 시작 전에 전체를 한 번에 추출해서 메모리/파일에 캐싱한다.

    캐시 파일(embedding_cache.pt)이 이미 있으면 로드해서 재사용.
    없으면 새로 계산 후 저장.

    Returns:
        {"1_0000": {"wavlm": np.ndarray(512,), "cam": np.ndarray(192,)}, ...}
    """
    if cache_path.exists():
        logger.info("임베딩 캐시 로드: %s", cache_path)
        cache = torch.load(cache_path, map_location="cpu")
        logger.info("캐시 로드 완료: %d개", len(cache))
        return cache

    logger.info("임베딩 캐시 생성 시작 (%d개 파일)...", len(samples))
    cache = {}

    for i, sample in enumerate(samples):
        audio = load_audio(sample["path"])
        wavlm_emb = wavlm.extract_embedding(audio)
        cam_emb = cosy.extract_embedding(audio)
        cache[sample["file_id"]] = {"wavlm": wavlm_emb, "cam": cam_emb}

        if (i + 1) % 100 == 0:
            logger.info("  캐시 진행: %d / %d", i + 1, len(samples))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    logger.info("임베딩 캐시 저장 완료: %s", cache_path)
    return cache


# ── 체크포인트 ────────────────────────────────────────────────────────────────


def save_checkpoint(
    agent: RLAgent,
    episode: int,
    epoch: int,
    best_reward: float,
    checkpoint_dir: Path,
    filename: str,
) -> None:
    """체크포인트를 저장한다."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / filename
    torch.save(
        {
            "episode": episode,
            "epoch": epoch,
            "best_reward": best_reward,
            "policy_state_dict": agent.policy.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
        },
        path,
    )
    logger.info("체크포인트 저장: %s", path)


def load_checkpoint(agent: RLAgent, path: Path) -> tuple[int, int, float]:
    """체크포인트를 로드하고 (episode, epoch, best_reward)를 반환한다."""
    checkpoint = torch.load(path, map_location="cpu")
    agent.policy.load_state_dict(checkpoint["policy_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    episode = checkpoint.get("episode", 0)
    epoch = checkpoint.get("epoch", 0)
    best_reward = checkpoint.get("best_reward", 0.0)
    logger.info("체크포인트 로드: %s (episode=%d, epoch=%d)", path, episode, epoch)
    return episode, epoch, best_reward


# ── 메인 훈련 루프 ────────────────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("디바이스: %s", device)

    checkpoint_dir = Path(args.checkpoint_dir)
    writer = SummaryWriter(log_dir=str(checkpoint_dir / "logs"))

    # ── 모델 초기화 ──────────────────────────────────────────────────
    logger.info("모델 초기화 중...")
    wavlm = WavLMSVAdapter(device=str(device))
    cosy = CosyVoiceAdapter(
        model_dir=args.model_dir,
        cosyvoice_root=args.cosyvoice_root,
        device=str(device),
    )
    masker = PsychoacousticMasker()
    mixer = Mixer()
    agent = RLAgent(config={"lr": args.lr})

    # ── 데이터 로드 ──────────────────────────────────────────────────
    samples = load_dataset(args.data_dir)
    total_files = len(samples)
    # 1 에폭 = 전체 파일 수만큼 에피소드
    # 에폭이 끝나면 파일 목록을 셔플해서 다시 순환
    logger.info("1 에폭 = %d 에피소드", total_files)

    # ── 임베딩 캐시 ──────────────────────────────────────────────────
    cache = build_embedding_cache(
        samples, wavlm, cosy, cache_path=checkpoint_dir / "embedding_cache.pt"
    )

    # ── Resume ───────────────────────────────────────────────────────
    start_episode = 0
    start_epoch = 0
    best_reward = float("-inf")
    if args.resume:
        start_episode, start_epoch, best_reward = load_checkpoint(agent, Path(args.resume))

    # ── 훈련 루프 ─────────────────────────────────────────────────────
    global_episode = start_episode
    transition_buffer: list[Transition] = []
    epoch_rewards: list[float] = []
    last_tts_defense: float = 0.0  # TTS 클로닝 방어 점수 캐시 (직전 값 재사용)

    for epoch in range(start_epoch, args.epochs):
        # 에폭마다 파일 목록 셔플 (패턴 암기 방지)
        epoch_samples = samples.copy()
        random.shuffle(epoch_samples)
        logger.info("에폭 %d/%d 시작 (%d 에피소드)", epoch + 1, args.epochs, total_files)

        for sample in epoch_samples:
            global_episode += 1
            file_id = sample["file_id"]
            text = sample["text"]

            # ── 원본 음성 로드 + 캐시에서 임베딩 가져오기 ──────────
            original = load_audio(sample["path"])
            orig_wavlm_emb = cache[file_id]["wavlm"]
            orig_cam_emb = cache[file_id]["cam"]

            # ── RLAgent: state 추출 + 노이즈 생성 ──────────────────
            # act()가 state도 반환 → 중복 추출 없음
            state, action, log_prob, value = agent.act(original)

            # ── 심리음향 마스킹 → 변조 음성 생성 ───────────────────
            safe_noise = masker.clamp(original, action)
            modified = mixer.mix(original, safe_noise)

            # ── 변조 음성 임베딩 추출 (WavLM + CAM++) ───────────────
            mod_wavlm_emb = wavlm.extract_embedding(modified)
            mod_cam_emb = cosy.extract_embedding(modified)

            # ── 임베딩 거리 계산 → reward ────────────────────────────
            wavlm_dist = cosine_distance(orig_wavlm_emb, mod_wavlm_emb)
            cam_dist = cosine_distance(orig_cam_emb, mod_cam_emb)
            # WavLM(0.4) + CAM++(0.4) + TTS 클로닝 방어(0.2) 가중 합산
            # TTS는 매 TTS_EVAL_INTERVAL마다 갱신, 나머지는 직전 값 재사용
            reward = float(
                REWARD_WEIGHT_WAVLM * wavlm_dist
                + REWARD_WEIGHT_CAM * cam_dist
                + REWARD_WEIGHT_TTS * last_tts_defense
            )
            epoch_rewards.append(reward)

            # ── next_state 추출 (변조 음성 기준) ────────────────────
            next_state = agent.extractor.extract(modified)

            # ── Transition 버퍼에 추가 ───────────────────────────────
            transition_buffer.append(
                Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=True,  # 에피소드마다 독립적 (단일 스텝)
                    log_prob=log_prob,
                    value=value,
                )
            )

            # ── PPO 업데이트: PPO_BUFFER_SIZE(32)개 쌓이면 업데이트 ─
            if len(transition_buffer) >= PPO_BUFFER_SIZE:
                metrics = agent.update(transition_buffer)
                transition_buffer.clear()

                writer.add_scalar("train/policy_loss", metrics["policy_loss"], global_episode)
                writer.add_scalar("train/value_loss", metrics["value_loss"], global_episode)
                writer.add_scalar("train/entropy", metrics["entropy"], global_episode)

            # ── TensorBoard: 에피소드 지표 ──────────────────────────
            writer.add_scalar("train/reward", reward, global_episode)
            writer.add_scalar("train/wavlm_dist", wavlm_dist, global_episode)
            writer.add_scalar("train/cam_dist", cam_dist, global_episode)

            # ── N 에피소드마다 CosyVoice3 실제 클로닝 평가 ──────────
            # TTS_EVAL_INTERVAL 조정으로 속도/정확도 트레이드오프 가능
            if global_episode % TTS_EVAL_INTERVAL == 0:
                logger.info("[TTS 평가] episode=%d, file=%s", global_episode, file_id)
                try:
                    # 변조 음성 + 원본 텍스트로 클로닝
                    cloned = cosy.clone(modified, text=text)
                    clone_wavlm_emb = wavlm.extract_embedding(cloned)
                    clone_cam_emb = cosy.extract_embedding(cloned)

                    tts_wavlm_dist = cosine_distance(orig_wavlm_emb, clone_wavlm_emb)
                    tts_cam_dist = cosine_distance(orig_cam_emb, clone_cam_emb)
                    tts_defense = float((tts_wavlm_dist + tts_cam_dist) / 2.0)

                    last_tts_defense = tts_defense  # 다음 에피소드 reward에 캐싱
                    writer.add_scalar("eval/tts_defense", tts_defense, global_episode)
                    writer.add_scalar("eval/tts_wavlm_dist", tts_wavlm_dist, global_episode)
                    writer.add_scalar("eval/tts_cam_dist", tts_cam_dist, global_episode)
                    logger.info(
                        "  TTS 방어 점수: %.4f (wavlm=%.4f, cam=%.4f)",
                        tts_defense,
                        tts_wavlm_dist,
                        tts_cam_dist,
                    )
                except Exception as e:
                    logger.warning("TTS 평가 실패 (건너뜀): %s", e)

            # ── 체크포인트 저장 ──────────────────────────────────────
            if global_episode % args.checkpoint_interval == 0:
                save_checkpoint(
                    agent,
                    global_episode,
                    epoch,
                    best_reward,
                    checkpoint_dir,
                    f"episode_{global_episode}.pt",
                )

            if global_episode % 100 == 0:
                logger.info(
                    "episode=%d | epoch=%d/%d | reward=%.4f",
                    global_episode,
                    epoch + 1,
                    args.epochs,
                    reward,
                )

        # ── 에폭 종료: best 체크포인트 갱신 ─────────────────────────
        epoch_mean_reward = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        writer.add_scalar("train/epoch_mean_reward", epoch_mean_reward, epoch + 1)
        logger.info(
            "에폭 %d/%d 완료 | 평균 reward=%.4f",
            epoch + 1,
            args.epochs,
            epoch_mean_reward,
        )

        if epoch_mean_reward > best_reward:
            best_reward = epoch_mean_reward
            save_checkpoint(
                agent,
                global_episode,
                epoch + 1,
                best_reward,
                checkpoint_dir,
                "best.pt",
            )
            logger.info("best 체크포인트 갱신: reward=%.4f", best_reward)

        epoch_rewards.clear()

    # 마지막 버퍼에 남은 transition 처리 (PPO_BUFFER_SIZE 미만으로 남은 경우)
    if len(transition_buffer) > 0:
        metrics = agent.update(transition_buffer)
        transition_buffer.clear()

    writer.close()
    logger.info("훈련 완료. 총 에피소드: %d", global_episode)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VoiceSecure 훈련 스크립트")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="../CosyVoice/kss",
        help="KSS 데이터셋 루트 경로 (Labels.txt + 화자 폴더 포함)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512",
        help="CosyVoice3 pretrained_models 경로",
    )
    parser.add_argument(
        "--cosyvoice_root",
        type=str,
        default="CosyVoice",
        help="CosyVoice 레포 루트 경로",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="훈련 에폭 수. 1 에폭 = 전체 데이터셋 파일 수만큼 에피소드.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="체크포인트 + 로그 저장 경로 (상대경로 권장)",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=CHECKPOINT_INTERVAL,
        help="체크포인트 저장 주기 (에피소드 단위). 기본값 1000.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="PPO 옵티마이저 learning rate",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="이어서 훈련할 체크포인트 경로. 예: checkpoints/episode_1000.pt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="재현성을 위한 랜덤 시드",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 재현성을 위한 시드 고정
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(args)
