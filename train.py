"""VoiceSecure 훈련 스크립트.

훈련 흐름:
    1. 전체 데이터셋 로드 + 원본 임베딩 캐싱 (ECAPA + CosyVoice3 CAM++)
    2. 에폭 루프 (1 에폭 = 전체 파일 수만큼 에피소드, 매 에폭 셔플)
    3. 에피소드마다:
        - RLAgent.act() → raw_noise (n_freq, n_time)
        - Masker.clamp() → safe_noise (심리음향 마스킹)
        - Mixer.mix() → 변조 음성
        - ECAPA + CAM++ 임베딩 거리 평균 → reward
        - Transition 버퍼에 추가
        - PPO_BUFFER_SIZE(32)개 쌓이면 PPO 업데이트
    4. 매 TTS_EVAL_INTERVAL 에피소드마다:
        - CosyVoice3 실제 클로닝 → 클론 임베딩 vs 원본 임베딩 → 방어 점수 로깅
    5. 매 CHECKPOINT_INTERVAL 에피소드마다 체크포인트 저장

Colab 사용법:
    python train.py --data_dir /path/to/kss --epochs 3
    python train.py --data_dir /path/to/kss --resume checkpoints/episode_1000.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.tensorboard import SummaryWriter

from voicesecure.evaluators.adapters.cosyvoice import CosyVoiceAdapter
from voicesecure.evaluators.adapters.ecapa_tdnn import ECAPATDNNAdapter
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

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────

PPO_BUFFER_SIZE = 32

# CosyVoice3 실제 클로닝 평가 주기 (에피소드 단위)
TTS_EVAL_INTERVAL = 10

# reward 가중치
REWARD_WEIGHT_ECAPA = 0.5   # ECAPA 임베딩 거리 가중치
REWARD_WEIGHT_CAM = 0.5     # CAM++ 임베딩 거리 가중치

# cosine distance 정규화 스케일
# ECAPA/CAM++ cosine distance 실측 분포: ~0.002~0.05 (심리음향 노이즈 한도 내)
# 0.05를 "충분한 방어" 기준으로 보고 그 이상은 1.0으로 clip
EMB_DIST_SCALE = 0.05

# TTS 평가 에피소드에서 emb_reward vs tts_reward 혼합 비율
# reward = REWARD_ALPHA * emb_reward + (1 - REWARD_ALPHA) * tts_reward
# 0.3: TTS 클로닝 방어(실제 공격 시나리오)에 70% 가중치
REWARD_ALPHA = 0.3

CHECKPOINT_INTERVAL = 1000
SAMPLE_RATE = 16000


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────


def load_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> AudioArray:
    """wav 파일을 16kHz mono float32로 로드 (soundfile + resample_poly).

    librosa 대신 soundfile 사용 — Colab에서 speechbrain k2 lazy-loader와
    librosa의 충돌을 피하기 위함.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = data.mean(axis=1).astype(np.float32)
    if sr != sample_rate:
        g = gcd(sample_rate, sr)
        audio = resample_poly(audio, sample_rate // g, sr // g).astype(np.float32)
    max_val = np.abs(audio).max()
    if max_val > 1e-6:
        audio = audio / max(max_val, 1.0)
    return audio


def load_dataset(data_dir: str) -> list[dict]:
    """KSS 데이터셋 로드.

    구조:
        data_dir/
            1/  2/  3/  4/   ← 화자 폴더
            Labels.txt       ← "파일명(확장자없음) 텍스트" 형식
    """
    data_dir = Path(data_dir)
    labels_path = data_dir / "Labels.txt"

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels.txt not found: {labels_path}")

    labels: dict[str, str] = {}
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                labels[parts[0]] = parts[1]

    samples = []
    for speaker_dir in sorted(data_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        for wav_path in sorted(speaker_dir.glob("*.wav")):
            file_id = wav_path.stem
            text = labels.get(file_id, "")
            if not text:
                logger.warning("라벨 없음, 건너뜀: %s", file_id)
                continue
            samples.append({"path": wav_path, "text": text, "file_id": file_id})

    logger.info("데이터셋 로드 완료: %d개 파일", len(samples))
    return samples


def load_dataset_aihub(
    data_dir: str,
    max_speakers: int | None = None,
    max_files_per_speaker: int | None = None,
) -> list[dict]:
    """AIHub 014 다화자 음성합성 데이터 로더.

    구조:
        data_dir/
            원천데이터/TS*/.../[화자ID]_[코드]_[이니셜]/*.wav
            라벨링데이터/TL*/.../[화자ID]_[코드]_[이니셜]/*.json

    각 JSON에는 전사/화자/녹음 메타데이터가 들어있다.
    WAV와 JSON은 파일명 stem이 동일하며, TL <-> TS 폴더에 1:1 대응된다.

    Args:
        data_dir:               원천/라벨링 폴더의 부모 경로.
        max_speakers:           로드할 최대 화자 수. None=전체.
        max_files_per_speaker:  화자당 최대 파일 수. None=전체. 데이터 균형 맞출 때 유용.
    """
    data_dir = Path(data_dir)
    wav_root = data_dir / "원천데이터"
    json_root = data_dir / "라벨링데이터"

    if not wav_root.exists() or not json_root.exists():
        raise FileNotFoundError(
            f"AIHub 구조 아님 (원천데이터/, 라벨링데이터/ 둘 다 필요): {data_dir}"
        )

    samples_by_speaker: dict[str, list[dict]] = {}
    missing_wav = 0

    for json_path in json_root.rglob("*.json"):
        # JSON 경로 -> WAV 경로 매핑 (TL* -> TS*, .json -> .wav)
        rel = json_path.relative_to(json_root)
        wav_rel_str = str(rel).replace("TL", "TS", 1)
        wav_rel = Path(wav_rel_str).with_suffix(".wav")
        wav_path = wav_root / wav_rel
        if not wav_path.exists():
            missing_wav += 1
            continue

        try:
            with open(json_path, encoding="utf-8") as f:
                meta = json.load(f)
            text = meta["전사정보"]["OrgLabelText"]
            speaker_id = meta["화자정보"]["SpeakerName"]
        except (KeyError, json.JSONDecodeError) as e:
            logger.warning("JSON 파싱 실패, 건너뜀 %s: %s", json_path.name, e)
            continue

        if not text:
            continue

        samples_by_speaker.setdefault(speaker_id, []).append(
            {
                "path": wav_path,
                "text": text,
                "file_id": wav_path.stem,
                "speaker_id": speaker_id,
            }
        )

    if missing_wav > 0:
        logger.warning("WAV 누락 %d개 (JSON은 있지만 매칭 WAV 없음)", missing_wav)

    # 화자별 정렬 후 cap 적용
    speakers = sorted(samples_by_speaker.keys())
    if max_speakers is not None:
        speakers = speakers[:max_speakers]

    samples: list[dict] = []
    for spk in speakers:
        spk_samples = sorted(samples_by_speaker[spk], key=lambda s: s["file_id"])
        if max_files_per_speaker is not None:
            spk_samples = spk_samples[:max_files_per_speaker]
        samples.extend(spk_samples)

    logger.info(
        "AIHub 로드 완료: %d 화자, %d 파일 (전체 화자 %d명 중)",
        len(speakers),
        len(samples),
        len(samples_by_speaker),
    )
    return samples


# ── 임베딩 캐시 ───────────────────────────────────────────────────────────────


def build_embedding_cache(
    samples: list[dict],
    ecapa: ECAPATDNNAdapter,
    cosy: CosyVoiceAdapter,
    cache_path: Path,
) -> dict[str, dict]:
    """전체 원본 음성의 ECAPA + CAM++ 임베딩을 미리 추출해 캐싱.

    캐시 파일(embedding_cache.pt)이 있으면 로드, 없으면 새로 계산 후 저장.

    Returns:
        {"1_0000": {"ecapa": np.ndarray(192,), "cam": np.ndarray(192,)}, ...}
    """
    if cache_path.exists():
        logger.info("임베딩 캐시 로드: %s", cache_path)
        # PyTorch 2.6+ weights_only 기본값이 True로 바뀌어 numpy 객체 포함된
        # cache (ECAPA/CAM++ embedding은 np.ndarray) 로드 실패. 본인이 만든
        # 파일이므로 신뢰 가능 → weights_only=False.
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        logger.info("캐시 로드 완료: %d개", len(cache))
        return cache

    logger.info("임베딩 캐시 생성 시작 (%d개 파일)...", len(samples))
    cache = {}

    for i, sample in enumerate(samples):
        audio = load_audio(sample["path"])
        ecapa_emb = ecapa.extract_embedding(audio)
        cam_emb = cosy.extract_embedding(audio)
        cache[sample["file_id"]] = {"ecapa": ecapa_emb, "cam": cam_emb}

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
    # weights_only=False — torch 2.6+ 호환 (optimizer state 등 비-tensor 객체 포함 가능)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
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
    ecapa = ECAPATDNNAdapter(device=str(device))
    cosy = CosyVoiceAdapter(
        model_dir=args.model_dir,
        cosyvoice_root=args.cosyvoice_root,
        device=str(device),
    )
    masker = PsychoacousticMasker()
    mixer = Mixer()
    agent = RLAgent(config={"lr": args.lr})

    # ── 데이터 로드 ──────────────────────────────────────────────────
    if args.data_format == "aihub":
        samples = load_dataset_aihub(
            args.data_dir,
            max_speakers=args.max_speakers,
            max_files_per_speaker=args.max_files_per_speaker,
        )
    else:
        samples = load_dataset(args.data_dir)
    total_files = len(samples)
    logger.info("1 에폭 = %d 에피소드", total_files)

    # ── 임베딩 캐시 ──────────────────────────────────────────────────
    cache = build_embedding_cache(
        samples, ecapa, cosy, cache_path=checkpoint_dir / "embedding_cache.pt"
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
    last_tts_defense: float = 0.0

    for epoch in range(start_epoch, args.epochs):
        epoch_samples = samples.copy()
        random.shuffle(epoch_samples)
        logger.info("에폭 %d/%d 시작 (%d 에피소드)", epoch + 1, args.epochs, total_files)

        for sample in epoch_samples:
            global_episode += 1
            file_id = sample["file_id"]
            text = sample["text"]

            # ── 원본 음성 로드 + 캐시 임베딩 ──────────────────────────
            original = load_audio(sample["path"])
            orig_ecapa_emb = cache[file_id]["ecapa"]
            orig_cam_emb = cache[file_id]["cam"]

            # ── RLAgent: state 추출 + 노이즈 생성 ──────────────────────
            state, action, log_prob, value, freq_pattern, time_gate = agent.act(original)

            # ── 심리음향 마스킹으로 노이즈 clamp ───────────────────────
            safe_noise = masker.clamp(original, action)

            # ── 변조 음성 생성 ──────────────────────────────────────────
            modified = mixer.mix(original, safe_noise)

            # ── 변조 음성 임베딩 추출 (ECAPA + CAM++) ──────────────────
            mod_ecapa_emb = ecapa.extract_embedding(modified)
            mod_cam_emb = cosy.extract_embedding(modified)

            # ── 임베딩 거리 (기본 reward 재료) ──────────────────────────
            ecapa_dist = cosine_distance(orig_ecapa_emb, mod_ecapa_emb)
            cam_dist = cosine_distance(orig_cam_emb, mod_cam_emb)
            raw_emb_dist = float(REWARD_WEIGHT_ECAPA * ecapa_dist + REWARD_WEIGHT_CAM * cam_dist)
            # cosine distance 실측값이 0.002~0.05 수준이므로 [0,1]로 정규화
            emb_dist_reward = min(raw_emb_dist / EMB_DIST_SCALE, 1.0)

            # ── next_state: done=True 고정 구조라 GAE에서 사용 안 됨
            # state 재사용으로 불필요한 StateExtractor forward pass 제거
            next_state = state

            # ── CosyVoice3 클로닝 평가 + reward 가중합 ─────────────────
            # 평소: reward = emb_dist_reward (빠른 근사 신호)
            # TTS 평가 에피소드(10마다): reward = ALPHA*emb + (1-ALPHA)*tts_defense
            #   → 실제 공격 시나리오(TTS 클로닝)를 primary 학습 신호로 유지하면서
            #     emb_reward를 30% 섞어 분포 충격 완화
            reward = emb_dist_reward
            if global_episode % TTS_EVAL_INTERVAL == 0:
                logger.info("[TTS 평가] episode=%d, file=%s", global_episode, file_id)
                try:
                    cloned = cosy.clone(modified, text=text)
                    clone_ecapa_emb = ecapa.extract_embedding(cloned)
                    clone_cam_emb = cosy.extract_embedding(cloned)

                    tts_ecapa_dist = cosine_distance(orig_ecapa_emb, clone_ecapa_emb)
                    tts_cam_dist = cosine_distance(orig_cam_emb, clone_cam_emb)
                    tts_defense = min(float((tts_ecapa_dist + tts_cam_dist) / 2.0) / EMB_DIST_SCALE, 1.0)

                    # 가중합: emb_reward 30% + tts_defense 70%
                    reward = REWARD_ALPHA * emb_dist_reward + (1.0 - REWARD_ALPHA) * tts_defense
                    last_tts_defense = tts_defense

                    writer.add_scalar("eval/tts_defense", tts_defense, global_episode)
                    writer.add_scalar("eval/tts_ecapa_dist", tts_ecapa_dist, global_episode)
                    writer.add_scalar("eval/tts_cam_dist", tts_cam_dist, global_episode)
                    writer.add_scalar("eval/tts_reward", reward, global_episode)
                    logger.info(
                        "  TTS reward: %.4f (emb=%.4f * %.1f + tts=%.4f * %.1f)",
                        reward, emb_dist_reward, REWARD_ALPHA,
                        tts_defense, 1.0 - REWARD_ALPHA,
                    )
                except Exception as e:
                    logger.warning("TTS 평가 실패, emb_dist 사용: %s", e)

            epoch_rewards.append(reward)

            # ── Transition 버퍼 ─────────────────────────────────────────
            transition_buffer.append(
                Transition(
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
            )

            # ── PPO 업데이트 ────────────────────────────────────────────
            if len(transition_buffer) >= PPO_BUFFER_SIZE:
                metrics = agent.update(transition_buffer)
                transition_buffer.clear()
                writer.add_scalar("train/policy_loss", metrics["policy_loss"], global_episode)
                writer.add_scalar("train/value_loss", metrics["value_loss"], global_episode)
                writer.add_scalar("train/entropy", metrics["entropy"], global_episode)

            writer.add_scalar("train/reward", reward, global_episode)
            writer.add_scalar("train/ecapa_dist", ecapa_dist, global_episode)
            writer.add_scalar("train/cam_dist", cam_dist, global_episode)

            # ── 체크포인트 저장 ─────────────────────────────────────────
            if global_episode % args.checkpoint_interval == 0:
                save_checkpoint(
                    agent, global_episode, epoch, best_reward,
                    checkpoint_dir, f"episode_{global_episode}.pt",
                )

            if global_episode % 100 == 0:
                logger.info(
                    "episode=%d | epoch=%d/%d | reward=%.4f",
                    global_episode, epoch + 1, args.epochs, reward,
                )

        # ── 에폭 종료 ───────────────────────────────────────────────
        epoch_mean_reward = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        writer.add_scalar("train/epoch_mean_reward", epoch_mean_reward, epoch + 1)
        logger.info("에폭 %d/%d 완료 | 평균 reward=%.4f", epoch + 1, args.epochs, epoch_mean_reward)

        if epoch_mean_reward > best_reward:
            best_reward = epoch_mean_reward
            save_checkpoint(
                agent, global_episode, epoch + 1, best_reward, checkpoint_dir, "best.pt"
            )
            logger.info("best 체크포인트 갱신: reward=%.4f", best_reward)

        epoch_rewards.clear()

    if len(transition_buffer) > 0:
        agent.update(transition_buffer)
        transition_buffer.clear()

    writer.close()
    logger.info("훈련 완료. 총 에피소드: %d", global_episode)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VoiceSecure 훈련 스크립트")

    parser.add_argument("--data_dir", type=str, default="../CosyVoice/kss")
    parser.add_argument(
        "--model_dir", type=str,
        default="CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512",
    )
    parser.add_argument("--cosyvoice_root", type=str, default="CosyVoice")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=CHECKPOINT_INTERVAL)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data_format",
        type=str,
        default="kss",
        choices=["kss", "aihub"],
        help="kss: data_dir/{1,2,3,4}/*.wav + Labels.txt. "
        "aihub: data_dir/{원천데이터,라벨링데이터}/...",
    )
    parser.add_argument(
        "--max_speakers",
        type=int,
        default=None,
        help="(aihub) 로드할 최대 화자 수. None=전체.",
    )
    parser.add_argument(
        "--max_files_per_speaker",
        type=int,
        default=None,
        help="(aihub) 화자당 최대 파일 수. 데이터 균형 맞출 때 사용. None=전체.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)
