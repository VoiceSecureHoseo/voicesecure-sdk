"""VoiceSecure 훈련 스크립트 (SDK Evaluator 기반).

의도된 매핑을 그대로 따른다:
    - 화자 식별  : WavLM-SV + CAM++       → SpeakerEvaluator
    - 공격 시뮬  : XTTS (저수준 API)        → TTSEvaluator
    - 음질        : wav2vec2-large-xlsr-korean → ASREvaluator
    - reward 결합 : RewardFunction (alpha·sv + beta·tts - lambda·max(0, cer - threshold))

훈련 흐름:
    1. 데이터셋 로드 (KSS 또는 AIHub 014)
    2. 전체 원본 음성의 화자 임베딩 캐싱
       (SpeakerEvaluator.precompute 결과 → 디스크에 .pt로 보존)
    3. 에폭 루프 (매 에폭 셔플)
    4. 에피소드마다:
        - RLAgent.act() → raw_noise
        - Masker.clamp() → safe_noise
        - Mixer.mix() → modified
        - SpeakerEvaluator.evaluate(..., precomputed_original_features=sv_cached) → sv_score
        - ASREvaluator.evaluate(..., original_text=sample[text]) → asr_cer
        - 매 TTS_EVAL_INTERVAL마다 TTSEvaluator.evaluate(...) → tts_score (그 외에는 last_tts_score 캐싱)
        - RewardFunction.compute({sv_score, tts_score, asr_cer}) → reward
        - Transition 버퍼 → PPO_BUFFER_SIZE 차면 agent.update()
    5. 매 CHECKPOINT_INTERVAL 에피소드마다 체크포인트 저장

Colab 사용법:
    python train.py --data_dir /path/to/aihub --data_format aihub --epochs 3
    python train.py --data_dir /path/to/kss  --data_format kss   --epochs 3
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

from voicesecure.evaluators import ASREvaluator, SpeakerEvaluator, TTSEvaluator
from voicesecure.evaluators.adapters.campplus import CAMPlusAdapter
from voicesecure.evaluators.adapters.wav2vec2_asr import Wav2Vec2KoreanAdapter
from voicesecure.evaluators.adapters.wavlm_sv import WavLMSVAdapter
from voicesecure.evaluators.adapters.xtts import XTTSAdapter
from voicesecure.modulation.masker import PsychoacousticMasker
from voicesecure.modulation.mixer import Mixer
from voicesecure.reward.function import RewardFunction
from voicesecure.rl.agent import RLAgent
from voicesecure.types import AudioArray, Transition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────

PPO_BUFFER_SIZE = 32

# XTTS 클로닝 평가 주기 (에피소드 단위). XTTS clone 1회가 무거워서 매 step은 비현실적.
TTS_EVAL_INTERVAL = 10

# cosine distance 정규화 스케일.
# WavLM/CAM++ cosine distance 실측 분포는 심리음향 마스킹 한도 내에서 ~0.002~0.05.
# 0.05를 "충분한 방어" 기준점으로 보고 그 이상은 1.0으로 clip → score가 [0,1] 전체에 펴짐.
EMB_DIST_SCALE = 0.05

# RewardFunction 기본 가중치 (SDK default와 동일)
#   reward = alpha * sv_score + beta * tts_score - lambda_asr * max(0, asr_cer - cer_threshold)
REWARD_ALPHA = 0.6     # sv_score 가중치
REWARD_BETA = 0.4      # tts_score 가중치
REWARD_LAMBDA_ASR = 1.0
REWARD_CER_THRESHOLD = 0.3

CHECKPOINT_INTERVAL = 1000
SAMPLE_RATE = 16000


def emb_dist_normalizer(raw_dist: float) -> float:
    """cosine distance(0~1) → 학습 가능한 스코어로 매핑.

    실측 cosine distance가 0.002~0.05 수준이라 그대로 쓰면 reward가 너무 작아
    학습 신호가 약함. EMB_DIST_SCALE(0.05)로 나눠 1.0에 clip.
    """
    return float(min(max(raw_dist, 0.0) / EMB_DIST_SCALE, 1.0))


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────


def load_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> AudioArray:
    """wav 파일을 16kHz mono float32로 로드 (soundfile + resample_poly).

    librosa 대신 soundfile 사용 — Colab에서 다른 패키지와 충돌을 피하기 위함.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = data.mean(axis=1).astype(np.float32)
    if sr != sample_rate:
        g = gcd(sample_rate, sr)
        audio = resample_poly(audio, sample_rate // g, sr // g).astype(np.float32)
    max_val = float(np.abs(audio).max())
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
    data_dir_p = Path(data_dir)
    labels_path = data_dir_p / "Labels.txt"

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

    samples: list[dict] = []
    for speaker_dir in sorted(data_dir_p.iterdir()):
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
    """
    data_dir_p = Path(data_dir)
    wav_root = data_dir_p / "원천데이터"
    json_root = data_dir_p / "라벨링데이터"

    if not wav_root.exists() or not json_root.exists():
        raise FileNotFoundError(
            f"AIHub 구조 아님 (원천데이터/, 라벨링데이터/ 둘 다 필요): {data_dir}"
        )

    samples_by_speaker: dict[str, list[dict]] = {}
    missing_wav = 0

    for json_path in json_root.rglob("*.json"):
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
    sv_eval: SpeakerEvaluator,
    cache_path: Path,
) -> dict[str, dict]:
    """전체 원본 음성의 화자 임베딩을 미리 추출해 캐싱.

    SpeakerEvaluator.precompute() 결과를 그대로 저장 — 캐시 형식은
    {"wavlm_embedding": np.ndarray, "cam_embedding": np.ndarray}.
    TTSEvaluator도 같은 wavlm_embedding 을 재활용하므로 별도 캐시 불필요.
    """
    if cache_path.exists():
        logger.info("임베딩 캐시 로드: %s", cache_path)
        # PyTorch 2.6+ weights_only 기본값 True 우회 (numpy 객체 포함)
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        logger.info("캐시 로드 완료: %d개", len(cache))
        return cache

    logger.info("임베딩 캐시 생성 시작 (%d개 파일)...", len(samples))
    cache: dict[str, dict] = {}

    for i, sample in enumerate(samples):
        audio = load_audio(sample["path"])
        cache[sample["file_id"]] = sv_eval.precompute(audio)

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

    # ── 어댑터 초기화 (의도된 매핑) ──────────────────────────────────
    logger.info("어댑터 로드 중...")
    wavlm = WavLMSVAdapter(device=str(device))
    cam = CAMPlusAdapter(device=str(device))
    xtts = XTTSAdapter(model_dir=args.xtts_model_dir) if args.use_tts else None
    asr_adapter = Wav2Vec2KoreanAdapter(device=str(device)) if args.use_asr else None

    # ── SDK Evaluator 조합 ──────────────────────────────────────────
    sv_eval = SpeakerEvaluator(
        wavlm_model=wavlm,
        cam_model=cam,
        normalizer=emb_dist_normalizer,
    )
    tts_eval: TTSEvaluator | None = None
    if xtts is not None:
        tts_eval = TTSEvaluator(
            xtts_model=xtts,
            speaker_model=wavlm,           # 의도된 매핑: WavLM이 TTS 평가의 화자 모델 역할
            normalizer=emb_dist_normalizer,
            sampling_interval=args.tts_eval_interval,
        )
    asr_eval: ASREvaluator | None = None
    if asr_adapter is not None:
        asr_eval = ASREvaluator(asr_model=asr_adapter)

    reward_fn = RewardFunction(
        alpha=args.reward_alpha,
        beta=args.reward_beta,
        lambda_asr=args.reward_lambda_asr,
        cer_threshold=args.reward_cer_threshold,
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

    # ── 원본 임베딩 캐시 (SpeakerEvaluator.precompute 결과) ──────────
    cache = build_embedding_cache(
        samples, sv_eval, cache_path=checkpoint_dir / "embedding_cache_sdk.pt"
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
    last_tts_score: float = 0.0          # TTS 미평가 에피소드에서 직전 값 재사용
    last_asr_cer: float = 0.0            # ASR 미평가 에피소드에서 직전 값 재사용

    for epoch in range(start_epoch, args.epochs):
        epoch_samples = samples.copy()
        random.shuffle(epoch_samples)
        logger.info("에폭 %d/%d 시작 (%d 에피소드)", epoch + 1, args.epochs, total_files)

        for sample in epoch_samples:
            global_episode += 1
            file_id = sample["file_id"]
            text = sample["text"]

            # ── 원본 음성 + 캐시 임베딩 ────────────────────────────────
            original = load_audio(sample["path"])
            sv_cached = cache[file_id]
            # TTSEvaluator의 캐시는 {"original_embedding": ...} 형식. WavLM 재사용.
            tts_cached = {"original_embedding": sv_cached["wavlm_embedding"]}

            # ── RLAgent: state 추출 + 노이즈 생성 ──────────────────────
            state, action, log_prob, value, freq_pattern, time_gate = agent.act(original)

            # ── 심리음향 마스킹 → 변조 음성 ──────────────────────────────
            safe_noise = masker.clamp(original, action)
            modified = mixer.mix(original, safe_noise)

            # ── SpeakerEvaluator (매 에피소드) ─────────────────────────
            sv_out = sv_eval.evaluate(
                original, modified, precomputed_original_features=sv_cached
            )
            sv_score = sv_out.score
            writer.add_scalar("train/sv_score", sv_score, global_episode)
            writer.add_scalar("train/sv_raw", sv_out.raw_metric, global_episode)

            # ── ASREvaluator (있을 때만, 매 에피소드) ──────────────────
            if asr_eval is not None:
                try:
                    asr_out = asr_eval.evaluate(original, modified, original_text=text)
                    last_asr_cer = float(asr_out.raw_metric)
                    writer.add_scalar("train/asr_cer", last_asr_cer, global_episode)
                except Exception as e:
                    logger.warning("ASR 평가 실패: %s", e)

            # ── TTSEvaluator (sampling_interval 마다만) ────────────────
            if tts_eval is not None and tts_eval.should_evaluate(global_episode):
                logger.info("[TTS 평가] episode=%d, file=%s", global_episode, file_id)
                try:
                    tts_out = tts_eval.evaluate(
                        original, modified,
                        precomputed_original_features=tts_cached,
                    )
                    last_tts_score = tts_out.score
                    writer.add_scalar("eval/tts_score", last_tts_score, global_episode)
                    writer.add_scalar("eval/tts_raw", tts_out.raw_metric, global_episode)
                except Exception as e:
                    logger.warning("TTS 평가 실패: %s", e)

            # ── RewardFunction ────────────────────────────────────────
            components = {
                "sv_score": float(np.clip(sv_score, 0.0, 1.0)),
                "tts_score": float(np.clip(last_tts_score, 0.0, 1.0)),
                "asr_cer": float(np.clip(last_asr_cer, 0.0, 1.0)),
            }
            reward = reward_fn.compute(components)
            epoch_rewards.append(reward)

            # ── Transition 버퍼 ─────────────────────────────────────────
            # done=True 고정 — 1-step episode 구조. next_state는 state 재사용.
            transition_buffer.append(
                Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=state,
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

            # ── 체크포인트 저장 ─────────────────────────────────────────
            if global_episode % args.checkpoint_interval == 0:
                save_checkpoint(
                    agent, global_episode, epoch, best_reward,
                    checkpoint_dir, f"episode_{global_episode}.pt",
                )

            if global_episode % 100 == 0:
                logger.info(
                    "episode=%d | epoch=%d/%d | reward=%.4f | sv=%.3f tts=%.3f cer=%.3f",
                    global_episode, epoch + 1, args.epochs,
                    reward, sv_score, last_tts_score, last_asr_cer,
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
    parser = argparse.ArgumentParser(description="VoiceSecure 훈련 스크립트 (SDK Evaluator 기반)")

    # 데이터
    parser.add_argument("--data_dir", type=str, default="../kss")
    parser.add_argument(
        "--data_format",
        type=str,
        default="kss",
        choices=["kss", "aihub"],
        help="kss: data_dir/{1,2,3,4}/*.wav + Labels.txt. "
        "aihub: data_dir/{원천데이터,라벨링데이터}/...",
    )
    parser.add_argument("--max_speakers", type=int, default=None)
    parser.add_argument("--max_files_per_speaker", type=int, default=None)

    # XTTS 모델
    parser.add_argument(
        "--xtts_model_dir",
        type=str,
        default=None,
        help="XTTS v2 모델 폴더. None이면 coqui-tts 캐시 자동 탐색 → HuggingFace 자동 다운로드.",
    )

    # Evaluator on/off
    parser.add_argument(
        "--use_tts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="XTTS 평가 사용 여부 (default True). 끄면 tts_score=0으로 고정.",
    )
    parser.add_argument(
        "--use_asr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="wav2vec2 ASR 평가 사용 여부 (default True). 끄면 asr_cer=0으로 고정.",
    )
    parser.add_argument("--tts_eval_interval", type=int, default=TTS_EVAL_INTERVAL)

    # Reward 가중치
    parser.add_argument("--reward_alpha", type=float, default=REWARD_ALPHA)
    parser.add_argument("--reward_beta", type=float, default=REWARD_BETA)
    parser.add_argument("--reward_lambda_asr", type=float, default=REWARD_LAMBDA_ASR)
    parser.add_argument("--reward_cer_threshold", type=float, default=REWARD_CER_THRESHOLD)

    # 학습
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=CHECKPOINT_INTERVAL)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)
