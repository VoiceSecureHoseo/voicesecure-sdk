"""
학습을 도와주는 잡다한 도구들

- compute_orig_embeddings : 원본 음성들을 CAM++에 미리 한 번씩 통과시켜서
  "이 사람 목소리는 이런 특징이다"라는 값(임베딩)을 저장해둔다. 매 학습
  스텝마다 다시 계산하면 너무 느려서, 한 번 계산해두고 재사용하는 것이다.
- save_checkpoint : 지금까지 학습된 모델을 파일로 저장한다. 중간에 꺼져도 안전하게.
- load_checkpoint : 저장해둔 모델을 다시 불러와서 이어서 학습한다.
"""

import os
import torch

from loss import campplus_embed_single
from dataset import MAX_SAMPLES


# -- 임베딩 캐싱 -----------------------------------------------------

def compute_orig_embeddings(
    campplus_model,
    dataset,
    device: torch.device,
    cache_path: str = None,
) -> dict:
    """
    전체 데이터셋의 원본 화자 임베딩을 미리 한 번에 계산해서 캐시로 만든다.

    원본 오디오는 학습 중에 변하지 않으니, 매 배치마다 다시 계산하면
    시간 낭비다. 한 번만 계산해두고 파일 경로를 key로 해서 저장해두면
    이후엔 꺼내 쓰기만 하면 된다. dataset은 augment=False로 만든 것을 넣어야 한다.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"  임베딩 캐시 로드: {cache_path}")
        emb_cache = torch.load(cache_path, map_location="cpu")
        print(f"  캐시 로드 완료: {len(emb_cache)}개")
        return emb_cache

    campplus_model.eval()
    emb_cache = {}

    from torch.utils.data import ConcatDataset
    if isinstance(dataset, ConcatDataset):
        all_paths = []
        for ds in dataset.datasets:
            all_paths.extend(ds.paths)
    else:
        all_paths = dataset.paths

    print(f"  원본 임베딩 캐싱 중... (총 {len(all_paths)}개)")

    import soundfile as sf
    import numpy as np
    from math import gcd
    from scipy.signal import resample_poly

    with torch.no_grad():
        for i, path in enumerate(all_paths):
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            audio = data.mean(axis=1).astype(np.float32)
            if sr != 16000:
                g = gcd(16000, sr)
                audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
            audio = np.clip(audio, -1.0, 1.0)

            if len(audio) >= MAX_SAMPLES:
                audio = audio[:MAX_SAMPLES]
            else:
                audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)), mode="constant")

            audio_t = torch.from_numpy(audio).unsqueeze(0).to(device)  # (1, MAX_SAMPLES)
            emb = campplus_embed_single(campplus_model, audio_t)        # (192,)
            emb_cache[path] = emb.cpu()

            if (i + 1) % 500 == 0 or (i + 1) == len(all_paths):
                print(f"    {i+1}/{len(all_paths)} 완료")

    print(f"  캐싱 완료: {len(emb_cache)}개")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(emb_cache, cache_path)
        print(f"  임베딩 캐시 저장: {cache_path}")

    return emb_cache


# -- 배치 조회 -------------------------------------------------------

def get_batch_embeddings(
    emb_cache: dict,
    paths: list,
    device: torch.device,
) -> torch.Tensor:
    """지금 배치에 해당하는 파일 경로들로 캐시에서 임베딩을 꺼내 모아준다."""
    embs = torch.stack([emb_cache[p] for p in paths]).to(device)
    return embs.detach()


# -- 체크포인트 저장/로드 ---------------------------------------------

def save_checkpoint(
    path: str,
    generator,
    optimizer,
    scheduler,
    epoch: int,
    best_dist: float,
    loss_history: list,
):
    """지금까지 학습된 모델과 학습 상태(optimizer, scheduler 등)를 파일로 저장한다."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":        epoch,
        "best_dist":    best_dist,
        "loss_history": loss_history,
        "generator":    generator.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "scheduler":    scheduler.state_dict() if scheduler is not None else None,
        "eps":          generator.eps,
    }, path)


def load_checkpoint(
    path: str,
    generator,
    optimizer,
    scheduler,
    device: torch.device,
    reset_scheduler: bool = False,
):
    """
    저장해둔 체크포인트를 불러와서 이어서 학습할 수 있게 준비한다.

    reset_scheduler를 True로 주면, 저장돼 있던 학습률 스케줄은 무시하고
    새로 만든 스케줄을 그대로 쓴다. 총 epoch 수를 늘려서 다시 돌릴 때 쓰는 옵션.
    """
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt["epoch"] + 1
    if reset_scheduler:
        # optimizer의 lr도 체크포인트 시점의 감쇠된 값이 저장돼 있으므로
        # scheduler 초기 lr(base_lrs)로 되돌려야 새 코사인 스케줄이 의도대로 동작
        for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
            group["lr"] = base_lr
        # scheduler는 T_max=args.epochs(총 epoch)로 만들어져 있지만 last_epoch=0부터
        # 다시 세므로, 실제로 도는 epoch 수(args.epochs - start_epoch)에 맞춰 T_max를 재설정
        scheduler.T_max = max(scheduler.T_max - start_epoch, 1)
    elif scheduler is not None and ckpt["scheduler"] is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return start_epoch, ckpt["best_dist"], ckpt.get("loss_history", [])
