"""
epoch 하나, 즉 전체 데이터를 한 바퀴 훑으면서 학습시키는 부분

데이터를 batch 단위로 조금씩 꺼내서 모델에 통과시키고(forward), 얼마나
잘했는지 점수를 매기고(loss), 그 점수를 바탕으로 모델을 조금 수정하는
(backward + step) 과정을 데이터가 끝날 때까지 반복한다. 보통 이 한
바퀴(1 epoch)를 여러 번 반복해서 모델을 점점 더 좋게 만든다.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader

from train_utils import get_batch_embeddings


# -- epoch 루프 ----------------------------------------------------

def train_one_epoch(
    generator,
    mel_extractor,
    campplus_model,
    criterion,
    optimizer,
    loader: DataLoader,
    emb_cache: dict,
    device: torch.device,
    grad_clip: float = 1.0,
) -> dict:
    """
    데이터 전체를 한 바퀴 돌면서 모델을 학습시키고, 끝나면 이번 epoch의
    평균 loss/dist 등을 담은 dict를 돌려준다.

    generator는 학습 대상(train 모드)이고, mel_extractor와 campplus_model은
    이미 완성된 도구라 값을 그대로 쓰기만 한다(eval 모드, 수정 안 됨).
    grad_clip은 한 번에 모델이 너무 크게 바뀌지 않도록 잡아주는 안전장치.
    """
    generator.train()
    mel_extractor.eval()
    campplus_model.eval()

    sum_loss   = 0.0
    sum_cam    = 0.0
    sum_snr    = 0.0
    sum_psycho = 0.0
    sum_linf   = 0.0
    sum_dist   = 0.0
    n_batch    = 0
    n_total    = len(loader)

    import time
    t_epoch = time.time()

    for orig_batch, paths in loader:
        # 원본 소리 배치와 각 파일 경로
        orig_batch = orig_batch.to(device)                     # (B, T)
        orig_batch_np = orig_batch.detach().cpu().numpy()      # psycho 계산용 numpy 버전

        # 미리 계산해둔 원본 화자 임베딩을 캐시에서 꺼낸다 (다시 계산 안 함)
        orig_embs = get_batch_embeddings(emb_cache, paths, device)  # (B, 192)

        # mel spectrogram 추출. MelExtractor는 학습 대상이 아니라서 그대로 계산만 한다
        with torch.no_grad():
            mel = mel_extractor(orig_batch).detach()           # (B, 1, 80, T')

        # Generator에 mel을 넣어 노이즈(delta)를 만든다.
        # 여기서부터 이어지는 계산 결과가 나중에 Generator를 수정하는 데 쓰인다.
        delta = generator(mel)                                 # (B, T)

        # 노이즈를 원본에 더해서 보호된(adv) 소리를 만든다
        adv = torch.clamp(orig_batch.detach() + delta, -1.0, 1.0)  # (B, T)

        # 점수 계산
        result = criterion(
            adv_batch=adv,
            orig_batch=orig_batch.detach(),
            orig_embs=orig_embs,
            orig_batch_np=orig_batch_np,
            delta_batch=delta,
        )

        # 점수를 바탕으로 Generator를 조금 수정한다
        optimizer.zero_grad()
        result["total"].backward()
        # 여러 GPU를 쓰는 경우를 대비한 파라미터 참조 방식
        params = (generator.module if hasattr(generator, "module") else generator).parameters()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()

        # dist는 l_cam의 부호만 바꾼 값이라 CAM++를 다시 돌릴 필요 없이 바로 계산
        avg_dist = -result["cam"].item()

        # 누적
        sum_loss   += result["total"].item()
        sum_cam    += result["cam"].item()
        sum_snr    += result["snr"].item()
        sum_psycho += result["psycho"].item()
        sum_linf   += result["linf"].item()
        sum_dist   += avg_dist
        n_batch    += 1

        if n_batch % 10 == 0:
            elapsed = time.time() - t_epoch
            spd = elapsed / n_batch
            eta = spd * (n_total - n_batch)
            w_p = result.get("w_psycho", None)
            w_p_str = f" w_psycho={w_p:.3f}" if w_p is not None else ""
            print(
                f"  step {n_batch:4d}/{n_total} "
                f"loss={sum_loss/n_batch:+.4f} "
                f"dist={sum_dist/n_batch:.4f} "
                f"psycho={sum_psycho/n_batch:.5f}"
                f"{w_p_str} "
                f"spd={spd:.2f}s/batch "
                f"ETA={eta:.0f}s",
                flush=True
            )

    if n_batch == 0:
        return {"loss": 0, "cam": 0, "snr": 0, "psycho": 0,
                "linf": 0, "dist": 0, "n_batch": 0}

    return {
        "loss":    sum_loss   / n_batch,
        "cam":     sum_cam    / n_batch,
        "snr":     sum_snr    / n_batch,
        "psycho":  sum_psycho / n_batch,
        "linf":    sum_linf   / n_batch,
        "dist":    sum_dist   / n_batch,
        "n_batch": n_batch,
    }
