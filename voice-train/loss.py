"""
Generator가 학습 중에 얼마나 잘하고 있는지 점수를 매기는 Loss 함수

4가지 목표를 동시에 만족시켜야 해서 점수를 4개 더해서 하나로 합친다.
  1. CAM++가 원본 목소리와 다르게 인식하게 만들기 (제일 중요한 목표)
  2. 사람 귀에는 최대한 원본과 비슷하게 들리도록 SNR 28dB 이상 유지하기
  3. 심리음향 마스킹 기준을 넘는 소음은 추가 페널티 주기 (더 안 들리게)
  4. perturbation 크기가 eps(허용 한계)를 넘지 않도록 잡아주기

이 4개 점수에 각각 가중치(w_cam, w_snr, w_psycho, w_linf)를 곱해서 더한 게 최종 loss.
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F

# psycho 모듈
_GEN_DIR = os.path.dirname(os.path.abspath(__file__))
if _GEN_DIR not in sys.path:
    sys.path.insert(0, _GEN_DIR)

# pgd_verify.py 경로
_PGD_DIR = os.path.dirname(_GEN_DIR)
if _PGD_DIR not in sys.path:
    sys.path.insert(0, _PGD_DIR)

from psycho import soft_psycho_loss, compute_threshold
from features import extract_kaldi_fbank

SNR_TARGET  = 28.0   # dB, PGD 최고 결과(#3) 기준
EPS_DEFAULT = 0.01


# -- 유틸 ------------------------------------------------------

def cosine_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    두 화자 임베딩(목소리 특징 값)이 얼마나 다른 방향을 가리키는지 재는 값.
    0에 가까우면 같은 사람처럼 인식된다는 뜻이고, 클수록(최대 2) 다르게 인식된다.
    """
    a_n = F.normalize(a.unsqueeze(0), dim=1)
    b_n = F.normalize(b.unsqueeze(0), dim=1)
    return 1.0 - (a_n * b_n).sum()


def campplus_embed_single(campplus_model, audio_t: torch.Tensor) -> torch.Tensor:
    """
    소리 하나를 CAM++에 통과시켜서 화자 임베딩(그 목소리의 특징을 숫자로
    표현한 값)을 뽑는다. 학습 중 역전파 신호가 계속 흐르도록 계산한다.
    """
    feat = extract_kaldi_fbank(audio_t)   # (T', 80)
    emb  = campplus_model(feat.unsqueeze(0))        # (1, 192)
    return emb.squeeze(0)                           # (192,)


def snr_db(original: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    원본 소리 대비 노이즈가 얼마나 작은지를 SNR(dB)로 계산한다.
    값이 클수록 노이즈가 상대적으로 작다는 뜻 (원음이 잘 보존됨).
    """
    signal_power = original.pow(2).mean() + 1e-12
    noise_power  = delta.pow(2).mean()   + 1e-12
    return 10.0 * torch.log10(signal_power / noise_power)


# -- 개별 Loss 항 ----------------------------------------------

def l_cam(campplus_model, adv_batch: torch.Tensor, orig_embs: torch.Tensor) -> torch.Tensor:
    """
    노이즈를 입힌 소리(adv_batch)가 원본과 최대한 다른 사람처럼 인식되도록
    유도하는 점수. 점수를 낮추는(minimize) 방향으로 학습하면 결과적으로
    화자 인식 거리(dist)가 최대화된다. (값을 음수로 만들어서 이렇게 처리)
    """
    dists = []
    for i in range(adv_batch.shape[0]):
        adv_emb = campplus_embed_single(campplus_model, adv_batch[i:i+1])
        d = cosine_dist(adv_emb, orig_embs[i])
        dists.append(d)
    return -torch.stack(dists).mean()


def l_snr(original_batch: torch.Tensor, delta_batch: torch.Tensor,
          snr_target: float = SNR_TARGET) -> torch.Tensor:
    """
    SNR이 목표치(기본 28dB)를 넘으면 벌점이 0이고, 못 미치면 부족한 만큼
    벌점을 준다. 사람 귀에 너무 티나는 노이즈가 안 생기게 막아주는 역할.
    """
    penalties = []
    for i in range(original_batch.shape[0]):
        snr = snr_db(original_batch[i], delta_batch[i])
        deficit = F.relu(snr_target - snr)           # 목표까지 부족한 dB
        ratio = deficit / snr_target                 # 0~1 정규화
        penalty = ratio ** 2 * snr_target            # 제곱 곡선: 목표 근처에서 급감
        penalties.append(penalty)
    return torch.stack(penalties).mean()


def l_psycho(delta_batch: torch.Tensor, orig_batch_np: np.ndarray) -> torch.Tensor:
    """
    노이즈가 "사람이 못 듣는 한계선"을 넘는 만큼 벌점을 준다.
    (psycho.py의 심리음향 마스킹 원리를 학습 loss로 쓰는 부분)
    """
    losses = []
    for i in range(delta_batch.shape[0]):
        threshold = compute_threshold(orig_batch_np[i])
        l = soft_psycho_loss(delta_batch[i], threshold)
        losses.append(l)
    return torch.stack(losses).mean()


def l_linf(delta_batch: torch.Tensor, eps: float = EPS_DEFAULT) -> torch.Tensor:
    """
    노이즈 크기가 eps(허용 한계)를 넘지 않도록 억제한다. eps의 80%까지는
    가볍게 페널티를 주고, 그걸 넘으면 훨씬 강하게 눌러서 한계를 벗어나지
    않게 하면서도, 한계 근처까지는 적극적으로 노이즈를 쓰도록 유도한다.
    """
    abs_delta = delta_batch.abs()
    threshold = eps * 0.8
    linear_part = F.relu(abs_delta - threshold) * 0.1
    quad_part   = F.relu(abs_delta - eps) ** 2 * 100.0
    return (linear_part + quad_part).mean()


# -- 통합 Loss -------------------------------------------------

class GeneratorLoss:
    """
    위의 4개 점수(l_cam, l_snr, l_psycho, l_linf)를 각자 가중치를 곱해서
    하나의 총점으로 합쳐주는 계산기. 학습 루프에서 이 클래스 하나만 호출하면 된다.
    """

    def __init__(
        self,
        campplus_model,
        eps:        float = EPS_DEFAULT,
        snr_target: float = SNR_TARGET,
        w_cam:      float = 1.0,
        w_snr:      float = 0.1,
        w_psycho:   float = 0.05,
        w_linf:     float = 10.0,
    ):
        self.campplus_model = campplus_model
        self.eps        = eps
        self.snr_target = snr_target
        self.w_cam      = w_cam
        self.w_snr      = w_snr
        self.w_psycho   = w_psycho
        self.w_linf     = w_linf

    def __call__(
        self,
        adv_batch:      torch.Tensor,   # (B, T) adv wav, gradient 연결됨
        orig_batch:     torch.Tensor,   # (B, T) 원본 wav, detach
        orig_embs:      torch.Tensor,   # (B, 192) 원본 CAM++ 임베딩, detach
        orig_batch_np:  np.ndarray,     # (B, T) 원본 wav numpy (psycho용)
        delta_batch:    torch.Tensor,   # (B, T) perturbation, gradient 연결됨
    ) -> dict:
        """4개 점수를 각각 계산하고 가중합한 총점(total)과 개별 값을 함께 돌려준다."""
        raw_cam    = l_cam(self.campplus_model, adv_batch, orig_embs)
        raw_snr    = l_snr(orig_batch, delta_batch, self.snr_target)
        raw_psycho = l_psycho(delta_batch, orig_batch_np)
        raw_linf   = l_linf(delta_batch, self.eps)

        # 동적 w_snr: 현재 배치 평균 SNR이 목표보다 낮을수록 w_snr 증가
        with torch.no_grad():
            cur_snr = torch.stack([
                snr_db(orig_batch[i], delta_batch[i])
                for i in range(orig_batch.shape[0])
            ]).mean().item()
        deficit_ratio = max(0.0, (self.snr_target - cur_snr) / self.snr_target)
        w_snr_dynamic = self.w_snr * (1.0 + deficit_ratio * 9.0)  # 최대 10x

        total = (self.w_cam    * raw_cam
               + w_snr_dynamic * raw_snr
               + self.w_psycho * raw_psycho
               + self.w_linf   * raw_linf)

        return {
            "total":  total,
            "cam":    raw_cam.detach(),     # 가중치 미적용 raw값 - 모니터링용
            "snr":    raw_snr.detach(),
            "psycho": raw_psycho.detach(),
            "linf":   raw_linf.detach(),
        }
