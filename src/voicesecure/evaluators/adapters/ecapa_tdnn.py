"""ECAPA-TDNN 화자 검증 어댑터 (SpeechBrain).

최신 논문(2024-2025)에서 가장 많이 공격 대상으로 사용되는 모델.
WavLM과 달리 noise-robust 설계가 아니라 adversarial perturbation에 취약.
→ 같은 심리음향 마스킹 제약 안에서 WavLM보다 훨씬 큰 cosine distance 기대.

References
----------
- MEP/I-MEP (2024): ECAPA-TDNN EER 6% → 44% (176배 악화)
- FoolHD (2021): x-vector 99.6% 공격 성공률
- Interspeech 2025: ECAPA-TDNN이 가장 광범위하게 평가됨
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from voicesecure.types import AudioArray, Embedding

logger = logging.getLogger(__name__)

_SPEECHBRAIN_MODEL = "speechbrain/spkrec-ecapa-voxceleb"


class ECAPATDNNAdapter:
    """SpeechBrain ECAPA-TDNN 화자 임베딩 어댑터.

    WavLM과 동일한 인터페이스(extract_embedding)를 제공.
    PyTorch 모델이라 gradient 추적 가능 → FGSM 공격에 활용 가능.

    Args:
        device: 'cpu' 또는 'cuda'. None이면 자동 감지.

    Example:
        ecapa = ECAPATDNNAdapter()
        emb = ecapa.extract_embedding(audio)  # shape (192,)
    """

    def __init__(self, device: str | None = None) -> None:
        from speechbrain.inference.speaker import EncoderClassifier

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info("ECAPA-TDNN 로드 중: %s", _SPEECHBRAIN_MODEL)
        self.model = EncoderClassifier.from_hparams(
            source=_SPEECHBRAIN_MODEL,
            run_opts={"device": device},
            savedir="pretrained_models/ecapa_tdnn",
        )
        self.model.eval()
        logger.info("ECAPA-TDNN 로드 완료 (device=%s)", device)

    def extract_embedding(self, audio: AudioArray) -> Embedding:
        """음성 → 화자 임베딩.

        Args:
            audio: shape (num_samples,), float32, [-1, 1], 16kHz mono

        Returns:
            shape (192,), float32
        """
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # (1, T)
        with torch.no_grad():
            emb = self.model.encode_batch(audio_tensor)  # (1, 1, 192)
        return emb.squeeze().cpu().numpy().astype(np.float32)

    def extract_embedding_grad(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """gradient 추적용 임베딩 추출 (FGSM에서 사용).

        Args:
            audio_tensor: shape (1, T), requires_grad=True

        Returns:
            shape (192,), gradient 추적 가능
        """
        emb = self.model.encode_batch(audio_tensor)  # (1, 1, 192)
        return emb.squeeze()
