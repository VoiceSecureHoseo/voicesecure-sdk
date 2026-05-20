"""WavLM-SV 화자 검증 어댑터.

SpeakerEvaluator (evaluators/speaker.py)의 wavlm_model 인자로 주입.
모델 weights는 HuggingFace에서 다운로드 (repo에 포함하지 않음).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

from voicesecure.types import SAMPLE_RATE, AudioArray, Embedding

logger = logging.getLogger(__name__)

_MODEL_NAME = "microsoft/wavlm-base-plus-sv"


class WavLMSVAdapter:
    """WavLM-SV를 SpeakerEvaluator에 끼울 수 있는 어댑터.

    SpeakerEvaluator는 extract_embedding(audio) 메서드를 가진 객체를 받는다.

    Args:
        model_name: HuggingFace 모델 ID. 기본값은 wavlm-base-plus-sv.
        device:     추론 디바이스. None이면 CUDA 자동 감지.

    Example:
        wavlm = WavLMSVAdapter()
        sv_eval = SpeakerEvaluator(wavlm_model=wavlm, cam_model=...)
    """

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        logger.info("Loading %s on %s...", model_name, self.device)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMForXVector.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info("WavLMSVAdapter ready.")

    def extract_embedding(self, audio: AudioArray) -> Embedding:
        """음성 → 화자 embedding.

        WavLMForXVector의 embeddings 출력을 사용한다.
        (last_hidden_state mean pooling이 아닌 SV head 출력)

        Args:
            audio: shape (num_samples,), float32, [-1, 1], 16kHz mono

        Returns:
            shape (512,), float32. 화자 특성 벡터.
        """
        if audio.ndim != 1:
            raise ValueError(f"audio must be 1-D, got shape {audio.shape}")
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        with torch.no_grad():
            inputs = self.feature_extractor(
                audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
            )
            input_values = inputs.input_values.to(self.device)
            outputs = self.model(input_values)

        # WavLMForXVector의 embeddings: (batch, 512) — SV용 head 출력
        embedding = outputs.embeddings.squeeze(0).cpu().numpy()
        return embedding.astype(np.float32)
