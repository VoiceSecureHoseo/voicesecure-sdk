"""wav2vec2 한국어 ASR 어댑터.

ASREvaluator (evaluators/asr.py)의 asr_model 인자로 주입.
모델 weights는 HuggingFace에서 다운로드 (repo에 포함하지 않음).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from voicesecure.types import SAMPLE_RATE, AudioArray

logger = logging.getLogger(__name__)

_MODEL_NAME = "kresnik/wav2vec2-large-xlsr-korean"


class Wav2Vec2KoreanAdapter:
    """wav2vec2 한국어 ASR을 ASREvaluator에 끼울 수 있는 어댑터.

    ASREvaluator는 transcribe(audio) 메서드를 가진 객체를 받는다.

    Args:
        model_name: HuggingFace 모델 ID. 기본값은 wav2vec2-large-xlsr-korean.
        device:     추론 디바이스. None이면 CUDA 자동 감지.

    Example:
        asr = Wav2Vec2KoreanAdapter()
        asr_eval = ASREvaluator(asr_model=asr, original_text="안녕하세요")
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
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info("Wav2Vec2KoreanAdapter ready.")

    def transcribe(self, audio: AudioArray) -> str:
        """음성 → 한국어 텍스트.

        Args:
            audio: shape (num_samples,), float32, [-1, 1], 16kHz mono

        Returns:
            인식된 한국어 텍스트. 인식 실패 시 빈 문자열.
        """
        if audio.ndim != 1:
            raise ValueError(f"audio must be 1-D, got shape {audio.shape}")
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        with torch.no_grad():
            inputs = self.processor(
                audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
            )
            input_values = inputs.input_values.to(self.device)
            logits = self.model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription
