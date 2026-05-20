"""CosyVoice3 음성 클로닝 + 화자 임베딩 어댑터.

훈련 환경: Colab (Linux + GPU) 전용.
역할:
    1. clone(reference_audio, text) — 변조 음성을 레퍼런스로 TTS 클로닝
    2. extract_embedding(audio)     — CosyVoice3 내부 CAM++ (192-dim) 화자 임베딩 추출

[중요] extract_embedding은 CosyVoice3 내부 CAM++을 직접 사용한다.
       Wespeaker CAM++ (campplus.py) 과 다른 모델/임베딩 공간 (192-dim vs 512-dim).
       훈련 시 cam_model 자리에 이 어댑터를 주입한다.

설치 (코랩):
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
    pip install -r CosyVoice/requirements.txt
    python -c "from huggingface_hub import snapshot_download; \
        snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', \
        local_dir='CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512')"
"""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
import soundfile as sf

from voicesecure.types import SAMPLE_RATE, AudioArray, Embedding

logger = logging.getLogger(__name__)

_MODEL_DIR = "pretrained_models/Fun-CosyVoice3-0.5B-2512"
_DEFAULT_TEXT = "안녕하세요"


class CosyVoiceAdapter:
    """CosyVoice3 클로닝 + 화자 임베딩 어댑터 (Colab 전용).

    clone()과 extract_embedding() 두 인터페이스를 모두 제공해서
    train.py에서 tts_model과 cam_model 자리에 동시에 주입 가능.

    Args:
        model_dir:      CosyVoice3 pretrained_models 경로.
        cosyvoice_root: CosyVoice 레포 루트 경로 (sys.path에 추가).

    Example (코랩):
        cosy = CosyVoiceAdapter(model_dir='CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512')
        # 클로닝
        cloned = cosy.clone(modified_audio, text="그는 괜찮은 척하려고 애쓰는 것 같았다.")
        # 화자 임베딩 (CAM++ 192-dim)
        embedding = cosy.extract_embedding(audio)
    """

    def __init__(
        self,
        model_dir: str = _MODEL_DIR,
        cosyvoice_root: str = "CosyVoice",
        device: str | None = None,
    ) -> None:
        import sys

        sys.path.insert(0, cosyvoice_root)
        sys.path.insert(0, os.path.join(cosyvoice_root, "third_party", "Matcha-TTS"))

        from cosyvoice.cli.cosyvoice import CosyVoice3

        # device 미지정 시 CUDA 자동 감지
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        logger.info("Loading CosyVoice3 from %s (device=%s)...", model_dir, device)
        self._model = CosyVoice3(model_dir, device=device)
        self._sample_rate = self._model.sample_rate
        logger.info("CosyVoiceAdapter ready (sample_rate=%d).", self._sample_rate)

    def clone(self, reference_audio: AudioArray, text: str = _DEFAULT_TEXT) -> AudioArray:
        """reference_audio 화자 목소리로 text를 합성한다.

        훈련 시 Labels.txt에서 읽은 원본 텍스트를 text 인자로 전달한다.
        → 같은 텍스트로 클로닝해서 텍스트 변수 없이 화자 방어 효과만 측정 가능.

        Args:
            reference_audio: 변조된 음성, shape (num_samples,), float32, [-1, 1], 16kHz mono
            text:            합성할 텍스트. 기본값 "안녕하세요".

        Returns:
            클론 음성, shape (num_samples,), float32, [-1, 1]
        """
        if reference_audio.ndim != 1:
            raise ValueError(f"reference_audio must be 1-D, got shape {reference_audio.shape}")
        if reference_audio.dtype != np.float32:
            reference_audio = reference_audio.astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, reference_audio, SAMPLE_RATE)
            ref_path = tmp.name

        try:
            chunks = []
            for result in self._model.inference_zero_shot(
                text,
                text,  # prompt_text = clone_text (같은 텍스트로 zero-shot)
                ref_path,
                stream=False,
            ):
                chunks.append(result["tts_speech"].squeeze().numpy())
        finally:
            os.unlink(ref_path)

        cloned = np.concatenate(chunks).astype(np.float32)
        cloned = np.clip(cloned, -1.0, 1.0)
        return cloned

    def extract_embedding(self, audio: AudioArray) -> Embedding:
        """CosyVoice3 내부 CAM++ (192-dim) 화자 임베딩을 추출한다.

        CosyVoice3가 클로닝 시 내부적으로 사용하는 것과 동일한 CAM++ 모델.
        → 클로닝 조건과 동일한 임베딩 공간에서 방어 효과 측정 가능.

        Args:
            audio: shape (num_samples,), float32, [-1, 1], 16kHz mono

        Returns:
            shape (192,), float32. 화자 특성 벡터.
        """
        if audio.ndim != 1:
            raise ValueError(f"audio must be 1-D, got shape {audio.shape}")
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, SAMPLE_RATE)
            wav_path = tmp.name

        try:
            # CosyVoice3 frontend의 CAM++ ONNX 세션으로 직접 임베딩 추출
            embedding = self._model.frontend._extract_spk_embedding(wav_path)
        finally:
            os.unlink(wav_path)

        # tensor → numpy 1-D float32
        emb_np = embedding.squeeze().cpu().numpy().astype(np.float32)
        return emb_np
