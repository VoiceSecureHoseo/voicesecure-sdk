"""CosyVoice3 음성 클로닝 + 화자 임베딩 어댑터.

═══════════════════════════════════════════════════════════════════════════
[V1] 원본 대비 수정사항
═══════════════════════════════════════════════════════════════════════════
1. clone(): 모델 출력(24kHz, cosyvoice2.yaml sample_rate=24000)을 SDK 계약인
   16kHz로 리샘플 후 반환하도록 수정.
   - 왜: 기존엔 24kHz를 그대로 반환했는데, extract_embedding()이 입력을
     16kHz 헤더(SAMPLE_RATE)로 디스크에 쓰기 때문에 CosyVoice frontend의
     load_wav가 헤더를 믿고 리샘플 없이 읽음 → 클론 임베딩이 1.5배
     시간-늘어진(피치 내려간) 오디오에서 계산됨. 원본 임베딩(정상 16kHz)과의
     cosine distance가 perturbation과 무관한 아티팩트로 부풀려져 tts reward가
     오염됨. WavLM(feature_extractor sampling_rate=16000 고정)·ECAPA
     (speechbrain 16kHz, 리샘플 없음)도 모두 16kHz 가정이므로 평가 셀의
     절대 수치도 같은 이유로 왜곡됐었음.
   - 수정 근거: 같은 SDK의 XTTSAdapter.clone()은 이미 "24kHz 출력 → 16kHz
     리샘플 후 반환"을 계약으로 명시·구현하고 있음(_resample_to_16k).
     CosyVoiceAdapter만 이 계약을 어기고 있었으므로 동일 방식
     (scipy.signal.resample_poly, gcd 기반 정수비)으로 통일.
2. clone() docstring의 Returns를 16kHz 명시로 갱신.
3. extract_embedding() docstring에 "입력은 반드시 16kHz" 계약을 명시.
   (sf.write가 SAMPLE_RATE=16000 헤더로 고정 저장하므로, 이 계약이 지켜져야
   CosyVoice frontend가 올바른 시간축으로 fbank를 계산함. 1번 수정으로
   clone 출력도 이 계약을 만족하게 됨.)
═══════════════════════════════════════════════════════════════════════════

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
from math import gcd  # [V1 추가] 16kHz 리샘플용 (XTTSAdapter._resample_to_16k와 동일 방식)

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly  # [V1 추가] 16kHz 리샘플용

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

        # model_dir 이름으로 v2/v3 자동 감지
        # CosyVoice2-0.5B 는 한국어 공식 지원, CosyVoice3 는 endofprompt 토큰 요구
        self._is_v2 = "CosyVoice2" in model_dir or "cosyvoice2" in model_dir.lower()

        if self._is_v2:
            from cosyvoice.cli.cosyvoice import CosyVoice2 as CosyVoiceClass

            version = "CosyVoice2"
        else:
            from cosyvoice.cli.cosyvoice import CosyVoice3 as CosyVoiceClass

            version = "CosyVoice3"

        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        logger.info("Loading %s from %s (auto device: %s)...", version, model_dir, device)
        self._model = CosyVoiceClass(model_dir, load_jit=False, load_trt=False, fp16=False)

        # transformers/Qwen2 호환성: LLM weights 는 BFloat16 인데 fp16=False 라
        # autocast 없음 → dtype mismatch 방지를 위해 fp32 강제.
        if hasattr(self._model, "model") and hasattr(self._model.model, "llm"):
            self._model.model.llm = self._model.model.llm.float()
            logger.info("%s LLM를 float32로 변환 (BF16 weights 호환성).", version)

        self._sample_rate = self._model.sample_rate
        logger.info(
            "CosyVoiceAdapter ready (version=%s, sample_rate=%d).", version, self._sample_rate
        )

    def clone(self, reference_audio: AudioArray, text: str = _DEFAULT_TEXT) -> AudioArray:
        """reference_audio 화자 목소리로 text를 합성한다.

        훈련 시 Labels.txt에서 읽은 원본 텍스트를 text 인자로 전달한다.
        → 같은 텍스트로 클로닝해서 텍스트 변수 없이 화자 방어 효과만 측정 가능.

        Args:
            reference_audio: 변조된 음성, shape (num_samples,), float32, [-1, 1], 16kHz mono
            text:            합성할 텍스트. 기본값 "안녕하세요".

        Returns:
            클론 음성, shape (num_samples,), float32, [-1, 1], **16kHz mono**
            (CosyVoice2/3의 24kHz 출력을 16kHz로 리샘플링한 결과 — XTTSAdapter와 동일 계약)
        """
        if reference_audio.ndim != 1:
            raise ValueError(f"reference_audio must be 1-D, got shape {reference_audio.shape}")
        if reference_audio.dtype != np.float32:
            reference_audio = reference_audio.astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, reference_audio, SAMPLE_RATE)
            ref_path = tmp.name

        try:
            # CosyVoice3 는 prompt_text 끝에 <|endofprompt|> (token 151646) 요구.
            # CosyVoice2 는 endofprompt 불필요 — assertion 안 함.
            if self._is_v2:
                prompt_text = text
            else:
                end_of_prompt = "<|endofprompt|>"
                prompt_text = text if text.endswith(end_of_prompt) else text + end_of_prompt
            chunks = []
            for result in self._model.inference_zero_shot(
                text,
                prompt_text,
                ref_path,
                stream=False,
            ):
                chunks.append(result["tts_speech"].squeeze().numpy())
        finally:
            os.unlink(ref_path)

        cloned = np.concatenate(chunks).astype(np.float32)

        # [V1 수정] 모델 출력(self._sample_rate, CosyVoice2-0.5B 기준 24000Hz)을
        # SDK 오디오 계약(AudioArray = 16kHz mono float32)으로 리샘플.
        # 기존: 24kHz 그대로 반환 → extract_embedding()의 16kHz 헤더 저장과 결합해
        #       클론 임베딩이 1.5배 시간-늘어진 오디오에서 계산됨 (reward 오염).
        # 수정: XTTSAdapter.clone()과 동일하게 어댑터 내부에서 16kHz로 통일.
        if self._sample_rate != SAMPLE_RATE:
            g = gcd(SAMPLE_RATE, self._sample_rate)
            cloned = resample_poly(
                cloned, SAMPLE_RATE // g, self._sample_rate // g
            ).astype(np.float32)

        cloned = np.clip(cloned, -1.0, 1.0)
        return cloned

    def extract_embedding(self, audio: AudioArray) -> Embedding:
        """CosyVoice3 내부 CAM++ (192-dim) 화자 임베딩을 추출한다.

        CosyVoice3가 클로닝 시 내부적으로 사용하는 것과 동일한 CAM++ 모델.
        → 클로닝 조건과 동일한 임베딩 공간에서 방어 효과 측정 가능.

        Args:
            audio: shape (num_samples,), float32, [-1, 1], **반드시 16kHz mono**
                [V1 명시] 내부에서 sf.write(..., SAMPLE_RATE=16000) 헤더로 고정
                저장하므로, 16kHz가 아닌 입력이 들어오면 CosyVoice frontend의
                load_wav가 잘못된 시간축으로 읽어 임베딩이 왜곡된다.
                (clone()이 V1부터 16kHz를 반환하므로 정상 경로에서는 항상 만족)

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
