"""XTTS v2 음성 클로닝 어댑터.

TTSEvaluator (evaluators/tts.py)의 xtts_model 인자로 주입.
coqui-tts (idiap fork) 패키지를 사용한다.
모델 weights는 HuggingFace에서 자동 다운로드.
"""

from __future__ import annotations

import logging
import tempfile

import numpy as np
import torch

from voicesecure.types import SAMPLE_RATE, AudioArray

logger = logging.getLogger(__name__)

_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"


def _ensure_ffmpeg_path() -> None:
    """Windows에서 winget으로 설치된 FFmpeg shared bin을 PATH에 자동 추가."""
    import os
    import sys

    if sys.platform != "win32":
        return

    winget_pkgs = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
    if not os.path.isdir(winget_pkgs):
        return

    for entry in os.listdir(winget_pkgs):
        if "FFmpeg.Shared" in entry:
            for sub in os.listdir(os.path.join(winget_pkgs, entry)):
                bin_path = os.path.join(winget_pkgs, entry, sub, "bin")
                if os.path.isdir(bin_path) and bin_path not in os.environ["PATH"]:
                    os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
                    logger.debug("Added FFmpeg shared bin to PATH: %s", bin_path)
                    return


_CLONE_TEXT = "안녕하세요"
_LANGUAGE = "ko"


class XTTSAdapter:
    """XTTS v2를 TTSEvaluator에 끼울 수 있는 어댑터.

    TTSEvaluator는 clone(reference_audio) 메서드를 가진 객체를 받는다.
    reference_audio를 레퍼런스 화자로 사용해 _CLONE_TEXT를 합성한다.

    Args:
        model_name: coqui-tts 모델 ID.
        language:   합성 언어 코드. 기본값 'ko'.
        clone_text: 클로닝 시 합성할 텍스트.
        device:     'cpu' 또는 'cuda'. None이면 자동 감지.

    Example:
        xtts = XTTSAdapter()
        tts_eval = TTSEvaluator(xtts_model=xtts, speaker_model=wavlm)
    """

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        language: str = _LANGUAGE,
        clone_text: str = _CLONE_TEXT,
        device: str | None = None,
    ) -> None:
        from TTS.api import TTS

        _ensure_ffmpeg_path()
        self.language = language
        self.clone_text = clone_text
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading XTTS v2 on %s...", self.device)
        self._tts = TTS(model_name).to(self.device)
        logger.info("XTTSAdapter ready.")

    def clone(self, reference_audio: AudioArray) -> AudioArray:
        """reference_audio 화자 목소리로 clone_text를 합성한다.

        Args:
            reference_audio: shape (num_samples,), float32, [-1, 1], 16kHz mono

        Returns:
            합성된 음성, shape (num_samples,), float32, [-1, 1]
        """
        if reference_audio.ndim != 1:
            raise ValueError(f"reference_audio must be 1-D, got shape {reference_audio.shape}")
        if reference_audio.dtype != np.float32:
            reference_audio = reference_audio.astype(np.float32)

        # reference_audio를 임시 wav 파일로 저장 (XTTS speaker_wav 인자 요구)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_ref:
            _write_wav(tmp_ref.name, reference_audio, SAMPLE_RATE)
            ref_path = tmp_ref.name

        samples = self._tts.tts(
            text=self.clone_text,
            language=self.language,
            speaker_wav=ref_path,
        )
        cloned = np.array(samples, dtype=np.float32)

        # XTTS 출력은 24000Hz — 16kHz로 리샘플링
        from math import gcd

        from scipy.signal import resample_poly

        xtts_sr = self._tts.synthesizer.output_sample_rate
        if xtts_sr != SAMPLE_RATE:
            g = gcd(SAMPLE_RATE, xtts_sr)
            cloned = resample_poly(cloned, SAMPLE_RATE // g, xtts_sr // g).astype(np.float32)

        return cloned


def _write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    """numpy array를 16-bit PCM wav 파일로 저장."""
    import soundfile as sf

    sf.write(path, audio, sample_rate, subtype="PCM_16")


def _read_wav(path: str) -> np.ndarray:
    """wav 파일을 float32 numpy array로 읽는다."""
    import soundfile as sf

    samples, _ = sf.read(path, dtype="float32")
    if samples.ndim > 1:
        samples = samples[:, 0]  # mono
    return samples
