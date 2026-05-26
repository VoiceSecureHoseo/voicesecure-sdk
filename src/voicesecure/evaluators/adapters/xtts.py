"""XTTS v2 음성 클로닝 어댑터 (저수준 Xtts API 직접 호출).

TTSEvaluator (evaluators/tts.py)의 xtts_model 인자로 주입.
coqui-tts 패키지의 고수준 ``TTS.api.TTS`` 대신 ``XttsConfig`` + ``Xtts``를
직접 사용해 디코딩 파라미터(temperature, repetition_penalty 등)를 노출한다.

배경
----
``TTS.api.TTS``로 한국어 합성 시 마침표 주변 아티팩트, 끝부분 반복/잡음,
출력 흔들림이 잦았다. 다음 세 가지로 개선:
  1) 텍스트 전처리: ``.`` → ``!`` (마침표 주변 아티팩트 감소)
  2) ``temperature=0.65`` (디폴트보다 낮춰 출력 안정화)
  3) ``repetition_penalty=10.0`` (반복/잡음 억제)

모델 weights는 다음 순서로 탐색:
  1) ``model_dir`` 인자로 명시된 경로
  2) coqui-tts 캐시 위치 (Windows: ``%LOCALAPPDATA%\\tts``,
     Linux/macOS: ``~/.local/share/tts``)
  3) ``Xtts.from_pretrained`` 로 HuggingFace에서 자동 다운로드 (가능 시)
없으면 ``FileNotFoundError`` — 수동으로 받아 ``model_dir`` 지정해야 한다.

참고
----
- https://developer-bing-gu.tistory.com/entry/Coqui-XTTS-v2-오류-해결-후기-speakerwav-적용부터-Python-API-최종-성공까지
"""

from __future__ import annotations

import glob
import logging
import os
import sys
import tempfile

import numpy as np
import torch

from voicesecure.types import SAMPLE_RATE, AudioArray

logger = logging.getLogger(__name__)

# ── 상수 ─────────────────────────────────────────────────────────────────────

_XTTS_OUTPUT_SR = 24000  # XTTS v2 고정 출력 샘플레이트
_DEFAULT_LANGUAGE = "ko"
_DEFAULT_CLONE_TEXT = "안녕하세요. 한국어 음성 합성 테스트입니다."

# 블로그 권장 디코딩 파라미터 — 한국어 안정성에 맞춰 튜닝됨
_DEFAULT_TEMPERATURE = 0.65
_DEFAULT_REPETITION_PENALTY = 10.0
_DEFAULT_TOP_K = 50
_DEFAULT_TOP_P = 0.85

# coqui-tts 캐시에서 XTTS v2 모델이 풀리는 폴더 이름
_XTTS_FOLDER_NAME = "tts_models--multilingual--multi-dataset--xtts_v2"


# ── 환경 유틸 ─────────────────────────────────────────────────────────────────


def _ensure_ffmpeg_path() -> None:
    """Windows에서 winget으로 설치된 FFmpeg shared bin을 PATH에 자동 추가."""
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


def _candidate_cache_roots() -> list[str]:
    """coqui-tts가 모델을 풀어두는 OS별 캐시 경로 후보."""
    if sys.platform == "win32":
        roots = [
            os.path.expandvars(r"%LOCALAPPDATA%\tts"),
        ]
    elif sys.platform == "darwin":
        roots = [
            os.path.expanduser("~/Library/Application Support/tts"),
        ]
    else:
        roots = [
            os.path.expanduser("~/.local/share/tts"),
        ]
    # 사용자 정의 캐시
    env = os.environ.get("TTS_HOME")
    if env:
        roots.insert(0, env)
    return roots


def _find_xtts_model_dir() -> str | None:
    """coqui-tts 캐시에서 XTTS v2 모델 폴더를 찾는다. 없으면 None."""
    for root in _candidate_cache_roots():
        pattern = os.path.join(root, _XTTS_FOLDER_NAME)
        for hit in glob.glob(pattern):
            if _looks_like_xtts_dir(hit):
                return hit
    return None


def _looks_like_xtts_dir(path: str) -> bool:
    """config.json + model 가중치 파일이 모두 있는지 가벼운 검증."""
    if not os.path.isdir(path):
        return False
    has_config = os.path.isfile(os.path.join(path, "config.json"))
    # XTTS v2는 model.pth (또는 model.safetensors)와 vocab.json을 갖는다
    has_weights = os.path.isfile(os.path.join(path, "model.pth")) or os.path.isfile(
        os.path.join(path, "model.safetensors")
    )
    return has_config and has_weights


# ── 텍스트 전처리 ─────────────────────────────────────────────────────────────


def preprocess_text(text: str) -> str:
    """마침표 주변 아티팩트를 줄이기 위해 ``.`` → ``!`` 치환.

    블로그 권장 처리. XTTS v2 한국어에서 ``.``로 문장이 끝나면 끝부분에
    잡음/끊김이 잦은 현상이 있다 → ``!``로 바꾸면 문장 종결 토큰이
    더 안정적으로 처리된다.
    """
    text = text.strip()
    text = text.replace(". ", "! ")
    text = text.replace(".", "!")
    return text


# ── 어댑터 ────────────────────────────────────────────────────────────────────


class XTTSAdapter:
    """XTTS v2를 TTSEvaluator에 끼울 수 있는 어댑터 (저수준 API).

    TTSEvaluator는 ``clone(reference_audio)`` 메서드를 가진 객체를 받는다.
    ``reference_audio``를 화자 레퍼런스로 사용해 ``clone_text``를 합성한다.

    Args:
        model_dir: XTTS v2 모델 폴더 경로. None이면 coqui-tts 캐시 자동 탐색.
        language: 합성 언어 코드. 기본 'ko'.
        clone_text: 클로닝 시 합성할 텍스트. 기본 한국어 문장.
        device: 'cpu' 또는 'cuda'. None이면 자동 감지.
        temperature: 디코딩 temperature. 기본 0.65 (낮을수록 결정적).
        repetition_penalty: 토큰 반복 penalty. 기본 10.0 (높을수록 반복 억제).
        top_k: top-k 샘플링. 기본 50.
        top_p: nucleus 샘플링 확률 컷오프. 기본 0.85.
        enable_text_splitting: 긴 문장을 청크 단위로 합성할지. 기본 True.
        download_if_missing: 캐시에 모델 없을 때 HuggingFace에서 자동 다운로드 시도.
            기본 True. False면 FileNotFoundError.

    Example:
        xtts = XTTSAdapter()
        tts_eval = TTSEvaluator(xtts_model=xtts, speaker_model=wavlm)
    """

    def __init__(
        self,
        model_dir: str | None = None,
        language: str = _DEFAULT_LANGUAGE,
        clone_text: str = _DEFAULT_CLONE_TEXT,
        device: str | None = None,
        temperature: float = _DEFAULT_TEMPERATURE,
        repetition_penalty: float = _DEFAULT_REPETITION_PENALTY,
        top_k: int = _DEFAULT_TOP_K,
        top_p: float = _DEFAULT_TOP_P,
        enable_text_splitting: bool = True,
        download_if_missing: bool = True,
    ) -> None:
        # 저수준 API import — TTS.api.TTS는 의도적으로 쓰지 않는다
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        _ensure_ffmpeg_path()

        self.language = language
        self.clone_text = clone_text
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_k = top_k
        self.top_p = top_p
        self.enable_text_splitting = enable_text_splitting
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 폴더 결정
        resolved_dir = model_dir or _find_xtts_model_dir()
        if resolved_dir is None and download_if_missing:
            resolved_dir = _download_xtts_to_cache()
        if resolved_dir is None:
            raise FileNotFoundError(
                "XTTS v2 모델 폴더를 찾을 수 없습니다.\n"
                "다음 중 하나를 시도하세요:\n"
                "  1) XTTSAdapter(model_dir='<직접 다운로드한 폴더>') 로 경로 지정\n"
                "  2) coqui-tts CLI로 모델 캐시:  tts --model_name "
                f"{_XTTS_FOLDER_NAME.replace('--', '/')} --text 'test' --out_path /tmp/x.wav\n"
                "  3) download_if_missing=True 로 자동 다운로드 (huggingface-hub 필요)"
            )
        if not _looks_like_xtts_dir(resolved_dir):
            raise FileNotFoundError(
                f"XTTS v2 모델 폴더가 유효하지 않습니다 (config.json 또는 model.pth 누락): "
                f"{resolved_dir}"
            )

        self.model_dir = resolved_dir
        logger.info("Loading XTTS v2 from %s on %s ...", self.model_dir, self.device)

        config = XttsConfig()
        config.load_json(os.path.join(self.model_dir, "config.json"))

        self._model = Xtts.init_from_config(config)
        self._model.load_checkpoint(config, checkpoint_dir=self.model_dir, eval=True)
        self._model.to(self.device)

        logger.info("XTTSAdapter ready (temperature=%.2f, repetition_penalty=%.1f).",
                    self.temperature, self.repetition_penalty)

    def clone(self, reference_audio: AudioArray) -> AudioArray:
        """reference_audio 화자 목소리로 clone_text를 합성한다.

        Args:
            reference_audio: shape (num_samples,), float32, [-1, 1], 16kHz mono

        Returns:
            합성된 음성, shape (num_samples,), float32, [-1, 1], 16kHz mono
            (XTTS v2의 24kHz 출력을 16kHz로 리샘플링한 결과)
        """
        if reference_audio.ndim != 1:
            raise ValueError(f"reference_audio must be 1-D, got shape {reference_audio.shape}")
        if reference_audio.dtype != np.float32:
            reference_audio = reference_audio.astype(np.float32)

        processed_text = preprocess_text(self.clone_text)

        # reference_audio → 임시 WAV (XTTS는 파일 경로를 받는다)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_ref:
            ref_path = tmp_ref.name
        try:
            _write_wav(ref_path, reference_audio, SAMPLE_RATE)

            with torch.no_grad():
                gpt_cond_latent, speaker_embedding = self._model.get_conditioning_latents(
                    audio_path=[ref_path]
                )
                out = self._model.inference(
                    processed_text,
                    self.language,
                    gpt_cond_latent,
                    speaker_embedding,
                    temperature=self.temperature,
                    repetition_penalty=self.repetition_penalty,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    enable_text_splitting=self.enable_text_splitting,
                )
        finally:
            try:
                os.unlink(ref_path)
            except OSError:
                pass

        cloned = np.asarray(out["wav"], dtype=np.float32)

        # XTTS 출력은 24kHz → 16kHz 리샘플링
        if _XTTS_OUTPUT_SR != SAMPLE_RATE:
            cloned = _resample_to_16k(cloned, _XTTS_OUTPUT_SR)

        # 안전 clip ([-1, 1] 범위 보장)
        return np.clip(cloned, -1.0, 1.0).astype(np.float32)


# ── 다운로드 헬퍼 ─────────────────────────────────────────────────────────────


def _download_xtts_to_cache() -> str | None:
    """huggingface-hub로 XTTS v2를 coqui-tts 캐시 위치에 받는다.

    실패 시 None 반환. 의존성(huggingface-hub) 없거나 네트워크 실패 등.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.warning("huggingface-hub 미설치 — XTTS 자동 다운로드 불가")
        return None

    cache_root = _candidate_cache_roots()[0]
    target_dir = os.path.join(cache_root, _XTTS_FOLDER_NAME)
    os.makedirs(target_dir, exist_ok=True)
    try:
        logger.info("Downloading XTTS v2 from HuggingFace to %s ...", target_dir)
        snapshot_download(
            repo_id="coqui/XTTS-v2",
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )
    except Exception as exc:  # 네트워크/권한/디스크 모두 포괄
        logger.error("XTTS 자동 다운로드 실패: %s", exc)
        return None
    return target_dir if _looks_like_xtts_dir(target_dir) else None


# ── WAV I/O / 리샘플링 ────────────────────────────────────────────────────────


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


def _resample_to_16k(audio: np.ndarray, src_sr: int) -> np.ndarray:
    """src_sr → 16kHz 리샘플링 (scipy.signal.resample_poly)."""
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(SAMPLE_RATE, src_sr)
    return resample_poly(audio, SAMPLE_RATE // g, src_sr // g).astype(np.float32)
