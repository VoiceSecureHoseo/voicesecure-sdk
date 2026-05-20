"""CAM++ 화자 검증 어댑터.

SpeakerEvaluator (evaluators/speaker.py)의 cam_model 인자로 주입.
Wespeaker의 ONNX 모델을 사용해 의존성을 최소화한다.
모델 weights는 HuggingFace에서 다운로드 (repo에 포함하지 않음).
"""

from __future__ import annotations

import logging

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from voicesecure.types import SAMPLE_RATE, AudioArray, Embedding

logger = logging.getLogger(__name__)

_REPO_ID = "Wespeaker/wespeaker-voxceleb-campplus"
_ONNX_FILE = "voxceleb_CAM++.onnx"

# fbank 파라미터 (config.yaml 기준)
_NUM_MEL_BINS = 80
_FRAME_LENGTH_MS = 25
_FRAME_SHIFT_MS = 10


def _extract_fbank(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """음성에서 log-fbank 피처를 추출한다.

    CAM++ config.yaml 기준:
      - num_mel_bins: 80
      - frame_length: 25ms
      - frame_shift: 10ms
      - dither: 0 (추론 시 비활성)

    Returns:
        shape (T, 80), float32
    """
    frame_length = int(sample_rate * _FRAME_LENGTH_MS / 1000)  # 400
    frame_shift = int(sample_rate * _FRAME_SHIFT_MS / 1000)  # 160
    n_fft = frame_length

    # pre-emphasis
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1]).astype(np.float32)

    # framing
    n_frames = 1 + (len(audio) - frame_length) // frame_shift
    if n_frames <= 0:
        raise ValueError(f"audio too short for fbank: {len(audio)} samples")

    indices = np.arange(frame_length)[None, :] + np.arange(n_frames)[:, None] * frame_shift
    frames = audio[indices]  # (T, frame_length)

    # window
    window = np.hamming(frame_length).astype(np.float32)
    frames = frames * window

    # power spectrum
    power = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2  # (T, n_fft/2+1)

    # mel filterbank
    low_freq, high_freq = 20.0, sample_rate / 2.0
    low_mel = 2595.0 * np.log10(1.0 + low_freq / 700.0)
    high_mel = 2595.0 * np.log10(1.0 + high_freq / 700.0)
    mel_points = np.linspace(low_mel, high_mel, _NUM_MEL_BINS + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    fbank = np.zeros((power.shape[0], _NUM_MEL_BINS), dtype=np.float32)
    for m in range(1, _NUM_MEL_BINS + 1):
        lo, center, hi = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(lo, center):
            if center != lo:
                fbank[:, m - 1] += power[:, k] * (k - lo) / (center - lo)
        for k in range(center, hi):
            if hi != center:
                fbank[:, m - 1] += power[:, k] * (hi - k) / (hi - center)

    # log
    fbank = np.log(fbank + 1e-10)

    # CMVN (utterance-level mean normalization)
    fbank -= fbank.mean(axis=0, keepdims=True)

    return fbank.astype(np.float32)


class CAMPlusAdapter:
    """CAM++ ONNX 모델을 SpeakerEvaluator에 끼울 수 있는 어댑터.

    SpeakerEvaluator는 extract_embedding(audio) 메서드를 가진 객체를 받는다.
    onnxruntime으로 추론하므로 별도 CAM++ 패키지 불필요.

    Args:
        repo_id:  HuggingFace repo ID.
        onnx_file: repo 내 ONNX 파일명.
        device:   'cpu' 또는 'cuda'. None이면 자동 감지.

    Example:
        cam = CAMPlusAdapter()
        sv_eval = SpeakerEvaluator(wavlm_model=wavlm, cam_model=cam)
    """

    def __init__(
        self,
        repo_id: str = _REPO_ID,
        onnx_file: str = _ONNX_FILE,
        device: str | None = None,
    ) -> None:
        self.repo_id = repo_id

        # 디바이스 설정
        use_cuda = (device == "cuda") or (device is None and _cuda_available())
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_cuda
            else ["CPUExecutionProvider"]
        )

        logger.info("Downloading %s/%s ...", repo_id, onnx_file)
        onnx_path = hf_hub_download(repo_id, onnx_file)

        self._session = ort.InferenceSession(onnx_path, providers=providers)
        logger.info("CAMPlusAdapter ready (provider: %s).", self._session.get_providers()[0])

    def extract_embedding(self, audio: AudioArray) -> Embedding:
        """음성 → 화자 embedding.

        Args:
            audio: shape (num_samples,), float32, [-1, 1], 16kHz mono

        Returns:
            shape (512,), float32.
        """
        if audio.ndim != 1:
            raise ValueError(f"audio must be 1-D, got shape {audio.shape}")
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        feats = _extract_fbank(audio)  # (T, 80)
        feats = feats[None, :, :]  # (1, T, 80)

        outputs = self._session.run(["embs"], {"feats": feats})
        embedding = outputs[0].squeeze(0)  # (512,)
        return embedding.astype(np.float32)


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
