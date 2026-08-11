from __future__ import annotations

import argparse
import time
from math import gcd
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from scipy.signal import resample_poly

from features import MelExtractor, SAMPLE_RATE


DEFAULT_ONNX_PATH = Path("checkpoints/generator.onnx")
UNET_DOWNSAMPLE_FACTOR = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VoiceSecure ONNX Generator WAV inference"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="입력 WAV 파일 경로",
    )

    parser.add_argument(
        "--output",
        default="output/protected_onnx.wav",
        help="보호 WAV 저장 경로",
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_ONNX_PATH),
        help="ONNX 모델 경로",
    )

    return parser.parse_args()


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """
    WAV 파일을 mono float32로 읽고 16kHz로 변환한다.

    Returns
    -------
    audio:
        shape=(T,), float32, 16kHz, [-1, 1]

    original_sr:
        입력 WAV의 원본 sample rate
    """
    if not path.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {path}")

    data, original_sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )

    # stereo 또는 다채널 음성을 mono로 변환
    audio = data.mean(axis=1).astype(np.float32)

    # 모델 입력 sample rate인 16kHz로 변환
    if original_sr != SAMPLE_RATE:
        common = gcd(SAMPLE_RATE, original_sr)

        audio = resample_poly(
            audio,
            SAMPLE_RATE // common,
            original_sr // common,
        ).astype(np.float32)

    audio = np.clip(audio, -1.0, 1.0)

    if audio.size == 0:
        raise ValueError("입력 WAV가 비어 있습니다.")

    if not np.all(np.isfinite(audio)):
        raise ValueError(
            "입력 WAV에 NaN 또는 Inf가 포함되어 있습니다."
        )

    return audio, original_sr


def extract_mel(audio: np.ndarray) -> np.ndarray:
    """
    waveform을 ONNX Generator 입력용 mel spectrogram으로 변환한다.

    Parameters
    ----------
    audio:
        shape=(T,)

    Returns
    -------
    mel:
        shape=(1, 1, 80, time_frames)
    """
    mel_extractor = MelExtractor().eval()

    audio_tensor = torch.from_numpy(
        audio
    ).unsqueeze(0)

    with torch.no_grad():
        mel_tensor = mel_extractor(audio_tensor)

    mel = (
        mel_tensor
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return mel


def align_mel_frames(
    mel: np.ndarray,
    factor: int = UNET_DOWNSAMPLE_FACTOR,
) -> np.ndarray:
    """
    ONNX U-Net skip connection 오류를 방지하기 위해
    mel time frame 길이를 factor의 배수로 맞춘다.

    예:
        723 frames -> 720 frames

    U-Net이 시간축을 3번 downsampling하므로
    2^3 = 8의 배수로 맞춘다.
    """
    if mel.ndim != 4:
        raise ValueError(
            "mel 입력은 4차원이어야 합니다. "
            f"현재 shape: {mel.shape}"
        )

    frames = mel.shape[-1]

    if frames < factor:
        raise ValueError(
            f"mel frame 수가 너무 작습니다: {frames}"
        )

    aligned_frames = (frames // factor) * factor

    if aligned_frames != frames:
        print(
            f"[Mel] frame alignment: "
            f"{frames} -> {aligned_frames}"
        )

        mel = mel[..., :aligned_frames]

    return np.ascontiguousarray(
        mel,
        dtype=np.float32,
    )


def create_onnx_session(
    model_path: Path,
) -> ort.InferenceSession:
    """
    ONNX Runtime CPU 세션을 생성한다.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"ONNX 모델이 없습니다: {model_path}"
        )

    external_data_path = Path(
        str(model_path) + ".data"
    )

    if not external_data_path.exists():
        raise FileNotFoundError(
            "ONNX 가중치 파일이 없습니다: "
            f"{external_data_path}"
        )

    session_options = ort.SessionOptions()

    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    return session


def match_delta_length(
    delta: np.ndarray,
    target_length: int,
) -> np.ndarray:
    """
    ONNX 출력 perturbation 길이를 원본 waveform 길이에 맞춘다.

    출력이 길면 자르고,
    출력이 짧으면 뒤쪽을 0으로 padding한다.
    """
    delta = np.asarray(
        delta,
        dtype=np.float32,
    ).reshape(-1)

    current_length = len(delta)

    if current_length > target_length:
        print(
            f"[Delta] crop length: "
            f"{current_length} -> {target_length}"
        )

        delta = delta[:target_length]

    elif current_length < target_length:
        print(
            f"[Delta] pad length: "
            f"{current_length} -> {target_length}"
        )

        delta = np.pad(
            delta,
            (0, target_length - current_length),
            mode="constant",
        )

    return delta.astype(np.float32)


def calculate_snr(
    original: np.ndarray,
    delta: np.ndarray,
) -> float:
    """
    원본 음성과 perturbation 사이의 SNR을 계산한다.
    """
    signal_power = (
        float(np.mean(original ** 2))
        + 1e-12
    )

    noise_power = (
        float(np.mean(delta ** 2))
        + 1e-12
    )

    return float(
        10.0
        * np.log10(
            signal_power / noise_power
        )
    )


def run_onnx_inference(
    session: ort.InferenceSession,
    mel: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    ONNX Runtime 추론을 실행한다.
    """
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    print(f"[ONNX] input name  : {input_info.name}")
    print(f"[ONNX] input shape : {input_info.shape}")
    print(f"[ONNX] output name : {output_info.name}")
    print(f"[ONNX] output shape: {output_info.shape}")
    print(f"[ONNX] actual input: {mel.shape}")

    start_time = time.perf_counter()

    outputs = session.run(
        [output_info.name],
        {
            input_info.name: mel,
        },
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    return outputs[0], elapsed


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(args.model)

    print("===================================")
    print("VoiceSecure ONNX WAV Inference")
    print("===================================")

    print(f"[Path] input : {input_path}")
    print(f"[Path] model : {model_path}")
    print(f"[Path] output: {output_path}")

    # 1. WAV 로드 및 16kHz 변환
    audio, original_sr = load_wav(
        input_path
    )

    duration = (
        len(audio) / SAMPLE_RATE
    )

    print("\n[Audio]")
    print(
        f"original sample rate : "
        f"{original_sr}"
    )
    print(
        f"inference sample rate: "
        f"{SAMPLE_RATE}"
    )
    print(
        f"samples              : "
        f"{len(audio)}"
    )
    print(
        f"duration             : "
        f"{duration:.3f} sec"
    )
    print(
        f"input min/max        : "
        f"{audio.min():.6f} / "
        f"{audio.max():.6f}"
    )

    # 2. Mel 추출
    mel_start = time.perf_counter()

    mel = extract_mel(audio)

    original_mel_frames = mel.shape[-1]

    # U-Net skip connection 오류 방지를 위해
    # mel frame 수를 8의 배수로 맞춘다.
    mel = align_mel_frames(mel)

    mel_elapsed = (
        time.perf_counter()
        - mel_start
    )

    print("\n[Mel]")
    print(
        f"original frames: "
        f"{original_mel_frames}"
    )
    print(
        f"aligned frames : "
        f"{mel.shape[-1]}"
    )
    print(
        f"mel shape      : "
        f"{mel.shape}"
    )
    print(
        f"mel dtype      : "
        f"{mel.dtype}"
    )
    print(
        f"mel time       : "
        f"{mel_elapsed:.3f} sec"
    )

    # 3. ONNX Runtime 세션 생성
    session_start = time.perf_counter()

    session = create_onnx_session(
        model_path
    )

    session_elapsed = (
        time.perf_counter()
        - session_start
    )

    print("\n[Session]")
    print(
        f"provider : "
        f"{session.get_providers()}"
    )
    print(
        f"load time: "
        f"{session_elapsed:.3f} sec"
    )

    # 4. ONNX 추론
    delta_raw, inference_elapsed = (
        run_onnx_inference(
            session,
            mel,
        )
    )

    # mel frame을 잘랐기 때문에 delta가
    # 원본 waveform보다 짧을 수 있다.
    delta = match_delta_length(
        delta_raw,
        target_length=len(audio),
    )

    # 5. 원본 음성에 perturbation 적용
    protected = np.clip(
        audio + delta,
        -1.0,
        1.0,
    ).astype(np.float32)

    # 6. 결과 지표 계산
    linf = float(
        np.max(np.abs(delta))
    )

    mean_abs_delta = float(
        np.mean(np.abs(delta))
    )

    snr = calculate_snr(
        audio,
        delta,
    )

    rtf = (
        inference_elapsed / duration
        if duration > 0
        else float("inf")
    )

    print("\n[Result]")
    print(
        f"raw delta shape    : "
        f"{delta_raw.shape}"
    )
    print(
        f"matched delta shape: "
        f"{delta.shape}"
    )
    print(
        f"delta min/max      : "
        f"{delta.min():.10f} / "
        f"{delta.max():.10f}"
    )
    print(
        f"L-inf              : "
        f"{linf:.10f}"
    )
    print(
        f"mean abs delta     : "
        f"{mean_abs_delta:.10f}"
    )
    print(
        f"SNR                : "
        f"{snr:.3f} dB"
    )
    print(
        f"ONNX inference     : "
        f"{inference_elapsed:.3f} sec"
    )
    print(
        f"RTF                : "
        f"{rtf:.4f}"
    )

    # 7. 보호 음성 저장
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        str(output_path),
        protected,
        SAMPLE_RATE,
        subtype="PCM_16",
    )

    print("\n===================================")
    print("ONNX WAV inference: SUCCESS")
    print(f"saved: {output_path}")
    print("===================================")


if __name__ == "__main__":
    main()