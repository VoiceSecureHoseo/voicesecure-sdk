from __future__ import annotations

import argparse
import subprocess
import time
from math import gcd
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sounddevice as sd
import soundfile as sf
import torch
from scipy.signal import resample_poly

from features import MelExtractor, SAMPLE_RATE


# =========================================================
# 기본 설정
# =========================================================

DEFAULT_MODEL_PATH = Path("checkpoints/generator.onnx")
DEFAULT_INPUT_PATH = Path("input/realtime_input.wav")
DEFAULT_OUTPUT_PATH = Path("output/realtime_protected_onnx.wav")

# Raspberry Pi HDMI 출력 장치
DEFAULT_PLAYBACK_DEVICE = "plughw:CARD=vc4hdmi1,DEV=0"

# U-Net이 시간축을 세 번 downsampling하므로 2^3
UNET_DOWNSAMPLE_FACTOR = 8


# =========================================================
# 명령행 옵션
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VoiceSecure Raspberry Pi ONNX realtime inference"
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="ONNX 모델 경로",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="녹음 원본 WAV 저장 경로",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="보호 음성 WAV 저장 경로",
    )

    parser.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="sounddevice 입력 장치 번호",
    )

    parser.add_argument(
        "--playback-device",
        type=str,
        default=DEFAULT_PLAYBACK_DEVICE,
        help="aplay 재생 장치",
    )

    parser.add_argument(
        "--no-play",
        action="store_true",
        help="추론 후 자동 재생하지 않음",
    )

    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="오디오 장치 목록 출력 후 종료",
    )

    return parser.parse_args()


# =========================================================
# 오디오 장치
# =========================================================

def print_audio_devices() -> None:
    print("===================================")
    print("Audio devices")
    print("===================================")
    print(sd.query_devices())


def get_input_sample_rate(
    device: int | None,
) -> int:
    """
    선택된 마이크의 기본 sample rate를 확인한다.
    """
    try:
        device_info = sd.query_devices(
            device=device,
            kind="input",
        )

        sample_rate = int(
            round(device_info["default_samplerate"])
        )

        if sample_rate <= 0:
            raise ValueError(
                f"잘못된 sample rate: {sample_rate}"
            )

        return sample_rate

    except Exception as exc:
        raise RuntimeError(
            "마이크 sample rate를 확인하지 못했습니다. "
            "--list-devices로 장치를 확인하세요."
        ) from exc


# =========================================================
# 녹음 및 리샘플링
# =========================================================

def record_audio(
    device: int | None,
) -> tuple[np.ndarray, int]:
    """
    Enter를 누르면 녹음을 시작하고,
    다시 Enter를 누르면 녹음을 종료한다.
    """
    native_sr = get_input_sample_rate(device)

    recorded_chunks: list[np.ndarray] = []

    def audio_callback(
        indata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            print(f"\n[Audio status] {status}")

        recorded_chunks.append(
            indata.copy()
        )

    print("\n[Recording]")
    print(f"device       : {device}")
    print(f"sample rate  : {native_sr}")

    input("Press ENTER to start recording...")

    print("===================================")
    print("🔴 Recording...")
    print("Press ENTER again to stop.")
    print("===================================")

    try:
        with sd.InputStream(
            samplerate=native_sr,
            channels=1,
            dtype="float32",
            device=device,
            callback=audio_callback,
        ):
            input()

    except KeyboardInterrupt:
        print("\nRecording interrupted.")

    print("Recording complete.")

    if not recorded_chunks:
        raise RuntimeError(
            "녹음된 데이터가 없습니다."
        )

    recording = np.concatenate(
        recorded_chunks,
        axis=0,
    )

    audio = np.asarray(
        recording,
        dtype=np.float32,
    ).reshape(-1)

    if audio.size == 0:
        raise RuntimeError(
            "녹음된 데이터가 없습니다."
        )

    if not np.all(np.isfinite(audio)):
        raise RuntimeError(
            "녹음 데이터에 NaN 또는 Inf가 있습니다."
        )

    audio = np.clip(
        audio,
        -1.0,
        1.0,
    )

    duration = len(audio) / native_sr

    print(f"duration     : {duration:.3f} sec")
    print(f"samples      : {len(audio)}")

    return audio, native_sr


def resample_audio(
    audio: np.ndarray,
    original_sr: int,
    target_sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    마이크 녹음 데이터를 모델 입력 sample rate로 변환한다.
    """
    if original_sr == target_sr:
        return audio.astype(np.float32)

    common = gcd(
        original_sr,
        target_sr,
    )

    up = target_sr // common
    down = original_sr // common

    print("\n[Resample]")
    print(f"{original_sr} Hz -> {target_sr} Hz")
    print(f"ratio: up={up}, down={down}")

    resampled = resample_poly(
        audio,
        up,
        down,
    ).astype(np.float32)

    return np.clip(
        resampled,
        -1.0,
        1.0,
    )


# =========================================================
# Mel 추출
# =========================================================

def extract_mel(
    audio: np.ndarray,
    extractor: MelExtractor,
) -> np.ndarray:
    """
    waveform을 Generator 입력 mel spectrogram으로 변환한다.

    반환 shape:
        (1, 1, 80, time_frames)
    """
    waveform = torch.from_numpy(
        audio
    ).unsqueeze(0)

    with torch.no_grad():
        mel_tensor = extractor(waveform)

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
    ONNX U-Net skip connection의 크기 불일치를 막기 위해
    mel frame 수를 8의 배수로 맞춘다.

    예:
        501 -> 496
        723 -> 720
    """
    if mel.ndim != 4:
        raise ValueError(
            "mel 입력은 4차원이어야 합니다. "
            f"현재 shape={mel.shape}"
        )

    frames = int(mel.shape[-1])

    if frames < factor:
        raise ValueError(
            f"mel frame 수가 너무 작습니다: {frames}"
        )

    aligned_frames = (
        frames // factor
    ) * factor

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


# =========================================================
# ONNX Runtime
# =========================================================

def create_onnx_session(
    model_path: Path,
) -> ort.InferenceSession:
    """
    ONNX Runtime CPU 추론 세션을 생성한다.
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
            "ONNX 외부 가중치 파일이 없습니다: "
            f"{external_data_path}"
        )

    options = ort.SessionOptions()

    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    return session


def warmup_onnx_session(
    session: ort.InferenceSession,
) -> None:
    """
    첫 추론 지연을 줄이기 위한 워밍업.
    """
    input_name = session.get_inputs()[0].name

    dummy_mel = np.zeros(
        (1, 1, 80, 400),
        dtype=np.float32,
    )

    print("\n[ONNX]")
    print("Warming up ONNX Runtime...")

    session.run(
        None,
        {
            input_name: dummy_mel,
        },
    )

    print("Warm-up complete.")


def run_onnx_inference(
    session: ort.InferenceSession,
    mel: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    ONNX Generator로 perturbation을 생성한다.
    """
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    print("\n[ONNX inference]")
    print(f"input name   : {input_info.name}")
    print(f"input shape  : {mel.shape}")
    print(f"output name  : {output_info.name}")

    start = time.perf_counter()

    outputs = session.run(
        [output_info.name],
        {
            input_info.name: mel,
        },
    )

    elapsed = time.perf_counter() - start

    delta = np.asarray(
        outputs[0],
        dtype=np.float32,
    )

    return delta, elapsed


# =========================================================
# Perturbation 후처리
# =========================================================

def match_delta_length(
    delta: np.ndarray,
    target_length: int,
) -> np.ndarray:
    """
    Generator 출력 길이를 원본 음성 길이에 맞춘다.
    """
    delta = np.asarray(
        delta,
        dtype=np.float32,
    ).reshape(-1)

    current_length = len(delta)

    if current_length > target_length:
        print(
            f"[Delta] crop: "
            f"{current_length} -> {target_length}"
        )

        delta = delta[:target_length]

    elif current_length < target_length:
        print(
            f"[Delta] pad: "
            f"{current_length} -> {target_length}"
        )

        delta = np.pad(
            delta,
            (
                0,
                target_length - current_length,
            ),
            mode="constant",
        )

    return delta.astype(np.float32)


def calculate_snr(
    original: np.ndarray,
    delta: np.ndarray,
) -> float:
    """
    원본 신호와 perturbation 사이의 SNR을 계산한다.
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
        10.0 * np.log10(
            signal_power / noise_power
        )
    )


# =========================================================
# 저장 및 재생
# =========================================================

def save_wav(
    path: Path,
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        str(path),
        audio,
        sample_rate,
        subtype="PCM_16",
    )


def play_wav(
    path: Path,
    playback_device: str,
) -> None:
    """
    Raspberry Pi HDMI 출력으로 WAV 파일을 재생한다.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"재생 파일이 없습니다: {path}"
        )

    command = [
        "aplay",
        "-D",
        playback_device,
        str(path),
    ]

    print("\n[Playback]")
    print(" ".join(command))

    try:
        subprocess.run(
            command,
            check=True,
        )

    except FileNotFoundError:
        print(
            "경고: aplay 명령을 찾을 수 없습니다."
        )

    except subprocess.CalledProcessError as exc:
        print(
            "경고: HDMI 재생 중 오류가 발생했습니다."
        )
        print(exc)


# =========================================================
# 메인 실행
# =========================================================

def main() -> None:
    args = parse_args()

    if args.list_devices:
        print_audio_devices()
        return

    print("===================================")
    print("VoiceSecure Raspberry Pi ONNX")
    print("===================================")

    print(f"model : {args.model}")
    print(f"input : {args.input}")
    print(f"output: {args.output}")

    total_start = time.perf_counter()

    # -----------------------------------------------------
    # 1. MelExtractor 준비
    # -----------------------------------------------------

    mel_extractor = MelExtractor().eval()

    # -----------------------------------------------------
    # 2. ONNX 세션 생성
    # -----------------------------------------------------

    session_start = time.perf_counter()

    session = create_onnx_session(
        args.model
    )

    session_load_time = (
        time.perf_counter()
        - session_start
    )

    print("\n[Session]")
    print(
        f"providers: "
        f"{session.get_providers()}"
    )
    print(
        f"load time: "
        f"{session_load_time:.3f} sec"
    )

    # 첫 실행 지연을 제외하기 위한 워밍업
    warmup_onnx_session(session)

    # -----------------------------------------------------
    # 3. 마이크 녹음
    # -----------------------------------------------------

    recorded_audio, native_sr = record_audio(
        device=args.input_device,
    )

    # -----------------------------------------------------
    # 4. 16kHz 변환
    # -----------------------------------------------------

    audio = resample_audio(
        recorded_audio,
        original_sr=native_sr,
        target_sr=SAMPLE_RATE,
    )

    duration = len(audio) / SAMPLE_RATE

    print("\n[Audio]")
    print(f"native sample rate: {native_sr}")
    print(f"model sample rate : {SAMPLE_RATE}")
    print(f"samples           : {len(audio)}")
    print(f"duration          : {duration:.3f} sec")
    print(
        f"min/max           : "
        f"{audio.min():.6f} / "
        f"{audio.max():.6f}"
    )

    # 녹음 원본 저장
    save_wav(
        args.input,
        audio,
        SAMPLE_RATE,
    )

    print(
        f"original saved    : {args.input}"
    )

    # -----------------------------------------------------
    # 5. Mel 추출
    # -----------------------------------------------------

    mel_start = time.perf_counter()

    mel = extract_mel(
        audio,
        mel_extractor,
    )

    original_frames = int(
        mel.shape[-1]
    )

    mel = align_mel_frames(mel)

    mel_time = (
        time.perf_counter()
        - mel_start
    )

    print("\n[Mel]")
    print(f"original frames: {original_frames}")
    print(f"aligned frames : {mel.shape[-1]}")
    print(f"shape          : {mel.shape}")
    print(f"extraction time: {mel_time:.3f} sec")

    # -----------------------------------------------------
    # 6. ONNX Generator 추론
    # -----------------------------------------------------

    delta_raw, inference_time = (
        run_onnx_inference(
            session,
            mel,
        )
    )

    delta = match_delta_length(
        delta_raw,
        target_length=len(audio),
    )

    # -----------------------------------------------------
    # 7. 보호 음성 생성
    # -----------------------------------------------------

    protected_audio = np.clip(
        audio + delta,
        -1.0,
        1.0,
    ).astype(np.float32)

    save_wav(
        args.output,
        protected_audio,
        SAMPLE_RATE,
    )

    # -----------------------------------------------------
    # 8. 결과 출력
    # -----------------------------------------------------

    total_time = (
        time.perf_counter()
        - total_start
    )

    rtf = (
        inference_time / duration
        if duration > 0
        else float("inf")
    )

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

    print("\n===================================")
    print("Inference result")
    print("===================================")

    print(
        f"raw delta shape  : "
        f"{delta_raw.shape}"
    )
    print(
        f"final delta shape: "
        f"{delta.shape}"
    )
    print(
        f"delta min/max    : "
        f"{delta.min():.10f} / "
        f"{delta.max():.10f}"
    )
    print(
        f"L-inf            : "
        f"{linf:.10f}"
    )
    print(
        f"mean abs delta   : "
        f"{mean_abs_delta:.10f}"
    )
    print(
        f"SNR              : "
        f"{snr:.3f} dB"
    )
    print(
        f"ONNX inference   : "
        f"{inference_time:.3f} sec"
    )
    print(
        f"RTF              : "
        f"{rtf:.4f}"
    )
    print(
        f"total elapsed    : "
        f"{total_time:.3f} sec"
    )
    print(
        f"protected saved  : "
        f"{args.output}"
    )

    print("===================================")
    print("VoiceSecure ONNX: SUCCESS")
    print("===================================")

    # -----------------------------------------------------
    # 9. HDMI 자동 재생
    # -----------------------------------------------------

    if not args.no_play:
        play_wav(
            args.output,
            args.playback_device,
        )


if __name__ == "__main__":
    main()