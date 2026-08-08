"""
음성 파일 1개에 실제로 노이즈를 입혀주는 실사용 스크립트

학습해둔 모델을 불러와서, 입력한 wav 파일 하나를 "보호된" wav로 바꿔준다.
모델에 한 번만 통과시키면 끝이라(~0.01초) 실시간으로도 쓸 수 있을 만큼 빠르다.

--no_psycho 옵션을 안 주면, 노이즈를 만든 뒤 한 번 더 "사람 귀에 안 들리는
한계선"을 넘는 부분을 깎아내는 후처리(hard_project)를 추가로 거친다.
이 후처리를 켜고 끈 상태의 SNR/L-inf 수치를 비교해서 출력해준다.

실행 방법:
  python infer.py --checkpoint checkpoints/best.pt --input voice.wav --output protected.wav
  python infer.py --checkpoint checkpoints/best.pt --input voice.wav  (output 자동 생성)
  python infer.py --checkpoint checkpoints/best.pt --input voice.wav --no_psycho  (hard_project 비활성화)
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PGD_DIR  = os.path.dirname(_THIS_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _PGD_DIR not in sys.path:
    sys.path.insert(0, _PGD_DIR)

from dataset import MAX_SAMPLES, SAMPLE_RATE
from features import MelExtractor
from model import PerturbationGenerator, EPS_DEFAULT
from psycho import hard_project, compute_threshold


# -- 인자 파싱 -------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="학습된 generator 체크포인트 경로 (.pt)")
    p.add_argument("--input",  type=str, required=True,
                   help="보호할 입력 wav 경로")
    p.add_argument("--output", type=str, default=None,
                   help="출력 wav 경로 (None -> input_protected.wav)")
    p.add_argument("--no_psycho", action="store_true",
                   help="심리음향 hard_project 후처리 비활성화 (2026-06-22 이전 동작)")
    return p.parse_args()


# -- 핵심 추론 함수 --------------------------------------------

def load_generator(checkpoint_path: str, device: torch.device) -> PerturbationGenerator:
    """
    체크포인트 파일에서 학습된 Generator를 불러온다.
    train.py로 저장한 것, model.py의 save()로 저장한 것 둘 다 읽을 수 있다.
    실제 사용할 때는 입력 길이가 매번 다르므로, 길이를 고정하지 않도록 설정한다.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    eps  = ckpt.get("eps", EPS_DEFAULT)
    gen  = PerturbationGenerator(eps=eps).to(device)
    state_key = "generator" if "generator" in ckpt else "state_dict"
    gen.load_state_dict(ckpt[state_key])
    gen.projector.t_out = None  # 추론 시 가변 길이
    gen.eval()
    return gen


def load_wav(path: str) -> tuple:
    """wav 파일을 읽어서 16kHz 모노로 바꾼 데이터와 원본 샘플레이트를 함께 돌려준다."""
    from math import gcd
    from scipy.signal import resample_poly

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = data.mean(axis=1).astype(np.float32)
    original_sr = sr

    if sr != SAMPLE_RATE:
        g = gcd(SAMPLE_RATE, sr)
        audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)

    audio = np.clip(audio, -1.0, 1.0)
    return audio, original_sr


def _psycho_exceed(delta_np: np.ndarray, threshold: np.ndarray) -> float:
    """만든 노이즈가 "안 들리는 한계선"을 평균적으로 얼마나 넘는지 계산한다. 0에 가까울수록 좋다."""
    import scipy.signal
    window = np.hanning(400).astype(np.float32)
    _, _, stft = scipy.signal.stft(
        delta_np, fs=SAMPLE_RATE, window=window,
        nperseg=400, noverlap=400 - 160, nfft=512, padded=True,
    )
    mag = np.abs(stft)
    n_frames = min(mag.shape[1], threshold.shape[1])
    exceed = np.maximum(mag[:, :n_frames] - threshold[:, :n_frames], 0.0)
    return float(exceed.mean())


def protect_wav(
    audio_np: np.ndarray,
    generator: PerturbationGenerator,
    mel_extractor: MelExtractor,
    device: torch.device,
    use_psycho: bool = True,
) -> tuple:
    """
    길이 상관없이 wav 하나를 통째로 보호한다 (여러 조각으로 안 잘라서
    이어붙일 때 생기는 경계 잡음이 없다).

    학습은 4초짜리로만 했기 때문에, 입력이 30초를 넘으면 경고를 띄운다.
    use_psycho를 켜두면(기본값) hard_project 후처리도 같이 적용하고,
    적용 전/후 지표(L-inf, SNR, psycho 초과량)를 함께 돌려준다.
    """
    total_len = len(audio_np)

    if total_len > SAMPLE_RATE * 30:
        print(f"[infer] 경고: 입력이 {total_len/SAMPLE_RATE:.1f}s로 30초를 초과합니다. "
              f"모델은 4초 기준으로 훈련되었으며 10초 이내 클립을 권장합니다.")

    with torch.no_grad():
        audio_t = torch.from_numpy(audio_np).unsqueeze(0).to(device)  # (1, T)
        mel     = mel_extractor(audio_t)                               # (1,1,80,T')
        delta   = generator(mel)                                       # (1, T'')

        # delta 길이와 원본 길이 맞춤 (ConvTranspose1d 출력이 근사값이므로)
        T = total_len
        T_delta = delta.shape[1]
        if T_delta > T:
            delta = delta[:, :T]
        elif T_delta < T:
            delta = F.pad(delta, [0, T - T_delta])

    delta_np = delta.squeeze(0).cpu().numpy()  # (T,)

    # hard_project 전 지표
    sig_pow = float(np.mean(audio_np ** 2)) + 1e-12
    def _snr(d):
        return 10.0 * np.log10(sig_pow / (float(np.mean(d ** 2)) + 1e-12))

    threshold = compute_threshold(audio_np) if use_psycho else None
    linf_before  = float(np.abs(delta_np).max())
    snr_before   = _snr(delta_np)
    psycho_before = _psycho_exceed(delta_np, threshold) if threshold is not None else None

    # 심리음향 hard_project
    if use_psycho:
        delta_np = hard_project(delta_np, threshold)
        delta_np = np.clip(delta_np, -generator.eps, generator.eps)  # iSTFT 오차 보정

    linf_after  = float(np.abs(delta_np).max())
    snr_after   = _snr(delta_np)
    psycho_after = _psycho_exceed(delta_np, threshold) if threshold is not None else None

    adv_np = np.clip(audio_np + delta_np, -1.0, 1.0)

    metrics = {
        "linf_before":    linf_before,
        "linf_after":     linf_after,
        "snr_before":     snr_before,
        "snr_after":      snr_after,
        "psycho_before":  psycho_before,
        "psycho_after":   psycho_after,
        "psycho_applied": use_psycho,
    }
    return adv_np, metrics


# -- 메인 -----------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_psycho = not args.no_psycho

    # 출력 경로 자동 생성
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = base + "_protected" + (ext if ext else ".wav")

    print(f"[infer] device        : {device}")
    print(f"[infer] input         : {args.input}")
    print(f"[infer] output        : {args.output}")
    print(f"[infer] checkpoint    : {args.checkpoint}")
    print(f"[infer] hard_project  : {'ON' if use_psycho else 'OFF (--no_psycho)'}")

    # 모델 로드
    generator     = load_generator(args.checkpoint, device)
    mel_extractor = MelExtractor().to(device).eval()
    print(f"[infer] Generator eps={generator.eps}")

    # 입력 오디오 로드
    audio_np, original_sr = load_wav(args.input)
    print(f"[infer] 입력 길이: {len(audio_np)} samples ({len(audio_np)/SAMPLE_RATE:.2f}s)")

    # Perturbation 적용
    t0 = time.time()
    protected_np, metrics = protect_wav(
        audio_np, generator, mel_extractor, device, use_psycho=use_psycho
    )
    elapsed = time.time() - t0
    print(f"[infer] 처리 시간: {elapsed*1000:.1f}ms")

    # 저장 (16kHz)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    sf.write(args.output, protected_np, SAMPLE_RATE)
    print(f"[infer] 저장 완료: {args.output}")

    # 지표 출력
    eps = generator.eps
    if use_psycho:
        print(f"[infer] -- hard_project 전 ----------------------")
        print(f"[infer]   L-inf  : {metrics['linf_before']:.6f}  (기준: <= {eps})")
        print(f"[infer]   SNR    : {metrics['snr_before']:.2f} dB  (기준: >= 28dB)")
        print(f"[infer]   psycho : {metrics['psycho_before']:.6f}  (threshold 초과 평균)")
        print(f"[infer] -- hard_project 후 ----------------------")
        print(f"[infer]   L-inf  : {metrics['linf_after']:.6f}  (기준: <= {eps})")
        print(f"[infer]   SNR    : {metrics['snr_after']:.2f} dB  (기준: >= 28dB)")
        print(f"[infer]   psycho : {metrics['psycho_after']:.6f}  (0에 가까울수록 좋음)")
        linf_delta = metrics['linf_before'] - metrics['linf_after']
        snr_delta  = metrics['snr_after']   - metrics['snr_before']
        print(f"[infer] -- 변화량 -------------------------------")
        print(f"[infer]   DeltaL-inf : {linf_delta:+.6f}  (음수 = delta 감소)")
        print(f"[infer]   DeltaSNR   : {snr_delta:+.2f} dB  (양수 = 음질 개선)")
    else:
        print(f"[infer] L-inf : {metrics['linf_after']:.6f}  (기준: <= {eps})")
        print(f"[infer] SNR   : {metrics['snr_after']:.2f} dB  (기준: >= 28dB)")


if __name__ == "__main__":
    main()
