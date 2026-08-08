"""
학습에 쓸 음성 파일들을 불러오고 살짝씩 변형해주는 부분

KSS(한국어 음성 데이터셋) 폴더에서 wav 파일들을 찾아서 읽어온다.
매번 똑같은 데이터만 보면 모델이 과적합(외워버리기)되기 쉬워서,
속도를 살짝 바꾸거나 잡음을 아주 조금 섞거나 볼륨을 바꾸는 식으로
매번 조금씩 다른 버전을 만들어 학습에 쓴다 (data augmentation).
"""

import os
import random
from math import gcd

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

SAMPLE_RATE = 16_000
MAX_SAMPLES = SAMPLE_RATE * 4   # 4초 고정 길이


# -- wav 전처리 유틸 ---------------------------------------------

def _load_wav(path: str) -> np.ndarray:
    """wav 파일을 읽어서 16kHz 모노 소리 데이터로 바꿔준다."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = data.mean(axis=1).astype(np.float32)
    if sr != SAMPLE_RATE:
        try:
            from scipy.signal import resample_poly
            g = gcd(SAMPLE_RATE, sr)
            audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)
        except ImportError:
            import torchaudio
            t = torch.from_numpy(audio).unsqueeze(0)
            t = torchaudio.functional.resample(t, sr, SAMPLE_RATE)
            audio = t.squeeze(0).numpy()
    return np.clip(audio, -1.0, 1.0)


def _pad_or_trim(audio: np.ndarray, length: int) -> np.ndarray:
    """길이를 정확히 length로 맞춘다. 길면 랜덤한 위치에서 자르고, 짧으면 뒤에 0을 채운다."""
    if len(audio) >= length:
        start = random.randint(0, len(audio) - length)
        return audio[start:start + length].copy()
    pad = length - len(audio)
    return np.pad(audio, (0, pad), mode="constant")


def _time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """
    소리의 재생 속도를 바꾼다. rate가 1보다 크면 빨라지고(길이는 짧아짐),
    1보다 작으면 느려진다(길이는 길어짐). 실패하면 원본을 그대로 돌려준다.
    """
    try:
        from scipy.signal import resample_poly
        factor_num = int(round(rate * 100))
        factor_den = 100
        g = gcd(factor_num, factor_den)
        stretched = resample_poly(audio, factor_den // g, factor_num // g)
        stretched = stretched.astype(np.float32)
        return stretched  # 길이는 이후 _pad_or_trim이 맞춤
    except Exception:
        return audio


def _add_gaussian_noise(audio: np.ndarray, snr_db: float = 40.0) -> np.ndarray:
    """아주 약한 잡음을 섞어준다. 기본값(40dB)이면 사람 귀엔 거의 안 들리는 수준."""
    signal_power = np.mean(audio ** 2) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(audio)).astype(np.float32) * np.sqrt(noise_power)
    return np.clip(audio + noise, -1.0, 1.0)


def _augment(audio: np.ndarray) -> np.ndarray:
    """
    아래 세 가지 변형 중 1~2개를 무작위로 골라 적용한다.
    - 속도 변화: 0.9~1.1배
    - 잡음 추가: SNR 35~45dB
    - 볼륨 변화: 0.8~1.0배
    """
    ops = random.sample(["stretch", "noise", "amplitude"], k=random.randint(1, 2))
    for op in ops:
        if op == "stretch":
            rate = random.uniform(0.9, 1.1)
            audio = _time_stretch(audio, rate)
        elif op == "noise":
            snr = random.uniform(35.0, 45.0)
            audio = _add_gaussian_noise(audio, snr)
        elif op == "amplitude":
            scale = random.uniform(0.8, 1.0)
            audio = (audio * scale).astype(np.float32)
    return audio


# -- 경로 수집 / Dataset 구현 --------------------------------

def _collect_wav_paths(root: str) -> list:
    """
    root 폴더와 그 하위 폴더를 전부 뒤져서 wav 파일 경로를 모은다.
    KSS는 kss/1, kss/2, kss/3, kss/4 폴더에 나뉘어 있는데 전부 찾아준다.
    파일명에 clone/adv/pgd가 들어간 건(다른 작업 결과물이라) 걸러낸다.
    """
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if (f.endswith(".wav")
                    and not f.startswith(".")
                    and "clone" not in f
                    and "adv" not in f
                    and "pgd" not in f):
                paths.append(os.path.normpath(os.path.join(dirpath, f)))
    return sorted(paths)


class AudioDataset(Dataset):
    """
    wav 파일들을 학습에 쓸 수 있는 형태로 하나씩 꺼내주는 클래스.

    wav_dir에 KSS 루트 폴더(1~4 서브폴더 자동 탐색)나 단일 폴더를 넣으면 되고,
    augment를 켜면 매번 살짝 다르게 변형해서 꺼내준다. max_samples는 오디오
    길이를 고정하는 샘플 수(기본 4초 분량).
    """

    def __init__(
        self,
        wav_dir: str,
        augment: bool = True,
        max_samples: int = MAX_SAMPLES,
    ):
        self.wav_dir = wav_dir
        self.augment = augment
        self.max_samples = max_samples

        self.paths = _collect_wav_paths(wav_dir)
        assert len(self.paths) > 0, f"WAV 파일 없음: {wav_dir}"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path  = self.paths[idx]
        audio = _load_wav(path)

        if self.augment:
            audio = _augment(audio)

        audio = _pad_or_trim(audio, self.max_samples)
        return torch.from_numpy(audio), path   # (MAX_SAMPLES,) float32, str

    @property
    def num_files(self):
        return len(self.paths)


def build_dataloader(
    wav_dir,
    batch_size: int = 8,
    augment: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    학습용 DataLoader를 만든다. wav_dir에 폴더 하나를 문자열로 줘도 되고,
    여러 폴더를 리스트로 줘서 한꺼번에 합쳐 쓸 수도 있다.
    """
    from torch.utils.data import ConcatDataset
    if isinstance(wav_dir, (list, tuple)):
        datasets = [AudioDataset(d, augment=augment) for d in wav_dir]
        dataset = ConcatDataset(datasets)
    else:
        dataset = AudioDataset(wav_dir, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
