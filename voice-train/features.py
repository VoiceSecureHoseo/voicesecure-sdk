"""
소리(파형)를 신경망이 이해하기 좋은 형태로 바꿔주는 부분

신경망은 파형 숫자 나열을 그대로 넣으면 학습이 잘 안 된다. 그래서 소리를
주파수별로 얼마나 세게 울렸는지를 나타내는, 이미지 같은 형태로 바꿔서 넣어준다.

- mel_spectrogram: 우리 Generator 모델에 넣을 입력 만들기용
- kaldi_fbank: CAM++(화자 인식 AI)에 넣을 입력 만들기용. 얘는 학습 중에도
  역전파 신호가 끊기지 않고 잘 흘러가야 해서 별도로 만들었다.
"""

import torch
import torch.nn as nn
import torchaudio
import torchaudio.compliance.kaldi as kaldi

SAMPLE_RATE = 16_000
N_FFT       = 512
HOP_LENGTH  = 160
WIN_LENGTH  = 400
N_MELS      = 80


# -- Mel spectrogram (Generator 입력용) --------------------------

class MelExtractor(nn.Module):
    """
    소리 데이터를 mel spectrogram(주파수별 세기를 나타내는 이미지 형태)으로 바꿔준다.
    이렇게 바꾼 결과를 우리 Generator 모델의 입력으로 쓴다.
    """

    def __init__(self):
        super().__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
            power=2.0,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: (B, T)
        mel = self.transform(audio)           # (B, N_MELS, T')
        mel = torch.log1p(mel)                # log 압축
        mel = mel.unsqueeze(1)                # (B, 1, N_MELS, T')
        return mel


# -- kaldi fbank (CAM++ 입력용) ----------------------------------

def extract_kaldi_fbank(audio_t: torch.Tensor) -> torch.Tensor:
    """
    소리 데이터를 fbank(CAM++가 요구하는 입력 형식)로 바꿔준다.

    MelExtractor와 달리 이 결과는 학습 중 역전파(모델을 수정하기 위한 신호
    전달)에 계속 쓰이기 때문에, 그 흐름이 안 끊기는 방식으로 계산한다.
    같은 입력이면 항상 같은 결과가 나오도록 dither=0을 준다.
    """
    feat = kaldi.fbank(
        audio_t,
        num_mel_bins=N_MELS,
        dither=0,
        sample_frequency=SAMPLE_RATE,
    )  # (T', 80)
    feat = feat - feat.mean(dim=0, keepdim=True)  # mean normalization
    return feat


# -- 유틸 ----------------------------------------------------------

def get_mel_time_frames(n_samples: int) -> int:
    """
    소리 길이(n_samples)를 mel spectrogram으로 바꿨을 때 시간 축 길이가
    몇 칸이 되는지 미리 계산해준다. 모델 입출력 크기를 맞출 때 필요하다.
    """
    pad = N_FFT // 2  # center=True 기본값
    return (n_samples + 2 * pad - N_FFT) // HOP_LENGTH + 1
