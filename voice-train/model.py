"""
목소리에 덧씌울 노이즈(perturbation)를 만들어내는 신경망, U-Net 구조를 쓴다

음성을 mel spectrogram(소리를 이미지처럼 표현한 것)으로 바꿔서 넣으면,
그 소리에 살짝 더해줄 노이즈를 만들어서 뱉어준다. 이 노이즈는 사람 귀에는
거의 안 들리지만 화자 인식 AI는 헷갈리게 만드는 게 목표.

U-Net 구조를 쓰는 이유는, 입력을 작게 압축했다가 다시 원래 크기로
복원하면서 압축 전 정보를 skip connection으로 같이 넘겨줘서 디테일을
살릴 수 있기 때문이다. 원래 이미지 분할 작업에서 자주 쓰는 구조다.

eps는 노이즈를 최대 얼마나 크게 만들지 정하는 한계값이다. 이 값이 크면
노이즈가 세져서 방어 효과는 좋아지지만 사람도 잡음을 느끼기 시작한다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import MAX_SAMPLES
from features import N_MELS, get_mel_time_frames

# Generator 출력 고정 길이
T_OUT = MAX_SAMPLES          # 64000
T_MEL = get_mel_time_frames(T_OUT)  # 401

EPS_DEFAULT = 0.01


# -- 기본 블록 -------------------------------------------------

class ConvBnRelu(nn.Module):
    """가장 기본이 되는 조합: 특징을 뽑고(Conv2d) 값을 정돈하고(BatchNorm2d) 음수는 0으로 자른다(ReLU)."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpConvBnRelu(nn.Module):
    """압축된 크기를 다시 키워주는 조합 (Decoder에서 원래 크기로 복원할 때 씀)."""
    def __init__(self, in_ch, out_ch, kernel=4, stride=2, padding=1, output_padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_ch, out_ch, kernel,
                stride=stride, padding=padding,
                output_padding=output_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# -- Encoder 블록 ----------------------------------------------

class EncoderBlock(nn.Module):
    """
    특징을 두 번 뽑은 뒤 크기를 절반으로 줄인다(다운샘플).
    줄이기 전 정보(skip)도 같이 돌려줘서, 나중에 Decoder가 디테일을 복원할 때 쓴다.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBnRelu(in_ch, out_ch)
        self.conv2 = ConvBnRelu(out_ch, out_ch)
        self.down  = nn.Conv2d(out_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.bn    = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x                           # skip connection 저장
        x = F.relu(self.bn(self.down(x)))  # 다운샘플
        return x, skip


# -- Decoder 블록 ----------------------------------------------

class DecoderBlock(nn.Module):
    """
    크기를 다시 키운 뒤(업샘플), Encoder에서 넘어온 skip 정보를 이어붙이고
    특징을 두 번 더 뽑는다. 크기가 살짝 안 맞으면 자동으로 잘라서 맞춘다.
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = UpConvBnRelu(in_ch, out_ch)
        self.conv = nn.Sequential(
            ConvBnRelu(out_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)

        # skip과 크기 불일치 보정 (H, W 축 독립적으로 처리)
        # x가 skip보다 작으면 pad, 크면 crop - 각 축 독립적으로
        dh = skip.shape[2] - x.shape[2]   # 양수: x가 작음, 음수: x가 큼
        dw = skip.shape[3] - x.shape[3]

        # H축 보정
        if dh > 0:
            x = F.pad(x, [0, 0, dh // 2, dh - dh // 2])
        elif dh < 0:
            x = x[:, :, :skip.shape[2], :]

        # W축 보정
        if dw > 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, 0, 0])
        elif dw < 0:
            x = x[:, :, :, :skip.shape[3]]

        x = torch.cat([x, skip], dim=1)   # (B, out_ch + skip_ch, H, W)
        return self.conv(x)


# -- Bottleneck ------------------------------------------------

class Bottleneck(nn.Module):
    """U자 모양의 가장 아래(가장 작게 압축된 지점)에서 특징을 한 번 더 다듬는다."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(in_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


# -- Time Domain Projection ------------------------------------

class TimeDomainProjector(nn.Module):
    """
    이미지 형태(mel)로 처리하던 결과를, 실제로 소리에 더할 수 있는
    노이즈 파형(waveform)으로 변환하는 마지막 단계.

    형태를 펼치고 -> 길이를 원하는 만큼 늘리고 -> Tanh에 eps를 곱해서
    노이즈 크기가 반드시 [-eps, eps] 범위 안에 들어오도록 보장한다.
    """
    def __init__(self, in_ch, n_freq, t_out, hop_length=160, eps=EPS_DEFAULT):
        super().__init__()
        self.n_freq = n_freq
        self.t_out  = t_out
        self.eps    = eps

        flat_ch = in_ch * n_freq   # reshape 후 채널 수
        self._flat_ch = flat_ch    # forward에서 일치 검증용

        # T' -> T_OUT: stride=hop_length로 업샘플
        self.up1 = nn.ConvTranspose1d(
            flat_ch, 64,
            kernel_size=hop_length * 2,
            stride=hop_length,
            padding=hop_length // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(64)

        self.up2 = nn.ConvTranspose1d(
            64, 16,
            kernel_size=9,
            stride=1,
            padding=4,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(16)

        self.out = nn.Conv1d(16, 1, kernel_size=7, padding=3, bias=True)

    def forward(self, x):
        # x: (B, C, n_freq, T')
        B, C, n_freq, T = x.shape
        assert C * n_freq == self._flat_ch, (
            f"채널 불일치: C*n_freq={C*n_freq} != 초기화 시 flat_ch={self._flat_ch}"
        )
        x = x.reshape(B, C * n_freq, T)           # (B, C*n_freq, T')

        x = torch.relu(self.bn1(self.up1(x)))  # (B, 64, ~T')
        x = torch.relu(self.bn2(self.up2(x)))  # (B, 16, ~T')
        x = self.out(x)                        # (B, 1, ~T')
        x = x.squeeze(1)                       # (B, ~T')

        # t_out이 지정된 경우(훈련)만 고정 길이로 맞춤, None이면 가변 길이 그대로 반환
        if self.t_out is not None:
            if x.shape[1] > self.t_out:
                x = x[:, :self.t_out]
            elif x.shape[1] < self.t_out:
                x = F.pad(x, [0, self.t_out - x.shape[1]])

        # Tanh -> [-eps, eps] L-inf 보장
        x = torch.tanh(x) * self.eps
        return x


# -- Generator 전체 --------------------------------------------

class PerturbationGenerator(nn.Module):
    """
    노이즈를 만들어내는 전체 신경망. Encoder 3단계 -> Bottleneck ->
    Decoder 3단계 -> TimeDomainProjector 순서로 이어붙인 U-Net 구조다.

    eps는 만들어낼 노이즈 크기의 최대 한계값 (기본 0.01).
    """

    def __init__(self, eps: float = EPS_DEFAULT):
        super().__init__()
        self.eps = eps

        # Encoder
        self.enc1 = EncoderBlock(1,   32)
        self.enc2 = EncoderBlock(32,  64)
        self.enc3 = EncoderBlock(64, 128)

        # Bottleneck
        self.bottleneck = Bottleneck(128, 256)

        # Decoder
        self.dec1 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128,  64,  64)
        self.dec3 = DecoderBlock( 64,  32,  32)

        # Time domain 출력
        self.projector = TimeDomainProjector(
            in_ch=32,
            n_freq=N_MELS,
            t_out=T_OUT,
            hop_length=160,
            eps=eps,
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel spectrogram을 입력받아 노이즈(delta)를 만들어 돌려준다."""
        # Encoder
        x, skip1 = self.enc1(mel)    # x:(B,32,40,200)  skip1:(B,32,80,401)
        x, skip2 = self.enc2(x)      # x:(B,64,20,100)  skip2:(B,64,40,200)
        x, skip3 = self.enc3(x)      # x:(B,128,10,50)  skip3:(B,128,20,100)

        # Bottleneck
        x = self.bottleneck(x)       # (B,256,10,50)

        # Decoder
        x = self.dec1(x, skip3)      # (B,128,20,100)
        x = self.dec2(x, skip2)      # (B,64,40,200)
        x = self.dec3(x, skip1)      # (B,32,80,401)

        # Time domain 출력
        delta = self.projector(x)    # (B,64000)
        return delta

    def save(self, path: str):
        torch.save({
            "state_dict": self.state_dict(),
            "eps": self.eps,
        }, path)

    @classmethod
    def load(cls, path: str, device: torch.device = None):
        ckpt = torch.load(path, map_location=device or "cpu")
        model = cls(eps=ckpt["eps"])
        model.load_state_dict(ckpt["state_dict"])
        model.projector.t_out = None  # 추론 시 가변 길이
        model.eval()
        if device is not None:
            model = model.to(device)
        return model
