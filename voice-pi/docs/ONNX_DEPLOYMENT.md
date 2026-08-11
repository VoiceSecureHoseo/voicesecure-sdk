# VoiceSecure ONNX Deployment

이 문서는 VoiceSecure U-Net Generator를 ONNX Runtime 형식으로 변환하고,
PyTorch 모델과 ONNX 모델의 출력을 검증한 과정을 기록합니다.

현재 Raspberry Pi 배포 저장소에서는
변환 및 검증이 완료된 ONNX 모델을 사용합니다.

---

# 1. 목적

VoiceSecure Generator는 Raspberry Pi와 같은 CPU 기반 환경에서
ONNX Runtime을 이용하여 추론할 수 있도록 ONNX 형식으로 변환되었습니다.

ONNX 변환 및 검증 과정의 주요 목적은 다음과 같습니다.

- PyTorch Generator의 ONNX 변환
- CPU 기반 ONNX Runtime 추론
- Raspberry Pi 배포
- PyTorch / ONNX 출력 동일성 검증
- 실제 WAV 기반 출력 검증
- CPU 추론 성능 비교

현재 Raspberry Pi 배포에는 다음 모델을 사용합니다.

```text
checkpoints/generator.onnx
checkpoints/generator.onnx.data
```

---

# 2. Deployment Pipeline

전체 모델 개발 및 배포 과정은 다음과 같습니다.

```text
PyTorch Training
        │
        ▼
PyTorch Generator Checkpoint
        │
        ▼
ONNX Export
        │
        ▼
generator.onnx
generator.onnx.data
        │
        ▼
Numerical Verification
        │
        ▼
WAV Output Verification
        │
        ▼
Raspberry Pi Deployment
        │
        ▼
ONNX Runtime
CPUExecutionProvider
```

본 Raspberry Pi 배포 저장소는
위 과정 중 ONNX 변환 및 검증이 완료된 이후의
배포 단계를 담당합니다.

---

# 3. PyTorch Checkpoint

ONNX 변환에는 학습이 완료된
U-Net 기반 `PerturbationGenerator` 체크포인트가 사용되었습니다.

해당 체크포인트에는 다음과 같은 학습 관련 정보가 포함되어 있었습니다.

```text
epoch
best_dist
loss_history
generator
optimizer
scheduler
eps
```

ONNX 배포에 필요한 것은 학습된 Generator의 구조와 가중치이며,
Optimizer와 Scheduler는 추론 과정에서 사용되지 않습니다.

PyTorch 학습 체크포인트는 현재 Raspberry Pi 배포 저장소에는
포함하지 않습니다.

---

# 4. ONNX Export

학습된 PyTorch Generator는 ONNX opset 18 형식으로 변환되었습니다.

변환 결과:

```text
generator.onnx
generator.onnx.data
```

현재 Raspberry Pi 배포 저장소에서는 다음 위치에 저장합니다.

```text
checkpoints/
├── generator.onnx
└── generator.onnx.data
```

ONNX 변환에 사용된 개발 스크립트와 PyTorch 모델 정의는
현재 Raspberry Pi 배포 저장소에 포함하지 않습니다.

본 저장소에서는 이미 변환된 ONNX 모델을 사용합니다.

---

# 5. ONNX External Data

현재 배포 모델은 다음 두 파일로 구성됩니다.

```text
generator.onnx
generator.onnx.data
```

`generator.onnx`는 ONNX 모델 그래프를 포함하고,
`generator.onnx.data`는 External Data 형식으로 저장된
모델 가중치 데이터를 포함합니다.

따라서 Raspberry Pi에 모델을 배포할 때는
두 파일을 반드시 함께 배포해야 합니다.

```text
checkpoints/
├── generator.onnx
└── generator.onnx.data
```

두 파일 중 하나라도 누락되면
현재 배포 모델을 정상적으로 로드할 수 없습니다.

---

# 6. Numerical Verification

ONNX 변환 후 PyTorch Generator와 ONNX Runtime이
동일한 입력에 대해 수치적으로 일치하는 출력을 생성하는지 검증했습니다.

검증 항목:

- Output Shape
- Maximum Absolute Error
- Mean Absolute Error
- Numerical Tolerance

검증 결과:

```text
PyTorch output shape : (1, 64160)
ONNX output shape    : (1, 64160)

Maximum absolute error : 1.0150e-06
Mean absolute error    : 9.3e-09

PyTorch vs ONNX : PASS
```

이 결과는 ONNX 변환 과정에서 수행한
기존 개발 환경의 검증 결과입니다.

현재 Raspberry Pi 배포 저장소에는
해당 PyTorch 검증 스크립트가 포함되어 있지 않습니다.

---

# 7. Mel Frame Alignment

Generator는 U-Net 구조를 사용하며,
Encoder 과정에서 시간축을 세 번 downsampling합니다.

각 단계의 downsampling factor가 2이므로
전체 factor는 다음과 같습니다.

```text
2 × 2 × 2 = 8
```

따라서 입력 Mel Spectrogram의 frame 수를
8의 배수로 정렬하여 Encoder와 Decoder의
skip connection shape mismatch를 방지합니다.

예:

```text
Original Frames : 723
Aligned Frames  : 720
```

또 다른 예:

```text
Original Frames : 521
Aligned Frames  : 520
```

정렬 방식:

```python
aligned = (frames // 8) * 8
```

이 frame alignment는 현재 Raspberry Pi 실행 코드에서도 적용됩니다.

---

# 8. WAV Output Verification

ONNX 변환 후 실제 WAV 파일을 입력하여
PyTorch와 ONNX Runtime이 생성한 보호 음성을 비교했습니다.

검증 항목:

- Output Shape
- PyTorch Inference Time
- ONNX Inference Time
- Maximum Absolute Error
- Mean Absolute Error
- RMSE
- Comparison SNR

기존 검증 결과:

```text
PyTorch inference time : 1.5593 sec
ONNX inference time    : 0.4640 sec
ONNX speed ratio       : 3.36x

Maximum absolute error : 0.0000001183
Mean absolute error    : 0.0000000068
RMSE                   : 0.0000000105
Comparison SNR         : 101.908 dB

PyTorch vs ONNX WAV: PASS
```

여기서 `Comparison SNR`은
원본 음성과 보호 음성 사이의 SNR이 아니라,
PyTorch 보호 출력과 ONNX 보호 출력의 차이를 비교한 지표입니다.

또한 `3.36x`는 해당 로컬 CPU 검증 환경에서 측정된
PyTorch 대비 ONNX Runtime의 속도비입니다.

Raspberry Pi에서 PyTorch와 ONNX를 직접 비교한 결과는 아닙니다.

---

# 9. Current Repository Deployment Model

현재 Raspberry Pi 배포 저장소에서는
ONNX 변환 과정을 다시 수행하지 않습니다.

배포에 필요한 모델은 다음 두 파일입니다.

```text
checkpoints/generator.onnx
checkpoints/generator.onnx.data
```

현재 저장소의 ONNX 추론 흐름은 다음과 같습니다.

```text
Input Audio
      │
      ▼
16 kHz Audio
      │
      ▼
Mel Spectrogram
      │
      ▼
Frame Alignment
      │
      ▼
generator.onnx
generator.onnx.data
      │
      ▼
ONNX Runtime
CPUExecutionProvider
      │
      ▼
Perturbation
      │
      ▼
Protected Audio
```

---

# 10. WAV Inference

현재 저장소에서 기존 WAV 파일에 대한
ONNX Runtime 추론은 다음 파일을 사용합니다.

```text
infer_onnx.py
```

실행:

```bash
python infer_onnx.py \
    --input input/input.wav \
    --output output/protected_onnx.wav
```

한 줄로 실행할 수도 있습니다.

```bash
python infer_onnx.py --input input/input.wav --output output/protected_onnx.wav
```

처리 과정:

```text
Input WAV
      │
      ▼
Audio Load
      │
      ▼
Mono Conversion
      │
      ▼
16 kHz Resampling
      │
      ▼
Mel Spectrogram
      │
      ▼
Frame Alignment
      │
      ▼
ONNX Runtime
      │
      ▼
Perturbation
      │
      ▼
Protected Audio
      │
      ▼
Output WAV
```

---

# 11. Raspberry Pi Deployment

Raspberry Pi에서는 ONNX Runtime의
`CPUExecutionProvider`를 사용하여 추론합니다.

실시간 실행 파일:

```text
realtime_onnx_pi.py
```

전체 처리 과정:

```text
USB Microphone
      │
      ▼
Audio Recording
      │
      ▼
Input Sample Rate Detection
      │
      ▼
16 kHz Resampling
      │
      ▼
Mel Spectrogram
      │
      ▼
Frame Alignment
      │
      ▼
ONNX Runtime
CPUExecutionProvider
      │
      ▼
Perturbation
      │
      ▼
Protected Audio
      │
      ├── WAV Save
      │
      └── HDMI Playback
```

Raspberry Pi에서 기본 실행:

```bash
python realtime_onnx_pi.py
```

오디오 장치 확인:

```bash
python realtime_onnx_pi.py --list-devices
```

자동 재생 없이 실행:

```bash
python realtime_onnx_pi.py --no-play
```

---

# 12. Raspberry Pi Verification

Raspberry Pi 5에서 ONNX Runtime을 사용한
실기기 추론이 정상적으로 수행됨을 확인했습니다.

검증 환경:

```text
Device       : Raspberry Pi 5
RAM          : 8 GB
Architecture : aarch64
OS           : Raspberry Pi OS 64-bit
Provider     : CPUExecutionProvider
```

실기기 검증 중 한 실행 결과:

```text
Audio duration : 5.208 sec
ONNX inference : 1.321 sec
RTF            : 0.2536
SNR            : 25.949 dB
Result         : SUCCESS
```

RTF 계산:

```text
RTF
= ONNX inference time / audio duration

= 1.321 / 5.208

≈ 0.2536
```

따라서 해당 실행에서는
ONNX 모델 추론 시간이 입력 음성 길이보다 짧았습니다.

위 값은 한 번의 실기기 실행 결과이며,
입력 길이와 Raspberry Pi 실행 상태에 따라 달라질 수 있습니다.

---

# 13. Development and Deployment Separation

VoiceSecure의 모델 개발 과정과
Raspberry Pi 배포 과정은 다음과 같이 구분합니다.

```text
[Model Development]

Training
   ↓
PyTorch Checkpoint
   ↓
ONNX Export
   ↓
PyTorch / ONNX Verification
   ↓
WAV Output Verification


[Raspberry Pi Deployment]

generator.onnx
generator.onnx.data
   ↓
ONNX Runtime
   ↓
Microphone / WAV Input
   ↓
Protected Audio
```

현재 Raspberry Pi 배포 저장소에는
배포 단계에 필요한 코드와 모델을 중심으로 구성합니다.

PyTorch 학습 체크포인트,
모델 학습 코드,
ONNX 변환 스크립트,
PyTorch / ONNX 비교 스크립트는
현재 배포 저장소에 포함하지 않습니다.

---

# 14. Related Documents

ONNX 및 Raspberry Pi 배포와 관련된 문서는 다음과 같습니다.

```text
docs/BENCHMARK.md
docs/PI_DEPLOYMENT.md
checkpoints/README.md
```

- `BENCHMARK.md`
  - PyTorch / ONNX 수치 비교
  - 로컬 CPU 추론 성능 비교
  - Raspberry Pi ONNX 추론 결과

- `PI_DEPLOYMENT.md`
  - Raspberry Pi 환경 설정
  - 모델 배치
  - 마이크 및 HDMI 설정
  - 실시간 실행 방법

- `checkpoints/README.md`
  - 배포용 ONNX 모델 파일 설명

---

# 15. Notes

- 현재 Raspberry Pi 배포 저장소에서는 ONNX 변환을 다시 수행하지 않습니다.
- PyTorch 학습 체크포인트는 Raspberry Pi ONNX 추론에 필요하지 않습니다.
- `generator.onnx`와 `generator.onnx.data`는 반드시 함께 배포해야 합니다.
- 두 모델 파일은 동일한 `checkpoints/` 디렉터리에 있어야 합니다.
- Raspberry Pi에서는 ONNX Runtime의 `CPUExecutionProvider`를 사용합니다.
- Mel frame 수는 U-Net 구조에 맞춰 8의 배수로 정렬합니다.
- 로컬 CPU의 3.36x 속도비와 Raspberry Pi의 RTF 0.2536은 서로 다른 환경에서 측정된 별도의 결과입니다.

---

# Conclusion

VoiceSecure U-Net Generator는 PyTorch 학습 모델에서
ONNX 형식으로 변환되었으며,
기존 검증 과정에서 PyTorch와 ONNX Runtime 출력 사이의
수치적 차이가 매우 작음을 확인했습니다.

실제 WAV 기반 비교에서도 PyTorch와 ONNX 출력의
높은 수치적 일치도를 확인했습니다.

현재 Raspberry Pi 배포 저장소에서는
변환 및 검증이 완료된 `generator.onnx`와
`generator.onnx.data`를 사용합니다.

또한 Raspberry Pi 5의 `CPUExecutionProvider` 환경에서
ONNX 추론 및 보호 음성 생성이 정상적으로 수행되었으며,
실기기 검증 중 한 실행에서는 5.208초 입력에 대해
1.321초의 ONNX 추론 시간과 0.2536의 RTF를 기록했습니다.