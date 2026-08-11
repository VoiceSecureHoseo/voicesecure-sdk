# VoiceSecure Raspberry Pi Deployment

VoiceSecure 음성 보호 시스템을 Raspberry Pi 환경에서 실행하기 위한
ONNX Runtime 기반 배포 모듈입니다.

사용자의 음성을 마이크로 녹음한 뒤, U-Net 기반 Generator가 생성한
보호 perturbation을 적용하여 보호 음성을 생성합니다.

생성된 보호 음성은 WAV 파일로 저장할 수 있으며,
Raspberry Pi의 HDMI 오디오 출력을 통해 직접 재생할 수 있습니다.

---

## 1. 프로젝트 목적

VoiceSecure는 사람이 이해할 수 있는 음성 품질을 유지하면서
AI 기반 화자 인증 및 음성 복제 모델의 화자 인식 성능을 낮추는 것을 목표로 합니다.

본 저장소는 학습 및 ONNX 변환이 완료된 VoiceSecure Generator를
Raspberry Pi 5의 CPU 환경에서 실행하기 위한 배포 코드를 제공합니다.

주요 기능은 다음과 같습니다.

- 기존 WAV 파일 ONNX 추론
- USB 마이크 실시간 녹음
- 입력 음성 16 kHz 리샘플링
- Mel Spectrogram 생성
- Mel frame alignment
- ONNX Runtime CPU 추론
- 보호 perturbation 생성
- 보호 음성 WAV 저장
- Raspberry Pi HDMI 오디오 재생
- SNR 측정
- RTF(Real-Time Factor) 측정
- Raspberry Pi 추론 성능 측정

---

## 2. 배포 모델

현재 Raspberry Pi 배포에는 U-Net 기반 `PerturbationGenerator`를
ONNX 형식으로 변환한 모델을 사용합니다.

배포용 모델:

```text
checkpoints/generator.onnx
checkpoints/generator.onnx.data
```

`generator.onnx`는 ONNX 모델 그래프를 포함하고,
`generator.onnx.data`는 External Data 형식으로 저장된 모델 가중치를 포함합니다.

두 파일은 반드시 같은 디렉터리에 존재해야 합니다.

PyTorch 학습 체크포인트와 ONNX 변환에 사용된 개발 코드는
본 Raspberry Pi 배포 저장소에 포함하지 않습니다.

본 저장소에서는 변환 및 검증이 완료된 ONNX 모델을 사용하여
Raspberry Pi에서 추론을 수행합니다.

---

## 3. 프로젝트 구조

```text
VOICESECURE_PI/
│
├── checkpoints/
│   ├── generator.onnx
│   ├── generator.onnx.data
│   └── README.md
│
├── docs/
│   ├── BENCHMARK.md
│   ├── ONNX_DEPLOYMENT.md
│   └── PI_DEPLOYMENT.md
│
├── input/
│   ├── input.wav
│   └── realtime_input.wav
│
├── output/
│
├── features.py
├── infer_onnx.py
├── realtime_onnx_pi.py
│
├── requirements_onnx.txt
├── README.md
└── .gitignore
```

로컬 가상환경(`venv/`, `.venv/`), Python cache 및 생성된 출력 파일은
Git 저장소에 포함하지 않습니다.

---

## 4. 주요 파일

### `features.py`

Generator 입력으로 사용되는 Mel Spectrogram을 생성합니다.

주요 설정:

```text
Sample Rate : 16000 Hz
FFT Size    : 512
Hop Length  : 160
Win Length  : 400
Mel Bins    : 80
```

ONNX Generator가 사용할 수 있는 형태의 Mel Spectrogram을 생성하는
전처리 기능을 담당합니다.

---

### `infer_onnx.py`

이미 존재하는 WAV 파일을 입력으로 사용하여
ONNX Runtime 기반 보호 음성을 생성합니다.

처리 과정:

```text
Input WAV
    │
    ▼
Audio Load
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

### `realtime_onnx_pi.py`

Raspberry Pi에서 실제 마이크 입력부터 보호 음성 재생까지
전체 파이프라인을 수행합니다.

```text
USB Microphone
      │
      ▼
Audio Recording
      │
      ▼
Sample Rate Detection
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
      ├── WAV Save
      │
      └── HDMI Playback
```

---

## 5. 검증 환경

### Windows 로컬 환경

ONNX 모델의 기본 동작 확인 및 WAV 기반 추론 검증에 사용한 환경입니다.

```text
OS           : Windows 11
Python       : 3.12
Runtime      : ONNX Runtime
Execution    : CPUExecutionProvider
```

### Raspberry Pi 환경

실제 배포 및 추론 검증에 사용한 환경입니다.

```text
Device       : Raspberry Pi 5
RAM          : 8 GB
Architecture : aarch64
OS           : Raspberry Pi OS 64-bit
Runtime      : ONNX Runtime
Execution    : CPUExecutionProvider
```

Raspberry Pi에서는 별도의 GPU 없이
CPU 기반으로 ONNX 추론을 수행합니다.

---

## 6. Python 환경 설정

### 가상환경 생성

Raspberry Pi 터미널에서 프로젝트 디렉터리로 이동한 뒤
가상환경을 생성합니다.

```bash
python3 -m venv .venv
```

### 가상환경 활성화

```bash
source .venv/bin/activate
```

가상환경이 활성화되면 터미널 앞에 다음과 같이 표시됩니다.

```text
(.venv)
```

### pip 업데이트

```bash
python -m pip install --upgrade pip
```

### 패키지 설치

```bash
pip install -r requirements_onnx.txt
```

주요 사용 패키지는 다음과 같습니다.

```text
numpy
scipy
soundfile
sounddevice
torch
torchaudio
onnx
onnxscript
onnxruntime
```

정확한 설치 조건은 `requirements_onnx.txt`를 기준으로 합니다.

---

## 7. 배포 모델 확인

Raspberry Pi에서 추론을 실행하기 전에
다음 두 파일이 존재하는지 확인합니다.

```text
checkpoints/generator.onnx
checkpoints/generator.onnx.data
```

두 파일은 반드시 같은 디렉터리에 있어야 합니다.

```text
checkpoints/
├── generator.onnx
└── generator.onnx.data
```

ONNX External Data 형식을 사용하기 때문에
`generator.onnx.data`가 누락되면 정상적으로 모델을 로드할 수 없습니다.

---

## 8. 기존 WAV 파일 ONNX 추론

기존 WAV 파일을 보호 음성으로 변환합니다.

```bash
python infer_onnx.py \
  --input input/input.wav \
  --output output/protected_onnx.wav
```

한 줄로도 실행할 수 있습니다.

```bash
python infer_onnx.py --input input/input.wav --output output/protected_onnx.wav
```

기존 검증 실행 예시:

```text
Input duration  : 7.227 sec
Mel frames      : 723 → 720
ONNX inference  : 0.412 sec
RTF             : 0.0570
SNR             : 28.116 dB
Result          : SUCCESS
```

위 결과는 기존 WAV 기반 ONNX 추론 검증 결과이며,
Raspberry Pi 실기기 성능 결과와는 별도로 구분합니다.

---

## 9. Raspberry Pi 실시간 실행

### 오디오 장치 확인

먼저 Raspberry Pi에서 사용 가능한 오디오 장치를 확인합니다.

```bash
python realtime_onnx_pi.py --list-devices
```

---

### 기본 마이크 및 기본 출력 장치 사용

```bash
python realtime_onnx_pi.py
```

프로그램 실행 후:

```text
Press ENTER to start recording...
```

이 표시되면 `ENTER`를 눌러 녹음을 시작합니다.

녹음 중:

```text
Recording...
Press ENTER again to stop.
```

다시 `ENTER`를 누르면 녹음이 종료되고
ONNX 추론이 자동으로 진행됩니다.

---

### 특정 입력 마이크 사용

```bash
python realtime_onnx_pi.py --input-device 2
```

장치 번호는 다음 명령으로 확인합니다.

```bash
python realtime_onnx_pi.py --list-devices
```

---

### 자동 재생 없이 실행

```bash
python realtime_onnx_pi.py --no-play
```

---

### 출력 장치 직접 지정

```bash
python realtime_onnx_pi.py \
  --playback-device "plughw:CARD=vc4hdmi1,DEV=0"
```

---

### 입력 및 출력 WAV 경로 지정

```bash
python realtime_onnx_pi.py \
  --input input/realtime_input.wav \
  --output output/realtime_protected_onnx.wav
```

---

## 10. Raspberry Pi 출력 지표

`realtime_onnx_pi.py` 실행이 완료되면 다음 정보를 확인할 수 있습니다.

```text
raw delta shape
final delta shape
delta min/max
L-inf
mean abs delta
SNR
ONNX inference
RTF
total elapsed
protected saved
```

### RTF

RTF(Real-Time Factor)는 다음과 같이 계산합니다.

```text
RTF = ONNX inference time / audio duration
```

예를 들어:

```text
Audio duration = 5.208 sec
ONNX inference = 1.321 sec

RTF = 1.321 / 5.208
    ≈ 0.2536
```

따라서:

```text
RTF < 1.0
```

이면 입력 음성 길이보다 짧은 시간 안에
ONNX 추론이 완료되었다는 의미입니다.

---

## 11. Raspberry Pi 실기기 검증 결과

Raspberry Pi 5에서 실제 수행한 검증 중 한 실행 결과는 다음과 같습니다.

```text
Audio duration   : 5.208 sec
ONNX inference   : 1.321 sec
RTF              : 0.2536
SNR              : 25.949 dB
Provider         : CPUExecutionProvider
Result           : SUCCESS
```

RTF 계산:

```text
1.321 / 5.208 ≈ 0.2536
```

따라서 해당 실행에서는
입력 음성 길이의 약 25.36%에 해당하는 시간으로 ONNX 추론을 완료했습니다.

위 값은 단일 실행 사례이며,
입력 음성 길이와 Raspberry Pi 실행 환경에 따라 달라질 수 있습니다.

---

## 12. HDMI 재생

현재 Raspberry Pi 검증에서 사용한 HDMI 출력 장치는 다음과 같습니다.

```text
plughw:CARD=vc4hdmi1,DEV=0
```

직접 WAV 파일을 재생하려면:

```bash
aplay -D plughw:CARD=vc4hdmi1,DEV=0 \
  output/realtime_protected_onnx.wav
```

HDMI 장치 이름은 Raspberry Pi의 연결 포트 및
운영체제 설정에 따라 달라질 수 있습니다.

따라서 다른 Raspberry Pi 환경에서는
사용 가능한 오디오 장치를 먼저 확인해야 합니다.

---

## 13. Mel Frame Alignment

Generator의 U-Net 구조는 시간축을 세 번 downsampling합니다.

따라서 downsampling factor는:

```text
2^3 = 8
```

입니다.

ONNX Runtime에서 Encoder / Decoder skip connection의 shape mismatch를
방지하기 위해 Mel frame 수를 8의 배수로 정렬합니다.

예:

```text
723 frames → 720 frames
521 frames → 520 frames
```

정렬은 다음 방식으로 수행합니다.

```text
aligned_frames = (frames // 8) * 8
```

Generator 출력 perturbation의 길이가 원본 오디오보다 짧은 경우
나머지 부분을 0으로 padding하여 원본 길이에 맞춥니다.

반대로 perturbation이 원본보다 긴 경우에는
원본 오디오 길이에 맞춰 crop합니다.

---

## 14. PyTorch → ONNX 변환 및 검증

본 Raspberry Pi 배포 저장소에서는
변환이 완료된 ONNX 모델을 사용합니다.

PyTorch 학습 체크포인트에서 ONNX 모델을 생성하는 과정과
PyTorch / ONNX 출력 비교 과정은 별도의 개발 환경에서 수행되었습니다.

검증 과정에는 다음 항목이 포함되었습니다.

```text
PyTorch Checkpoint
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
```

관련 검증 결과와 과정은 다음 문서에 기록되어 있습니다.

```text
docs/ONNX_DEPLOYMENT.md
docs/BENCHMARK.md
```

---

## 15. 기존 ONNX 검증 결과

ONNX 변환 과정에서 PyTorch와 ONNX Runtime 출력의
수치적 동일성을 비교한 결과는 다음과 같습니다.

```text
PyTorch output shape : (1, 64160)
ONNX output shape    : (1, 64160)

Maximum absolute error : 0.0000010150
Mean absolute error    : 0.0000000093

PyTorch vs ONNX: PASS
```

실제 WAV 기반 비교 결과:

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

위 결과는 ONNX 변환 및 출력 동일성을 확인하기 위해
기존 개발 환경에서 수행한 검증 결과입니다.

Raspberry Pi 실기기 성능 결과와는 구분합니다.

---

## 16. 현재 진행 상태

### 완료

- U-Net Generator 체크포인트 분석
- ONNX opset 18 변환
- ONNX External Data 생성
- PyTorch ↔ ONNX 출력 검증
- 실제 WAV 기반 PyTorch ↔ ONNX 비교
- WAV 기반 ONNX 추론
- Raspberry Pi 5 실기기 배포
- Raspberry Pi OS 64-bit / aarch64 실행 확인
- ONNX Runtime `CPUExecutionProvider` 실행
- USB 마이크 녹음
- 입력 샘플레이트 자동 확인
- 16 kHz 리샘플링
- Mel Spectrogram 생성
- Mel frame alignment
- 보호 perturbation 생성
- 보호 음성 WAV 저장
- HDMI 오디오 재생
- Raspberry Pi ONNX inference time 측정
- Raspberry Pi RTF 측정
- Raspberry Pi SNR 측정

### 남은 작업

- Raspberry Pi 배포 저장소 최종 정리
- GitHub Repository 반영
- 기존 VoiceSecure SDK와의 통합

---

## 17. 관련 문서

세부 내용은 다음 문서를 참고합니다.

```text
docs/ONNX_DEPLOYMENT.md
docs/BENCHMARK.md
docs/PI_DEPLOYMENT.md
checkpoints/README.md
```

### `docs/ONNX_DEPLOYMENT.md`

PyTorch Generator를 ONNX 형식으로 변환하고
출력을 검증한 과정을 기록합니다.

### `docs/BENCHMARK.md`

PyTorch / ONNX 비교 결과와
Raspberry Pi 추론 성능 측정 결과를 기록합니다.

### `docs/PI_DEPLOYMENT.md`

Raspberry Pi 환경 설정 및
실행 절차를 설명합니다.

### `checkpoints/README.md`

Raspberry Pi에서 사용하는 ONNX 모델 파일을 설명합니다.

---

## 18. 주의사항

- `generator.onnx`와 `generator.onnx.data`는 반드시 같은 폴더에 있어야 합니다.
- Raspberry Pi 배포에는 변환 완료된 ONNX 모델을 사용합니다.
- PyTorch 학습 체크포인트는 Raspberry Pi ONNX 추론에 필요하지 않습니다.
- ONNX 모델은 Raspberry Pi에서 `CPUExecutionProvider`로 실행합니다.
- 마이크 입력은 추론 전에 16 kHz mono 형식으로 변환됩니다.
- 마이크 장치 번호는 실행 환경에 따라 달라질 수 있습니다.
- HDMI 재생 장치 이름은 연결 환경에 따라 달라질 수 있습니다.
- RTF 및 추론 시간은 입력 길이와 실행 환경에 따라 달라질 수 있습니다.
- `input/`과 `output/`의 WAV 파일은 테스트 및 실행 과정에서 생성될 수 있으며 Git 저장소에는 포함하지 않습니다.