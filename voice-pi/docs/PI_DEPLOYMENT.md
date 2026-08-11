# Raspberry Pi Deployment

이 문서는 VoiceSecure ONNX 모델을 Raspberry Pi 5 환경에서 실행하기 위한
설치, 모델 준비, 오디오 장치 확인 및 실시간 추론 절차를 설명합니다.

현재 Raspberry Pi 배포에서는 이미 변환 및 검증이 완료된
ONNX 모델을 사용합니다.

---

# 1. Overview

VoiceSecure Raspberry Pi 배포 파이프라인은 다음과 같습니다.

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

Raspberry Pi에서는 별도의 GPU 없이
ONNX Runtime의 `CPUExecutionProvider`를 사용합니다.

---

# 2. Verified Hardware

실제 검증에 사용한 하드웨어 환경은 다음과 같습니다.

```text
Device       : Raspberry Pi 5
RAM          : 8 GB
Architecture : aarch64
Storage      : 256 GB microSD
Power        : Official 27 W Power Adapter
Audio Input  : USB Microphone
Audio Output : HDMI
```

---

# 3. Software Environment

실제 배포 환경:

```text
OS       : Raspberry Pi OS 64-bit
Runtime  : ONNX Runtime
Provider : CPUExecutionProvider
```

Python 패키지는 다음 파일을 기준으로 설치합니다.

```text
requirements_onnx.txt
```

---

# 4. Project Structure

현재 Raspberry Pi 배포 저장소의 구조는 다음과 같습니다.

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

PyTorch 학습 체크포인트와
ONNX 변환용 개발 스크립트는 현재 배포 저장소에 포함하지 않습니다.

---

# 5. Environment Setup

프로젝트 디렉터리로 이동합니다.

```bash
cd ~/voicesecure_pi
```

가상환경 생성:

```bash
python3 -m venv .venv
```

가상환경 활성화:

```bash
source .venv/bin/activate
```

pip 업데이트:

```bash
python -m pip install --upgrade pip
```

필요 패키지 설치:

```bash
pip install -r requirements_onnx.txt
```

설치 후 ONNX Runtime Provider 확인:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

현재 Raspberry Pi에서는 다음 Provider를 사용합니다.

```text
CPUExecutionProvider
```

---

# 6. Model Preparation

Raspberry Pi 추론에 필요한 모델 파일은 다음 두 개입니다.

```text
checkpoints/generator.onnx
checkpoints/generator.onnx.data
```

두 파일은 반드시 같은 디렉터리에 위치해야 합니다.

정상 구조:

```text
checkpoints/
├── generator.onnx
└── generator.onnx.data
```

`generator.onnx.data`가 누락되면
현재 ONNX 모델을 정상적으로 로드할 수 없습니다.

---

# 7. Audio Device Check

사용 가능한 오디오 장치를 확인합니다.

```bash
python realtime_onnx_pi.py --list-devices
```

USB 마이크가 정상적으로 표시되는지 확인합니다.

특정 입력 장치를 사용할 경우:

```bash
python realtime_onnx_pi.py --input-device 2
```

장치 번호 `2`는 예시입니다.

실제 장치 번호는 `--list-devices` 결과를 기준으로 지정해야 합니다.

---

# 8. Offline WAV Inference

이미 존재하는 WAV 파일을 ONNX Runtime으로 처리하려면
다음 명령을 사용합니다.

```bash
python infer_onnx.py \
  --input input/input.wav \
  --output output/protected_onnx.wav
```

한 줄로도 실행할 수 있습니다.

```bash
python infer_onnx.py --input input/input.wav --output output/protected_onnx.wav
```

처리 흐름:

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

# 9. Real-Time Microphone Inference

Raspberry Pi에서 마이크 기반 ONNX 추론을 실행합니다.

```bash
python realtime_onnx_pi.py
```

프로그램 시작 후 다음 메시지가 출력됩니다.

```text
Press ENTER to start recording...
```

`ENTER`를 누르면 녹음이 시작됩니다.

녹음 중:

```text
Recording...
Press ENTER again to stop.
```

다시 `ENTER`를 누르면 녹음을 종료합니다.

이후 다음 과정이 자동으로 수행됩니다.

```text
Recording
   ↓
Resampling
   ↓
Mel Spectrogram
   ↓
Frame Alignment
   ↓
ONNX Inference
   ↓
Perturbation
   ↓
Protected Audio
   ↓
WAV Save
   ↓
Playback
```

---

# 10. Input and Output Paths

기본 녹음 원본 저장 경로:

```text
input/realtime_input.wav
```

기본 보호 음성 저장 경로:

```text
output/realtime_protected_onnx.wav
```

다른 경로를 지정하려면:

```bash
python realtime_onnx_pi.py \
  --input input/realtime_input.wav \
  --output output/realtime_protected_onnx.wav
```

---

# 11. Disable Automatic Playback

추론 후 자동 재생을 사용하지 않으려면:

```bash
python realtime_onnx_pi.py --no-play
```

보호 WAV 파일은 저장되지만
HDMI 자동 재생은 수행하지 않습니다.

---

# 12. HDMI Playback

실제 Raspberry Pi 검증에서 사용한 HDMI 출력 장치는 다음과 같습니다.

```text
plughw:CARD=vc4hdmi1,DEV=0
```

직접 보호 음성을 재생하려면:

```bash
aplay -D plughw:CARD=vc4hdmi1,DEV=0 \
  output/realtime_protected_onnx.wav
```

실행 코드에서 출력 장치를 직접 지정할 수도 있습니다.

```bash
python realtime_onnx_pi.py \
  --playback-device "plughw:CARD=vc4hdmi1,DEV=0"
```

HDMI 장치 이름은 연결 포트와 OS 설정에 따라
달라질 수 있습니다.

---

# 13. Mel Frame Alignment

VoiceSecure Generator의 U-Net 구조는
시간축을 세 번 downsampling합니다.

전체 downsampling factor:

```text
2 × 2 × 2 = 8
```

따라서 Mel Spectrogram frame 수를
8의 배수로 정렬합니다.

예:

```text
521 → 520
723 → 720
```

정렬 방식:

```python
aligned = (frames // 8) * 8
```

이를 통해 Encoder / Decoder skip connection에서
발생할 수 있는 shape mismatch를 방지합니다.

Generator 출력 perturbation 길이가 원본 오디오보다 짧으면
0으로 padding하고,
더 길면 원본 길이에 맞춰 crop합니다.

---

# 14. Runtime Metrics

`realtime_onnx_pi.py`는 추론 완료 후
다음 정보를 출력합니다.

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

---

# 15. Real-Time Factor

RTF(Real-Time Factor)는 다음과 같이 계산합니다.

```text
RTF = ONNX inference time / audio duration
```

해석:

```text
RTF < 1.0  → 입력 음성 길이보다 짧은 시간에 추론 완료
RTF = 1.0  → 입력 음성 길이와 추론 시간이 동일
RTF > 1.0  → 입력 음성 길이보다 추론 시간이 오래 걸림
```

현재 코드의 RTF는
**ONNX 모델 inference time을 기준으로 계산한 값**입니다.

녹음 대기 시간이나 HDMI 재생 시간은
RTF 계산에 포함되지 않습니다.

---

# 16. Raspberry Pi Verification Result

Raspberry Pi 5에서 실제 수행한 한 번의 검증 결과:

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
ONNX 모델 추론 시간이 입력 음성 길이보다 짧았습니다.

이 값은 단일 실행 결과이며,
입력 음성 길이와 Raspberry Pi 실행 상태에 따라
달라질 수 있습니다.

---

# 17. Troubleshooting

## ONNX Model Not Found

다음 파일이 존재하는지 확인합니다.

```text
checkpoints/generator.onnx
checkpoints/generator.onnx.data
```

---

## External Data Error

`generator.onnx.data`가
`generator.onnx`와 같은 디렉터리에 존재하는지 확인합니다.

```text
checkpoints/
├── generator.onnx
└── generator.onnx.data
```

---

## Shape Mismatch

Mel frame alignment가 정상적으로 적용되는지 확인합니다.

```python
aligned = (frames // 8) * 8
```

---

## Microphone Not Detected

오디오 장치 목록을 확인합니다.

```bash
python realtime_onnx_pi.py --list-devices
```

USB 마이크가 표시되는지 확인한 뒤
필요한 경우 `--input-device`로 장치를 직접 지정합니다.

---

## No Audio Playback

생성된 WAV 파일 존재 여부를 먼저 확인합니다.

```bash
ls -lh output/realtime_protected_onnx.wav
```

HDMI 장치 직접 재생 테스트:

```bash
aplay -D plughw:CARD=vc4hdmi1,DEV=0 \
  output/realtime_protected_onnx.wav
```

---

## ONNX Runtime GPU Warning

Raspberry Pi에서 ONNX Runtime 실행 시
GPU 장치 탐색 관련 warning이 출력될 수 있습니다.

현재 배포에서는 다음 Provider를 사용합니다.

```text
CPUExecutionProvider
```

실행 결과에:

```text
providers: ['CPUExecutionProvider']
```

가 표시되고 추론이 정상 완료된다면
CPU 기반 ONNX 추론이 수행되고 있는 것입니다.

---

# 18. Deployment Verification Checklist

Raspberry Pi 5에서 다음 항목을 실제로 확인했습니다.

- ONNX Runtime 세션 생성
- `CPUExecutionProvider` 사용
- ONNX warm-up
- USB 마이크 녹음
- 입력 sample rate 확인
- 48 kHz → 16 kHz 리샘플링
- Mel Spectrogram 생성
- Mel frame alignment
- ONNX Generator 추론
- Perturbation 생성
- 원본 길이에 맞춘 delta pad / crop
- 보호 음성 WAV 저장
- HDMI 오디오 재생
- SNR 측정
- RTF 측정

---

# 19. Deployment Summary

| Step | Description |
| ---: | --- |
| 1 | Prepare Raspberry Pi 5 |
| 2 | Prepare deployment repository |
| 3 | Create Python virtual environment |
| 4 | Install dependencies |
| 5 | Prepare ONNX model files |
| 6 | Check audio devices |
| 7 | Run offline WAV inference |
| 8 | Run microphone-based ONNX inference |
| 9 | Save protected WAV |
| 10 | Verify HDMI playback |
| 11 | Check SNR and RTF |

---

# 20. Notes

- 현재 저장소는 Raspberry Pi 실행 및 배포용입니다.
- PyTorch 학습 체크포인트는 현재 배포 저장소에 포함하지 않습니다.
- ONNX 변환용 개발 스크립트도 현재 배포 저장소에 포함하지 않습니다.
- `generator.onnx`와 `generator.onnx.data`는 반드시 함께 배포해야 합니다.
- Raspberry Pi에서는 ONNX Runtime의 `CPUExecutionProvider`를 사용합니다.
- ONNX Generator 추론에는 별도의 GPU가 필요하지 않습니다.
- 마이크 장치 번호는 연결 환경에 따라 달라질 수 있습니다.
- HDMI 출력 장치 이름은 연결 포트와 OS 설정에 따라 달라질 수 있습니다.
- RTF와 추론 시간은 입력 길이와 실행 환경에 따라 달라질 수 있습니다.
- `input/` 및 `output/` WAV 파일은 Git 저장소에 포함하지 않습니다.

---

# Conclusion

VoiceSecure의 변환 완료 ONNX Generator를
Raspberry Pi 5에서 ONNX Runtime의
`CPUExecutionProvider`로 실행했습니다.

실제 Raspberry Pi 환경에서 다음 전체 파이프라인의
동작을 확인했습니다.

```text
USB Microphone
→ Recording
→ 16 kHz Resampling
→ Mel Spectrogram
→ Frame Alignment
→ ONNX Inference
→ Protected Audio
→ WAV Save
→ HDMI Playback
```

또한 실제 실행 과정에서 ONNX inference time,
RTF 및 SNR을 측정했습니다.

실기기 검증 중 한 실행에서는
5.208초 입력에 대해 1.321초의 ONNX 추론 시간과
0.2536의 RTF를 기록했습니다.

따라서 현재 Raspberry Pi 배포 파이프라인은
Raspberry Pi 5 실기기에서 실행 검증이 완료된 상태입니다.