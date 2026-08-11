# VoiceSecure ONNX Model

이 디렉터리는 VoiceSecure Raspberry Pi 배포에 사용되는
ONNX 모델 파일을 저장합니다.

현재 Raspberry Pi 추론에 필요한 파일은 다음 두 개입니다.

```text
checkpoints/
├── generator.onnx
└── generator.onnx.data
```

---

## 1. `generator.onnx`

VoiceSecure U-Net 기반 Generator를
ONNX 형식으로 변환한 배포용 모델 파일입니다.

```text
checkpoints/generator.onnx
```

ONNX 모델의 그래프 구조를 포함하며,
Raspberry Pi에서는 ONNX Runtime을 통해 로드합니다.

현재 Raspberry Pi 배포에서는 다음 Execution Provider를 사용합니다.

```text
CPUExecutionProvider
```

---

## 2. `generator.onnx.data`

`generator.onnx`에서 사용하는 ONNX External Data 파일입니다.

```text
checkpoints/generator.onnx.data
```

모델의 가중치 데이터가 저장되어 있으며,
`generator.onnx`와 반드시 같은 디렉터리에 있어야 합니다.

```text
checkpoints/
├── generator.onnx
└── generator.onnx.data
```

두 파일 중 하나라도 누락되면
현재 배포용 ONNX 모델을 정상적으로 로드할 수 없습니다.

---

## 3. Model Deployment

본 Raspberry Pi 배포 저장소에서는
이미 변환 및 검증이 완료된 ONNX 모델을 사용합니다.

전체 실행 흐름은 다음과 같습니다.

```text
generator.onnx
generator.onnx.data
        │
        ▼
ONNX Runtime
CPUExecutionProvider
        │
        ▼
VoiceSecure Generator Inference
        │
        ▼
Perturbation
        │
        ▼
Protected Audio
```

Raspberry Pi 실시간 실행은 프로젝트 루트의 다음 파일을 사용합니다.

```text
realtime_onnx_pi.py
```

기존 WAV 파일을 이용한 ONNX 추론은 다음 파일을 사용합니다.

```text
infer_onnx.py
```

---

## 4. Model Generation

현재 저장소에는 PyTorch 학습 체크포인트와
ONNX 변환용 개발 코드를 포함하지 않습니다.

ONNX 모델은 별도의 개발 과정에서 다음 순서로 생성 및 검증되었습니다.

```text
PyTorch Generator
        │
        ▼
ONNX Export
        │
        ▼
generator.onnx
generator.onnx.data
        │
        ▼
PyTorch / ONNX Verification
        │
        ▼
Raspberry Pi Deployment
```

ONNX 변환 및 검증 과정에 대한 자세한 내용은 다음 문서를 참고합니다.

```text
docs/ONNX_DEPLOYMENT.md
docs/BENCHMARK.md
```

---

## 5. Raspberry Pi Usage

Raspberry Pi에서 모델을 실행하기 전에
다음 두 파일이 존재하는지 확인합니다.

```bash
ls -lh checkpoints
```

정상적인 구조:

```text
generator.onnx
generator.onnx.data
```

실시간 마이크 기반 추론:

```bash
python realtime_onnx_pi.py
```

기존 WAV 기반 추론:

```bash
python infer_onnx.py \
  --input input/input.wav \
  --output output/protected_onnx.wav
```

---

## 6. Notes

- `generator.onnx`와 `generator.onnx.data`는 반드시 함께 유지해야 합니다.
- 두 파일은 같은 `checkpoints/` 디렉터리에 위치해야 합니다.
- Raspberry Pi에서는 ONNX Runtime의 `CPUExecutionProvider`를 사용합니다.
- PyTorch 학습 체크포인트는 Raspberry Pi ONNX 추론에 필요하지 않습니다.
- PyTorch 체크포인트 및 ONNX 변환용 개발 코드는 본 배포 저장소에 포함하지 않습니다.
- 모델을 교체할 경우 `generator.onnx`와 대응되는 `generator.onnx.data`를 함께 교체해야 합니다.