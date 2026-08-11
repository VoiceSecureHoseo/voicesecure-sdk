# VoiceSecure ONNX Benchmark

이 문서는 VoiceSecure U-Net Generator의 ONNX 변환 검증 결과와
Raspberry Pi 5에서 수행한 ONNX Runtime 추론 성능 측정 결과를 정리합니다.

PyTorch와 ONNX Runtime의 직접 비교 결과는 ONNX 변환 과정에서 수행한
로컬 CPU 환경의 검증 결과이며,
Raspberry Pi 결과는 실제 Raspberry Pi 5에서 수행한 별도의 실기기 측정 결과입니다.

---

# 1. Benchmark Overview

검증은 크게 두 단계로 구분됩니다.

```text
[ONNX 변환 검증]

PyTorch Generator
        │
        ├───────────────┐
        ▼               ▼
     PyTorch       ONNX Runtime
        │               │
        └───────┬───────┘
                ▼
       Numerical Comparison
       Audio Comparison
       Inference Speed Comparison


[Raspberry Pi 배포 검증]

generator.onnx
generator.onnx.data
        │
        ▼
Raspberry Pi 5
        │
        ▼
ONNX Runtime
CPUExecutionProvider
        │
        ▼
Protected Audio
        │
        ▼
Inference Time / RTF / SNR
```

---

# 2. ONNX Verification Environment

PyTorch와 ONNX Runtime의 출력 및 추론 속도를 비교한
기존 개발 환경은 다음과 같습니다.

| Item | Specification |
| --- | --- |
| Model | U-Net PerturbationGenerator |
| Framework | PyTorch / ONNX Runtime |
| Input | 16 kHz WAV |
| Feature | 80-bin Mel Spectrogram |
| Device | CPU |
| Runtime Provider | CPUExecutionProvider |
| Output | Protected Audio |

이 결과는 ONNX 변환의 정확성과
CPU 환경에서의 추론 성능을 확인하기 위한 검증 결과입니다.

Raspberry Pi 실기기 결과와는 별도로 구분합니다.

---

# 3. PyTorch / ONNX Output Verification

## Output Shape

| Framework | Output Shape |
| --- | --- |
| PyTorch | (1, 64160) |
| ONNX Runtime | (1, 64160) |

동일한 입력에 대해 PyTorch와 ONNX Runtime이
동일한 출력 shape를 생성함을 확인했습니다.

---

## Numerical Accuracy

| Metric | Result |
| --- | ---: |
| Maximum Absolute Error | 1.183e-07 |
| Mean Absolute Error | 6.8e-09 |
| RMSE | 1.05e-08 |
| Comparison SNR | 101.908 dB |

PyTorch와 ONNX Runtime으로 생성한 보호 음성의
수치적 차이가 매우 작음을 확인했습니다.

여기서 `Comparison SNR`은 원본 음성과 보호 음성 사이의 SNR이 아니라,
**PyTorch 보호 음성과 ONNX 보호 음성의 차이를 기준으로 계산한 비교 지표**입니다.

---

# 4. Local CPU Inference Performance

동일한 실제 WAV 입력을 사용하여
PyTorch와 ONNX Runtime의 추론 시간을 비교한 결과입니다.

| Framework | Inference Time |
| --- | ---: |
| PyTorch | 1.5593 sec |
| ONNX Runtime | 0.4640 sec |

---

## Speed Ratio

속도비는 다음과 같이 계산했습니다.

```text
Speed Ratio
= PyTorch inference time / ONNX inference time

= 1.5593 / 0.4640

≈ 3.36
```

따라서 해당 로컬 CPU 검증 실행에서:

```text
ONNX speed ratio ≈ 3.36x
```

를 확인했습니다.

이 값은 **해당 로컬 CPU 환경에서 측정된 PyTorch 대비 ONNX Runtime의 추론 속도비**이며,
Raspberry Pi에서 PyTorch와 ONNX를 직접 비교한 결과가 아닙니다.

---

# 5. WAV Output Verification

동일한 실제 WAV 파일을 이용하여
PyTorch와 ONNX Runtime의 보호 음성을 생성하고 비교했습니다.

검증 항목:

- Output Shape
- Maximum Absolute Error
- Mean Absolute Error
- RMSE
- Comparison SNR
- PyTorch Inference Time
- ONNX Inference Time
- Speed Ratio

검증 결과:

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

이 검증을 통해 ONNX 변환 후에도
PyTorch Generator와 거의 동일한 보호 음성이 생성됨을 확인했습니다.

---

# 6. Raspberry Pi Test Environment

ONNX 모델의 실제 배포 검증은 다음 환경에서 수행했습니다.

| Item | Specification |
| --- | --- |
| Device | Raspberry Pi 5 |
| RAM | 8 GB |
| Architecture | aarch64 |
| OS | Raspberry Pi OS 64-bit |
| Runtime | ONNX Runtime |
| Execution Provider | CPUExecutionProvider |
| Audio Input | USB Microphone |
| Target Sample Rate | 16 kHz |
| Output | Protected WAV / HDMI Audio |

Raspberry Pi에서는 별도의 GPU를 사용하지 않고
CPU 기반으로 ONNX 추론을 수행했습니다.

---

# 7. Raspberry Pi ONNX Inference Result

Raspberry Pi 5 실기기에서 수행한 한 번의 검증 결과는 다음과 같습니다.

| Metric | Result |
| --- | ---: |
| Audio Duration | 5.208 sec |
| ONNX Inference Time | 1.321 sec |
| RTF | 0.2536 |
| SNR | 25.949 dB |
| Execution Provider | CPUExecutionProvider |
| Result | SUCCESS |

---

# 8. Real-Time Factor

RTF(Real-Time Factor)는 ONNX 추론 시간이
입력 음성 길이에 비해 어느 정도인지 나타내는 지표입니다.

계산:

```text
RTF
= ONNX inference time / audio duration

= 1.321 / 5.208

≈ 0.2536
```

따라서 해당 실행에서는:

```text
RTF = 0.2536
```

이 측정되었습니다.

`RTF < 1.0`이므로 해당 실행에서는
입력 음성 길이보다 짧은 시간 안에 ONNX 모델 추론이 완료되었습니다.

또한:

```text
0.2536 × 100 = 25.36%
```

이므로 ONNX 모델 추론 시간은
입력 음성 길이의 약 25.36%에 해당합니다.

이 RTF는 **ONNX 모델 inference time을 입력 음성 길이로 나눈 값**이며,
녹음 대기 시간, 오디오 재생 시간 등 전체 사용자 파이프라인 시간을 의미하지 않습니다.

---

# 9. Raspberry Pi SNR

해당 Raspberry Pi 실행에서 측정된 원본 음성과
보호 음성 사이의 SNR은 다음과 같습니다.

```text
SNR = 25.949 dB
```

이 값은 앞에서 사용한 `Comparison SNR = 101.908 dB`와
서로 다른 의미를 갖습니다.

```text
Comparison SNR
→ PyTorch 보호 출력과 ONNX 보호 출력의 차이 비교

Raspberry Pi SNR
→ 원본 음성과 최종 보호 음성의 차이 측정
```

따라서 두 SNR 값은 직접 비교하는 지표가 아닙니다.

---

# 10. Result Summary

## ONNX Conversion Verification

| Category | Result |
| --- | --- |
| PyTorch / ONNX Output Shape | Identical |
| Maximum Absolute Error | 1.183e-07 |
| Mean Absolute Error | 6.8e-09 |
| RMSE | 1.05e-08 |
| Comparison SNR | 101.908 dB |
| Local CPU ONNX Speed Ratio | 3.36x |
| Verification | PASS |

---

## Raspberry Pi Deployment Verification

| Category | Result |
| --- | --- |
| Device | Raspberry Pi 5 |
| Architecture | aarch64 |
| Execution Provider | CPUExecutionProvider |
| Audio Duration | 5.208 sec |
| ONNX Inference | 1.321 sec |
| RTF | 0.2536 |
| Original / Protected SNR | 25.949 dB |
| ONNX Execution | SUCCESS |

---

# 11. Interpretation

기존 개발 환경에서 수행한 PyTorch / ONNX 비교에서는
ONNX Runtime으로 변환된 Generator가 PyTorch 모델과
매우 작은 수치 차이로 동일한 출력을 생성함을 확인했습니다.

해당 로컬 CPU 검증 실행에서는 ONNX Runtime의 추론 시간이
PyTorch보다 약 3.36배 짧았습니다.

또한 Raspberry Pi 5 실기기에서 ONNX Runtime의
`CPUExecutionProvider`를 사용하여 보호 음성을 정상적으로 생성했으며,
한 번의 측정에서 5.208초 입력에 대해 1.321초의 ONNX 추론 시간과
0.2536의 RTF를 기록했습니다.

이 결과는 **ONNX 모델이 Raspberry Pi 5 CPU 환경에서 실제로 실행되었고,
해당 측정 입력에 대해 RTF가 1 미만이었음**을 보여줍니다.

---

# 12. Notes

- PyTorch / ONNX의 3.36x 속도비는 기존 로컬 CPU 환경에서 측정된 결과입니다.
- 3.36x 값을 Raspberry Pi의 PyTorch 대비 성능 향상 수치로 해석하면 안 됩니다.
- Raspberry Pi의 0.2536 RTF는 한 번의 실기기 측정 결과입니다.
- Raspberry Pi 성능은 입력 길이, 시스템 상태 및 실행 환경에 따라 달라질 수 있습니다.
- `Comparison SNR`과 원본 / 보호 음성 SNR은 서로 다른 지표입니다.
- Raspberry Pi의 RTF는 ONNX inference time을 기준으로 계산한 값입니다.

---

# Conclusion

VoiceSecure U-Net Generator의 ONNX 변환 결과는
기존 PyTorch 모델과 매우 작은 수치 차이를 보였으며,
기존 로컬 CPU 검증에서는 ONNX Runtime이 PyTorch 대비
약 3.36배 짧은 추론 시간을 기록했습니다.

또한 Raspberry Pi 5의 `CPUExecutionProvider` 환경에서
ONNX 모델을 이용한 보호 음성 생성이 실제로 정상 수행되었습니다.

실기기 검증 중 한 실행에서는 5.208초 음성에 대해
1.321초의 ONNX 추론 시간과 0.2536의 RTF를 기록하여,
해당 조건에서 ONNX 모델 추론이 입력 음성 길이보다 짧은 시간 안에
완료됨을 확인했습니다.