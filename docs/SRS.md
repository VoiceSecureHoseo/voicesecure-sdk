# VoiceSecure SDK — Software Requirements Specification (SRS) v1.0

| 항목 | 내용 |
|---|---|
| **문서 상태** | v1.0 (확정) |
| **작성일** | 2025-03-31 |
| **참조 문서** | PRD v1.0, ARCHITECTURE.md v1.0 |
| **다음 리뷰** | 모듈 통합 직후 (5월 초) |

> **PRD가 "왜·무엇"이라면, SRS는 "정확히 어떻게"다.**
> 이 문서는 각 기능의 입출력·동작·예외처리·성능 요구를 측정 가능한 형태로 명세한다.

---

## 1. 시스템 개요

### 1.1 시스템 컨텍스트

```
┌──────────────────────────────────────────────────────┐
│  Host Application (사용자 앱)                         │
│  - 통화 앱, 메신저, 녹음 앱 등                        │
│                                                       │
│  ┌────────────────────────────────────────────────┐  │
│  │  VoiceSecure SDK                                │  │
│  │                                                 │  │
│  │   audio in → [변형 엔진] → audio out           │  │
│  │                                                 │  │
│  │   (RL Agent + Masker + Mixer + Safety Check)   │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
                          ↓ (offline only, 학습 시)
                  Evaluation Models
                  (WavLM-SV, CAM++, OpenVoice2, XTTS, wav2vec2)
```

### 1.2 시스템 모드

| 모드 | 사용 시점 | 활성 컴포넌트 |
|---|---|---|
| **Inference** | 사용자 기기에서 실시간 음성 보호 | StateExtractor, RLAgent (frozen), Masker, Mixer, SafetyChecker |
| **Training** | 개발 단계 (서버) | 위 + 3개 Evaluator + RewardFunction + PPO 업데이트 |

---

## 2. 기능 요구사항 (Functional Requirements)

각 기능에 ID 부여. 추적성을 위해 PRD의 G1~G9와 매핑.

### FR-1: 음성 입력 처리

**ID**: FR-1
**관련 PRD**: G4, G7
**우선순위**: P0 (필수)

#### 입력
- 오디오 데이터: PCM, 16kHz, 16bit, mono
- 길이: 정확히 1초 청크 (16000 샘플)
  - 짧은 청크 (100ms ~ 1초): 무음 zero-padding으로 1초 채움
  - 긴 음성 (> 1초): 호출자가 1초 청크로 분할 후 순차 호출

#### v1.0 설계 결정 (긴 음성 처리)
실시간 음성 보호는 청크 단위 스트리밍 처리가 표준이므로, SDK는 1초 청크만 처리.
긴 음성은 호출자가 분할해서 호출 (예시는 README 참조).
types.py의 `ACTION_N_TIME = 100`과 정합.

#### 동작
1. 입력 검증: sample rate, dtype, 청크 크기 확인
2. 정규화: int16 → float32 [-1, 1]
3. resampling: 16kHz가 아니면 16kHz로 변환
4. 길이 보정:
   - `len(audio) < 16000`: zero-padding으로 16000 샘플 채움
   - `len(audio) > 16000`: `ChunkSizeError` 발생 (호출자가 청크 분할 책임)
5. 모듈 간 표준 타입 (`AudioArray`)으로 변환

#### 출력
- `AudioArray`: shape `(num_samples,)`, float32, [-1, 1]

#### 예외
- 입력 길이 < 100ms (1600 샘플): `InsufficientAudioError`
- 입력 길이 > 16000 샘플: `ChunkSizeError` — 호출자가 분할해서 다시 호출 필요
- 무음 구간 (RMS < 0.001): pass-through (변형 안 함)

---

### FR-2: State 추출

**ID**: FR-2
**관련 PRD**: 내부 모듈
**우선순위**: P0
**구현 위치**: `rl/state.py:StateExtractor`

#### 입력
- `AudioArray`: shape `(num_samples,)`

#### 동작
1. MFCC 계산 (n_mfcc=13, n_fft=512, hop_length=160)
2. 시간축 평균 → 13-dim
3. F0 추정 (librosa.yin), 평균 → 1-dim
4. duration (초) 정규화 → 1-dim
5. RMS energy → 1-dim
6. 16-dim 벡터로 결합, float32 torch.Tensor 반환

#### 출력
- `State`: shape `(16,)`, dtype=float32

#### 성능 요구
- 처리 시간 ≤ 10ms (1초 입력 기준, Raspberry Pi 4B)

#### 예외
- F0 추정 실패: 0으로 채움 + 경고 로그

---

### FR-3: 노이즈 정책 추론 (RL Agent)

**ID**: FR-3
**관련 PRD**: G1, G2 (방어), G4 (실시간)
**우선순위**: P0
**구현 위치**: `rl/agent.py:RLAgent`

#### 입력
- `State`: shape `(16,)`, float32

#### 동작
1. PolicyNetwork.forward(state)
2. Inference 모드 (`deterministic=True`) 시 mean action 반환
3. Training 모드 (`deterministic=False`) 시 정책 분포에서 sampling

#### 출력
- `Action` (raw noise): shape `(n_freq_bins, n_time_frames)`, float32
  - `n_freq_bins = 257` (n_fft=512 / 2 + 1)
  - `n_time_frames = 100` (1초 청크 기준 고정, `types.py:ACTION_N_TIME`)

#### 성능 요구
- 추론 시간 ≤ 5ms (1초 입력, Raspberry Pi 4B, ONNX)
- 모델 크기 ≤ 1MB (ONNX 변환 후)

#### 예외
- NaN 출력 감지: 0으로 대체 + `RLInferenceWarning`
- GPU 메모리 부족 (학습 시): batch size 자동 감소

---

### FR-4: 심리음향 마스킹

**ID**: FR-4
**관련 PRD**: G3 (명료성), G5 (음질)
**우선순위**: P0
**구현 위치**: `modulation/masker.py:PsychoacousticMasker`

#### 입력
- `audio`: `AudioArray`, shape `(num_samples,)`
- `raw_noise`: `Action`, shape `(257, n_time_frames)`

#### 동작
1. 원본 음성 STFT → 주파수 영역
2. 청각 임계치 계산:
   - Bark scale 변환
   - Spreading function 적용
   - Absolute threshold of hearing 적용
3. raw_noise를 임계치로 element-wise clamp:
   ```
   safe_noise = sign(raw_noise) * min(|raw_noise|, threshold)
   ```

#### 출력
- `safe_noise`: shape (raw_noise와 동일), 모든 element가 임계치 이내 보장

#### 성능 요구
- 처리 시간 ≤ 30ms (1초 입력, Raspberry Pi 4B)
- 결과 검증: `assert torch.all(torch.abs(safe_noise) <= threshold)` 통과

#### 예외
- 임계치 계산 실패 (예: 무음 입력): threshold = 0 → safe_noise = 0

---

### FR-5: 음성 합성 (Mixing)

**ID**: FR-5
**관련 PRD**: 핵심 출력 단계
**우선순위**: P0
**구현 위치**: `modulation/mixer.py:Mixer`

#### 입력
- `audio`: 원본 `AudioArray`
- `safe_noise`: 안전 노이즈 spectrogram

#### 동작
1. audio STFT → complex spectrogram
2. spectrogram + safe_noise (magnitude만 더하고 phase는 원본 유지)
3. iSTFT → 시간 영역 음성
4. clipping: `np.clip(out, -1.0, 1.0)`
5. 원본 길이로 truncate (STFT padding 제거)

#### 출력
- `modified_audio`: `AudioArray`, shape `(num_samples,)` (원본과 동일 길이)

#### 성능 요구
- 처리 시간 ≤ 20ms (1초 입력, 벡터 PRO 병렬연산 활용)

---

### FR-6: 안전성 검증

**ID**: FR-6
**관련 PRD**: 안전장치
**우선순위**: P0
**구현 위치**: `modulation/safety.py:SafetyChecker`

#### 입력
- `original`: 원본 `AudioArray`
- `modified`: Mixer 출력 `AudioArray`

#### 동작 (순서대로)
1. NaN/Inf 검출
   - 발견 시: 해당 샘플을 0으로 대체 + 로그
2. 범위 검증: `[-1, 1]` 초과 sample 카운트
   - > 1% 초과: clipping 적용
3. RMS 비율 검증: `rms(modified) / rms(original)`
   - 비율 < 0.5 또는 > 2.0: fallback (modified = 0.7 × modified + 0.3 × original)
4. Length 검증: `len(modified) == len(original)`
   - 불일치: `SafetyError` 발생 (구현 버그 의미)

#### 출력
- `safe_modified`: 검증 통과한 `AudioArray`

#### 예외
- `SafetyError`: 복구 불가능한 경우 (length mismatch 등)

#### 로깅
- 모든 보정 작업은 `WARNING` 레벨로 로깅 (운영 시 빈도 모니터링용)

---

### FR-7: 화자 식별 평가 (학습 시)

**ID**: FR-7
**관련 PRD**: G1
**우선순위**: P0
**구현 위치**: `evaluators/speaker.py:SpeakerEvaluator`

#### 입력
- `original`, `modified`: `AudioArray`

#### 동작
1. WavLM-SV로 두 음성의 embedding 추출
2. CAM++로 두 음성의 embedding 추출
3. cosine distance 계산:
   - `wavlm_dist = 1 - cosine_similarity(wavlm_emb_orig, wavlm_emb_mod)`
   - `cam_dist = 1 - cosine_similarity(cam_emb_orig, cam_emb_mod)`
4. 평균: `score = (wavlm_dist + cam_dist) / 2`
5. score를 [0, 1]로 정규화 (sigmoid 또는 clipping)

#### 출력
- `EvaluatorOutput`:
  - `score`: 정규화된 거리 [0, 1]
  - `raw_metric`: 평균 cosine distance (정규화 전)
  - `metadata`: `{"wavlm_dist": float, "cam_dist": float}`

#### 성능 요구
- 처리 시간: 학습 환경에서 ≤ 500ms (GPU)
- Feature caching: 원본 embedding은 1회만 계산

---

### FR-8: TTS 복제 평가 (학습 시)

**ID**: FR-8
**관련 PRD**: G2
**우선순위**: P0
**구현 위치**: `evaluators/tts.py:TTSEvaluator`

#### 입력
- `original`, `modified`: `AudioArray`

#### 동작
1. modified를 reference voice로 OpenVoice2 복제 → `clone_ov2`
2. modified를 reference voice로 XTTS 복제 → `clone_xtts`
3. 각 클론과 `original`의 cosine distance 측정 (WavLM-SV로)
4. 평균: `clone_failure = (dist(clone_ov2, original) + dist(clone_xtts, original)) / 2`
5. 정규화 → score [0, 1]

#### 출력
- `EvaluatorOutput`

#### 성능 요구
- TTS 추론이 무거우므로 학습 시 sampling 적용 (예: 매 10 episode마다 1회)

---

### FR-9: ASR 명료성 평가 (학습 시)

**ID**: FR-9
**관련 PRD**: G3
**우선순위**: P0
**구현 위치**: `evaluators/asr.py:ASREvaluator`

#### 입력
- `original`, `modified`: `AudioArray`
- `original_text`: 원본 정답 텍스트 (학습 데이터로 제공)

#### 동작
1. wav2vec2-xlsr-korean으로 modified → 텍스트 인식
2. 인식 결과와 `original_text`의 CER (Character Error Rate) 계산
3. score = `max(0, 1 - cer)`

#### 출력
- `EvaluatorOutput`:
  - `score`: [0, 1]
  - `raw_metric`: CER

---

### FR-10: Reward 계산

**ID**: FR-10
**관련 PRD**: 학습 신호
**우선순위**: P0
**구현 위치**: `reward/function.py:RewardFunction`

#### 입력
- `RewardComponents` (TypedDict):
  - `sv_score`: float
  - `tts_score`: float
  - `asr_cer`: float

#### 동작
```
R = α·sv_score + β·tts_score − λ·max(0, cer − cer_threshold)

기본값: α=0.6, β=0.4, λ=1.0, cer_threshold=0.3
```

#### 출력
- `Reward`: float

#### 검증
- 기본 하이퍼파라미터에서 R 범위는 약 [-0.7, 1.0]

---

### FR-11: Feature 캐싱

**ID**: FR-11
**관련 PRD**: 학습 효율화
**우선순위**: P1
**구현 위치**: `utils/cache.py:FeatureCache`

#### 동작
1. `audio_id` 기반 dict 자료구조
2. `get_or_compute(audio_id, evaluator, audio)`:
   - 캐시 hit: 저장된 features 반환
   - 캐시 miss: `evaluator.precompute(audio)` 호출 후 저장
3. v1.0은 in-memory dict (영속화는 v1.1)

#### 성능 효과
- WavLM/CAM++ embedding 재계산 제거 → 학습 속도 약 5x

---

### FR-12: 학습 진입점

**ID**: FR-12
**우선순위**: P1
**구현 위치**: `scripts/train.py`

#### 입력
- config YAML 경로 (CLI 인자)
- 데이터셋 경로

#### 동작
1. config 로드, 모듈 인스턴스 생성
2. 학습 루프:
   - 데이터셋에서 (audio, text) 샘플
   - StateExtractor → RLAgent → Masker → Mixer → SafetyCheck
   - Evaluators 호출, Reward 계산
   - Transition 저장
   - N개 Transition 모이면 PPO 업데이트
3. 매 epoch마다 체크포인트 저장
4. wandb/tensorboard 로깅

#### 종료
- N epoch 도달
- 또는 사용자 인터럽트 (Ctrl+C)

---

### FR-13: 추론 진입점

**ID**: FR-13
**우선순위**: P0
**구현 위치**: `scripts/infer.py`

#### 입력
- 모델 체크포인트 경로
- 입력 오디오 파일 또는 stream

#### 동작
1. 체크포인트 로드 → RLAgent
2. 오디오 파일: 한 번에 처리
3. Stream 모드: 1초 청크 단위로 처리, 결과 즉시 출력

#### 출력
- 변형된 오디오 파일 또는 stream

---

## 3. 비기능 요구사항 (Non-Functional Requirements)

### NFR-1: 성능 (Performance)

| 환경 | 입력 | 목표 latency | 비고 |
|---|---|---|---|
| Raspberry Pi 4B (Phase 1) | 1초 음성 | ≤ 200ms | ONNX, 벡터 PRO |
| Pixel 6 (Phase 2) | 1초 음성 | ≤ 300ms | ONNX, NNAPI |
| 서버 (학습) | batch 32 | ≤ 5초/iter | GPU |

### NFR-2: 메모리

| 환경 | peak RAM | 모델 weight |
|---|---|---|
| Raspberry Pi 4B | ≤ 500MB | ≤ 1MB (RLAgent) |
| Android | ≤ 200MB | ≤ 1MB |

### NFR-3: 정확도/품질

PRD G1~G5와 동일.

### NFR-4: 안정성

- 단위 테스트 커버리지 ≥ 70%
- 24시간 연속 추론 부하 테스트에서 메모리 누수 없음
- 1000회 추론 중 SafetyError 발생률 ≤ 0.1%

### NFR-5: 호환성

- Python 3.10, 3.11
- PyTorch 2.0+
- ONNX Runtime 1.15+
- Android API 26+ (Phase 2)

### NFR-6: 보안/프라이버시

- 사용자 음성은 디스크에 저장하지 않음 (메모리 처리만)
- 외부 네트워크 요청 없음 (학습 후 추론 시)

### NFR-7: 사용성 (Integrator 관점)

- API 진입점 ≤ 3개
- 통합에 필요한 코드 ≤ 10 lines
- README + API 문서 + 동작 sample app 제공

### NFR-8: 유지보수성

- 모듈 의존성 단방향 (ARCHITECTURE.md 1번 항목)
- 모든 공개 메서드에 docstring + type hint
- pre-commit hooks 통과 강제

---

## 4. 외부 인터페이스 (Interface Requirements)

### 4.1 Python API (Phase 1)

```python
from voicesecure import VoiceSecure

# 초기화 (체크포인트 로드)
vs = VoiceSecure(model_path="checkpoints/v1.0.onnx")

# 추론 — 파일
vs.protect_file("input.wav", "output.wav")

# 추론 — numpy array
audio_in = librosa.load("input.wav", sr=16000)[0]
audio_out = vs.protect(audio_in)

# Stream 모드
for chunk_in in audio_stream():
    chunk_out = vs.protect_chunk(chunk_in)
    play(chunk_out)
```

### 4.2 Android SDK API (Phase 2)

```kotlin
// 초기화
val voiceSecure = VoiceSecure(context, modelPath = "voicesecure_v1.onnx")

// PCM 16-bit short array 처리
val protectedAudio: ShortArray = voiceSecure.protect(rawAudio)

// AudioRecord 통합
voiceSecure.attachToRecord(audioRecord) { protected ->
    // 변형된 음성 자동 출력
}
```

### 4.3 학습 CLI

```bash
# 학습
python scripts/train.py --config configs/default.yaml

# 추론
python scripts/infer.py --model checkpoints/v1.0.onnx --input input.wav --output output.wav
```

### 4.4 데이터 형식

| 항목 | 형식 |
|---|---|
| 입력 오디오 파일 | WAV, FLAC, MP3 (librosa로 디코딩) |
| 출력 오디오 파일 | WAV, 16-bit PCM, 16kHz mono |
| 체크포인트 | PyTorch `.pt` (학습) → ONNX `.onnx` (배포) |
| Config | YAML |
| 로그 | JSON Lines (한 줄 = 한 이벤트) |

---

## 5. 검증 계획 (Verification Plan)

각 FR에 대한 검증 방법.

### 5.1 단위 테스트 (Unit Test)

| FR | 테스트 파일 | 검증 항목 |
|---|---|---|
| FR-1 | `test_audio.py` | 형식 변환, 길이 검증, 예외 |
| FR-2 | `test_state.py` | 출력 shape, dtype, 정규화 |
| FR-3 | `test_agent.py` | forward pass, NaN handling |
| FR-4 | `test_masker.py` | clamp 결과 ≤ threshold |
| FR-5 | `test_mixer.py` | 출력 길이, 범위 |
| FR-6 | `test_safety.py` | NaN/Inf/clipping 처리 |
| FR-7 | `test_speaker.py` | mock 모델로 score 계산 |
| FR-8 | `test_tts.py` | (slow) 실제 TTS 모델 |
| FR-9 | `test_asr.py` | (slow) 실제 ASR 모델 |
| FR-10 | `test_reward.py` | 가중합, threshold 동작 |
| FR-11 | `test_cache.py` | hit/miss, key 충돌 |

### 5.2 통합 테스트 (Integration Test)

- **end-to-end**: `audio in → 전체 파이프라인 → audio out` 동작
- **학습 1 step**: 더미 데이터로 PPO 업데이트 1회 성공
- **체크포인트 round-trip**: save → load → 동일 결과

### 5.3 성능 테스트 (Performance Test)

- Raspberry Pi 4B 실기 latency 측정 (`pytest-benchmark`)
- 메모리 프로파일링 (`memory-profiler`)
- 24시간 연속 부하 테스트

### 5.4 정확도 테스트 (Accuracy Test)

- KsponSpeech 100문장 샘플로 PRD G1~G3 측정
- 결과를 `evaluation/results.csv`에 기록, 매 주차 비교

### 5.5 청취 테스트 (Listening Test)

- ABX 테스트: 팀원 4명 + 외부 5명, 각 50쌍 청취
- PRD G5 측정

---

## 6. 데이터 요구사항

### 6.1 학습 데이터

| 데이터셋 | 용도 | 크기 |
|---|---|---|
| KsponSpeech (한국어) | 메인 학습 | 약 1000시간 |
| LibriSpeech-clean (영어) | 일반화 검증 | 약 100시간 |
| 자체 수집 (선택) | 추가 화자 다양성 | 미정 |

### 6.2 평가 모델

| 모델 | 출처 | 라이선스 | 가중치 다운로드 |
|---|---|---|---|
| WavLM-SV | microsoft/wavlm-base-plus-sv | MIT | HuggingFace |
| CAM++ | modelscope/CAM++ | Apache 2.0 | ModelScope |
| OpenVoice2 | myshell-ai/OpenVoice | MIT | HuggingFace |
| XTTS v2 | coqui/XTTS-v2 | CPML | HuggingFace |
| wav2vec2-xlsr-korean | kresnik/wav2vec2-large-xlsr-korean | Apache 2.0 | HuggingFace |

---

## 7. 변경 이력 (Change Log)

| 버전 | 날짜 | 변경 사항 | 작성자 |
|---|---|---|---|
| v1.0 | 2025-03-31 | 초안 확정 | dPsk |
| v1.1 | 2025-05-18 | FR-1, FR-3을 types.py (ACTION_N_TIME=100)와 정합화 | dPsk |

---

**문서 책임자**: dPsk
**문의**: GitHub Issues (label: `srs`)
