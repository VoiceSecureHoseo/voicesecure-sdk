# Voice Train

CAM++ 화자 임베딩 기준으로 원본 목소리와 최대한 다르게 인식되도록, perturbation을
실시간으로 생성하는 Generator 신경망을 학습하고 추론하는 코드다.

기존 PGD 방식은 파일마다 수십~수백 스텝을 최적화해야 해서 파일 하나에 수 초~수십 초가
걸린다. 이 코드는 perturbation을 만드는 신경망 자체를 한 번 학습해두는 방식이라, 학습이
끝난 뒤에는 forward pass 한 번(~0.01초)으로 추론이 끝난다.

## 구조

```
setup.py            설치 스크립트 (의존성 설치 + .env 생성 + 필수 파일 확인)
train.py            학습 진입점
evaluate.py          평가 스크립트
infer.py             추론 스크립트 (학습된 generator로 음성 보호)
dataset.py           오디오 데이터셋 로더
features.py          mel feature 추출
model.py             PerturbationGenerator 정의
loss.py              CAM++ 코사인 거리 기반 loss
psycho.py            심리음향 마스킹 기반 loss
train_epoch.py       epoch 단위 학습 루프
train_utils.py       체크포인트 저장/로드 유틸

run_train.bat        Windows용 학습 실행 (더블클릭)
run_evaluate.bat     Windows용 평가 실행
run_infer.bat        Windows용 추론 실행
```

## 대용량 파일 (git에는 없음)

용량 문제로 아래 파일은 저장소에 포함하지 않았다. `.gitignore`에 등록되어 있다.

- `campplus.onnx` — CAM++ 화자 임베딩 모델. [FunASR / CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
  등에서 받아 `CAMPLUS_ONNX` 환경변수로 경로를 지정한다.

## 설치

```bash
python setup.py
```

의존성 설치, `.env` 생성(이미 있으면 건드리지 않음), `campplus.onnx`/`kss` 같은 필수
파일이 있는지 확인까지 한 번에 처리한다. 없는 파일이 있으면 어디서 받아야 하는지
안내해준다. `.env`가 새로 생성됐다면 그 안의 값을 채워야 한다.

- `KSS_DIR` — KSS(한국어 단일 화자 음성) 데이터셋 경로.
- `KOREAN_SPEECH_DIR` — 추가 한국어 음성 데이터셋 경로 (선택).
- `CAMPPLUS_ONNX` — CAM++ onnx 모델 경로.
- `SAMPLE_WAV` — 학습 후 빠른 추론 테스트에 쓸 샘플 wav 경로 (선택, 기본값은 KSS 첫 파일).

`.env`에 채운 값은 실행 전에 셸에 로드해야 한다.

```bash
set -a; source .env; set +a   # bash
```

## 실행

```bash
python train.py --epochs 100 --batch 8
python train.py --epochs 50 --batch 16 --resume   # 이어서 학습

python evaluate.py --checkpoint checkpoints/best.pt --wav_dir <경로>
python infer.py --checkpoint checkpoints/best.pt --input <wav경로>
```

학습이 끝나면 `checkpoints/` 아래에 `best.pt`(최고 성능), `last.pt`(마지막 체크포인트),
`emb_cache.pt`(화자 임베딩 캐시), `train_log.json`(epoch별 지표),
`quick_output/`(샘플 음성 보호 결과)가 생성된다.

### Windows에서 더블클릭으로 실행

`run_train.bat` / `run_evaluate.bat` / `run_infer.bat`을 더블클릭하면 환경변수
(`KSS_DIR`, `CAMPPLUS_ONNX`)를 자동으로 설정해서 실행한다. 기본값은 `kss_mini`
(빠른 동작 확인용 100개 미니 데이터셋)를 사용하며, 전체 데이터로 학습하려면
`run_train.bat`을 열어 `KSS_DIR`을 `kss`로 바꾼다.

```
run_train.bat --epochs 1 --batch 2      (인자 없이 실행하면 기본값 --epochs 10 --batch 16)
run_infer.bat kss_mini\1\1_0000.wav
```

---

## 모델 설계

파일: `model.py`
파라미터 수: 약 55M
입력: mel spectrogram `(B, 1, 80, 401)`
출력: perturbation delta `(B, 64000)`, range `[-eps, +eps]` (기본 eps=0.01)

mel 도메인에서 U-Net으로 특징을 뽑아낸 뒤, 시간 도메인 waveform으로 복원하는 구조다.

```
입력 mel (B,1,80,401)
  -> Encoder 3단계 (Conv x2 + stride-2 다운샘플, skip 저장)
  -> Bottleneck (Conv x2)
  -> Decoder 3단계 (업샘플 + skip concat + Conv x2)
  -> TimeDomainProjector: reshape -> ConvTranspose1d x2 -> Conv1d
     -> [:64000] 슬라이스/패딩 -> Tanh * eps
  -> 출력 delta (B, 64000)
```

- skip connection으로 세밀한 주파수 정보를 보존한다 (U-Net 구조).
- Decoder에서 skip과 크기가 안 맞으면 pad/crop으로 자동 보정한다.
- 마지막에 `Tanh * eps`를 곱해서 L-inf <= eps를 수학적으로 보장한다.
- `projector.t_out`을 학습 때는 64000으로 고정하고, 추론 때는 None으로 풀어서 가변 길이 입력을 처리한다.

## Loss 설계

파일: `loss.py`

```
Total = w_cam * L_cam + w_snr_dynamic * L_snr + w_psycho * L_psycho + w_linf * L_linf
```

| 항 | 역할 | 기본 가중치 |
|----|------|-----------|
| L_cam | CAM++ 코사인 거리 최대화 (화자 인식 헷갈리게 만들기, 핵심 목표) | 1.0 |
| L_snr | SNR 28dB 이상 유지 | 0.1 (미달 시 동적으로 최대 10배까지 증가) |
| L_psycho | 심리음향 마스킹 초과분에 페널티 | 0.05 |
| L_linf | perturbation 크기가 eps를 넘지 않도록 억제 | 10.0 |

## 학습 파이프라인

```
KSS wav 파일
  -> AudioDataset: 4초로 자르거나 zero-pad, augmentation 적용
  -> DataLoader (배치 단위)
  -> MelExtractor(no_grad)로 mel 추출  +  캐싱해둔 원본 CAM++ 임베딩 조회
  -> Generator(mel) -> delta
  -> adv = clamp(orig + delta, -1, 1)
  -> L_cam / L_snr / L_psycho / L_linf 계산 후 합산
  -> backward -> grad clip -> Adam.step()
```

기본 하이퍼파라미터는 Adam, lr 1e-4(CosineAnnealingLR로 감쇠), batch 16, grad_clip 1.0,
eps 0.01, SNR 목표 28dB다.

원본 오디오의 CAM++ 임베딩은 매 배치마다 다시 계산하면 낭비라서, 학습 시작 전 한 번만
전체를 훑어서 캐싱해두고(`checkpoints/emb_cache.pt`) 이후 배치마다 꺼내 쓴다.

`infer.py`는 노이즈를 만든 뒤 `--no_psycho`를 안 주면 심리음향 한계선을 넘는 부분을
`hard_project`로 한 번 더 깎아내는 후처리를 추가로 거친다. 사람 귀에는 덜 들리게 되지만
CAM++ dist로 나타나는 방어 효과는 다소 줄어드는 트레이드오프가 있다.
