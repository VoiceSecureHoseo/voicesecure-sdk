"""마스킹 + 노이즈 적용 후 실제 음성 파일로 저장하고 임베딩 거리 측정하는 테스트.

Original/ 폴더에서 음성 하나 골라서
PsychoacousticMasker + Mixer 적용 후 tests/voice/ 에 저장.
WavLM으로 원본 vs 변조 임베딩 거리 측정 → 방어 효과 사전 검증.

실행 (Original/ 폴더 필요):
    pytest -m slow tests/test_modulation/test_masker_audio.py -v -s
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F

from voicesecure.evaluators.adapters.wavlm_sv import WavLMSVAdapter
from voicesecure.evaluators.base import cosine_distance
from voicesecure.modulation.masker import PsychoacousticMasker
from voicesecure.modulation.mixer import Mixer

pytestmark = pytest.mark.slow

ORIGINAL_DIR = Path(__file__).parents[2] / "Original"
OUTPUT_DIR = Path(__file__).parents[1] / "voice"

SAMPLE_RATE = 16000


@pytest.fixture
def sample_audio() -> tuple[np.ndarray, Path]:
    """Original/ 에서 첫 번째 wav 파일 로드."""
    wav_files = sorted(ORIGINAL_DIR.rglob("*.wav"))
    if not wav_files:
        pytest.skip("Original/ 폴더에 wav 파일 없음")
    path = wav_files[0]
    import librosa

    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32), path


def test_masking_and_save(sample_audio: tuple[np.ndarray, Path]) -> None:
    """마스킹 + 노이즈 적용 후 tests/voice/ 에 저장."""
    audio, src_path = sample_audio
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    masker = PsychoacousticMasker()
    mixer = Mixer()

    # 랜덤 노이즈 (257 x 100)
    torch.manual_seed(42)
    n_freq = masker.config.n_fft // 2 + 1
    raw_noise = torch.randn(n_freq, 100) * 0.5

    # 마스킹 적용
    safe_noise = masker.clamp(audio, raw_noise)

    # 원본 + 노이즈 믹싱
    modified = mixer.mix(audio, safe_noise)

    # 저장
    out_original = OUTPUT_DIR / f"original_{src_path.stem}.wav"
    out_modified = OUTPUT_DIR / f"modified_{src_path.stem}.wav"

    sf.write(str(out_original), audio, SAMPLE_RATE)
    sf.write(str(out_modified), modified, SAMPLE_RATE)

    print(f"\n원본 저장: {out_original}")
    print(f"변조 저장: {out_modified}")
    print(f"원본 길이: {len(audio)} samples ({len(audio)/SAMPLE_RATE:.2f}s)")
    print(f"변조 길이: {len(modified)} samples ({len(modified)/SAMPLE_RATE:.2f}s)")
    print(f"노이즈 RMS: {safe_noise.abs().mean().item():.6f}")
    print(f"원본 RMS:   {np.sqrt(np.mean(audio**2)):.6f}")

    assert out_original.exists()
    assert out_modified.exists()
    assert len(modified) > 0


def fgsm_direction(
    audio: np.ndarray,
    wavlm: WavLMSVAdapter,
    masker: PsychoacousticMasker,
) -> torch.Tensor:
    """FGSM으로 WavLM 임베딩을 최대로 흔드는 노이즈 방향 계산.

    waveform에서 직접 gradient를 구해 STFT magnitude 방향으로 변환.
    feature_extractor가 numpy 경유 → gradient 끊김 → input_values에 직접 grad 붙임.
    """
    device = wavlm.device
    model = wavlm.model

    # input_values를 직접 만들어서 gradient 추적
    # feature_extractor는 내부적으로 mean0/std1 normalize → 간단히 직접 normalize
    audio_np = audio.astype(np.float64)
    mean = audio_np.mean()
    std = audio_np.std() + 1e-8
    normed = ((audio_np - mean) / std).astype(np.float32)

    input_values = torch.from_numpy(normed).float().unsqueeze(0).to(device)
    input_values.requires_grad_(True)

    # 원본 임베딩 (gradient 없이, raw input 그대로)
    with torch.no_grad():
        orig_emb = model(input_values.detach()).embeddings  # (1, 512)

    # forward (gradient 추적)
    mod_emb = model(input_values).embeddings  # (1, 512)

    # cosine similarity 최소화 = 거리 최대화
    loss = -F.cosine_similarity(orig_emb, mod_emb).mean()
    loss.backward()

    # waveform gradient → STFT magnitude 방향으로 변환
    grad = input_values.grad.squeeze(0).cpu()  # (T,)

    window = torch.hann_window(masker.config.win_length)
    grad_spec = torch.stft(
        grad,
        n_fft=masker.config.n_fft,
        hop_length=masker.config.hop_length,
        win_length=masker.config.win_length,
        window=window,
        return_complex=True,
    )
    # STFT magnitude의 gradient → 각 bin의 magnitude를 어느 방향으로 바꿔야 하는지
    direction = torch.sign(torch.abs(grad_spec))
    direction = torch.where(direction == 0, torch.ones_like(direction), direction)
    return direction


def test_embedding_distance(sample_audio: tuple[np.ndarray, Path]) -> None:
    """원본 vs 변조 음성의 WavLM 임베딩 거리 측정.

    거리가 0에 가까우면 AI가 같은 사람으로 인식 → 방어 효과 없음.
    거리가 클수록 방어 효과 있음 (최대 2.0).
    """
    audio, _ = sample_audio

    masker = PsychoacousticMasker()
    mixer = Mixer()
    wavlm = WavLMSVAdapter()

    torch.manual_seed(42)
    n_freq = masker.config.n_fft // 2 + 1
    raw_noise = torch.randn(n_freq, 100) * 0.5
    safe_noise = masker.clamp(audio, raw_noise)
    modified = mixer.mix(audio, safe_noise)

    orig_emb = wavlm.extract_embedding(audio)
    mod_emb = wavlm.extract_embedding(modified)
    dist = cosine_distance(orig_emb, mod_emb)

    print(f"\n원본 vs 변조 WavLM 코사인 거리 (노이즈+워핑): {dist:.6f}")
    print("  → 0에 가까울수록 AI가 같은 사람으로 인식 (방어 효과 없음)")
    print("  → 클수록 방어 효과 있음 (최대 2.0)")
    print(f"노이즈 RMS: {safe_noise.abs().mean().item():.6f}")
    print(f"원본 RMS:   {np.sqrt(np.mean(audio**2)):.6f}")


def test_fgsm_embedding_distance(sample_audio: tuple[np.ndarray, Path]) -> None:
    """FGSM 방향 노이즈 vs 랜덤 노이즈 임베딩 거리 비교.

    FGSM으로 WavLM gradient 방향을 구해서 노이즈 방향으로 사용.
    랜덤 노이즈(~0.000813)와 비교해서 방어 효과 차이 확인.
    """
    audio, src_path = sample_audio
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    masker = PsychoacousticMasker()
    mixer = Mixer()
    wavlm = WavLMSVAdapter()

    # FGSM 방향으로 노이즈 생성 (2D direction → warp_direction은 zeros)
    direction = fgsm_direction(audio, wavlm, masker)
    safe_noise = masker.clamp(audio, direction)
    modified = mixer.mix(audio, safe_noise)

    # 저장
    out_fgsm = OUTPUT_DIR / f"fgsm_{src_path.stem}.wav"
    sf.write(str(out_fgsm), modified, SAMPLE_RATE)

    orig_emb = wavlm.extract_embedding(audio)
    mod_emb = wavlm.extract_embedding(modified)
    dist = cosine_distance(orig_emb, mod_emb)

    print(f"\nFGSM 방향 WavLM 코사인 거리: {dist:.6f}")
    print("랜덤 노이즈 거리 (이전):      0.000813")
    print(f"  → FGSM이 랜덤 대비 {dist / 0.000813:.1f}배 효과")
    print(f"노이즈 RMS: {safe_noise.abs().mean().item():.6f}")
    print(f"원본 RMS:   {np.sqrt(np.mean(audio**2)):.6f}")
    print(f"FGSM 저장: {out_fgsm}")
