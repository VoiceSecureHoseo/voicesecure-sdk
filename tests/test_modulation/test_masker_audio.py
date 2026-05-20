"""마스킹 + 노이즈 적용 후 실제 음성 파일로 저장하는 테스트.

Original/ 폴더에서 음성 하나 골라서
PsychoacousticMasker + Mixer 적용 후 tests/voice/ 에 저장.

실행:
    pytest tests/test_modulation/test_masker_audio.py -v -s
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from voicesecure.modulation.masker import PsychoacousticMasker
from voicesecure.modulation.mixer import Mixer

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
