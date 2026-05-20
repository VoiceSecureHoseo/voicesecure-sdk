"""CAMPlusAdapter 테스트.

@pytest.mark.slow 테스트는 실제 모델 다운로드가 필요하므로
nightly 또는 수동 실행: pytest -m slow tests/test_evaluators/test_adapters_campplus.py
"""

import numpy as np
import pytest

from voicesecure.types import SAMPLE_RATE


def _make_audio(freq: float, duration: float = 1.0) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * duration), dtype=np.float32) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ── import / 클래스 구조 (모델 로드 없이 확인) ──────────────────────────────

def test_adapter_importable():
    from voicesecure.evaluators.adapters import CAMPlusAdapter
    assert CAMPlusAdapter is not None


def test_adapter_has_required_method():
    from voicesecure.evaluators.adapters import CAMPlusAdapter
    assert hasattr(CAMPlusAdapter, "extract_embedding")


# ── fbank 추출 단독 테스트 (모델 로드 없이 확인) ─────────────────────────────

def test_fbank_shape():
    """fbank 출력 shape 확인 — (T, 80)."""
    from voicesecure.evaluators.adapters.campplus import _extract_fbank
    audio = _make_audio(200.0)
    feats = _extract_fbank(audio)
    assert feats.ndim == 2
    assert feats.shape[1] == 80
    assert feats.dtype == np.float32


def test_fbank_no_nan():
    """fbank에 NaN/Inf 없는지 확인."""
    from voicesecure.evaluators.adapters.campplus import _extract_fbank
    audio = _make_audio(200.0)
    feats = _extract_fbank(audio)
    assert np.all(np.isfinite(feats))


def test_fbank_too_short_raises():
    """너무 짧은 음성은 ValueError."""
    from voicesecure.evaluators.adapters.campplus import _extract_fbank
    audio = np.zeros(10, dtype=np.float32)
    with pytest.raises(ValueError):
        _extract_fbank(audio)


# ── 실제 모델 로드 테스트 (slow) ─────────────────────────────────────────────

@pytest.mark.slow
def test_campplus_loads():
    """모델이 에러 없이 로드되는지 확인."""
    from voicesecure.evaluators.adapters import CAMPlusAdapter
    adapter = CAMPlusAdapter()
    assert adapter._session is not None


@pytest.mark.slow
def test_embedding_shape():
    """embedding shape이 (512,) float32인지 확인."""
    from voicesecure.evaluators.adapters import CAMPlusAdapter
    adapter = CAMPlusAdapter()
    audio = _make_audio(200.0)
    emb = adapter.extract_embedding(audio)

    print(f"\n  embedding shape: {emb.shape}, dtype: {emb.dtype}")
    assert emb.shape == (512,)
    assert emb.dtype == np.float32


@pytest.mark.slow
def test_embedding_no_nan():
    """embedding에 NaN/Inf 없는지 확인."""
    from voicesecure.evaluators.adapters import CAMPlusAdapter
    adapter = CAMPlusAdapter()
    audio = _make_audio(200.0)
    emb = adapter.extract_embedding(audio)
    assert np.all(np.isfinite(emb)), "embedding에 NaN 또는 Inf 포함"


@pytest.mark.slow
def test_same_audio_same_embedding():
    """같은 입력 → 같은 embedding (재현성)."""
    from voicesecure.evaluators.adapters import CAMPlusAdapter
    adapter = CAMPlusAdapter()
    audio = _make_audio(200.0)
    emb1 = adapter.extract_embedding(audio)
    emb2 = adapter.extract_embedding(audio)
    assert np.allclose(emb1, emb2), "같은 입력인데 embedding이 다름"


@pytest.mark.slow
def test_different_audio_different_embedding():
    """다른 음성 → 다른 embedding."""
    from voicesecure.evaluators.adapters import CAMPlusAdapter
    adapter = CAMPlusAdapter()
    audio_a = _make_audio(200.0)
    audio_b = _make_audio(800.0)
    emb_a = adapter.extract_embedding(audio_a)
    emb_b = adapter.extract_embedding(audio_b)
    assert not np.allclose(emb_a, emb_b), "다른 음성인데 embedding이 동일"


@pytest.mark.slow
def test_self_cosine_distance_near_zero():
    """같은 음성의 cosine distance는 0에 가까워야 함."""
    from voicesecure.evaluators.adapters import CAMPlusAdapter
    from voicesecure.evaluators.base import cosine_distance
    adapter = CAMPlusAdapter()
    audio = _make_audio(200.0)
    emb = adapter.extract_embedding(audio)
    dist = cosine_distance(emb, emb)
    print(f"\n  self cosine distance: {dist:.6f}  (0.0이어야 함)")
    assert dist < 1e-4, f"self distance가 너무 큼: {dist}"


@pytest.mark.slow
def test_integration_with_speaker_evaluator():
    """WavLM + CAM++ 둘 다 주입해서 SpeakerEvaluator.evaluate() 동작 확인."""
    from voicesecure.evaluators.adapters import CAMPlusAdapter, WavLMSVAdapter
    from voicesecure.evaluators.speaker import SpeakerEvaluator

    wavlm = WavLMSVAdapter()
    cam = CAMPlusAdapter()
    evaluator = SpeakerEvaluator(wavlm_model=wavlm, cam_model=cam)

    original = _make_audio(200.0)
    modified = _make_audio(800.0)
    result = evaluator.evaluate(original, modified)

    print(f"\n  score      : {result.score:.4f}")
    print(f"  raw_metric : {result.raw_metric:.4f}")
    print(f"  wavlm_dist : {result.metadata['wavlm_dist']:.4f}")
    print(f"  cam_dist   : {result.metadata['cam_dist']:.4f}")

    assert 0.0 <= result.score <= 1.0
    assert np.isfinite(result.raw_metric)
    assert "wavlm_dist" in result.metadata
    assert "cam_dist" in result.metadata
