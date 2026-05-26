"""XTTSAdapter 테스트.

XTTS는 현재 train 파이프라인에서 사용하지 않음 (CosyVoice3 사용).
수동 실행: pytest -m slow tests/test_evaluators/test_adapters_xtts.py
"""

import numpy as np
import pytest

from voicesecure.types import SAMPLE_RATE

pytestmark = pytest.mark.slow


def _make_audio(freq: float, duration: float = 3.0) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * duration), dtype=np.float32) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ── import / 클래스 구조 (모델 로드 없이 확인) ──────────────────────────────


def test_adapter_importable():
    from voicesecure.evaluators.adapters import XTTSAdapter

    assert XTTSAdapter is not None


def test_adapter_has_required_method():
    from voicesecure.evaluators.adapters import XTTSAdapter

    assert hasattr(XTTSAdapter, "clone")


# ── 실제 모델 로드 테스트 (slow) ─────────────────────────────────────────────


@pytest.mark.slow
def test_xtts_loads():
    """모델이 에러 없이 로드되는지 확인."""
    from voicesecure.evaluators.adapters import XTTSAdapter

    adapter = XTTSAdapter()
    # 저수준 API 기반 — TTS.api.TTS 대신 Xtts 직접 사용
    assert adapter._model is not None
    assert adapter.model_dir and adapter.model_dir.endswith("xtts_v2")


@pytest.mark.slow
def test_clone_returns_array():
    """clone() 반환값이 1-D float32 numpy array인지 확인."""
    from voicesecure.evaluators.adapters import XTTSAdapter

    adapter = XTTSAdapter()
    audio = _make_audio(200.0)
    result = adapter.clone(audio)

    print(f"\n  cloned shape: {result.shape}, dtype: {result.dtype}")
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert result.dtype == np.float32


@pytest.mark.slow
def test_clone_output_range():
    """clone() 출력이 [-1, 1] 범위인지 확인."""
    from voicesecure.evaluators.adapters import XTTSAdapter

    adapter = XTTSAdapter()
    audio = _make_audio(200.0)
    result = adapter.clone(audio)

    assert np.all(result >= -1.0), f"min={result.min()}"
    assert np.all(result <= 1.0), f"max={result.max()}"


@pytest.mark.slow
def test_clone_no_nan():
    """clone() 출력에 NaN/Inf 없는지 확인."""
    from voicesecure.evaluators.adapters import XTTSAdapter

    adapter = XTTSAdapter()
    audio = _make_audio(200.0)
    result = adapter.clone(audio)
    assert np.all(np.isfinite(result)), "clone 출력에 NaN 또는 Inf 포함"


@pytest.mark.slow
def test_clone_wrong_shape_raises():
    """2-D 입력은 ValueError."""
    from voicesecure.evaluators.adapters import XTTSAdapter

    adapter = XTTSAdapter()
    audio_2d = np.zeros((2, SAMPLE_RATE), dtype=np.float32)
    with pytest.raises(ValueError):
        adapter.clone(audio_2d)


@pytest.mark.slow
def test_integration_with_tts_evaluator():
    """TTSEvaluator에 실제로 주입해서 evaluate() 동작 확인."""
    from voicesecure.evaluators.adapters import WavLMSVAdapter, XTTSAdapter
    from voicesecure.evaluators.tts import TTSEvaluator

    xtts = XTTSAdapter()
    wavlm = WavLMSVAdapter()
    evaluator = TTSEvaluator(xtts_model=xtts, speaker_model=wavlm)  # openvoice_model 생략

    original = _make_audio(200.0)
    modified = _make_audio(800.0)
    result = evaluator.evaluate(original, modified)

    print(f"\n  score      : {result.score:.4f}")
    print(f"  raw_metric : {result.raw_metric:.4f}")
    print(f"  xtts_dist  : {result.metadata['xtts_dist']:.4f}")

    assert 0.0 <= result.score <= 1.0
    assert np.isfinite(result.raw_metric)
    assert "xtts_dist" in result.metadata
