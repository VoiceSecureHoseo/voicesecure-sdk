"""Wav2Vec2KoreanAdapter 테스트.

@pytest.mark.slow 테스트는 실제 모델 다운로드가 필요하므로
nightly 또는 수동 실행: pytest -m slow tests/test_evaluators/test_adapters_wav2vec2.py
"""

import numpy as np
import pytest

from voicesecure.types import SAMPLE_RATE


def _make_audio(freq: float, duration: float = 1.0) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * duration), dtype=np.float32) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ── import / 클래스 구조 (모델 로드 없이 확인) ──────────────────────────────

def test_adapter_importable():
    from voicesecure.evaluators.adapters import Wav2Vec2KoreanAdapter
    assert Wav2Vec2KoreanAdapter is not None


def test_adapter_has_required_method():
    from voicesecure.evaluators.adapters import Wav2Vec2KoreanAdapter
    assert hasattr(Wav2Vec2KoreanAdapter, "transcribe")


# ── 실제 모델 로드 테스트 (slow) ─────────────────────────────────────────────

@pytest.mark.slow
def test_wav2vec2_loads():
    """모델이 에러 없이 로드되는지 확인."""
    from voicesecure.evaluators.adapters import Wav2Vec2KoreanAdapter
    adapter = Wav2Vec2KoreanAdapter()
    assert adapter.model is not None
    assert adapter.processor is not None


@pytest.mark.slow
def test_transcribe_returns_string():
    """transcribe() 반환값이 str인지 확인."""
    from voicesecure.evaluators.adapters import Wav2Vec2KoreanAdapter
    adapter = Wav2Vec2KoreanAdapter()
    audio = _make_audio(200.0)
    result = adapter.transcribe(audio)

    print(f"\n  transcription: '{result}'")
    print(f"  type: {type(result).__name__}")
    assert isinstance(result, str)


@pytest.mark.slow
def test_transcribe_no_error_on_silence():
    """무음 입력에서도 에러 없이 빈 문자열 또는 텍스트 반환."""
    from voicesecure.evaluators.adapters import Wav2Vec2KoreanAdapter
    adapter = Wav2Vec2KoreanAdapter()
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    result = adapter.transcribe(silence)

    print(f"\n  silence transcription: '{result}'")
    assert isinstance(result, str)


@pytest.mark.slow
def test_transcribe_wrong_shape_raises():
    """2-D 입력은 ValueError."""
    from voicesecure.evaluators.adapters import Wav2Vec2KoreanAdapter
    adapter = Wav2Vec2KoreanAdapter()
    audio_2d = np.zeros((2, SAMPLE_RATE), dtype=np.float32)
    with pytest.raises(ValueError):
        adapter.transcribe(audio_2d)


@pytest.mark.slow
def test_transcribe_float64_auto_cast():
    """float64 입력도 내부에서 float32로 변환되어 정상 동작."""
    from voicesecure.evaluators.adapters import Wav2Vec2KoreanAdapter
    adapter = Wav2Vec2KoreanAdapter()
    audio_f64 = _make_audio(200.0).astype(np.float64)
    result = adapter.transcribe(audio_f64)
    assert isinstance(result, str)


@pytest.mark.slow
def test_same_audio_same_transcription():
    """같은 입력 → 같은 결과 (재현성)."""
    from voicesecure.evaluators.adapters import Wav2Vec2KoreanAdapter
    adapter = Wav2Vec2KoreanAdapter()
    audio = _make_audio(200.0)
    result1 = adapter.transcribe(audio)
    result2 = adapter.transcribe(audio)
    assert result1 == result2, "같은 입력인데 결과가 다름"


@pytest.mark.slow
def test_integration_with_asr_evaluator():
    """ASREvaluator에 실제로 주입해서 evaluate() 동작 확인."""
    from voicesecure.evaluators.adapters import Wav2Vec2KoreanAdapter
    from voicesecure.evaluators.asr import ASREvaluator

    adapter = Wav2Vec2KoreanAdapter()
    evaluator = ASREvaluator(asr_model=adapter, original_text="안녕하세요")

    original = _make_audio(200.0)
    modified = _make_audio(800.0)
    result = evaluator.evaluate(original, modified)

    print(f"\n  score      : {result.score:.4f}")
    print(f"  cer        : {result.raw_metric:.4f}")
    print(f"  transcription: '{result.metadata['transcription']}'")

    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.raw_metric <= 1.0
    assert isinstance(result.metadata["transcription"], str)
