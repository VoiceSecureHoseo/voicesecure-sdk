import numpy as np
import pytest

from voicesecure.evaluators import ASREvaluator, character_error_rate


class ConstantASR:
    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe(self, audio):
        return self.text


def test_character_error_rate_handles_korean_text():
    assert character_error_rate("안녕하세요", "안녕하세오") == pytest.approx(0.2)
    assert character_error_rate("A B", "ab") == pytest.approx(0.0)
    assert character_error_rate("", "x") == pytest.approx(1.0)


def test_asr_evaluator_returns_clarity_score():
    audio = np.zeros(8, dtype=np.float32)
    evaluator = ASREvaluator(ConstantASR("안녕하세오"), original_text="안녕하세요")

    result = evaluator.evaluate(audio, audio)

    assert result.raw_metric == pytest.approx(0.2)
    assert result.score == pytest.approx(0.8)
    assert result.metadata["transcription"] == "안녕하세오"


def test_asr_evaluator_requires_reference_text():
    audio = np.zeros(8, dtype=np.float32)
    evaluator = ASREvaluator(ConstantASR("hello"))

    with pytest.raises(ValueError):
        evaluator.evaluate(audio, audio)

