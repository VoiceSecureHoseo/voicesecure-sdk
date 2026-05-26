import numpy as np
import pytest

from voicesecure.evaluators import TTSEvaluator


class CloneModel:
    def __init__(self, gain: float) -> None:
        self.gain = gain

    def clone(self, audio):
        return np.clip(audio * self.gain, -1.0, 1.0).astype(np.float32)


class EnergyEmbeddingModel:
    def extract_embedding(self, audio):
        return np.array([float(np.mean(audio)), float(np.std(audio)) + 1.0], dtype=np.float32)


def test_tts_evaluator_scores_xtts_clone():
    original = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    modified = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    # OpenVoice2 인자가 제거된 새 시그니처: (xtts_model, speaker_model)
    evaluator = TTSEvaluator(CloneModel(0.5), EnergyEmbeddingModel())

    result = evaluator.evaluate(original, modified)

    assert 0.0 <= result.score <= 1.0
    assert result.raw_metric == pytest.approx(result.metadata["xtts_dist"])
    assert set(result.metadata) == {"xtts_dist"}


def test_tts_evaluator_sampling_interval():
    evaluator = TTSEvaluator(CloneModel(1.0), EnergyEmbeddingModel(), sampling_interval=10)

    assert evaluator.should_evaluate(0)
    assert not evaluator.should_evaluate(9)
    assert evaluator.should_evaluate(10)

    with pytest.raises(ValueError):
        TTSEvaluator(sampling_interval=0)
