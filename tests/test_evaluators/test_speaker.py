import numpy as np
import pytest

from voicesecure.evaluators import ModelNotConfiguredError, SpeakerEvaluator


class MeanEmbeddingModel:
    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset

    def extract_embedding(self, audio):
        return np.array([float(audio.mean()) + self.offset, 1.0], dtype=np.float32)


def test_speaker_evaluator_computes_average_distance():
    original = np.array([0.2, 0.2, 0.2], dtype=np.float32)
    modified = np.array([0.8, 0.8, 0.8], dtype=np.float32)
    evaluator = SpeakerEvaluator(MeanEmbeddingModel(), MeanEmbeddingModel(offset=0.5))

    result = evaluator.evaluate(original, modified)

    assert 0.0 <= result.score <= 1.0
    assert result.raw_metric == pytest.approx(
        (result.metadata["wavlm_dist"] + result.metadata["cam_dist"]) / 2.0
    )
    assert set(result.metadata) == {"wavlm_dist", "cam_dist"}


def test_speaker_evaluator_requires_model_adapters():
    audio = np.zeros(4, dtype=np.float32)
    evaluator = SpeakerEvaluator()

    with pytest.raises(ModelNotConfiguredError):
        evaluator.evaluate(audio, audio)
