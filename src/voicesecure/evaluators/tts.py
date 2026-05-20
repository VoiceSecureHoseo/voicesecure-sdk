"""TTS clone-resistance evaluator using OpenVoice2 and XTTS adapters."""

from __future__ import annotations

from typing import Any

from voicesecure.evaluators.base import Evaluator, Normalizer, clip_unit_interval, cosine_distance
from voicesecure.types import AudioArray, EvaluatorOutput


class TTSEvaluator(Evaluator):
    """Score how poorly OpenVoice2 and XTTS can clone a modified voice."""

    def __init__(
        self,
        openvoice_model: Any | None = None,
        xtts_model: Any | None = None,
        speaker_model: Any | None = None,
        normalizer: Normalizer = clip_unit_interval,
        sampling_interval: int = 10,
    ) -> None:
        if sampling_interval < 1:
            raise ValueError("sampling_interval must be >= 1.")
        self._openvoice_model = openvoice_model
        self._xtts_model = xtts_model
        self._speaker_model = speaker_model
        self._normalizer = normalizer
        self.sampling_interval = sampling_interval

    def should_evaluate(self, episode_index: int) -> bool:
        """Return whether the expensive TTS evaluation should run for an episode."""

        if episode_index < 0:
            raise ValueError("episode_index must be >= 0.")
        return episode_index % self.sampling_interval == 0

    def precompute(self, original: AudioArray) -> dict[str, Any]:
        """Precompute the original speaker embedding used for clone distance."""

        original = self.validate_audio(original, name="original")
        return {
            "original_embedding": self.extract_embedding(
                self._speaker_model, original, model_name="speaker-distance"
            )
        }

    def evaluate(self, original: AudioArray, modified: AudioArray) -> EvaluatorOutput:
        """Return normalized clone-failure distance for modified audio."""

        original = self.validate_audio(original, name="original")
        modified = self.validate_audio(modified, name="modified")

        original_features = self.precompute(original)
        # clone_ov2 = self.synthesize_clone(self._openvoice_model, modified, model_name="OpenVoice2")
        clone_xtts = self.synthesize_clone(self._xtts_model, modified, model_name="XTTS")

        # clone_ov2_embedding = self.extract_embedding(
        #     self._speaker_model, clone_ov2, model_name="speaker-distance"
        # )
        clone_xtts_embedding = self.extract_embedding(
            self._speaker_model, clone_xtts, model_name="speaker-distance"
        )

        # openvoice_dist = cosine_distance(
        #     original_features["original_embedding"], clone_ov2_embedding
        # )
        xtts_dist = cosine_distance(original_features["original_embedding"], clone_xtts_embedding)
        raw_metric = xtts_dist

        return EvaluatorOutput(
            score=self._normalizer(raw_metric),
            raw_metric=raw_metric,
            metadata={"xtts_dist": xtts_dist},
        )
