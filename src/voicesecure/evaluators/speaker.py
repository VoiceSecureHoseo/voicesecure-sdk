"""Speaker-verification evaluator using WavLM-SV and CAM++ adapters."""

from __future__ import annotations

from typing import Any

from voicesecure.evaluators.base import Evaluator, Normalizer, clip_unit_interval, cosine_distance
from voicesecure.types import AudioArray, EvaluatorOutput


class SpeakerEvaluator(Evaluator):
    """Measure speaker-embedding distance with WavLM-SV and CAM++."""

    def __init__(
        self,
        wavlm_model: Any | None = None,
        cam_model: Any | None = None,
        normalizer: Normalizer = clip_unit_interval,
    ) -> None:
        self._wavlm_model = wavlm_model
        self._cam_model = cam_model
        self._normalizer = normalizer

    def precompute(self, original: AudioArray) -> dict[str, Any]:
        """Precompute original WavLM-SV and CAM++ embeddings for cache use."""

        original = self.validate_audio(original, name="original")
        return {
            "wavlm_embedding": self.extract_embedding(
                self._wavlm_model, original, model_name="WavLM-SV"
            ),
            "cam_embedding": self.extract_embedding(self._cam_model, original, model_name="CAM++"),
        }

    def evaluate(
        self,
        original: AudioArray,
        modified: AudioArray,
        precomputed_original_features: dict[str, Any] | None = None,
    ) -> EvaluatorOutput:
        """Return normalized speaker distance between original and modified audio.

        If ``precomputed_original_features`` is provided (e.g. fetched from
        :class:`voicesecure.utils.cache.FeatureCache`), the WavLM-SV/CAM++
        embeddings of ``original`` are reused instead of being recomputed.
        Mirrors the same kwarg already exposed by :class:`TTSEvaluator.evaluate`.
        """

        original = self.validate_audio(original, name="original")
        modified = self.validate_audio(modified, name="modified")

        if precomputed_original_features is not None:
            original_features = precomputed_original_features
        else:
            original_features = self.precompute(original)
        wavlm_mod = self.extract_embedding(self._wavlm_model, modified, model_name="WavLM-SV")
        cam_mod = self.extract_embedding(self._cam_model, modified, model_name="CAM++")

        wavlm_dist = cosine_distance(original_features["wavlm_embedding"], wavlm_mod)
        cam_dist = cosine_distance(original_features["cam_embedding"], cam_mod)
        raw_metric = (wavlm_dist + cam_dist) / 2.0

        return EvaluatorOutput(
            score=self._normalizer(raw_metric),
            raw_metric=raw_metric,
            metadata={"wavlm_dist": wavlm_dist, "cam_dist": cam_dist},
        )
