"""TTS clone-resistance evaluator using an XTTS adapter.

OpenVoice2는 SDK에서 의도적으로 제외됨. 의도된 "공격 시뮬레이션 2종"
중 XTTS만 정식 어댑터(``XTTSAdapter``, 저수준 Xtts API)로 구현되어 있다.
"""

from __future__ import annotations

from typing import Any

from voicesecure.evaluators.base import Evaluator, Normalizer, clip_unit_interval, cosine_distance
from voicesecure.types import AudioArray, EvaluatorOutput


class TTSEvaluator(Evaluator):
    """Score how poorly XTTS can clone a modified voice."""

    def __init__(
        self,
        xtts_model: Any | None = None,
        speaker_model: Any | None = None,
        normalizer: Normalizer = clip_unit_interval,
        sampling_interval: int = 10,
    ) -> None:
        if sampling_interval < 1:
            raise ValueError("sampling_interval must be >= 1.")
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

    def evaluate(
        self,
        original: AudioArray,
        modified: AudioArray,
        precomputed_original_features: dict[str, Any] | None = None,
    ) -> EvaluatorOutput:
        """Return normalized clone-failure distance for modified audio.

        If ``precomputed_original_features`` is provided (e.g. fetched from
        :class:`voicesecure.utils.cache.FeatureCache`), the original speaker
        embedding is reused instead of being recomputed.
        """

        original = self.validate_audio(original, name="original")
        modified = self.validate_audio(modified, name="modified")

        if precomputed_original_features is not None:
            original_features = precomputed_original_features
        else:
            original_features = self.precompute(original)

        clone_xtts = self.synthesize_clone(self._xtts_model, modified, model_name="XTTS")
        clone_xtts_embedding = self.extract_embedding(
            self._speaker_model, clone_xtts, model_name="speaker-distance"
        )

        xtts_dist = cosine_distance(original_features["original_embedding"], clone_xtts_embedding)
        raw_metric = xtts_dist

        return EvaluatorOutput(
            score=self._normalizer(raw_metric),
            raw_metric=raw_metric,
            metadata={"xtts_dist": xtts_dist},
        )
