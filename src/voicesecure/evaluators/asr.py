"""ASR clarity evaluator using a wav2vec2-xlsr-korean adapter."""

from __future__ import annotations

from typing import Any

from voicesecure.evaluators.base import Evaluator, clip_unit_interval
from voicesecure.types import AudioArray, EvaluatorOutput


def normalize_text(text: str) -> str:
    """Normalize text before CER calculation."""

    return "".join(text.split()).lower()


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Compute character error rate with Levenshtein distance."""

    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    previous = list(range(len(hyp) + 1))
    for i, ref_char in enumerate(ref, start=1):
        current = [i]
        for j, hyp_char in enumerate(hyp, start=1):
            substitution_cost = 0 if ref_char == hyp_char else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1] / len(ref)


class ASREvaluator(Evaluator):
    """Measure intelligibility through ASR CER."""

    def __init__(self, asr_model: Any | None = None, original_text: str | None = None) -> None:
        self._asr_model = asr_model
        self._original_text = original_text

    def precompute(self, original: AudioArray) -> dict[str, Any]:
        """Validate original audio. ASR scoring uses text labels rather than audio features."""

        self.validate_audio(original, name="original")
        return {}

    def evaluate(
        self,
        original: AudioArray,
        modified: AudioArray,
        original_text: str | None = None,
    ) -> EvaluatorOutput:
        """Return ASR clarity score and raw CER for modified audio."""

        self.validate_audio(original, name="original")
        modified = self.validate_audio(modified, name="modified")

        reference_text = original_text if original_text is not None else self._original_text
        if reference_text is None:
            raise ValueError("original_text is required for ASR evaluation.")

        transcription = self.transcribe(
            self._asr_model, modified, model_name="wav2vec2-xlsr-korean"
        )
        cer = character_error_rate(reference_text, transcription)
        score = clip_unit_interval(1.0 - cer)

        return EvaluatorOutput(
            score=score,
            raw_metric=cer,
            metadata={"transcription": transcription, "reference_text": reference_text},
        )

