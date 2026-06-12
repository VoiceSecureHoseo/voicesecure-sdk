"""TTS clone-resistance evaluator using an XTTS adapter.

═══════════════════════════════════════════════════════════════════════════
[V1] 원본 대비 수정사항
═══════════════════════════════════════════════════════════════════════════
1. evaluate()에 text 파라미터 추가 → synthesize_clone()으로 전파.
   - 왜: 기존엔 text 전달 경로가 아예 없어서, CosyVoice 클론이 항상 기본값
     "안녕하세요"로 합성됐음(레퍼런스 오디오의 실제 전사와 불일치 →
     prompt 조건화가 망가져 클론 품질이 perturbation과 무관하게 붕괴 →
     tts reward가 학습 신호로 기능하지 못함).
   - 설계 의도 구현: 원본 녹음과 클론이 같은 문장을 말하게 하여(텍스트 변수
     통제) "원본 화자 임베딩 ↔ 클론 화자 임베딩" 거리가 순수하게 화자 방어
     효과만 반영하도록 함.
   - text=None이면 기존 동작(어댑터 기본 텍스트) 그대로 — 하위 호환 유지.
═══════════════════════════════════════════════════════════════════════════

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
        text: str | None = None,  # [V1 추가] 클론이 말할 텍스트 = 레퍼런스(modified)의 전사
    ) -> EvaluatorOutput:
        """Return normalized clone-failure distance for modified audio.

        If ``precomputed_original_features`` is provided (e.g. fetched from
        :class:`voicesecure.utils.cache.FeatureCache`), the original speaker
        embedding is reused instead of being recomputed.

        [V1] ``text``가 주어지면 클론 합성 시 해당 텍스트를 사용한다
        (CosyVoice 계열은 prompt_text가 레퍼런스 오디오 전사와 일치해야 함).
        """

        original = self.validate_audio(original, name="original")
        modified = self.validate_audio(modified, name="modified")

        if precomputed_original_features is not None:
            original_features = precomputed_original_features
        else:
            original_features = self.precompute(original)

        # [V1 수정] text 전파 — 기존: synthesize_clone(self._xtts_model, modified, model_name="XTTS")
        clone_xtts = self.synthesize_clone(
            self._xtts_model, modified, model_name="XTTS", text=text
        )
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
