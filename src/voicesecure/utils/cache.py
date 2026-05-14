"""Feature cache module."""

# evaluators/base.py 구현 전까지 임시 Any 사용
# from voicesecure.evaluators.base import Evaluator
from typing import Any

Evaluator = Any


class FeatureCache:
    """Evaluator별 feature를 메모리에 캐싱하는 클래스."""

    def __init__(self) -> None:
        """빈 캐시 초기화."""
        self._cache: dict[tuple[str, str], dict] = {}

    def get_or_compute(
        self,
        audio_id: str,
        evaluator: Evaluator,
        audio: Any,
    ) -> dict:
        """캐시된 feature 반환 또는 새로 계산.

        Args:
            audio_id: 오디오 고유 ID.
            evaluator: precompute(audio) 메서드를 가진 evaluator.
            audio: 원본 오디오.

        Returns:
            evaluator.precompute(audio)의 결과 dict.
        """
        cache_key = self._make_cache_key(audio_id, evaluator)

        # cache hit
        if cache_key in self._cache:
            return self._cache[cache_key]

        # cache miss → 새 feature 계산
        features = evaluator.precompute(audio)

        # evaluator별/audio별 결과 저장
        self._cache[cache_key] = features

        return features

    def clear(self) -> None:
        """캐시 전체 삭제."""
        self._cache.clear()

    @staticmethod
    def _make_cache_key(
        audio_id: str,
        evaluator: Evaluator,
    ) -> tuple[str, str]:
        """evaluator 이름과 audio_id를 조합해 cache key를 만든다."""
        evaluator_name = evaluator.__class__.__name__

        return evaluator_name, audio_id
