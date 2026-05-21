"""Feature cache module."""

from collections import OrderedDict
from typing import Any

from voicesecure.types import AudioArray


class FeatureCache:
    """Evaluator별 feature를 메모리에 캐싱하는 클래스.

    cache key는 ``(evaluator.__class__.__name__, audio_id)`` 이며,
    evaluator 객체는 duck typing으로만 사용된다 — ``precompute(audio)`` 메서드와
    ``__class__.__name__`` 속성만 있으면 어떤 타입이든 허용한다.
    (Evaluator 추상 클래스를 import하지 않으므로 utils → evaluators 순환 의존이 없다.)
    """

    def __init__(self, max_size: int = 10000) -> None:
        """캐시 초기화.

        Args:
            max_size: 캐시에 저장할 최대 entry 수.
        """
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")

        self._cache: OrderedDict[tuple[str, str], dict] = OrderedDict()
        self._max_size = max_size

    def get_or_compute(
        self,
        audio_id: str,
        evaluator: Any,
        audio: AudioArray,
    ) -> dict:
        """캐시된 feature 반환 또는 새로 계산.

        Args:
            audio_id: 오디오 고유 ID.
            evaluator: precompute(audio) 메서드를 가진 evaluator (duck typed).
            audio: 원본 오디오.

        Returns:
            evaluator.precompute(audio)의 결과 dict.
        """
        cache_key = self._make_cache_key(audio_id, evaluator)

        # cache hit
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        # cache miss → 새 feature 계산
        features = evaluator.precompute(audio)

        # evaluator별/audio별 결과 저장
        self._cache[cache_key] = features
        self._cache.move_to_end(cache_key)

        # max_size 초과 시 가장 오래된 entry 제거
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

        return features

    def clear(self, evaluator: Any | None = None) -> None:
        """캐시 삭제. evaluator 지정 시 해당 evaluator만 삭제."""
        if evaluator is None:
            self._cache.clear()
            return

        evaluator_name = evaluator.__class__.__name__
        keys_to_remove = [key for key in self._cache if key[0] == evaluator_name]

        for key in keys_to_remove:
            del self._cache[key]

    @staticmethod
    def _make_cache_key(
        audio_id: str,
        evaluator: Any,
    ) -> tuple[str, str]:
        """evaluator 이름과 audio_id를 조합해 cache key를 만든다."""
        evaluator_name = evaluator.__class__.__name__

        return evaluator_name, audio_id
