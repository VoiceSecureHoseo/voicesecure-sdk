"""FeatureCache unit tests."""

import pytest

from voicesecure.utils.cache import FeatureCache


class DummyEvaluator:
    """테스트용 evaluator."""

    def __init__(self):
        self.call_count = 0

    def precompute(self, audio):
        """호출 횟수 증가 후 dummy feature 반환."""
        self.call_count += 1

        return {
            "feature": 123,
        }


def test_cache_miss():
    """cache miss 시 feature 계산."""
    cache = FeatureCache()

    evaluator = DummyEvaluator()

    result = cache.get_or_compute(
        audio_id="audio_1",
        evaluator=evaluator,
        audio=None,
    )

    assert result["feature"] == 123
    assert evaluator.call_count == 1


def test_cache_hit():
    """같은 audio_id면 캐시 재사용."""
    cache = FeatureCache()

    evaluator = DummyEvaluator()

    cache.get_or_compute(
        audio_id="audio_1",
        evaluator=evaluator,
        audio=None,
    )

    cache.get_or_compute(
        audio_id="audio_1",
        evaluator=evaluator,
        audio=None,
    )

    # 두 번째는 cache hit라 precompute 다시 호출 안 됨
    assert evaluator.call_count == 1


def test_cache_clear():
    """clear() 후 cache 제거."""
    cache = FeatureCache()

    evaluator = DummyEvaluator()

    cache.get_or_compute(
        audio_id="audio_1",
        evaluator=evaluator,
        audio=None,
    )

    cache.clear()

    cache.get_or_compute(
        audio_id="audio_1",
        evaluator=evaluator,
        audio=None,
    )

    # clear 이후 다시 계산됨
    assert evaluator.call_count == 2


class AnotherDummyEvaluator:
    """다른 evaluator를 흉내내는 테스트용 클래스."""

    def __init__(self):
        self.call_count = 0

    def precompute(self, audio):
        """호출 횟수 증가 후 다른 dummy feature 반환."""
        self.call_count += 1

        return {
            "feature": 999,
        }


def test_cache_separates_different_evaluators():
    """같은 audio_id라도 evaluator가 다르면 서로 다른 cache entry를 사용해야 한다."""
    cache = FeatureCache()

    evaluator_a = DummyEvaluator()
    evaluator_b = AnotherDummyEvaluator()

    result_a = cache.get_or_compute(
        audio_id="audio_1",
        evaluator=evaluator_a,
        audio=None,
    )

    result_b = cache.get_or_compute(
        audio_id="audio_1",
        evaluator=evaluator_b,
        audio=None,
    )

    assert result_a["feature"] == 123
    assert result_b["feature"] == 999
    assert evaluator_a.call_count == 1
    assert evaluator_b.call_count == 1


def test_cache_invalid_max_size_raises_value_error():
    """max_size가 0 이하이면 ValueError가 발생해야 한다."""
    with pytest.raises(ValueError):
        FeatureCache(max_size=0)


def test_cache_evicts_oldest_entry_when_max_size_exceeded():
    """max_size 초과 시 가장 오래된 cache entry가 제거되어야 한다."""
    cache = FeatureCache(max_size=1)

    evaluator = DummyEvaluator()

    cache.get_or_compute(
        audio_id="audio_1",
        evaluator=evaluator,
        audio=None,
    )

    cache.get_or_compute(
        audio_id="audio_2",
        evaluator=evaluator,
        audio=None,
    )

    cache.get_or_compute(
        audio_id="audio_1",
        evaluator=evaluator,
        audio=None,
    )

    assert evaluator.call_count == 3


def test_cache_clear_specific_evaluator_only():
    """evaluator를 지정하면 해당 evaluator의 cache만 삭제되어야 한다."""
    cache = FeatureCache()

    evaluator_a = DummyEvaluator()
    evaluator_b = AnotherDummyEvaluator()

    cache.get_or_compute(
        audio_id="audio_1",
        evaluator=evaluator_a,
        audio=None,
    )
    cache.get_or_compute(
        audio_id="audio_1",
        evaluator=evaluator_b,
        audio=None,
    )

    cache.clear(evaluator=evaluator_a)

    cache.get_or_compute(
        audio_id="audio_1",
        evaluator=evaluator_a,
        audio=None,
    )
    cache.get_or_compute(
        audio_id="audio_1",
        evaluator=evaluator_b,
        audio=None,
    )

    assert evaluator_a.call_count == 2
    assert evaluator_b.call_count == 1
