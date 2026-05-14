"""RewardFunction unit tests."""

import pytest

from voicesecure.reward.function import RewardFunction


def test_reward_without_asr_penalty():
    """CER가 threshold 이하이면 ASR penalty가 적용되지 않아야 한다."""
    reward_fn = RewardFunction()

    reward = reward_fn.compute(
        {
            "sv_score": 0.8,
            "tts_score": 0.5,
            "asr_cer": 0.2,
        }
    )

    expected = 0.6 * 0.8 + 0.4 * 0.5
    assert reward == expected


def test_reward_with_asr_penalty():
    """CER가 threshold를 초과하면 초과분만큼 penalty가 적용되어야 한다."""
    reward_fn = RewardFunction()

    reward = reward_fn.compute(
        {
            "sv_score": 0.8,
            "tts_score": 0.5,
            "asr_cer": 0.5,
        }
    )

    expected = 0.6 * 0.8 + 0.4 * 0.5 - 1.0 * (0.5 - 0.3)
    assert reward == expected


def test_reward_custom_weights():
    """사용자가 지정한 reward 가중치가 계산에 반영되어야 한다."""
    reward_fn = RewardFunction(
        alpha=1.0,
        beta=1.0,
        lambda_asr=2.0,
        cer_threshold=0.25,
    )

    reward = reward_fn.compute(
        {
            "sv_score": 0.4,
            "tts_score": 0.3,
            "asr_cer": 0.5,
        }
    )

    expected = 1.0 * 0.4 + 1.0 * 0.3 - 2.0 * (0.5 - 0.25)
    assert reward == expected


def test_reward_missing_component_raises_key_error():
    """필수 component가 없으면 KeyError가 발생해야 한다."""
    reward_fn = RewardFunction()

    with pytest.raises(KeyError):
        reward_fn.compute(
            {
                "sv_score": 0.8,
                "tts_score": 0.5,
            }
        )


def test_reward_invalid_score_range_raises_value_error():
    """점수가 [0, 1] 범위를 벗어나면 ValueError가 발생해야 한다."""
    reward_fn = RewardFunction()

    with pytest.raises(ValueError):
        reward_fn.compute(
            {
                "sv_score": 1.2,
                "tts_score": 0.5,
                "asr_cer": 0.2,
            }
        )


def test_reward_invalid_weight_raises_value_error():
    """가중치가 음수이면 ValueError가 발생해야 한다."""
    with pytest.raises(ValueError):
        RewardFunction(alpha=-0.1)
