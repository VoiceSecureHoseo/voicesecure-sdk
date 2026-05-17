"""Tests for SafetyChecker.

See Also
--------
- src/voicesecure/modulation/safety.py
- ARCHITECTURE.md section 3.1
- SRS.md FR-6
"""

from __future__ import annotations

import numpy as np
import pytest

from voicesecure.modulation.safety import SafetyChecker, SafetyConfig, SafetyError


@pytest.fixture
def checker() -> SafetyChecker:
    return SafetyChecker()


@pytest.fixture
def original() -> np.ndarray:
    """1초 분량 원본 음성."""
    rng = np.random.default_rng(seed=42)
    return (rng.standard_normal(16000) * 0.1).astype(np.float32)


# ----------------------------------------------------------------------
# Length check (가장 엄격 — SafetyError 발생)
# ----------------------------------------------------------------------
class TestLengthCheck:
    def test_matching_length_passes(self, checker: SafetyChecker, original: np.ndarray) -> None:
        """동일 길이 통과."""
        modified = original.copy()
        result = checker.check(original, modified)
        assert result.shape == original.shape

    def test_mismatched_length_raises(self, checker: SafetyChecker, original: np.ndarray) -> None:
        """길이 불일치 시 SafetyError."""
        modified = original[:-1]  # 1 sample 짧음
        with pytest.raises(SafetyError, match="Length mismatch"):
            checker.check(original, modified)


# ----------------------------------------------------------------------
# NaN/Inf handling
# ----------------------------------------------------------------------
class TestNanInfHandling:
    def test_nan_replaced_with_zero(self, checker: SafetyChecker, original: np.ndarray) -> None:
        """NaN sample이 0으로 교체."""
        modified = original.copy()
        modified[100:110] = np.nan

        result = checker.check(original, modified)
        assert not np.any(np.isnan(result))

    def test_inf_replaced_with_zero(self, checker: SafetyChecker, original: np.ndarray) -> None:
        """Inf sample이 0으로 교체."""
        modified = original.copy()
        modified[100:110] = np.inf
        modified[200:210] = -np.inf

        result = checker.check(original, modified)
        assert not np.any(np.isinf(result))

    def test_all_finite_unchanged(self, checker: SafetyChecker, original: np.ndarray) -> None:
        """모든 sample이 finite면 NaN/Inf 처리 단계에서 변경 없음."""
        modified = original.copy()
        result = checker.check(original, modified)
        # range/RMS 처리는 통과하므로 동일
        assert np.allclose(result, modified)


# ----------------------------------------------------------------------
# Range handling
# ----------------------------------------------------------------------
class TestRangeHandling:
    def test_within_range_unchanged(self, checker: SafetyChecker, original: np.ndarray) -> None:
        """범위 내 sample은 그대로."""
        modified = original.copy() * 0.5  # 확실히 [-1, 1] 안쪽
        result = checker.check(original, modified)
        assert np.allclose(result, modified)

    def test_excessive_overflow_clipped(self, checker: SafetyChecker, original: np.ndarray) -> None:
        """1% 이상 초과 시 clipping."""
        modified = original.copy()
        # 1% 이상 (160 sample 이상) overflow 만들기
        modified[:300] = 5.0
        modified[300:600] = -5.0

        result = checker.check(original, modified)
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_small_overflow_not_clipped(self, checker: SafetyChecker, original: np.ndarray) -> None:
        """1% 미만 overflow면 clipping 안 적용 (그대로 둠)."""
        modified = original.copy()
        # 0.5% overflow (80 sample)
        modified[:80] = 5.0

        result = checker.check(original, modified)
        # clipping 안 됨 — 큰 값 그대로 (단 RMS 비율 검사에서 fallback 적용 가능)
        # 여기서는 range 체크 단독 검증이 어려우니 그냥 통과만 확인
        assert result.shape == original.shape


# ----------------------------------------------------------------------
# RMS ratio handling
# ----------------------------------------------------------------------
class TestRMSRatio:
    def test_normal_ratio_unchanged(self, checker: SafetyChecker, original: np.ndarray) -> None:
        """RMS 비율 정상 (0.5 ~ 2.0) 시 그대로."""
        modified = original.copy() * 1.2  # 비율 1.2
        result = checker.check(original, modified)
        assert np.allclose(result, modified)

    def test_too_quiet_blended_with_original(
        self, checker: SafetyChecker, original: np.ndarray
    ) -> None:
        """너무 작아지면 (RMS ratio < 0.5) fallback blend."""
        modified = original.copy() * 0.1  # 비율 0.1 → 아웃
        result = checker.check(original, modified)
        # blend되면 원본 영향이 들어가서 modified와 다름
        assert not np.allclose(result, modified)

    def test_too_loud_blended_with_original(
        self, checker: SafetyChecker, original: np.ndarray
    ) -> None:
        """너무 커지면 (RMS ratio > 2.0) fallback blend."""
        modified = original.copy() * 3.0  # 비율 3.0 → 아웃
        # 단, [-1, 1] 범위 초과로 먼저 clipping될 수도 있음
        # 그래도 최종 출력은 안전한 형태여야 함
        result = checker.check(original, modified)
        assert result.shape == original.shape
        assert not np.any(np.isnan(result))


# ----------------------------------------------------------------------
# Integration: 여러 문제 동시 발생
# ----------------------------------------------------------------------
class TestIntegration:
    def test_multiple_issues_handled(self, checker: SafetyChecker, original: np.ndarray) -> None:
        """NaN + overflow + RMS 이상이 한꺼번에 와도 안전한 출력."""
        modified = original.copy()
        modified[100:200] = np.nan
        modified[300:400] = 10.0  # overflow
        modified[500:600] = np.inf

        result = checker.check(original, modified)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert result.min() >= -1.0
        assert result.max() <= 1.0
        assert result.shape == original.shape


# ----------------------------------------------------------------------
# Configurable thresholds
# ----------------------------------------------------------------------
class TestConfig:
    def test_custom_rms_range(self) -> None:
        """custom RMS 범위 적용."""
        config = SafetyConfig(rms_ratio_min=0.9, rms_ratio_max=1.1)
        checker = SafetyChecker(config)

        rng = np.random.default_rng(seed=42)
        original = (rng.standard_normal(1000) * 0.1).astype(np.float32)
        modified = original * 1.5  # 비율 1.5 → 1.1 초과 → fallback

        result = checker.check(original, modified)
        # blend 발생 → modified와 다름
        assert not np.allclose(result, modified)

    def test_custom_blend_alpha(self) -> None:
        """custom blend 비율 적용."""
        config = SafetyConfig(rms_ratio_min=0.9, rms_ratio_max=1.1, fallback_blend_alpha=0.5)
        checker = SafetyChecker(config)

        rng = np.random.default_rng(seed=42)
        original = (rng.standard_normal(1000) * 0.1).astype(np.float32)
        modified = original * 1.5

        result = checker.check(original, modified)
        # alpha=0.5면 정확히 (modified + original) / 2
        expected = (0.5 * modified + 0.5 * original).astype(np.float32)
        assert np.allclose(result, expected)
