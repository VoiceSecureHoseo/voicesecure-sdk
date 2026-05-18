"""Safety check module.

Mixer 출력의 안전성을 검증하고 필요시 보정한다.
NaN/Inf, 범위 초과, RMS 비율 이상, 길이 불일치 등 검사.

See Also
--------
- ARCHITECTURE.md section 3.1: SafetyChecker contract
- SRS.md FR-6: Safety validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from voicesecure.types import AudioArray

logger = logging.getLogger(__name__)


class SafetyError(Exception):
    """복구 불가능한 안전성 위반."""


@dataclass
class SafetyConfig:
    """Safety check thresholds."""

    rms_ratio_min: float = 0.5
    rms_ratio_max: float = 2.0
    clipping_fraction_max: float = 0.01
    fallback_blend_alpha: float = 0.7
    rms_epsilon: float = 1e-10


class SafetyChecker:
    """Mixer 출력의 안전성을 검증하고 필요시 보정.

    검사 항목 (SRS.md FR-6 순서대로):
        1. Length 검증 — 불일치 시 SafetyError (가장 먼저)
        2. NaN/Inf 검출 → 0으로 대체
        3. 범위 [-1, 1] 초과 sample 카운트 → 1% 초과 시 clipping
        4. RMS 비율 (modified / original) 검증 → 범위 이탈 시 fallback blend
        5. Final clipping → 보정 후에도 [-1, 1] 보장

    모든 보정 작업은 WARNING 로그.
    """

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()

    def check(self, original: AudioArray, modified: AudioArray) -> AudioArray:
        """안전성 검사 + 필요시 보정.

        Parameters
        ----------
        original
            원본 음성, shape (num_samples,)
        modified
            Mixer 출력, shape (num_samples,)

        Returns
        -------
        safe_modified
            검증/보정 통과한 음성, shape (num_samples,)

        Raises
        ------
        SafetyError
            length mismatch 등 복구 불가능한 경우
        """
        safe = modified.copy()

        # Step 1: length 검증 (가장 먼저 — 복구 불가)
        if len(safe) != len(original):
            raise SafetyError(f"Length mismatch: original={len(original)}, modified={len(safe)}")

        # Step 2: NaN/Inf 처리
        safe = self._handle_nan_inf(safe)

        # Step 3: 범위 검증
        safe = self._handle_range(safe)

        # Step 4: RMS 비율 검증
        safe = self._handle_rms_ratio(original, safe)

        # Step 5: final safety clipping
        # 위 보정 후에도 [-1, 1] 벗어날 수 있으므로 마지막으로 확실히 자른다.
        safe = np.clip(safe, -1.0, 1.0).astype(np.float32)

        return safe

    def _handle_nan_inf(self, audio: AudioArray) -> AudioArray:
        """NaN/Inf를 0으로 대체."""
        n_bad = int(np.sum(~np.isfinite(audio)))
        if n_bad > 0:
            logger.warning("SafetyChecker: %d non-finite samples → replaced with 0", n_bad)
            audio = np.where(np.isfinite(audio), audio, 0.0).astype(np.float32)
        return audio

    def _handle_range(self, audio: AudioArray) -> AudioArray:
        """[-1, 1] 범위 초과 처리. 1% 초과 시 clipping."""
        out_of_range = int(np.sum(np.abs(audio) > 1.0))
        fraction = out_of_range / max(len(audio), 1)

        if fraction > self.config.clipping_fraction_max:
            logger.warning(
                "SafetyChecker: %.2f%% samples out of [-1, 1] -> clipping applied",
                fraction * 100,
            )
            audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        return audio

    def _handle_rms_ratio(self, original: AudioArray, modified: AudioArray) -> AudioArray:
        """RMS 비율 검증. 범위 이탈 시 fallback blend."""
        rms_orig = float(np.sqrt(np.mean(original**2) + self.config.rms_epsilon))
        rms_mod = float(np.sqrt(np.mean(modified**2) + self.config.rms_epsilon))
        ratio = rms_mod / rms_orig

        if ratio < self.config.rms_ratio_min or ratio > self.config.rms_ratio_max:
            logger.warning(
                "SafetyChecker: RMS ratio %.2f out of [%.2f, %.2f] -> fallback blend",
                ratio,
                self.config.rms_ratio_min,
                self.config.rms_ratio_max,
            )
            alpha = self.config.fallback_blend_alpha
            modified = (alpha * modified + (1 - alpha) * original).astype(np.float32)
        return modified
