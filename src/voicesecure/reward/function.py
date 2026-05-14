"""Reward function module.

Evaluator들이 산출한 SV/TTS/ASR 점수를 하나의 scalar reward로 변환한다.
"""

from voicesecure.types import Reward, RewardComponents


class RewardFunction:
    """평가 결과를 단일 reward 값으로 변환하는 클래스.

    Reward 공식:
        R = alpha * sv_score
            + beta * tts_score
            - lambda_asr * max(0, asr_cer - cer_threshold)
    """

    def __init__(
        self,
        alpha: float = 0.6,
        beta: float = 0.4,
        lambda_asr: float = 1.0,
        cer_threshold: float = 0.3,
    ) -> None:
        """Reward 가중치와 ASR penalty 기준값을 초기화한다.

        Args:
            alpha: SV score 가중치.
            beta: TTS score 가중치.
            lambda_asr: ASR CER penalty 가중치.
            cer_threshold: 이 CER 값을 초과한 부분만 penalty로 적용.
        """
        self._validate_non_negative("alpha", alpha)
        self._validate_non_negative("beta", beta)
        self._validate_non_negative("lambda_asr", lambda_asr)
        self._validate_score_range("cer_threshold", cer_threshold)

        self.alpha = alpha
        self.beta = beta
        self.lambda_asr = lambda_asr
        self.cer_threshold = cer_threshold

    def compute(self, components: RewardComponents) -> Reward:
        """Reward를 계산한다.

        Args:
            components: SV/TTS/ASR 평가 점수 묶음.
                - sv_score: 화자 식별 방어 점수. 일반적으로 [0, 1].
                - tts_score: 음성 복제 방어 점수. 일반적으로 [0, 1].
                - asr_cer: ASR Character Error Rate. 일반적으로 [0, 1].

        Returns:
            계산된 scalar reward.

        Raises:
            KeyError: 필요한 component key가 없는 경우.
            ValueError: 점수 값이 허용 범위를 벗어난 경우.
        """
        sv_score = self._get_required_component(components, "sv_score")
        tts_score = self._get_required_component(components, "tts_score")
        asr_cer = self._get_required_component(components, "asr_cer")

        self._validate_score_range("sv_score", sv_score)
        self._validate_score_range("tts_score", tts_score)
        self._validate_score_range("asr_cer", asr_cer)

        # CER가 threshold를 넘는 경우에만 명료성 penalty를 적용한다.
        asr_penalty = max(0.0, asr_cer - self.cer_threshold)

        reward = self.alpha * sv_score + self.beta * tts_score - self.lambda_asr * asr_penalty

        return float(reward)

    @staticmethod
    def _get_required_component(
        components: RewardComponents,
        key: str,
    ) -> float:
        """필수 reward component를 가져온다."""
        if key not in components:
            raise KeyError(f"Missing reward component: {key}")

        return float(components[key])

    @staticmethod
    def _validate_non_negative(
        name: str,
        value: float,
    ) -> None:
        """가중치가 0 이상인지 검증한다."""
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}")

    @staticmethod
    def _validate_score_range(
        name: str,
        value: float,
    ) -> None:
        """점수가 [0, 1] 범위인지 검증한다."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
