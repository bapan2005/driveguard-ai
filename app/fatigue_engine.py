"""
Driver Fatigue Fusion Engine.

Combines independent fatigue indicators into an explainable 0-100
Driver Fatigue Index.

Signals:
- Microsleep / prolonged eye closure
- PERCLOS
- Excessive yawning
- Sustained downward head pose
- Strong sideways head orientation

This is an engineering prototype score, not a clinically validated
medical or safety standard.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from app.eye_closure_detector import (
    EyeClosureDetector,
    EyeClosureState,
    EyeClosureResult,
)


class FatigueLevel(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    HIGH_RISK = "HIGH_RISK"
    CRITICAL = "CRITICAL"


@dataclass
class FatigueWeights:
    """Maximum contribution of each fatigue signal."""

    microsleep: float = 40.0
    perclos: float = 25.0
    excessive_yawning: float = 15.0
    sustained_downward_head: float = 15.0
    sideways_head: float = 5.0

    @property
    def total(self) -> float:
        return (
            self.microsleep
            + self.perclos
            + self.excessive_yawning
            + self.sustained_downward_head
            + self.sideways_head
        )


@dataclass
class FatigueInput:
    """
    Inputs supplied by the existing detection modules.
    """

    microsleep_detected: bool = False

    # PERCLOS percentage, e.g. 18.5 means 18.5%.
    perclos_percent: Optional[float] = None
    perclos_available: bool = False

    # Number of yawns in the configured rolling window.
    yawn_count: int = 0
    excessive_yawning: bool = False

    # Head-pose evidence.
    sustained_downward_pose: bool = False
    head_looking_left: bool = False
    head_looking_right: bool = False


@dataclass
class FatigueResult:
    """Final explainable fatigue assessment."""

    score: float
    level: FatigueLevel
    contributions: Dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "level": self.level.value,
            "contributions": self.contributions,
            "reasons": self.reasons,
        }


class FatigueEngine:
    """
    Combines fatigue evidence into a configurable 0-100 score.

    Default levels:
        0-24   NORMAL
        25-49  WARNING
        50-74  HIGH_RISK
        75-100 CRITICAL
    """

    def __init__(
        self,
        weights: Optional[FatigueWeights] = None,
        warning_threshold: float = 25.0,
        high_risk_threshold: float = 50.0,
        critical_threshold: float = 75.0,
    ):
        self.weights = weights or FatigueWeights()

        if self.weights.total <= 0:
            raise ValueError("At least one fatigue weight must be positive.")

        self.warning_threshold = warning_threshold
        self.high_risk_threshold = high_risk_threshold
        self.critical_threshold = critical_threshold

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _perclos_contribution(self, perclos: float) -> float:
        """
        Convert PERCLOS into a gradual contribution.

        Prototype mapping:
            <= 10%  -> 0
            10-40%  -> gradual increase
            >= 40%  -> maximum contribution
        """
        if perclos <= 10.0:
            return 0.0

        normalized = (perclos - 10.0) / 30.0
        normalized = self._clamp(normalized, 0.0, 1.0)

        return normalized * self.weights.perclos

    def evaluate(self, inputs: FatigueInput) -> FatigueResult:
        contributions: Dict[str, float] = {}
        reasons: list[str] = []

        # -------------------------------------------------
        # 1. Microsleep
        # -------------------------------------------------
        if inputs.microsleep_detected:
            contributions["microsleep"] = self.weights.microsleep
            reasons.append("Prolonged eye closure / microsleep detected")
        else:
            contributions["microsleep"] = 0.0

        # -------------------------------------------------
        # 2. PERCLOS
        # -------------------------------------------------
        if inputs.perclos_available and inputs.perclos_percent is not None:
            perclos = self._clamp(
                float(inputs.perclos_percent),
                0.0,
                100.0,
            )

            contributions["perclos"] = self._perclos_contribution(perclos)

            if perclos > 10.0:
                reasons.append(
                    f"Elevated PERCLOS ({perclos:.1f}%)"
                )
        else:
            contributions["perclos"] = 0.0

        # -------------------------------------------------
        # 3. Excessive yawning
        # -------------------------------------------------
        if inputs.excessive_yawning:
            contributions["excessive_yawning"] = (
                self.weights.excessive_yawning
            )
            reasons.append(
                f"Excessive yawning detected ({inputs.yawn_count} yawns)"
            )
        else:
            contributions["excessive_yawning"] = 0.0

        # -------------------------------------------------
        # 4. Sustained downward head pose
        # -------------------------------------------------
        if inputs.sustained_downward_pose:
            contributions["sustained_downward_head"] = (
                self.weights.sustained_downward_head
            )
            reasons.append("Sustained downward head pose detected")
        else:
            contributions["sustained_downward_head"] = 0.0

        # -------------------------------------------------
        # 5. Sideways head orientation
        # -------------------------------------------------
        if inputs.head_looking_left or inputs.head_looking_right:
            contributions["sideways_head"] = self.weights.sideways_head

            if inputs.head_looking_left:
                reasons.append("Head oriented significantly left")
            else:
                reasons.append("Head oriented significantly right")
        else:
            contributions["sideways_head"] = 0.0

        # -------------------------------------------------
        # Calculate score
        # -------------------------------------------------
        raw_score = sum(contributions.values())

        # Normalize to 0-100 in case custom weights are used.
        score = (raw_score / self.weights.total) * 100.0
        score = self._clamp(score, 0.0, 100.0)

        # -------------------------------------------------
        # Determine level
        # -------------------------------------------------
        if score >= self.critical_threshold:
            level = FatigueLevel.CRITICAL
        elif score >= self.high_risk_threshold:
            level = FatigueLevel.HIGH_RISK
        elif score >= self.warning_threshold:
            level = FatigueLevel.WARNING
        else:
            level = FatigueLevel.NORMAL

        # A confirmed microsleep is treated as critical evidence.
        if inputs.microsleep_detected:
            level = FatigueLevel.CRITICAL

        return FatigueResult(
            score=round(score, 2),
            level=level,
            contributions=contributions,
            reasons=reasons,
        )


# Preserve the existing public exports.
__all__ = [
    "EyeClosureDetector",
    "EyeClosureState",
    "EyeClosureResult",
    "FatigueLevel",
    "FatigueWeights",
    "FatigueInput",
    "FatigueResult",
    "FatigueEngine",
]