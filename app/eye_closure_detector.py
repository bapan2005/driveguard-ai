"""
Temporal Eye-Closure & Microsleep Detector Module for DriveGuard AI.

Implements a state machine for tracking temporal eye closure and microsleep events:
- NORMAL: Eyes open (EAR >= EAR_THRESHOLD) or invalid frame.
- CLOSING: Eyes closed (EAR < EAR_THRESHOLD) for short duration (< 1.5s).
- DROWSY: Prolonged eye closure (EAR < EAR_THRESHOLD for >= 1.5s). Triggers Microsleep alarm.
- RECOVERY: Reopening eyes following a DROWSY state, resetting alarm triggers before returning to NORMAL.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
from app.config import EyeClosureConfig


class EyeClosureState(Enum):
    NORMAL = "NORMAL"
    CLOSING = "CLOSING"
    DROWSY = "DROWSY"
    RECOVERY = "RECOVERY"


@dataclass
class EyeClosureResult:
    """Output container for frame-by-frame temporal eye closure tracking."""
    state: EyeClosureState
    closed_duration_seconds: float
    closed_frame_count: int
    is_microsleep: bool
    trigger_alarm: bool  # True ONLY on the initial frame entering DROWSY state for an event

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "state": self.state.value,
            "closed_duration_seconds": self.closed_duration_seconds,
            "closed_frame_count": self.closed_frame_count,
            "is_microsleep": self.is_microsleep,
            "trigger_alarm": self.trigger_alarm,
        }


class EyeClosureDetector:
    """
    State machine for temporal eye-closure tracking and microsleep detection.
    """

    def __init__(
        self,
        fps: float = EyeClosureConfig.DEFAULT_FPS,
        ear_threshold: float = EyeClosureConfig.EAR_THRESHOLD,
        microsleep_duration_seconds: float = EyeClosureConfig.MICROSLEEP_DURATION_SECONDS,
    ):
        """
        Initialize EyeClosureDetector.

        Args:
            fps: Target video frame rate (default: 30 FPS).
            ear_threshold: Threshold below which eyes are considered closed (default: 0.22).
            microsleep_duration_seconds: Minimum duration of low EAR to trigger microsleep (default: 1.5s).
        """
        self.fps = fps
        self.ear_threshold = ear_threshold
        self.microsleep_duration_seconds = microsleep_duration_seconds

        self.state = EyeClosureState.NORMAL
        self.closed_frame_count = 0
        self.alarm_triggered_for_current_event = False
        self.total_microsleep_events = 0

    def process_ear(
        self,
        avg_ear: float,
        is_valid: bool = True,
        dt: Optional[float] = None
    ) -> EyeClosureResult:
        """
        Process average EAR for the current frame and update the eye-closure state machine.

        Args:
            avg_ear: Average Eye Aspect Ratio for both eyes.
            is_valid: True if face landmark detection was valid; False if face tracking was lost.
            dt: Optional frame duration in seconds (if None, 1.0 / fps is used).

        Returns:
            EyeClosureResult object summarizing current state, closed duration, and alarm flags.
        """
        frame_dt = dt if (dt is not None and dt > 0) else (1.0 / self.fps)
        trigger_alarm = False

        # Guard against invalid landmark data / missing face tracking
        if not is_valid:
            # Face lost or invalid landmark data - reset closed count and return safe state
            if self.state == EyeClosureState.DROWSY or self.alarm_triggered_for_current_event:
                self.state = EyeClosureState.RECOVERY
                self.alarm_triggered_for_current_event = False
            else:
                self.state = EyeClosureState.NORMAL

            self.closed_frame_count = 0
            return EyeClosureResult(
                state=self.state,
                closed_duration_seconds=0.0,
                closed_frame_count=0,
                is_microsleep=False,
                trigger_alarm=False,
            )

        if avg_ear < self.ear_threshold:
            self.closed_frame_count += 1
            duration = self.closed_frame_count * frame_dt

            if duration >= self.microsleep_duration_seconds:
                if self.state != EyeClosureState.DROWSY:
                    self.state = EyeClosureState.DROWSY
                    self.total_microsleep_events += 1
                    if not self.alarm_triggered_for_current_event:
                        trigger_alarm = True
                        self.alarm_triggered_for_current_event = True
                else:
                    # Already in DROWSY state; avoid repeated alarm triggers for the same event
                    trigger_alarm = False
            else:
                self.state = EyeClosureState.CLOSING
        else:
            # Eyes are open (EAR >= ear_threshold)
            duration = 0.0

            if self.state == EyeClosureState.DROWSY or self.alarm_triggered_for_current_event:
                self.state = EyeClosureState.RECOVERY
                self.closed_frame_count = 0
                self.alarm_triggered_for_current_event = False
            elif self.state == EyeClosureState.RECOVERY:
                self.state = EyeClosureState.NORMAL
                self.closed_frame_count = 0
            else:
                self.state = EyeClosureState.NORMAL
                self.closed_frame_count = 0

        is_microsleep = (self.state == EyeClosureState.DROWSY)

        return EyeClosureResult(
            state=self.state,
            closed_duration_seconds=round(self.closed_frame_count * frame_dt, 3),
            closed_frame_count=self.closed_frame_count,
            is_microsleep=is_microsleep,
            trigger_alarm=trigger_alarm,
        )

    def reset(self) -> None:
        """Reset state machine counters."""
        self.state = EyeClosureState.NORMAL
        self.closed_frame_count = 0
        self.alarm_triggered_for_current_event = False
        self.total_microsleep_events = 0
