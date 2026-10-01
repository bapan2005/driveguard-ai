"""
Temporal Yawn Detector Module for DriveGuard AI.

Implements a state machine for temporal mouth aspect ratio tracking, yawn event logging,
and rolling 2-minute excessive yawning analysis.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from app.config import YawnConfig


class YawnState(Enum):
    NORMAL = "NORMAL"
    OPENING = "OPENING"
    YAWNING = "YAWNING"
    RECOVERY = "RECOVERY"


@dataclass
class YawnResult:
    """Output container for frame-by-frame temporal yawn tracking."""
    state: YawnState
    open_duration_seconds: float
    open_frame_count: int
    new_yawn_event: bool          # True ONLY on the single frame a yawn event is confirmed
    total_yawns: int             # Total yawns recorded in session
    yawns_in_window: int         # Yawns recorded within rolling time window
    excessive_yawning: bool      # True if yawns_in_window > excessive_yawn_threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "state": self.state.value,
            "open_duration_seconds": self.open_duration_seconds,
            "open_frame_count": self.open_frame_count,
            "new_yawn_event": self.new_yawn_event,
            "total_yawns": self.total_yawns,
            "yawns_in_window": self.yawns_in_window,
            "excessive_yawning": self.excessive_yawning,
        }


class YawnDetector:
    """
    Temporal yawn detection state machine with 2-minute period-based counting.
    """

    def __init__(
        self,
        fps: float = YawnConfig.DEFAULT_FPS,
        mar_threshold: float = YawnConfig.MAR_THRESHOLD,
        min_yawn_duration_seconds: float = YawnConfig.YAWN_MIN_DURATION_SECONDS,
        rolling_window_seconds: float = YawnConfig.YAWN_ROLLING_WINDOW_SECONDS,
        excessive_yawn_threshold: int = YawnConfig.EXCESSIVE_YAWN_COUNT_THRESHOLD,
    ):
        """
        Initialize YawnDetector.

        Args:
            fps: Video stream frame rate (default: 30 FPS).
            mar_threshold: Mouth Aspect Ratio threshold for open mouth (default: 0.60).
            min_yawn_duration_seconds: Minimum duration of open mouth for a yawn event (default: 1.5s).
            rolling_window_seconds: Duration of yawn counting period in seconds (default: 120.0s / 2 mins).
            excessive_yawn_threshold: Yawn count threshold in counting period to set excessive_yawning flag (default: 3).
        """
        self.fps = fps
        self.mar_threshold = mar_threshold
        self.min_yawn_duration_seconds = min_yawn_duration_seconds
        self.rolling_window_seconds = rolling_window_seconds
        self.excessive_yawn_threshold = excessive_yawn_threshold

        self.state = YawnState.NORMAL
        self.open_frame_count = 0
        self.yawn_event_recorded = False
        self.total_yawns = 0
        self.period_start_time = 0.0
        self.period_yawn_count = 0
        self.current_time_seconds = 0.0

    def process_mar(
        self,
        mar: float,
        is_valid: bool = True,
        timestamp: Optional[float] = None,
        dt: Optional[float] = None
    ) -> YawnResult:
        """
        Process frame MAR and update yawn detection state machine and period-based counter.

        Args:
            mar: Calculated Mouth Aspect Ratio.
            is_valid: True if face landmark detection was valid; False if face tracking was lost.
            timestamp: Optional timestamp in seconds. If None, internal clock increments by dt.
            dt: Optional frame duration in seconds (defaults to 1.0 / fps).

        Returns:
            YawnResult object summarizing state, yawn count, period metrics, and excessive flag.
        """
        frame_dt = dt if (dt is not None and dt > 0) else (1.0 / self.fps)

        if timestamp is not None:
            self.current_time_seconds = timestamp
        else:
            self.current_time_seconds += frame_dt

        # Check if 2-minute counting period has elapsed -> reset count to 0 and start new period
        if (self.current_time_seconds - self.period_start_time) >= self.rolling_window_seconds:
            self.period_yawn_count = 0
            self.period_start_time = self.current_time_seconds

        new_yawn_event = False

        # Guard against invalid landmark data / missing face tracking
        if not is_valid:
            if self.state == YawnState.YAWNING or self.yawn_event_recorded:
                self.state = YawnState.RECOVERY
                self.yawn_event_recorded = False
            else:
                self.state = YawnState.NORMAL

            self.open_frame_count = 0
            excessive_yawning = (self.period_yawn_count >= self.excessive_yawn_threshold)

            return YawnResult(
                state=self.state,
                open_duration_seconds=0.0,
                open_frame_count=0,
                new_yawn_event=False,
                total_yawns=self.total_yawns,
                yawns_in_window=self.period_yawn_count,
                excessive_yawning=excessive_yawning,
            )

        if mar >= self.mar_threshold:
            self.open_frame_count += 1
            duration = self.open_frame_count * frame_dt

            if duration >= self.min_yawn_duration_seconds:
                self.state = YawnState.YAWNING
                if not self.yawn_event_recorded:
                    self.total_yawns += 1
                    self.period_yawn_count += 1
                    self.yawn_event_recorded = True
                    new_yawn_event = True

                    # Rule 6: If yawnCount > 3 (4th yawn), immediately reset count to 0 and start a new period
                    if self.period_yawn_count > self.excessive_yawn_threshold:
                        self.period_yawn_count = 0
                        self.period_start_time = self.current_time_seconds
                else:
                    new_yawn_event = False
            else:
                self.state = YawnState.OPENING
                new_yawn_event = False
        else:
            # Mouth closed (MAR < mar_threshold)
            if self.state == YawnState.YAWNING or self.yawn_event_recorded:
                self.state = YawnState.RECOVERY
            elif self.state == YawnState.RECOVERY:
                self.state = YawnState.NORMAL
            else:
                self.state = YawnState.NORMAL

            self.open_frame_count = 0
            self.yawn_event_recorded = False
            new_yawn_event = False

        excessive_yawning = (self.period_yawn_count >= self.excessive_yawn_threshold)

        return YawnResult(
            state=self.state,
            open_duration_seconds=round(self.open_frame_count * frame_dt, 3),
            open_frame_count=self.open_frame_count,
            new_yawn_event=new_yawn_event,
            total_yawns=self.total_yawns,
            yawns_in_window=self.period_yawn_count,
            excessive_yawning=excessive_yawning,
        )

    def reset(self) -> None:
        """Reset state machine counters and period timer."""
        self.state = YawnState.NORMAL
        self.open_frame_count = 0
        self.yawn_event_recorded = False
        self.total_yawns = 0
        self.period_yawn_count = 0
        self.period_start_time = 0.0
        self.current_time_seconds = 0.0
