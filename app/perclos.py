"""
PERCLOS (Percentage of Eye Closure) Calculator Module for DriveGuard AI.

Computes the percentage of time within a rolling time window during which the
driver's eyes are closed (EAR < EAR_THRESHOLD).
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from app.config import PERCLOSConfig


@dataclass
class SampleRecord:
    """Store one valid observation interval."""
    timestamp: float
    duration: float
    is_closed: bool


@dataclass
class PERCLOSResult:
    """Result returned by the PERCLOS calculator."""

    perclos_percentage: float
    total_valid_seconds: float
    closed_seconds: float
    has_enough_data: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a dictionary."""
        return {
            "perclos_percentage": self.perclos_percentage,
            "total_valid_seconds": self.total_valid_seconds,
            "closed_seconds": self.closed_seconds,
            "has_enough_data": self.has_enough_data,
        }


class PERCLOSCalculator:
    """
    Calculates PERCLOS using a rolling time window.

    PERCLOS = (closed-eye valid duration / total valid observation duration) * 100
    """

    def __init__(
        self,
        fps: float = PERCLOSConfig.DEFAULT_FPS,
        ear_threshold: float = PERCLOSConfig.EAR_THRESHOLD,
        window_seconds: float = PERCLOSConfig.WINDOW_SECONDS,
        min_valid_duration_seconds: float = (
            PERCLOSConfig.MIN_VALID_DURATION_SECONDS
        ),
    ):
        self.fps = fps
        self.ear_threshold = ear_threshold
        self.window_seconds = window_seconds
        self.min_valid_duration_seconds = min_valid_duration_seconds

        self.samples: List[SampleRecord] = []
        self.current_time_seconds = 0.0

    def update(
        self,
        avg_ear: float,
        is_valid: bool = True,
        timestamp: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> PERCLOSResult:
        """
        Update PERCLOS using the current frame.

        Args:
            avg_ear: Average Eye Aspect Ratio.
            is_valid: Whether face/eye tracking is valid.
            timestamp: Optional absolute timestamp.
            dt: Frame duration in seconds.

        Returns:
            PERCLOSResult.
        """

        # Determine frame duration.
        frame_dt = (
            dt
            if dt is not None and dt > 0
            else 1.0 / self.fps
        )

        # Update internal time.
        if timestamp is not None:
            self.current_time_seconds = timestamp
        else:
            self.current_time_seconds += frame_dt

        # Add only valid observations.
        if is_valid:
            is_closed = avg_ear < self.ear_threshold

            self.samples.append(
                SampleRecord(
                    timestamp=self.current_time_seconds,
                    duration=frame_dt,
                    is_closed=is_closed,
                )
            )

        # Remove observations outside rolling window.
        cutoff_time = (
            self.current_time_seconds
            - self.window_seconds
            + 1e-5
        )

        self.samples = [
            sample
            for sample in self.samples
            if sample.timestamp >= cutoff_time
        ]

        # Calculate valid observation duration.
        total_valid_seconds = sum(
            sample.duration
            for sample in self.samples
        )

        # Calculate closed-eye duration.
        closed_seconds = sum(
            sample.duration
            for sample in self.samples
            if sample.is_closed
        )

        # Check whether enough data is available.
        has_enough_data = (
            total_valid_seconds
            >= self.min_valid_duration_seconds
        )

        # Calculate PERCLOS.
        if total_valid_seconds > 0:
            perclos_percentage = (
                closed_seconds
                / total_valid_seconds
            ) * 100.0
        else:
            perclos_percentage = 0.0

        return PERCLOSResult(
            perclos_percentage=round(
                perclos_percentage,
                2,
            ),
            total_valid_seconds=round(
                total_valid_seconds,
                3,
            ),
            closed_seconds=round(
                closed_seconds,
                3,
            ),
            has_enough_data=has_enough_data,
        )

    def reset(self) -> None:
        """Reset the PERCLOS calculation."""
        self.samples.clear()
        self.current_time_seconds = 0.0