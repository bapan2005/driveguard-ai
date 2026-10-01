"""
Centralized Configuration & Hyperparameters for DriveGuard AI.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class CameraConfig:
    """Camera Stream Settings."""
    DEVICE_INDEX: int = 0
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    FPS_TARGET: int = 30


@dataclass
class VisionConfig:
    """MediaPipe Face Landmarker & Vision Settings."""
    MAX_NUM_FACES: int = 1
    REFINE_LANDMARKS: bool = True
    MIN_DETECTION_CONFIDENCE: float = 0.5
    MIN_TRACKING_CONFIDENCE: float = 0.5


@dataclass
class EyeLandmarkIndices:
    """
    MediaPipe Face Mesh landmark indices for Eye Aspect Ratio calculation.
    Ordered as 6-tuple: (p1, p2, p3, p4, p5, p6)
    - p1, p4: Horizontal corner points
    - p2, p6: First vertical landmark pair
    - p3, p5: Second vertical landmark pair
    """
    LEFT_EYE: Tuple[int, int, int, int, int, int] = (33, 160, 158, 133, 153, 144)
    RIGHT_EYE: Tuple[int, int, int, int, int, int] = (362, 385, 387, 263, 373, 380)


@dataclass
class EyeClosureConfig:
    """Temporal Eye-Closure & Microsleep Detection Configuration."""
    DEFAULT_FPS: float = 30.0
    EAR_THRESHOLD: float = 0.22
    MICROSLEEP_DURATION_SECONDS: float = 1.5


@dataclass
class PERCLOSConfig:
    """PERCLOS (Percentage of Eye Closure) Calculation Configuration."""
    DEFAULT_FPS: float = 30.0
    EAR_THRESHOLD: float = 0.22
    WINDOW_SECONDS: float = 60.0             # 60-second rolling window
    MIN_VALID_DURATION_SECONDS: float = 10.0 # Minimum valid observation time required


@dataclass
class MouthLandmarkIndices:
    """
    MediaPipe Face Mesh landmark indices for Mouth Aspect Ratio (MAR) calculation.
    Ordered as 6-tuple: (m1, m2, m3, m4, m5, m6)
    - m1, m4: Left and Right mouth corners (61, 291)
    - m2, m6: Upper and Lower lip vertical pair 1 (82, 87)
    - m3, m5: Upper and Lower lip vertical pair 2 (13, 14)
    """
    INNER_LIP: Tuple[int, int, int, int, int, int] = (61, 82, 13, 291, 14, 87)


@dataclass
class YawnConfig:
    """Temporal Yawn Detection & Rolling Window Configuration."""
    DEFAULT_FPS: float = 30.0
    MAR_THRESHOLD: float = 0.60
    YAWN_MIN_DURATION_SECONDS: float = 1.5
    YAWN_ROLLING_WINDOW_SECONDS: float = 120.0  # 2 minutes
    EXCESSIVE_YAWN_COUNT_THRESHOLD: int = 3     # > 3 yawns triggers excessive yawning evidence


@dataclass
class HeadPoseConfig:
    """Head Pose Orientation & Nodding Detection Configuration."""
    DEFAULT_FPS: float = 30.0
    PITCH_DOWN_THRESHOLD: float = 15.0          # Degrees down (head drooping)
    PITCH_UP_THRESHOLD: float = 15.0            # Degrees up
    YAW_LEFT_THRESHOLD: float = 20.0            # Degrees left
    YAW_RIGHT_THRESHOLD: float = 20.0           # Degrees right
    SUSTAINED_NOD_DURATION_SECONDS: float = 1.5 # Duration of downward pitch for fatigue evidence


@dataclass
class ThresholdConfig:
    """Detection Thresholds & Frame Accumulators."""
    # Eye Aspect Ratio (EAR) Thresholds
    EAR_THRESHOLD: float = 0.22
    EAR_CONSEC_FRAMES_DROWSY: int = 45  # 1.5s at 30 FPS
    EAR_CONSEC_FRAMES_WARNING: int = 15 # 0.5s at 30 FPS

    # Mouth Aspect Ratio (MAR) Thresholds
    MAR_THRESHOLD: float = 0.60
    YAWN_CONSEC_FRAMES: int = 45        # 1.5s at 30 FPS

    # Head Pose Thresholds (Degrees)
    PITCH_DOWN_THRESHOLD: float = 15.0
    YAW_TILT_THRESHOLD: float = 20.0

    # Sliding Window Size (Seconds)
    FATIGUE_WINDOW_SECONDS: int = 60


@dataclass
class AlertConfig:
    """Audio & Visual Alert Settings."""
    ENABLE_AUDIO: bool = True
    ALARM_SOUND_PATH: str = "assets/alarm.wav"
    ALARM_COOLDOWN_SECONDS: float = 2.0
    ALERT_COLOR_NORMAL: Tuple[int, int, int] = (0, 255, 0)     # Green (BGR)
    ALERT_COLOR_WARNING: Tuple[int, int, int] = (0, 255, 255)  # Yellow (BGR)
    ALERT_COLOR_DANGER: Tuple[int, int, int] = (0, 0, 255)     # Red (BGR)
