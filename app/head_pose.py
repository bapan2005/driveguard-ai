"""
Head Pose Estimation & Temporal Head Pose Detector Module for DriveGuard AI.

Estimates 3D head orientation angles (Pitch, Yaw, Roll) from MediaPipe 2D facial landmarks
using OpenCV Perspective-n-Point (solvePnP) geometry, and tracks sustained downward pitch (head nodding).
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np
from app.config import HeadPoseConfig


class HeadPoseState(Enum):
    NORMAL = "NORMAL"
    LOOKING_DOWN = "LOOKING_DOWN"
    LOOKING_LEFT = "LOOKING_LEFT"
    LOOKING_RIGHT = "LOOKING_RIGHT"
    RECOVERY = "RECOVERY"


@dataclass
class HeadPoseResult:
    """
    Container for 3D Head Pose orientation angles.

    Attributes:
        pitch: Pitch angle in degrees (positive = looking down, negative = looking up).
        yaw: Yaw angle in degrees (positive = looking right, negative = looking left).
        roll: Roll angle in degrees (head tilt).
        is_valid: True if PnP solve succeeded; False if landmarks invalid/missing.
    """
    pitch: float
    yaw: float
    roll: float
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "pitch": self.pitch,
            "yaw": self.yaw,
            "roll": self.roll,
            "is_valid": self.is_valid,
        }


@dataclass
class HeadPoseDetectorResult:
    """
    Output container for temporal head pose tracking and sustained downward pose evidence.

    Attributes:
        state: HeadPoseState enum value.
        pitch: Pitch angle in degrees.
        yaw: Yaw angle in degrees.
        roll: Roll angle in degrees.
        sustained_down_seconds: Duration of continuous downward pitch in seconds.
        sustained_downward_pose: True if sustained_down_seconds >= sustained_nod_duration_seconds.
        is_valid: True if input tracking is valid.
    """
    state: HeadPoseState
    pitch: float
    yaw: float
    roll: float
    sustained_down_seconds: float
    sustained_downward_pose: bool
    is_valid: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "state": self.state.value,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "roll": self.roll,
            "sustained_down_seconds": self.sustained_down_seconds,
            "sustained_downward_pose": self.sustained_downward_pose,
            "is_valid": self.is_valid,
        }


class HeadPoseEstimator:
    """Estimates head orientation (Pitch, Yaw, Roll) using OpenCV solvePnP."""

    # Standard 3D facial model points (in mm)
    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),          # Nose tip (landmark 1)
        (0.0, -330.0, -65.0),     # Chin (landmark 152)
        (-225.0, 170.0, -135.0),  # Left eye outer corner (landmark 33)
        (225.0, 170.0, -135.0),   # Right eye outer corner (landmark 263)
        (-150.0, -150.0, -125.0), # Left mouth corner (landmark 61)
        (150.0, -150.0, -125.0),  # Right mouth corner (landmark 291)
    ], dtype=np.float64)

    # Corresponding MediaPipe Face Mesh landmark indices
    LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

    def __init__(self, frame_size: Tuple[int, int] = (640, 480)):
        """
        Initialize HeadPoseEstimator.

        Args:
            frame_size: (width, height) frame dimensions.
        """
        self.frame_width, self.frame_height = frame_size

    def estimate_pose(
        self,
        landmarks: np.ndarray,
        frame_size: Optional[Tuple[int, int]] = None
    ) -> HeadPoseResult:
        """
        Estimate pitch, yaw, and roll angles in degrees from facial landmarks.

        Args:
            landmarks: NumPy array of shape (N, 2) or (N, 3) containing landmark coordinates.
            frame_size: Optional (width, height) tuple to override default frame size.

        Returns:
            HeadPoseResult object containing pitch, yaw, roll, and validity flag.
        """
        if landmarks is None or len(landmarks) == 0:
            return HeadPoseResult(pitch=0.0, yaw=0.0, roll=0.0, is_valid=False)

        max_idx = max(self.LANDMARK_INDICES)
        if len(landmarks) <= max_idx:
            return HeadPoseResult(pitch=0.0, yaw=0.0, roll=0.0, is_valid=False)

        w, h = frame_size if frame_size is not None else (self.frame_width, self.frame_height)

        # Extract 2D image coordinates (converting normalized coordinates if values are <= 1.0)
        image_points = []
        for idx in self.LANDMARK_INDICES:
            pt = landmarks[idx]
            x = pt[0] * w if pt[0] <= 1.0 else pt[0]
            y = pt[1] * h if pt[1] <= 1.0 else pt[1]
            image_points.append([x, y])

        image_points = np.array(image_points, dtype=np.float64)

        # Camera Intrinsic Matrix approximation
        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        try:
            success, rvec, tvec = cv2.solvePnP(
                self.MODEL_POINTS,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return HeadPoseResult(pitch=0.0, yaw=0.0, roll=0.0, is_valid=False)

            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            # Extract Euler angles (pitch, yaw, roll) in degrees
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

            pitch = angles[0]
            yaw = angles[1]
            roll = angles[2]

            return HeadPoseResult(
                pitch=round(float(pitch), 2),
                yaw=round(float(yaw), 2),
                roll=round(float(roll), 2),
                is_valid=True
            )

        except Exception:
            return HeadPoseResult(pitch=0.0, yaw=0.0, roll=0.0, is_valid=False)


class HeadPoseDetector:
    """
    Temporal head pose state machine for detecting sustained head drooping/nodding.
    """

    def __init__(
        self,
        fps: float = HeadPoseConfig.DEFAULT_FPS,
        pitch_down_threshold: float = HeadPoseConfig.PITCH_DOWN_THRESHOLD,
        yaw_left_threshold: float = HeadPoseConfig.YAW_LEFT_THRESHOLD,
        yaw_right_threshold: float = HeadPoseConfig.YAW_RIGHT_THRESHOLD,
        sustained_nod_duration_seconds: float = HeadPoseConfig.SUSTAINED_NOD_DURATION_SECONDS,
    ):
        """
        Initialize HeadPoseDetector.

        Args:
            fps: Video stream frame rate (default: 30 FPS).
            pitch_down_threshold: Pitch angle threshold in degrees for looking down (default: 15.0).
            yaw_left_threshold: Yaw angle threshold in degrees for looking left (default: 20.0).
            yaw_right_threshold: Yaw angle threshold in degrees for looking right (default: 20.0).
            sustained_nod_duration_seconds: Duration of downward pitch required for fatigue evidence (default: 1.5s).
        """
        self.fps = fps
        self.pitch_down_threshold = pitch_down_threshold
        self.yaw_left_threshold = yaw_left_threshold
        self.yaw_right_threshold = yaw_right_threshold
        self.sustained_nod_duration_seconds = sustained_nod_duration_seconds

        self.state = HeadPoseState.NORMAL
        self.down_frame_count = 0

    def process_pose(
        self,
        pose_result: HeadPoseResult,
        dt: Optional[float] = None
    ) -> HeadPoseDetectorResult:
        """
        Process HeadPoseResult for current frame and update state machine.

        Args:
            pose_result: HeadPoseResult object containing pitch, yaw, roll, and validity flag.
            dt: Optional frame interval duration in seconds (defaults to 1.0 / fps).

        Returns:
            HeadPoseDetectorResult object summarizing state, angles, and sustained downward evidence.
        """
        frame_dt = dt if (dt is not None and dt > 0) else (1.0 / self.fps)

        if not pose_result.is_valid:
            if self.state != HeadPoseState.NORMAL:
                self.state = HeadPoseState.RECOVERY
            self.down_frame_count = 0
            return HeadPoseDetectorResult(
                state=self.state,
                pitch=0.0,
                yaw=0.0,
                roll=0.0,
                sustained_down_seconds=0.0,
                sustained_downward_pose=False,
                is_valid=False
            )

        pitch = pose_result.pitch
        yaw = pose_result.yaw
        roll = pose_result.roll

        # Check directional orientation:
        # Looking down condition: pitch > pitch_down_threshold or pitch < -pitch_down_threshold
        is_looking_down = (abs(pitch) >= self.pitch_down_threshold)
        is_looking_left = (yaw <= -abs(self.yaw_left_threshold))
        is_looking_right = (yaw >= abs(self.yaw_right_threshold))

        if is_looking_down:
            self.down_frame_count += 1
            self.state = HeadPoseState.LOOKING_DOWN
        elif is_looking_left:
            self.down_frame_count = 0
            self.state = HeadPoseState.LOOKING_LEFT
        elif is_looking_right:
            self.down_frame_count = 0
            self.state = HeadPoseState.LOOKING_RIGHT
        else:
            if self.state != HeadPoseState.NORMAL and self.state != HeadPoseState.RECOVERY:
                self.state = HeadPoseState.RECOVERY
            elif self.state == HeadPoseState.RECOVERY:
                self.state = HeadPoseState.NORMAL
            else:
                self.state = HeadPoseState.NORMAL

            self.down_frame_count = 0

        sustained_down_seconds = round(self.down_frame_count * frame_dt, 3)
        sustained_downward_pose = (sustained_down_seconds >= self.sustained_nod_duration_seconds)

        return HeadPoseDetectorResult(
            state=self.state,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            sustained_down_seconds=sustained_down_seconds,
            sustained_downward_pose=sustained_downward_pose,
            is_valid=True
        )

    def reset(self) -> None:
        """Reset state machine counters."""
        self.state = HeadPoseState.NORMAL
        self.down_frame_count = 0
