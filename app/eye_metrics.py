"""
Eye Aspect Ratio (EAR) Computation Module for DriveGuard AI.

Calculates Eye Aspect Ratio (EAR) for left, right, and average eye metrics
from facial landmark coordinates.

Mathematical Formulation:
-------------------------
The Eye Aspect Ratio (EAR) quantifies eye openness by comparing vertical distances 
between eye eyelids to horizontal distance between eye corners.

Formula for a single eye defined by 6 landmark points [p1, p2, p3, p4, p5, p6]:

    EAR = ( ||p2 - p6|| + ||p3 - p5|| ) / ( 2.0 * ||p1 - p4|| )

Where:
  - p1, p4: Horizontal corner points of the eye (outer and inner corners).
  - p2, p6: First vertical pair of eye eyelid landmarks.
  - p3, p5: Second vertical pair of eye eyelid landmarks.
  - ||p_a - p_b||: Euclidean distance between landmark points p_a and p_b.

When the eye is open, EAR remains relatively constant (~0.25 - 0.35).
When the eye closes during a blink or drowsiness, EAR rapidly drops towards zero (~0.0 - 0.15).
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np
from app.config import EyeLandmarkIndices


@dataclass
class EARMetrics:
    """
    Structured container for calculated Eye Aspect Ratio (EAR) metrics.

    Attributes:
        left_ear: Calculated EAR for the left eye.
        right_ear: Calculated EAR for the right eye.
        avg_ear: Average EAR across both eyes.
        is_valid: True if landmark inputs were valid and computation succeeded.
    """
    left_ear: float
    right_ear: float
    avg_ear: float
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "left_ear": self.left_ear,
            "right_ear": self.right_ear,
            "avg_ear": self.avg_ear,
            "is_valid": self.is_valid,
        }


class EyeMetricsCalculator:
    """Calculates independent and average Eye Aspect Ratio (EAR) from facial landmarks."""

    def __init__(
        self,
        left_eye_indices: Tuple[int, int, int, int, int, int] = EyeLandmarkIndices.LEFT_EYE,
        right_eye_indices: Tuple[int, int, int, int, int, int] = EyeLandmarkIndices.RIGHT_EYE,
    ):
        """
        Initialize EyeMetricsCalculator with landmark index configurations.

        Args:
            left_eye_indices: 6-tuple of landmark indices for the left eye.
            right_eye_indices: 6-tuple of landmark indices for the right eye.
        """
        self.left_eye_indices = left_eye_indices
        self.right_eye_indices = right_eye_indices

    @staticmethod
    def calculate_single_eye_ear(
        landmarks: np.ndarray,
        indices: Tuple[int, int, int, int, int, int]
    ) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for a single eye.

        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

        Args:
            landmarks: NumPy array of shape (N, 2) or (N, 3) containing landmark coordinates.
            indices: 6-tuple (p1, p2, p3, p4, p5, p6) specifying landmark indices.

        Returns:
            Calculated EAR value as a float (0.0 if calculation is invalid or division by zero).
        """
        if landmarks is None or len(landmarks) == 0:
            return 0.0

        p1_idx, p2_idx, p3_idx, p4_idx, p5_idx, p6_idx = indices

        max_index = max(indices)
        if len(landmarks) <= max_index:
            return 0.0

        # Extract 2D or 3D coordinate points (taking first 2 dimensions [x, y] for Euclidean distance)
        p1 = landmarks[p1_idx][:2]
        p2 = landmarks[p2_idx][:2]
        p3 = landmarks[p3_idx][:2]
        p4 = landmarks[p4_idx][:2]
        p5 = landmarks[p5_idx][:2]
        p6 = landmarks[p6_idx][:2]

        # Vertical Euclidean distances
        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)

        # Horizontal Euclidean distance
        h = np.linalg.norm(p1 - p4)

        if h == 0.0:
            return 0.0

        # Compute Eye Aspect Ratio
        ear = (v1 + v2) / (2.0 * h)
        return float(ear)

    def compute_metrics(self, landmarks: np.ndarray) -> EARMetrics:
        """
        Compute left EAR, right EAR, and average EAR from facial landmarks.

        Args:
            landmarks: NumPy array of shape (N, 2) or (N, 3) representing face landmarks.

        Returns:
            EARMetrics object containing left_ear, right_ear, avg_ear, and validity flag.
        """
        if landmarks is None or len(landmarks) == 0:
            return EARMetrics(left_ear=0.0, right_ear=0.0, avg_ear=0.0, is_valid=False)

        left_ear = self.calculate_single_eye_ear(landmarks, self.left_eye_indices)
        right_ear = self.calculate_single_eye_ear(landmarks, self.right_eye_indices)

        avg_ear = (left_ear + right_ear) / 2.0

        return EARMetrics(
            left_ear=round(left_ear, 4),
            right_ear=round(right_ear, 4),
            avg_ear=round(avg_ear, 4),
            is_valid=True
        )
