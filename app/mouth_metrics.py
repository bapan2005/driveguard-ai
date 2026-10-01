"""
Mouth Aspect Ratio (MAR) Computation Module for DriveGuard AI.

Calculates Mouth Aspect Ratio (MAR) from facial landmark coordinates.

Mathematical Formulation:
-------------------------
The Mouth Aspect Ratio (MAR) measures mouth opening by comparing vertical distances
between inner/outer lip points to horizontal distance between mouth corners.

Formula for mouth landmarks [m1, m2, m3, m4, m5, m6]:

    MAR = ( ||m2 - m6|| + ||m3 - m5|| ) / ( 2.0 * ||m1 - m4|| )

Where:
  - m1, m4: Horizontal mouth corner landmarks (left and right corners).
  - m2, m6: First vertical lip landmark pair.
  - m3, m5: Second vertical lip landmark pair.
  - ||m_a - m_b||: 2D Euclidean distance between landmark points m_a and m_b.

When the mouth is closed or speaking normally, MAR is low (< 0.40 - 0.50).
When yawning, the mouth opens wide and MAR exceeds the threshold (>= 0.60).
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np
from app.config import MouthLandmarkIndices, YawnConfig


@dataclass
class MARMetrics:
    """
    Structured container for calculated Mouth Aspect Ratio (MAR) metrics.

    Attributes:
        mar: Calculated Mouth Aspect Ratio value.
        mouth_open: True if MAR >= mar_threshold.
        is_valid: True if landmark inputs were valid and computation succeeded.
    """
    mar: float
    mouth_open: bool
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "mar": self.mar,
            "mouth_open": self.mouth_open,
            "is_valid": self.is_valid,
        }


class MouthMetricsCalculator:
    """Calculates Mouth Aspect Ratio (MAR) from facial landmarks."""

    def __init__(
        self,
        mouth_indices: Tuple[int, int, int, int, int, int] = MouthLandmarkIndices.INNER_LIP,
        mar_threshold: float = YawnConfig.MAR_THRESHOLD,
    ):
        """
        Initialize MouthMetricsCalculator.

        Args:
            mouth_indices: 6-tuple of landmark indices (m1, m2, m3, m4, m5, m6).
            mar_threshold: Threshold above which mouth is considered open.
        """
        self.mouth_indices = mouth_indices
        self.mar_threshold = mar_threshold

    def calculate_mar(self, landmarks: np.ndarray) -> float:
        """
        Calculate Mouth Aspect Ratio (MAR) for given landmarks.

        MAR = (||m2-m6|| + ||m3-m5||) / (2 * ||m1-m4||)

        Args:
            landmarks: NumPy array of shape (N, 2) or (N, 3) containing landmark coordinates.

        Returns:
            Calculated MAR value as float (0.0 if invalid or division by zero).
        """
        if landmarks is None or len(landmarks) == 0:
            return 0.0

        max_idx = max(self.mouth_indices)
        if len(landmarks) <= max_idx:
            return 0.0

        m1_idx, m2_idx, m3_idx, m4_idx, m5_idx, m6_idx = self.mouth_indices

        m1 = landmarks[m1_idx][:2]
        m2 = landmarks[m2_idx][:2]
        m3 = landmarks[m3_idx][:2]
        m4 = landmarks[m4_idx][:2]
        m5 = landmarks[m5_idx][:2]
        m6 = landmarks[m6_idx][:2]

        v1 = np.linalg.norm(m2 - m6)
        v2 = np.linalg.norm(m3 - m5)
        h = np.linalg.norm(m1 - m4)

        if h == 0.0:
            return 0.0

        mar = (v1 + v2) / (2.0 * h)
        return float(mar)

    def compute_metrics(self, landmarks: np.ndarray) -> MARMetrics:
        """
        Compute MAR, mouth_open state, and validity.

        Args:
            landmarks: NumPy array of shape (N, 2) or (N, 3) representing face landmarks.

        Returns:
            MARMetrics object.
        """
        if landmarks is None or len(landmarks) == 0:
            return MARMetrics(mar=0.0, mouth_open=False, is_valid=False)

        max_idx = max(self.mouth_indices)
        if len(landmarks) <= max_idx:
            return MARMetrics(mar=0.0, mouth_open=False, is_valid=False)

        mar = self.calculate_mar(landmarks)
        mouth_open = (mar >= self.mar_threshold)

        return MARMetrics(
            mar=round(mar, 4),
            mouth_open=mouth_open,
            is_valid=True
        )
