"""
Unit tests for Eye Aspect Ratio (EAR) metrics computation.
"""

import pytest
import numpy as np
from app.eye_metrics import EyeMetricsCalculator, EARMetrics
from app.config import EyeLandmarkIndices


def create_synthetic_landmarks_for_eye(
    horizontal_dist: float = 20.0,
    vertical_dist: float = 6.0
) -> np.ndarray:
    """
    Helper function to generate synthetic landmark points (N, 2)
    where landmark indices match EyeLandmarkIndices.
    """
    # Create 400 empty landmark points
    landmarks = np.zeros((400, 2), dtype=np.float32)

    # Left eye: [33, 160, 158, 133, 153, 144]
    # p1=33 (left corner), p4=133 (right corner)
    # p2=160 (top 1), p6=144 (bot 1)
    # p3=158 (top 2), p5=153 (bot 2)
    landmarks[33] = [10.0, 50.0]                      # p1
    landmarks[133] = [10.0 + horizontal_dist, 50.0]  # p4
    landmarks[160] = [15.0, 50.0 - vertical_dist / 2.0]  # p2
    landmarks[144] = [15.0, 50.0 + vertical_dist / 2.0]  # p6
    landmarks[158] = [25.0, 50.0 - vertical_dist / 2.0]  # p3
    landmarks[153] = [25.0, 50.0 + vertical_dist / 2.0]  # p5

    # Right eye: [362, 385, 387, 263, 373, 380]
    landmarks[362] = [100.0, 50.0]                     # p1
    landmarks[263] = [100.0 + horizontal_dist, 50.0] # p4
    landmarks[385] = [105.0, 50.0 - vertical_dist / 2.0] # p2
    landmarks[380] = [105.0, 50.0 + vertical_dist / 2.0] # p6
    landmarks[387] = [115.0, 50.0 - vertical_dist / 2.0] # p3
    landmarks[373] = [115.0, 50.0 + vertical_dist / 2.0] # p5

    return landmarks


def test_open_eye_ear_calculation():
    """Test EAR calculation for an open eye (vertical_dist = 6.0, horizontal_dist = 20.0 => EAR = 0.30)."""
    calculator = EyeMetricsCalculator()
    landmarks = create_synthetic_landmarks_for_eye(horizontal_dist=20.0, vertical_dist=6.0)

    metrics = calculator.compute_metrics(landmarks)

    assert metrics.is_valid is True
    # Expected EAR: (6.0 + 6.0) / (2 * 20.0) = 12.0 / 40.0 = 0.30
    assert pytest.approx(metrics.left_ear, abs=1e-3) == 0.30
    assert pytest.approx(metrics.right_ear, abs=1e-3) == 0.30
    assert pytest.approx(metrics.avg_ear, abs=1e-3) == 0.30


def test_closed_eye_ear_calculation():
    """Test EAR calculation for a closed eye (vertical_dist = 0.0 => EAR = 0.0)."""
    calculator = EyeMetricsCalculator()
    landmarks = create_synthetic_landmarks_for_eye(horizontal_dist=20.0, vertical_dist=0.0)

    metrics = calculator.compute_metrics(landmarks)

    assert metrics.is_valid is True
    assert metrics.left_ear == 0.0
    assert metrics.right_ear == 0.0
    assert metrics.avg_ear == 0.0


def test_independent_asymmetric_eye_ear():
    """Test independent EAR calculation when left eye is closed and right eye is open."""
    calculator = EyeMetricsCalculator()
    landmarks = create_synthetic_landmarks_for_eye(horizontal_dist=20.0, vertical_dist=6.0)

    # Close left eye only: p2=160, p6=144, p3=158, p5=153 set to vertical 0
    landmarks[160] = [15.0, 50.0]
    landmarks[144] = [15.0, 50.0]
    landmarks[158] = [25.0, 50.0]
    landmarks[153] = [25.0, 50.0]

    metrics = calculator.compute_metrics(landmarks)

    assert metrics.is_valid is True
    assert metrics.left_ear == 0.0
    assert pytest.approx(metrics.right_ear, abs=1e-3) == 0.30
    assert pytest.approx(metrics.avg_ear, abs=1e-3) == 0.15


def test_invalid_or_empty_landmarks():
    """Test handling of None or empty landmark arrays."""
    calculator = EyeMetricsCalculator()

    metrics_none = calculator.compute_metrics(None)
    assert metrics_none.is_valid is False
    assert metrics_none.left_ear == 0.0
    assert metrics_none.avg_ear == 0.0

    metrics_empty = calculator.compute_metrics(np.array([]))
    assert metrics_empty.is_valid is False


def test_division_by_zero_safety():
    """Test division by zero safety when horizontal points p1 and p4 coincide."""
    calculator = EyeMetricsCalculator()
    landmarks = create_synthetic_landmarks_for_eye(horizontal_dist=0.0, vertical_dist=5.0)

    metrics = calculator.compute_metrics(landmarks)

    assert metrics.is_valid is True
    assert metrics.left_ear == 0.0
    assert metrics.right_ear == 0.0
