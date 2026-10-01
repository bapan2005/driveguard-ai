"""
Unit tests for the MediaPipe Face Landmarker module.
"""

import os
import pytest
import numpy as np
from app.face_landmarker import FaceLandmarkDetector, LandmarkResult, DEFAULT_MODEL_PATH


def test_face_landmarker_initialization():
    """Test that FaceLandmarkDetector initializes and downloads model if missing."""
    detector = FaceLandmarkDetector()
    assert os.path.exists(detector.model_path)
    assert detector.landmarker is not None
    detector.close()


def test_process_empty_or_black_frame():
    """Test process_frame on a black frame with no face."""
    detector = FaceLandmarkDetector()
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.process_frame(black_frame)
    assert result is None  # No face in blank image
    detector.close()


def test_process_invalid_frame():
    """Test process_frame handling of None and empty arrays."""
    detector = FaceLandmarkDetector()
    assert detector.process_frame(None) is None
    assert detector.process_frame(np.array([])) is None
    detector.close()
