"""
Integration test for the live vision pipeline without requiring a physical webcam hardware device.
"""

import pytest
import numpy as np
from app.face_landmarker import FaceLandmarkDetector
from app.eye_metrics import EyeMetricsCalculator, EARMetrics
from app.eye_closure_detector import EyeClosureDetector, EyeClosureState
from app.mouth_metrics import MouthMetricsCalculator, MARMetrics
from app.yawn_detector import YawnDetector, YawnState
from scripts.demo_live_pipeline import draw_hud_overlay


def test_live_pipeline_synthetic_frame_processing():
    """Test full live vision pipeline execution on synthetic image frames."""
    # 1. Initialize all pipeline components
    landmarker = FaceLandmarkDetector(auto_download=True)
    eye_calculator = EyeMetricsCalculator()
    eye_detector = EyeClosureDetector(fps=30.0)
    mouth_calculator = MouthMetricsCalculator()
    yawn_detector = YawnDetector(fps=30.0)

    # 2. Synthetic blank frame (No face detected)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    landmark_result = landmarker.process_frame(frame)
    face_detected = (landmark_result is not None and landmark_result.normalized_landmarks is not None)
    assert face_detected is False

    ear_metrics = EARMetrics(left_ear=0.0, right_ear=0.0, avg_ear=0.0, is_valid=False)
    eye_result = eye_detector.process_ear(avg_ear=0.0, is_valid=False)
    mar_metrics = MARMetrics(mar=0.0, mouth_open=False, is_valid=False)
    yawn_result = yawn_detector.process_mar(mar=0.0, is_valid=False)

    # 3. Test HUD rendering on synthetic frame
    hud_frame = draw_hud_overlay(
        frame=frame,
        fps=30.0,
        face_detected=face_detected,
        ear_metrics=ear_metrics,
        eye_result=eye_result,
        mar_metrics=mar_metrics,
        yawn_result=yawn_result,
    )

    assert hud_frame is not None
    assert hud_frame.shape == (480, 640, 3)
    assert eye_result.state == EyeClosureState.NORMAL
    assert yawn_result.state == YawnState.NORMAL

    landmarker.close()


def test_live_pipeline_synthetic_microsleep_hud_rendering():
    """Test HUD rendering when microsleep alert is active."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    eye_detector = EyeClosureDetector(fps=30.0, ear_threshold=0.22, microsleep_duration_seconds=1.5)

    # Simulate 45 frames of closed eyes to trigger microsleep
    for _ in range(45):
        eye_result = eye_detector.process_ear(avg_ear=0.15, is_valid=True)

    assert eye_result.is_microsleep is True
    assert eye_result.state == EyeClosureState.DROWSY

    ear_metrics = EARMetrics(left_ear=0.15, right_ear=0.15, avg_ear=0.15, is_valid=True)
    mar_metrics = MARMetrics(mar=0.20, mouth_open=False, is_valid=True)
    yawn_detector = YawnDetector(fps=30.0)
    yawn_result = yawn_detector.process_mar(mar=0.20, is_valid=True)

    hud_frame = draw_hud_overlay(
        frame=frame,
        fps=30.0,
        face_detected=True,
        ear_metrics=ear_metrics,
        eye_result=eye_result,
        mar_metrics=mar_metrics,
        yawn_result=yawn_result,
    )

    assert hud_frame is not None
    assert hud_frame.shape == (480, 640, 3)
