"""
Unit tests for Mouth Aspect Ratio (MAR) metrics computation and temporal yawn detection.
"""

import pytest
import numpy as np
from app.mouth_metrics import MouthMetricsCalculator, MARMetrics
from app.yawn_detector import YawnDetector, YawnState, YawnResult
from app.config import MouthLandmarkIndices, YawnConfig


def create_synthetic_mouth_landmarks(
    horizontal_dist: float = 30.0,
    vertical_dist: float = 5.0
) -> np.ndarray:
    """
    Helper function to generate synthetic mouth landmark points (N, 2)
    where landmark indices match MouthLandmarkIndices.INNER_LIP: (61, 82, 13, 291, 14, 87).
    """
    landmarks = np.zeros((350, 2), dtype=np.float32)

    # m1=61 (left corner), m4=291 (right corner)
    # m2=82 (upper left), m6=87 (lower left)
    # m3=13 (upper center), m5=14 (lower center)
    landmarks[61] = [10.0, 50.0]                       # m1
    landmarks[291] = [10.0 + horizontal_dist, 50.0]   # m4

    landmarks[82] = [20.0, 50.0 - vertical_dist / 2.0] # m2
    landmarks[87] = [20.0, 50.0 + vertical_dist / 2.0] # m6

    landmarks[13] = [25.0, 50.0 - vertical_dist / 2.0] # m3
    landmarks[14] = [25.0, 50.0 + vertical_dist / 2.0] # m5

    return landmarks


# ====================================================================
# MAR Metrics Calculator Tests
# ====================================================================

def test_mar_calculation_normal_and_open():
    """Test MAR calculation for closed/normal mouth and open mouth."""
    calculator = MouthMetricsCalculator(mar_threshold=0.60)

    # Closed mouth (vertical=5.0, horizontal=30.0 => MAR = (5+5)/(2*30) = 0.1667)
    closed_landmarks = create_synthetic_mouth_landmarks(horizontal_dist=30.0, vertical_dist=5.0)
    metrics_closed = calculator.compute_metrics(closed_landmarks)
    assert metrics_closed.is_valid is True
    assert pytest.approx(metrics_closed.mar, abs=1e-3) == 0.1667
    assert metrics_closed.mouth_open is False

    # Wide open mouth (vertical=40.0, horizontal=30.0 => MAR = (40+40)/(2*30) = 1.3333)
    open_landmarks = create_synthetic_mouth_landmarks(horizontal_dist=30.0, vertical_dist=40.0)
    metrics_open = calculator.compute_metrics(open_landmarks)
    assert metrics_open.is_valid is True
    assert pytest.approx(metrics_open.mar, abs=1e-3) == 1.3333
    assert metrics_open.mouth_open is True


def test_mar_invalid_landmarks_and_zero_division():
    """Test MAR calculation handling of None, empty landmarks, and zero division."""
    calculator = MouthMetricsCalculator()

    assert calculator.compute_metrics(None).is_valid is False
    assert calculator.compute_metrics(np.array([])).is_valid is False

    # Zero horizontal distance
    zero_h_landmarks = create_synthetic_mouth_landmarks(horizontal_dist=0.0, vertical_dist=10.0)
    metrics_zero = calculator.compute_metrics(zero_h_landmarks)
    assert metrics_zero.is_valid is True
    assert metrics_zero.mar == 0.0
    assert metrics_zero.mouth_open is False


# ====================================================================
# Yawn Detector State Machine Tests (8 Mandatory Scenarios)
# ====================================================================

def test_1_normal_closed_mouth_no_yawn():
    """Test 1: Normal closed mouth -> no yawn."""
    detector = YawnDetector(fps=30.0, mar_threshold=0.60, min_yawn_duration_seconds=1.5)

    for _ in range(60):
        res = detector.process_mar(0.25)
        assert res.state == YawnState.NORMAL
        assert res.new_yawn_event is False
        assert res.total_yawns == 0
        assert res.excessive_yawning is False


def test_2_short_mouth_opening_no_yawn():
    """Test 2: Short mouth opening (< 1.5s, e.g. talking/smiling) -> no yawn."""
    detector = YawnDetector(fps=30.0, mar_threshold=0.60, min_yawn_duration_seconds=1.5)

    # 20 frames open (0.67s < 1.5s)
    for _ in range(20):
        res = detector.process_mar(0.75)
        assert res.state == YawnState.OPENING
        assert res.new_yawn_event is False
        assert res.total_yawns == 0

    # Mouth closes
    res_closed = detector.process_mar(0.20)
    assert res_closed.state == YawnState.NORMAL
    assert res_closed.total_yawns == 0


def test_3_sufficient_mouth_opening_duration_one_yawn():
    """Test 3: Sufficient mouth opening duration (>= 1.5s = 45 frames at 30 FPS) -> exactly one yawn."""
    detector = YawnDetector(fps=30.0, mar_threshold=0.60, min_yawn_duration_seconds=1.5)

    # Frames 1 to 44 -> OPENING state, no yawn yet
    for _ in range(44):
        res = detector.process_mar(0.80)
        assert res.state == YawnState.OPENING
        assert res.new_yawn_event is False
        assert res.total_yawns == 0

    # Frame 45 (Exactly 1.5s) -> YAWNING state, new_yawn_event = True
    res_45 = detector.process_mar(0.80)
    assert res_45.state == YawnState.YAWNING
    assert res_45.new_yawn_event is True
    assert res_45.total_yawns == 1


def test_4_mouth_remains_open_still_one_yawn():
    """Test 4: Mouth remains open for additional frames -> still exactly one yawn."""
    detector = YawnDetector(fps=30.0, mar_threshold=0.60, min_yawn_duration_seconds=1.5)

    # Trigger yawn at frame 45
    for _ in range(45):
        detector.process_mar(0.80)

    assert detector.total_yawns == 1

    # Frames 46 to 75: Mouth stays open
    for _ in range(30):
        res = detector.process_mar(0.80)
        assert res.state == YawnState.YAWNING
        assert res.new_yawn_event is False  # No duplicate yawn events!
        assert res.total_yawns == 1        # Yawn count remains 1


def test_5_mouth_closes_and_opens_again_second_yawn():
    """Test 5: Mouth closes and opens again -> second yawn can be detected."""
    detector = YawnDetector(fps=30.0, mar_threshold=0.60, min_yawn_duration_seconds=1.5)

    # First yawn (45 frames open)
    for _ in range(45):
        detector.process_mar(0.80)
    assert detector.total_yawns == 1

    # Mouth closes (10 frames)
    for _ in range(10):
        detector.process_mar(0.20)

    # Second yawn (45 frames open)
    for i in range(45):
        res = detector.process_mar(0.80)

    assert res.state == YawnState.YAWNING
    assert detector.total_yawns == 2


def test_6_fourth_yawn_resets_count_to_zero_and_clears_warning():
    """Test 6: 3 yawns -> warning active. 4th yawn -> immediately resets count to 0 and clears warning."""
    detector = YawnDetector(
        fps=30.0,
        mar_threshold=0.60,
        min_yawn_duration_seconds=1.5,
        rolling_window_seconds=120.0,
        excessive_yawn_threshold=3
    )

    # Perform 3 yawns
    for _ in range(3):
        for _ in range(45):
            res = detector.process_mar(0.85)
        for _ in range(10):
            res = detector.process_mar(0.20)

    assert res.yawns_in_window == 3
    assert res.excessive_yawning is True

    # 4th Yawn occurs -> resets count to 0 immediately
    for _ in range(45):
        res = detector.process_mar(0.85)

    assert detector.total_yawns == 4
    assert res.yawns_in_window == 0  # Reset to 0 on > 3
    assert res.excessive_yawning is False  # Warning cleared!


def test_7_yawns_separated_beyond_two_minute_period_reset():
    """Test 7: After 2 minutes elapse, period yawn count resets to 0 and starts a new period."""
    detector = YawnDetector(
        fps=30.0,
        mar_threshold=0.60,
        min_yawn_duration_seconds=1.5,
        rolling_window_seconds=120.0,
        excessive_yawn_threshold=3
    )

    # Yawn 1 at t = 1s
    for _ in range(45):
        detector.process_mar(0.85, timestamp=1.0)
    detector.process_mar(0.20, timestamp=2.0)

    # Yawn 2 at t = 10s
    for _ in range(45):
        detector.process_mar(0.85, timestamp=10.0)
    res_2 = detector.process_mar(0.20, timestamp=11.0)
    assert res_2.yawns_in_window == 2

    # Advance timestamp past 2-minute period (t = 125.0s) -> count resets to 0
    res_future = detector.process_mar(0.20, timestamp=125.0)

    assert detector.total_yawns == 2        # Total session yawns remains 2
    assert res_future.yawns_in_window == 0  # Reset to 0 for new 2-minute period
    assert res_future.excessive_yawning is False


def test_8_invalid_landmarks_no_crash():
    """Test 8: Invalid landmarks -> no crash and invalid result."""
    detector = YawnDetector(fps=30.0, mar_threshold=0.60, min_yawn_duration_seconds=1.5)

    for _ in range(50):
        res = detector.process_mar(mar=0.0, is_valid=False)
        assert res.state == YawnState.NORMAL
        assert res.new_yawn_event is False
        assert res.total_yawns == 0
        assert res.excessive_yawning is False


def test_9_exact_3_yawns_triggers_excessive_yawning_and_4th_resets():
    """Test 9: 2 yawns -> excessive_yawning False, 3 yawns -> excessive_yawning True, 4th yawn -> resets to 0."""
    detector = YawnDetector(
        fps=30.0,
        mar_threshold=0.60,
        min_yawn_duration_seconds=1.5,
        rolling_window_seconds=120.0,
        excessive_yawn_threshold=3
    )

    # 1st Yawn
    for _ in range(45):
        detector.process_mar(0.85, timestamp=1.0)
    detector.process_mar(0.20, timestamp=2.0)

    # 2nd Yawn
    for _ in range(45):
        detector.process_mar(0.85, timestamp=10.0)
    res2 = detector.process_mar(0.20, timestamp=11.0)
    assert res2.yawns_in_window == 2
    assert res2.excessive_yawning is False  # 2 yawns < 3 threshold

    # 3rd Yawn
    for _ in range(45):
        detector.process_mar(0.85, timestamp=20.0)
    res3 = detector.process_mar(0.20, timestamp=21.0)
    assert res3.yawns_in_window == 3
    assert res3.excessive_yawning is True  # Exactly 3 yawns >= 3 threshold -> True

    # 4th Yawn -> Resets count to 0 and clears warning
    for _ in range(45):
        res4 = detector.process_mar(0.85, timestamp=30.0)
    assert res4.yawns_in_window == 0
    assert res4.excessive_yawning is False


