"""
Unit tests for temporal eye-closure and microsleep detection state machine.
"""

import pytest
from app.eye_closure_detector import EyeClosureDetector, EyeClosureState


def test_normal_blink():
    """Test normal short blink (5 frames closed, then open). Should not trigger alarm."""
    detector = EyeClosureDetector(fps=30.0, ear_threshold=0.22, microsleep_duration_seconds=1.5)

    # Eyes open
    res = detector.process_ear(0.28)
    assert res.state == EyeClosureState.NORMAL
    assert res.trigger_alarm is False
    assert res.is_microsleep is False

    # Blink: 5 frames closed (EAR = 0.15)
    for _ in range(5):
        res = detector.process_ear(0.15)
        assert res.state == EyeClosureState.CLOSING
        assert res.trigger_alarm is False
        assert res.is_microsleep is False

    # Eyes open again
    res = detector.process_ear(0.28)
    assert res.state == EyeClosureState.NORMAL
    assert res.trigger_alarm is False
    assert res.is_microsleep is False
    assert detector.total_microsleep_events == 0


def test_short_eye_closure():
    """Test short eye closure (~0.8s = 24 frames at 30 FPS). Should transition to CLOSING but not trigger alarm."""
    detector = EyeClosureDetector(fps=30.0, ear_threshold=0.22, microsleep_duration_seconds=1.5)

    # 24 frames of low EAR (0.8 seconds)
    for i in range(1, 25):
        res = detector.process_ear(0.18)
        assert res.state == EyeClosureState.CLOSING
        assert res.trigger_alarm is False
        assert res.is_microsleep is False
        assert res.closed_frame_count == i

    # Eyes reopen
    res = detector.process_ear(0.30)
    assert res.state == EyeClosureState.NORMAL
    assert res.trigger_alarm is False
    assert detector.total_microsleep_events == 0


def test_prolonged_eye_closure_microsleep():
    """Test prolonged eye closure (>= 1.5s = 45 frames at 30 FPS). Triggers alarm ONCE on frame 45."""
    detector = EyeClosureDetector(fps=30.0, ear_threshold=0.22, microsleep_duration_seconds=1.5)

    # Frames 1 to 44: CLOSING state, no alarm
    for i in range(1, 45):
        res = detector.process_ear(0.15)
        assert res.state == EyeClosureState.CLOSING
        assert res.trigger_alarm is False
        assert res.is_microsleep is False

    # Frame 45 (Exactly 1.5 seconds): Transitions to DROWSY, trigger_alarm = True
    res_45 = detector.process_ear(0.15)
    assert res_45.state == EyeClosureState.DROWSY
    assert res_45.trigger_alarm is True
    assert res_45.is_microsleep is True
    assert detector.total_microsleep_events == 1

    # Frames 46 to 60: Remains DROWSY, but trigger_alarm = False (prevents duplicate alarms)
    for _ in range(15):
        res_subsequent = detector.process_ear(0.15)
        assert res_subsequent.state == EyeClosureState.DROWSY
        assert res_subsequent.trigger_alarm is False
        assert res_subsequent.is_microsleep is True

    # Total microsleep event count remains 1 for this single continuous event
    assert detector.total_microsleep_events == 1


def test_recovery_after_opening_eyes():
    """Test state transition to RECOVERY when eyes reopen after a microsleep event, then back to NORMAL."""
    detector = EyeClosureDetector(fps=30.0, ear_threshold=0.22, microsleep_duration_seconds=1.5)

    # Trigger microsleep (45 frames closed)
    for _ in range(45):
        detector.process_ear(0.15)

    assert detector.state == EyeClosureState.DROWSY

    # Frame 46: Eyes reopen (EAR = 0.28) => State transitions to RECOVERY
    res_recovery = detector.process_ear(0.28)
    assert res_recovery.state == EyeClosureState.RECOVERY
    assert res_recovery.trigger_alarm is False
    assert res_recovery.is_microsleep is False
    assert res_recovery.closed_frame_count == 0

    # Frame 47: Eyes stay open (EAR = 0.28) => State transitions to NORMAL
    res_normal = detector.process_ear(0.28)
    assert res_normal.state == EyeClosureState.NORMAL
    assert res_normal.trigger_alarm is False
    assert res_normal.is_microsleep is False

    # Second microsleep event can now be detected after recovery
    for _ in range(45):
        res_event2 = detector.process_ear(0.15)

    assert res_event2.state == EyeClosureState.DROWSY
    assert detector.total_microsleep_events == 2


def test_invalid_ear_or_missing_face_landmarks():
    """Test that invalid landmark data / missing face tracking (is_valid=False) does NOT trigger false alarms."""
    detector = EyeClosureDetector(fps=30.0, ear_threshold=0.22, microsleep_duration_seconds=1.5)

    # Simulate 50 consecutive frames of invalid face landmarks (e.g. face lost / camera obscured)
    for _ in range(50):
        res = detector.process_ear(avg_ear=0.0, is_valid=False)
        assert res.state == EyeClosureState.NORMAL
        assert res.trigger_alarm is False
        assert res.is_microsleep is False
        assert res.closed_frame_count == 0

    assert detector.total_microsleep_events == 0


def test_custom_fps_configuration():
    """Test microsleep detection at custom FPS (e.g., 60 FPS requires 90 frames for 1.5s)."""
    detector = EyeClosureDetector(fps=60.0, ear_threshold=0.22, microsleep_duration_seconds=1.5)

    # Frames 1 to 89 at 60 FPS (1.483s < 1.5s) -> CLOSING
    for _ in range(89):
        res = detector.process_ear(0.15)
        assert res.state == EyeClosureState.CLOSING
        assert res.trigger_alarm is False

    # Frame 90 at 60 FPS (Exactly 1.5s) -> DROWSY, trigger_alarm = True
    res_90 = detector.process_ear(0.15)
    assert res_90.state == EyeClosureState.DROWSY
    assert res_90.trigger_alarm is True
    assert detector.total_microsleep_events == 1
