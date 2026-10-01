import numpy as np

from app.head_pose import (
    HeadPoseEstimator,
    HeadPoseResult,
    HeadPoseDetector,
    HeadPoseState,
)


def make_landmarks(
    nose_x=320.0,
    nose_y=240.0,
    width=640,
    height=480,
):
    """Create synthetic MediaPipe-style (x, y) landmarks."""

    landmarks = np.zeros((468, 2), dtype=np.float64)

    # Required landmarks used by HeadPoseEstimator
    landmarks[1] = [nose_x, nose_y]       # Nose
    landmarks[152] = [320.0, 400.0]       # Chin
    landmarks[33] = [220.0, 200.0]        # Left eye
    landmarks[263] = [420.0, 200.0]       # Right eye
    landmarks[61] = [260.0, 330.0]        # Left mouth
    landmarks[291] = [380.0, 330.0]       # Right mouth

    return landmarks


def test_invalid_landmarks():
    estimator = HeadPoseEstimator()

    result = estimator.estimate_pose(None)

    assert result.is_valid is False


def test_empty_landmarks():
    estimator = HeadPoseEstimator()

    result = estimator.estimate_pose(np.array([]))

    assert result.is_valid is False


def test_insufficient_landmarks():
    estimator = HeadPoseEstimator()

    landmarks = np.zeros((100, 2))

    result = estimator.estimate_pose(landmarks)

    assert result.is_valid is False


def test_valid_landmarks():
    estimator = HeadPoseEstimator()

    landmarks = make_landmarks()

    result = estimator.estimate_pose(
        landmarks,
        frame_size=(640, 480),
    )

    assert result.is_valid is True


def test_pose_values_are_finite():
    estimator = HeadPoseEstimator()

    result = estimator.estimate_pose(
        make_landmarks(),
        frame_size=(640, 480),
    )

    assert np.isfinite(result.pitch)
    assert np.isfinite(result.yaw)
    assert np.isfinite(result.roll)


def test_head_pose_result_to_dict():
    result = HeadPoseResult(
        pitch=10.0,
        yaw=-5.0,
        roll=2.0,
        is_valid=True,
    )

    data = result.to_dict()

    assert data["pitch"] == 10.0
    assert data["yaw"] == -5.0
    assert data["roll"] == 2.0
    assert data["is_valid"] is True


def test_detector_starts_normal():
    detector = HeadPoseDetector()

    pose = HeadPoseResult(
        pitch=0.0,
        yaw=0.0,
        roll=0.0,
        is_valid=True,
    )

    result = detector.process_pose(pose)

    assert result.state == HeadPoseState.NORMAL
    assert result.sustained_downward_pose is False


def test_sustained_downward_pose():
    detector = HeadPoseDetector(
        fps=10.0,
        pitch_down_threshold=15.0,
        sustained_nod_duration_seconds=1.5,
    )

    pose = HeadPoseResult(
        pitch=30.0,
        yaw=0.0,
        roll=0.0,
        is_valid=True,
    )

    result = None

    for _ in range(15):
        result = detector.process_pose(pose)

    assert result.state == HeadPoseState.LOOKING_DOWN
    assert result.sustained_downward_pose is True
    assert result.sustained_down_seconds >= 1.5


def test_left_head_pose():
    detector = HeadPoseDetector(
        fps=10.0,
        yaw_left_threshold=20.0,
    )

    pose = HeadPoseResult(
        pitch=0.0,
        yaw=-30.0,
        roll=0.0,
        is_valid=True,
    )

    result = detector.process_pose(pose)

    assert result.state == HeadPoseState.LOOKING_LEFT


def test_right_head_pose():
    detector = HeadPoseDetector(
        fps=10.0,
        yaw_right_threshold=20.0,
    )

    pose = HeadPoseResult(
        pitch=0.0,
        yaw=30.0,
        roll=0.0,
        is_valid=True,
    )

    result = detector.process_pose(pose)

    assert result.state == HeadPoseState.LOOKING_RIGHT


def test_recovery_after_downward_pose():
    detector = HeadPoseDetector(
        fps=10.0,
        pitch_down_threshold=15.0,
        sustained_nod_duration_seconds=1.0,
    )

    down_pose = HeadPoseResult(
        pitch=30.0,
        yaw=0.0,
        roll=0.0,
        is_valid=True,
    )

    normal_pose = HeadPoseResult(
        pitch=0.0,
        yaw=0.0,
        roll=0.0,
        is_valid=True,
    )

    for _ in range(10):
        detector.process_pose(down_pose)

    result = detector.process_pose(normal_pose)

    assert result.state == HeadPoseState.RECOVERY


def test_invalid_pose_is_handled():
    detector = HeadPoseDetector()

    invalid_pose = HeadPoseResult(
        pitch=0.0,
        yaw=0.0,
        roll=0.0,
        is_valid=False,
    )

    result = detector.process_pose(invalid_pose)

    assert result.is_valid is False


def test_detector_reset():
    detector = HeadPoseDetector()

    pose = HeadPoseResult(
        pitch=30.0,
        yaw=0.0,
        roll=0.0,
        is_valid=True,
    )

    detector.process_pose(pose)

    detector.reset()

    assert detector.state == HeadPoseState.NORMAL
    assert detector.down_frame_count == 0


def test_custom_dt():
    detector = HeadPoseDetector(
        pitch_down_threshold=15.0,
        sustained_nod_duration_seconds=1.0,
    )

    pose = HeadPoseResult(
        pitch=30.0,
        yaw=0.0,
        roll=0.0,
        is_valid=True,
    )

    detector.process_pose(pose, dt=0.5)
    result = detector.process_pose(pose, dt=0.5)

    assert result.sustained_down_seconds >= 1.0
    assert result.sustained_downward_pose is True