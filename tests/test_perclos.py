"""
Unit tests for PERCLOS (Percentage of Eye Closure) calculator.
"""

import pytest
from app.perclos import PERCLOSCalculator, PERCLOSResult


def test_all_eyes_open_zero_perclos():
    """Test all eyes open (EAR >= 0.22) across full window. PERCLOS should be 0%."""
    calculator = PERCLOSCalculator(fps=30.0, ear_threshold=0.22, window_seconds=60.0, min_valid_duration_seconds=10.0)

    # 60 seconds of open eyes (1800 frames)
    for _ in range(1800):
        res = calculator.update(avg_ear=0.30, is_valid=True)

    assert res.has_enough_data is True
    assert res.perclos_percentage == 0.0
    assert res.closed_seconds == 0.0
    assert pytest.approx(res.total_valid_seconds, abs=0.1) == 60.0


def test_partially_closed_eyes_perclos():
    """Test eyes closed for 15s out of 60s total valid time (15 / 60 = 25% PERCLOS)."""
    calculator = PERCLOSCalculator(fps=30.0, ear_threshold=0.22, window_seconds=60.0, min_valid_duration_seconds=10.0)

    # 45 seconds open (1350 frames)
    for _ in range(1350):
        calculator.update(avg_ear=0.30, is_valid=True)

    # 15 seconds closed (450 frames)
    for _ in range(450):
        res = calculator.update(avg_ear=0.15, is_valid=True)

    assert res.has_enough_data is True
    assert pytest.approx(res.perclos_percentage, abs=0.5) == 25.0
    assert pytest.approx(res.closed_seconds, abs=0.1) == 15.0
    assert pytest.approx(res.total_valid_seconds, abs=0.1) == 60.0


def test_entirely_closed_eyes_100_perclos():
    """Test eyes closed for entire valid window (60s closed). PERCLOS should be 100%."""
    calculator = PERCLOSCalculator(fps=30.0, ear_threshold=0.22, window_seconds=60.0, min_valid_duration_seconds=10.0)

    # 60 seconds closed (1800 frames)
    for _ in range(1800):
        res = calculator.update(avg_ear=0.15, is_valid=True)

    assert res.has_enough_data is True
    assert res.perclos_percentage == 100.0
    assert pytest.approx(res.closed_seconds, abs=0.1) == 60.0
    assert pytest.approx(res.total_valid_seconds, abs=0.1) == 60.0


def test_invalid_missing_samples():
    """Test handling of invalid/missing samples (is_valid=False). Invalid frames must not count as valid observation time."""
    calculator = PERCLOSCalculator(fps=30.0, ear_threshold=0.22, window_seconds=60.0, min_valid_duration_seconds=10.0)

    # 15 seconds of valid open eyes
    for _ in range(450):
        calculator.update(avg_ear=0.30, is_valid=True)

    # 30 seconds of invalid frames (e.g. face lost / camera obscured)
    for _ in range(900):
        res = calculator.update(avg_ear=0.0, is_valid=False)

    assert res.has_enough_data is True
    assert res.perclos_percentage == 0.0
    assert pytest.approx(res.total_valid_seconds, abs=0.1) == 15.0
    assert res.closed_seconds == 0.0


def test_insufficient_observation_data():
    """Test insufficient observation data (< min_valid_duration_seconds). has_enough_data should be False."""
    calculator = PERCLOSCalculator(fps=30.0, ear_threshold=0.22, window_seconds=60.0, min_valid_duration_seconds=10.0)

    # 5 seconds of valid data (< 10s min threshold)
    for _ in range(150):
        res = calculator.update(avg_ear=0.15, is_valid=True)

    assert res.has_enough_data is False  # Insufficient data for reliable PERCLOS
    assert res.perclos_percentage == 100.0
    assert pytest.approx(res.total_valid_seconds, abs=0.1) == 5.0


def test_rolling_window_expiration():
    """Test rolling-window expiration of old samples beyond window_seconds."""
    calculator = PERCLOSCalculator(fps=30.0, ear_threshold=0.22, window_seconds=60.0, min_valid_duration_seconds=10.0)

    # Window 1: 60s of closed eyes (PERCLOS = 100%)
    for _ in range(1800):
        calculator.update(avg_ear=0.15, is_valid=True, dt=1.0/30.0)

    # Window 2: Next 60s of open eyes (pushes old closed samples out of the 60s rolling window)
    for _ in range(1800):
        res = calculator.update(avg_ear=0.30, is_valid=True, dt=1.0/30.0)

    assert res.has_enough_data is True
    assert res.perclos_percentage == 0.0  # Old closed samples expired!
    assert res.closed_seconds == 0.0
    assert pytest.approx(res.total_valid_seconds, abs=0.1) == 60.0


def test_configurable_window_duration():
    """Test custom window duration (e.g. 30 seconds window)."""
    calculator = PERCLOSCalculator(fps=30.0, ear_threshold=0.22, window_seconds=30.0, min_valid_duration_seconds=5.0)

    # 15s open, 15s closed in a 30s window => 50% PERCLOS
    for _ in range(450):
        calculator.update(avg_ear=0.30, is_valid=True)
    for _ in range(450):
        res = calculator.update(avg_ear=0.15, is_valid=True)

    assert res.has_enough_data is True
    assert pytest.approx(res.perclos_percentage, abs=0.5) == 50.0
    assert pytest.approx(res.total_valid_seconds, abs=0.1) == 30.0
    assert pytest.approx(res.closed_seconds, abs=0.1) == 15.0


def test_boundary_cases_and_reset():
    """Test boundary conditions: exact threshold equality, zero valid samples, reset method."""
    calculator = PERCLOSCalculator(fps=30.0, ear_threshold=0.22, window_seconds=60.0, min_valid_duration_seconds=10.0)

    # Zero valid samples
    res_zero = calculator.update(avg_ear=0.0, is_valid=False)
    assert res_zero.perclos_percentage == 0.0
    assert res_zero.has_enough_data is False

    # Exact threshold boundary: EAR == 0.22 (not < 0.22) => considered OPEN
    for _ in range(300):
        res_boundary = calculator.update(avg_ear=0.22, is_valid=True)

    assert res_boundary.perclos_percentage == 0.0

    # Test reset
    calculator.reset()
    assert len(calculator.samples) == 0
    assert calculator.current_time_seconds == 0.0
