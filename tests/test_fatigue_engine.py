from app.fatigue_engine import (
    FatigueEngine,
    FatigueInput,
    FatigueLevel,
    FatigueWeights,
)


def test_no_fatigue_is_normal():
    engine = FatigueEngine()

    result = engine.evaluate(FatigueInput())

    assert result.score == 0.0
    assert result.level == FatigueLevel.NORMAL
    assert result.reasons == []


def test_microsleep_is_critical():
    engine = FatigueEngine()

    result = engine.evaluate(
        FatigueInput(
            microsleep_detected=True,
        )
    )

    assert result.level == FatigueLevel.CRITICAL
    assert result.score == 40.0
    assert "microsleep" in result.reasons[0].lower()


def test_elevated_perclos_increases_score():
    engine = FatigueEngine()

    result = engine.evaluate(
        FatigueInput(
            perclos_percent=25.0,
            perclos_available=True,
        )
    )

    assert result.score > 0.0
    assert result.contributions["perclos"] > 0.0


def test_low_perclos_has_no_contribution():
    engine = FatigueEngine()

    result = engine.evaluate(
        FatigueInput(
            perclos_percent=5.0,
            perclos_available=True,
        )
    )

    assert result.contributions["perclos"] == 0.0
    assert result.score == 0.0


def test_excessive_yawning_adds_evidence():
    engine = FatigueEngine()

    result = engine.evaluate(
        FatigueInput(
            yawn_count=4,
            excessive_yawning=True,
        )
    )

    assert result.contributions["excessive_yawning"] > 0
    assert result.score > 0


def test_sustained_downward_head_adds_evidence():
    engine = FatigueEngine()

    result = engine.evaluate(
        FatigueInput(
            sustained_downward_pose=True,
        )
    )

    assert result.contributions["sustained_downward_head"] > 0
    assert result.score > 0


def test_sideways_head_adds_evidence():
    engine = FatigueEngine()

    result = engine.evaluate(
        FatigueInput(
            head_looking_left=True,
        )
    )

    assert result.contributions["sideways_head"] > 0
    assert result.score > 0


def test_multiple_signals_combine():
    engine = FatigueEngine()

    result = engine.evaluate(
        FatigueInput(
            perclos_percent=30.0,
            perclos_available=True,
            excessive_yawning=True,
            yawn_count=4,
            sustained_downward_pose=True,
        )
    )

    assert result.score > 40.0
    assert result.level == FatigueLevel.WARNING
    assert len(result.reasons) >= 3


def test_invalid_perclos_is_ignored():
    engine = FatigueEngine()

    result = engine.evaluate(
        FatigueInput(
            perclos_percent=None,
            perclos_available=False,
        )
    )

    assert result.contributions["perclos"] == 0.0


def test_score_is_bounded():
    engine = FatigueEngine()

    result = engine.evaluate(
        FatigueInput(
            microsleep_detected=True,
            perclos_percent=100.0,
            perclos_available=True,
            excessive_yawning=True,
            yawn_count=10,
            sustained_downward_pose=True,
            head_looking_left=True,
        )
    )

    assert 0.0 <= result.score <= 100.0


def test_custom_weights():
    weights = FatigueWeights(
        microsleep=50.0,
        perclos=20.0,
        excessive_yawning=10.0,
        sustained_downward_head=10.0,
        sideways_head=10.0,
    )

    engine = FatigueEngine(weights=weights)

    result = engine.evaluate(
        FatigueInput(
            sustained_downward_pose=True,
        )
    )

    assert result.score == 10.0


def test_to_dict():
    engine = FatigueEngine()

    result = engine.evaluate(
        FatigueInput(
            excessive_yawning=True,
            yawn_count=4,
        )
    )

    data = result.to_dict()

    assert "score" in data
    assert "level" in data
    assert "contributions" in data
    assert "reasons" in data