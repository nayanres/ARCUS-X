import pytest

from audit.correction_2026_11.recalculate_sampling_figure import calculate


def test_figure_includes_calibration_but_excludes_invalid_from_accuracy():
    rows = [
        {"model": "ministral-14b-2512", "z": 3, "tier": 1, "gravity": 0.0,
         "valid": True, "step_accuracy": 0.4},
        {"model": "ministral-14b-2512", "z": 20, "tier": 0, "gravity": 0.5,
         "valid": True, "step_accuracy": 0.8},
        {"model": "ministral-14b-2512", "z": 20, "tier": 0, "gravity": 0.5,
         "valid": False, "step_accuracy": None},
    ]
    calibration, later = calculate(rows)
    assert calibration["horizon_z"] == 3
    assert calibration["calibration_observations"] == 1
    assert calibration["step_accuracy_pct_valid"] == pytest.approx(40.0)
    assert later["step_accuracy_pct_valid"] == pytest.approx(80.0)
    assert later["invalid_observations"] == 1
    assert later["probe_density_pct_all_observations"] == pytest.approx(200 / 3)


def test_figure_rejects_invalid_only_horizon():
    with pytest.raises(ValueError, match="undefined conditional accuracy"):
        calculate([{"model": "ministral-14b-2512", "z": 20, "valid": False}])
