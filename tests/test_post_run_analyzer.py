import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from post_run_analyzer import (  # noqa: E402
    _canonicalize_taxonomy,
    _compute_aggregated_metrics,
    _compute_report_metrics,
    _is_ssot_fully_correct,
    _render_report,
)


def test_fully_correct_probe_overrides_stale_failure_label():
    result = {
        "step_accuracy": 1.0,
        "exact_match": False,
        "probe_classification": {
            "trajectory_correct": False,
            "exact_match": False,
        },
    }

    assert _is_ssot_fully_correct(result, result["probe_classification"])
    assert _canonicalize_taxonomy("Horizon Collapse", True) == "None"


def test_aggregated_taxonomy_uses_same_correctness_rule_as_report_metrics():
    results = {
        "1_0.0_0_0_grid": {
            "step_accuracy": 1.0,
            "exact_match": False,
            "continuity_score": 1.0,
            "error_mode": "Horizon Collapse",
            "probe_classification": {
                "valid": True,
                "trajectory_correct": False,
                "exact_match": False,
                "mechanism": "Horizon Failure",
            },
        },
        "2_0.0_0_1_grid": {
            "step_accuracy": 0.5,
            "exact_match": False,
            "continuity_score": 0.5,
            "error_mode": "Unknown",
            "probe_classification": {
                "valid": True,
                "trajectory_correct": False,
                "exact_match": False,
                "mechanism": "Unknown Failure",
            },
        },
    }

    aggregated = _compute_aggregated_metrics(results, [42])

    assert aggregated["failure_analysis"]["None"] == 50.0
    assert aggregated["failure_analysis"]["Unknown / Unmapped"] == 50.0
    assert aggregated["failure_analysis"]["Horizon Collapse"] == 0.0


def test_density_reports_each_horizon_as_share_of_total_probes():
    results = {
        "1_0.0_0_0": {"step_accuracy": 1.0, "probe_classification": {"valid": True}},
        "1_0.0_0_1": {"step_accuracy": 1.0, "probe_classification": {"valid": True}},
        "2_0.0_0_2": {"step_accuracy": 1.0, "probe_classification": {"valid": True}},
        "3_0.0_0_3": {"step_accuracy": 1.0, "probe_classification": {"valid": True}},
    }
    metrics = _compute_report_metrics(results, "test", "COMPLETE", 4, 4, 1.0)

    assert metrics["per_horizon_density"] == {"1": 50.0, "2": 25.0, "3": 25.0}

    data = {"results_matrix": results, "seeds": [42]}
    report = _render_report(data, show_density=True)
    assert "Probe Density by Horizon:" in report
    assert "z=1   : 50.0%" in report
    assert "z=3   : 25.0%" in report
    assert "Probe Density by Horizon:" not in _render_report(data)
