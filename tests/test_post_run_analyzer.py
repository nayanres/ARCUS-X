import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from post_run_analyzer import (  # noqa: E402
    _canonicalize_taxonomy,
    _compute_aggregated_metrics,
    _is_ssot_fully_correct,
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
