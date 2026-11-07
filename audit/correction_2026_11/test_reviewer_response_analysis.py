import audit.correction_2026_11.reviewer_response_analysis as analysis
from audit.correction_2026_11.reviewer_response_analysis import (
    CheckpointStore,
    cluster_bootstrap,
    cohort_tier_summary,
    conditional_response_metrics,
    selected_horizon_profiles,
    synthetic_cases,
    summarize_grid_coverage,
)


def test_synthetic_error_injections_match_classifier_signatures():
    checks = synthetic_cases()

    assert len(checks) == 6
    assert all(row["classifier_consistency"] for row in checks)
    fallback = [row for row in checks if row["state_tracking_fallback"]]
    assert [row["synthetic_case"] for row in fallback] == ["unmapped_divergence"]


def test_requested_z_scoring_keeps_complete_reference_and_excludes_initial_state():
    z = 7
    reference = [f"[{x},0]" for x in range(z + 1)]
    scored = conditional_response_metrics(True, reference[:z], reference, z)

    assert scored["full_reference_states"] == z + 1
    assert scored["scored_reference_states"] == z
    assert scored["scored_transitions"] == z - 1
    assert scored["step_accuracy"] == 1.0
    assert scored["requested_length_compliant"] is True


def test_invalid_responses_have_no_accuracy_value():
    z = 7
    reference = [f"[{x},0]" for x in range(z + 1)]
    scored = conditional_response_metrics(False, [], reference, z)

    assert scored["step_accuracy"] is None
    assert scored["early_accuracy"] is None
    assert scored["middle_accuracy"] is None
    assert scored["late_accuracy"] is None
    assert scored["requested_length_compliant"] is None


def test_grid_summary_aggregates_each_horizon_once_and_separates_missing_cells():
    coverage = [
        {
            "horizon_z": 10,
            "coverage_status": "invalid_only",
            "observations": 2,
            "valid_observations": 0,
            "invalid_observations": 2,
        },
        {
            "horizon_z": 10,
            "coverage_status": "missing",
            "observations": 0,
            "valid_observations": 0,
            "invalid_observations": 0,
        },
    ]

    summary = summarize_grid_coverage(coverage)
    horizon_10 = next(row for row in summary if row["horizon_z"] == 10)
    horizon_100 = next(row for row in summary if row["horizon_z"] == 100)

    assert len(summary) == 15
    assert horizon_10["model_tier_cells_observed"] == 1
    assert horizon_10["model_tier_cells_missing"] == 51
    assert horizon_10["observations"] == 2
    assert horizon_10["valid_observations"] == 0
    assert horizon_10["invalid_observations"] == 2
    assert horizon_100["model_tier_cells_observed"] == 0
    assert horizon_100["model_tier_cells_missing"] == 52
    assert horizon_100["observations"] == 0


def test_selected_horizon_profiles_exclude_invalid_accuracy_values():
    rows = [
        {
            "model": "GPT-5-mini",
            "z": 20,
            "tier": 0,
            "gravity": "0.5",
            "valid": True,
            "original_step_accuracy": 0.8,
        },
        {
            "model": "GPT-5-mini",
            "z": 20,
            "tier": 0,
            "gravity": "0.5",
            "valid": False,
            "original_step_accuracy": 0.0,
        },
    ]

    profile = selected_horizon_profiles(rows)

    assert len(profile) == 1
    assert profile[0]["valid_observations"] == 1
    assert profile[0]["invalid_observations"] == 1
    assert profile[0]["mean_valid_step_accuracy"] == 0.8


def test_cohort_tier_summary_is_valid_only_and_reports_tier_zero_contrasts():
    rows = [
        {
            "model": "GPT-5-mini",
            "z": 20,
            "tier": 0,
            "seed": 1,
            "valid": True,
            "step_accuracy": 0.8,
            "taxonomy": "None",
        },
        {
            "model": "GPT-5-mini",
            "z": 20,
            "tier": 0,
            "seed": 1,
            "valid": False,
            "step_accuracy": 0.0,
            "taxonomy": "",
        },
        {
            "model": "GPT-5-mini",
            "z": 20,
            "tier": 1,
            "seed": 1,
            "valid": True,
            "step_accuracy": 0.5,
            "taxonomy": "State Tracking Failure",
        },
    ]

    summary = cohort_tier_summary(rows)
    tier_0 = next(row for row in summary if row["horizon_z"] == 20 and row["tier"] == 0)
    tier_1 = next(row for row in summary if row["horizon_z"] == 20 and row["tier"] == 1)

    assert tier_0["valid_observations"] == 1
    assert tier_0["mean_step_accuracy"] == 0.8
    assert tier_1["valid_observations"] == 1
    assert tier_1["mean_step_accuracy"] == 0.5
    assert abs(tier_1["delta_vs_tier0"] + 0.3) < 1e-12


def test_cluster_bootstrap_matches_resampling_whole_rows(monkeypatch):
    monkeypatch.setattr(analysis, "BOOTSTRAP_REPLICATES", 100)
    rows = [
        {"seed": "1", "valid": True, "step_accuracy": 0.25, "taxonomy": "None"},
        {"seed": "1", "valid": False, "step_accuracy": None, "taxonomy": ""},
        {"seed": "2", "valid": True, "step_accuracy": 0.75, "taxonomy": "Horizon Collapse"},
        {"seed": "2", "valid": True, "step_accuracy": 1.0, "taxonomy": "None"},
    ]
    metric, taxonomy = cluster_bootstrap(rows, "comparison")
    rng = analysis.deterministic_rng("comparison")
    accuracy, invalid, none = [], [], []
    for _ in range(100):
        chosen = [rng.choice(["1", "2"]) for _ in range(2)]
        sample = [row for seed in chosen for row in rows if row["seed"] == seed]
        valid = [row for row in sample if row["valid"]]
        accuracy.append(sum(row["step_accuracy"] for row in valid) / len(valid))
        invalid.append(sum(not row["valid"] for row in sample) / len(sample))
        none.append(sum(row["taxonomy"] == "None" for row in valid) / len(valid))
    assert metric["step_accuracy"] == analysis.percentile_interval(accuracy)
    assert metric["invalid_rate"] == analysis.percentile_interval(invalid)
    assert taxonomy["None"] == analysis.percentile_interval(none)


def test_checkpoint_resume_is_isolated_by_namespace(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.sqlite3"
    store = CheckpointStore(path, "first")
    store.put("stage", {"rows": [1, 2, 3]})
    store.close()
    resumed = CheckpointStore(path, "first")
    assert resumed.get("stage") == {"rows": [1, 2, 3]}
    resumed.close()
    changed = CheckpointStore(path, "changed")
    assert changed.get("stage") is None
    changed.close()


def test_difference_bootstrap_includes_unshared_seed_clusters(monkeypatch):
    monkeypatch.setattr(analysis, "BOOTSTRAP_REPLICATES", 500)
    left = [
        {"seed": "1", "step_accuracy": 0.0},
        {"seed": "2", "step_accuracy": 1.0},
    ]
    right = [{"seed": "1", "step_accuracy": 0.0}]
    low, high = analysis.difference_cluster_bootstrap(left, right, "partial-seeds")
    assert low == 0.0
    assert high == 0.5
