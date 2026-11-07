from collections import defaultdict

import pytest

from audit.correction_2026_11 import horizon_local_analysis as local


def rows(z, mean, *, task_prefix="task", seed_offset=0):
    return [
        {
            "z": z, "valid": True, "seed": str(seed_offset + i % 3),
            "task_hash": f"{task_prefix}-{i}", "gravity": 1.0, "step_accuracy": mean,
        }
        for i in range(12)
    ]


def test_window_inclusion_preserves_observed_horizons_and_overlap():
    groups = defaultdict(list)
    for z in (44, 45, 47, 50, 53, 55, 56):
        groups[("model", 0, z)].append({"z": z})
    assert [r["z"] for r in local.window_rows(groups, "model", 0, 50, 3)] == [47, 50, 53]
    assert [r["z"] for r in local.window_rows(groups, "model", 0, 50, 5)] == [45, 47, 50, 53, 55]
    assert [r["z"] for r in local.window_rows(groups, "model", 0, 50, 0)] == [50]
    assert 55 in [r["z"] for r in local.window_rows(groups, "model", 0, 60, 5)]


def test_support_codes_use_one_uniform_rule():
    assert local.support([]) == "N/A-U"
    assert local.support([{"valid": False}]) == "N/A-N"
    assert local.support(rows(20, 0.5)[:9]) == "N/A-I"
    assert local.support(rows(20, 0.5, task_prefix="x")) == "OK"
    assert local.support([dict(r, seed="1") for r in rows(20, 0.5)]) == "N/A-I"
    assert local.support([dict(r, task_hash="x") for r in rows(20, 0.5)]) == "N/A-I"
    assert local.standardized_comparison([], rows(20, 0.5), "missing")["paired_status"] == "N/A-U"
    assert local.standardized_comparison(
        [{"z": 20, "valid": False}], rows(20, 0.5), "invalid"
    )["paired_status"] == "N/A-N"


def test_standardization_weights_shared_horizons_equally(monkeypatch):
    monkeypatch.setattr(local.base, "BOOTSTRAP_REPLICATES", 100)
    left = rows(47, 0.2) + rows(50, 0.8) * 2
    right = rows(47, 0.4, task_prefix="other") + rows(50, 0.5, task_prefix="other")
    result = local.standardized_comparison(left, right, "test")
    assert result["comparison_status"] == "OK"
    assert result["supported_horizon_intersection"] == "47;50"
    assert result["standardized_left"] == pytest.approx(0.5)
    assert result["standardized_right"] == pytest.approx(0.45)
    assert result["difference"] == pytest.approx(0.05)
    assert result["matched_seed_task_pairs"] == 0


def test_shared_horizon_without_support_is_not_a_controlled_comparison(monkeypatch):
    monkeypatch.setattr(local.base, "BOOTSTRAP_REPLICATES", 100)
    left = rows(47, 0.2) + rows(50, 0.8)[:3]
    right = rows(50, 0.4) + rows(53, 0.5)
    result = local.standardized_comparison(left, right, "no-overlap")
    assert result["observed_horizon_intersection"] == "50"
    assert result["supported_horizon_intersection"] == ""
    assert result["comparison_status"] == "N/A-I"
    assert result["difference"] == ""
    assert result["matched_seed_task_pairs"] == 3


def test_task_pairs_require_identical_seed_and_experiment_hash(monkeypatch):
    monkeypatch.setattr(local.base, "BOOTSTRAP_REPLICATES", 100)
    left = rows(20, 0.8)
    different_seeds = rows(20, 0.3, seed_offset=10)
    assert local.standardized_comparison(left, different_seeds, "different")["matched_seed_task_pairs"] == 0
    matched = local.standardized_comparison(left, rows(20, 0.3), "matched")
    assert matched["matched_seed_task_pairs"] == 12
    assert matched["paired_status"] == "OK"
    assert matched["paired_difference"] == pytest.approx(0.5)
    other_gravity = [dict(r, gravity=2.0) for r in rows(20, 0.3)]
    assert local.standardized_comparison(left, other_gravity, "gravity")["matched_seed_task_pairs"] == 0


def test_compressed_bootstrap_matches_seed_block_draws(monkeypatch):
    monkeypatch.setattr(local.base, "BOOTSTRAP_REPLICATES", 100)
    left = rows(20, 0.8)
    right = rows(20, 0.3)
    left[0]["step_accuracy"] = 0.2
    right[1]["step_accuracy"] = 0.9
    actual = sorted(local.standard_bootstrap({20: left}, {20: right}, [20], "ignored"))
    seeds = ("0", "1", "2")
    rng = local.base.deterministic_rng(f"local-seed-draws-v1|{seeds}|{seeds}|100")
    expected = []
    for _ in range(100):
        chosen = [rng.choice(list(seeds)) for _ in seeds]
        lsample = [r["step_accuracy"] for seed in chosen for r in left if r["seed"] == seed]
        rsample = [r["step_accuracy"] for seed in chosen for r in right if r["seed"] == seed]
        expected.append(sum(lsample) / len(lsample) - sum(rsample) / len(rsample))
    assert actual == pytest.approx(sorted(expected))
