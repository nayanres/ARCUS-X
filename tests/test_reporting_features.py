"""Quick verification of the new interpretable reporting features (Task 3)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcus.experiments.benchmark_runner import BenchmarkRunner


def _stub_evaluator_class():
    class StubEvaluator:
        def __init__(self, *a, **k):
            pass

        def evaluate_single_probe(self, probe, tier, depth=1):
            return {
                "step_accuracy": 0.9 if tier == 0 else 0.5,
                "exact_match": tier == 0,
                "continuity_score": 0.8 if tier == 0 else 0.4,
                "predicted_length": depth,
                "ground_truth_length": depth,
                "horizon_compliance": 1.0,
                "generation_bloat_index": 0.1,
                "generation_efficiency": 0.9,
                "total_tokens": depth * 10,
                "fol_depth": depth,
                "fracture_depth": depth if tier > 0 else 0,
                "error_mode": "None",
                "raw_output": "x",
                "finish_reason": "stop",
            }
    return StubEvaluator


def test_aggregated_metrics_include_per_tier_and_per_gravity():
    runner = BenchmarkRunner(
        model_name="stub",
        evaluator_class=_stub_evaluator_class(),
        n_probes=1,
        gravity_levels=[0.0, 1.0],
        seed=42,
    )

    runner.results_matrix = {
        (1, 0.0, 0, 0, "8x8"): {"step_accuracy": 0.9, "exact_match": True,
                                 "continuity_score": 0.8, "fracture_depth": 0,
                                 "fol_depth": 1, "total_tokens": 10,
                                 "horizon_compliance": 1.0,
                                 "generation_bloat_index": 0.1,
                                 "generation_efficiency": 0.9, "error_mode": "None"},
        (2, 0.0, 0, 0, "8x8"): {"step_accuracy": 0.85, "exact_match": False,
                                 "continuity_score": 0.7, "fracture_depth": 0,
                                 "fol_depth": 2, "total_tokens": 20,
                                 "horizon_compliance": 1.0,
                                 "generation_bloat_index": 0.1,
                                 "generation_efficiency": 0.9, "error_mode": "None"},
        (3, 0.0, 0, 0, "8x8"): {"step_accuracy": 0.4, "exact_match": False,
                                 "continuity_score": 0.3, "fracture_depth": 0,
                                 "fol_depth": 3, "total_tokens": 30,
                                 "horizon_compliance": 1.0,
                                 "generation_bloat_index": 0.1,
                                 "generation_efficiency": 0.9, "error_mode": "None"},
        (1, 1.0, 1, 0, "8x8"): {"step_accuracy": 0.5, "exact_match": False,
                                 "continuity_score": 0.4, "fracture_depth": 1,
                                 "fol_depth": 1, "total_tokens": 10,
                                 "horizon_compliance": 1.0,
                                 "generation_bloat_index": 0.1,
                                 "generation_efficiency": 0.9, "error_mode": "None"},
        (2, 1.0, 1, 0, "8x8"): {"step_accuracy": 0.3, "exact_match": False,
                                 "continuity_score": 0.2, "fracture_depth": 1,
                                 "fol_depth": 2, "total_tokens": 20,
                                 "horizon_compliance": 1.0,
                                 "generation_bloat_index": 0.1,
                                 "generation_efficiency": 0.9, "error_mode": "None"},
    }

    agg = runner._compute_aggregated_metrics()

    assert "per_tier" in agg
    assert "tier_0" in agg["per_tier"]
    assert "tier_1" in agg["per_tier"]
    assert agg["per_tier"]["tier_0"]["fracture_depth"] == 3.0
    assert agg["per_tier"]["tier_1"]["fracture_depth"] == 2.0

    assert "per_gravity_fracture_curve" in agg
    assert 0.0 in agg["per_gravity_fracture_curve"]
    curve = agg["per_gravity_fracture_curve"][0.0]
    assert curve["depths"] == [1, 2, 3]
    assert curve["accuracies"] == [0.9, 0.85, 0.4]

    assert "cri" in agg
    cri = agg["cri"]
    assert "trajectory_fidelity" in cri
    assert "horizon_robustness" in cri
    assert "semantic_robustness" in cri
    assert "generation_efficiency" in cri
    assert "cri" in cri

    print("Task 3 reporting features verified:")
    print("  per_tier:", agg["per_tier"])
    print("  per_gravity_fracture_curve:", agg["per_gravity_fracture_curve"])
    print("  cri components:", {k: round(v, 3) if isinstance(v, (int, float)) else v
                                for k, v in cri.items() if k != "components"})


if __name__ == "__main__":
    test_aggregated_metrics_include_per_tier_and_per_gravity()
    print("OK")
