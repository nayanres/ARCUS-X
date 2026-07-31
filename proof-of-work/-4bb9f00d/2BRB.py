import sys, json
sys.path.insert(0, '.')

# Minimal stub evaluator that never calls a model.
class StubEvaluator:
    def __init__(self, *a, **k):
        pass
    def evaluate(self, *a, **k):
        # Return a minimal result the runner can consume.
        return {
            "raw_output": "x",
            "step_accuracy": 0.5,
            "exact_match": False,
            "continuity": 0.5,
            "correct_transitions": 1,
            "output_tokens": 10,
            "latency_ms": 1.0,
        }

from arcus.experiments.benchmark_runner import BenchmarkRunner

runner = BenchmarkRunner(
    model_name="stub",
    evaluator_class=StubEvaluator,
    seeds=[42, 1337, 2026],
    n_probes=1,
    gravity_levels=[0.0],
)

# Monkeypatch the heavy evaluation so no real work happens but seeds still loop.
def fake_evaluate_all(base_probes):
    # Populate a tiny results matrix so compute_cri / fracture work.
    runner.results_matrix = {
        "0|0.0|0|0|grid": {
            "step_accuracy": 0.6, "exact_match": False, "continuity": 0.6,
            "correct_transitions": 2, "output_tokens": 10, "latency_ms": 1.0,
            "raw_output": "x",
        }
    }
    runner.fracture_cache = {("0.0", "final"): 3}

runner._evaluate_all_dimensions = fake_evaluate_all

agg = runner._compute_aggregated_metrics()
print("master_seeds:", agg.get("master_seeds"))
print("per_seed keys:", list(agg.get("per_seed", {}).keys()))
print("seed_variance:", agg.get("seed_variance"))

assert agg.get("master_seeds") == [42, 1337, 2026], "master_seeds missing/wrong"
assert set(agg.get("per_seed", {}).keys()) == {"42", "1337", "2026"}, "per_seed keys wrong"
for seed, m in agg["per_seed"].items():
    assert "cri" in m and "fracture_depth" in m, f"per_seed[{seed}] incomplete"

# Report parameters block
report = runner._generate_report(agg)
print("report parameters seeds:", report["parameters"].get("seeds"))
assert report["parameters"].get("seeds") == [42, 1337, 2026]

print("OK: master_seeds + per_seed both present and correct")
