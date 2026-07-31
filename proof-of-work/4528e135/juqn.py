import io, json
from arcus.experiments.benchmark_runner import BenchmarkRunner


class StubEvaluator:
    def __init__(self, *a, **k):
        pass

    def evaluate_single_probe(self, probe, tier, depth=1):
        gt = probe.get("ground_truth_path", [])
        return {
            "raw_output": "->".join(gt),
            "fol_depth": depth,
            "context_exhausted": False,
            "completion_tokens": 10,
            "total_tokens": 20,
            "provider_response": None,
            "finish_reason": "stop",
            # latency fields (normally set by the real evaluator)
            "latency_ms": 1000.0 * (depth / 3.0),
            "input_tokens": 10,
            "output_tokens": 10,
        }


runner = BenchmarkRunner(
    model_name="stub", evaluator_class=StubEvaluator,
    n_probes=2, gravity_levels=[0.0, 1.0],
)
base = runner._generate_base_probes()
runner._evaluate_all_dimensions(base)
agg = runner._compute_aggregated_metrics()
lat = agg.get("latency_diagnostics", {})
io.open("outputs/_sanity_latency.json", "w", encoding="utf-8").write(json.dumps(lat, indent=2))
print("latency_diagnostics keys:", sorted(lat.keys()))
print("mean_latency_ms:", lat.get("mean_latency_ms"))
print("ale:", lat.get("ale"))
print("correct_steps_per_sec:", lat.get("correct_steps_per_sec"))
print("per_horizon:", lat.get("per_horizon"))
print("has diagnostic_note:", "diagnostic_note" in lat)
