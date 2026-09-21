import io, json, os
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
            "latency_ms": 1000.0 * (depth / 3.0),
            "input_tokens": 10,
            "output_tokens": 10,
        }


# Capture the terminal summary report text.
import contextlib, io as _io
runner = BenchmarkRunner(
    model_name="verify-stub", evaluator_class=StubEvaluator,
    n_probes=2, gravity_levels=[0.0, 1.0],
)
buf = _io.StringIO()
with contextlib.redirect_stdout(buf):
    runner.run(output_filepath="outputs/_verify_results.json")
report_text = buf.getvalue()

# Extract the Latency Diagnostics section from the printed report.
start = report_text.find("Latency Diagnostics:")
end = report_text.find("=" * 60, start)
section = report_text[start:end] if start != -1 else "(NOT FOUND)"

io.open("outputs/_verify_report.txt", "w", encoding="utf-8").write(section)

# Load the exported metrics JSON and confirm latency_diagnostics is present + clean.
with open("outputs/metrics_verify-stub.json", encoding="utf-8") as f:
    metrics = json.load(f)
lat = metrics.get("latency_diagnostics", {})

io.open("outputs/_verify_metrics.json", "w", encoding="utf-8").write(json.dumps(lat, indent=2))

required = ["mean_latency_ms", "median_latency_ms", "p95_latency_ms",
            "max_latency_ms", "mean_output_tokens_per_sec",
            "mean_correct_steps_per_sec", "ale", "correct_steps_per_sec",
            "per_horizon", "diagnostic_note"]
missing = [k for k in required if k not in lat]
print("REPORT SECTION PRESENT:", start != -1)
print("MISSING KEYS:", missing)
print("ALE:", lat.get("ale"))
print("CORRECT STEPS/SEC:", lat.get("correct_steps_per_sec"))
print("PER_HORIZON ENTRIES:", len(lat.get("per_horizon", [])))
print("DIAGNOSTIC NOTE PRESENT:", bool(lat.get("diagnostic_note")))
