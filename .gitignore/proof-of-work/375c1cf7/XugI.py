"""Smoke test for the ARCUS-X UX (banner, progress, chart) using a stub evaluator."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arcus.experiments.benchmark_runner import BenchmarkRunner
from arcus.analysis.fracture import FracturePointFinder

MODES = ["None", "State Tracking Failure", "Transition Rule Failure",
         "Semantic Interpretation Failure", "Horizon Collapse", "Formatting Failure"]


class StubEvaluator:
    def __init__(self, *a, **k):
        self.status_callback = None

    def evaluate_single_probe(self, probe, tier, depth=1):
        # Vary the error mode so the taxonomy chart shows a distribution.
        mode = MODES[(tier + depth) % len(MODES)]
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
            "error_mode": mode,
            "raw_output": "x",
            "finish_reason": "stop",
        }


if __name__ == "__main__":
    runner = BenchmarkRunner(
        model_name="glm-5.2",
        evaluator_class=StubEvaluator,
        fracture_finder_class=FracturePointFinder,
        n_probes=2,
        gravity_levels=[0.0, 1.0],
        seed=42,
        hard_ceiling=8,
    )
    report = runner.run(output_filepath="outputs/_ux_smoke_results.json")
    print("\n[smoke] run() returned; step_accuracy =", report["metrics"]["step_accuracy"])
