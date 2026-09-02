"""Tests for ARCUS-X latency-normalized efficiency diagnostics.

These diagnostics are *systems-level only*. Latency is affected by provider
infrastructure, architecture, batching, and deployment conditions, so it must
never be treated as a model capability score. The tests assert:

* per-probe latency fields are recorded by the evaluator,
* ALE and Correct Steps/sec are computed correctly and are diagnostic-only,
* the aggregate report contains the required latency sections,
* latency is NOT folded into CRI / GBI / trajectory / fracture metrics,
* multi-run / partial inputs are tolerated (stable schema).
"""

import time

from arcus.evaluation.metrics import LatencyDiagnostics
from arcus.evaluation.evaluator import ModelEvaluator


# ---------------------------------------------------------------------------
# Unit tests: LatencyDiagnostics math
# ---------------------------------------------------------------------------
def test_ale_basic():
    # Accuracy 0.8 over 10s -> ALE 0.08 (matches the spec example).
    assert abs(LatencyDiagnostics.accuracy_latency_efficiency(0.8, 10.0) - 0.08) < 1e-9


def test_ale_zero_latency_safe():
    # Latency <= 0 must not divide-by-zero; returns 0.0.
    assert LatencyDiagnostics.accuracy_latency_efficiency(0.8, 0.0) == 0.0
    assert LatencyDiagnostics.accuracy_latency_efficiency(0.8, None) == 0.0


def test_correct_steps_per_second_basic():
    # 1000 correct transitions in 20s -> 50 steps/sec (spec example).
    assert abs(LatencyDiagnostics.correct_steps_per_second(1000, 20.0) - 50.0) < 1e-9


def test_correct_steps_per_second_zero_latency_safe():
    assert LatencyDiagnostics.correct_steps_per_second(1000, 0.0) == 0.0


def test_aggregate_latency_report_shape():
    probes = [
        {"latency_ms": 1000.0, "output_tokens": 100, "accuracy": 0.8,
         "correct_transitions": 8, "horizon": 50},
        {"latency_ms": 2000.0, "output_tokens": 200, "accuracy": 0.6,
         "correct_transitions": 12, "horizon": 50},
        {"latency_ms": 3000.0, "output_tokens": 300, "accuracy": 0.9,
         "correct_transitions": 27, "horizon": 100},
    ]
    agg = LatencyDiagnostics.aggregate(probes)
    # Required aggregate keys.
    for key in ("mean_latency_ms", "median_latency_ms", "p95_latency_ms",
                "max_latency_ms", "mean_output_tokens_per_sec",
                "mean_correct_steps_per_sec", "ale", "correct_steps_per_sec",
                "per_horizon", "diagnostic_note"):
        assert key in agg, f"missing aggregate key: {key}"
    # Latency stats.
    assert abs(agg["mean_latency_ms"] - 2000.0) < 1e-6
    assert abs(agg["max_latency_ms"] - 3000.0) < 1e-6
    assert abs(agg["median_latency_ms"] - 2000.0) < 1e-6
    # P95 of [1000,2000,3000] = 2000 + 0.95*(3000-1000) interpolated -> 2900.
    assert abs(agg["p95_latency_ms"] - 2900.0) < 1e-6
    # Output tokens/sec: 100/1 + 200/2 + 300/3 = 100+100+100 = 300 /3 = 100.
    assert abs(agg["mean_output_tokens_per_sec"] - 100.0) < 1e-6
    # Correct steps/sec: 8/1 + 12/2 + 27/3 = 8+6+9 = 23 /3 = 7.6667.
    assert abs(agg["mean_correct_steps_per_sec"] - (23.0 / 3.0)) < 1e-3
    # ALE: 0.8/1 + 0.6/2 + 0.9/3 = 0.8+0.3+0.3 = 1.4 /3 = 0.4667.
    assert abs(agg["ale"] - (1.4 / 3.0)) < 1e-3
    # Per-horizon breakdown present and sorted.
    horizons = [ph["horizon"] for ph in agg["per_horizon"]]
    assert horizons == ["50", "100"], horizons
    z50 = next(ph for ph in agg["per_horizon"] if ph["horizon"] == "50")
    # z=50: mean latency (1000+2000)/2 = 1500; cps (8/1 + 12/2)/2 = 7.0.
    assert abs(z50["mean_latency_ms"] - 1500.0) < 1e-6
    assert abs(z50["correct_steps_per_sec"] - 7.0) < 1e-6


def test_aggregate_tolerates_missing_fields():
    # Probes without latency are skipped; probes without accuracy/correct
    # transitions still contribute latency stats (those diagnostics omitted).
    probes = [
        {"latency_ms": 500.0, "output_tokens": 50},
        {"latency_ms": 1500.0},  # no output_tokens/accuracy/correct_transitions
        {"accuracy": 0.5},        # no latency -> ignored entirely
    ]
    agg = LatencyDiagnostics.aggregate(probes)
    assert abs(agg["mean_latency_ms"] - 1000.0) < 1e-6
    # No accuracy/correct_transitions -> those means are 0.0 (stable schema).
    assert agg["ale"] == 0.0
    assert agg["mean_correct_steps_per_sec"] == 0.0
    assert agg["per_horizon"] == []


def test_aggregate_empty():
    agg = LatencyDiagnostics.aggregate([])
    assert agg["mean_latency_ms"] == 0.0
    assert agg["max_latency_ms"] == 0.0
    assert agg["per_horizon"] == []
    assert "diagnostic_note" in agg


# ---------------------------------------------------------------------------
# Integration: evaluator records per-probe latency fields
# ---------------------------------------------------------------------------
class _StubEvaluator(ModelEvaluator):
    """Minimal evaluator subclass that avoids vLLM/API construction."""

    def __init__(self):
        # Bypass the heavy __init__ (vLLM / API) entirely.
        self.model_name = "stub"
        self.provider = "local"
        self.capability_cache = None
        self.adaptation_log = None
        self.status_callback = None

    def evaluate(self, prompt, max_tokens=None):
        return {
            "output": "[0,0]->[0,1]->[0,2]",
            "finish_reason": "stop",
            "thinking_tokens": 0,
            "output_tokens": 10,
            "provider_response": None,
        }


def test_evaluator_records_latency_fields():
    ev = _StubEvaluator()
    probe = {
        "id": "p1", "seed": 1, "fol_depth": 3, "gravity_target": 0.0,
        "axioms": ["A"], "question": "Go.",
        "ground_truth_path": ["[0,0]", "[0,1]", "[0,2]"],
    }
    result = ev.evaluate_single_probe(probe, tier=0, depth=3)
    # Per-probe latency fields present.
    assert "request_start_time" in result
    assert "response_end_time" in result
    assert "latency_ms" in result
    assert result["latency_ms"] >= 0.0
    assert result["response_end_time"] >= result["request_start_time"]
    # input_tokens recorded (prompt token accounting).
    assert "input_tokens" in result
    assert result["input_tokens"] >= 0


def test_evaluator_latency_is_monotonic_across_calls():
    ev = _StubEvaluator()
    probe = {
        "id": "p1", "seed": 1, "fol_depth": 3, "gravity_target": 0.0,
        "axioms": ["A"], "question": "Go.",
        "ground_truth_path": ["[0,0]", "[0,1]", "[0,2]"],
    }
    t0 = ev.evaluate_single_probe(probe, tier=0, depth=3)["response_end_time"]
    time.sleep(0.001)
    t1 = ev.evaluate_single_probe(probe, tier=0, depth=3)["request_start_time"]
    assert t1 >= t0


# ---------------------------------------------------------------------------
# Guardrail: latency must NOT leak into CRI / trajectory metrics
# ---------------------------------------------------------------------------
def test_latency_excluded_from_cri_and_trajectory():
    # Latency diagnostics are computed independently and never feed CRI.
    # This test documents the contract: the CRI computation path does not
    # accept latency inputs, and LatencyDiagnostics is a separate module.
    from arcus.evaluation.metrics import ARCUSRobustnessIndex

    cri = ARCUSRobustnessIndex.compute(
        step_accuracy=0.7, continuity_score=0.6, exact_match=0.5,
        horizon_points=[(3.0, 0.7), (6.0, 0.6)],
        generation_efficiency=0.9,
    )
    cri_dict = cri.as_dict()
    # CRI components are the four trajectory/efficiency terms only.
    assert set(cri_dict["components"].keys()) == {
        "trajectory_fidelity", "horizon_robustness",
        "semantic_robustness", "generation_efficiency",
    }
    # Latency is never part of CRI.
    assert "latency" not in cri_dict
    assert "ale" not in cri_dict
    assert "correct_steps_per_sec" not in cri_dict
