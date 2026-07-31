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
    assert abs(agg["mean_correct_steps_per_sec"] - (23.0 / 3.0)) < 1e-6
    # ALE: 0.8/1 + 0.6/2 + 0.9/3 = 0.8+0.3+0.3 = 1.4 /3 = 0.4667.
    assert abs(agg["ale"] - (1.4 / 3.0)) < 1e-6
    # Per-horizon breakdown present and sorted.
    horizons = [ph["horizon"] for ph in agg["per_horizon"]]
    assert horizons == ["50", "100"], horizons
    z50 = next(ph for ph in agg["per_horizon"] if ph["horizon"] == "50")
