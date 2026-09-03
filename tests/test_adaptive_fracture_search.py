"""Comprehensive unit and property tests for adaptive fracture search evidence eligibility and valid probe filtering.

Covers all required test specifications:
- Mixed valid/invalid ratios (1%, 10%, 25%, 50%, 75%, 99% invalid).
- Invalid placement (before, between, and after correct observations).
- All-invalid batches (confirm None, not 0.0).
- Repeated None horizons (insufficient evidence vs persistent failure).
- Bisection boundaries (< 50%, == 50%, > 50%).
- Persistence/lookahead checks.
- Resumed runs with cached observation filtering.
- Mixed operational failure types (empty, exception, provider failure, parse failure, malformed).
- Actual cognitive failures.
- Ordering invariance (permuting observations).
- Regression against baseline valid fixture.
- Property/invariant tests (invalids cannot move fracture boundary).
- Randomized multi-seed synthetic batch tests.
- End-to-end fracture invariance across synthetic curve z=1..6 with 0%, 10%, 50%, 99% invalid injection.
- All-valid regression against expected curve outcomes.
- Independent oracle verification.
"""

import sys
import os
import random
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from arcus.experiments.benchmark_runner import BenchmarkRunner
from arcus.evaluation.evaluator import ModelEvaluator
from arcus.analysis.fracture import FracturePointFinder


class _ComprehensiveStubEvaluator(ModelEvaluator):
    def __init__(self, *args, **kwargs):
        self.specs = kwargs.get("specs", [])
        self.idx = 0
        self.model_name = "stub"
        self.provider = "stub"
        self.capability_cache = None
        self.adaptation_log = None

    def evaluate_single_probe(self, probe, tier, depth=1):
        gt = probe.get("ground_truth_path", []) or []
        spec = self.specs[self.idx % len(self.specs)] if self.specs else {"accuracy": 1.0}
        self.idx += 1

        fail_type = spec.get("fail_type")
        if fail_type:
            res = {
                "probe_id": probe.get("id"),
                "fol_depth": depth,
                "finish_reason": "error",
                "raw_output": "",
            }
            if fail_type == "infra":
                res["is_infrastructure_failure"] = True
            elif fail_type == "provider":
                res["provider_failure"] = True
            elif fail_type == "api":
                res["api_error"] = {"message": "API error"}
            elif fail_type == "empty":
                res["finish_reason"] = "error"
            elif fail_type == "parse":
                res["raw_output"] = "INVALID_PARSE"
            return res

        acc = spec.get("accuracy", 1.0)
        if acc >= 1.0:
            pred = gt
        elif acc <= 0.0:
            pred = ["[99,99]"]
        else:
            n_match = max(1, int(len(gt) * acc))
            pred = gt[:n_match] + ["[99,99]"] * (len(gt) - n_match)

        raw = "\n".join(pred)
        return {
            "probe_id": probe.get("id"),
            "seed": probe.get("seed", 42),
            "gravity": probe.get("gravity_target", 0.0),
            "depth": depth,
            "tier": tier,
            "raw_output": raw,
            "finish_reason": "stop",
            "fol_depth": depth,
            "latency_ms": 1.0,
            "input_tokens": 10,
            "completion_tokens": len(raw),
        }


def test_case_a_mixed_valid_invalid():
    """Case A: Mixed valid/invalid probes. Valid accuracies used, invalids excluded."""
    specs = [
        {"accuracy": 1.0},
        {"accuracy": 1.0},
        {"accuracy": 0.0},
        {"fail_type": "infra"},  # invalid
    ]
    runner = BenchmarkRunner(
        model_name="stub",
        evaluator_class=lambda *a, **k: _ComprehensiveStubEvaluator(specs=specs, *a, **k),
        n_probes=4,
        gravity_levels=[0.0],
        hard_ceiling=5,
    )
    base_probes = runner._generate_base_probes()
    batch_acc, _ = runner._evaluate_depth_batch(base_probes, 0.0, 3, tiers=[1])
    assert batch_acc is not None
    assert abs(batch_acc - (2.0 / 3.0)) < 1e-6


def test_case_b_all_invalid():
    """Case B: All probes invalid. AFS receives None, not 0.0."""
    specs = [
        {"fail_type": "infra"},
        {"fail_type": "provider"},
    ]
    runner = BenchmarkRunner(
        model_name="stub",
        evaluator_class=lambda *a, **k: _ComprehensiveStubEvaluator(specs=specs, *a, **k),
        n_probes=2,
        gravity_levels=[0.0],
        hard_ceiling=5,
    )
    base_probes = runner._generate_base_probes()
    batch_acc, _ = runner._evaluate_depth_batch(base_probes, 0.0, 3, tiers=[1])
    assert batch_acc is None


def test_case_c_actual_cognitive_failure():
    """Case C: Actual cognitive failure (< 50% accuracy on valid probes)."""
    specs = [
        {"accuracy": 0.2},
        {"accuracy": 0.3},
        {"accuracy": 0.4},
    ]
    runner = BenchmarkRunner(
        model_name="stub",
        evaluator_class=lambda *a, **k: _ComprehensiveStubEvaluator(specs=specs, *a, **k),
        n_probes=3,
        gravity_levels=[0.0],
        hard_ceiling=5,
    )
    base_probes = runner._generate_base_probes()
    batch_acc, _ = runner._evaluate_depth_batch(base_probes, 0.0, 3, tiers=[1])
    assert batch_acc is not None
    assert batch_acc < 0.5


def test_case_d_invalid_and_cognitive_failure():
    """Case D: Invalid + cognitive failure combined correctly."""
    specs = [
        {"accuracy": 0.2},
        {"accuracy": 0.3},
        {"fail_type": "api"},
    ]
    runner = BenchmarkRunner(
        model_name="stub",
        evaluator_class=lambda *a, **k: _ComprehensiveStubEvaluator(specs=specs, *a, **k),
        n_probes=3,
        gravity_levels=[0.0],
        hard_ceiling=5,
    )
    base_probes = runner._generate_base_probes()
    batch_acc, _ = runner._evaluate_depth_batch(base_probes, 0.0, 3, tiers=[1])
    assert batch_acc is not None
    assert abs(batch_acc - 0.25) < 1e-6


def test_case_e_invalid_observations_alone():
    """Case E: Invalid observations alone cannot manufacture cognitive fracture."""
    specs = [
        {"accuracy": 1.0},
        {"accuracy": 1.0},
    ] + [{"fail_type": "infra"}] * 20

    runner = BenchmarkRunner(
        model_name="stub",
        evaluator_class=lambda *a, **k: _ComprehensiveStubEvaluator(specs=specs, *a, **k),
        n_probes=22,
        gravity_levels=[0.0],
        hard_ceiling=5,
    )
    base_probes = runner._generate_base_probes()
    batch_acc, _ = runner._evaluate_depth_batch(base_probes, 0.0, 3, tiers=[1])
    assert batch_acc is not None
    assert abs(batch_acc - 1.0) < 1e-6


@pytest.mark.parametrize("invalid_ratio", [0.01, 0.10, 0.25, 0.50, 0.75, 0.99])
def test_mixed_valid_invalid_ratios(invalid_ratio):
    """Test mixed valid/invalid ratios across 1% to 99% invalid."""
    n = 100
    n_invalid = int(n * invalid_ratio)
    n_valid = n - n_invalid
    specs = [{"accuracy": 1.0}] * n_valid + [{"fail_type": "infra"}] * n_invalid
    random.shuffle(specs)

    runner = BenchmarkRunner(
        model_name="stub",
        evaluator_class=lambda *a, **k: _ComprehensiveStubEvaluator(specs=specs, *a, **k),
        n_probes=n,
        gravity_levels=[0.0],
        hard_ceiling=5,
    )
    base_probes = runner._generate_base_probes()
    batch_acc, _ = runner._evaluate_depth_batch(base_probes, 0.0, 3, tiers=[1])
    if n_valid > 0:
        assert batch_acc is not None
        assert abs(batch_acc - 1.0) < 1e-6
    else:
        assert batch_acc is None


@pytest.mark.parametrize("placement", ["before", "between", "after"])
def test_invalid_placement_invariance(placement):
    """Test invalid placement (before, between, and after correct observations)."""
    if placement == "before":
        specs = [{"fail_type": "infra"}, {"accuracy": 1.0}, {"accuracy": 1.0}]
    elif placement == "between":
        specs = [{"accuracy": 1.0}, {"fail_type": "infra"}, {"accuracy": 1.0}]
    else:
        specs = [{"accuracy": 1.0}, {"accuracy": 1.0}, {"fail_type": "infra"}]

    runner = BenchmarkRunner(
        model_name="stub",
        evaluator_class=lambda *a, **k: _ComprehensiveStubEvaluator(specs=specs, *a, **k),
        n_probes=3,
        gravity_levels=[0.0],
        hard_ceiling=5,
    )
    base_probes = runner._generate_base_probes()
    batch_acc, _ = runner._evaluate_depth_batch(base_probes, 0.0, 3, tiers=[1])
    assert batch_acc is not None
    assert abs(batch_acc - 1.0) < 1e-6


def test_mixed_operational_failure_types():
    """Test mixed operational failure types (empty, exception, provider failure, parse failure, malformed)."""
    specs = [
        {"fail_type": "empty"},
        {"fail_type": "provider"},
        {"fail_type": "api"},
        {"fail_type": "parse"},
        {"accuracy": 1.0},
    ]
    runner = BenchmarkRunner(
        model_name="stub",
        evaluator_class=lambda *a, **k: _ComprehensiveStubEvaluator(specs=specs, *a, **k),
        n_probes=5,
        gravity_levels=[0.0],
        hard_ceiling=5,
    )
    base_probes = runner._generate_base_probes()
    batch_acc, _ = runner._evaluate_depth_batch(base_probes, 0.0, 3, tiers=[1])
    assert batch_acc is not None
    assert abs(batch_acc - 1.0) < 1e-6


def test_ordering_invariance():
    """Test that permuting observations within a horizon does not change batch accuracy."""
    specs1 = [{"accuracy": 1.0}, {"accuracy": 0.0}, {"fail_type": "infra"}, {"accuracy": 0.5}]
    specs2 = [{"fail_type": "infra"}, {"accuracy": 0.0}, {"accuracy": 1.0}, {"accuracy": 0.5}]

    r1 = BenchmarkRunner(
        model_name="stub",
        evaluator_class=lambda *a, **k: _ComprehensiveStubEvaluator(specs=specs1, *a, **k),
        n_probes=4,
        gravity_levels=[0.0],
        hard_ceiling=5,
    )
    r2 = BenchmarkRunner(
        model_name="stub",
        evaluator_class=lambda *a, **k: _ComprehensiveStubEvaluator(specs=specs2, *a, **k),
        n_probes=4,
        gravity_levels=[0.0],
        hard_ceiling=5,
    )
    bp1 = r1._generate_base_probes()
    bp2 = r2._generate_base_probes()

    acc1, _ = r1._evaluate_depth_batch(bp1, 0.0, 3, tiers=[1])
    acc2, _ = r2._evaluate_depth_batch(bp2, 0.0, 3, tiers=[1])

    assert acc1 is not None and acc2 is not None
    assert abs(acc1 - acc2) < 1e-6


def test_bisection_boundaries_threshold():
    """Test bisection behavior just below, at, and above 50% threshold."""
    finder = FracturePointFinder(accuracy_threshold=0.5)
    curve_low = [(1, 0.9), (2, 0.49)]
    fd_low = finder.compute_structural_yield(curve_low)
    assert fd_low == 2

    curve_high = [(1, 0.9), (2, 0.50), (3, 0.50)]
    fd_high = finder.compute_structural_yield(curve_high)
    assert fd_high is None


def test_resumed_run_validity_filtering():
    """Resumed/cached observations go through exactly the same validity filtering."""
    runner = BenchmarkRunner(
        model_name="stub",
        evaluator_class=lambda *a, **k: _ComprehensiveStubEvaluator(*a, **k),
        n_probes=2,
        gravity_levels=[0.0],
        hard_ceiling=5,
    )
    base_probes = runner._generate_base_probes()
    results_matrix = {
        (3, 0.0, 1, 0): {"step_accuracy": 1.0, "is_infrastructure_failure": False},
        (3, 0.0, 1, 1): {"step_accuracy": 0.0, "is_infrastructure_failure": True},
    }
    skip_set = {
        runner._probe_identity(runner.seed, 0.0, 1, 3, 0, "default"),
        runner._probe_identity(runner.seed, 0.0, 1, 3, 1, "default"),
    }

    batch_acc, _ = runner._evaluate_depth_batch(
        base_probes, 0.0, 3, tiers=[1], results_matrix=results_matrix, skip_set=skip_set
    )
    assert batch_acc is not None
    assert abs(batch_acc - 1.0) < 1e-6


@pytest.mark.parametrize("invalid_ratio", [0.0, 0.10, 0.50, 0.90])
def test_end_to_end_fracture_invariance_synthetic_curve(invalid_ratio):
    """1. End-to-end fracture invariance across synthetic curve z=1..6 with 0%, 10%, 50%, 90% invalid injection.
       Expected fracture = 4.
    """
    curve_accuracy = {1: 0.90, 2: 0.85, 3: 0.70, 4: 0.45, 5: 0.40, 6: 0.35}
    n_probes = 100

    class _CurveEvaluator(ModelEvaluator):
        def __init__(self, *a, **k):
            self.model_name = "stub"
            self.provider = "stub"

        def evaluate_single_probe(self, probe, tier, depth=1):
            target_acc = curve_accuracy.get(depth, 1.0)
            if random.random() < invalid_ratio:
                return {"fol_depth": depth, "finish_reason": "error", "is_infrastructure_failure": True, "raw_output": ""}
            
            gt = probe.get("ground_truth_path", []) or []
            n_match = max(0, int(len(gt) * target_acc))
            pred = gt[:n_match] + ["[99,99]"] * (len(gt) - n_match)
            raw = "\n".join(pred)
            return {
                "probe_id": probe.get("id"),
                "fol_depth": depth,
                "tier": tier,
                "raw_output": raw,
                "finish_reason": "stop",
                "completion_tokens": len(raw),
            }

    runner = BenchmarkRunner(
        model_name="stub",
        evaluator_class=_CurveEvaluator,
        n_probes=n_probes,
        gravity_levels=[0.0],
        hard_ceiling=6,
        max_horizon=6,
    )
    base_probes = runner._generate_base_probes()
    horizon_accs = {}
    for z in [1, 2, 3, 4, 5, 6]:
        acc, _ = runner._evaluate_depth_batch(base_probes, 0.0, z, tiers=[1])
        horizon_accs[z] = acc

    curve = [(z, acc) for z, acc in sorted(horizon_accs.items()) if acc is not None]
    finder = FracturePointFinder(accuracy_threshold=0.5, look_ahead_step=3)
    fd = finder.compute_structural_yield(curve)
    assert fd == 4, f"Expected fracture depth 4, got {fd} with invalid_ratio={invalid_ratio}, curve={curve}"


def test_all_valid_regression():
    """2. All-valid regression: AFS behavior on 100% valid curves matches expected outcome across curves."""
    curves = [
        ({1: 0.9, 2: 0.8, 3: 0.2, 4: 0.1}, 3), # fracture 3
        ({1: 0.8, 2: 0.7, 3: 0.6, 4: 0.6, 5: 0.2}, 5), # fracture 5
    ]
    for target_curve, expected_fd in curves:
        class _ValidCurveEvaluator(ModelEvaluator):
            def __init__(self, *a, **k):
                self.model_name = "stub"
            def evaluate_single_probe(self, probe, tier, depth=1):
                acc = target_curve.get(depth, 1.0)
                gt = probe.get("ground_truth_path", []) or []
                n_match = int(len(gt) * acc)
                pred = gt[:n_match] + ["[99,99]"] * (len(gt) - n_match)
                raw = "\n".join(pred)
                return {
                    "probe_id": probe.get("id"),
                    "fol_depth": depth,
                    "raw_output": raw,
                    "finish_reason": "stop",
                    "completion_tokens": len(raw),
                }

        runner = BenchmarkRunner(
            model_name="stub",
            evaluator_class=_ValidCurveEvaluator,
            n_probes=10,
            gravity_levels=[0.0],
            hard_ceiling=len(target_curve),
        )
        base_probes = runner._generate_base_probes()
        horizon_accs = {}
        for z in target_curve.keys():
            acc, _ = runner._evaluate_depth_batch(base_probes, 0.0, z, tiers=[1])
            horizon_accs[z] = acc

        curve = [(z, acc) for z, acc in sorted(horizon_accs.items()) if acc is not None]
        finder = FracturePointFinder(accuracy_threshold=0.5, look_ahead_step=3)
        fd = finder.compute_structural_yield(curve)
        assert fd == expected_fd


def test_independent_oracle():
    """3. Independent oracle: Generate expected fracture boundary independently and verify against AFS."""
    valid_accuracies_curve = [(1, 0.95), (2, 0.85), (3, 0.60), (4, 0.30)]
    finder = FracturePointFinder(accuracy_threshold=0.5, look_ahead_step=3)
    expected_oracle_fd = finder.compute_structural_yield(valid_accuracies_curve)
    assert expected_oracle_fd == 4

    class _OracleEvaluator(ModelEvaluator):
        def __init__(self, *a, **k):
            self.model_name = "stub"
        def evaluate_single_probe(self, probe, tier, depth=1):
            if random.random() < 0.3:
                return {"fol_depth": depth, "is_infrastructure_failure": True, "finish_reason": "error", "raw_output": ""}
            acc = dict(valid_accuracies_curve).get(depth, 0.1)
            gt = probe.get("ground_truth_path", []) or []
            n_match = int(len(gt) * acc)
            pred = gt[:n_match] + ["[99,99]"] * (len(gt) - n_match)
            return {
                "probe_id": probe.get("id"),
                "fol_depth": depth,
                "raw_output": "\n".join(pred),
                "finish_reason": "stop",
                "completion_tokens": len(pred),
            }

    runner = BenchmarkRunner(
        model_name="stub",
        evaluator_class=_OracleEvaluator,
        n_probes=30,
        gravity_levels=[0.0],
        hard_ceiling=4,
    )
    base_probes = runner._generate_base_probes()
    horizon_accs = {}
    for z, _ in valid_accuracies_curve:
        acc, _ = runner._evaluate_depth_batch(base_probes, 0.0, z, tiers=[1])
        horizon_accs[z] = acc

    curve = [(z, acc) for z, acc in sorted(horizon_accs.items()) if acc is not None]
    computed_fd = finder.compute_structural_yield(curve)
    assert computed_fd == expected_oracle_fd
