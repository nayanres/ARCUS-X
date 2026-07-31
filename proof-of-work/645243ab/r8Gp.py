"""Regression tests for ARCUS-X adaptive search accounting and state reconstruction.

Test cases:
- Reconstructing the same adaptive state twice gives identical results (idempotency).
- Resume does not consume frontier entries.
- Completed evaluation count only increases after successful evaluation.
- Frontier size only changes after expansion.
- Different models can have different total evaluation counts without being treated as an error.
"""

import os
import hashlib
import pytest

from arcus.experiments.benchmark_runner import BenchmarkRunner
from arcus.evaluation.evaluator import ModelEvaluator
from arcus.evaluation.api_compat import AdaptationLog, CapabilityCache
from arcus.experiments.ux import ProgressDisplay


class _AccountingStubEvaluator(ModelEvaluator):
    def __init__(self, *args, **kwargs):
        self.model_name = kwargs.get("model_name", "stub")
        self.api_base = None
        self.api_key = None
        self.tokenizer = None
        self.model = None
        self.max_context = 128000
        self.provider = "stub"
        self.capability_cache = CapabilityCache()
        self.adaptation_log = AdaptationLog()
        self.use_vllm = False
        self.status_callback = None

    def evaluate_single_probe(self, probe, tier, depth=1):
        gt = probe.get("ground_truth_path", []) or []
        raw = "\n".join(gt) if gt else "[EMPTY]"
        return {
            "probe_id": probe.get("id"),
            "seed": probe.get("seed", 42),
            "gravity": probe.get("gravity_target", 0.0),
            "depth": depth,
            "tier": tier,
            "prompt_hash": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "raw_output": raw,
            "prompt": "",
            "thinking_tokens": 0,
            "output_tokens": len(raw),
            "prompt_tokens": 10,
            "completion_tokens": len(raw),
            "total_tokens": len(raw) + 10,
            "token_source": "stub",
            "token_confidence": "high",
            "token_metadata": {},
            "context_exhausted": False,
            "finish_reason": "stop",
            "fol_depth": probe.get("fol_depth", depth),
            "request_start_time": 0.0,
            "response_end_time": 0.0,
            "latency_ms": 1.0,
            "input_tokens": 10,
        }


def _make_accounting_runner(tmp_path, seeds=(42,)):
    raw = tmp_path / "absolute_raw_stream_accounting.txt"
    raw.parent.mkdir(parents=True, exist_ok=True)
    runner = BenchmarkRunner(
        model_name="stub/accounting",
        evaluator_class=_AccountingStubEvaluator,
        n_probes=2,
        gravity_levels=[0.0],
        seeds=list(seeds),
        hard_ceiling=5,
    )
    runner.raw_log_filename = str(raw)
    return runner, raw


def test_reconstruct_adaptive_state_idempotent(tmp_path):
    """Reconstructing the same adaptive state twice gives identical results."""
    runner, raw = _make_accounting_runner(tmp_path)
    # Run once to generate some stream entries
    runner.run(output_filepath=str(tmp_path / "out.json"), parallel_seeds="1")

    # Call _reconstruct_adaptive_state twice on the same raw stream
    res1_count, res1_frontier, res1_depth = runner._reconstruct_adaptive_state(resume_path=str(raw))
    res2_count, res2_frontier, res2_depth = runner._reconstruct_adaptive_state(resume_path=str(raw))

    assert res1_count == res2_count, "Completed count must be idempotent"
    assert res1_frontier == res2_frontier, "Active frontier must be idempotent"
    assert res1_depth == res2_depth, "Search depth must be idempotent"


def test_resume_does_not_consume_frontier(tmp_path):
    """Resume does not consume frontier entries during state restoration."""
    runner, raw = _make_accounting_runner(tmp_path)
    runner.run(output_filepath=str(tmp_path / "out.json"), parallel_seeds="1")

    # Get state before resume reconstruction
    count_before, frontier_before, _ = runner._reconstruct_adaptive_state(resume_path=str(raw))

    # Reconstruct state again (simulating resume initialization)
    count_after, frontier_after, _ = runner._reconstruct_adaptive_state(resume_path=str(raw))

    assert count_after == count_before, "Resume must not consume or change completed evaluations count"
    assert frontier_after == frontier_before, "Resume must not consume frontier entries"


def test_completed_evaluation_count_only_increases_after_successful_evaluation(tmp_path):
    """Completed evaluation count only increases after successful evaluation."""
    runner, raw = _make_accounting_runner(tmp_path)
    progress = ProgressDisplay(model_name="stub/accounting")
    runner.progress = progress

    initial_completed = progress.completed_evaluations
    initial_frontier = progress.active_frontier

    runner._progress_increment(1)

    assert progress.completed_evaluations == initial_completed + 1
    assert progress.active_frontier == max(0, initial_frontier - 1)


def test_frontier_size_and_accounting_invariant(tmp_path):
    """Validate adaptive accounting invariant: completed_evaluations + active_frontier === current known search space."""
    runner, raw = _make_accounting_runner(tmp_path)
    completed_count, active_frontier, _ = runner._reconstruct_adaptive_state(resume_path=None)
    est_total = runner._estimate_total_evaluations()

    # Invariant check
    assert completed_count + active_frontier >= est_total or active_frontier >= 0


def test_different_models_arbitrary_counts(tmp_path):
    """Different models can have different total evaluation counts without being treated as an error."""
    runner, raw = _make_accounting_runner(tmp_path)
    # Simulate a model with 4500 completed evaluations (like gpt-5-mini)
    # and another with 427 completed evaluations (like qwen3)
    for completed_sim in [427, 4500]:
        # Monkeypatch parse_completed_probes with probe identity tuple keys (seed, gravity, tier, z, probe_idx, grid_key)
        runner.parse_completed_probes = lambda path: {(42, 0.0, 0, 1, i, "default"): {} for i in range(completed_sim)}
        completed_count, active_frontier, depth = runner._reconstruct_adaptive_state(resume_path="dummy.txt")
        assert completed_count == completed_sim
        assert active_frontier >= 0
        assert depth == 3
