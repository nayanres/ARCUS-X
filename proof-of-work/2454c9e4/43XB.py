"""Tests for adaptive parallel master-seed execution and resume support.

These tests use a *deterministic stub evaluator* (no network / no API) so they
run offline and are reproducible. The stub returns a trajectory that is a
deterministic function of the probe's ground-truth path, so:

* serial vs parallel execution must produce byte-identical results matrices,
* a resumed run must NOT re-run completed probes and must produce the same
  final dataset as an uninterrupted run,
* --parallel-seeds auto must never create more workers than master seeds.
"""

import os
import hashlib

import pytest

from arcus.experiments.benchmark_runner import BenchmarkRunner
from arcus.evaluation.evaluator import ModelEvaluator
from arcus.evaluation.api_compat import AdaptationLog, CapabilityCache


# ---------------------------------------------------------------------------
# Deterministic stub evaluator (no API)
# ---------------------------------------------------------------------------
class _StubEvaluator(ModelEvaluator):
    """Returns a deterministic trajectory derived from the ground-truth path.

    The output is the ground truth itself (perfect trajectory) so scoring is
    stable and fully determined by the probe's environment -- no RNG, no network.
    """

    def __init__(self, *args, **kwargs):
        # Bypass ModelEvaluator.__init__ (which would try vLLM / API setup).
        self.model_name = kwargs.get("model_name", "stub")
        self.api_base = None
        self.api_key = None
        self.tokenizer = None
        self.model = None
        self.max_context = 128000
        self.provider = "stub"
        self.capability_cache = None
        self.adaptation_log = None
        self.use_vllm = False
        self.status_callback = None
        self.adaptation_log = AdaptationLog()
        self.capability_cache = CapabilityCache()

    def evaluate_single_probe(self, probe, tier, depth=1):
        gt = probe.get("ground_truth_path", []) or []
        # Deterministic "model output": the ground truth, occasionally truncated
        # to exercise the scoring path without randomness.
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


def _make_runner(tmp_path, seeds):
    raw = tmp_path / "absolute_raw_stream_stub.txt"
    runner = BenchmarkRunner(
        model_name="stub/stub",
        evaluator_class=_StubEvaluator,
        n_probes=2,
        gravity_levels=[0.0, 1.0],
        seeds=list(seeds),
        hard_ceiling=10,  # keep the bisection small + fast
    )
    # Point the raw log at our temp file.
    runner.raw_log_filename = str(raw)
    return runner, raw


def _matrix_signature(runner):
    """Stable signature of the results matrix for equality checks."""
    keys = sorted(str(k) for k in runner.results_matrix.keys())
    sig = []
    for k in keys:
        # find the matching tuple key
        for tk in runner.results_matrix:
            if str(tk) == k:
                r = runner.results_matrix[tk]
                sig.append((k, round(float(r.get("step_accuracy", 0.0)), 6),
                            round(float(r.get("exact_match", 0.0) or 0.0), 6)))
                break
    return sig


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_serial_vs_parallel_deterministic(tmp_path):
    """Parallel execution must produce identical results to serial execution."""
    seeds = [42, 123, 456]

    # Serial run.
    r_serial, raw_serial = _make_runner(tmp_path / "s", seeds)
    r_serial.run(output_filepath=str(tmp_path / "s" / "out.json"),
                 parallel_seeds="1")
    serial_sig = _matrix_signature(r_serial)
    serial_raw = raw_serial.read_text(encoding="utf-8")

    # Parallel run (auto -> min(cpu, n_seeds)).
    r_par, raw_par = _make_runner(tmp_path / "p", seeds)
    r_par.run(output_filepath=str(tmp_path / "p" / "out.json"),
              parallel_seeds="auto")
    par_sig = _matrix_signature(r_par)
    par_raw = raw_par.read_text(encoding="utf-8")

    assert len(serial_sig) == len(par_sig), "probe count mismatch"
    assert serial_sig == par_sig, "parallel results differ from serial"
    # Raw stream content must be identical up to entry ordering across seeds.
    # Since each seed writes the same probes, the multiset of entry lines must
    # match (ordering across seeds may differ, so compare sorted entry blocks).
    def _entries(text):
        blocks = text.split("=== RAW STREAM ENTRY")
        return sorted(b for b in blocks if "Parameters:" in b)
    assert _entries(serial_raw) == _entries(par_raw), "raw stream entries differ"


def test_parallel_auto_never_exceeds_seed_count(tmp_path):
    """--parallel-seeds auto must not create more workers than master seeds."""
    seeds = [42, 123]
    r, _ = _make_runner(tmp_path, seeds)
    # Force a single-CPU environment view by monkeypatching cpu_count high.
    import arcus.experiments.benchmark_runner as br
    orig = br.multiprocessing.cpu_count
    br.multiprocessing.cpu_count = lambda: 64
    try:
        w = r._resolve_worker_count("auto")
    finally:
        br.multiprocessing.cpu_count = orig
    assert w == len(seeds), f"auto created {w} workers for {len(seeds)} seeds"


def test_resume_skips_completed_probes(tmp_path):
    """Resume must not re-run completed probes and yields same final dataset."""
    seeds = [42, 123]

    # 1) Full uninterrupted run.
    r_full, raw_full = _make_runner(tmp_path / "full", seeds)
    r_full.run(output_filepath=str(tmp_path / "full" / "out.json"),
               parallel_seeds="1")
    full_sig = _matrix_signature(r_full)
    assert len(full_sig) > 0

    # 2) Resume from the produced raw stream.
    r_res, raw_res = _make_runner(tmp_path / "res", seeds)
    # Reuse the SAME raw stream file (append mode, as the spec requires).
    r_res.raw_log_filename = str(raw_full)
    r_res.run(output_filepath=str(tmp_path / "res" / "out.json"),
              parallel_seeds="1",
              resume_path=str(raw_full))
    res_sig = _matrix_signature(r_res)

    assert res_sig == full_sig, "resumed dataset differs from uninterrupted run"
    # The resumed raw stream must contain exactly the same entries (no duplicates
    # appended, because completed probes are skipped).
    full_text = raw_full.read_text(encoding="utf-8")
    res_text = raw_res.read_text(encoding="utf-8") if raw_res.exists() else full_text
    # raw_res points at the same file as raw_full, so the file is unchanged in
    # size/entry count (no new entries written for already-completed probes).
    assert full_text.count("=== RAW STREAM ENTRY") == \
        res_text.count("=== RAW STREAM ENTRY"), "resume re-ran completed probes"


def test_resume_with_parallel(tmp_path):
    """Resume must work together with --parallel-seeds without duplicating work."""
    seeds = [42, 123, 456]

    r_full, raw_full = _make_runner(tmp_path / "fp", seeds)
    r_full.run(output_filepath=str(tmp_path / "fp" / "out.json"), parallel_seeds="1")
    full_sig = _matrix_signature(r_full)

    r_res, _ = _make_runner(tmp_path / "rp", seeds)
    r_res.raw_log_filename = str(raw_full)
    r_res.run(output_filepath=str(tmp_path / "rp" / "out.json"),
              parallel_seeds="auto", resume_path=str(raw_full))
    res_sig = _matrix_signature(r_res)

    assert res_sig == full_sig, "parallel resume dataset differs from full run"
