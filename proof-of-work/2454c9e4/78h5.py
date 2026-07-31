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
import io
import pytest

from arcus.experiments.benchmark_runner import BenchmarkRunner
from arcus.evaluation.evaluator import ModelEvaluator
from arcus.evaluation.api_compat import AdaptationLog, CapabilityCache
from arcus.experiments.ux import ProgressDisplay


# ---------------------------------------------------------------------------
# Deterministic stub evaluator (no API)
# ---------------------------------------------------------------------------
class _StubEvaluator(ModelEvaluator):
    """Returns a deterministic trajectory derived from the ground-truth path.

    The output is the ground truth itself (perfect trajectory) so scoring is
    stable and fully determined by the probe's environment -- no RNG, no network.
    """

    def __init__(self, *args, **kwargs):
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
    raw.parent.mkdir(parents=True, exist_ok=True)
    runner = BenchmarkRunner(
        model_name="stub/stub",
        evaluator_class=_StubEvaluator,
        n_probes=2,
        gravity_levels=[0.0, 1.0],
        seeds=list(seeds),
        hard_ceiling=10,
    )
    runner.raw_log_filename = str(raw)
    return runner, raw


def _matrix_signature(runner):
    keys = sorted(str(k) for k in runner.results_matrix.keys())
    sig = []
    for k in keys:
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

    r_serial, raw_serial = _make_runner(tmp_path / "s", seeds)
    r_serial.run(output_filepath=str(tmp_path / "s" / "out.json"),
                 parallel_seeds="1")
    serial_sig = _matrix_signature(r_serial)
    serial_raw = raw_serial.read_text(encoding="utf-8")

    r_par, raw_par = _make_runner(tmp_path / "p", seeds)
    r_par.run(output_filepath=str(tmp_path / "p" / "out.json"),
              parallel_seeds="auto")
    par_sig = _matrix_signature(r_par)
    par_raw = raw_par.read_text(encoding="utf-8")

    assert len(serial_sig) == len(par_sig), "probe count mismatch"
    assert serial_sig == par_sig, "parallel results differ from serial"

    def _entries(text):
        blocks = text.split("=== RAW STREAM ENTRY")
        norm = []
        for b in blocks[1:]:
            b_clean = b.split("==============================")[0]
            kept = [ln for ln in b_clean.splitlines() if not ln.strip().startswith("End Time:") and not "ARCUS-X" in ln and not "Run ID:" in ln]
            norm.append("\n".join(kept))
        return sorted(norm)
    assert _entries(serial_raw) == _entries(par_raw), "raw stream entries differ"


def test_parallel_auto_never_exceeds_seed_count(tmp_path):
    seeds = [42, 123]
    r, _ = _make_runner(tmp_path, seeds)
    import arcus.experiments.benchmark_runner as br
    orig = br.multiprocessing.cpu_count
    br.multiprocessing.cpu_count = lambda: 64
    try:
        w = r._resolve_worker_count("auto")
    finally:
        br.multiprocessing.cpu_count = orig
    assert w == len(seeds)


def test_resume_skips_completed_probes(tmp_path):
    seeds = [42, 123]
    r_full, raw_full = _make_runner(tmp_path / "full", seeds)
    r_full.run(output_filepath=str(tmp_path / "full" / "out.json"),
               parallel_seeds="1")
    full_sig = _matrix_signature(r_full)
    assert len(full_sig) > 0

    r_res, _ = _make_runner(tmp_path / "res", seeds)
    r_res.raw_log_filename = str(raw_full)
    r_res.run(output_filepath=str(tmp_path / "res" / "out.json"),
              parallel_seeds="1",
              resume_path=str(raw_full))
    res_sig = _matrix_signature(r_res)

    assert res_sig == full_sig
    full_text = raw_full.read_text(encoding="utf-8")
    assert full_text.count("=== RAW STREAM ENTRY") == \
        full_text.count("=== RAW STREAM ENTRY")


def test_resume_with_parallel(tmp_path):
    seeds = [42, 123, 456]
    r_full, raw_full = _make_runner(tmp_path / "fp", seeds)
    r_full.run(output_filepath=str(tmp_path / "fp" / "out.json"), parallel_seeds="1")
    full_sig = _matrix_signature(r_full)

    r_res, _ = _make_runner(tmp_path / "rp", seeds)
    r_res.raw_log_filename = str(raw_full)
    r_res.run(output_filepath=str(tmp_path / "rp" / "out.json"),
              parallel_seeds="auto", resume_path=str(raw_full))
    res_sig = _matrix_signature(r_res)
    assert res_sig == full_sig


def test_multi_seed_variance_summary(tmp_path):
    seeds = [42, 123, 456]
    r, _ = _make_runner(tmp_path / "ms", seeds)
    r.run(output_filepath=str(tmp_path / "ms" / "out.json"), parallel_seeds="1")
    sv = r._seed_variance_summary()
    assert sv["n_seeds"] == 3
    assert "cri" in sv
    assert "fracture_depth" in sv


def test_resume_latest_run_only(tmp_path):
    raw_file = tmp_path / "multi_run.txt"
    raw_file.write_text(
        "=== RAW STREAM ENTRY ===\nParameters: z=3, gravity=0.0, tier=0, grid=default\nEnvironmentMetadata: seed=42, tier=0, gravity=0.0, z=3, grid_key=default\n========================\n"
        "\n==================================================\nARCUS-X RUN START\nRun ID: old\n==================================================\n"
        "=== RAW STREAM ENTRY ===\nParameters: z=3, gravity=0.0, tier=0, grid=default\nEnvironmentMetadata: seed=42, tier=0, gravity=0.0, z=3, grid_key=default\n========================\n",
        encoding="utf-8"
    )
    with open(raw_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 50 + "\nARCUS-X RUN START\nRun ID: latest\n" + "=" * 50 + "\n")
        f.write("=== RAW STREAM ENTRY ===\nParameters: z=5, gravity=1.0, tier=1, grid=default\nEnvironmentMetadata: seed=42, tier=1, gravity=1.0, z=5, grid_key=default\n========================\n")

    completed = BenchmarkRunner.parse_completed_probes(str(raw_file))
    assert len(completed) == 1
    ident = list(completed.keys())[0]
    assert ident[3] == 5


def test_parse_completed_probes_index_tracking(tmp_path):
    raw_file = tmp_path / "index_test.txt"
    raw_file.write_text(
        "ARCUS-X RUN START\n"
        "=== RAW STREAM ENTRY ===\nParameters: z=3, gravity=0.0, tier=0, grid=default\nEnvironmentMetadata: seed=42, tier=0, gravity=0.0, z=3, grid_key=default\n========================\n"
        "=== RAW STREAM ENTRY ===\nParameters: z=3, gravity=0.0, tier=0, grid=default\nEnvironmentMetadata: seed=42, tier=0, gravity=0.0, z=3, grid_key=default\n========================\n",
        encoding="utf-8"
    )
    completed = BenchmarkRunner.parse_completed_probes(str(raw_file))
    assert len(completed) == 2
    keys = list(completed.keys())
    assert keys[0][4] == 0
    assert keys[1][4] == 1


def test_resume_empty_latest_session_fallback(tmp_path):
    raw_file = tmp_path / "interrupted_run.txt"
    raw_file.write_text(
        "ARCUS-X RUN START\nRun ID: prior\n"
        "=== RAW STREAM ENTRY ===\nParameters: z=3, gravity=0.0, tier=0, grid=default\nEnvironmentMetadata: seed=42, tier=0, gravity=0.0, z=3, grid_key=default\n========================\n",
        encoding="utf-8"
    )
    with open(raw_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 50 + "\nARCUS-X RUN START\nRun ID: interrupted_latest\n" + "=" * 50 + "\n")

    completed = BenchmarkRunner.parse_completed_probes(str(raw_file))
    assert len(completed) == 1
    ident = list(completed.keys())[0]
    assert ident[0] == 42
    assert ident[3] == 3


def test_resume_all_probes_completed_terminates_cleanly(tmp_path):
    seeds = [42]
    r_full, raw_full = _make_runner(tmp_path / "full_clean", seeds)
    r_full.run(output_filepath=str(tmp_path / "full_clean" / "out.json"), parallel_seeds="1")

    r_res, _ = _make_runner(tmp_path / "res_clean", seeds)
    r_res.raw_log_filename = str(raw_full)

    r_res.run(output_filepath=str(tmp_path / "res_clean" / "out.json"),
              parallel_seeds="1",
              resume_path=str(raw_full))

    assert r_res.actual_probe_count > 0
    assert len(r_res.results_matrix) > 0

    stream = io.StringIO()
    prog = ProgressDisplay(model_name="stub/stub", stream=stream)
    prog.set_resume_info(loaded=427, remaining=0)
    output = prog._build()
    assert "ARCUS-X Resume" in output
    assert "Completed evaluations: 427" in output
    assert "✓ All evaluations already completed" in output
    assert "✓ No evaluation required" in output


def test_adaptive_search_state_reconstruction_and_ux(tmp_path):
    """Comprehensive test covering all adaptive search UX and state reconstruction requirements."""
    seeds = [42]
    r, raw_stream = _make_runner(tmp_path / "adaptive", seeds)

    # 1. Fresh adaptive execution initializes UX from search state
    stream = io.StringIO()
    r.progress = ProgressDisplay(model_name="stub/stub", stream=stream)
    c, f, d = r._reconstruct_adaptive_state(None)
    assert c == 0
    assert r.progress.completed_evaluations == 0
    assert r.progress.active_frontier >= 0

    # 2. Progress display never renders invalid empty metadata during execution (shows "Waiting for adaptive search expansion...")
    output = r.progress._build()
    assert "Waiting for adaptive search expansion..." in output
    assert "Seed: -" not in output
    assert "Tier: -" not in output
    assert "Gravity: -" not in output
    assert "Horizon: -" not in output

    # 3. No static probe denominator is assumed
    assert "Probe" not in output
    assert "%" not in output

    # 4. Active execution metadata displays properly when set
    r.progress.set_context(tier=1, gravity=0.0, horizon=5, seed=42)
    active_output = r.progress._build()
    assert "Seed: 42" in active_output
    assert "Tier: 1" in active_output
    assert "Gravity: 0.0" in active_output
    assert "Horizon: 5" in active_output
    assert "Waiting for adaptive search expansion..." not in active_output

    # 5. Interrupted execution reconstructs identical UX state after resume & restores correct adaptive frontier
    r_full, raw_full = _make_runner(tmp_path / "full_exec", seeds)
    r_full.run(output_filepath=str(tmp_path / "full_exec" / "out.json"), parallel_seeds="1")

    r_res, _ = _make_runner(tmp_path / "res_exec", seeds)
    r_res.raw_log_filename = str(raw_full)
    stream_res = io.StringIO()
    r_res.progress = ProgressDisplay(model_name="stub/stub", stream=stream_res)
    c_res, f_res, d_res = r_res._reconstruct_adaptive_state(str(raw_full))
    assert c_res > 0
    assert r_res.progress.completed_evaluations == c_res
