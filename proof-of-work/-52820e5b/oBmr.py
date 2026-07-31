"""Tests for the multi-run raw-log parser and analyzer.

These verify that the post-run analyzer can split a single raw log file into
independent run segments (via the ARCUS-X RUN START / RUN END state machine),
report per-run metrics, and produce an aggregate view -- without runs bleeding
into each other.
"""

import os
import tempfile

import pytest

from post_run_analyzer import (
    analyze_raw_log,
    load_manifest,
    parse_raw_log_text,
)


def _entry(z, gravity, tier, model_path, truth_path, tokens=10):
    """Build a minimal RAW STREAM ENTRY block string."""
    return (
        f"\n=== RAW STREAM ENTRY (Tokens: {tokens}) ===\n"
        f"Parameters: z={z}, gravity={gravity}, tier={tier}, grid=8x8\n"
        f"Grid: 8x8\n"
        f"InitialState: [0,0]\n"
        f"Actions: [[0,0],[1,0]]\n"
        f"TransitionRules: step=1\n"
        f"EnvironmentMetadata: gravity={gravity}\n"
        f"[MODEL OUTPUT]\n{model_path}\n"
        f"[GROUND TRUTH EXPECTED]\n{truth_path}\n"
    )


def _run_block(run_id, model, total, status, completed, failures, entries):
    """Build a full RUN START + entries + RUN END block."""
    block = (
        "\n" + "=" * 50 + "\n"
        "ARCUS-X RUN START\n"
        f"Run ID: {run_id}\n"
        f"Model: {model}\n"
        f"Total Probes: {total}\n"
        "=" * 50 + "\n"
    )
    block += "".join(entries)
    block += (
        "\n" + "=" * 50 + "\n"
        "ARCUS-X RUN END\n"
        f"Status: {status}\n"
        f"Completed: {completed}/{total}\n"
        f"Failures: {failures}\n"
        f"End Time: 2026-07-16T04:12:22\n"
        "=" * 50 + "\n"
    )
    return block


def _make_log():
    # Run 1: complete, 2 perfect probes.
    run1 = _run_block(
        run_id="2026-07-16T02:31:44",
        model="deepseek/deepseek-v4-pro",
        total=2,
        status="COMPLETE",
        completed=2,
        failures=0,
        entries=[
            _entry(3, 1.0, 0, "[[0,0],[1,0],[2,0]]", "[[0,0],[1,0],[2,0]]"),
            _entry(5, 2.0, 1, "[[0,0],[1,1],[2,2]]", "[[0,0],[1,1],[2,2]]"),
        ],
    )
    # Run 2: partial, 1 perfect + 1 wrong (so accuracy 50%).
    run2 = _run_block(
        run_id="2026-07-16T05:00:00",
        model="deepseek/deepseek-v4-pro",
        total=4,
        status="PARTIAL",
        completed=2,
        failures=1,
        entries=[
            _entry(3, 1.0, 0, "[[0,0],[1,0],[2,0]]", "[[0,0],[1,0],[2,0]]"),
            _entry(5, 2.0, 1, "[[0,0],[9,9],[9,9]]", "[[0,0],[1,1],[2,2]]"),
        ],
    )
    return run1 + run2


def test_split_runs_separates_segments():
    log = _make_log()
    parsed = parse_raw_log_text(log)
    # parse_raw_log_text itself does not split runs; analyze_raw_log does.
    result = analyze_raw_log_text_helper(log)
    assert len(result["runs"]) == 2
    assert result["runs"][0]["run_id"] == "2026-07-16T02:31:44"
    assert result["runs"][1]["run_id"] == "2026-07-16T05:00:00"


def analyze_raw_log_text_helper(content):
    """Thin helper: write content to a temp file and analyze it."""
    import tempfile
    import os
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="arcus_test_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    try:
        return analyze_raw_log(path)
    finally:
        os.remove(path)


def test_per_run_metrics():
    result = analyze_raw_log_text_helper(_make_log())
    runs = result["runs"]
    # Run 1: both perfect -> 100% accuracy, COMPLETE.
    assert runs[0]["status"] == "COMPLETE"
    assert runs[0]["completed"] == 2
    assert runs[0]["total"] == 2
    assert runs[0]["metrics"]["accuracy"] == 100.0
    # Run 2: one perfect, one wrong -> 50% accuracy, PARTIAL.
    assert runs[1]["status"] == "PARTIAL"
    assert runs[1]["completed"] == 2
    assert runs[1]["total"] == 4
    assert runs[1]["metrics"]["accuracy"] == 50.0


def test_aggregate_view():
    result = analyze_raw_log_text_helper(_make_log())
    agg = result["metrics"]
    # Weighted by completed probes: (100*2 + 50*2) / 4 = 75.0
    assert agg["accuracy"] == 75.0
    assert agg["completed_probes"] == 4
    assert agg["total_probes"] == 6
    # Any partial run -> aggregate is PARTIAL.
    assert agg["run_status"] == "PARTIAL"
    assert agg["num_runs"] == 2


def test_legacy_single_run_no_markers():
    """A log with no RUN START/END markers is treated as one legacy run."""
    log = _entry(3, 1.0, 0, "[[0,0],[1,0]]", "[[0,0],[1,0]]")
    result = analyze_raw_log_text_helper(log)
    assert len(result["runs"]) == 1
    assert result["runs"][0]["run_id"] is None
    assert result["metrics"]["accuracy"] == 100.0


def test_run_end_status_overrides_inferred():
    """The explicit RUN END Status wins over inferred status."""
    # Log claims COMPLETE in the END block but only 1/4 probes present.
    log = _run_block(
        run_id="2026-07-16T02:31:44",
        model="deepseek/deepseek-v4-pro",
        total=4,
        status="COMPLETE",
        completed=1,
        failures=0,
        entries=[_entry(3, 1.0, 0, "[[0,0],[1,0]]", "[[0,0],[1,0]]")],
    )
    result = analyze_raw_log_text_helper(log)
    assert result["runs"][0]["status"] == "COMPLETE"
    # But the aggregate still reflects the real completed/total counts.
    assert result["metrics"]["completed_probes"] == 1
    assert result["metrics"]["total_probes"] == 4


def test_load_manifest(tmp_path):
    manifest = {
        "experiment_id": "ARCUS-X-exp-001",
        "benchmark_version": "1.0.1",
        "models": ["deepseek/deepseek-v4-pro"],
    }
    p = tmp_path / "manifest.json"
    p.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    loaded = load_manifest(str(p))
    assert loaded["experiment_id"] == "ARCUS-X-exp-001"
    # Missing file -> None.
    assert load_manifest(str(tmp_path / "nope.json")) is None
