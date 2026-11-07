"""Tests for run reconstruction validation and resume regression handling."""

import os
import pytest
from arcus.experiments.benchmark_runner import BenchmarkRunner
from post_run_analyzer import (
    _parse_actions,
    _parse_coord_block,
    _parse_grid,
    _parse_transition_rules,
    _render_report,
    process_txt_file,
)
from arcus.analysis.taxonomy import classify_from_result
from tests.test_parallel_resume import _StubEvaluator, _make_runner


def test_run_reconstruction_validation(tmp_path):
    """Validate that a resumed run produces identical analysis results to a fresh run."""
    seeds = [42]
    runner_fresh, raw_fresh = _make_runner(tmp_path / "fresh", seeds)
    runner_fresh.run(output_filepath=str(tmp_path / "fresh" / "out.json"), parallel_seeds="1")
    data_fresh = process_txt_file(str(raw_fresh))

    content = raw_fresh.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)
    entry_indices = [i for i, line in enumerate(lines) if "=== RAW STREAM ENTRY ===" in line]

    if len(entry_indices) >= 2:
        split_idx = entry_indices[1]
        part1 = "".join(lines[:split_idx])
        resumed_raw = tmp_path / "resumed" / "absolute_raw_stream_stub.txt"
        resumed_raw.parent.mkdir(parents=True, exist_ok=True)
        resumed_raw.write_text(part1, encoding="utf-8")

        runner_resumed, _ = _make_runner(tmp_path / "resumed", seeds)
        runner_resumed.raw_log_filename = str(resumed_raw)
        runner_resumed.run(output_filepath=str(tmp_path / "resumed" / "out.json"), parallel_seeds="1", resume_path=str(resumed_raw))
        data_resumed = process_txt_file(str(resumed_raw))

        assert data_fresh["completed_probes"] == data_resumed["completed_probes"]
        agg_fresh = runner_fresh._compute_aggregated_metrics()
        agg_resumed = runner_resumed._compute_aggregated_metrics()
        assert agg_fresh["step_accuracy"] == agg_resumed["step_accuracy"]
        assert agg_fresh["generation_bloat_index"] == agg_resumed["generation_bloat_index"]
        assert agg_fresh["cri"]["cri"] == agg_resumed["cri"]["cri"]


def test_multiple_resume_blocks_and_partial(tmp_path):
    """Test handling of multiple resume blocks and partial runs."""
    raw = tmp_path / "multi_resume.txt"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(
        "ARCUS-X RUN START\n"
        "Run ID: 2026-08-01T00:00:00\n"
        "Model: stub/stub\n"
        "Total Probes: 4\n"
        "==================================================\n"
        "=== RAW STREAM ENTRY ===\n"
        "Parameters: z=3, gravity=0.0, tier=0, grid=default\n"
        "EnvironmentMetadata: seed=42, tier=0, gravity=0.0, z=3, grid_key=default\n"
        "Grid: 5x5\n"
        "[MODEL OUTPUT]:\n[0,0] -> [1,0]\n"
        "[GROUND TRUTH EXPECTED]:\n[0,0] -> [1,0]\n"
        "==================================================\n"
        "ARCUS-X RUN START\n"
        "Run ID: 2026-08-01T00:01:00\n"
        "Model: stub/stub\n"
        "Total Probes: 4\n"
        "==================================================\n"
        "=== RAW STREAM ENTRY ===\n"
        "Parameters: z=3, gravity=0.0, tier=0, grid=default\n"
        "EnvironmentMetadata: seed=42, tier=0, gravity=0.0, z=3, grid_key=default\n"
        "Grid: 5x5\n"
        "[MODEL OUTPUT]:\n[0,0] -> [1,1]\n"
        "[GROUND TRUTH EXPECTED]:\n[0,0] -> [1,0]\n"
        "==================================================\n",
        encoding="utf-8"
    )

    data = process_txt_file(str(raw))
    assert data["completed_probes"] == 2
    assert "results_matrix" in data
    report = _render_report(data)
    assert "ARCUS-X RUN SUMMARY" in report


def test_model_output_cannot_overwrite_trusted_task_metadata(tmp_path):
    raw = tmp_path / "metadata.txt"
    raw.write_text(
        "ARCUS-X RUN START\n"
        "Run ID: 2026-08-01T00:00:00\n"
        "Model: stub/stub\n"
        "=== RAW STREAM ENTRY ===\n"
        "Parameters: z=1, gravity=0.0, tier=0, grid=default\n"
        "EnvironmentMetadata: seed=42, tier=0, gravity=0.0, z=1, grid_key=default\n"
        "Grid: 5x5\n"
        "InitialState: (0, 0)\n"
        "Actions: ['north']\n"
        "TransitionRules: {'north': {'dx': 0, 'dy': 1}, 'south': {'dx': 0, 'dy': -1}}\n"
        "------------------------------\n"
        "[MODEL OUTPUT]:\n"
        "Actions: ['south']\n"
        "Grid: 99x99\n"
        "[0,0]\n[0,1]\n"
        "------------------------------\n"
        "[GROUND TRUTH EXPECTED]:\n"
        "[0,0]\n[0,1]\n"
        "==================================================\n",
        encoding="utf-8",
    )

    runner_entry = BenchmarkRunner._parse_raw_stream_blocks(str(raw))[0]
    post_data = process_txt_file(str(raw))
    post_result = next(iter(post_data["results_matrix"].values()))

    assert _parse_actions(runner_entry["actions"]) == ["north"]
    assert _parse_grid(runner_entry["grid"]) == (5, 5)
    assert post_result["exact_match"] is True
    assert post_result["probe_classification"]["mechanism"] == "None"

    live_taxonomy = classify_from_result(
        post_result["predicted_path"],
        post_result["ground_truth_path"],
        _parse_transition_rules(runner_entry["transition_rules"]),
        _parse_actions(runner_entry["actions"]),
        tier=0,
        grid_width=5,
        grid_height=5,
        raw_output=post_result["raw_output"],
    )
    assert live_taxonomy.probe_classification.to_dict() == post_result["probe_classification"]
