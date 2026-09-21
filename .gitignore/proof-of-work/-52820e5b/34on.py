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
