#!/usr/bin/env python3
"""Post-run analyzer for ARCUS-X raw stream logs.

This is a STANDALONE utility. It does NOT call any model APIs and performs no
benchmark generation, scoring, gravity, tier, transition-rule, or evaluation
logic. It only consumes a completed raw log file (the
``absolute_raw_stream_{model}.txt`` produced by :mod:`arcus.experiments.benchmark_runner`)
and derives reporting metrics from it.

Usage:
    python post_run_analyzer.py --input absolute_raw_data_deepseek-deepseek-v4-flash.txt

The runner writes ``absolute_raw_data_{provider}_{model}.txt``; the legacy
``absolute_raw_stream_{model}.txt`` name is also accepted as an alias.

It prints a clean terminal summary (mirroring the in-run report) and exports a
``metrics_{model}.json`` file alongside the input.
"""

import argparse
import ast
import json
import os
import re
import sys
from typing import Optional

# Allow running from the repo root without an install.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Real taxonomy (read-only classification; no API calls, no benchmark logic).
from arcus.analysis.taxonomy import (
    classify_from_result,
    Outcome,
    FailureMechanism,
    FailureSubtype,
    ErrorPattern,
    TaxonomyResult,
)
# Canonical evaluation metrics (read-only; no API calls, no benchmark logic).
from arcus.evaluation.metrics import (
    compare_trajectories,
    aggregate_trajectory_results,
    HorizonCompliance,
    GenerationBloatIndex,
    GenerationEfficiency,
    ARCUSRobustnessIndex,
    LatencyDiagnostics,
    metric_tier,
    _token_density,
)


# ---------------------------------------------------------------------------
# Raw log parsing (read-only; no API calls, no benchmark logic)
# ---------------------------------------------------------------------------
# Entries are split on their header line; each block is then parsed
# line-by-line. This is robust to both the new format (with Grid/Actions/
# TransitionRules env lines) and old logs that omit them (heuristic fallback).
ENTRY_HEADER_RE = re.compile(r"=== RAW STREAM ENTRY \(Tokens:\s*(\d+)\) ===")
PARAMS_RE = re.compile(
    r"Parameters:\s*z=([^,]+),\s*gravity=([^,]+),\s*tier=([^,]+),\s*grid=([^\n]+)"
)
GRID_RE = re.compile(r"Grid:\s*(\d+)x(\d+)")
INITIAL_STATE_RE = re.compile(r"InitialState:\s*(.*)")
ACTIONS_RE = re.compile(r"Actions:\s*(.*)")
RULES_RE = re.compile(r"TransitionRules:\s*(.*)")
ENV_META_RE = re.compile(r"EnvironmentMetadata:\s*(.*)")
# Per-entry provider telemetry written by the runner (new-format logs). These
# appear as plain "Latency(ms):", "InputTokens:", "OutputTokens:" lines and
# must be captured separately so they are not mistaken for model output.
LATENCY_RE = re.compile(r"Latency\(ms\):\s*([\d.]+)")
INPUT_TOKENS_RE = re.compile(r"InputTokens:\s*(\d+)")
OUTPUT_TOKENS_RE = re.compile(r"OutputTokens:\s*(\d+)")
# One-time metadata header written by the runner (new-format logs only).
EXPECTED_RE = re.compile(r"ExpectedProbes:\s*(\d+)")
MODEL_RE = re.compile(r"Model:\s*(.+)\s*$")
# Per-run delimiter blocks written by the runner at the start/end of every run.
# Splitting the log on these markers yields independent run segments, each
# carrying its own identity, intended probe total, and completion status so
# PARTIAL detection and multi-run reporting work without inference.
RUN_START_RE = re.compile(r"^ARCUS-X RUN START\s*$", re.MULTILINE)
RUN_END_RE = re.compile(r"^ARCUS-X RUN END\s*$", re.MULTILINE)
RUN_ID_RE = re.compile(r"Run ID:\s*(.+?)\s*$", re.MULTILINE)
RUN_MODEL_RE = re.compile(r"Model:\s*(.+?)\s*$", re.MULTILINE)
RUN_TOTAL_RE = re.compile(r"Total Probes:\s*(\d+)")
RUN_STATUS_RE = re.compile(r"Status:\s*(\w+)")
RUN_COMPLETED_RE = re.compile(r"Completed:\s*(\d+)/(\d+)")
RUN_FAILURES_RE = re.compile(r"Failures:\s*(\d+)")
RUN_REASON_RE = re.compile(r"Reason:\s*(.+?)\s*$", re.MULTILINE)
RUN_END_TIME_RE = re.compile(r"End Time:\s*(.+?)\s*$", re.MULTILINE)
# API compatibility adaptation blocks written by the runner whenever a
# provider/model requires a token-parameter change (e.g. max_tokens ->
# max_completion_tokens). Parsed so the analyzer can report exactly what
# compatibility changes occurred, with structured metadata.
ADAPT_PROVIDER_RE = re.compile(r"Provider:\s*(.+?)\s*$", re.MULTILINE)
ADAPT_MODEL_RE = re.compile(r"Model:\s*(.+?)\s*$", re.MULTILINE)
ADAPT_ADJUST_RE = re.compile(r"Adjustment:\s*(.+?)\s*$", re.MULTILINE)
ADAPT_REASON_RE = re.compile(r"Reason:\s*(.+?)\s*$", re.MULTILINE)


def load_manifest(manifest_path: str) -> Optional[dict]:
    """Load an experiment manifest JSON sidecar if present.

    The manifest records the experiment identity, software versions, and exact
    parameters, so results can be attributed and reproduced even when model
    aliases or provider names change. Returns ``None`` if missing/unreadable.
    """
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
# Optional provider metrics block (kept separate from trajectory tokens).
PROVIDER_RE = re.compile(
    r"ProviderMetrics:\s*input_tokens=([^,]+),\s*output_tokens=([^,]+),"
    r"\s*total_tokens=([^,]+),\s*latency=([^,]+),\s*cost=([^\n]+)"
)


def sanitize_model_name(model_name: str) -> str:
    """Mirror the runner's filename sanitization for deriving output names."""
    sanitized = model_name.replace("/", "-")
    sanitized = re.sub(r'[\\:*?"<>|]', "-", sanitized)
    sanitized = sanitized.strip(". ")
    return sanitized


def _parse_coord_list(text: str):
    """Parse a ground-truth / model path list into a list of [r,c] pairs."""
    coords = re.findall(r"\[(\d+),\s*(\d+)\]", text)
    return [(int(r), int(c)) for r, c in coords]


def _step_accuracy(model_path, truth_path) -> float:
    """Position-wise step accuracy over the trajectory (0.0-1.0 scale).

    Purely a reporting helper over already-logged strings; it does not invoke
    any evaluation module.
    """
    if not truth_path:
        return 0.0
    if not model_path:
        return 0.0
    n = min(len(model_path), len(truth_path))
    if n == 0:
        return 0.0
    correct = sum(
        1 for i in range(n) if model_path[i] == truth_path[i]
    )
    return correct / len(truth_path)


def parse_raw_log_text(content: str) -> dict:
    """Parse raw log *content* (a string) into a list of probe records.

    Entries are split on their header line and each block parsed line-by-line.
    Supports both the new format (with Grid/Actions/TransitionRules env lines)
    and old logs that omit them (heuristic fallback in analyze_raw_log).

    Returns a dict with ``records`` (list of dicts) and ``parse_errors`` (int).
    """
    # The preamble (before the first entry) may carry a one-time RUN METADATA
    # header written by the runner, including the intended probe total. This is
    # how the analyzer learns the expected count for PARTIAL detection without
    # any runner state or API access.
    preamble = re.split(ENTRY_HEADER_RE, content, 1)[0]
    expected_probes = None
    meta_model = None
    m_exp = EXPECTED_RE.search(preamble)
    if m_exp:
        try:
            expected_probes = int(m_exp.group(1))
        except ValueError:
            expected_probes = None
    m_model = MODEL_RE.search(preamble)
    if m_model:
        meta_model = m_model.group(1).strip()

    # Split into entry blocks on the header line.
    parts = ENTRY_HEADER_RE.split(content)
    # parts[0] is preamble; subsequent items alternate: (tokens, block).
    records = []
    parse_errors = 0
    for i in range(1, len(parts), 2):
        tokens_str = parts[i]
        block = parts[i + 1] if i + 1 < len(parts) else ""
        try:
            rec = _parse_entry_block(tokens_str, block)
            if rec is None:
                parse_errors += 1
            else:
                records.append(rec)
        except Exception:
            parse_errors += 1

    return {
        "records": records,
        "parse_errors": parse_errors,
        "expected_probes": expected_probes,
        "meta_model": meta_model,
    }


def parse_raw_log(input_file: str):
    """Parse a raw stream log file into a list of probe records.

    Thin file wrapper around :func:`parse_raw_log_text`.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()
    return parse_raw_log_text(content)


def _parse_entry_block(tokens_str: str, block: str):
    """Parse a single entry block (everything between two headers)."""
    tokens = int(tokens_str)
    lines = block.splitlines()

    z = gravity = tier = grid = None
    grid_w = grid_h = None
    initial_state_raw = None
    actions_raw = rules_raw = None
    env_meta_raw = None
    provider_metrics = None
    model_output = None
    ground_truth = None
    entry_latency = None
    entry_input_tokens = None
    entry_output_tokens = None

    section = None  # "model" or "truth"
    model_lines = []
    truth_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Parameters:"):
            # Parameters: z=3, gravity=0.0, tier=1, grid=default
            for kv in stripped[len("Parameters:"):].split(","):
                kv = kv.strip()
                if kv.startswith("z="):
                    try:
                        z = int(kv[2:])
                    except ValueError:
                        z = None
                elif kv.startswith("gravity="):
                    try:
                        gravity = float(kv[len("gravity="):])
                    except ValueError:
                        gravity = None
                elif kv.startswith("tier="):
                    try:
                        tier = int(kv[len("tier="):])
                    except ValueError:
                        tier = None
                elif kv.startswith("grid="):
                    grid = kv[len("grid="):]
        elif stripped.startswith("[MODEL OUTPUT]:"):
            section = "model"
            continue
        elif stripped.startswith("[GROUND TRUTH EXPECTED]:"):
            section = "truth"
            continue
        elif stripped.startswith("Initial State:"):
            initial_state_raw = stripped[len("Initial State:"):].strip()
        elif stripped.startswith("Actions:"):
            actions_raw = stripped[len("Actions:"):].strip()
        elif stripped.startswith("Transition Rules:"):
            rules_raw = stripped[len("Transition Rules:"):].strip()
        elif stripped.startswith("Grid:"):
            m = GRID_RE.search(stripped)
            if m:
                try:
                    grid_w = int(m.group(1))
                    grid_h = int(m.group(2))
                except ValueError:
                    grid_w = grid_h = None
        elif stripped.startswith("EnvironmentMetadata:"):
            env_meta_raw = stripped[len("EnvironmentMetadata:"):].strip()
        elif stripped.startswith("Provider Metrics:"):
            try:
                import json
                provider_metrics = json.loads(stripped[len("Provider Metrics:"):].strip())
            except Exception:
                provider_metrics = None
        elif LATENCY_RE.match(stripped):
            try:
                entry_latency = float(LATENCY_RE.match(stripped).group(1))
            except (ValueError, AttributeError):
                entry_latency = None
        elif INPUT_TOKENS_RE.match(stripped):
            try:
                entry_input_tokens = int(INPUT_TOKENS_RE.match(stripped).group(1))
            except (ValueError, AttributeError):
                entry_input_tokens = None
        elif OUTPUT_TOKENS_RE.match(stripped):
            try:
                entry_output_tokens = int(OUTPUT_TOKENS_RE.match(stripped).group(1))
            except (ValueError, AttributeError):
                entry_output_tokens = None
        else:
            if section == "model":
                model_lines.append(stripped)
            elif section == "truth":
                truth_lines.append(stripped)

    model_output = "\n".join(model_lines).strip()
    ground_truth = "\n".join(truth_lines).strip()

    # Model path: support both arrow format "[0,1]->[0,0]" and list format.
    if "->" in model_output:
        model_path = _parse_coord_list(model_output)
    else:
        model_path = _parse_coord_list(model_output)

    truth_path = _parse_coord_list(ground_truth)

    empty = (not model_output) or model_output.upper().startswith("[EMPTY")

    step_accuracy = _step_accuracy(model_path, truth_path) if not empty else 0.0

    # Canonical trajectory-compliance metrics from arcus.evaluation.metrics.
    # compare_trajectories expects canonical "[x,y]" token lists.
    model_tokens = [f"[{r},{c}]" for r, c in model_path]
    truth_tokens = [f"[{r},{c}]" for r, c in truth_path]
    traj = compare_trajectories(model_tokens, truth_tokens)
    exact_match = traj.exact_match
    continuity_score = traj.continuity_score
    first_divergence = traj.first_divergence
    horizon_match = traj.horizon_match
    final_state_correct = traj.final_state_correct
    parsed_ok = traj.parsed
    # Correct transition count = number of matching positions in the overlap.
    correct_transitions = sum(
        1 for i in range(min(len(model_path), len(truth_path)))
        if model_path[i] == truth_path[i]
    )

    # Reconstruct a minimal env dict if grid metadata is present; otherwise None.
    env = None
    if env_meta_raw or rules_raw or actions_raw or grid_w is not None:
        try:
            import json
            # The runner writes TransitionRules / Actions as Python repr
            # (single-quoted dicts/lists), so use ast.literal_eval, falling back
            # to json.loads for any JSON-formatted variants.
            def _parse_literal(text):
                if not text:
                    return None
                try:
                    return ast.literal_eval(text)
                except Exception:
                    try:
                        return json.loads(text)
                    except Exception:
                        return None

            meta = {}
            if env_meta_raw:
                # The runner writes EnvironmentMetadata as "key=value, ..." pairs
                # (not JSON), so parse it as such, tolerating a JSON object too.
                try:
                    meta = json.loads(env_meta_raw)
                except Exception:
                    meta = {}
                    for kv in env_meta_raw.split(","):
                        kv = kv.strip()
                        if "=" in kv:
                            k, v = kv.split("=", 1)
                            meta[k.strip()] = v.strip()
            # Prefer explicit Grid: WxH; fall back to EnvironmentMetadata.
            if grid_w is None:
                try:
                    grid_w = int(meta.get("width")) if meta.get("width") is not None else None
                except (ValueError, TypeError):
                    grid_w = None
            if grid_h is None:
                try:
                    grid_h = int(meta.get("height")) if meta.get("height") is not None else None
                except (ValueError, TypeError):
                    grid_h = None
            try:
                meta_tier = int(meta.get("tier")) if meta.get("tier") is not None else tier
            except (ValueError, TypeError):
                meta_tier = tier
            try:
                meta_gravity = float(meta.get("gravity")) if meta.get("gravity") is not None else gravity
            except (ValueError, TypeError):
                meta_gravity = gravity
            env = {
                "transition_rules": _parse_literal(rules_raw) or {},
                "actions": _parse_literal(actions_raw) or [],
                "grid_width": grid_w,
                "grid_height": grid_h,
                "tier": meta_tier,
                "gravity": meta_gravity,
            }
        except Exception:
            env = None

    return {
        "z": z,
        "gravity": gravity,
        "tier": tier,
        "grid": grid,
        "tokens": tokens,
        "empty": empty,
        "model_output": model_output,
        "ground_truth": ground_truth,
        "step_accuracy": step_accuracy,
        "exact_match": exact_match,
        "continuity_score": continuity_score,
        "first_divergence": first_divergence,
        "horizon_match": horizon_match,
        "final_state_correct": final_state_correct,
        "parsed_ok": parsed_ok,
        "correct_transitions": correct_transitions,
        "model_path": model_path,
        "truth_path": truth_path,
        "model_path_len": len(model_path),
        "truth_path_len": len(truth_path),
        "_model_coords": model_path,
        "_truth_coords": truth_path,
        "env": env,
        "provider_metrics": provider_metrics,
        "_raw_output": model_output,
        # Per-entry provider telemetry (latency / token counts) captured from
        # the runner's "Latency(ms):" / "InputTokens:" / "OutputTokens:" lines.
        # Normalized into the same shape as the JSON "Provider Metrics:" block so
        # downstream aggregation is uniform.
        "_entry_latency": entry_latency,
        "_entry_input_tokens": entry_input_tokens,
        "_entry_output_tokens": entry_output_tokens,
    }


def _split_runs(content: str) -> list:
    """Split a raw log into independent run segments using a state machine.

    The runner appends an ``ARCUS-X RUN START`` block before each run and an
    ``ARCUS-X RUN END`` block after it. We walk the lines and collect everything
    between a START and its matching END into one segment. This keeps multiple
    runs of the same model (or different models in one file) cleanly separated
    so old/new experiments never bleed into each other's metrics.

    Returns a list of ``(start_meta, segment_text)`` tuples, where ``start_meta``
    is a dict parsed from the RUN START block (run_id, model, total_probes) and
    ``segment_text`` is the raw text of that run (entries + RUN END block).
    """
    lines = content.splitlines(keepends=True)
    runs = []
    current = None  # dict: {"meta": {...}, "lines": [...], "ended": bool}
    SEP_RE = re.compile(r"^=+\s*$")
    for line in lines:
        stripped = line.strip()
        if RUN_START_RE.match(stripped):
            # Close any previously-open run that lacked an END marker (treat the
            # stray START as the start of a new run; the prior one is kept as-is).
            if current is not None:
                runs.append((current["meta"], "".join(current["lines"])))
            current = {"meta": {}, "lines": [line], "ended": False}
        elif RUN_END_RE.match(stripped):
            if current is not None:
                current["lines"].append(line)
                current["ended"] = True
            else:
                # END without a preceding START: ignore (defensive).
                continue
        else:
            if current is not None:
                current["lines"].append(line)
                # Once the RUN END marker is seen, the run stays open only until
                # the closing separator line (or the next START) so the trailing
                # Status / Completed / Failures / End Time metadata is captured.
                if current["ended"] and SEP_RE.match(stripped):
                    runs.append((current["meta"], "".join(current["lines"])))
                    current = None
            # Lines outside any run block are ignored for per-run analysis.

    # A run that started but never ended (crash / truncated log) is still useful:
    # keep it so PARTIAL detection can flag it. Its END metadata will be absent.
    if current is not None:
        runs.append((current["meta"], "".join(current["lines"])))

    # Parse the RUN START metadata for each segment.
    parsed_runs = []
    for meta, text in runs:
        m_id = RUN_ID_RE.search(text)
        m_model = RUN_MODEL_RE.search(text)
        m_total = RUN_TOTAL_RE.search(text)
        start_meta = {
            "run_id": m_id.group(1).strip() if m_id else None,
            "model": m_model.group(1).strip() if m_model else None,
            "total_probes": int(m_total.group(1)) if m_total else None,
        }
        parsed_runs.append((start_meta, text))
    return parsed_runs


def _classify_with_taxonomy(record: dict):
    """Classify a single parsed record with the evidence-based taxonomy.

    Returns a :class:`TaxonomyResult` on success, or ``None`` when the record
    lacks the environment metadata required for classification (e.g. legacy
    logs that only carry Parameters / MODEL OUTPUT / GROUND TRUTH).
    """
    env = record.get("env")
    if not env:
        return None
    try:
        # Reconstruct canonical [x,y] token lists from the parsed coord tuples.
        model_tokens = [f"[{r},{c}]" for r, c in record.get("_model_coords", [])]
        truth_tokens = [f"[{r},{c}]" for r, c in record.get("_truth_coords", [])]
        result = classify_from_result(
            model_tokens,
            truth_tokens,
            env["transition_rules"],
            env["actions"],
            int(record["tier"]),
            env["grid_width"],
            env["grid_height"],
            raw_output=record.get("_raw_output"),
        )
        if result.valid:
            return result
    except Exception:
        return None
    return None


def _analyze_segment(segment_text: str, start_meta: dict,
                     expected_override: Optional[int] = None) -> dict:
    """Analyze a single run segment's text into a metrics dict.

    Mirrors the legacy single-run aggregation but operates on one run segment
    only. The intended probe total comes from (in priority order): the explicit
    ``--expected`` override, the RUN START ``Total Probes`` header, or a
    structural estimate from distinct (z, gravity, tier) combos.
    """
    parsed = parse_raw_log_text(segment_text)
    records = parsed["records"]
    parse_errors = parsed["parse_errors"]
    meta_model = parsed.get("meta_model") or start_meta.get("model")

    base = {
        "model": "unknown",
        "run_status": "FAILED",
        "completed_probes": 0,
        "total_probes": 0,
        "accuracy": 0.0,
        "per_horizon_accuracy": {},
        "per_gravity_accuracy": {},
        "per_tier_accuracy": {},
        "avg_output_tokens": 0.0,
        "max_output_tokens": 0.0,
        "avg_latency": 0.0,
        "total_cost": None,
        "failure_count": 0,
        "failure_mechanism_distribution": {},
        "failure_subtype_distribution": {},
        "error_pattern_distribution": {},
        "recovery_statistics": {},
        "mean_divergence_by_horizon": {},
        "failure_dynamics_by_gravity": {},
        "failure_dynamics_by_tier": {},
        "provider_metrics": None,
        "parse_errors": parse_errors,
    }

    if not records:
        return base

    # Derive a model name: prefer the RUN START Model header, then the filename.
    model = meta_model or "unknown"

    completed = len(records)

    if expected_override is not None:
        total_probes = expected_override
    elif start_meta.get("total_probes") is not None:
        total_probes = start_meta["total_probes"]
    else:
        combos = {(r["z"], r["gravity"], r["tier"]) for r in records}
        total_probes = max(len(combos), completed)

    if completed == 0:
        run_status = "FAILED"
    elif completed < total_probes:
        run_status = "PARTIAL"
    else:
        run_status = "COMPLETE"

    per_horizon: dict = {}
    per_gravity: dict = {}
    per_tier: dict = {}
    token_list: list = []
    failure_count = 0

    mechanism_dist: dict = {}
    subtype_dist: dict = {}
    pattern_dist: dict = {}
    recovery_count = 0
    non_success = 0
    persistence_values: list = []
    div_by_horizon: dict = {}
    dyn_by_gravity: dict = {}
    dyn_by_tier: dict = {}

    # Collections for the canonical arcus.evaluation.metrics suite.
    traj_results = []          # compare_trajectories results (or dicts)
    horizon_points = []         # (z, step_accuracy) for CRI horizon robustness
    tier_acc = {0: [], 1: [], 2: [], 3: []}  # per-tier accuracy lists
    latency_probes = []        # per-probe dicts for LatencyDiagnostics.aggregate

    prov_input = []
    prov_output = []
    prov_total = []
    prov_latency = []
    prov_cost = []

    for r in records:
        per_horizon.setdefault(r["z"], []).append(r["step_accuracy"])
        per_gravity.setdefault(r["gravity"], []).append(r["step_accuracy"])
        per_tier.setdefault(r["tier"], []).append(r["step_accuracy"])
        if r["tokens"]:
            token_list.append(r["tokens"])
        if r["empty"] or r["step_accuracy"] == 0.0:
            failure_count += 1

        # Canonical trajectory-compliance result for this probe.
        traj_results.append({
            "parsed": r.get("parsed_ok", True),
            "exact_match": r.get("exact_match", False),
            "step_accuracy": r["step_accuracy"],
            "continuity_score": r.get("continuity_score", 0.0),
            "first_divergence": r.get("first_divergence"),
            "horizon_match": r.get("horizon_match", False),
            "final_state_correct": r.get("final_state_correct", False),
            "predicted_length": r.get("model_path_len", 0),
            "ground_truth_length": r.get("truth_path_len", 0),
        })
        horizon_points.append((float(r["z"]), r["step_accuracy"]))
        if r["tier"] in tier_acc:
            tier_acc[r["tier"]].append(r["step_accuracy"])
        # Latency probe (systems-level diagnostic only).
        probe = {"z": r["z"], "accuracy": r["step_accuracy"]}
        if r.get("_entry_latency") is not None:
            probe["latency_ms"] = r["_entry_latency"]
        if r.get("_entry_output_tokens") is not None:
            probe["output_tokens"] = r["_entry_output_tokens"]
        if r.get("correct_transitions") is not None:
            probe["correct_transitions"] = r["correct_transitions"]
        latency_probes.append(probe)

        pm = r.get("provider_metrics")
        if pm:
            if pm.get("input_tokens") is not None:
                prov_input.append(pm["input_tokens"])
            if pm.get("output_tokens") is not None:
                prov_output.append(pm["output_tokens"])
            if pm.get("total_tokens") is not None:
                prov_total.append(pm["total_tokens"])
            if pm.get("latency") is not None:
                prov_latency.append(pm["latency"])
            if pm.get("cost") is not None:
                prov_cost.append(pm["cost"])
        else:
            # Fall back to the per-entry telemetry captured from the runner's
            # "Latency(ms):" / "InputTokens:" / "OutputTokens:" lines.
            if r.get("_entry_input_tokens") is not None:
                prov_input.append(r["_entry_input_tokens"])
            if r.get("_entry_output_tokens") is not None:
                prov_output.append(r["_entry_output_tokens"])
                prov_total.append(r["_entry_output_tokens"])
                token_list.append(r["_entry_output_tokens"])
            if r.get("_entry_latency") is not None:
                prov_latency.append(r["_entry_latency"])

        tax = _classify_with_taxonomy(r)
        if tax is not None:
            mech = tax.mechanism.value
            mechanism_dist[mech] = mechanism_dist.get(mech, 0) + 1
            subtype_dist[tax.subtype.value] = subtype_dist.get(tax.subtype.value, 0) + 1
            for p in tax.patterns:
                pattern_dist[p.value] = pattern_dist.get(p.value, 0) + 1
            if tax.outcome != Outcome.SUCCESS:
                non_success += 1
                if tax.dynamics and tax.dynamics.recovered:
                    recovery_count += 1
                if tax.dynamics:
                    persistence_values.append(tax.dynamics.error_persistence)
            if tax.impact:
                div_by_horizon.setdefault(r["z"], []).append(tax.impact.mean_divergence)
            if tax.dynamics:
                dyn_by_gravity.setdefault(r["gravity"], []).append(
                    tax.dynamics.divergence_evolution.value
                )
                dyn_by_tier.setdefault(r["tier"], []).append(
                    tax.dynamics.divergence_evolution.value
                )
        else:
            if r["empty"]:
                mechanism_dist["Output Reliability Failure"] = \
                    mechanism_dist.get("Output Reliability Failure", 0) + 1
                pattern_dist["None"] = pattern_dist.get("None", 0) + 1
            elif r["model_path_len"] != r["truth_path_len"]:
                mechanism_dist["Transition Execution Failure"] = \
                    mechanism_dist.get("Transition Execution Failure", 0) + 1
                pattern_dist["None"] = pattern_dist.get("None", 0) + 1
            elif r["model_path_len"] > 0 and r["step_accuracy"] < 1.0:
                mechanism_dist["State Retention Failure"] = \
                    mechanism_dist.get("State Retention Failure", 0) + 1
                pattern_dist["None"] = pattern_dist.get("None", 0) + 1
            else:
                mechanism_dist["Unknown"] = mechanism_dist.get("Unknown", 0) + 1
                pattern_dist["None"] = pattern_dist.get("None", 0) + 1

    def _avg(vals):
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    def _avg4(vals):
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    overall_acc = round(_avg([r["step_accuracy"] for r in records]) * 100, 1)
    recovery_rate = round(recovery_count / non_success, 4) if non_success else 0.0
    avg_persistence = round(_avg(persistence_values), 1) if persistence_values else 0.0

    provider_metrics = None
    if prov_total:
        provider_metrics = {
            "avg_input_tokens": round(_avg(prov_input), 1),
            "avg_output_tokens": round(_avg(prov_output), 1),
            "avg_total_tokens": round(_avg(prov_total), 1),
            "avg_latency": round(_avg(prov_latency), 3),
            "total_cost": round(sum(prov_cost), 4) if prov_cost else None,
        }

    # --- Canonical arcus.evaluation.metrics suite -------------------------
    # Aggregated trajectory-compliance metrics (primary + secondary).
    agg_traj = aggregate_trajectory_results(traj_results)

    # Generation efficiency: GB I from actual vs expected output tokens, then GE.
    expected_tokens = total_probes  # 1 token per requested transition (heuristic)
    actual_tokens = sum(token_list) if token_list else 0
    gbi = GenerationBloatIndex.calculate(actual_tokens, expected_tokens) if expected_tokens > 0 else 0.0
    ge = GenerationEfficiency.calculate(gbi)

    # Horizon compliance rate (did the model emit the requested depth z?).
    hc_values = [
        HorizonCompliance.calculate(int(r["z"]), int(r["model_path_len"]))
        for r in records
    ]
    horizon_compliance_rate = round(sum(hc_values) / len(hc_values), 4) if hc_values else 0.0

    # Token density (tokens per horizon step), averaged across probes.
    td_values = [
        _token_density(r["tokens"], int(r["model_path_len"]))
        for r in records if r.get("tokens")
    ]
    mean_token_density = round(sum(td_values) / len(td_values), 4) if td_values else 0.0

    # ARCUS Robustness Index (secondary summary statistic).
    tier_acc_avg = {
        t: (round(sum(v) / len(v), 4) if v else None) for t, v in tier_acc.items()
    }
    cri_summary = ARCUSRobustnessIndex.compute(
        step_accuracy=agg_traj["mean_step_accuracy"],
        continuity_score=agg_traj["mean_continuity"],
        exact_match=agg_traj["exact_match_rate"],
        horizon_points=horizon_points,
        tier0_acc=tier_acc_avg.get(0),
        tier1_acc=tier_acc_avg.get(1),
        tier2_acc=tier_acc_avg.get(2),
        tier3_acc=tier_acc_avg.get(3),
        generation_efficiency=ge,
    )
    cri_dict = cri_summary.as_dict()

    # Latency diagnostics (systems-level only; never a capability score).
    latency_diagnostics = LatencyDiagnostics.aggregate(latency_probes)

    metrics = dict(base)
    metrics.update({
        "model": model,
        "run_status": run_status,
        "completed_probes": completed,
        "total_probes": total_probes,
        "accuracy": overall_acc,
        # Primary trajectory-compliance metrics.
        "exact_match_rate": agg_traj["exact_match_rate"],
        "mean_step_accuracy": agg_traj["mean_step_accuracy"],
        "mean_continuity": agg_traj["mean_continuity"],
        # Secondary trajectory-compliance metrics.
        "mean_final_state": agg_traj["mean_final_state"],
        "horizon_compliance_rate": horizon_compliance_rate,
        "parse_failure_rate": agg_traj["parse_failure_rate"],
        # Efficiency / generation metrics.
        "generation_bloat_index": round(gbi, 4),
        "generation_efficiency": round(ge, 4),
        "mean_token_density": mean_token_density,
        # ARCUS Robustness Index (secondary summary).
        "arcus_robustness_index": cri_dict,
        # Latency diagnostics (systems-level only).
        "latency_diagnostics": latency_diagnostics,
        "per_horizon_accuracy": {k: _avg(v) for k, v in sorted(per_horizon.items())},
        "per_gravity_accuracy": {
            k: _avg(v) for k, v in sorted(per_gravity.items(), key=lambda kv: float(kv[0]))
        },
        "per_tier_accuracy": {f"tier_{k}": _avg(v) for k, v in sorted(per_tier.items())},
        "avg_output_tokens": round(sum(token_list) / len(token_list), 1) if token_list else 0.0,
        "max_output_tokens": round(max(token_list), 1) if token_list else 0.0,
        "avg_latency": provider_metrics["avg_latency"] if provider_metrics else 0.0,
        "total_cost": provider_metrics["total_cost"] if provider_metrics else None,
        "failure_count": failure_count,
        "failure_mechanism_distribution": dict(sorted(mechanism_dist.items())),
        "failure_subtype_distribution": dict(sorted(subtype_dist.items())),
        "error_pattern_distribution": dict(sorted(pattern_dist.items())),
        "recovery_statistics": {
            "recovered": recovery_count,
            "failures": non_success,
            "recovery_rate": recovery_rate,
            "avg_error_persistence": avg_persistence,
        },
        "mean_divergence_by_horizon": {
            k: _avg4(v) for k, v in sorted(div_by_horizon.items())
        },
        "failure_dynamics_by_gravity": {
            k: _mode(v) for k, v in sorted(dyn_by_gravity.items(), key=lambda kv: float(kv[0]))
        },
        "failure_dynamics_by_tier": {
            f"tier_{k}": _mode(v) for k, v in sorted(dyn_by_tier.items())
        },
        "provider_metrics": provider_metrics,
    })
    return metrics


def parse_compatibility_adaptations(content: str) -> list:
    """Parse ``API COMPATIBILITY ADAPTATION`` blocks from raw log content.

    Returns a list of dicts with ``provider``, ``model``, ``adjustment`` and
    ``reason`` keys. Pure read-only parsing -- no API calls, no benchmark logic.
    """
    adaptations = []
    # Each block is delimited by the header line; split and inspect each piece.
    parts = re.split(r"API COMPATIBILITY ADAPTATION\s*", content)
    for block in parts[1:]:
        # Stop at the next dashed separator that ends the block.
        block = block.split("-" * 30)[0]
        prov = ADAPT_PROVIDER_RE.search(block)
        mod = ADAPT_MODEL_RE.search(block)
        adj = ADAPT_ADJUST_RE.search(block)
        reason = ADAPT_REASON_RE.search(block)
        if adj and prov and mod:
            adaptations.append({
                "provider": prov.group(1).strip(),
                "model": mod.group(1).strip(),
                "adjustment": adj.group(1).strip(),
                "reason": reason.group(1).strip() if reason else "",
            })
    return adaptations


def analyze_raw_log(input_file: str, expected_override: Optional[int] = None) -> dict:
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:  # pragma: no cover - defensive
        return {"status": "FAILED", "error": str(e), "metrics": {}, "runs": []}

    run_segments = _split_runs(content)

    # No RUN START markers at all -> treat the whole file as a single legacy run.
    if not run_segments:
        seg_metrics = _analyze_segment(content, {}, expected_override=expected_override)
        return {
            "status": "COMPLETE",
            "error": None,
            "metrics": seg_metrics,
            "runs": [{
                "run_id": None,
                "model": seg_metrics["model"],
                "status": seg_metrics["run_status"],
                "completed": seg_metrics["completed_probes"],
                "total": seg_metrics["total_probes"],
                "metrics": seg_metrics,
            }],
        }

    runs = []
    for start_meta, seg_text in run_segments:
        seg_metrics = _analyze_segment(seg_text, start_meta, expected_override=expected_override)
        # Prefer the explicit RUN END status if present in the segment.
        m_status = RUN_STATUS_RE.search(seg_text)
        m_completed = RUN_COMPLETED_RE.search(seg_text)
        m_failures = RUN_FAILURES_RE.search(seg_text)
        m_reason = RUN_REASON_RE.search(seg_text)
        m_end = RUN_END_TIME_RE.search(seg_text)
        status = m_status.group(1).strip() if m_status else seg_metrics["run_status"]
        completed = int(m_completed.group(1)) if m_completed else seg_metrics["completed_probes"]
        total = int(m_completed.group(2)) if m_completed else seg_metrics["total_probes"]
        failures = int(m_failures.group(1)) if m_failures else seg_metrics["failure_count"]
        # API compatibility adaptations observed in this run segment.
        adaptations = parse_compatibility_adaptations(seg_text)
        runs.append({
            "run_id": start_meta.get("run_id"),
            "model": start_meta.get("model") or seg_metrics["model"],
            "status": status,
            "completed": completed,
            "total": total,
            "failures": failures,
            "reason": m_reason.group(1).strip() if m_reason else "",
            "end_time": m_end.group(1).strip() if m_end else None,
            "api_compatibility_adaptations": adaptations,
            "metrics": seg_metrics,
        })

    # Aggregate view across all runs (weighted by completed probe count).
    total_completed = sum(r["completed"] for r in runs)
    total_intended = sum(r["total"] for r in runs)
    if total_completed > 0:
        agg_accuracy = round(
            sum(r["metrics"]["accuracy"] * r["completed"] for r in runs) / total_completed, 1
        )
    else:
        agg_accuracy = 0.0
    any_partial = any(r["status"] != "COMPLETE" for r in runs)
    agg_status = "PARTIAL" if any_partial else "COMPLETE"

    # Aggregate the detailed per-run metrics (taxonomy distributions, provider
    # telemetry, recovery statistics) so the exported JSON and summary carry the
    # full reporting picture, not just the headline accuracy.
    def _merge_dist(target, src):
        for k, v in (src or {}).items():
            target[k] = target.get(k, 0) + v
        return target

    agg_mechanism = {}
    agg_subtype = {}
    agg_pattern = {}
    agg_div_by_horizon = {}
    agg_dyn_by_gravity = {}
    agg_dyn_by_tier = {}
    agg_recovery = {"recovered": 0, "failures": 0, "avg_error_persistence": []}
    agg_traj_fields = {}
    agg_cri = []
    agg_latency = []
    prov_input = []
    prov_output = []
    prov_total = []
    prov_latency = []
    prov_cost = []
    for r in runs:
        m = r["metrics"]
        _merge_dist(agg_mechanism, m.get("failure_mechanism_distribution", {}))
        _merge_dist(agg_subtype, m.get("failure_subtype_distribution", {}))
        _merge_dist(agg_pattern, m.get("error_pattern_distribution", {}))
        for k, v in (m.get("mean_divergence_by_horizon", {}) or {}).items():
            agg_div_by_horizon.setdefault(k, []).append(v)
        for k, v in (m.get("failure_dynamics_by_gravity", {}) or {}).items():
            agg_dyn_by_gravity.setdefault(k, []).append(v)
        for k, v in (m.get("failure_dynamics_by_tier", {}) or {}).items():
            agg_dyn_by_tier.setdefault(k, []).append(v)
        rs = m.get("recovery_statistics", {}) or {}
        agg_recovery["recovered"] += rs.get("recovered", 0)
        agg_recovery["failures"] += rs.get("failures", 0)
        if rs.get("avg_error_persistence") is not None:
            agg_recovery["avg_error_persistence"].append(rs["avg_error_persistence"])
        pm = m.get("provider_metrics")
        if pm:
            if pm.get("avg_input_tokens") is not None:
                prov_input.append(pm["avg_input_tokens"])
            if pm.get("avg_output_tokens") is not None:
                prov_output.append(pm["avg_output_tokens"])
            if pm.get("avg_total_tokens") is not None:
                prov_total.append(pm["avg_total_tokens"])
            if pm.get("avg_latency") is not None:
                prov_latency.append(pm["avg_latency"])
            if pm.get("total_cost") is not None:
                prov_cost.append(pm["total_cost"])
        # Aggregate the canonical arcus.evaluation.metrics fields.
        for fld in (
            "exact_match_rate", "mean_step_accuracy", "mean_continuity",
            "mean_final_state", "horizon_compliance_rate", "parse_failure_rate",
            "generation_bloat_index", "generation_efficiency", "mean_token_density",
        ):
            v = m.get(fld)
            if v is not None:
                agg_traj_fields.setdefault(fld, []).append(v)
        if m.get("arcus_robustness_index") is not None:
            agg_cri.append(m["arcus_robustness_index"])
        if m.get("latency_diagnostics") is not None:
            agg_latency.append(m["latency_diagnostics"])

    def _avg(vals):
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def _avg1(vals):
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    agg_provider = None
    if prov_total:
        agg_provider = {
            "avg_input_tokens": _avg1(prov_input),
            "avg_output_tokens": _avg1(prov_output),
            "avg_total_tokens": _avg1(prov_total),
            "avg_latency": _avg(prov_latency),
            "total_cost": round(sum(prov_cost), 4) if prov_cost else None,
        }

    aggregate = {
        "model": runs[0]["model"] if runs else "unknown",
        "run_status": agg_status,
        "completed_probes": total_completed,
        "total_probes": total_intended,
        "accuracy": agg_accuracy,
        "num_runs": len(runs),
        "avg_output_tokens": _avg1([
            r["metrics"].get("avg_output_tokens", 0.0) for r in runs
            if r["metrics"].get("avg_output_tokens")
        ]),
        "max_output_tokens": max(
            [r["metrics"].get("max_output_tokens", 0.0) for r in runs], default=0.0
        ),
        "avg_latency": agg_provider["avg_latency"] if agg_provider else 0.0,
        "total_cost": agg_provider["total_cost"] if agg_provider else None,
        "failure_count": sum(r["metrics"].get("failure_count", 0) for r in runs),
        "failure_mechanism_distribution": dict(sorted(agg_mechanism.items())),
        "failure_subtype_distribution": dict(sorted(agg_subtype.items())),
        "error_pattern_distribution": dict(sorted(agg_pattern.items())),
        "recovery_statistics": {
            "recovered": agg_recovery["recovered"],
            "failures": agg_recovery["failures"],
            "recovery_rate": round(
                agg_recovery["recovered"] / agg_recovery["failures"], 4
            ) if agg_recovery["failures"] else 0.0,
            "avg_error_persistence": _avg1(agg_recovery["avg_error_persistence"]),
        },
        "mean_divergence_by_horizon": {
            k: _avg(v) for k, v in sorted(agg_div_by_horizon.items())
        },
        "failure_dynamics_by_gravity": {
            k: _mode(v) for k, v in sorted(
                agg_dyn_by_gravity.items(), key=lambda kv: float(kv[0])
            )
        },
        "failure_dynamics_by_tier": {
            k: _mode(v) for k, v in sorted(agg_dyn_by_tier.items())
        },
        "provider_metrics": agg_provider,
        # All API compatibility adaptations across every run segment, so the
        # report can show exactly what parameter changes occurred.
        "api_compatibility_adaptations": [
            a for r in runs for a in r.get("api_compatibility_adaptations", [])
        ],
        "runs_summary": [
            {
                "run_id": r["run_id"],
                "model": r["model"],
                "status": r["status"],
                "completed": r["completed"],
                "total": r["total"],
                "failures": r["failures"],
                "accuracy": r["metrics"]["accuracy"],
            }
            for r in runs
        ],
    }

    return {"status": "COMPLETE", "error": None, "metrics": aggregate, "runs": runs}


def _mode(values: list) -> str:
    """Return the most common value in a list (mode)."""
    if not values:
        return "none"
    counts: dict = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


# ---------------------------------------------------------------------------
# Terminal reporting (mirrors the in-run summary)
# ---------------------------------------------------------------------------
def print_summary_report(metrics: dict, runs: Optional[list] = None):
    """Print the multi-run summary report.

    When ``runs`` is provided (multi-run log), prints the ``ARCUS-X MULTI-RUN
    SUMMARY`` header with one block per run (Model / Status / Probes / Accuracy /
    Cost) followed by the aggregate totals. When ``runs`` is None, prints the
    legacy single-run summary for the given ``metrics`` dict.
    """
    if runs is not None:
        lines = [
            "",
            "=" * 50,
            "ARCUS-X MULTI-RUN SUMMARY",
            "=" * 50,
        ]
        for i, r in enumerate(runs, start=1):
            m = r["metrics"]
            pm = m.get("provider_metrics")
            cost = pm["total_cost"] if pm else None
            lines.extend([
                f"Run {i}:",
                f"  Model:     {r['model']}",
                f"  Status:    {r['status']}",
                f"  Probes:    {r['completed']}/{r['total']}",
                f"  Accuracy:  {m['accuracy']:.1f}%",
                f"  Cost:      {cost}",
            ])
            if r.get("reason"):
                lines.append(f"  Reason:    {r['reason']}")
        lines.extend([
            "",
            "-" * 50,
            "AGGREGATE:",
            f"  Runs:      {len(runs)}",
            f"  Probes:    {metrics['completed_probes']}/{metrics['total_probes']}",
            f"  Accuracy:  {metrics['accuracy']:.1f}%",
            f"  Status:    {metrics['run_status']}",
            "=" * 50,
        ])
        # API compatibility adaptations (provider/model parameter changes).
        adaptations = metrics.get("api_compatibility_adaptations", [])
        if adaptations:
            lines.extend(["", "API COMPATIBILITY ADAPTATIONS:"])
            for a in adaptations:
                lines.append(
                    f"  - Provider: {a['provider']} | Model: {a['model']} | "
                    f"Adjustment: {a['adjustment']}"
                )
        print("\n".join(lines))
        return

    # Legacy single-run report.
    lines = [
        "",
        "=" * 60,
        "ARCUS-X RUN SUMMARY (post-run analyzer)",
        "=" * 60,
        f"Model:               {metrics['model']}",
        f"Run Status:          {metrics['run_status']}",
        f"Completed Probes:    {metrics['completed_probes']} / {metrics['total_probes']}",
        f"Accuracy:            {metrics['accuracy']:.1f}%",
        "",
        "Per-Horizon Accuracy:",
    ]
    for z, acc in metrics["per_horizon_accuracy"].items():
        lines.append(f"  z={z:<4}: {acc * 100:.1f}%")
    lines.append("")
    lines.append("Per-Gravity Accuracy:")
    for g, acc in metrics["per_gravity_accuracy"].items():
        lines.append(f"  gravity={g:<4}: {acc * 100:.1f}%")
    lines.append("")
    lines.append("Per-Tier Accuracy:")
    for t, acc in metrics["per_tier_accuracy"].items():
        lines.append(f"  {t:<8}: {acc * 100:.1f}%")
    lines.append("")

    # Provider metrics (separate from trajectory token counts).
    pm = metrics.get("provider_metrics")
    if pm:
        lines.extend([
            "Provider Metrics (separate from trajectory tokens):",
            f"  Avg Input Tokens:   {pm['avg_input_tokens']}",
            f"  Avg Output Tokens:  {pm['avg_output_tokens']}",
            f"  Avg Total Tokens:   {pm['avg_total_tokens']}",
            f"  Avg Latency:        {pm['avg_latency']}",
            f"  Total Cost:         {pm['total_cost']}",
        ])
    else:
        lines.extend([
            f"Average Output Tokens: {metrics['avg_output_tokens']}",
            f"Maximum Output Tokens: {metrics['max_output_tokens']}",
            f"Average Latency:       {metrics['avg_latency']}",
        ])
    lines.extend([
        f"Failure Count:         {metrics['failure_count']}",
        "",
        "ARCUS-X Failure Analysis",
        "",
        "Failure Mechanism Distribution:",
    ])
    for mech, count in metrics.get("failure_mechanism_distribution", {}).items():
        lines.append(f"  {mech + ':':<32} {count}")
    lines.append("")
    lines.append("Error Pattern Distribution:")
    for pat, count in metrics.get("error_pattern_distribution", {}).items():
        lines.append(f"  {pat + ':':<32} {count}")
    lines.append("")
    lines.append("Recovery Statistics:")
    rs = metrics.get("recovery_statistics", {})
    lines.append(f"  Recovered:           {rs.get('recovered', 0)}")
    lines.append(f"  Failures:            {rs.get('failures', 0)}")
    lines.append(f"  Recovery Rate:       {rs.get('recovery_rate', 0.0)}")
    lines.append(f"  Avg Error Persistence: {rs.get('avg_error_persistence', 0.0)} steps")
    lines.append("")
    lines.append("Mean Divergence by Horizon:")
    for z, div in metrics.get("mean_divergence_by_horizon", {}).items():
        lines.append(f"  z={z:<4}: {div} cells")
    lines.append("")
    lines.append("Failure Dynamics by Gravity:")
    for g, evo in metrics.get("failure_dynamics_by_gravity", {}).items():
        lines.append(f"  gravity={g:<4}: {evo}")
    lines.append("")
    lines.append("Failure Dynamics by Tier:")
    for t, evo in metrics.get("failure_dynamics_by_tier", {}).items():
        lines.append(f"  {t:<8}: {evo}")
    lines.append("=" * 60)
    print("\n".join(lines))


def print_partial_run_report(metrics: dict):
    completed = metrics["completed_probes"]
    total = metrics["total_probes"]
    lines = [
        "=" * 35,
        "RUN STATUS:",
        "PARTIAL" if metrics["run_status"] == "PARTIAL" else metrics["run_status"],
        "",
        "Completed:",
        f"{completed} / {total}",
        "Statistics generated from completed probes only.",
        "=" * 35,
    ]
    print("\n".join(lines))


def export_metrics_json(metrics: dict, input_file: str, runs: Optional[list] = None):
    out_name = f"metrics_{sanitize_model_name(metrics['model'])}.json"
    out_path = os.path.join(os.path.dirname(os.path.abspath(input_file)), out_name)
    payload = dict(metrics)
    if runs is not None:
        payload["runs"] = runs
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def _resolve_input_path(input_arg: str) -> str:
    """Resolve the raw-log path, tolerating the two filename conventions.

    The runner writes ``absolute_raw_data_{provider}_{model}.txt``
    (see :mod:`arcus.experiments.benchmark_runner`), but some commands/docs
    reference ``absolute_raw_stream_{model}.txt``. Accept either form by
    transparently swapping the ``data`` <-> ``stream`` token when the literal
    path does not exist.
    """
    if os.path.exists(input_arg):
        return input_arg
    base, name = os.path.split(input_arg)
    swapped = None
    if name.startswith("absolute_raw_stream_"):
        swapped = "absolute_raw_data_" + name[len("absolute_raw_stream_"):]
    elif name.startswith("absolute_raw_data_"):
        swapped = "absolute_raw_stream_" + name[len("absolute_raw_data_"):]
    if swapped:
        candidate = os.path.join(base, swapped) if base else swapped
        if os.path.exists(candidate):
            return candidate
    return input_arg


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyze a completed ARCUS-X raw stream log (no API calls)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the raw log file, e.g. absolute_raw_data_{provider}_{model}.txt "
             "(the 'stream' prefix is also accepted as an alias).",
    )
    parser.add_argument(
        "--expected",
        type=int,
        default=None,
        help="Override the expected total probe count for PARTIAL detection "
             "(otherwise read from each run's RUN START header).",
    )
    args = parser.parse_args(argv)

    input_path = _resolve_input_path(args.input)
    result = analyze_raw_log(input_path, expected_override=args.expected)
    if result["status"] != "COMPLETE":
        print(f"[ERROR] Analysis failed: {result.get('error')}")
        return 1

    metrics = result["metrics"]
    runs = result.get("runs", [])

    # Link results back to the experiment manifest if one exists alongside the
    # raw log (single sidecar every result points back to).
    manifest_path = os.path.join(os.path.dirname(os.path.abspath(args.input)), "manifest.json")
    manifest = load_manifest(manifest_path)
    if manifest:
        print(f"Experiment: {manifest.get('experiment_id')} "
              f"(arcus {manifest.get('benchmark_version')})")

    print_summary_report(metrics, runs=runs)

    if metrics["run_status"] != "COMPLETE":
        print_partial_run_report(metrics)

    out_path = export_metrics_json(metrics, args.input, runs=runs)
    print(f"\nExported metrics to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
