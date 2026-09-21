#!/usr/bin/env python3
"""
Post-run analyzer for ARCUS-X benchmark results.

This script now handles both JSON result files and TXT log files. For TXT files, it parses structured log entries into a format compatible with the existing metrics computation.
"""

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr.encoding != "utf-8":
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Re-implement the small helpers used by benchmark_runner
# ---------------------------------------------------------------------------

def _coerce_float(value: Any) -> float:
    """Safely coerce values to float for aggregation."""
    try:
        if value is None:
            return 0.0
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _unpack_matrix_key(key: str) -> Tuple[Any, Any, Any, Any, str]:
    """Normalise a results_matrix key to ``(z, gravity, tier, probe_idx, grid_key)``.

    Mirrors ``BenchmarkRunner._unpack_matrix_key``. The quickstart JSON keys
    are 4-tuples of strings like ``"20_0.5_1_0"``; we keep them as strings so
    sorting (numeric for z, float for gravity) works as in the runner.
    """
    parts = key.split("_")
    if len(parts) >= 5:
        return (parts[0], parts[1], parts[2], parts[3], parts[4])
    z, gravity, tier, probe_idx = parts[0], parts[1], parts[2], parts[3]
    return (z, gravity, tier, probe_idx, "default")


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _avg_round1(vals: List[float]) -> float:
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def _compute_report_metrics(results_matrix: Dict[str, Any], model: str,
                            run_status: str, completed: int, total: int,
                            accuracy: float) -> Dict[str, Any]:
    """Replicates BenchmarkRunner._compute_report_metrics."""
    per_horizon: Dict[Any, List[float]] = {}
    per_gravity: Dict[Any, List[float]] = {}
    per_tier: Dict[Any, List[float]] = {}

    output_tokens_list: List[float] = []
    latency_list: List[float] = []
    total_cost = 0.0
    failure_count = 0

    taxonomy_counts = {
        "Wraparound errors": 0,
        "State drift": 0,
        "Formatting failures": 0,
        "Unknown": 0,
    }

    for key, result in results_matrix.items():
        z, gravity, tier, _probe_idx, _grid_key = _unpack_matrix_key(key)
        acc = _coerce_float(result.get("step_accuracy", 0.0))

        per_horizon.setdefault(z, []).append(acc)
        per_gravity.setdefault(gravity, []).append(acc)
        per_tier.setdefault(tier, []).append(acc)

        ot = result.get("output_tokens")
        if ot is None:
            ot = result.get("completion_tokens", 0)
        if ot:
            output_tokens_list.append(_coerce_float(ot))

        lat = result.get("latency_ms")
        if lat is None:
            lat = result.get("latency")
        if lat is not None:
            latency_list.append(_coerce_float(lat))

        cost = result.get("cost")
        if cost is not None:
            total_cost += _coerce_float(cost)

        mode = result.get("error_mode", "None")
        if result.get("finish_reason") in ("error",) or mode not in ("None", "Unknown"):
            failure_count += 1

        # Bucket the real taxonomy ErrorMode into the 4 reporting categories.
        if mode in ("Transition Rule Failure", "Horizon Collapse"):
            taxonomy_counts["Wraparound errors"] += 1
        elif mode == "State Tracking Failure":
            taxonomy_counts["State drift"] += 1
        elif mode == "Formatting Failure":
            taxonomy_counts["Formatting failures"] += 1
        else:
            taxonomy_counts["Unknown"] += 1

    per_horizon_acc = {str(k): _avg_round1(v) for k, v in sorted(per_horizon.items())}
    per_gravity_acc = {str(k): _avg_round1(v) for k, v in sorted(per_gravity.items(), key=lambda kv: float(kv[0]))}
    per_tier_acc = {f"tier_{k}": _avg_round1(v) for k, v in sorted(per_tier.items())}

    avg_output_tokens = round(_mean(output_tokens_list), 1) if output_tokens_list else 0.0
    max_output_tokens = round(max(output_tokens_list), 1) if output_tokens_list else 0.0
    avg_latency = round(_mean(latency_list), 3) if latency_list else 0.0

    return {
        "model": model,
        "run_status": run_status or "UNKNOWN",
        "completed_probes": completed,
        "total_probes": total,
        "accuracy": accuracy,
        "per_horizon_accuracy": per_horizon_acc,
        "per_gravity_accuracy": per_gravity_acc,
        "per_tier_accuracy": per_tier_acc,
        "avg_output_tokens": avg_output_tokens,
        "max_output_tokens": max_output_tokens,
        "avg_latency": avg_latency,
        "total_cost": round(total_cost, 4) if total_cost else (0.0 if total_cost == 0.0 else None),
        "failure_count": failure_count,
        "failure_taxonomy": taxonomy_counts,
    }


def _compute_aggregated_metrics(results_matrix: Dict[str, Any], seeds,
                                fracture_cache: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Replicates BenchmarkRunner._compute_aggregated_metrics."""
    from arcus.evaluation.metrics import (
        aggregate_trajectory_results,
        LatencyDiagnostics,
        ARCUSRobustnessIndex,
        mean,
    )
    from arcus.analysis.fracture import FracturePointFinder

    if not results_matrix:
        return {
            "step_accuracy": 0.0, "exact_match": 0.0, "continuity": 0.0,
            "horizon_compliance": 0.0, "generation_bloat_index": 0.0,
            "efficiency": 0.0, "fracture_depth": 0, "mean_overshoot": 0.0,
        }

    traj_results = []
    total_runs = len(results_matrix)

    for result in results_matrix.values():
        step_acc = _coerce_float(result.get("step_accuracy", 0.0))
        exact = 1.0 if result.get("exact_match", False) else 0.0
        continuity = _coerce_float(result.get("continuity_score", 0.0))
        traj_results.append({
            "step_accuracy": step_acc,
            "exact_match": bool(exact),
            "continuity_score": continuity,
            "predicted_length": result.get("predicted_length", 0),
            "ground_truth_length": result.get("ground_truth_length", 0),
        })

    agg = aggregate_trajectory_results(traj_results)

    all_hc = [
        _coerce_float(res.get("horizon_compliance", 0.0))
        for res in results_matrix.values() if "horizon_compliance" in res
    ]
    all_gbi = [
        _coerce_float(res.get("generation_bloat_index", 0.0))
        for res in results_matrix.values() if "generation_bloat_index" in res
    ]
    all_ge = [
        _coerce_float(res.get("generation_efficiency", 0.0))
        for res in results_matrix.values() if "generation_efficiency" in res
    ]

    # Fracture depth: read from the stored fracture_cache (keyed by
    # "<gravity>_final") exactly as BenchmarkRunner._compute_aggregated_metrics
    # does, falling back to per-probe fracture_depth values when absent.
    final_fractures = []
    if fracture_cache:
        for key, depth in fracture_cache.items():
            if key.endswith("_final") and _coerce_float(depth) > 0:
                final_fractures.append(_coerce_float(depth))
    if not final_fractures:
        final_fractures = [
            _coerce_float(res.get("fracture_depth", 0))
            for res in results_matrix.values()
            if _coerce_float(res.get("fracture_depth", 0)) > 0
        ]
    fracture_depth = min(final_fractures) if final_fractures else 0

    # Failure Analysis (from the real error taxonomy)
    failure_analysis = {
        "State Tracking Failure": 0,
        "Transition Rule Failure": 0,
        "Semantic Interpretation Failure": 0,
        "Horizon Collapse": 0,
        "Formatting Failure": 0,
        "None": 0,
    }
    for res in results_matrix.values():
        mode = res.get("error_mode", "Unknown")
        if mode in failure_analysis:
            failure_analysis[mode] += 1
        else:
            failure_analysis["None"] += 1

    for k in failure_analysis:
        failure_analysis[k] = (failure_analysis[k] / total_runs) * 100

    # Per-tier aggregation
    per_tier: Dict[Any, Dict[str, Any]] = {}
    for key, result in results_matrix.items():
        z, gravity, tier, probe_idx, grid_key = _unpack_matrix_key(key)
        if tier not in per_tier:
            per_tier[tier] = {"accuracies": [], "token_density_values": [], "depth_acc": {}}

        per_tier[tier]["accuracies"].append(_coerce_float(result.get("step_accuracy", 0.0)))

        total_tokens = result.get("total_tokens", 1)
        depth = result.get("fol_depth", 1)
        token_density = float(total_tokens) / float(depth) if depth and depth > 0 else 0.0
        per_tier[tier]["token_density_values"].append(token_density)

        # Collect per-depth accuracy so we can compute the canonical fracture
        # depth for this tier from its own accuracy curve.
        try:
            zi = int(float(z))
        except (TypeError, ValueError):
            zi = 0
        per_tier[tier]["depth_acc"].setdefault(zi, []).append(
            _coerce_float(result.get("step_accuracy", 0.0))
        )

    fracture_finder = FracturePointFinder(look_ahead_step=2)
    per_tier_summary = {}
    for tier, data in per_tier.items():
        # Build the per-tier accuracy curve (z, mean step_accuracy) and locate
        # the canonical fracture depth for that tier. A 2-step look-ahead is
        # used so a single anomalous high-accuracy outlier at a larger horizon
        # (e.g. a sparse z=23=1.0 point) does not mask the real collapse.
        curve = [
            (z, _mean(accs)) for z, accs in sorted(data["depth_acc"].items())
        ]
        tier_fracture = fracture_finder.compute_structural_yield(curve)
        per_tier_summary[f"tier_{tier}"] = {
            "accuracy": _mean(data["accuracies"]) if data["accuracies"] else 0.0,
            "token_density": _mean(data["token_density_values"]) if data["token_density_values"] else 0.0,
            "fracture_depth": float(tier_fracture) if tier_fracture is not None else 0.0,
        }

    # Latency diagnostics (systems-level only)
    latency_probes = []
    for key, result in results_matrix.items():
        z, gravity, tier, probe_idx, grid_key = _unpack_matrix_key(key)
        lat = result.get("latency_ms")
        if lat is None:
            lat = result.get("latency")
        if lat is None:
            continue
        latency_probes.append({
            "latency_ms": float(lat),
            "output_tokens": result.get("output_tokens", result.get("completion_tokens", 0)),
            "accuracy": _coerce_float(result.get("step_accuracy", 0.0)),
            "correct_transitions": int(result.get("correct_transitions", 0) or 0),
            "horizon": z,
        })
    latency_diagnostics = LatencyDiagnostics.aggregate(latency_probes)

    # CRI computation
    cri = _compute_cri(results_matrix, ARCUSRobustnessIndex, mean)

    return {
        "step_accuracy": round(agg["mean_step_accuracy"] * 100.0, 1),
        "exact_match": round(agg["exact_match_rate"] * 100.0, 1),
        "continuity": round(agg["mean_continuity"] * 100.0, 1),
        "horizon_compliance": round((_mean(all_hc) if all_hc else 0.0) * 100.0, 1),
        "generation_bloat_index": round(_mean(all_gbi) if all_gbi else 0.0, 2),
        "efficiency": round((_mean(all_ge) if all_ge else 0.0) * 100.0, 1),
        "failure_analysis": failure_analysis,
        "fracture_depth": fracture_depth,
        "per_tier": per_tier_summary,
        "cri": cri,
        "latency_diagnostics": latency_diagnostics,
    }


def _compute_cri(results_matrix: Dict[str, Any], ARCUSRobustnessIndex, mean) -> Dict[str, Any]:
    """Replicates BenchmarkRunner.compute_cri."""
    if not results_matrix:
        return {
            "cri": 0.0, "trajectory_fidelity": 0.0, "horizon_robustness": 0.0,
            "semantic_robustness": 0.0, "generation_efficiency": 0.0,
            "components": dict(ARCUSRobustnessIndex.WEIGHTS),
        }

    step_accs: List[float] = []
    cont_accs: List[float] = []
    exact_accs: List[float] = []
    ge_accs: List[float] = []
    horizon_points: List[Tuple[float, Optional[float]]] = []
    tier_accs = {0: [], 1: [], 2: [], 3: []}

    for key, result in results_matrix.items():
        z, gravity, tier, probe_idx, grid_key = _unpack_matrix_key(key)
        sa = _coerce_float(result.get("step_accuracy", 0.0))
        cs = _coerce_float(result.get("continuity_score", 0.0))
        em = 1.0 if result.get("exact_match", False) else 0.0
        step_accs.append(sa)
        cont_accs.append(cs)
        exact_accs.append(em)
        horizon_points.append((float(z), sa))
        try:
            ti = int(tier)
        except (TypeError, ValueError):
            ti = -1
        if ti in tier_accs:
            tier_accs[ti].append(sa)
        ge = result.get("generation_efficiency", None)
        if ge is not None:
            ge_accs.append(_coerce_float(ge))

    summary = ARCUSRobustnessIndex.compute(
        step_accuracy=mean(step_accs),
        continuity_score=mean(cont_accs),
        exact_match=mean(exact_accs),
        horizon_points=horizon_points,
        tier0_acc=mean(tier_accs[0]) if tier_accs[0] else None,
        tier1_acc=mean(tier_accs[1]) if tier_accs[1] else None,
        tier2_acc=mean(tier_accs[2]) if tier_accs[2] else None,
        tier3_acc=mean(tier_accs[3]) if tier_accs[3] else None,
        generation_efficiency=mean(ge_accs) if ge_accs else None,
    )
    return summary.as_dict()


# ---------------------------------------------------------------------------
# Report rendering (mirrors BenchmarkRunner._print_summary_report + chart)
# ---------------------------------------------------------------------------

def _bar(percentage: float, width: int = 20) -> str:
    """Render a progress bar of the given width (percentage in 0-100)."""
    filled = int(round(percentage / 100.0 * width))
    filled = max(0, min(width, filled))
    return "\u2588" * filled + "\u2591" * (width - filled)


def _render_report(data: Dict[str, Any]) -> str:
    results_matrix = data.get("results_matrix", {})
    model = data.get("model", "unknown")
    run_status = data.get("run_status", "COMPLETE")
    total = len(results_matrix)
    completed = data.get("completed_probes", total)
    seeds = data.get("seeds", [42])

    aggregated = _compute_aggregated_metrics(results_matrix, seeds, data.get("fracture_cache"))
    accuracy = aggregated["step_accuracy"]
    m = _compute_report_metrics(
        results_matrix, model, run_status, completed, total, accuracy
    )

    lines: List[str] = []
    lines += [
        "",
        "=" * 60,
        "ARCUS-X RUN SUMMARY",
        "=" * 60,
        f"Model:               {m['model']}",
        f"Run Status:          {m['run_status']}",
        f"Completed Probes:    {m['completed_probes']} / {m['total_probes']}",
        f"Accuracy:            {m['accuracy']:.1f}%",
        "",
        "Per-Horizon Accuracy:",
    ]
    for z in sorted(m["per_horizon_accuracy"].keys(), key=int):
        acc = m["per_horizon_accuracy"][z]
        lines.append(f"  z={z:<4}: {acc * 100:.1f}%")
    lines.append("")
    lines.append("Per-Gravity Accuracy:")
    for g, acc in m["per_gravity_accuracy"].items():
        lines.append(f"  gravity={g:<4}: {acc * 100:.1f}%")
    lines.append("")
    lines.append("Per-Tier Accuracy:")
    for t, acc in m["per_tier_accuracy"].items():
        lines.append(f"  {t:<8}: {acc * 100:.1f}%")
    lines.append("")
    cost_str = f"{m['total_cost']:.4f}" if m["total_cost"] is not None else "N/A"
    lines += [
        f"Total Provider Cost: {cost_str}",
        f"Average Output Tokens: {m['avg_output_tokens']}",
        f"Maximum Output Tokens: {m['max_output_tokens']}",
        f"Average Latency:       {m['avg_latency']}",
        f"Failure Count:         {m['failure_count']}",
        "",
        "Multi-Seed Variance (mean +/- std across master seeds):",
    ]
    if len(seeds) > 1:
        lines.append(f"  Seeds:               {seeds}")
    else:
        lines.append(f"  Seeds:               {seeds} (single seed; no variance reported)")
    lines.append("")
    lines.append("Failure Taxonomy:")
    for bucket, count in m.get("failure_taxonomy", {}).items():
        lines.append(f"  {bucket + ':':<22} {count}")
    lines.append("")

    # Latency Diagnostics (systems-level only)
    lat = aggregated.get("latency_diagnostics")
    if lat:
        lines += [
            "Latency Diagnostics:",
            "-" * 60,
            f"Mean Latency:        {lat.get('mean_latency_ms', 0.0):.3f} ms",
            f"Median Latency:      {lat.get('median_latency_ms', 0.0):.3f} ms",
            f"P95 Latency:         {lat.get('p95_latency_ms', 0.0):.3f} ms",
            f"Maximum Latency:     {lat.get('max_latency_ms', 0.0):.3f} ms",
            f"Mean Output Tokens/sec: {lat.get('mean_output_tokens_per_sec', 0.0):.3f}",
            f"Mean Correct Steps/sec: {lat.get('mean_correct_steps_per_sec', 0.0):.3f}",
            "",
            "Normalized Efficiency:",
            f"  ALE: {lat.get('ale', 0.0):.4f}",
            f"  Correct Steps/sec: {lat.get('correct_steps_per_sec', 0.0):.3f}",
            "",
            "Latency by Horizon:",
        ]
        for ph in lat.get("per_horizon", []):
            lines.append(f"  z={ph['horizon']}:")
            lines.append(f"    Mean Latency: {ph['mean_latency_ms']:.3f} ms")
            lines.append(f"    Correct Steps/sec: {ph['correct_steps_per_sec']:.3f}")
        lines.append("=" * 60)

    # ---- ARCUS-X EVALUATION CHART ----
    lines += [
        "",
        "=" * 64,
        "ARCUS-X EVALUATION CHART",
        "=" * 64,
        "",
        "Headline Metrics:",
    ]
    headline = [
        ("Step Accuracy", aggregated["step_accuracy"]),
        ("Exact Match", aggregated["exact_match"]),
        ("Continuity", aggregated["continuity"]),
        ("Horizon Compliance", aggregated["horizon_compliance"]),
        ("Efficiency", aggregated["efficiency"]),
    ]
    for name, val in headline:
        bar = _bar(val, 20)
        lines.append(f"  {name:<22} {bar:<20} {val:>5.1f}%")

    lines.append(f"\n  {'GBI':<22} {aggregated['generation_bloat_index']:>6.2f}  (lower is better)")

    cri = aggregated.get("cri", {})
    if isinstance(cri, dict):
        lines.append(f"  {'CRI':<22} {cri.get('cri', 0.0):>6.3f}")
        lines.append(f"  {'Trajectory Fidelity':<22} {cri.get('trajectory_fidelity', 0.0):>6.3f}")
        lines.append(f"  {'Horizon Robustness':<22} {cri.get('horizon_robustness', 0.0):>6.3f}")
        lines.append(f"  {'Semantic Robustness':<22} {cri.get('semantic_robustness', 0.0):>6.3f}")
        lines.append(f"  {'Generation Efficiency':<22} {cri.get('generation_efficiency', 0.0):>6.3f}")

    fd_val = aggregated["fracture_depth"]
    fd_str = str(int(fd_val)) if isinstance(fd_val, float) and fd_val.is_integer() else str(fd_val)
    lines.append(f"\n  Fracture Depth: {fd_str}")

    per_tier_data = aggregated.get("per_tier", {})
    if per_tier_data:
        lines.append("\nPer-Tier Fracture Depth:")
        for tier in sorted(per_tier_data.keys()):
            fdv = float(per_tier_data[tier].get("fracture_depth", 0.0))
            bar = _bar(fdv, 20)
            lines.append(f"  {tier:<22} {bar:<20} {fdv:>5.1f}")

    failure_analysis = aggregated.get("failure_analysis", {})
    if failure_analysis:
        lines.append("\nError Taxonomy (failure distribution):")
        for mode, pct in failure_analysis.items():
            bar = _bar(pct, 20)
            lines.append(f"  {mode:<30} {bar:<20} {pct:>5.1f}%")
    lines.append("=" * 64)

    return "\n".join(lines)


def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load and parse a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_quickstart(filepath: str) -> str:
    """Analyze a quickstart JSON file and return the metrics report string."""
    data = load_json_file(filepath)
    return _render_report(data)


# ---------------------------------------------------------------------------
# TXT file parsing logic
# ---------------------------------------------------------------------------

def parse_txt_entry(line: str) -> Dict[str, Any]:
    """Parse a single log line from TXT file into structured data."""
    data = {}
    if line.startswith("Parameters:"):
        params = line.split(": ")[1].split(", ")
        data["z"] = int(params[0].split()[1])
        data["gravity"] = float(params[1].split()[1])
        data["tier"] = int(params[2].split()[1])
        data["grid_key"] = params[3].split()[1]
    elif line.startswith("Actions:"):
        data["actions"] = line.split(": ")[1].strip("[]").split(", ")
    elif line.startswith("TransitionRules:"):
        data["transition_rules"] = line.split(": ")[1].strip()
    elif line.startswith("InitialState:"):
        data["initial_state"] = line.split(": ")[1].strip("[]").split(", ")
    elif line.startswith("Latency(ms):"):
        data["latency_ms"] = float(line.split(": ")[1])
    elif line.startswith("OutputTokens:"):
        data["output_tokens"] = int(line.split(": ")[1])
    return data


def process_txt_file(filepath: str) -> Dict[str, Any]:
    """Parse entire TXT file into results matrix format compatible with JSON structure."""
    results_matrix = {}
    current_key = None
    model = None
    timestamp = None
    total_probes_from_header = None
    probe_count = 0
    gravities_set = set()

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Parse header (RUN START)
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("ARCUS-X RUN START"):
            # Parse the next few lines for metadata
            i += 1
            while i < len(lines) and not lines[i].startswith("===="):
                header_line = lines[i].strip()
                if header_line.startswith("Run ID:"):
                    timestamp = header_line.split(": ")[1]
                elif header_line.startswith("Model:"):
                    model = header_line.split(": ")[1]
                elif header_line.startswith("Total Probes:"):
                    total_probes_from_header = int(header_line.split(": ")[1])
                i += 1
            continue
        i += 1

    # Parse probes
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("Parameters:"):
            # Parse the line to get z, gravity, tier, grid
            try:
                param_str = line.split(": ")[1]
                params = [p.strip() for p in param_str.split(",")]
                param_dict = {}
                for p in params:
                    if '=' in p:
                        key, value = p.split('=', 1)
                        param_dict[key.strip()] = value.strip()
                z = int(param_dict['z'])
                gravity = float(param_dict['gravity'])
                tier = int(param_dict['tier'])
                grid_key = param_dict['grid']
                key = f"{z}_{gravity}_{tier}_{grid_key}"
                current_key = key
                results_matrix[key] = {}
                probe_count += 1
                gravities_set.add(gravity)
            except (IndexError, ValueError, KeyError):
                # If we can't parse, skip this line and set current_key to None
                current_key = None
            i += 1
        elif current_key is not None:
            try:
                if line.startswith("Latency(ms):"):
                    value = float(line.split(": ")[1])
                    results_matrix[current_key]["latency_ms"] = value
                elif line.startswith("OutputTokens:"):
                    value = int(line.split(": ")[1])
                    results_matrix[current_key]["output_tokens"] = value
                elif line.startswith("[MODEL OUTPUT]:"):
                    value = line.split("[MODEL OUTPUT]:")[1].strip()
                    results_matrix[current_key]["raw_output"] = value
                elif line.startswith("[GROUND TRUTH EXPECTED]:"):
                    value = line.split("[GROUND TRUTH EXPECTED]:")[1].strip()
                    results_matrix[current_key]["ground_truth"] = value
                # We can add more fields as needed
            except (IndexError, ValueError):
                # If we can't parse this line, just skip it
                pass
            i += 1
        else:
            i += 1

    # Build the final data structure
    parameters = {
        "n_probes": probe_count,
        "gravity_levels": sorted(list(gravities_set)),
        "seed": 42,  # default seed
        "grid_size_levels": None,
    }

    data = {
        "model": model if model else "unknown",
        "timestamp": timestamp if timestamp else "unknown",
        "parameters": parameters,
        "results_matrix": results_matrix,
        "seeds": [42],  # default
        "fracture_cache": None,
    }

    return data


# ---------------------------------------------------------------------------
# Main function with file type detection
# ---------------------------------------------------------------------------
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Reproduce the ARCUS-X completed-run report from quickstart JSON files or TXT log files."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="A quickstart JSON file OR a directory containing quickstart_results_*.json files OR a TXT log file"
    )
    parser.add_argument(
        "--pattern",
        default="quickstart_results_*.json",
        help="Glob pattern when --input is a directory (default: quickstart_results_*.json)"
    )

    args = parser.parse_args()
    input_path = args.input
    pattern = args.pattern

    files: List[str] = []
    if os.path.isdir(input_path):
        search = os.path.join(input_path, pattern)
        files = sorted(glob.glob(search))
    elif os.path.isfile(input_path):
        files = [input_path]
    else:
        print(f"Input path not found: {input_path}")
        sys.exit(1)

    if not files:
        print(f"No files found matching pattern '{pattern}' in '{input_path}'")
        sys.exit(1)

    for filepath in files:
        print(f"\n{'=' * 60}\nProcessing: {filepath}\n{'=' * 60}")
        try:
            if filepath.endswith(".json"):
                data = load_json_file(filepath)
            elif filepath.endswith(".txt"):
                data = process_txt_file(filepath)
            else:
                raise ValueError(f"Unsupported file type: {filepath}")
            report = _render_report(data)
        except Exception as exc:
            print(f"Failed to analyze {filepath}: {exc}")
            continue
        print(report)


if __name__ == "__main__":
    main()
