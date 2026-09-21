#!/usr/bin/env python3
"""
Post-run analyzer for ARCUS-X benchmark results.

This script reads quickstart JSON result files (and the corresponding
absolute_raw_stream_*.txt files) and reproduces the exact terminal report
that a completed benchmark run emits -- the "ARCUS-X RUN SUMMARY" block
followed by the "ARCUS-X EVALUATION CHART" block.

It reuses the same aggregation logic that ``arcus.experiments.benchmark_runner``
uses internally (``_compute_report_metrics``, ``_compute_aggregated_metrics``,
``compute_cri``) so the numbers and formatting match a live run.

USAGE:
    python post_run_analyzer.py --input <path> [--pattern <glob>]

    --input   Path to a single quickstart JSON file OR a directory containing
              one or more quickstart_results_*.json files.
    --pattern Glob pattern used when --input is a directory
              (default: quickstart_results_*.json).

EXAMPLES:
    # Analyze a single quickstart result file
    python post_run_analyzer.py --input outputs/quickstart_results_20260718_103856.json

    # Analyze every quickstart result file in a directory
    python post_run_analyzer.py --input outputs/

    # Analyze with a custom glob pattern
    python post_run_analyzer.py --input outputs/ --pattern "quickstart_results_*.json"

The report printed to the terminal is byte-for-byte equivalent (modulo the
per-tier accuracies, which are recomputed from the stored probe results) to
the report a completed run would have produced.
"""

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Force UTF-8 output on Windows (cp1252 cannot encode the block glyphs).
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
# Re-implement the small helpers used by benchmark_runner so this script has no
# hard dependency on an instantiated BenchmarkRunner object.
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

    # Fracture depth: the quickstart JSON does not carry fracture_cache, so we
    # derive it from the per-probe fracture_depth values that the runner now
    # propagates onto each result (or fall back to 0 when absent).
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
    per_tier: Dict[Any, Dict[str, List[float]]] = {}
    for key, result in results_matrix.items():
        z, gravity, tier, probe_idx, grid_key = _unpack_matrix_key(key)
        if tier not in per_tier:
            per_tier[tier] = {"accuracies": [], "token_density_values": [], "fracture_depths": []}

        per_tier[tier]["accuracies"].append(_coerce_float(result.get("step_accuracy", 0.0)))

        total_tokens = result.get("total_tokens", 1)
        depth = result.get("fol_depth", 1)
        token_density = float(total_tokens) / float(depth) if depth and depth > 0 else 0.0
        per_tier[tier]["token_density_values"].append(token_density)

        fd = _coerce_float(result.get("fracture_depth", 0))
        if fd and fd > 0:
            per_tier[tier]["fracture_depths"].append(fd)

    per_tier_summary = {}
    for tier, data in per_tier.items():
        per_tier_summary[f"tier_{tier}"] = {
            "accuracy": _mean(data["accuracies"]) if data["accuracies"] else 0.0,
            "token_density": _mean(data["token_density_values"]) if data["token_density_values"] else 0.0,
            "fracture_depth": _mean(data["fracture_depths"]) if data["fracture_depths"] else 0.0,
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

    aggregated = _compute_aggregated_metrics(results_matrix, seeds)
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
    for z, acc in m["per_horizon_accuracy"].items():
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

    lines.append(f"\n  Fracture Depth: {aggregated['fracture_depth']}")

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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Reproduce the ARCUS-X completed-run report from quickstart JSON files."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="A quickstart JSON file OR a directory containing quickstart_results_*.json files",
    )
    parser.add_argument(
        "--pattern",
        default="quickstart_results_*.json",
        help="Glob pattern when --input is a directory (default: quickstart_results_*.json)",
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
        print(f"\n{'=' * 60}")
        print(f"Processing: {filepath}")
        print("=" * 60)
        try:
            report = analyze_quickstart(filepath)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Failed to analyze {filepath}: {exc}")
            continue
        print(report)


if __name__ == "__main__":
    main()
