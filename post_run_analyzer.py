#!/usr/bin/env python3
"""
Post-run analyzer for ARCUS-X benchmark results.

This script now handles both JSON result files and TXT log files. For TXT files,
it parses the structured ``=== RAW STREAM ENTRY ===`` blocks and scores each
probe with the *same* evaluation pipeline the live runner uses
(``compare_trajectories`` + ``classify_from_result`` + the efficiency metrics),
so the post-run report is faithful to a live run without re-calling the model.
"""

import argparse
import ast
import glob
import json
import os
import re
import sys
import statistics
from typing import Any, Dict, List, Optional, Tuple
from arcus.analysis.reporting import (
    TAXONOMY_ALIAS_MAP,
    canonicalize_taxonomy,
    is_fully_correct,
    normalize_taxonomy_label,
)

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
# Taxonomy alias mapping and validity classification constants
# ---------------------------------------------------------------------------
CANONICAL_TAXONOMY_BUCKETS = (
    "State Tracking Failure",
    "Transition Rule Failure",
    "Semantic Interpretation Failure",
    "Horizon Collapse",
    "Formatting Failure",
    "Unknown / Unmapped",
    "None",
)


_normalize_taxonomy_label = normalize_taxonomy_label
_is_ssot_fully_correct = is_fully_correct
_canonicalize_taxonomy = canonicalize_taxonomy

class ExceptionCategory:
    EXCEPTION_TOKEN = "EXCEPTION_TOKEN"
    EMPTY_OUTPUT = "EMPTY_OUTPUT"
    PARSE_FAILURE = "PARSE_FAILURE"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    UNKNOWN_INVALID = "UNKNOWN_INVALID"

class ErrorCode:
    E100 = "E100"
    E101 = "E101"
    E102 = "E102"
    E103 = "E103"
    E104 = "E104"

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

    Mirrors ``BenchmarkRunner._unpack_matrix_key``. Supports 4, 5, or 6 part keys
    (including multi-seed prefix or grid key).
    """
    parts = str(key).split("_")
    if len(parts) == 6:
        return (parts[1], parts[2], parts[3], parts[4], parts[5])
    if len(parts) == 5:
        if "x" in parts[4] or parts[4] == "default":
            return (parts[0], parts[1], parts[2], parts[3], parts[4])
        else:
            return (parts[1], parts[2], parts[3], parts[4], "default")
    if len(parts) == 4:
        return (parts[0], parts[1], parts[2], parts[3], "default")
    return (parts[0], parts[1], parts[2], parts[3], parts[4] if len(parts) > 4 else "default")


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _avg_round1(vals: List[float]) -> float:
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def _avg_round4(vals: List[float]) -> float:
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _compute_report_metrics(results_matrix: Dict[str, Any], model: str,
                            run_status: str, completed: int, total: int,
                            accuracy: float) -> Dict[str, Any]:
    """Replicates BenchmarkRunner._compute_report_metrics with binary validity."""
    per_horizon: Dict[Any, List[float]] = {}
    per_horizon_error_counts: Dict[Any, int] = {}
    per_horizon_error_codes: Dict[Any, Dict[str, int]] = {}
    per_gravity: Dict[Any, List[float]] = {}
    per_tier: Dict[Any, List[float]] = {}

    output_tokens_list: List[float] = []
    latency_list: List[float] = []
    total_cost = 0.0
    failure_count = 0
    valid_count = 0
    invalid_count = 0

    output_compliance_counts = {
        "Perfect Format": 0,
        "Verbose Correct": 0,
        "Malformed": 0,
        "Missing Trajectory": 0,
        "Parse Failure": 0,
    }
    cog_taxonomy_counts_new = {
        "State Tracking": 0,
        "Transition": 0,
        "Semantic": 0,
        "Horizon": 0,
        "None": 0,
    }
    taxonomy_counts = {
        "State Tracking Failure": 0,
        "Transition Rule Failure": 0,
        "Semantic Interpretation Failure": 0,
        "Horizon Collapse": 0,
        "Formatting Failure": 0,
        "Unknown / Unmapped": 0,
        "None": 0,
    }
    invalid_category_counts = {
        ExceptionCategory.EXCEPTION_TOKEN: 0,
        ExceptionCategory.EMPTY_OUTPUT: 0,
        ExceptionCategory.PARSE_FAILURE: 0,
        ExceptionCategory.PROVIDER_ERROR: 0,
        ExceptionCategory.UNKNOWN_INVALID: 0,
    }
    error_code_counts = {
        ErrorCode.E100: 0,
        ErrorCode.E101: 0,
        ErrorCode.E102: 0,
        ErrorCode.E103: 0,
        ErrorCode.E104: 0,
    }
    exact_match_count = 0
    fully_correct_valid_probe_count = 0
    unknown_taxonomy_audit = []
    total_probes_counted = len(results_matrix)

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

        raw_output = str(result.get("raw_output", "") or "")
        is_infra = result.get("is_infrastructure_failure", False) or result.get("provider_failure", False) or result.get("api_error") is not None or result.get("finish_reason") == "error"
        error_mode = result.get("error_mode", "Unknown")

        pc = result.get("probe_classification")
        if not pc:
            tax = classify_from_result(
                result.get("predicted_path", []),
                result.get("ground_truth_path", []),
                grid_width=result.get("grid_width"),
                grid_height=result.get("grid_height"),
                raw_output=raw_output,
            )
            pc = tax.probe_classification.to_dict() if tax.probe_classification else {"valid": True, "exception_code": None, "mechanism": "None"}

        oc = pc.get("output_compliance", "Verbose Correct")
        if oc == "Perfect Format":
            output_compliance_counts["Perfect Format"] += 1
        elif oc == "Verbose Correct":
            output_compliance_counts["Verbose Correct"] += 1
        elif oc == "Malformed Output":
            output_compliance_counts["Malformed"] += 1
        elif oc == "No Trajectory":
            output_compliance_counts["Missing Trajectory"] += 1
        elif oc == "Parse Failure":
            output_compliance_counts["Parse Failure"] += 1
        else:
            output_compliance_counts["Verbose Correct"] += 1

        mech = pc.get("mechanism", "None")
        is_success = _is_ssot_fully_correct(result, pc)

        if is_success:
            cog_taxonomy_counts_new["None"] += 1
        elif mech in ("State Tracking Failure", "State Tracking", "Unknown Failure", "UNKNOWN_FAILURE", "Unknown"):
            cog_taxonomy_counts_new["State Tracking"] += 1
        elif mech in ("Transition Failure", "Transition"):
            cog_taxonomy_counts_new["Transition"] += 1
        elif mech in ("Semantic Failure", "Semantic"):
            cog_taxonomy_counts_new["Semantic"] += 1
        elif mech in ("Horizon Failure", "Horizon"):
            cog_taxonomy_counts_new["Horizon"] += 1
        else:
            cog_taxonomy_counts_new["State Tracking"] += 1

        if result.get("exact_match", False) or pc.get("exact_match", False):
            exact_match_count += 1

        is_valid = pc.get("valid", True) and not is_infra
        if not is_valid:
            invalid_count += 1
            exc = pc.get("exception_code", ExceptionCategory.UNKNOWN_INVALID)
            if ExceptionCategory.EXCEPTION_TOKEN in str(exc):
                error_code = ErrorCode.E100
                invalid_category_counts[ExceptionCategory.EXCEPTION_TOKEN] += 1
            elif ExceptionCategory.EMPTY_OUTPUT in str(exc):
                error_code = ErrorCode.E101
                invalid_category_counts[ExceptionCategory.EMPTY_OUTPUT] += 1
            elif ExceptionCategory.PARSE_FAILURE in str(exc):
                error_code = ErrorCode.E102
                invalid_category_counts[ExceptionCategory.PARSE_FAILURE] += 1
            elif ExceptionCategory.PROVIDER_ERROR in str(exc):
                error_code = ErrorCode.E103
                invalid_category_counts[ExceptionCategory.PROVIDER_ERROR] += 1
            else:
                error_code = ErrorCode.E104
                invalid_category_counts[ExceptionCategory.UNKNOWN_INVALID] += 1
            error_code_counts[error_code] += 1
            per_horizon_error_counts[z] = per_horizon_error_counts.get(z, 0) + 1
            horizon_code_counts = per_horizon_error_codes.setdefault(
                z, {code: 0 for code in error_code_counts}
            )
            horizon_code_counts[error_code] += 1
        else:
            valid_count += 1
            is_fully_correct = _is_ssot_fully_correct(result, pc)
            canonical_mode = _canonicalize_taxonomy(error_mode, is_fully_correct)
            if is_fully_correct:
                fully_correct_valid_probe_count += 1
            if canonical_mode == "None" and not is_fully_correct:
                raise AssertionError(
                    f"Taxonomy classification violated SSOT correctness for probe {key}: "
                    f"canonical_mode was None when trajectory was not fully correct; "
                    f"step_accuracy={acc}, exact_match={result.get('exact_match', False)}, "
                    f"probe_classification={pc}"
                )
            taxonomy_counts[canonical_mode] += 1
            if canonical_mode == "Unknown / Unmapped":
                unknown_taxonomy_audit.append({
                    "probe_id": key,
                    "raw_taxonomy": str(error_mode),
                    "normalized_taxonomy": _normalize_taxonomy_label(error_mode),
                    "step_accuracy": acc,
                    "probe_classification": pc,
                })
            elif canonical_mode not in ("None", "Unknown / Unmapped"):
                failure_count += 1

    # Enforce full accounting invariants
    if (valid_count + invalid_count) != completed:
        raise AssertionError(
            f"Validity partition invariant violated: valid ({valid_count}) + invalid ({invalid_count}) != completed ({completed})"
        )
    total_tax = sum(taxonomy_counts.values())
    if total_tax != valid_count:
        raise AssertionError(
            f"Taxonomy invariant violated: sum of taxonomy counts ({total_tax}) != valid probes ({valid_count})"
        )
    if taxonomy_counts.get("None", 0) != fully_correct_valid_probe_count:
        raise AssertionError(
            f"None-count invariant violated: taxonomy_counts['None']={taxonomy_counts.get('None', 0)} != "
            f"fully_correct_valid_probe_count={fully_correct_valid_probe_count}. "
            f"sample_unknowns={unknown_taxonomy_audit[:5]}"
        )

    per_horizon_acc = {str(k): _avg_round1(v) for k, v in sorted(per_horizon.items())}
    per_horizon_density = {
        str(k): round(len(values) / total_probes_counted * 100.0, 1)
        for k, values in sorted(per_horizon.items())
    } if total_probes_counted else {}
    per_horizon_error_density = {
        str(k): round(per_horizon_error_counts.get(k, 0) / len(values) * 100.0, 1)
        for k, values in sorted(per_horizon.items())
        if per_horizon_error_counts.get(k, 0) > 0
    }
    per_horizon_probe_counts = {
        str(k): len(values)
        for k, values in sorted(per_horizon.items())
    }
    per_horizon_error_codes_report = {
        str(k): per_horizon_error_codes[k]
        for k in sorted(per_horizon_error_codes)
    }
    per_gravity_acc = {str(k): _avg_round4(v) for k, v in sorted(per_gravity.items(), key=lambda kv: float(kv[0]))}
    per_tier_acc = {f"tier_{k}": _avg_round4(v) for k, v in sorted(per_tier.items())}

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
        "per_horizon_density": per_horizon_density,
        "per_horizon_probe_counts": per_horizon_probe_counts,
        "per_horizon_error_density": per_horizon_error_density,
        "per_horizon_error_codes": per_horizon_error_codes_report,
        "per_gravity_accuracy": per_gravity_acc,
        "per_tier_accuracy": per_tier_acc,
        "avg_output_tokens": avg_output_tokens,
        "max_output_tokens": max_output_tokens,
        "avg_latency": avg_latency,
        "total_cost": round(total_cost, 4) if total_cost else (0.0 if total_cost == 0.0 else None),
        "failure_count": failure_count,
        "failure_taxonomy": taxonomy_counts,
        "invalid_category_counts": invalid_category_counts,
        "error_code_counts": error_code_counts,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "unknown_taxonomy_audit": unknown_taxonomy_audit,
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
    
    # If still no fracture depth, compute from global accuracy curve
    # (same approach as per-tier fracture depth computation)
    if not final_fractures:
        global_depth_acc = {}
        for key, result in results_matrix.items():
            z, gravity, tier, probe_idx, grid_key = _unpack_matrix_key(key)
            try:
                zi = int(float(z))
            except (TypeError, ValueError):
                zi = 0
            global_depth_acc.setdefault(zi, []).append(
                _coerce_float(result.get("step_accuracy", 0.0))
            )
        
        if global_depth_acc:
            curve = [
                (z, _mean(accs)) for z, accs in sorted(global_depth_acc.items())
            ]
            fracture_finder = FracturePointFinder(look_ahead_step=2)
            global_fracture = fracture_finder.compute_structural_yield(curve)
            if global_fracture is not None and global_fracture > 0:
                final_fractures.append(global_fracture)
    
    fracture_depth = min(final_fractures) if final_fractures else None

    # Failure Analysis (from the real error taxonomy)
    failure_analysis = {
        "State Tracking Failure": 0,
        "Transition Rule Failure": 0,
        "Semantic Interpretation Failure": 0,
        "Horizon Collapse": 0,
        "Formatting Failure": 0,
        "Unknown / Unmapped": 0,
        "None": 0,
    }
    valid_failure_count = 0
    for res in results_matrix.values():
        raw_mode = res.get("error_mode", "Unknown")
        pc = res.get("probe_classification")
        acc = _coerce_float(res.get("step_accuracy", 0.0))
        is_infra = res.get("is_infrastructure_failure", False) or res.get("provider_failure", False) or res.get("api_error") is not None or res.get("finish_reason") == "error"
        is_valid = True
        if pc and isinstance(pc, dict) and not pc.get("valid", True):
            is_valid = False
        if not is_valid or is_infra:
            continue

        valid_failure_count += 1
        normalized_mode = str(raw_mode).strip().lower()
        is_fully_correct = _is_ssot_fully_correct(res, pc)
        canonical_mode = _canonicalize_taxonomy(raw_mode, is_fully_correct)

        if canonical_mode in failure_analysis:
            failure_analysis[canonical_mode] += 1
        else:
            failure_analysis["Unknown / Unmapped"] += 1

    denom = valid_failure_count if valid_failure_count > 0 else 1
    for k in failure_analysis:
        failure_analysis[k] = (failure_analysis[k] / denom) * 100

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
            "fracture_depth": float(tier_fracture) if tier_fracture is not None else None,
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


def _render_report(
    data: Dict[str, Any],
    show_density: bool = False,
    show_error_density: bool = False,
) -> str:
    results_matrix = data.get("results_matrix", {})
    model = data.get("model", "unknown")
    run_status = data.get("run_status", "COMPLETE")
    total = len(results_matrix)
    completed = data.get("completed_probes", total)
    seeds = data.get("seeds") or data.get("parameters", {}).get("seeds") or [42]

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
    for z in sorted(m["per_horizon_accuracy"].keys(), key=lambda x: float(x)):
        acc = m["per_horizon_accuracy"][z]
        lines.append(f"  z={z:<4}: {acc * 100:.1f}%")
    if show_density:
        lines.append("")
        lines.append("Probe Density by Horizon:")
        for z in sorted(m["per_horizon_density"].keys(), key=lambda x: float(x)):
            density = m["per_horizon_density"][z]
            lines.append(f"  z={z:<4}: {density:.1f}%")
    if show_error_density:
        lines.append("")
        lines.append("Error Density by Horizon:")
        for z in sorted(m["per_horizon_error_density"].keys(), key=lambda x: float(x)):
            error_density = m["per_horizon_error_density"][z]
            total_probes = m["per_horizon_probe_counts"][z]
            code_counts = m["per_horizon_error_codes"].get(z, {})
            codes = "   ".join(
                f"{code}: {code_counts.get(code, 0)}"
                for code in (ErrorCode.E100, ErrorCode.E101, ErrorCode.E102, ErrorCode.E103, ErrorCode.E104)
            )
            lines.append(
                f"  z={z:<4} {total_probes:>4} probes   {codes}   "
                f"error density: {error_density:.1f}%"
            )
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
        # Compute per-seed CRI and fracture depth for variance reporting
        from arcus.evaluation.metrics import ARCUSRobustnessIndex, mean
        per_seed_cri = []
        per_seed_fd = []
        for seed in seeds:
            # Filter results_matrix for this seed
            seed_results = {
                k: v for k, v in results_matrix.items()
                if v.get("seed") == seed or str(k).startswith(f"{seed}_")
            }
            if seed_results:
                # Compute CRI for this seed
                step_accs = []
                cont_accs = []
                exact_accs = []
                ge_accs = []
                horizon_points = []
                tier_accs = {0: [], 1: [], 2: [], 3: []}
                for key, result in seed_results.items():
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
                cri_summary = ARCUSRobustnessIndex.compute(
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
                per_seed_cri.append(cri_summary.cri)
                # Fracture depth for this seed
                final_fractures = []
                for key, result in seed_results.items():
                    fd = _coerce_float(result.get("fracture_depth", 0))
                    if fd > 0:
                        final_fractures.append(fd)
                if final_fractures:
                    per_seed_fd.append(min(final_fractures))
        if per_seed_cri:
            cri_mean = mean(per_seed_cri)
            cri_std = statistics.pstdev(per_seed_cri) if len(per_seed_cri) > 1 else 0.0
            lines.append(f"  CRI:                 {cri_mean:.3f} +/- {cri_std:.3f}")
        if per_seed_fd:
            fd_mean = mean(per_seed_fd)
            fd_std = statistics.pstdev(per_seed_fd) if len(per_seed_fd) > 1 else 0.0
            lines.append(f"  Fracture Depth:      {fd_mean:.1f} +/- {fd_std:.1f}")
    else:
        lines.append(f"  Seeds:               {seeds} (single seed; no variance reported)")
    lines.append("")
    valid_count = m.get("valid_count", total)
    invalid_count = m.get("invalid_count", 0)
    valid_pct = (valid_count / total * 100.0) if total > 0 else 100.0
    invalid_pct = (invalid_count / total * 100.0) if total > 0 else 0.0

    lines.append("Response Validity:")
    lines.append(f"  VALID:   {valid_count} ({valid_pct:.1f}%)")
    lines.append(f"  INVALID: {invalid_count} ({invalid_pct:.1f}%)")
    lines.append("")

    lines.append("Invalid Breakdown:")
    invalid_cats = [
        (ErrorCode.E100, "E100  Exception Token", m.get("invalid_category_counts", {}).get(ExceptionCategory.EXCEPTION_TOKEN, 0)),
        (ErrorCode.E101, "E101  Empty Output      ", m.get("invalid_category_counts", {}).get(ExceptionCategory.EMPTY_OUTPUT, 0)),
        (ErrorCode.E102, "E102  Parse Failure     ", m.get("invalid_category_counts", {}).get(ExceptionCategory.PARSE_FAILURE, 0)),
        (ErrorCode.E103, "E103  Provider Error    ", m.get("invalid_category_counts", {}).get(ExceptionCategory.PROVIDER_ERROR, 0)),
        (ErrorCode.E104, "E104  Unknown           ", m.get("invalid_category_counts", {}).get(ExceptionCategory.UNKNOWN_INVALID, 0)),
    ]
    for code, label, count in invalid_cats:
        lines.append(f"  {label:<22} {count}")
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


    per_tier_data = aggregated.get("per_tier", {})
    if per_tier_data:
        lines.append("\nPer-Tier Fracture Depth:")
        for tier in sorted(per_tier_data.keys()):
            fdv = per_tier_data[tier].get("fracture_depth")
            if fdv is not None:
                fdv_f = float(fdv)
                bar = _bar(fdv_f, 20)
                fdv_str = f"{fdv_f:>5.1f}"
            else:
                bar = "-" * 20
                fdv_str = "  N/A"
            lines.append(f"  {tier:<22} {bar:<20} {fdv_str}")

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

# Import the real evaluation pipeline so TXT logs are scored identically to the
# live benchmark runner (no re-running the model required).
from arcus.evaluation.parser import extract_model_path, estimated_optimal_path_length  # noqa: E402
from arcus.evaluation.metrics import (  # noqa: E402
    compare_trajectories,
    HorizonCompliance,
    GenerationBloatIndex,
    GenerationEfficiency,
)
from arcus.analysis.taxonomy import classify_from_result  # noqa: E402


def _parse_coord_block(block: str) -> List[str]:
    """Parse a multi-line ``[x,y]`` block into canonical tokens."""
    tokens: List[str] = []
    for raw in block.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        # Tolerate a leading bullet / dash / comma.
        cleaned = raw.lstrip("-•* ").rstrip(",")
        if cleaned.startswith("[") and "," in cleaned:
            tokens.append(cleaned)
    return tokens


def _parse_grid(grid_str: str) -> Tuple[int, int]:
    """Parse ``WxH`` into ``(width, height)``."""
    try:
        w, h = grid_str.lower().split("x")
        return int(w.strip()), int(h.strip())
    except (ValueError, AttributeError):
        return 0, 0


def _parse_actions(actions_value: Any) -> List[str]:
    """Parse a Python list-literal string, tolerating scalar action metadata."""
    try:
        parsed = ast.literal_eval(actions_value) if isinstance(actions_value, str) else actions_value
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    return [str(action) for action in parsed]


def _parse_transition_rules(rules_str: str) -> Optional[Dict[str, Dict[str, int]]]:
    """Parse a Python dict-literal string of transition rules."""
    try:
        return ast.literal_eval(rules_str)
    except (ValueError, SyntaxError):
        return None


def _score_probe(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the full trajectory metric bundle for a single parsed probe.

    Mirrors ``BenchmarkRunner._evaluate_depth_batch`` scoring so that the
    post-run report is identical to a live run.
    """
    z = entry.get("z", 0)
    gravity = entry.get("gravity", 0.0)
    tier = entry.get("tier", 0)
    grid_key = entry.get("grid_key", "default")
    seed = entry.get("seed")
    raw_output = entry.get("raw_output", "") or ""
    ground_truth_block = entry.get("ground_truth", "") or ""

    predicted_path = extract_model_path(raw_output)
    ground_truth_path = _parse_coord_block(ground_truth_block)

    traj = compare_trajectories(predicted_path, ground_truth_path)

    gt_len = len(ground_truth_path)
    overlap = min(len(predicted_path), gt_len)
    correct_transitions = sum(
        1 for i in range(overlap) if predicted_path[i] == ground_truth_path[i]
    )

    transition_rules = _parse_transition_rules(entry.get("transition_rules", "") or "")
    actions = _parse_actions(entry.get("actions", "") or "")
    grid_width, grid_height = _parse_grid(entry.get("grid", "") or "")

    tax = classify_from_result(
        predicted_path,
        ground_truth_path,
        transition_rules,
        actions,
        tier,
        grid_width,
        grid_height,
        raw_output,
    )
    error_mode = tax.legacy_mode.value if tax.valid else "Unknown"
    if tax.valid and tax.legacy_mode.value == "None" and (traj.step_accuracy < 1.0 or not traj.exact_match):
        error_mode = "Unknown"

    generated_depth = max(0, len(predicted_path) - 1)
    horizon_compliance = HorizonCompliance.calculate(z, generated_depth)

    output_tokens = int(entry.get("output_tokens", 0) or 0)
    baseline_tokens = estimated_optimal_path_length(z)
    gbi = GenerationBloatIndex.calculate(output_tokens, baseline_tokens)
    generation_efficiency = GenerationEfficiency.calculate(gbi)

    return {
        "z": z,
        "gravity": gravity,
        "tier": tier,
        "grid_key": grid_key,
        "seed": seed,
        "step_accuracy": traj.step_accuracy,
        "exact_match": traj.exact_match,
        "continuity_score": traj.continuity_score,
        "first_divergence": traj.first_divergence,
        "final_state_correct": traj.final_state_correct,
        "predicted_length": traj.predicted_length,
        "ground_truth_length": traj.ground_truth_length,
        "correct_transitions": correct_transitions,
        "error_mode": error_mode,
        "horizon_compliance": horizon_compliance,
        "generation_bloat_index": gbi,
        "generation_efficiency": generation_efficiency,
        "fol_depth": z,
        "latency_ms": _coerce_float(entry.get("latency_ms", 0.0)),
        "output_tokens": output_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": output_tokens,
        "input_tokens": int(entry.get("input_tokens", 0) or 0),
        "finish_reason": "truncated" if "[OUTPUT_TRUNCATED]" in raw_output else "stop",
        "raw_output": raw_output,
        "predicted_path": predicted_path,
        "ground_truth_path": ground_truth_path,
        "grid_width": grid_width,
        "grid_height": grid_height,
        "probe_classification": tax.probe_classification.to_dict() if tax.probe_classification else {"valid": tax.valid, "exception_code": None, "mechanism": "None"},
    }


def process_txt_file(filepath: str, last_run_only: bool = False) -> Dict[str, Any]:
    """Parse an ARCUS-X absolute raw data TXT log into a results matrix.

    Each ``=== RAW STREAM ENTRY ===`` block is parsed for its environment
    metadata, then scored with the real evaluation pipeline (the same
    ``compare_trajectories`` / ``classify_from_result`` / efficiency metrics the
    live runner uses) so the post-run report is faithful to a live run.
    """
    results_matrix: Dict[str, Any] = {}
    model = None
    timestamp = None
    total_probes_from_header = None
    probe_count = 0
    gravities_set = set()
    # TXT logs repeat the same (z, gravity, tier, grid) combo for many
    # probes/seeds, so we append a per-combo running index to keep keys unique
    # (mirroring the 5-part JSON keys like "20_0.5_1_0").
    combo_counter: Dict[str, int] = {}

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if last_run_only:
        run_starts = [idx for idx, line in enumerate(lines) if "ARCUS-X RUN START" in line]
        if run_starts:
            lines = lines[run_starts[-1]:]

    # First pass: collect header metadata.
    for line in lines:
        s = line.strip()
        if s.startswith("Run ID:"):
            timestamp = s.split(": ", 1)[1].strip()
        elif s.startswith("Model:"):
            model = s.split(": ", 1)[1].strip()
        elif s.startswith("Total Probes:"):
            try:
                total_probes_from_header = int(float(s.split(": ", 1)[1].strip()))
            except ValueError:
                pass

    # Second pass: walk entry blocks.
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if not line.startswith("=== RAW STREAM ENTRY"):
            i += 1
            continue

        entry: Dict[str, Any] = {}
        i += 1
        # Read until the closing ``==============================`` separator.
        block_lines: List[str] = []
        while i < n and not lines[i].strip().startswith("==="):
            block_lines.append(lines[i])
            i += 1
        # Skip the separator line itself.
        if i < n:
            i += 1

        # Parse the structured fields within the block.
        section = None  # "output" | "truth" | None
        for bl in block_lines:
            bl_stripped = bl.strip()
            if bl_stripped.startswith("EnvironmentMetadata:"):
                meta_str = bl_stripped.split(":", 1)[1]
                for item in meta_str.split(","):
                    if "=" in item:
                        k, v = item.split("=", 1)
                        k, v = k.strip(), v.strip()
                        if k == "seed":
                            try:
                                entry["seed"] = int(float(v))
                            except ValueError:
                                pass
                        elif k == "tier":
                            try:
                                entry["tier"] = int(float(v))
                            except ValueError:
                                pass
                        elif k == "z":
                            try:
                                entry["z"] = int(float(v))
                            except ValueError:
                                pass
                        elif k == "gravity":
                            try:
                                entry["gravity"] = float(v)
                            except ValueError:
                                pass
                        elif k == "grid_key":
                            entry["grid_key"] = v
            elif bl_stripped.startswith("Parameters:"):
                param_str = bl_stripped.split(": ", 1)[1]
                param_dict = {}
                for p in param_str.split(","):
                    p = p.strip()
                    if "=" in p:
                        k, v = p.split("=", 1)
                        param_dict[k.strip()] = v.strip()
                try:
                    if "z" in param_dict:
                        entry["z"] = int(float(param_dict["z"]))
                    if "gravity" in param_dict:
                        entry["gravity"] = float(param_dict["gravity"])
                    if "tier" in param_dict:
                        entry["tier"] = int(float(param_dict["tier"]))
                    if "grid" in param_dict:
                        entry["grid_key"] = param_dict["grid"]
                except (KeyError, ValueError):
                    pass
            elif bl_stripped.startswith("Grid:"):
                entry["grid"] = bl_stripped.split(":", 1)[1].strip()
            elif bl_stripped.startswith("InitialState:"):
                entry["initial_state"] = bl_stripped.split(":", 1)[1].strip()
            elif bl_stripped.startswith("Actions:"):
                entry["actions"] = bl_stripped.split(":", 1)[1].strip()
            elif bl_stripped.startswith("TransitionRules:"):
                entry["transition_rules"] = bl_stripped.split(":", 1)[1].strip()
            elif bl_stripped.startswith("Latency(ms):"):
                try:
                    entry["latency_ms"] = float(bl_stripped.split(":", 1)[1].strip())
                except ValueError:
                    entry["latency_ms"] = 0.0
            elif bl_stripped.startswith("InputTokens:"):
                try:
                    entry["input_tokens"] = int(float(bl_stripped.split(":", 1)[1].strip()))
                except ValueError:
                    entry["input_tokens"] = 0
            elif bl_stripped.startswith("OutputTokens:"):
                try:
                    entry["output_tokens"] = int(float(bl_stripped.split(":", 1)[1].strip()))
                except ValueError:
                    entry["output_tokens"] = 0
            elif bl_stripped.startswith("[MODEL OUTPUT]:"):
                section = "output"
                # Capture any content on the same line after the marker.
                rest = bl_stripped.split("[MODEL OUTPUT]:", 1)[1].strip()
                entry["raw_output"] = rest + "\n"
            elif bl_stripped.startswith("[GROUND TRUTH EXPECTED]:"):
                section = "truth"
                rest = bl_stripped.split("[GROUND TRUTH EXPECTED]:", 1)[1].strip()
                entry["ground_truth"] = rest + "\n"
            else:
                # Continuation lines belong to the active section.
                if section == "output":
                    entry["raw_output"] = entry.get("raw_output", "") + bl
                elif section == "truth":
                    entry["ground_truth"] = entry.get("ground_truth", "") + bl

        if not entry:
            continue

        z = entry.get("z")
        gravity = entry.get("gravity")
        tier = entry.get("tier")
        grid_key = entry.get("grid_key", "default")
        seed = entry.get("seed")
        combo = f"{seed}_{z}_{gravity}_{tier}_{grid_key}" if seed is not None else f"{z}_{gravity}_{tier}_{grid_key}"
        probe_idx = combo_counter.get(combo, 0)
        combo_counter[combo] = probe_idx + 1
        key = f"{combo}_{probe_idx}"

        results_matrix[key] = _score_probe(entry)
        probe_count += 1
        if gravity is not None:
            gravities_set.add(gravity)

    found_seeds = {v["seed"] for v in results_matrix.values() if v.get("seed") is not None}
    seeds_list = sorted(list(found_seeds)) if found_seeds else [42]

    parameters = {
        "n_probes": probe_count,
        "gravity_levels": sorted(list(gravities_set)),
        "seed": seeds_list[0],
        "seeds": seeds_list,
        "grid_size_levels": None,
    }

    data = {
        "model": model if model else "unknown",
        "timestamp": timestamp if timestamp else "unknown",
        "parameters": parameters,
        "results_matrix": results_matrix,
        "seeds": seeds_list,
        "completed_probes": probe_count,
        "fracture_cache": None,
    }

    return data


def run_audit_checks(data: Dict[str, Any], aggregated: Dict[str, Any], m: Dict[str, Any]) -> Dict[str, bool]:
    results_matrix = data.get("results_matrix", {})
    completed = data.get("completed_probes", len(results_matrix))
    total_probes = len(results_matrix)
    params = data.get("parameters", {})
    n_p = params.get("n_probes")
    seeds = data.get("seeds", [42])
    scheduled = (n_p * len(seeds)) if isinstance(n_p, int) else total_probes
    if scheduled <= 0:
        scheduled = total_probes

    checks = {}
    
    # A) coverage == completed / scheduled
    cov_ratio = (completed / scheduled) if scheduled > 0 else 1.0
    expected_cov = m.get("coverage_ratio", cov_ratio)
    checks["A) Coverage (completed/scheduled)"] = abs(expected_cov - cov_ratio) < 1e-4

    # B) taxonomy counts sum to valid count
    tax = m.get("failure_taxonomy", {})
    total_tax = sum(tax.values())
    valid_c = m.get("valid_count", 0)
    checks["B) Taxonomy counts sum to valid count"] = total_tax == valid_c
    checks["B1) None is SSOT-grounded"] = tax.get("None", 0) == sum(
        1
        for result in results_matrix.values()
        if ((result.get("probe_classification") or {}).get("valid", True))
        and not (
            result.get("is_infrastructure_failure", False)
            or result.get("provider_failure", False)
            or result.get("api_error") is not None
            or result.get("finish_reason") == "error"
        )
        and _is_ssot_fully_correct(result, result.get("probe_classification") or {})
    )

    # C) valid + invalid == completed
    invalid_c = m.get("invalid_count", 0)
    checks["C) Valid + Invalid == Completed"] = (valid_c + invalid_c) == completed

    # D) fracture depth consistency (authoritative derivation vs aggregated)
    final_fractures = []
    fracture_cache = data.get("fracture_cache")
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
    if not final_fractures:
        global_depth_acc = {}
        for key, result in results_matrix.items():
            z, gravity, tier, probe_idx, grid_key = _unpack_matrix_key(key)
            try:
                zi = int(float(z))
            except (TypeError, ValueError):
                zi = 0
            global_depth_acc.setdefault(zi, []).append(
                _coerce_float(result.get("step_accuracy", 0.0))
            )
        if global_depth_acc:
            curve = [(z, _mean(accs)) for z, accs in sorted(global_depth_acc.items())]
            from arcus.analysis.fracture import FracturePointFinder
            finder = FracturePointFinder(look_ahead_step=2)
            global_fracture = finder.compute_structural_yield(curve)
            if global_fracture is not None and global_fracture > 0:
                final_fractures.append(global_fracture)
    expected_fd = min(final_fractures) if final_fractures else None

    agg_fd = aggregated.get("fracture_depth")
    if agg_fd is not None and expected_fd is not None:
        checks["D) Fracture depth consistency"] = abs(float(agg_fd) - float(expected_fd)) < 1e-4
    else:
        checks["D) Fracture depth consistency"] = (agg_fd == expected_fd)

    # E) CRI and component bounds validity
    cri_dict = aggregated.get("cri", {})
    cri_val = cri_dict.get("cri", 0.0)
    tf = cri_dict.get("trajectory_fidelity", 0.0)
    hr = cri_dict.get("horizon_robustness", 0.0)
    sr = cri_dict.get("semantic_robustness", 0.0)
    checks["E) CRI and component bounds valid"] = (0.0 <= cri_val <= 1.0) and (0.0 <= tf <= 1.0) and (0.0 <= hr <= 1.0) and (0.0 <= sr <= 1.0)

    return checks


# ---------------------------------------------------------------------------
# Main function with file type detection
# ---------------------------------------------------------------------------
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Reproduce the ARCUS-X completed-run report from quickstart JSON files or TXT log files."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="A quickstart JSON file OR a directory containing quickstart_results_*.json files OR a TXT log file"
    )
    parser.add_argument(
        "--input",
        dest="input_flag",
        help="A quickstart JSON file OR a directory containing quickstart_results_*.json files OR a TXT log file"
    )
    parser.add_argument(
        "--pattern",
        default="quickstart_results_*.json",
        help="Glob pattern when --input is a directory (default: quickstart_results_*.json)"
    )
    parser.add_argument(
        "--lastrun",
        action="store_true",
        help="When analyzing a TXT log file, parse only the most recent run (delimited by the last 'ARCUS-X RUN START')"
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Run post-run consistency audit checks (A, B, C, D, E)"
    )
    parser.add_argument(
        "--density",
        action="store_true",
        help="Include the percentage of all probes at each horizon"
    )
    parser.add_argument(
        "--error-density",
        action="store_true",
        help="Include non-zero error rates and error codes at each horizon"
    )

    args = parser.parse_args()
    input_path = args.input_flag or args.input_path
    if not input_path:
        parser.error("the following arguments are required: INPUT or --input")
    pattern = args.pattern
    last_run_only = args.lastrun

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
                data = process_txt_file(filepath, last_run_only=last_run_only)
            else:
                raise ValueError(f"Unsupported file type: {filepath}")
            report = _render_report(
                data,
                show_density=args.density,
                show_error_density=args.error_density,
            )
            print(report)
            if args.audit:
                aggregated = _compute_aggregated_metrics(data.get("results_matrix", {}), data.get("seeds", [42]), data.get("fracture_cache"))
                m = _compute_report_metrics(
                    data.get("results_matrix", {}), data.get("model", "unknown"),
                    data.get("run_status", "COMPLETE"), data.get("completed_probes", len(data.get("results_matrix", {}))),
                    len(data.get("results_matrix", {})), aggregated["step_accuracy"]
                )
                audit_results = run_audit_checks(data, aggregated, m)
                print("\n" + "=" * 60)
                print("ARCUS-X POST-RUN AUDIT CONSISTENCY REPORT")
                print("=" * 60)
                for check_name, passed in audit_results.items():
                    status_str = "PASS" if passed else "FAIL"
                    print(f"  {check_name:<30} [{status_str}]")
                if m.get("unknown_taxonomy_audit"):
                    print("\nUnknown / Unmapped taxonomy diagnostics:")
                    for item in m["unknown_taxonomy_audit"][:10]:
                        probe_id = item.get("probe_id", "unknown")
                        raw = item.get("raw_taxonomy", "")
                        normalized = item.get("normalized_taxonomy", "")
                        step_acc = item.get("step_accuracy", 0.0)
                        pc = item.get("probe_classification", {})
                        print(
                            f"  probe={probe_id}: raw={raw!r}, normalized={normalized!r}, "
                            f"step_accuracy={step_acc}, mechanism={pc.get('mechanism', 'unknown')}, "
                            f"output_compliance={pc.get('output_compliance', 'unknown')}"
                        )
                print("=" * 60)
        except Exception as exc:
            print(f"Failed to analyze {filepath}: {exc}")
            continue


if __name__ == "__main__":
    main()
