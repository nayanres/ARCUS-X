#!/usr/bin/env python3
"""Read-only, reproducible reanalysis of ARCUS-X raw response logs.

Run from the repository root with:
    python audit/analyze_artifacts.py

The script reads outputs/absolute_raw_data_local_*.txt and saved
quickstart_results_*.json files. It does not write to the evaluation pipeline
or alter any source or raw results.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import post_run_analyzer as analyzer  # noqa: E402
from arcus.analysis.reporting import canonicalize_taxonomy  # noqa: E402


COORD_RE = re.compile(r"\[\s*-?\d+\s*,\s*-?\d+\s*\]")
META_RE = re.compile(
    r"(?<!\w)(seed|tier|gravity|z|grid_key|experiment_hash)=([^,]+)"
)
FALLBACK_EXPLANATION = "State synchronization lost after early divergence."
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 241106


def _literal(text: str, default: Any) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return default


def _parse_raw_context(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    run_id = ""
    in_model_output = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("Run ID:"):
            run_id = stripped.split(":", 1)[1].strip()
        if stripped.startswith("=== RAW STREAM ENTRY"):
            current = {"run_id": run_id, "_entry_lines": [line.rstrip("\r\n")]}
            in_model_output = False
            continue
        if current is None:
            continue
        if stripped.startswith("=============================="):
            current["_entry_lines"].append(line.rstrip("\r\n"))
            current["entry_hash"] = hashlib.sha256(
                "\n".join(current.pop("_entry_lines")).encode("utf-8")
            ).hexdigest()
            entries.append(current)
            current = None
            continue
        current["_entry_lines"].append(line.rstrip("\r\n"))
        if stripped.startswith("[MODEL OUTPUT]:"):
            in_model_output = True
            continue
        if in_model_output:
            continue
        if stripped.startswith("EnvironmentMetadata:"):
            for key, value in META_RE.findall(stripped.split(":", 1)[1]):
                current[key] = value.strip()
        elif stripped.startswith("Actions:"):
            current["actions"] = _literal(stripped.split(":", 1)[1].strip(), [])
        elif stripped.startswith("TransitionRules:"):
            current["transition_rules"] = _literal(
                stripped.split(":", 1)[1].strip(), {}
            )
        elif stripped.startswith("Grid:"):
            match = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", stripped.split(":", 1)[1])
            if match:
                current["grid_width"] = int(match.group(1))
                current["grid_height"] = int(match.group(2))
    return entries


def _independent_metrics(
    predicted: list[str], ground_truth: list[str], requested_horizon: int
) -> dict[str, Any]:
    gt_n = len(ground_truth)
    overlap = min(len(predicted), gt_n)
    mismatches = [
        i for i in range(overlap) if predicted[i].strip() != ground_truth[i].strip()
    ]
    transitions = max(gt_n - 1, 0)
    correct = sum(
        1 for i in range(1, overlap)
        if predicted[i].strip() == ground_truth[i].strip()
    )
    step_accuracy = correct / transitions if transitions else 0.0
    exact = len(predicted) == gt_n and not mismatches

    if mismatches:
        first_position = mismatches[0] + 1
        first_kind = "coordinate"
    elif len(predicted) > gt_n:
        first_position = gt_n + 1
        first_kind = "extra_coordinate"
    elif len(predicted) < gt_n:
        first_position = len(predicted) + 1
        first_kind = "missing_coordinate"
    else:
        first_position = None
        first_kind = ""

    if first_position is None:
        prefix_correct = transitions
    elif first_position <= 1:
        prefix_correct = 0
    else:
        prefix_correct = max(0, first_position - 2)
    continuity = prefix_correct / transitions if transitions else 0.0

    transition_flags = [
        (i < len(predicted) and predicted[i].strip() == ground_truth[i].strip())
        for i in range(1, gt_n)
    ]
    thirds: dict[str, float | None] = {
        "early_accuracy": None,
        "middle_accuracy": None,
        "late_accuracy": None,
    }
    if len(transition_flags) >= 3:
        third = len(transition_flags) // 3
        slices = (
            ("early_accuracy", transition_flags[:third]),
            ("middle_accuracy", transition_flags[third : 2 * third]),
            ("late_accuracy", transition_flags[2 * third :]),
        )
        thirds.update({
            name: sum(values) / len(values) if values else None
            for name, values in slices
        })

    generated = max(0, len(predicted) - 1)
    if requested_horizon <= 0 or generated <= 0:
        horizon_score = 0.0
    else:
        horizon_score = max(
            0.0,
            1.0 - abs(generated - requested_horizon) / max(generated, requested_horizon),
        )
    return {
        "step_accuracy": step_accuracy,
        "exact_match": exact,
        "continuity": continuity,
        "horizon_match": len(predicted) == gt_n,
        "horizon_compliance": horizon_score,
        "first_divergence_position": first_position,
        "first_divergence_kind": first_kind,
        "first_predicted_coordinate": (
            predicted[first_position - 1] if first_position and first_position <= len(predicted) else ""
        ),
        "first_ground_truth_coordinate": (
            ground_truth[first_position - 1]
            if first_position and first_position <= len(ground_truth)
            else ""
        ),
        "ground_truth_length_matches_z": gt_n == requested_horizon + 1,
        **thirds,
    }


def _transition_first_matches(
    predicted: list[str],
    ground_truth: list[str],
    rules: dict[str, dict[str, int]],
    actions: list[str],
    width: int,
    height: int,
) -> bool:
    """Whether a transition-first precedence would select a structured rule error."""
    if not isinstance(actions, (list, tuple)) or not isinstance(rules, dict):
        return False
    n = min(len(predicted), len(ground_truth))
    if predicted and ground_truth and predicted[0] != ground_truth[0]:
        return False
    mismatched: list[int] = []
    sign_inversions = axis_swaps = magnitude_errors = parity_errors = 0
    wraparound_errors = 0
    for i in range(1, n):
        action = actions[i - 1] if i - 1 < len(actions) else None
        rule = rules.get(action) if action else None
        if rule is None:
            continue
        rule_dx, rule_dy = int(rule["dx"]), int(rule["dy"])
        exp_dx = (
            (rule_dx + width // 2) % width - width // 2
            if width > 0 else rule_dx
        )
        exp_dy = (
            (rule_dy + height // 2) % height - height // 2
            if height > 0 else rule_dy
        )
        exp = (exp_dx, exp_dy)
        p0 = tuple(int(v) for v in COORD_RE.findall(predicted[i - 1])[0][1:-1].split(","))
        p1 = tuple(int(v) for v in COORD_RE.findall(predicted[i])[0][1:-1].split(","))
        act = (p1[0] - p0[0], p1[1] - p0[1])
        if width > 0:
            act = ((act[0] + width // 2) % width - width // 2, act[1])
        if height > 0:
            act = (act[0], (act[1] + height // 2) % height - height // 2)
        gt0 = tuple(int(v) for v in ground_truth[i - 1][1:-1].split(","))
        if p1 == (gt0[0] + rule_dx, gt0[1] + rule_dy) and (
            p1[0] < 0 or p1[0] >= width or p1[1] < 0 or p1[1] >= height
        ):
            wraparound_errors += 1
            mismatched.append(i)
            continue
        if act == exp:
            continue
        mismatched.append(i)
        gt0 = tuple(int(v) for v in ground_truth[i - 1][1:-1].split(","))
        gt1 = tuple(int(v) for v in ground_truth[i][1:-1].split(","))
        de = (gt1[0] - gt0[0], gt1[1] - gt0[1])
        da = (p1[0] - p0[0], p1[1] - p0[1])
        if da == (-de[0], -de[1]) and da != (0, 0):
            sign_inversions += 1
        elif da == (de[1], de[0]) and da != de and de[0] != de[1]:
            axis_swaps += 1
        else:
            if abs(act[0]) != abs(exp[0]) or abs(act[1]) != abs(exp[1]):
                magnitude_errors += 1
            if (act[0] % 2) != (exp[0] % 2) or (act[1] % 2) != (exp[1] % 2):
                parity_errors += 1

    if not mismatched:
        return False
    total = len(mismatched)
    if (
        wraparound_errors > 0
        and sign_inversions == axis_swaps == magnitude_errors == parity_errors == 0
    ):
        return False
    structured = sign_inversions + axis_swaps + magnitude_errors + parity_errors
    return structured >= total / 2


def _canonical_mechanism(mechanism: str) -> str:
    return canonicalize_taxonomy(mechanism, False)


def _row_values(row: dict[str, Any], tax: Any, independent: dict[str, Any]) -> dict[str, Any]:
    pc = tax.probe_classification.to_dict() if tax.probe_classification else {}
    valid = bool(pc.get("valid", tax.valid))
    exact = bool(independent["exact_match"])
    reported_none = valid and bool(
        exact or pc.get("exact_match", False) or pc.get("trajectory_correct", False)
    )
    mechanism = tax.mechanism.value
    legacy = tax.legacy_mode.value
    category = "None" if reported_none else canonicalize_taxonomy(legacy, False)
    if category == "None" and not reported_none:
        category = "Unknown / Unmapped"
    strict_category = "None" if exact else _canonical_mechanism(mechanism)
    if strict_category == "None" and not exact:
        strict_category = "Unknown / Unmapped"
    fallback = (
        valid
        and mechanism == "State Tracking Failure"
        and tax.explanation == FALLBACK_EXPLANATION
    )
    if fallback:
        fallback_unknown = "Unknown / Unmapped"
    else:
        fallback_unknown = category
    strict_fallback_unknown = (
        "Unknown / Unmapped" if fallback and not exact else strict_category
    )

    return {
        "valid": valid,
        "trajectory_correct": bool(pc.get("trajectory_correct", False)),
        "classifier_exact_match": bool(pc.get("exact_match", False)),
        "exception_code": pc.get("exception_code") or "",
        "output_compliance": pc.get("output_compliance") or "",
        "formatting_subcode": pc.get("formatting_subcode") or "",
        "mechanism": mechanism,
        "legacy_mode": legacy,
        "subtype": tax.subtype.value,
        "explanation": tax.explanation or "",
        "state_tracking_fallback": fallback,
        "reported_taxonomy": category,
        "fallback_unknown_taxonomy": fallback_unknown,
        "strict_horizon_taxonomy": strict_category,
        "combined_counterfactual_taxonomy": strict_fallback_unknown,
        "transition_first": _transition_first_matches(
            row.get("predicted_path", []),
            row.get("ground_truth_path", []),
            row.get("_rules") if isinstance(row.get("_rules"), dict) else {},
            row.get("_actions") if isinstance(row.get("_actions"), (list, tuple)) else [],
            int(row.get("grid_width", 0) or 0),
            int(row.get("grid_height", 0) or 0),
        ),
        "probe_classification": pc,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _cluster_bootstrap(
    rows: list[dict[str, Any]],
    value_key: str,
    valid_only: bool = False,
) -> tuple[float | None, float | None, float | None, int]:
    """Bootstrap equal-weight seed clusters after averaging repeated gravity calls by task."""
    task_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    task_seed: dict[tuple[Any, ...], Any] = {}
    for row in rows:
        if valid_only and not row["valid"]:
            continue
        key = (row["seed"], row["tier"], row["z"], row["task_hash"])
        task_values[key].append(float(row.get(value_key, 0.0)))
        task_seed[key] = row["seed"]
    seed_values: dict[Any, list[float]] = defaultdict(list)
    for key, values in task_values.items():
        seed_values[task_seed[key]].append(sum(values) / len(values))
    clusters = [sum(v) / len(v) for v in seed_values.values() if v]
    if not clusters:
        return None, None, None, 0
    point = sum(clusters) / len(clusters)
    rng = random.Random(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        draws.append(sum(sample) / len(sample))
    draws.sort()
    lower = draws[int(0.025 * (len(draws) - 1))]
    upper = draws[int(0.975 * (len(draws) - 1))]
    return point, lower, upper, len(clusters)


def _bootstrap_share(rows: list[dict[str, Any]], category: str, variant: str) -> tuple[float | None, float | None, float | None, int]:
    valid_rows = [r for r in rows if r["valid"]]
    task_categories: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in valid_rows:
        key = (row["seed"], row["tier"], row["z"], row["task_hash"])
        task_categories[key].append(float(row[variant] == category))
    by_seed: dict[Any, list[float]] = defaultdict(list)
    for key, values in task_categories.items():
        by_seed[key[0]].append(sum(values) / len(values))
    seed_shares = [
        sum(values) / len(values) for values in by_seed.values() if values
    ]
    if not seed_shares:
        return None, None, None, 0
    point = sum(seed_shares) / len(seed_shares)
    rng = random.Random(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = [seed_shares[rng.randrange(len(seed_shares))] for _ in seed_shares]
        draws.append(sum(sample) / len(sample))
    draws.sort()
    return (
        point,
        draws[int(0.025 * (len(draws) - 1))],
        draws[int(0.975 * (len(draws) - 1))],
        len(seed_shares),
    )


def _deduplicate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[tuple[Any, ...]] = set()
    unique = []
    duplicates = 0
    for row in rows:
        key = (
            row["model"],
            row["seed"],
            row["z"],
            row["gravity"],
            row["tier"],
            row["grid_key"],
            row["task_hash"],
            row.get("entry_hash", row.get("source_entry_hash", "")),
        )
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(row)
    return unique, duplicates


def _classify_file(path: Path) -> tuple[str, list[dict[str, Any]], int]:
    context = _parse_raw_context(path)
    captured: list[Any] = []
    original = analyzer.classify_from_result

    def capture(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    analyzer.classify_from_result = capture
    try:
        data = analyzer.process_txt_file(str(path))
    finally:
        analyzer.classify_from_result = original

    matrix_rows = list(data["results_matrix"].items())
    if not (len(context) == len(matrix_rows) == len(captured)):
        raise RuntimeError(
            f"Raw parser/classifier alignment failed for {path.name}: "
            f"context={len(context)}, matrix={len(matrix_rows)}, classified={len(captured)}"
        )

    rows = []
    for index, ((matrix_key, parsed), metadata, tax) in enumerate(
        zip(matrix_rows, context, captured)
    ):
        pred = parsed.get("predicted_path", [])
        gt = parsed.get("ground_truth_path", [])
        z = int(metadata.get("z", parsed.get("z", 0)) or 0)
        tier = int(metadata.get("tier", parsed.get("tier", 0)) or 0)
        seed = int(metadata["seed"]) if metadata.get("seed") else parsed.get("seed")
        gravity = float(metadata.get("gravity", parsed.get("gravity", 0.0)) or 0.0)
        grid_key = metadata.get("grid_key", parsed.get("grid_key", "default"))
        hashes = metadata.get("experiment_hash", "")
        independent = _independent_metrics(pred, gt, z)
        parsed.update(
            _rules=metadata.get("transition_rules", {}),
            _actions=metadata.get("actions", []),
        )
        tax_data = _row_values(parsed, tax, independent)
        rows.append({
            "model": data["model"],
            "source_file": path.name,
            "source_row": index + 1,
            "run_id": metadata.get("run_id", ""),
            "matrix_key": matrix_key,
            "seed": seed,
            "z": z,
            "tier": tier,
            "gravity": gravity,
            "grid_key": grid_key,
            "task_hash": hashes,
            "entry_hash": metadata.get("entry_hash", ""),
            "predicted_path": pred,
            "ground_truth_path": gt,
            "predicted_length": len(pred),
            "ground_truth_length": len(gt),
            "step_accuracy": independent["step_accuracy"],
            "exact_match": independent["exact_match"],
            "continuity": independent["continuity"],
            "horizon_match": independent["horizon_match"],
            "horizon_compliance": independent["horizon_compliance"],
            "first_divergence_position": independent["first_divergence_position"],
            "first_divergence_kind": independent["first_divergence_kind"],
            "first_predicted_coordinate": independent["first_predicted_coordinate"],
            "first_ground_truth_coordinate": independent["first_ground_truth_coordinate"],
            "ground_truth_length_matches_z": independent["ground_truth_length_matches_z"],
            "early_accuracy": independent["early_accuracy"],
            "middle_accuracy": independent["middle_accuracy"],
            "late_accuracy": independent["late_accuracy"],
            "grid_width": parsed.get("grid_width"),
            "grid_height": parsed.get("grid_height"),
            "actions": json.dumps(parsed.get("_actions", [])),
            "transition_rules": json.dumps(parsed.get("_rules", {}), sort_keys=True),
            "predicted_path": json.dumps(pred),
            "ground_truth_path": json.dumps(gt),
            **tax_data,
        })

    return str(data.get("model", path.stem)), rows, len(context)


def _write_probe_csv(rows: list[dict[str, Any]]) -> None:
    columns = [
        "model", "source_file", "source_row", "run_id", "matrix_key", "seed",
        "z", "tier", "gravity", "grid_key", "task_hash", "predicted_length",
        "entry_hash",
        "ground_truth_length", "ground_truth_length_matches_z", "valid",
        "exception_code", "output_compliance", "formatting_subcode",
        "step_accuracy", "exact_match", "continuity", "horizon_match",
        "horizon_compliance", "first_divergence_position", "first_divergence_kind",
        "first_predicted_coordinate", "first_ground_truth_coordinate",
        "early_accuracy", "middle_accuracy", "late_accuracy", "mechanism",
        "legacy_mode", "subtype", "explanation", "state_tracking_fallback",
        "reported_taxonomy", "fallback_unknown_taxonomy", "strict_horizon_taxonomy",
        "combined_counterfactual_taxonomy", "transition_first",
        "trajectory_correct", "classifier_exact_match", "grid_width",
        "grid_height", "actions", "transition_rules", "predicted_path",
        "ground_truth_path",
    ]
    _write_csv(OUTPUT / "probe_level_audit.csv", rows, columns)


def _write_taxonomy_outputs(rows: list[dict[str, Any]]) -> None:
    categories = [
        "None", "State Tracking Failure", "Transition Rule Failure",
        "Semantic Interpretation Failure", "Horizon Collapse",
        "Formatting Failure", "Unknown / Unmapped",
    ]
    variants = [
        ("A_original_reporting", "reported_taxonomy"),
        ("B_fallback_unknown", "fallback_unknown_taxonomy"),
        ("C_exact_horizon_invariant", "strict_horizon_taxonomy"),
        ("D_combined", "combined_counterfactual_taxonomy"),
    ]
    sensitivity: list[dict[str, Any]] = []
    keys = sorted({(r["model"], r["tier"], r["z"]) for r in rows})
    for model, tier, z in keys:
        subset = [r for r in rows if (r["model"], r["tier"], r["z"]) == (model, tier, z)]
        subset, _ = _deduplicate(subset)
        valid = [r for r in subset if r["valid"]]
        transition_overrides = sum(
            r["valid"] and r["reported_taxonomy"] == "State Tracking Failure"
            and r["transition_first"] for r in subset
        )
        for name, field in variants:
            counts = Counter(r[field] for r in valid)
            for category in categories:
                count = counts.get(category, 0)
                sensitivity.append({
                    "model": model, "tier": tier, "z": z, "variant": name,
                    "category": category, "count_valid": count,
                    "valid_denominator": len(valid),
                    "proportion_valid": count / len(valid) if valid else None,
                    "fallback_state_tracking_count": sum(r["state_tracking_fallback"] for r in valid),
                    "transition_first_candidate_count": transition_overrides,
                })
        # Variant E retains the intentional reporting override but reorders
        # structured transition diagnostics ahead of state tracking.
        e_counts = Counter()
        for r in valid:
            label = r["reported_taxonomy"]
            if label == "State Tracking Failure" and r["transition_first"]:
                label = "Transition Rule Failure"
            e_counts[label] += 1
        for category in categories:
            count = e_counts.get(category, 0)
            sensitivity.append({
                "model": model, "tier": tier, "z": z,
                "variant": "E_transition_first_precedence",
                "category": category, "count_valid": count,
                "valid_denominator": len(valid),
                "proportion_valid": count / len(valid) if valid else None,
                "fallback_state_tracking_count": sum(r["state_tracking_fallback"] for r in valid),
                "transition_first_candidate_count": transition_overrides,
            })
    _write_csv(OUTPUT / "taxonomy_sensitivity.csv", sensitivity)

    model_summary: list[dict[str, Any]] = []
    for model in sorted({r["model"] for r in rows}):
        subset = [r for r in rows if r["model"] == model]
        unique, duplicate_count = _deduplicate(subset)
        valid = [r for r in unique if r["valid"]]
        none_count = sum(r["reported_taxonomy"] == "None" for r in valid)
        exact_count = sum(r["exact_match"] for r in valid)
        wrong_none = [
            r for r in valid
            if r["reported_taxonomy"] == "None" and not r["exact_match"]
        ]
        step_perfect_none = [
            r for r in wrong_none if r["step_accuracy"] >= 1.0
        ]
        classifier_none_subperfect = [
            r for r in wrong_none
            if r["mechanism"] == "None" and r["step_accuracy"] < 1.0
        ]
        overlength_prefix = [
            r for r in wrong_none
            if r["predicted_length"] > r["ground_truth_length"]
            and r["step_accuracy"] == 1.0
            and r["first_divergence_kind"] == "extra_coordinate"
        ]
        initial_state_mismatch = [
            r for r in wrong_none
            if r["first_divergence_kind"] == "coordinate"
            and r["first_divergence_position"] == 1
            and r["step_accuracy"] == 1.0
        ]
        state = [r for r in valid if r["mechanism"] == "State Tracking Failure"]
        fallback = [r for r in state if r["state_tracking_fallback"]]
        model_summary.append({
            "model": model,
            "raw_entries": len(subset),
            "exact_duplicate_raw_entry_count_removed": duplicate_count,
            "unique_logged_observations": len(unique),
            "unique_task_condition_count": len({
                (
                    r["model"], r["seed"], r["z"], r["gravity"], r["tier"],
                    r["grid_key"], r["task_hash"],
                )
                for r in unique
            }),
            "repeated_task_condition_observations": len(unique) - len({
                (
                    r["model"], r["seed"], r["z"], r["gravity"], r["tier"],
                    r["grid_key"], r["task_hash"],
                )
                for r in unique
            }),
            "valid_probes": len(valid),
            "invalid_probes": len(unique) - len(valid),
            "exact_match_count_valid": exact_count,
            "reported_none_count_valid": none_count,
            "reported_none_proportion_valid": none_count / len(valid) if valid else None,
            "exact_match_proportion_valid": exact_count / len(valid) if valid else None,
            "none_not_exact_count": len(wrong_none),
            "none_not_exact_step_perfect_override_count": len(step_perfect_none),
            "none_not_exact_classifier_none_subperfect_count": len(classifier_none_subperfect),
            "none_not_exact_overlength_prefix_count": len(overlength_prefix),
            "none_not_exact_initial_state_mismatch_count": len(initial_state_mismatch),
            "state_tracking_count_valid": len(state),
            "state_tracking_explicit_count": len(state) - len(fallback),
            "state_tracking_fallback_count": len(fallback),
            "fallback_fraction_state_tracking": len(fallback) / len(state) if state else 0.0,
            "fallback_fraction_all_valid": len(fallback) / len(valid) if valid else 0.0,
            "strict_exact_none_count": sum(r["strict_horizon_taxonomy"] == "None" for r in valid),
            "strict_horizon_failure_count": sum(r["strict_horizon_taxonomy"] == "Horizon Collapse" for r in valid),
        })
    _write_csv(OUTPUT / "taxonomy_model_summary.csv", model_summary)


def _write_horizon_and_matching(rows: list[dict[str, Any]]) -> None:
    deduplicated_rows, _ = _deduplicate(rows)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["tier"], row["gravity"], row["z"])].append(row)
    model_tier_gravity_totals: Counter = Counter()
    for (model, tier, gravity, _z), values in grouped.items():
        model_tier_gravity_totals[(model, tier, gravity)] += len(values)
    dedup_grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in deduplicated_rows:
        dedup_grouped[(row["model"], row["tier"], row["gravity"], row["z"])].append(row)
    dedup_totals: Counter = Counter()
    for (model, tier, gravity, _z), values in dedup_grouped.items():
        dedup_totals[(model, tier, gravity)] += len(values)
    dist = []
    for key, values in sorted(grouped.items()):
        model, tier, gravity, z = key
        dedup_values = dedup_grouped[key]
        unique_hashes = {r["task_hash"] for r in values if r["task_hash"]}
        dist.append({
            "model": model, "tier": tier, "gravity": gravity, "z": z,
            "raw_probe_count": len(values),
            "deduplicated_probe_count": len(dedup_values),
            "unique_task_hash_count": len(unique_hashes),
            "share_within_model_tier_gravity": (
                len(values) / model_tier_gravity_totals[(model, tier, gravity)]
            ),
            "deduplicated_share_within_model_tier_gravity": (
                len(dedup_values) / dedup_totals[(model, tier, gravity)]
            ),
            "valid_count": sum(r["valid"] for r in values),
            "invalid_count": sum(not r["valid"] for r in values),
        })
    _write_csv(OUTPUT / "horizon_distribution.csv", dist)

    # Exact task-hash support by model/tier/horizon. Hashes are generated from
    # seed, tier, grid, initial state, horizon, actions, rules, and label map.
    support: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    support_seeds: dict[tuple[str, int, int], dict[str, Any]] = defaultdict(dict)
    for row in rows:
        if row["task_hash"]:
            support[(row["model"], row["tier"], row["z"])].add(row["task_hash"])
            support_seeds[(row["model"], row["tier"], row["z"])][row["task_hash"]] = row["seed"]
    task_rows = []
    models = sorted({r["model"] for r in rows})
    for tier in sorted({r["tier"] for r in rows}):
        for z in sorted({r["z"] for r in rows}):
            sets = [support.get((model, tier, z), set()) for model in models]
            common = set.intersection(*sets) if sets and all(sets) else set()
            task_rows.append({
                "tier": tier, "z": z, "models_in_repository": len(models),
                "models_with_observations": sum(bool(s) for s in sets),
                "all_model_common_task_hashes": len(common),
                "all_model_common_seed_count": (
                    len({
                        support_seeds[(model, tier, z)][task_hash]
                        for task_hash in common
                        for model in models
                        if task_hash in support_seeds[(model, tier, z)]
                    })
                ),
                "status": "exact task match across every listed model" if common else "no all-model common set",
            })
    _write_csv(OUTPUT / "task_support_by_model_tier_horizon.csv", task_rows)

    tier_support: list[dict[str, Any]] = []
    model_hashes: dict[tuple[str, int, int, Any], set[str]] = defaultdict(set)
    for row in rows:
        if row["task_hash"]:
            model_hashes[(row["model"], row["tier"], row["z"], row["seed"])].add(
                row["task_hash"]
            )
    for model in models:
        for z in sorted({r["z"] for r in rows}):
            for seed in sorted({r["seed"] for r in rows if r["model"] == model}, key=str):
                by_tier = [model_hashes.get((model, t, z, seed), set()) for t in range(4)]
                available = [hashes for hashes in by_tier if hashes]
                common = set.intersection(*available) if available else set()
                tier_support.append({
                    "model": model, "seed": seed, "z": z,
                    "tiers_with_observations": len(available),
                    "task_hashes_common_to_all_available_tiers": len(common),
                    "all_four_tier_tasks_matched": len(available) == 4 and bool(common),
                })
    _write_csv(OUTPUT / "tier_task_matching.csv", tier_support)


def _write_matched_horizon_metrics(rows: list[dict[str, Any]]) -> None:
    metrics: list[dict[str, Any]] = []
    for model in sorted({r["model"] for r in rows}):
        for tier in range(4):
            for z in (20, 30):
                subset = [
                    r for r in rows
                    if r["model"] == model and r["tier"] == tier and r["z"] == z
                ]
                if not subset:
                    continue
                unique, _ = _deduplicate(subset)
                fields: dict[str, tuple[Callable[[dict[str, Any]], float], bool]] = {
                    "step_accuracy_unconditional": (lambda r: r["step_accuracy"] if r["valid"] else 0.0, False),
                    "step_accuracy_valid": (lambda r: r["step_accuracy"], True),
                    "exact_match_unconditional": (lambda r: float(r["exact_match"] and r["valid"]), False),
                    "exact_match_valid": (lambda r: float(r["exact_match"]), True),
                    "continuity_unconditional": (lambda r: r["continuity"] if r["valid"] else 0.0, False),
                    "continuity_valid": (lambda r: r["continuity"], True),
                    "horizon_match_unconditional": (lambda r: float(r["horizon_match"] and r["valid"]), False),
                    "horizon_compliance_valid": (lambda r: r["horizon_compliance"], True),
                    "invalid_response_rate": (lambda r: float(not r["valid"]), False),
                    "early_accuracy_valid": (lambda r: float(r["early_accuracy"] or 0.0), True),
                    "middle_accuracy_valid": (lambda r: float(r["middle_accuracy"] or 0.0), True),
                    "late_accuracy_valid": (lambda r: float(r["late_accuracy"] or 0.0), True),
                }
                for metric, (fn, valid_only) in fields.items():
                    for row in unique:
                        row["_metric_value"] = fn(row)
                    estimate, lower, upper, n_seeds = _cluster_bootstrap(
                        unique, "_metric_value", valid_only=valid_only
                    )
                    denominator = (
                        sum(r["valid"] for r in unique) if valid_only else len(unique)
                    )
                    metrics.append({
                        "model": model, "tier": tier, "z": z, "metric": metric,
                        "estimate": estimate, "ci95_lower_cluster_seed": lower,
                        "ci95_upper_cluster_seed": upper, "n_probe_conditions": len(unique),
                        "n_valid_probe_conditions": sum(r["valid"] for r in unique),
                        "n_unique_task_hashes": len({r["task_hash"] for r in unique}),
                        "denominator_n": denominator, "n_seed_clusters": n_seeds,
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    })
                for category in [
                    "None", "State Tracking Failure", "Transition Rule Failure",
                    "Semantic Interpretation Failure", "Horizon Collapse",
                    "Formatting Failure", "Unknown / Unmapped",
                ]:
                    estimate, lower, upper, n_seeds = _bootstrap_share(
                        unique, category, "reported_taxonomy"
                    )
                    metrics.append({
                        "model": model, "tier": tier, "z": z,
                        "metric": f"taxonomy_{category.replace(' ', '_')}_share_valid",
                        "estimate": estimate, "ci95_lower_cluster_seed": lower,
                        "ci95_upper_cluster_seed": upper,
                        "n_probe_conditions": len(unique),
                        "n_valid_probe_conditions": sum(r["valid"] for r in unique),
                        "n_unique_task_hashes": len({r["task_hash"] for r in unique}),
                        "denominator_n": sum(r["valid"] for r in unique),
                        "n_seed_clusters": n_seeds,
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    })
    _write_csv(OUTPUT / "matched_horizon_metrics.csv", metrics)


def _fracture_depth(curve: list[tuple[int, float]], threshold: float, look_ahead: int) -> int | None:
    for index, (z, accuracy) in enumerate(curve):
        if accuracy < threshold:
            window = curve[index + 1 : index + 1 + look_ahead]
            if not any(score >= threshold for _, score in window):
                return z
    return None


def _write_fracture_sensitivity(rows: list[dict[str, Any]]) -> None:
    items = []
    for model in sorted({r["model"] for r in rows}):
        for tier in range(4):
            subset, _ = _deduplicate(
                [r for r in rows if r["model"] == model and r["tier"] == tier]
            )
            for condition, source in (
                ("all_probes_invalid_as_zero", subset),
                ("valid_only", [r for r in subset if r["valid"]]),
            ):
                per_z: dict[int, list[float]] = defaultdict(list)
                for row in source:
                    value = row["step_accuracy"] if row["valid"] else 0.0
                    per_z[row["z"]].append(value)
                curve = [
                    (z, sum(values) / len(values))
                    for z, values in sorted(per_z.items())
                ]
                for look_ahead in (2, 3):
                    items.append({
                        "model": model, "tier": tier, "condition": condition,
                        "threshold": 0.5, "look_ahead_sampled_points": look_ahead,
                        "fracture_depth": _fracture_depth(curve, 0.5, look_ahead),
                        "sampled_horizons": len(curve),
                        "curve": json.dumps(curve),
                        "probes": len(source),
                        "valid_probes": sum(r["valid"] for r in subset),
                    })

    json_by_model: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted((ROOT / "outputs").glob("quickstart_results_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        model = str(data.get("model", ""))
        json_by_model[_normal_model(model)] = (path, data)
    for item in items:
        snapshot_record = json_by_model.get(_normal_model(item["model"]))
        item["quickstart_snapshot_file"] = ""
        item["quickstart_matrix_n"] = ""
        item["afs_fracture_cache_json"] = ""
        item["afs_fracture_cache_provenance"] = "not in raw text log; read from snapshot only"
        if snapshot_record:
            snapshot_path, snapshot = snapshot_record
            item["quickstart_snapshot_file"] = snapshot_path.name
            item["quickstart_matrix_n"] = len(snapshot.get("results_matrix", {}))
            item["afs_fracture_cache_json"] = json.dumps(snapshot.get("fracture_cache", {}))
    _write_csv(OUTPUT / "fracture_sensitivity.csv", items)


def _normal_model(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _write_artifact_inventory(rows: list[dict[str, Any]]) -> None:
    quickstart = {}
    for path in sorted((ROOT / "outputs").glob("quickstart_results_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        quickstart[_normal_model(str(data.get("model", "")))] = (path, data)
    inventory = []
    for model in sorted({r["model"] for r in rows}):
        subset = [r for r in rows if r["model"] == model]
        unique, duplicates = _deduplicate(subset)
        saved = quickstart.get(_normal_model(model))
        inventory.append({
            "model": model,
            "raw_log_files": ";".join(sorted({r["source_file"] for r in subset})),
            "raw_entry_count": len(subset),
            "unique_logged_observation_count": len(unique),
            "exact_duplicate_raw_entry_count_removed": duplicates,
            "unique_task_condition_count": len({
                (
                    r["model"], r["seed"], r["z"], r["gravity"], r["tier"],
                    r["grid_key"], r["task_hash"],
                )
                for r in unique
            }),
            "repeated_task_condition_observation_count": len(unique) - len({
                (
                    r["model"], r["seed"], r["z"], r["gravity"], r["tier"],
                    r["grid_key"], r["task_hash"],
                )
                for r in unique
            }),
            "raw_distinct_horizon_count": len({r["z"] for r in unique}),
            "raw_distinct_tier_count": len({r["tier"] for r in unique}),
            "raw_distinct_seed_count": len({r["seed"] for r in unique}),
            "quickstart_file": saved[0].name if saved else "",
            "quickstart_matrix_count": len(saved[1].get("results_matrix", {})) if saved else "",
            "quickstart_matches_raw_count": (
                len(saved[1].get("results_matrix", {})) == len(unique) if saved else ""
            ),
        })
    _write_csv(OUTPUT / "artifact_inventory.csv", inventory)


def _write_synthetic_tests() -> None:
    from arcus.tasks.generator import GridTaskGenerator
    from arcus.analysis.taxonomy import classify_trajectory
    from arcus.environment.grid import parse_coord

    task = GridTaskGenerator(seed=29).generate(tier=0, task_index=7, horizon=12)
    gt = list(task.ground_truth_trajectory)
    cases: dict[str, list[str]] = {"known_correct": gt}
    cases["length_mismatch_overlong_correct_prefix"] = gt + [gt[-1]]
    cases["truncated_correct_prefix"] = gt[:-1]
    sign_error = [gt[0]]
    for i in range(1, len(gt)):
        x0, y0 = parse_coord(task.ground_truth_trajectory[i - 1])
        x1, y1 = parse_coord(task.ground_truth_trajectory[i])
        dx, dy = x1 - x0, y1 - y0
        x = (x0 - dx) % task.grid_width
        y = (y0 - dy) % task.grid_height
        sign_error.append(f"[{x},{y}]")
    cases["sign_inversion"] = sign_error
    delayed_offset = list(gt)
    for i in range(5, len(delayed_offset)):
        x, y = parse_coord(task.ground_truth_trajectory[i])
        delayed_offset[i] = f"[{(x + 1) % task.grid_width},{y}]"
    cases["late_persistent_offset"] = delayed_offset
    cases["unparseable"] = ["no coordinate output"]

    results = []
    for name, predicted in cases.items():
        outcome = classify_trajectory(
            predicted, gt, task.transition_rules, task.actions, task.tier,
            task.grid_width, task.grid_height, "\n".join(predicted),
        )
        results.append({
            "synthetic_case": name,
            "valid": outcome.valid,
            "outcome": outcome.outcome.value,
            "mechanism": outcome.mechanism.value,
            "subtype": outcome.subtype.value,
            "explanation": outcome.explanation,
            "trajectory_correct": (
                outcome.probe_classification.trajectory_correct
                if outcome.probe_classification else False
            ),
            "exact_match": (
                outcome.probe_classification.exact_match
                if outcome.probe_classification else False
            ),
            "predicted_length": len(predicted),
            "ground_truth_length": len(gt),
            "validation_scope": "synthetic rule-pattern recovery only; not human validity",
        })
    _write_csv(OUTPUT / "synthetic_classifier_checks.csv", results)


def main() -> None:
    raw_files = sorted(
        p for p in (ROOT / "outputs").glob("absolute_raw_data_local_*.txt")
        if ".analyzed." not in p.name
    )
    if not raw_files:
        raise SystemExit("No outputs/absolute_raw_data_local_*.txt files found")

    all_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for path in raw_files:
        model, parsed_rows, raw_count = _classify_file(path)
        all_rows.extend(parsed_rows)
        model_rows.append({
            "model": model,
            "source_file": path.name,
            "parsed_entries": len(parsed_rows),
            "raw_context_entries": raw_count,
        })

    _write_probe_csv(all_rows)
    _write_taxonomy_outputs(all_rows)
    _write_horizon_and_matching(all_rows)
    _write_matched_horizon_metrics(all_rows)
    _write_fracture_sensitivity(all_rows)
    _write_artifact_inventory(all_rows)
    _write_synthetic_tests()
    _write_csv(OUTPUT / "raw_parser_alignment.csv", model_rows)
    print(
        f"Audited {len(raw_files)} raw logs, {len(all_rows)} logged responses; "
        f"wrote analysis CSVs under {OUTPUT}"
    )


if __name__ == "__main__":
    main()
