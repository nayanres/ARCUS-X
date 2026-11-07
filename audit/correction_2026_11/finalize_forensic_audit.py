"""Reproduce the final, calibration-excluded ARCUS-X audit tables.

This script reads preserved raw response logs and existing probe-level
reanalysis files. It writes only under audit/correction_2026_11.
"""

from __future__ import annotations

import csv
import hashlib
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import post_run_analyzer as post_run  # noqa: E402


PUBLICATION_MODELS = [
    "qwen.qwen3-235b-a22b-2507",
    "moonshotai.kimi-k2.5",
    "qwen.qwen3-coder-480b-a35b-v1:0",
    "mistral-small-2603",
    "mistral-large-2512",
    "mistral.devstral-2-123b",
    "ministral-14b-2512",
    "ministral-8b-2512",
    "ministral-3b-2512",
    "nvidia.nemotron-super-3-120b",
    "zai.glm-5",
    "GPT-5-mini",
    "MAI-Thinking-1",
]

CATEGORIES = [
    "None",
    "State Tracking Failure",
    "Transition Rule Failure",
    "Semantic Interpretation Failure",
    "Horizon Collapse",
    "Formatting Failure",
    "Unknown / Unmapped",
    "Unclassified",
]

METRICS = (
    "step_accuracy",
    "exact_match",
    "continuity",
    "horizon_compliance",
    "early_accuracy",
    "middle_accuracy",
    "late_accuracy",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def integer(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def is_calibration_anchor(row: dict[str, Any]) -> bool:
    return (
        integer(row.get("z")) == 3
        and integer(row.get("tier")) == 1
        and abs(number(row.get("gravity")) - 0.0) < 1e-12
    )


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def metric_value(row: dict[str, Any], metric: str) -> float:
    if metric == "exact_match":
        return float(truth(row.get("exact_match")))
    return number(row.get(metric))


def seed_cluster_ci(
    rows: list[dict[str, Any]],
    metric: str,
    valid_only: bool,
    *,
    replicates: int = 2_000,
) -> tuple[float | None, float | None, int]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if valid_only and not truth(row.get("valid")):
            continue
        clusters[str(row.get("seed", ""))].append(row)
    clusters = {seed: group for seed, group in clusters.items() if seed}
    if not clusters:
        return None, None, 0

    seed_ids = sorted(clusters)
    cluster_totals = {}
    for seed, group in clusters.items():
        values = [metric_value(row, metric) for row in group]
        cluster_totals[seed] = (sum(values), len(values))
    rng = random.Random(241106)
    draws = []
    for _ in range(replicates):
        selected = [rng.choice(seed_ids) for _ in seed_ids]
        total = sum(cluster_totals[seed][0] for seed in selected)
        count = sum(cluster_totals[seed][1] for seed in selected)
        if count:
            draws.append(total / count)
    draws.sort()
    lower = draws[int(0.025 * (len(draws) - 1))]
    upper = draws[int(0.975 * (len(draws) - 1))]
    return lower, upper, len(seed_ids)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if truth(row.get("valid"))]
    unique_tasks = {
        row.get("task_hash", "")
        for row in rows
        if row.get("task_hash")
    }
    summary: dict[str, Any] = {
        "observations": len(rows),
        "valid": len(valid),
        "invalid": len(rows) - len(valid),
        "validity": len(valid) / len(rows) if rows else None,
        "unique_task_hashes": len(unique_tasks),
        "distinct_horizons": len({integer(row.get("z")) for row in rows}),
        "valid_seed_clusters": len({
            str(row.get("seed")) for row in valid if row.get("seed")
        }),
        "none_count": sum(row.get("reported_taxonomy") == "None" for row in valid),
        "exact_match_valid_count": sum(truth(row.get("exact_match")) for row in valid),
        "state_tracking_fallback_count": sum(
            truth(row.get("state_tracking_fallback")) for row in valid
        ),
    }
    for metric in METRICS:
        summary[f"{metric}_all"] = mean([metric_value(row, metric) for row in rows])
        summary[f"{metric}_valid"] = mean(
            [metric_value(row, metric) for row in valid]
        )
    for category in CATEGORIES:
        count = sum(row.get("reported_taxonomy") == category for row in valid)
        key = re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")
        summary[f"{key}_count"] = count
        summary[f"{key}_proportion_valid"] = count / len(valid) if valid else None
    return summary


def output_model_summaries(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_summary: list[dict[str, Any]] = []
    anchor_sensitivity: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        anchors = [row for row in model_rows if is_calibration_anchor(row)]
        primary = [row for row in model_rows if not is_calibration_anchor(row)]
        full = summarize_rows(model_rows)
        included = summarize_rows(primary)
        all_summary.append({
            "model": model,
            "publication_cohort": model in PUBLICATION_MODELS,
            **included,
            "calibration_anchor_observations_excluded": len(anchors),
            "logged_observations_including_anchor": len(model_rows),
        })
        for metric in (
            "observations",
            "valid",
            "invalid",
            "validity",
            "step_accuracy_all",
            "step_accuracy_valid",
            "exact_match_all",
            "exact_match_valid",
            "continuity_all",
            "continuity_valid",
            "horizon_compliance_all",
            "horizon_compliance_valid",
            "none_count",
            "exact_match_valid_count",
        ):
            anchor_sensitivity.append({
                "model": model,
                "metric": metric,
                "including_calibration_anchor": full.get(metric),
                "excluding_calibration_anchor": included.get(metric),
                "calibration_anchor_observations": len(anchors),
                "change_excluded_minus_included": (
                    included.get(metric) - full.get(metric)
                    if isinstance(included.get(metric), (int, float))
                    and isinstance(full.get(metric), (int, float))
                    else None
                ),
            })
    write_csv(OUT / "analysis_A_publication_model_summary.csv", all_summary)
    write_csv(OUT / "calibration_anchor_sensitivity.csv", anchor_sensitivity)
    return all_summary, anchor_sensitivity


def output_probe_crosswalk(
    rows: list[dict[str, str]],
    old_rows: list[dict[str, str]],
) -> None:
    old_by_source = {
        (row["model"], row["source_row"]): row
        for row in old_rows
    }
    fields = (
        "model",
        "source_file",
        "source_row",
        "run_id",
        "matrix_key",
        "seed",
        "tier",
        "gravity",
        "z",
        "task_hash",
        "valid",
        "exception_code",
        "output_compliance",
        "reported_taxonomy",
        "strict_fully_correct",
        "exact_match",
        "horizon_match",
        "predicted_length",
        "ground_truth_length",
        "horizon_compliance",
        "step_accuracy",
        "first_divergence_position",
        "first_divergence_kind",
        "first_predicted_coordinate",
        "first_ground_truth_coordinate",
        "state_tracking_fallback",
        "mechanism",
        "subtype",
    )
    output = []
    for row in rows:
        old = old_by_source.get((row["model"], row["source_row"]), {})
        output.append({
            **{field: row.get(field, "") for field in fields},
            "historical_taxonomy": old.get("reported_taxonomy", ""),
            "publication_included": not is_calibration_anchor(row),
            "publication_exclusion_reason": (
                "fixed parsing/environment anchor (g=0,z=3,tier=1)"
                if is_calibration_anchor(row) else ""
            ),
        })
    write_csv(OUT / "taxonomy_probe_crosswalk.csv", output)


def output_taxonomy_cross_tabs(
    rows: list[dict[str, str]],
    old_rows: list[dict[str, str]],
) -> None:
    old_by_source = {
        (row["model"], row["source_row"]): row
        for row in old_rows
    }
    counts: Counter[tuple[str, str, str, bool]] = Counter()
    for row in rows:
        if is_calibration_anchor(row) or not truth(row.get("valid")):
            continue
        exact = truth(row.get("exact_match"))
        counts[(
            row["model"], "strict", row.get("reported_taxonomy", ""), exact
        )] += 1
        old = old_by_source.get((row["model"], row["source_row"]), {})
        if truth(old.get("valid")):
            counts[(
                row["model"], "historical", old.get("reported_taxonomy", ""), exact
            )] += 1
    cross = []
    for (model, version, category, exact), count in sorted(counts.items()):
        cross.append({
            "model": model,
            "taxonomy_version": version,
            "taxonomy_label": category,
            "exact_match": exact,
            "valid_probe_count": count,
        })
    write_csv(OUT / "taxonomy_exact_match_cross_tab.csv", cross)


def output_state_tracking(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for model in sorted({row["model"] for row in rows}):
        valid = [
            row for row in rows
            if row["model"] == model
            and not is_calibration_anchor(row)
            and truth(row.get("valid"))
        ]
        state = [
            row for row in valid
            if row.get("reported_taxonomy") == "State Tracking Failure"
        ]
        fallback = sum(truth(row.get("state_tracking_fallback")) for row in state)
        count = len(state)
        output.append({
            "model": model,
            "valid_probe_denominator": len(valid),
            "state_tracking_total": count,
            "state_tracking_explicit_rule": count - fallback,
            "state_tracking_fallback": fallback,
            "fallback_fraction_of_state_tracking": fallback / count if count else 0.0,
            "fallback_fraction_of_valid_probes": fallback / len(valid) if valid else 0.0,
            "state_tracking_share_valid": count / len(valid) if valid else 0.0,
            "state_tracking_share_if_fallback_unknown": (
                (count - fallback) / len(valid) if valid else 0.0
            ),
        })
    write_csv(OUT / "state_tracking_breakdown.csv", output)
    return output


def output_taxonomy_sensitivity(rows: list[dict[str, str]]) -> None:
    counts: Counter[tuple[str, str, str]] = Counter()
    denominators: Counter[tuple[str, str]] = Counter()
    for row in rows:
        if is_calibration_anchor(row) or not truth(row.get("valid")):
            continue
        model = row["model"]
        primary = row.get("reported_taxonomy", "Unknown / Unmapped")
        mismatch = (
            integer(row.get("predicted_length"))
            != integer(row.get("ground_truth_length"))
        )
        fallback = truth(row.get("state_tracking_fallback"))
        transition = truth(row.get("transition_first"))
        policies = {
            "A_strict_original_precedence": primary,
            "B_unmapped_divergences_unknown": (
                "Unknown / Unmapped" if fallback else primary
            ),
            "C_any_horizon_mismatch_is_horizon_failure": (
                "Horizon Collapse" if mismatch else primary
            ),
            "D_horizon_and_unmapped_unknown": (
                "Unknown / Unmapped"
                if fallback
                else "Horizon Collapse" if mismatch else primary
            ),
            "E_transition_first": (
                "Transition Rule Failure" if transition else primary
            ),
        }
        for policy, category in policies.items():
            counts[(model, policy, category)] += 1
            denominators[(model, policy)] += 1
    output = []
    categories = sorted(set(CATEGORIES) | {"Unknown / Unmapped"})
    for model, policy in sorted(denominators):
        for category in categories:
            count = counts[(model, policy, category)]
            denominator = denominators[(model, policy)]
            output.append({
                "model": model,
                "policy": policy,
                "category": category,
                "count_valid": count,
                "valid_denominator": denominator,
                "proportion_valid": count / denominator,
            })
    write_csv(OUT / "taxonomy_sensitivity_publication.csv", output)


def output_horizon_distribution(rows: list[dict[str, str]]) -> None:
    by_condition: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if is_calibration_anchor(row):
            continue
        by_condition[(row["model"], integer(row.get("tier")), integer(row.get("z")))].append(row)
    totals: Counter[tuple[str, int]] = Counter()
    unique_totals: dict[tuple[str, int], set[str]] = defaultdict(set)
    for (model, tier, _z), group in by_condition.items():
        totals[(model, tier)] += len(group)
        unique_totals[(model, tier)].update(
            row["task_hash"] for row in group if row.get("task_hash")
        )
    output = []
    for (model, tier, z), group in sorted(by_condition.items()):
        hashes = {row["task_hash"] for row in group if row.get("task_hash")}
        output.append({
            "model": model,
            "publication_cohort": model in PUBLICATION_MODELS,
            "tier": tier,
            "horizon": z,
            "raw_observations": len(group),
            "valid_observations": sum(truth(row.get("valid")) for row in group),
            "invalid_observations": sum(not truth(row.get("valid")) for row in group),
            "unique_task_hashes": len(hashes),
            "share_of_model_tier_observations": (
                len(group) / totals[(model, tier)] if totals[(model, tier)] else None
            ),
            "share_of_model_tier_unique_tasks": (
                len(hashes) / len(unique_totals[(model, tier)])
                if unique_totals[(model, tier)] else None
            ),
            "gravity_values_collapsed": "0.5;3.0;15.0",
            "gravity_interpretation": "same task hash/prompt repeated; not independent conditions",
        })
    write_csv(OUT / "horizon_distribution_publication.csv", output)


def output_tier_task_matching(rows: list[dict[str, str]]) -> None:
    task_sets: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    for row in rows:
        if is_calibration_anchor(row):
            continue
        key = (row["model"], integer(row.get("tier")), integer(row.get("z")))
        if row.get("task_hash"):
            task_sets[key].add(row["task_hash"])
    output = []
    models = sorted({key[0] for key in task_sets})
    for model in models:
        tiers = sorted({key[1] for key in task_sets if key[0] == model})
        horizons = sorted({key[2] for key in task_sets if key[0] == model})
        for z in horizons:
            for i, left_tier in enumerate(tiers):
                left = task_sets.get((model, left_tier, z), set())
                for right_tier in tiers[i + 1:]:
                    right = task_sets.get((model, right_tier, z), set())
                    common = left & right
                    output.append({
                        "model": model,
                        "horizon": z,
                        "tier_left": left_tier,
                        "tier_right": right_tier,
                        "left_unique_tasks": len(left),
                        "right_unique_tasks": len(right),
                        "common_task_hashes": len(common),
                        "exact_task_match": bool(common),
                    })
    write_csv(OUT / "tier_task_matching_publication.csv", output)


def score_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    valid = [row for row in rows if truth(row.get("valid"))]
    out: dict[str, float | None] = {
        "invalid_response_rate": (
            sum(not truth(row.get("valid")) for row in rows) / len(rows)
            if rows else None
        ),
        "probe_count": float(len(rows)),
        "valid_probe_count": float(len(valid)),
        "unique_task_hashes": float(len({
            row.get("task_hash") for row in rows if row.get("task_hash")
        })),
    }
    for metric in METRICS:
        out[f"{metric}_all"] = mean([
            metric_value(row, metric) if truth(row.get("valid")) else 0.0
            for row in rows
        ])
        out[f"{metric}_valid"] = mean([
            metric_value(row, metric) for row in valid
        ])
    for category in CATEGORIES:
        out[f"taxonomy_{re.sub(r'[^a-z0-9]+', '_', category.lower()).strip('_')}_share_valid"] = (
            sum(row.get("reported_taxonomy") == category for row in valid) / len(valid)
            if valid else None
        )
    return out


def output_matched_horizon_metrics(rows: list[dict[str, str]]) -> None:
    groups: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if is_calibration_anchor(row):
            continue
        key = (row["model"], integer(row.get("tier")), integer(row.get("z")))
        if key[2] in (20, 30):
            groups[key].append(row)
    output = []
    for (model, tier, z), group in sorted(groups.items()):
        metrics = score_metrics(group)
        base = {
            "model": model,
            "tier": tier,
            "horizon": z,
            "n_seed_clusters": len({row["seed"] for row in group}),
            "uncertainty_unit": "master seed; cluster bootstrap, 2,000 resamples",
            "matching_scope": "fixed tier/horizon; task hashes reported separately",
        }
        for metric, estimate in metrics.items():
            if metric in {"probe_count", "valid_probe_count", "unique_task_hashes"}:
                output.append({**base, "metric": metric, "estimate": estimate})
                continue
            key = metric.removesuffix("_all").removesuffix("_valid")
            valid_only = metric.endswith("_valid")
            if key.startswith("taxonomy_") or key == "invalid_response_rate":
                # Bootstrap binary indicators at the probe level, clustered by seed.
                if key == "invalid_response_rate":
                    boot_key = "__invalid__"
                    bootstrap_rows = [
                        {**row, boot_key: float(not truth(row.get("valid")))}
                        for row in group
                    ]
                    lower, upper, clusters = seed_cluster_ci(
                        bootstrap_rows, boot_key, False
                    )
                elif valid_only:
                    category = key.removeprefix("taxonomy_").removesuffix("_share")
                    label = category.replace("_", " ").title()
                    # Preserve known capitalization in the taxonomy names.
                    label = next(
                        (candidate for candidate in CATEGORIES
                         if re.sub(r"[^a-z0-9]+", "_", candidate.lower()).strip("_") == category),
                        label,
                    )
                    bootstrap_rows = [
                        {**row, "__category__": float(row.get("reported_taxonomy") == label)}
                        for row in group
                    ]
                    lower, upper, clusters = seed_cluster_ci(
                        bootstrap_rows, "__category__", True
                    )
                else:
                    lower = upper = None
                    clusters = len({row["seed"] for row in group})
            else:
                lower, upper, clusters = seed_cluster_ci(
                    group, key, valid_only
                )
            output.append({
                **base,
                "metric": metric,
                "estimate": estimate,
                "ci95_lower_cluster_seed": lower,
                "ci95_upper_cluster_seed": upper,
                "n_seed_clusters": clusters,
            })
    write_csv(OUT / "matched_horizon_metrics_publication.csv", output)


def clustered_pair_ci(
    differences: dict[str, list[float]],
    *,
    replicates: int = 2_000,
) -> tuple[float | None, float | None]:
    seeds = sorted(differences)
    if not seeds:
        return None, None
    rng = random.Random(241106)
    draws = []
    for _ in range(replicates):
        selected = [rng.choice(seeds) for _ in seeds]
        values = [
            value
            for seed in selected
            for value in differences[seed]
        ]
        draws.append(sum(values) / len(values) if values else 0.0)
    draws.sort()
    return draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]


def output_matched_pairs(rows: list[dict[str, str]]) -> None:
    by_condition: dict[tuple[str, int, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if is_calibration_anchor(row):
            continue
        z = integer(row.get("z"))
        if z in (20, 30) and row.get("task_hash"):
            by_condition[(row["model"], integer(row.get("tier")), z, row["task_hash"])].append(row)
    models = sorted({key[0] for key in by_condition if key[0] in PUBLICATION_MODELS})
    output = []
    for tier in range(4):
        for z in (20, 30):
            for i, left in enumerate(models):
                for right in models[i + 1:]:
                    left_hashes = {
                        key[3] for key in by_condition
                        if key[:3] == (left, tier, z)
                    }
                    right_hashes = {
                        key[3] for key in by_condition
                        if key[:3] == (right, tier, z)
                    }
                    common = sorted(left_hashes & right_hashes)
                    if not common:
                        continue
                    for metric in METRICS:
                        for valid_only in (False, True):
                            per_seed: dict[str, list[float]] = defaultdict(list)
                            n_pairs = 0
                            for task_hash in common:
                                lrows = by_condition[(left, tier, z, task_hash)]
                                rrows = by_condition[(right, tier, z, task_hash)]
                                lsel = [r for r in lrows if not valid_only or truth(r.get("valid"))]
                                rsel = [r for r in rrows if not valid_only or truth(r.get("valid"))]
                                if not lsel or not rsel:
                                    continue
                                if not valid_only:
                                    lv = mean([
                                        metric_value(r, metric) if truth(r.get("valid")) else 0.0
                                        for r in lsel
                                    ])
                                    rv = mean([
                                        metric_value(r, metric) if truth(r.get("valid")) else 0.0
                                        for r in rsel
                                    ])
                                else:
                                    lv = mean([metric_value(r, metric) for r in lsel])
                                    rv = mean([metric_value(r, metric) for r in rsel])
                                if lv is None or rv is None:
                                    continue
                                n_pairs += 1
                                seed = str(lsel[0].get("seed", ""))
                                per_seed[seed].append(lv - rv)
                            seed_means = {
                                seed: values for seed, values in per_seed.items() if seed
                            }
                            estimate = mean([
                                value
                                for seed_values in seed_means.values()
                                for value in seed_values
                            ])
                            lower, upper = clustered_pair_ci(seed_means)
                            output.append({
                                "model_left": left,
                                "model_right": right,
                                "tier": tier,
                                "horizon": z,
                                "metric": metric,
                                "estimand": (
                                    "valid-conditioned; retain task hash only when both models have valid responses"
                                    if valid_only
                                    else "unconditional; invalid responses scored zero"
                                ),
                                "estimate_left_minus_right": estimate,
                                "ci95_lower_cluster_seed": lower,
                                "ci95_upper_cluster_seed": upper,
                                "common_task_hashes": len(common),
                                "paired_task_hashes_used": n_pairs,
                                "seed_clusters": len(seed_means),
                                "bootstrap_replicates": 2_000 if seed_means else 0,
                            })
    write_csv(OUT / "matched_horizon_pair_contrasts.csv", output)


def output_post_run_rerun() -> list[dict[str, Any]]:
    raw_files = sorted(
        path for path in (ROOT / "outputs").glob("absolute_raw_data_local_*.txt")
        if ".analyzed." not in path.name and "_stub" not in path.name
    )
    if len(raw_files) != 14:
        raise RuntimeError(f"Expected 14 named raw logs; found {len(raw_files)}")
    table_rows = []
    run_manifest = []
    for path in raw_files:
        data = post_run.process_txt_file(str(path))
        matrix = data["results_matrix"]
        filtered = {
            key: value for key, value in matrix.items()
            if not (
                integer(value.get("z")) == 3
                and integer(value.get("tier")) == 1
                and abs(number(value.get("gravity")) - 0.0) < 1e-12
            )
        }
        aggregate = post_run._compute_aggregated_metrics(
            filtered, data.get("seeds", []), None
        )
        report = post_run._compute_report_metrics(
            filtered,
            data.get("model", "unknown"),
            "COMPLETE",
            len(filtered),
            len(filtered),
            aggregate["step_accuracy"],
        )
        cri = aggregate["cri"]
        row: dict[str, Any] = {
            "model": data.get("model", "unknown"),
            "raw_log": path.name,
            "logged_observations": len(matrix),
            "calibration_observations_excluded": len(matrix) - len(filtered),
            "analysis_observations": len(filtered),
            "valid": report["valid_count"],
            "invalid": report["invalid_count"],
            "validity": report["valid_count"] / len(filtered) if filtered else None,
            "step_accuracy_all_pct": aggregate["step_accuracy"],
            "exact_match_all_pct": aggregate["exact_match"],
            "continuity_all_pct": aggregate["continuity"],
            "horizon_compliance_all_pct": aggregate["horizon_compliance"],
            "generation_bloat_index": aggregate["generation_bloat_index"],
            "generation_efficiency_pct": aggregate["efficiency"],
            "correct_steps_per_second": aggregate["latency_diagnostics"].get(
                "mean_correct_steps_per_sec"
            ),
            "average_output_tokens": report["avg_output_tokens"],
            "total_provider_cost_reported": report["total_cost"],
            "posthoc_fracture_depth": aggregate["fracture_depth"],
            "cri": cri.get("cri"),
            "trajectory_fidelity": cri.get("trajectory_fidelity"),
            "horizon_robustness": cri.get("horizon_robustness"),
            "semantic_robustness": cri.get("semantic_robustness"),
            "cri_generation_efficiency": cri.get("generation_efficiency"),
        }
        for tier, values in sorted(aggregate["per_tier"].items()):
            row[f"tier_{tier}_step_accuracy_pct"] = round(values["accuracy"] * 100, 1)
            row[f"tier_{tier}_fracture_depth"] = values["fracture_depth"]
        table_rows.append(row)
        report_path = (
            OUT
            / "post_run_raw"
            / (path.stem + "_report.txt")
        )
        report_text = report_path.read_text(encoding="utf-8-sig") if report_path.exists() else ""
        checks = re.findall(r"^\s*([A-E]\)[^\r\n]*\[(PASS|FAIL)\])\s*$", report_text, re.M)
        run_manifest.append({
            "input_log": path.name,
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "output_report": str(report_path.relative_to(ROOT)) if report_path.exists() else "",
            "post_run_command_completed": report_path.exists(),
            "logged_entries": len(matrix),
            "calibration_anchor_entries": len(matrix) - len(filtered),
            "analysis_entries": len(filtered),
            "validity_partition_pass": report["valid_count"] + report["invalid_count"] == len(filtered),
            "strict_none_equals_exact_match_count": (
                report["failure_taxonomy"]["None"]
                == sum(
                    bool(item.get("exact_match"))
                    for item in filtered.values()
                    if item.get("probe_classification", {}).get("valid", True)
                )
            ),
            "post_run_audit_checks": "; ".join(
                f"{item.strip()}" for item in re.findall(
                    r"^\s*[A-E]\)[^\r\n]*\[(?:PASS|FAIL)\]", report_text, re.M
                )
            ),
            "reported_dollar_cost_is_measured": any(
                result.get("cost") is not None for result in filtered.values()
            ),
        })
    write_csv(OUT / "post_run_table_metrics_publication.csv", table_rows)
    write_csv(OUT / "post_run_rerun_manifest.csv", run_manifest)
    return table_rows


def main() -> None:
    rows = read_csv(OUT / "analysis_A_probe_level.csv")
    old_rows = read_csv(OUT / "probe_level_audit.csv")
    if len(rows) != 55_033:
        raise RuntimeError(f"Unexpected corrected probe count: {len(rows)}")
    if any(row.get("model") == "stub" for row in rows):
        raise RuntimeError("The test stub must not enter the model analysis")

    output_model_summaries(rows)
    output_probe_crosswalk(rows, old_rows)
    output_taxonomy_cross_tabs(rows, old_rows)
    output_state_tracking(rows)
    output_taxonomy_sensitivity(rows)
    output_horizon_distribution(rows)
    output_tier_task_matching(rows)
    output_matched_horizon_metrics(rows)
    output_matched_pairs(rows)
    output_post_run_rerun()
    print(
        "Generated calibration-excluded forensic summaries, cross-tabs, "
        "horizon analyses, matched comparisons, and post-run metric tables."
    )


if __name__ == "__main__":
    main()
