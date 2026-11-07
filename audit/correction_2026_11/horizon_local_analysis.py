#!/usr/bin/env python3
"""Observed-only exact and horizon-local reviewer analyses.

Predeclared support rule: at least 10 valid observations, 3 master seeds, and
5 distinct recorded experiment hashes. The same rule applies to cells, each
common exact horizon in a standardized comparison, and paired task matches.
Intervals resample whole master-seed blocks; at least 95% of draws must have
support for every included horizon. Three to five clusters remain a limitation.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from audit.correction_2026_11 import reviewer_response_analysis as base

MIN_VALID = 10
MIN_SEEDS = 3
MIN_TASKS = 5
MIN_BOOTSTRAP_FRACTION = 0.95
SCOPES = (("exact", 0, (20, 30)), ("local_3", 3, base.GRID), ("local_5", 5, base.GRID))
OUTPUT_PREFIX = "horizon_local_"


def valid_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r["valid"]]


def support(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "N/A-U"
    valid = valid_rows(rows)
    if not valid:
        return "N/A-N"
    seeds = {str(r["seed"]) for r in valid if str(r["seed"]).strip()}
    tasks = {r["task_hash"] for r in valid if r["task_hash"]}
    if len(valid) < MIN_VALID or len(seeds) < MIN_SEEDS or len(tasks) < MIN_TASKS:
        return "N/A-I"
    return "OK"


def cell_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    valid = valid_rows(rows)
    status = support(rows)
    frequencies: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        frequencies[row["z"]]["valid" if row["valid"] else "invalid"] += 1
    for r in valid:
        if r["scored_transitions"] != r["z"] - 1 or r["full_reference_states"] != r["z"] + 1:
            raise AssertionError("Invalid requested-z scoring denominator or full reference")
        if r["taxonomy"] == "None" and (
            not r["exact_match_full_reference"] or r["coordinate_interpretation"] != "canonical"
        ):
            raise AssertionError("None classification without full canonical exact match")
    counts = Counter(r["taxonomy"] for r in valid)
    if sum(counts.values()) != len(valid):
        raise AssertionError("Taxonomy partition does not match valid denominator")
    metric_ci, taxonomy_ci = base.cluster_bootstrap(valid, f"local-v1|{key}") if status == "OK" else ({}, {})
    result: dict[str, Any] = {
        "status": status, "observations": len(rows), "valid": len(valid),
        "invalid": len(rows) - len(valid),
        "master_seeds_all": len({str(r["seed"]) for r in rows if str(r["seed"]).strip()}),
        "master_seeds_valid": len({str(r["seed"]) for r in valid if str(r["seed"]).strip()}),
        "task_hashes_all": len({r["task_hash"] for r in rows if r["task_hash"]}),
        "task_hashes_valid": len({r["task_hash"] for r in valid if r["task_hash"]}),
        "horizon_min": min(frequencies) if frequencies else "",
        "horizon_max": max(frequencies) if frequencies else "",
        "horizon_frequencies": json.dumps(
            {str(z): {"valid": frequencies[z]["valid"], "invalid": frequencies[z]["invalid"]}
             for z in sorted(frequencies)}, separators=(",", ":")
        ),
        "task_match_exists": False,
        "paired_comparison_supported": False,
        "support_rule": ">=10 valid observations; >=3 valid master seeds; >=5 valid experiment hashes",
    }
    for metric in base.METRICS:
        values = [r[metric] for r in valid if r[metric] is not None]
        result[f"mean_{metric}"] = sum(values) / len(values) if values else ""
        interval = metric_ci.get(metric, (None, None))
        result[f"{metric}_ci95_lower"] = interval[0] if interval[0] is not None else ""
        result[f"{metric}_ci95_upper"] = interval[1] if interval[1] is not None else ""
    for label in base.TAXONOMY:
        keyname = {
            "None": "none", "Horizon Collapse": "horizon",
            "State Tracking Failure": "state_tracking", "Transition Rule Failure": "transition",
            "Semantic Interpretation Failure": "semantic", "Formatting Failure": "format",
            "Unknown / Unmapped": "unknown",
        }[label]
        interval = taxonomy_ci.get(label, (None, None))
        result[f"taxonomy_{keyname}_count"] = counts[label]
        result[f"taxonomy_{keyname}_proportion"] = counts[label] / len(valid) if valid else ""
        result[f"taxonomy_{keyname}_ci95_lower"] = interval[0] if interval[0] is not None else ""
        result[f"taxonomy_{keyname}_ci95_upper"] = interval[1] if interval[1] is not None else ""
    return result


def standardized_comparison(
    left: list[dict[str, Any]], right: list[dict[str, Any]], key: str
) -> dict[str, Any]:
    left_by_z: dict[int, list[dict[str, Any]]] = defaultdict(list)
    right_by_z: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in left:
        left_by_z[r["z"]].append(r)
    for r in right:
        right_by_z[r["z"]].append(r)
    common_observed = sorted(set(left_by_z) & set(right_by_z))
    common = [z for z in common_observed if support(left_by_z[z]) == support(right_by_z[z]) == "OK"]
    comparison_status = ("N/A-U" if not left or not right else
                         "N/A-N" if not valid_rows(left) or not valid_rows(right) else
                         "N/A-I")
    result: dict[str, Any] = {
        "left_status": support(left), "right_status": support(right),
        "comparison_status": comparison_status,
        "observed_horizon_intersection": ";".join(map(str, common_observed)),
        "supported_horizon_intersection": ";".join(map(str, common)),
        "common_horizon_count": len(common),
        "weights": "equal across supported shared exact horizons" if common else "",
        "comparison_kind": "exact-horizon-standardized, not task-paired",
        "standardized_left": "", "standardized_right": "", "difference": "",
        "difference_ci95_lower": "", "difference_ci95_upper": "",
        "bootstrap_successful_draws": 0,
        "matched_task_hashes": 0, "matched_seed_task_pairs": 0,
        "paired_horizons": "", "paired_status": comparison_status,
        "paired_difference": "", "paired_ci95_lower": "", "paired_ci95_upper": "",
    }
    paired_by_z = {}
    for z in common_observed:
        ltask: dict[tuple[str, str, float], list[float]] = defaultdict(list)
        rtask: dict[tuple[str, str, float], list[float]] = defaultdict(list)
        for r in valid_rows(left_by_z[z]):
            if r["task_hash"]:
                ltask[(str(r["seed"]), r["task_hash"], float(r["gravity"]))].append(r["step_accuracy"])
        for r in valid_rows(right_by_z[z]):
            if r["task_hash"]:
                rtask[(str(r["seed"]), r["task_hash"], float(r["gravity"]))].append(r["step_accuracy"])
        paired = [
            {"seed": seed, "task_hash": task, "valid": True,
             "step_accuracy": sum(ltask[(seed, task, gravity)]) / len(ltask[(seed, task, gravity)])
             - sum(rtask[(seed, task, gravity)]) / len(rtask[(seed, task, gravity)])}
            for seed, task, gravity in sorted(ltask.keys() & rtask.keys())
        ]
        if paired:
            paired_by_z[z] = paired
    pairs = [p for values in paired_by_z.values() for p in values]
    result["matched_seed_task_pairs"] = len(pairs)
    result["matched_task_hashes"] = len({p["task_hash"] for p in pairs})
    if not common or support(left) != "OK" or support(right) != "OK":
        return result
    left_valid = {z: valid_rows(left_by_z[z]) for z in common}
    right_valid = {z: valid_rows(right_by_z[z]) for z in common}
    result["comparison_status"] = "OK"
    lmean = sum(sum(r["step_accuracy"] for r in left_valid[z]) / len(left_valid[z]) for z in common) / len(common)
    rmean = sum(sum(r["step_accuracy"] for r in right_valid[z]) / len(right_valid[z]) for z in common) / len(common)
    result["standardized_left"], result["standardized_right"] = lmean, rmean
    result["difference"] = lmean - rmean
    draws = standard_bootstrap(left_valid, right_valid, common, key)
    result["bootstrap_successful_draws"] = len(draws)
    if len(draws) >= math.ceil(base.BOOTSTRAP_REPLICATES * MIN_BOOTSTRAP_FRACTION):
        result["difference_ci95_lower"], result["difference_ci95_upper"] = base.percentile_interval(draws)

    paired_horizons = [z for z in common if support(paired_by_z.get(z, [])) == "OK"]
    result["paired_horizons"] = ";".join(map(str, paired_horizons))
    if paired_horizons:
        result["paired_status"] = "OK"
        result["paired_difference"] = sum(
            sum(p["step_accuracy"] for p in paired_by_z[z]) / len(paired_by_z[z])
            for z in paired_horizons
        ) / len(paired_horizons)
        zero = {z: [dict(p, step_accuracy=0.0) for p in paired_by_z[z]] for z in paired_horizons}
        paired_draws = standard_bootstrap(paired_by_z, zero, paired_horizons, key + "|paired")
        if len(paired_draws) >= math.ceil(base.BOOTSTRAP_REPLICATES * MIN_BOOTSTRAP_FRACTION):
            result["paired_ci95_lower"], result["paired_ci95_upper"] = base.percentile_interval(paired_draws)
    return result


def standard_bootstrap(
    left: dict[int, list[dict[str, Any]]], right: dict[int, list[dict[str, Any]]],
    horizons: list[int], key: str,
) -> list[float]:
    aggregates = []
    for side in (left, right):
        values: dict[tuple[int, str], list[float]] = defaultdict(list)
        for z in horizons:
            for row in side[z]:
                values[(z, str(row["seed"]))].append(row["step_accuracy"])
        aggregates.append({k: (sum(v), len(v)) for k, v in values.items()})
    lseeds = sorted({seed for _, seed in aggregates[0]})
    rseeds = sorted({seed for _, seed in aggregates[1]})
    shared = bool(set(lseeds) & set(rseeds))
    union = sorted(set(lseeds) | set(rseeds))
    draws = []
    for (lpick, rpick), frequency in bootstrap_weights(tuple(lseeds), tuple(rseeds), shared):
        picks = (lpick, rpick)
        estimates = []
        for z in horizons:
            values = []
            for side, weights in enumerate(picks):
                counts = sum(aggregates[side].get((z, seed), (0, 0))[1] * n for seed, n in weights)
                if not counts:
                    break
                values.append(sum(aggregates[side].get((z, seed), (0, 0))[0] * n for seed, n in weights) / counts)
            if len(values) != 2:
                break
            estimates.append(values[0] - values[1])
        if len(estimates) == len(horizons):
            draws.extend([sum(estimates) / len(estimates)] * frequency)
    return draws


@lru_cache(maxsize=256)
def bootstrap_weights(
    left: tuple[str, ...], right: tuple[str, ...], shared: bool,
) -> tuple[tuple[tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]], int], ...]:
    """Reuse and compress identical multinomial seed draws across comparisons."""
    rng = base.deterministic_rng(f"local-seed-draws-v1|{left}|{right}|{base.BOOTSTRAP_REPLICATES}")
    union = sorted(set(left) | set(right))
    counts: Counter[Any] = Counter()
    for _ in range(base.BOOTSTRAP_REPLICATES):
        if shared:
            selected = tuple(sorted(Counter(rng.choice(union) for _ in union).items()))
            counts[(selected, selected)] += 1
        else:
            lsel = tuple(sorted(Counter(rng.choice(left) for _ in left).items()))
            rsel = tuple(sorted(Counter(rng.choice(right) for _ in right).items()))
            counts[(lsel, rsel)] += 1
    return tuple(counts.items())


def window_rows(
    groups: dict[tuple[str, int, int], list[dict[str, Any]]],
    model: str, tier: int, target: int, bandwidth: int,
) -> list[dict[str, Any]]:
    return [
        row for z in range(target - bandwidth, target + bandwidth + 1)
        for row in groups.get((model, tier, z), [])
    ]


def main() -> None:
    rows = [r for r in base.read_csv(base.PROBE_FILE) if r["model"] in base.PUBLICATION_MODELS]
    if {r["model"] for r in rows} != set(base.PUBLICATION_MODELS):
        raise ValueError("Incomplete publication cohort")
    namespace = base.checkpoint_namespace(sorted({r["source_file"] for r in rows}))
    base.CHECKPOINTS = base.CheckpointStore(base.OUT / ".reviewer_response_cache" / "checkpoints.sqlite3", namespace)
    local_cache = base.CheckpointStore(
        base.OUT / ".reviewer_response_cache" / "horizon_local.sqlite3",
        namespace + "|" + base.file_sha256(Path(__file__)),
    )
    try:
        records, _manifest = base.load_and_verify_raw(rows)
        records = [r for r in records if not base.is_calibration_anchor({
            "z": str(r["z"]), "tier": str(r["tier"]), "gravity": str(r["gravity"])
        })]
        groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
        for r in records:
            groups[(r["model"], r["tier"], r["z"])].append(r)
        cells, comparisons = [], []
        for scope, bandwidth, targets in SCOPES:
            for h in targets:
                base.progress(f"[{scope} target {h}] building cells and controlled comparisons")
                cell_rows = {
                    (model, tier): window_rows(groups, model, tier, h, bandwidth)
                    for model in base.PUBLICATION_MODELS for tier in base.TIERS
                }
                summary = {
                    key: cell_summary(group, f"{scope}|{h}|{key[0]}|{key[1]}")
                    for key, group in cell_rows.items()
                }
                pairs = []
                for model in base.PUBLICATION_MODELS:
                    pairs.extend(((model, a), (model, b), "tier") for a, b in itertools.combinations(base.TIERS, 2))
                for tier in base.TIERS:
                    pairs.extend(((a, tier), (b, tier), "model") for a, b in itertools.combinations(base.PUBLICATION_MODELS, 2))
                for left, right, kind in pairs:
                    key = f"{scope}|{h}|{left}|{right}"
                    comp = local_cache.get(key)
                    if comp is None:
                        comp = standardized_comparison(cell_rows[left], cell_rows[right], key)
                        local_cache.put(key, comp)
                    if comp["matched_seed_task_pairs"]:
                        summary[left]["task_match_exists"] = True
                        summary[right]["task_match_exists"] = True
                    if comp["paired_status"] == "OK":
                        summary[left]["paired_comparison_supported"] = True
                        summary[right]["paired_comparison_supported"] = True
                    comparisons.append({
                        "scope": scope, "bandwidth": bandwidth, "target_horizon": h,
                        "comparison_axis": kind, "model_left": left[0], "tier_left": left[1],
                        "model_right": right[0], "tier_right": right[1], **comp,
                    })
                cells.extend({
                    "scope": scope, "bandwidth": bandwidth, "target_horizon": h,
                    "model": model, "tier": tier, **summary[(model, tier)],
                } for model in base.PUBLICATION_MODELS for tier in base.TIERS)
                base.CHECKPOINTS.flush()
                local_cache.flush()
        base.write_csv(base.OUT / f"{OUTPUT_PREFIX}cells.csv", cells)
        base.write_csv(base.OUT / f"{OUTPUT_PREFIX}comparisons.csv", comparisons)
        write_sensitivity(cells, comparisons)
        base.progress(f"Wrote {len(cells)} cells and {len(comparisons)} comparisons")
    finally:
        local_cache.close()
        base.CHECKPOINTS.close()
        base.CHECKPOINTS = None


def write_sensitivity(cells: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> None:
    index = {(r["scope"], r["target_horizon"], r["model"], r["tier"]): r for r in cells}
    cindex = {
        (r["scope"], r["target_horizon"], r["comparison_axis"],
         r["model_left"], r["tier_left"], r["model_right"], r["tier_right"]): r
        for r in comparisons
    }
    rows = []
    for r in comparisons:
        if r["scope"] != "local_3":
            continue
        key = ("local_5", r["target_horizon"], r["comparison_axis"],
               r["model_left"], r["tier_left"], r["model_right"], r["tier_right"])
        wider = cindex[key]
        both = r["comparison_status"] == wider["comparison_status"] == "OK"
        d3, d5 = r["difference"], wider["difference"]
        rows.append({
            "target_horizon": r["target_horizon"], "comparison_axis": r["comparison_axis"],
            "model_left": r["model_left"], "tier_left": r["tier_left"],
            "model_right": r["model_right"], "tier_right": r["tier_right"],
            "status_3": r["comparison_status"], "status_5": wider["comparison_status"],
            "shared_horizons_3": r["supported_horizon_intersection"],
            "shared_horizons_5": wider["supported_horizon_intersection"],
            "difference_3": d3, "difference_5": d5,
            "difference_change_5_minus_3": d5 - d3 if both else "",
            "same_direction": (d3 * d5 > 0) if both and d3 != 0 and d5 != 0 else "",
            "same_horizon_weights": r["supported_horizon_intersection"] == wider["supported_horizon_intersection"] if both else "",
            "interpretation": "descriptive; overlapping neighborhoods are dependent",
        })
    base.write_csv(base.OUT / f"{OUTPUT_PREFIX}bandwidth_comparisons.csv", rows)
    cell_rows = []
    for r in cells:
        if r["scope"] != "local_3":
            continue
        wider = index[("local_5", r["target_horizon"], r["model"], r["tier"])]
        cell_rows.append({
            "target_horizon": r["target_horizon"], "model": r["model"], "tier": r["tier"],
            "status_3": r["status"], "status_5": wider["status"],
            "valid_3": r["valid"], "valid_5": wider["valid"],
            "horizons_3": r["horizon_frequencies"], "horizons_5": wider["horizon_frequencies"],
            "mean_3": r["mean_step_accuracy"], "mean_5": wider["mean_step_accuracy"],
            "dominant_taxonomy_3": dominant(r), "dominant_taxonomy_5": dominant(wider),
        })
    base.write_csv(base.OUT / f"{OUTPUT_PREFIX}bandwidth_cells.csv", cell_rows)


def dominant(row: dict[str, Any]) -> str:
    if row["status"] != "OK":
        return ""
    cats = ("none", "horizon", "state_tracking", "transition", "semantic", "format", "unknown")
    largest = max(row[f"taxonomy_{cat}_count"] for cat in cats)
    return ";".join(cat for cat in cats if row[f"taxonomy_{cat}_count"] == largest)


if __name__ == "__main__":
    main()
