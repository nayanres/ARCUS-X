#!/usr/bin/env python3
"""Reproduce reviewer-response summaries from preserved ARCUS-X artifacts.

This script performs no model inference and does not write to source logs or
the benchmark scoring implementation. The extended horizon metrics use the
requested z-coordinate output convention: the full z+1-state reference is
retained, the first z reference states are scored, and the initial state is
excluded from Step Accuracy.

Run ``python audit\\correction_2026_11\\reviewer_response_analysis.py`` to
resume completed raw-file and bootstrap checkpoints. Use ``--no-resume`` for
a fresh run or ``--checkpoint PATH`` to choose a checkpoint database.
Input and source-code hashes prevent reuse after the evidence changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sqlite3
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
PACKAGE = OUT / "Formatting_Instructions_for_ICLR_2027_Conference_Submissions"
RAW_DIR = ROOT / "outputs"
PROBE_FILE = OUT / "analysis_A_probe_level.csv"
SENSITIVITY_PROBES_FILE = OUT / "analysis_B_probe_level.csv"
STATE_BREAKDOWN_FILE = OUT / "state_tracking_breakdown.csv"
FRACTURE_FILE = OUT / "fracture_sensitivity.csv"
GRID = tuple(range(10, 151, 10))
RECONCILIATION_HORIZONS = (1, 2, 5, 10, 20, 30)
TIERS = (0, 1, 2, 3)
PUBLICATION_MODELS = (
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
)
TAXONOMY = (
    "None",
    "Horizon Collapse",
    "State Tracking Failure",
    "Transition Rule Failure",
    "Semantic Interpretation Failure",
    "Formatting Failure",
    "Unknown / Unmapped",
)
METRICS = (
    "step_accuracy",
    "early_accuracy",
    "middle_accuracy",
    "late_accuracy",
)
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 241106
CALIBRATION_ANCHOR = (3, 1, 0.0)
SELECTED_PROFILE_MODELS = {
    "GPT-5-mini": ("GPT-5-mini", "#CC79A7", "o"),
    "moonshotai.kimi-k2.5": ("Kimi K2.5", "#0072B2", "s"),
    "zai.glm-5": ("GLM-5", "#D55E00", "^"),
    "qwen.qwen3-coder-480b-a35b-v1:0": (
        "Qwen 3 Coder 480B A35B", "#009E73", "D"
    ),
    "mistral-large-2512": ("Mistral Large 2512", "#E69F00", "*"),
}

CHECKPOINTS: "CheckpointStore | None" = None


class CheckpointStore:
    """Persist completed units under a fingerprint of their inputs and code."""

    def __init__(self, path: Path, namespace: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS checkpoints "
            "(namespace TEXT NOT NULL, item TEXT NOT NULL, value BLOB NOT NULL, "
            "PRIMARY KEY(namespace, item))"
        )
        self.connection.commit()
        self.namespace = namespace
        self.pending = 0

    def get(self, item: str) -> Any:
        row = self.connection.execute(
            "SELECT value FROM checkpoints WHERE namespace = ? AND item = ?",
            (self.namespace, item),
        ).fetchone()
        return json.loads(zlib.decompress(row[0])) if row else None

    def put(self, item: str, value: Any) -> None:
        payload = zlib.compress(json.dumps(value, separators=(",", ":")).encode(), 1)
        self.connection.execute(
            "INSERT OR REPLACE INTO checkpoints(namespace, item, value) VALUES (?, ?, ?)",
            (self.namespace, item, payload),
        )
        self.pending += 1
        if self.pending >= 25:
            self.flush()

    def flush(self) -> None:
        self.connection.commit()
        self.pending = 0

    def close(self) -> None:
        self.flush()
        self.connection.close()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_namespace(raw_files: list[str]) -> str:
    digest = hashlib.sha256()
    paths = (
        Path(__file__), ROOT / "inter_horizon_analysis.py",
        ROOT / "arcus" / "evaluation" / "parser.py",
        ROOT / "arcus" / "analysis" / "reporting.py",
        ROOT / "arcus" / "analysis" / "taxonomy.py",
        ROOT / "arcus" / "analysis" / "fracture.py",
        PROBE_FILE, SENSITIVITY_PROBES_FILE, STATE_BREAKDOWN_FILE, FRACTURE_FILE,
    )
    for path in paths:
        digest.update(str(path.resolve()).encode())
        digest.update(file_sha256(path).encode())
    for filename in sorted(raw_files):
        digest.update(filename.encode())
        digest.update(file_sha256(RAW_DIR / filename).encode())
    digest.update(sys.version.encode())
    return digest.hexdigest()


def progress(message: str) -> None:
    print(message, flush=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required correction artifact is missing: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty results file: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def integer(value: Any) -> int:
    return int(float(value))


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_path(value: str, field: str) -> list[str]:
    try:
        path = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {field}") from exc
    if not isinstance(path, list) or not all(isinstance(item, str) for item in path):
        raise ValueError(f"Expected a list of coordinate strings in {field}")
    return path


def parse_json_object(value: str, field: str) -> dict[str, Any]:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {field}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"Expected an object in {field}")
    return result


def is_calibration_anchor(row: dict[str, str]) -> bool:
    return (
        integer(row["z"]) == CALIBRATION_ANCHOR[0]
        and integer(row["tier"]) == CALIBRATION_ANCHOR[1]
        and abs(float(row["gravity"]) - CALIBRATION_ANCHOR[2]) < 1e-12
    )


def percentile_interval(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    ordered = sorted(values)
    low = ordered[int(0.025 * (len(ordered) - 1))]
    high = ordered[int(0.975 * (len(ordered) - 1))]
    return low, high


def deterministic_rng(key: str) -> random.Random:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big") ^ BOOTSTRAP_SEED
    return random.Random(seed)


def cluster_bootstrap(
    rows: list[dict[str, Any]],
    key: str,
    categories: tuple[str, ...] = TAXONOMY,
) -> tuple[dict[str, tuple[float | None, float | None]], dict[str, tuple[float | None, float | None]]]:
    """Resample master-seed blocks, keeping repeated tasks/gravity calls together."""
    cache_key = f"cluster|{key}|{','.join(categories)}"
    if CHECKPOINTS is not None:
        cached = CHECKPOINTS.get(cache_key)
        if cached is not None:
            return cached
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        seed = str(row.get("seed", "")).strip()
        if seed:
            clusters[seed].append(row)
    seed_ids = sorted(clusters)
    if not seed_ids:
        return {}, {}

    metric_draws: dict[str, list[float]] = {metric: [] for metric in METRICS}
    metric_draws["invalid_rate"] = []
    taxonomy_draws: dict[str, list[float]] = {category: [] for category in categories}
    rng = deterministic_rng(key)

    aggregates = {}
    for seed, cluster in clusters.items():
        valid = [row for row in cluster if row["valid"]]
        aggregates[seed] = (
            len(cluster), len(valid),
            {metric: (
                sum(row[metric] for row in valid if row.get(metric) is not None),
                sum(row.get(metric) is not None for row in valid),
            ) for metric in METRICS},
            Counter(row["taxonomy"] for row in valid),
        )

    for _ in range(BOOTSTRAP_REPLICATES):
        selected = [rng.choice(seed_ids) for _ in seed_ids]
        sample_count = sum(aggregates[seed][0] for seed in selected)
        valid_count = sum(aggregates[seed][1] for seed in selected)
        for metric in METRICS:
            count = sum(aggregates[seed][2][metric][1] for seed in selected)
            if count:
                metric_draws[metric].append(
                    sum(aggregates[seed][2][metric][0] for seed in selected) / count
                )
        if sample_count:
            metric_draws["invalid_rate"].append(
                (sample_count - valid_count) / sample_count
            )
        if valid_count:
            for category in categories:
                taxonomy_draws[category].append(
                    sum(aggregates[seed][3][category] for seed in selected) / valid_count
                )

    result = (
        {name: percentile_interval(values) for name, values in metric_draws.items()},
        {name: percentile_interval(values) for name, values in taxonomy_draws.items()},
    )
    if CHECKPOINTS is not None:
        CHECKPOINTS.put(cache_key, result)
    return result


def requested_z_metrics(
    predicted: list[str], full_reference: list[str], z: int
) -> dict[str, Any]:
    """Score only the requested z output coordinates; retain the z+1 reference."""
    if len(full_reference) != z + 1:
        raise ValueError(
            f"z={z} requires a complete z+1 reference; found {len(full_reference)} states"
        )
    scoring_reference = full_reference[:z]
    transitions = [
        i < len(predicted) and predicted[i] == scoring_reference[i]
        for i in range(1, z)
    ]
    denominator = len(transitions)
    step_accuracy = (
        sum(transitions) / denominator if denominator else None
    )
    sizes = split_horizon(denominator)
    cursor = 0
    regions: dict[str, float | None] = {}
    for name in ("early", "middle", "end"):
        size = sizes[name]
        values = transitions[cursor:cursor + size]
        cursor += size
        metric_name = "late_accuracy" if name == "end" else f"{name}_accuracy"
        regions[metric_name] = (
            sum(values) / len(values) if values else None
        )
    return {
        "step_accuracy": step_accuracy,
        **regions,
        "requested_length_compliant": len(predicted) == z,
        "exact_match_requested_output": predicted == scoring_reference,
        "full_reference_states": len(full_reference),
        "scored_reference_states": len(scoring_reference),
        "scored_transitions": denominator,
    }


def conditional_response_metrics(
    valid: bool, predicted: list[str], full_reference: list[str], z: int
) -> dict[str, Any]:
    if len(full_reference) != z + 1:
        raise ValueError(
            f"z={z} requires a complete z+1 reference; found {len(full_reference)} states"
        )
    if valid:
        return requested_z_metrics(predicted, full_reference, z)
    return {
        **{name: None for name in METRICS},
        "requested_length_compliant": None,
        "exact_match_requested_output": None,
        "full_reference_states": len(full_reference),
        "scored_reference_states": z,
        "scored_transitions": max(0, z - 1),
    }


def selected_horizon_profiles(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, Any]] = defaultdict(
        lambda: {"observations": 0, "valid": 0, "accuracy": []}
    )
    for row in records:
        if row["model"] not in SELECTED_PROFILE_MODELS or is_calibration_anchor({
            "z": str(row["z"]),
            "tier": str(row["tier"]),
            "gravity": str(row["gravity"]),
        }):
            continue
        group = grouped[(row["model"], row["z"])]
        group["observations"] += 1
        if row["valid"]:
            group["valid"] += 1
            if row["original_step_accuracy"] is not None:
                group["accuracy"].append(row["original_step_accuracy"])

    output = []
    for (model, z), group in sorted(grouped.items()):
        valid = group["valid"]
        values = group["accuracy"]
        output.append({
            "model": model,
            "horizon_z": z,
            "observations": group["observations"],
            "valid_observations": valid,
            "invalid_observations": group["observations"] - valid,
            "coverage_status": "observed_with_valid" if valid else "invalid_only",
            "mean_valid_step_accuracy": (
                sum(values) / len(values) if values else ""
            ),
            "accuracy_denominator": (
                "valid responses only; all z action transitions"
                if values else ""
            ),
        })
    return output


def write_selected_horizon_plot(rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.2, 3.45))
    for model, (label, color, marker) in SELECTED_PROFILE_MODELS.items():
        points = [
            row for row in rows
            if row["model"] == model and row["mean_valid_step_accuracy"] != ""
        ]
        if not points:
            continue
        points.sort(key=lambda row: row["horizon_z"])
        axis.plot(
            [row["horizon_z"] for row in points],
            [100 * row["mean_valid_step_accuracy"] for row in points],
            marker=marker,
            markersize=3.8,
            linewidth=1.0,
            label=label,
            color=color,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.35,
        )
    axis.axhline(50, color="0.5", linestyle="--", linewidth=0.7, label="50% floor")
    axis.set_xlabel("Horizon $z$", fontsize=8)
    axis.set_ylabel("Conditional step accuracy (%)", fontsize=8)
    axis.set_ylim(0, 105)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.tick_params(axis="both", labelsize=7, pad=1)
    axis.grid(True, alpha=0.25, linewidth=0.5)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.98, bottom=0.16)
    fig.savefig(
        (PACKAGE if PACKAGE.is_dir() else OUT) / "horizon_profiles_selected_valid_only.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def write_taxonomy_profile_figure(rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    categories = (
        ("State Tracking Failure", "State Tracking"),
        ("Transition Rule Failure", "Transition Rule"),
        ("Semantic Interpretation Failure", "Semantic Interpretation"),
        ("Horizon Collapse", "Horizon Collapse"),
        ("Formatting Failure", "Formatting"),
    )
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row["validity"] == "valid":
            counts[row["model"]][row["primary_taxonomy"]] += int(row["count"])
    models = list(PUBLICATION_MODELS)
    values = np.array([
        [
            100 * counts[model][category] / sum(counts[model].values())
            for category, _ in categories
        ]
        for model in models
    ])
    model_labels = {
        "qwen.qwen3-235b-a22b-2507": "Qwen3-235B-A22B-2507*",
        "moonshotai.kimi-k2.5": "Kimi K2.5",
        "qwen.qwen3-coder-480b-a35b-v1:0": "Qwen 3 Coder 480B A35B",
        "mistral-small-2603": "Mistral Small 2603",
        "mistral-large-2512": "Mistral Large 2512",
        "mistral.devstral-2-123b": "Devstral 2 123B*",
        "ministral-14b-2512": "Ministral 14B 2512",
        "ministral-8b-2512": "Ministral 8B 2512",
        "ministral-3b-2512": "Ministral 3B 2512",
        "nvidia.nemotron-super-3-120b": "Nemotron Super 3 120B",
        "zai.glm-5": "GLM-5",
        "GPT-5-mini": "GPT-5-mini",
        "MAI-Thinking-1": "MAI-Thinking-1*",
    }

    fig, axis = plt.subplots(figsize=(11, 7.0))
    image = axis.imshow(values, vmin=0, vmax=100, aspect="auto", cmap="Greys")
    axis.set_xticks(range(len(categories)))
    axis.set_xticklabels([label for _, label in categories], fontsize=8)
    axis.set_yticks(range(len(models)))
    axis.set_yticklabels([model_labels[model] for model in models], fontsize=7)
    for i in range(len(models)):
        for j in range(len(categories)):
            value = values[i, j]
            axis.text(
                j, i, f"{value:.1f}%", ha="center", va="center", fontsize=7,
                color="white" if value > 50 else "black",
            )
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Failure proportion among valid probes", fontsize=9)
    colorbar.set_ticks([0, 20, 40, 60, 80, 100])
    colorbar.set_ticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    axis.set_xlabel("Failure signature", fontsize=10)
    axis.set_ylabel("Model", fontsize=10)
    axis.set_title("Failure-signature profiles", fontsize=12, pad=12)
    fig.tight_layout()
    figure_dir = PACKAGE if PACKAGE.is_dir() else OUT
    figure_name = "failure_profiles" if PACKAGE.is_dir() else "reviewer_response_failure_profiles"
    fig.savefig(figure_dir / f"{figure_name}.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / f"{figure_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def split_horizon(transitions: int) -> dict[str, int]:
    if transitions < 0:
        raise ValueError("transition count must be non-negative")
    base, remainder = divmod(transitions, 3)
    return {"early": base, "middle": base, "end": base + remainder}


def load_and_verify_raw(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Cross-check corrected path records against their preserved source entries."""
    from inter_horizon_analysis import parse_raw_file
    from arcus.evaluation.parser import extract_model_path

    by_file: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["model"] in PUBLICATION_MODELS:
            by_file[row["source_file"]].append(row)

    source_hashes: dict[str, str] = {}
    verified_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for index, (filename, file_rows) in enumerate(sorted(by_file.items()), 1):
        progress(f"[raw {index}/{len(by_file)}] {filename}")
        raw_path = RAW_DIR / filename
        if not raw_path.is_file():
            raise FileNotFoundError(f"Raw source log is missing: {raw_path}")
        source_hashes[filename] = file_sha256(raw_path)
        cached = CHECKPOINTS.get(f"raw|{filename}") if CHECKPOINTS is not None else None
        if cached is not None:
            if cached["manifest"]["sha256"] != source_hashes[filename]:
                raise ValueError(f"Cached raw checksum mismatch for {filename}")
            verified_rows.extend(cached["records"])
            manifest.append(cached["manifest"])
            progress(f"  resumed {len(cached['records'])} verified rows")
            continue
        entries = parse_raw_file(raw_path)
        indexed_rows = {integer(row["source_row"]): row for row in file_rows}
        if len(indexed_rows) != len(file_rows):
            raise ValueError(f"Duplicate source_row values in {filename}")
        if set(indexed_rows) != set(range(1, len(entries) + 1)):
            raise ValueError(
                f"Raw/corrected row counts differ for {filename}: "
                f"{len(entries)} raw entries vs {len(indexed_rows)} corrected rows"
            )

        file_verified = []
        for source_row, entry in enumerate(entries, start=1):
            row = indexed_rows[source_row]
            predicted = extract_model_path(entry.raw_output)
            full_reference = extract_model_path(entry.ground_truth)
            if integer(row["z"]) != entry.z:
                raise ValueError(f"Horizon mismatch in {filename} row {source_row}")
            if parse_path(row["predicted_path"], "predicted_path") != predicted:
                raise ValueError(f"Predicted path mismatch in {filename} row {source_row}")
            if parse_path(row["ground_truth_path"], "ground_truth_path") != full_reference:
                raise ValueError(f"Reference path mismatch in {filename} row {source_row}")
            if entry.actions != json.loads(row["actions"]):
                raise ValueError(f"Action metadata mismatch in {filename} row {source_row}")

            z = integer(row["z"])
            valid = truth(row["valid"])
            metrics = conditional_response_metrics(
                valid, predicted, full_reference, z
            )
            requested_length_compliant = metrics["requested_length_compliant"]
            exact_requested = metrics["exact_match_requested_output"]

            taxonomy = row.get("analysis_A_taxonomy") or row.get("reported_taxonomy") or ""
            if valid and taxonomy not in TAXONOMY:
                raise ValueError(
                    f"Unexpected primary taxonomy label {taxonomy!r} in "
                    f"{filename} row {source_row}"
                )
            if not valid:
                taxonomy = ""

            file_verified.append({
                "model": row["model"],
                "source_file": filename,
                "source_row": source_row,
                "seed": str(row["seed"]),
                "task_hash": row.get("task_hash", ""),
                "z": z,
                "tier": integer(row["tier"]),
                "gravity": row["gravity"],
                "valid": valid,
                "predicted_path": predicted,
                "ground_truth_path_full": full_reference,
                "taxonomy": taxonomy,
                "state_tracking_fallback": truth(row.get("state_tracking_fallback")),
                "coordinate_interpretation": row.get("coordinate_interpretation", ""),
                "original_step_accuracy": finite_float(row.get("step_accuracy")),
                "exact_match_full_reference": (
                    predicted == full_reference if valid else None
                ),
                "requested_length_compliant": requested_length_compliant,
                "exact_match_requested_output": exact_requested,
                **metrics,
            })

        file_manifest = {
            "raw_file": filename,
            "raw_entries": len(entries),
            "corrected_rows_verified": len(indexed_rows),
            "sha256": source_hashes[filename],
        }
        verified_rows.extend(file_verified)
        manifest.append(file_manifest)
        if CHECKPOINTS is not None:
            CHECKPOINTS.put(f"raw|{filename}", {
                "manifest": file_manifest, "records": file_verified,
            })
            CHECKPOINTS.flush()
        progress(f"  verified {len(file_verified)} rows")

    for filename, before in source_hashes.items():
        after = file_sha256(RAW_DIR / filename)
        if before != after:
            raise RuntimeError(f"Raw source log changed during analysis: {filename}")
    return verified_rows, manifest


def coverage_and_metrics(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, int, int], list[dict[str, Any]]]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(row["model"], row["z"], row["tier"])].append(row)

    coverage_rows = []
    metric_rows = []
    taxonomy_rows = []
    grid_summary = []
    categories = TAXONOMY

    for model_index, model in enumerate(PUBLICATION_MODELS, 1):
        progress(f"[coverage {model_index}/{len(PUBLICATION_MODELS)}] {model}")
        for z in GRID:
            for tier in TIERS:
                key = (model, z, tier)
                group = groups.get(key, [])
                valid = [row for row in group if row["valid"]]
                invalid_count = len(group) - len(valid)
                status = (
                    "missing" if not group
                    else "invalid_only" if not valid
                    else "observed_with_valid"
                )
                coverage_rows.append({
                    "model": model,
                    "horizon_z": z,
                    "tier": tier,
                    "coverage_status": status,
                    "observations": len(group),
                    "valid_observations": len(valid),
                    "invalid_observations": invalid_count,
                    "seed_clusters": len({row["seed"] for row in group}),
                    "valid_seed_clusters": len({row["seed"] for row in valid}),
                    "unique_task_hashes": len({
                        row["task_hash"] for row in group if row["task_hash"]
                    }),
                    "reference_state_count": z + 1 if group else "",
                    "scored_reference_state_count": z if group else "",
                })

                metric_values = {
                    metric: [row[metric] for row in valid if row[metric] is not None]
                    for metric in METRICS
                }
                metric_ci, taxonomy_ci = cluster_bootstrap(
                    group, f"{model}|{z}|{tier}"
                ) if group else ({}, {})
                metric_row: dict[str, Any] = {
                    "model": model,
                    "horizon_z": z,
                    "tier": tier,
                    "coverage_status": status,
                    "observations": len(group),
                    "valid_observations": len(valid),
                    "invalid_observations": invalid_count,
                    "accuracy_denominator": (
                        f"valid responses only; {max(0, z - 1)} post-initial coordinates"
                        if valid else ""
                    ),
                    "bootstrap_unit": "master-seed block; repeated task/gravity rows retained",
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES if valid else 0,
                    "valid_seed_clusters": len({row["seed"] for row in valid}),
                    "requested_length_compliant_valid": sum(
                        bool(row["requested_length_compliant"]) for row in valid
                    ),
                }
                for metric in METRICS:
                    values = metric_values[metric]
                    metric_row[f"mean_{metric}"] = (
                        sum(values) / len(values) if values else ""
                    )
                    low, high = metric_ci.get(metric, (None, None))
                    metric_row[f"{metric}_ci95_lower"] = low if low is not None else ""
                    metric_row[f"{metric}_ci95_upper"] = high if high is not None else ""
                    metric_row[f"{metric}_n"] = len(values)
                low, high = metric_ci.get("invalid_rate", (None, None))
                metric_row["invalid_rate"] = (
                    invalid_count / len(group) if group else ""
                )
                metric_row["invalid_rate_ci95_lower"] = low if low is not None else ""
                metric_row["invalid_rate_ci95_upper"] = high if high is not None else ""
                metric_rows.append(metric_row)

                tax_counts = Counter(row["taxonomy"] for row in valid)
                if sum(tax_counts.values()) != len(valid):
                    raise AssertionError(f"Taxonomy partition failed for {key}")
                if valid and abs(sum(tax_counts[c] / len(valid) for c in categories) - 1.0) > 1e-12:
                    raise AssertionError(f"Taxonomy shares do not sum to one for {key}")
                for category in categories:
                    low, high = taxonomy_ci.get(category, (None, None))
                    taxonomy_rows.append({
                        "model": model,
                        "horizon_z": z,
                        "tier": tier,
                        "coverage_status": status,
                        "valid_denominator": len(valid),
                        "taxonomy": category,
                        "count": tax_counts[category],
                        "proportion_valid": (
                            tax_counts[category] / len(valid) if valid else ""
                        ),
                        "share_ci95_lower": low if low is not None else "",
                        "share_ci95_upper": high if high is not None else "",
                        "bootstrap_unit": "master-seed block",
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES if valid else 0,
                    })

    grid_summary = summarize_grid_coverage(coverage_rows)
    return coverage_rows, metric_rows, taxonomy_rows, grid_summary, groups


def summarize_grid_coverage(
    coverage_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary = []
    for z in GRID:
        horizon_cells = [
            row for row in coverage_rows if row["horizon_z"] == z
        ]
        expected = len(PUBLICATION_MODELS) * len(TIERS)
        observed = sum(row["coverage_status"] != "missing" for row in horizon_cells)
        summary.append({
            "horizon_z": z,
            "model_tier_cells_expected": expected,
            "model_tier_cells_observed": observed,
            "model_tier_cells_missing": expected - observed,
            "observations": sum(row["observations"] for row in horizon_cells),
            "valid_observations": sum(row["valid_observations"] for row in horizon_cells),
            "invalid_observations": sum(row["invalid_observations"] for row in horizon_cells),
        })
    return summary


def tier_comparisons(
    groups: dict[tuple[str, int, int], list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    output = []
    for model_index, model in enumerate(PUBLICATION_MODELS, 1):
        progress(f"[tier comparisons {model_index}/{len(PUBLICATION_MODELS)}] {model}")
        for z in GRID:
            for left_tier in TIERS:
                for right_tier in TIERS:
                    if left_tier >= right_tier:
                        continue
                    left = [
                        row for row in groups.get((model, z, left_tier), [])
                        if row["valid"] and row["step_accuracy"] is not None
                    ]
                    right = [
                        row for row in groups.get((model, z, right_tier), [])
                        if row["valid"] and row["step_accuracy"] is not None
                    ]
                    left_mean = (
                        sum(row["step_accuracy"] for row in left) / len(left)
                        if left else None
                    )
                    right_mean = (
                        sum(row["step_accuracy"] for row in right) / len(right)
                        if right else None
                    )
                    left_pairs: dict[tuple[str, str], list[float]] = defaultdict(list)
                    right_pairs: dict[tuple[str, str], list[float]] = defaultdict(list)
                    for row in left:
                        if row["task_hash"]:
                            left_pairs[(row["seed"], row["task_hash"])].append(
                                row["step_accuracy"]
                            )
                    for row in right:
                        if row["task_hash"]:
                            right_pairs[(row["seed"], row["task_hash"])].append(
                                row["step_accuracy"]
                            )
                    common = sorted(set(left_pairs) & set(right_pairs))
                    common_hashes = len({
                        task_hash for _seed, task_hash in common
                    })
                    if common:
                        paired = [
                            {
                                "seed": seed,
                                "difference": (
                                    sum(left_pairs[(seed, task_hash)])
                                    / len(left_pairs[(seed, task_hash)])
                                    - sum(right_pairs[(seed, task_hash)])
                                    / len(right_pairs[(seed, task_hash)])
                                ),
                            }
                            for seed, task_hash in common
                        ]
                        clusters: dict[str, list[float]] = defaultdict(list)
                        for row in paired:
                            clusters[row["seed"]].append(row["difference"])
                        estimate = sum(
                            sum(values) / len(values) for values in clusters.values()
                        ) / len(clusters)
                        low, high = bootstrap_cluster_values(
                            clusters, f"paired|{model}|{z}|{left_tier}|{right_tier}"
                        )
                        comparison_kind = "task-paired"
                    elif left and right:
                        estimate = left_mean - right_mean
                        low, high = difference_cluster_bootstrap(
                            left,
                            right,
                            f"unpaired|{model}|{z}|{left_tier}|{right_tier}",
                        )
                        comparison_kind = "horizon-matched-not-task-paired"
                        paired = []
                    else:
                        estimate = low = high = None
                        comparison_kind = "not_estimable_missing_valid_observations"
                        paired = []

                    output.append({
                        "model": model,
                        "horizon_z": z,
                        "tier_left": left_tier,
                        "tier_right": right_tier,
                        "coverage_status": (
                            "both_tiers_observed"
                            if groups.get((model, z, left_tier))
                            and groups.get((model, z, right_tier))
                            else "one_tier_missing"
                            if groups.get((model, z, left_tier))
                            or groups.get((model, z, right_tier))
                            else "both_tiers_missing"
                        ),
                        "valid_left": len(left),
                        "valid_right": len(right),
                        "mean_step_accuracy_left": left_mean if left_mean is not None else "",
                        "mean_step_accuracy_right": right_mean if right_mean is not None else "",
                        "estimate_left_minus_right": estimate if estimate is not None else "",
                        "ci95_lower": low if low is not None else "",
                        "ci95_upper": high if high is not None else "",
                        "comparison_kind": comparison_kind,
                        "exact_paired_seed_task_rows": len(common),
                        "exact_paired_task_hashes": common_hashes,
                        "left_seed_clusters": len({row["seed"] for row in left}),
                        "right_seed_clusters": len({row["seed"] for row in right}),
                        "bootstrap_unit": (
                            "master seed; exact task-hash pairs" if common
                            else "master-seed blocks; observations not task-paired"
                        ),
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES if left and right else 0,
                    })
    return output


def cohort_tier_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for z in GRID:
        by_tier = {
            tier: [
                row for row in records
                if row["z"] == z and row["tier"] == tier
                and row["valid"] and row["step_accuracy"] is not None
            ]
            for tier in TIERS
        }
        means = {
            tier: (
                sum(row["step_accuracy"] for row in rows) / len(rows)
                if rows else None
            )
            for tier, rows in by_tier.items()
        }
        baseline = by_tier[0]
        for tier in TIERS:
            rows = by_tier[tier]
            metric_ci, _taxonomy_ci = cluster_bootstrap(
                rows, f"cohort-tier-{z}-{tier}"
            )
            low, high = metric_ci.get("step_accuracy", (None, None))
            if tier == 0:
                difference = None
                difference_low = None
                difference_high = None
            elif rows and baseline:
                difference = means[tier] - means[0]
                difference_low, difference_high = difference_cluster_bootstrap(
                    rows, baseline, f"cohort-tier-difference-{z}-{tier}"
                )
            else:
                difference = None
                difference_low = None
                difference_high = None
            output.append({
                "horizon_z": z,
                "tier": tier,
                "coverage_status": "observed_with_valid" if rows else "no_valid_observations",
                "valid_observations": len(rows),
                "models_with_valid_observations": len({row["model"] for row in rows}),
                "mean_step_accuracy": means[tier] if means[tier] is not None else "",
                "step_accuracy_ci95_lower": low if low is not None else "",
                "step_accuracy_ci95_upper": high if high is not None else "",
                "delta_vs_tier0": difference if difference is not None else "",
                "delta_ci95_lower": difference_low if difference_low is not None else "",
                "delta_ci95_upper": difference_high if difference_high is not None else "",
                "bootstrap_unit": "master-seed block; repeated model/task/gravity rows retained",
                "bootstrap_replicates": BOOTSTRAP_REPLICATES if rows else 0,
            })
    return output


def bootstrap_cluster_values(
    clusters: dict[str, list[float]], key: str
) -> tuple[float | None, float | None]:
    if not clusters:
        return None, None
    cache_key = f"paired|{key}"
    if CHECKPOINTS is not None:
        cached = CHECKPOINTS.get(cache_key)
        if cached is not None:
            return tuple(cached)
    seeds = sorted(clusters)
    means = {seed: sum(values) / len(values) for seed, values in clusters.items() if values}
    rng = deterministic_rng(key)
    draws = []
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = [rng.choice(seeds) for _ in seeds]
        sample = [
            means[seed] for seed in selected if seed in means
        ]
        if sample:
            draws.append(sum(sample) / len(sample))
    result = percentile_interval(draws)
    if CHECKPOINTS is not None:
        CHECKPOINTS.put(cache_key, result)
    return result


def difference_cluster_bootstrap(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    key: str,
) -> tuple[float | None, float | None]:
    if not left or not right:
        return None, None
    cache_key = f"difference|{key}"
    if CHECKPOINTS is not None:
        cached = CHECKPOINTS.get(cache_key)
        if cached is not None:
            return tuple(cached)
    left_clusters: dict[str, list[float]] = defaultdict(list)
    right_clusters: dict[str, list[float]] = defaultdict(list)
    for row in left:
        left_clusters[row["seed"]].append(row["step_accuracy"])
    for row in right:
        right_clusters[row["seed"]].append(row["step_accuracy"])
    shared = sorted(set(left_clusters) & set(right_clusters))
    ldata = {seed: (sum(values), len(values)) for seed, values in left_clusters.items()}
    rdata = {seed: (sum(values), len(values)) for seed, values in right_clusters.items()}
    rng = deterministic_rng(key)
    draws = []
    if shared:
        seeds = sorted(set(left_clusters) | set(right_clusters))
        for _ in range(BOOTSTRAP_REPLICATES):
            selected = [rng.choice(seeds) for _ in seeds]
            lcount = sum(ldata.get(seed, (0, 0))[1] for seed in selected)
            rcount = sum(rdata.get(seed, (0, 0))[1] for seed in selected)
            if lcount and rcount:
                draws.append(
                    sum(ldata.get(seed, (0, 0))[0] for seed in selected) / lcount
                    - sum(rdata.get(seed, (0, 0))[0] for seed in selected) / rcount
                )
    else:
        lseeds = sorted(left_clusters)
        rseeds = sorted(right_clusters)
        for _ in range(BOOTSTRAP_REPLICATES):
            lselected = [rng.choice(lseeds) for _ in lseeds]
            rselected = [rng.choice(rseeds) for _ in rseeds]
            lcount = sum(ldata[seed][1] for seed in lselected)
            rcount = sum(rdata[seed][1] for seed in rselected)
            draws.append(
                sum(ldata[seed][0] for seed in lselected) / lcount
                - sum(rdata[seed][0] for seed in rselected) / rcount
            )
    result = percentile_interval(draws)
    if CHECKPOINTS is not None:
        CHECKPOINTS.put(cache_key, result)
    return result


def taxonomy_reconciliation(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    counts: Counter[tuple[str, str, str, str, str]] = Counter()
    model_totals: dict[str, Counter[str]] = defaultdict(Counter)
    fallback_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        if is_calibration_anchor({
            "z": str(row["z"]),
            "tier": str(row["tier"]),
            "gravity": str(row["gravity"]),
        }):
            continue
        model = row["model"]
        valid = row["valid"]
        if valid:
            taxonomy = row["taxonomy"]
            exact = bool(row["exact_match_full_reference"])
            length_compliant = len(row["predicted_path"]) == row["z"]
            if taxonomy == "None" and not exact:
                raise AssertionError(
                    f"{model}: None without a complete full-reference exact match"
                )
            if taxonomy == "None" and row["coordinate_interpretation"] != "canonical":
                raise AssertionError(
                    f"{model}: None without canonical coordinates"
                )
            if exact and taxonomy != "None":
                raise AssertionError(
                    f"{model}: complete exact match not classified as None"
                )
            if taxonomy not in TAXONOMY:
                raise AssertionError(f"Unknown primary taxonomy label: {taxonomy}")
            label = taxonomy
            validity = "valid"
            exact_value = str(exact)
            length_value = str(length_compliant)
            model_totals[model]["valid"] += 1
            model_totals[model]["taxonomy_" + taxonomy] += 1
            if taxonomy == "State Tracking Failure":
                if row["state_tracking_fallback"]:
                    fallback_counts[model]["fallback"] += 1
                else:
                    fallback_counts[model]["explicit"] += 1
                model_totals[model]["state_tracking"] += 1
        else:
            label = "Invalid / Excluded"
            validity = "invalid"
            exact_value = "not_assessed"
            length_value = "not_assessed"
            model_totals[model]["invalid"] += 1
        counts[(model, validity, exact_value, length_value, label)] += 1

    cross_tab = [
        {
            "model": model,
            "validity": valid,
            "exact_match_full_reference": exact,
            "requested_length_compliant_z_coordinates": length,
            "primary_taxonomy": label,
            "count": count,
        }
        for (model, valid, exact, length, label), count in sorted(counts.items())
    ]

    fallback_rows = []
    for model in PUBLICATION_MODELS:
        total_state = fallback_counts[model]["fallback"] + fallback_counts[model]["explicit"]
        valid_total = model_totals[model]["valid"]
        fallback = fallback_counts[model]["fallback"]
        explicit = fallback_counts[model]["explicit"]
        fallback_rows.append({
            "model": model,
            "valid_observations": valid_total,
            "state_tracking_assignments": total_state,
            "explicit_state_tracking_assignments": explicit,
            "fallback_state_tracking_assignments": fallback,
            "fallback_percent_of_state_tracking": (
                100 * fallback / total_state if total_state else ""
            ),
            "fallback_percent_of_valid": (
                100 * fallback / valid_total if valid_total else ""
            ),
        })
    model_reconciliation = []
    for model in PUBLICATION_MODELS:
        valid_total = model_totals[model]["valid"]
        none_count = model_totals[model]["taxonomy_None"]
        exact_count = sum(
            count
            for (m, validity, exact, _length, _label), count in counts.items()
            if m == model and validity == "valid" and exact == "True"
        )
        if none_count != exact_count:
            raise AssertionError(
                f"{model}: valid None count {none_count} != exact-match count {exact_count}"
            )
        model_reconciliation.append({
            "model": model,
            "valid_observations": valid_total,
            "invalid_observations": model_totals[model]["invalid"],
            "valid_none_count": none_count,
            "valid_full_reference_exact_match_count": exact_count,
            "none_equals_exact_match": none_count == exact_count,
            "taxonomy_partition_sum": sum(
                model_totals[model]["taxonomy_" + category]
                for category in TAXONOMY
            ),
            "taxonomy_partition_share": (
                sum(model_totals[model]["taxonomy_" + category] for category in TAXONOMY)
                / valid_total if valid_total else ""
            ),
        })
    return cross_tab, fallback_rows, model_reconciliation


def existing_precedence_sensitivity() -> list[dict[str, Any]]:
    """Reaggregate saved per-probe labels, excluding calibration anchors."""
    policy_columns = (
        "A_corrected_original_precedence",
        "B_horizon_first",
        "B_transition_first",
        "B_conservative_unclassified",
        "B_fallback_unclassified",
        "B_modulo_equivalent",
    )
    categories = (*TAXONOMY, "Unclassified")
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    denominators: Counter[str] = Counter()
    reclassified: Counter[tuple[str, str]] = Counter()
    with SENSITIVITY_PROBES_FILE.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            model = row.get("model", "")
            if (
                model not in PUBLICATION_MODELS
                or not truth(row.get("valid"))
                or is_calibration_anchor(row)
            ):
                continue
            primary = row[policy_columns[0]]
            if primary not in categories:
                raise ValueError(f"Unknown saved primary policy label: {primary!r}")
            denominators[model] += 1
            for policy in policy_columns:
                category = row[policy]
                if category not in categories:
                    raise ValueError(f"Unknown saved sensitivity label: {category!r}")
                counts[(model, policy)][category] += 1
                if policy != policy_columns[0] and category != primary:
                    reclassified[(model, policy)] += 1

    output = []
    for model in PUBLICATION_MODELS:
        for policy in policy_columns:
            denominator = denominators[model]
            changed = reclassified[(model, policy)]
            for category in categories:
                count = counts[(model, policy)][category]
                output.append({
                    "model": model,
                    "policy": policy,
                    "category": category,
                    "count_valid": count,
                    "valid_denominator": denominator,
                    "proportion_valid": count / denominator if denominator else "",
                    "changed_valid_probes": changed,
                    "proportion_changed": changed / denominator if denominator else "",
                })
    return output


def verify_precedence_baseline(
    taxonomy_cross_tab: list[dict[str, Any]],
    sensitivity: list[dict[str, Any]],
) -> None:
    primary: Counter[tuple[str, str]] = Counter()
    for row in taxonomy_cross_tab:
        if row["validity"] == "valid":
            primary[(row["model"], row["primary_taxonomy"])] += int(row["count"])
    saved_primary = {
        (row["model"], row["category"]): int(row["count_valid"])
        for row in sensitivity
        if row["policy"] == "A_corrected_original_precedence"
    }
    for model in PUBLICATION_MODELS:
        for category in TAXONOMY:
            if primary.get((model, category), 0) != saved_primary.get((model, category), 0):
                raise AssertionError(
                    f"{model}: saved Analysis B primary label count for {category} "
                    "does not match the corrected Analysis A reconciliation"
                )


def verify_state_tracking_breakdown(
    fallback_rows: list[dict[str, Any]],
) -> None:
    source = {
        row["model"]: row for row in read_csv(STATE_BREAKDOWN_FILE)
        if row.get("model") in PUBLICATION_MODELS
    }
    for row in fallback_rows:
        model = row["model"]
        if model not in source:
            raise ValueError(f"Missing prior State Tracking breakdown for {model}")
        prior = source[model]
        fields = (
            ("valid_observations", "valid_probe_denominator"),
            ("state_tracking_assignments", "state_tracking_total"),
            ("explicit_state_tracking_assignments", "state_tracking_explicit_rule"),
            ("fallback_state_tracking_assignments", "state_tracking_fallback"),
        )
        for current_field, source_field in fields:
            if int(row[current_field]) != integer(prior[source_field]):
                raise AssertionError(
                    f"{model}: fallback audit does not match stored "
                    f"{source_field} ({row[current_field]} != {prior[source_field]})"
                )


def summarize_precedence_sensitivity(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    denominators: dict[str, int] = defaultdict(int)
    changed_counts: Counter[str] = Counter()
    for row in rows:
        policy = row["policy"]
        counts[policy][row["category"]] += int(row["count_valid"])
        if row["category"] == "None":
            denominators[policy] += int(row["valid_denominator"])
            changed_counts[policy] += int(row["changed_valid_probes"])

    baseline = counts.get("A_corrected_original_precedence", Counter())
    baseline_denominator = denominators.get("A_corrected_original_precedence", 0)
    output = []
    for policy, category_counts in sorted(counts.items()):
        denominator = denominators[policy]
        changed_count = changed_counts[policy]
        for category in (*TAXONOMY, "Unclassified"):
            proportion = category_counts[category] / denominator if denominator else ""
            baseline_proportion = (
                baseline[category] / baseline_denominator
                if baseline_denominator else 0.0
            )
            output.append({
                "policy": policy,
                "taxonomy": category,
                "count_valid": category_counts[category],
                "valid_denominator": denominator,
                "proportion_valid": proportion,
                "difference_vs_corrected_primary": (
                    proportion - baseline_proportion
                    if proportion != "" else ""
                ),
                "changed_valid_probes": changed_count,
                "changed_valid_denominator": denominator,
                "proportion_reclassified": (
                    changed_count / denominator if denominator else ""
                ),
            })
    return output


def summarize_fracture_artifacts(
    source_rows: list[dict[str, str]],
    records: list[dict[str, Any]],
    grid_groups: dict[tuple[str, int, int], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from arcus.analysis.fracture import FracturePointFinder

    source = read_csv(FRACTURE_FILE)
    selected: dict[tuple[str, int, str, int], dict[str, str]] = {}
    for row in source:
        if row["model"] not in PUBLICATION_MODELS:
            continue
        key = (
            row["model"],
            integer(row["tier"]),
            row["condition"],
            integer(row["look_ahead_sampled_points"]),
        )
        if key in selected:
            old = selected[key]
            compare_fields = (
                "threshold", "look_ahead_sampled_points", "fracture_depth",
                "sampled_horizons", "curve", "probes", "valid_probes",
                "afs_fracture_cache_json",
            )
            if any(old[field] != row[field] for field in compare_fields):
                raise ValueError(f"Conflicting duplicate fracture records for {key}")
        else:
            selected[key] = row

    result_rows = []
    fracture_rows = []
    curve_reconciliation = []
    curve_index: dict[tuple[str, int, str, int], dict[int, float]] = {}
    for model in PUBLICATION_MODELS:
        for tier in TIERS:
            valid_key = (model, tier, "valid_only", 2)
            all_key = (model, tier, "all_probes_invalid_as_zero", 2)
            if valid_key not in selected or all_key not in selected:
                raise ValueError(f"Missing stored fracture conditions for {model}, tier {tier}")
            valid_row = selected[valid_key]
            historical_row = selected[all_key]
            historical_curve = {
                int(z): float(acc) for z, acc in json.loads(historical_row["curve"])
            }
            stored_curve_pairs = json.loads(valid_row["curve"])
            stored_curve = {
                int(z): float(acc) for z, acc in stored_curve_pairs
            }
            curve_index[valid_key] = stored_curve
            all_model_tier_rows = [
                row for row in records
                if row["model"] == model
                and row["tier"] == tier
            ]
            valid_observations = [
                row for row in all_model_tier_rows
                if row["valid"]
                and not is_calibration_anchor({
                    "z": str(row["z"]),
                    "tier": str(row["tier"]),
                    "gravity": str(row["gravity"]),
                })
            ]
            valid_anchors = [
                row for row in all_model_tier_rows
                if row["valid"]
                and is_calibration_anchor({
                    "z": str(row["z"]),
                    "tier": str(row["tier"]),
                    "gravity": str(row["gravity"]),
                })
            ]
            observed_curve: dict[int, list[float]] = defaultdict(list)
            for row in valid_observations:
                value = row["original_step_accuracy"]
                if value is not None:
                    observed_curve[row["z"]].append(value)
            recomputed = {
                z: sum(values) / len(values)
                for z, values in observed_curve.items()
            }
            stored_extra_horizons = set(stored_curve) - set(recomputed)
            if (
                set(recomputed) - set(stored_curve)
                or stored_extra_horizons - ({3} if valid_anchors else set())
                or any(
                    not math.isclose(stored_curve[z], recomputed[z], rel_tol=0, abs_tol=1e-10)
                    for z in set(stored_curve) & set(recomputed)
                )
            ):
                raise ValueError(
                    f"Stored valid-only curve disagrees with corrected probe records for {model}, tier {tier}"
                )
            if integer(valid_row["valid_probes"]) != len(valid_observations) + len(valid_anchors):
                raise ValueError(
                    f"Stored valid denominator disagrees for {model}, tier {tier}; "
                    "the difference is not explained by calibration-anchor observations"
                )
            lookahead = integer(valid_row["look_ahead_sampled_points"])
            finder = FracturePointFinder(
                accuracy_threshold=float(valid_row["threshold"]),
                look_ahead_step=lookahead,
            )
            calculated = finder.compute_structural_yield(
                sorted(recomputed.items())
            )
            stored_depth = (
                integer(valid_row["fracture_depth"])
                if valid_row["fracture_depth"] else None
            )
            max_sampled = max(recomputed) if recomputed else None
            terminal_censored = (
                calculated is not None
                and calculated == max_sampled
                and len(recomputed) - list(sorted(recomputed)).index(calculated) - 1 < lookahead
            )
            curve_reconciliation.append({
                "model": model,
                "tier": tier,
                "look_ahead_sampled_points": lookahead,
                "saved_valid_curve_probes": integer(valid_row["valid_probes"]),
                "recomputed_nonanchor_valid_probes": len(valid_observations),
                "valid_calibration_anchor_rows": len(valid_anchors),
                "saved_curve_horizons": len(stored_curve),
                "recomputed_nonanchor_curve_horizons": len(recomputed),
                "saved_extra_horizons": ";".join(map(str, sorted(stored_extra_horizons))),
                "saved_fracture_depth": stored_depth if stored_depth is not None else "",
                "recomputed_nonanchor_fracture_depth": calculated if calculated is not None else "",
                "reconciliation": (
                    "matches_after_removing_calibration_anchor"
                    if stored_extra_horizons else "exact_match"
                ),
            })
            fracture_rows.append({
                "model": model,
                "tier": tier,
                "valid_probes": len(valid_observations),
                "sampled_horizons": len(recomputed),
                "threshold": valid_row["threshold"],
                "look_ahead_sampled_points": lookahead,
                "valid_only_fracture_depth": calculated if calculated is not None else "",
                "fracture_status": (
                    "terminal_candidate_right_censored" if terminal_censored
                    else "candidate" if calculated is not None
                    else "no_fracture_on_sampled_curve"
                ),
                "last_sampled_horizon": max_sampled if max_sampled is not None else "",
                "afs_cache": valid_row["afs_fracture_cache_json"],
            })

            for z in RECONCILIATION_HORIZONS:
                cell = grid_groups.get((model, z, tier), [])
                valid_cell = [row for row in cell if row["valid"]]
                valid_curve_accuracy = recomputed.get(z)
                new_metric_rows = [
                    row for row in valid_cell if row["step_accuracy"] is not None
                ]
                result_rows.append({
                    "model": model,
                    "tier": tier,
                    "horizon_z": z,
                    "observations": len(cell),
                    "valid_observations": len(valid_cell),
                    "invalid_observations": len(cell) - len(valid_cell),
                    "valid_only_stored_step_accuracy_full_transition_metric": (
                        valid_curve_accuracy if valid_curve_accuracy is not None else ""
                    ),
                    "extended_analysis_step_accuracy_z_coordinate_metric": (
                        sum(row["step_accuracy"] for row in new_metric_rows) / len(new_metric_rows)
                        if new_metric_rows else ""
                    ),
                    "historical_fracture_depth": (
                        historical_row["fracture_depth"]
                        if historical_row["fracture_depth"] else ""
                    ),
                    "historical_condition_label": historical_row["condition"],
                    "historical_all_probes_invalid_as_zero_step_accuracy": (
                        historical_curve[z] if z in historical_curve else ""
                    ),
                    "valid_only_fracture_depth": (
                        calculated if calculated is not None else ""
                    ),
                    "valid_only_fracture_status": (
                        "terminal_candidate_right_censored" if terminal_censored
                        else "candidate" if calculated is not None
                        else "no_fracture_on_sampled_curve"
                    ),
                    "stored_valid_only_fracture_depth": (
                        stored_depth if stored_depth is not None else ""
                    ),
                    "valid_only_sampled_horizons": len(recomputed),
                    "afs_cache": valid_row["afs_fracture_cache_json"],
                    "accuracy_convention_note": (
                        "stored curve uses corrected z-transition metric; extended analysis uses z output coordinates"
                    ),
                })
    return fracture_rows, result_rows, curve_reconciliation, {
        "curves": curve_index,
        "stored_rows": selected,
    }


def synthetic_cases() -> list[dict[str, Any]]:
    """Small deterministic deviations for classifier-consistency checks."""
    from arcus.analysis.taxonomy import classify_from_result

    width = height = 8
    z = 7
    actions = ["east"] * z
    rules = {
        "east": {"dx": 1, "dy": 0},
        "west": {"dx": -1, "dy": 0},
    }
    full_reference = [f"[{x},0]" for x in range(z + 1)]
    scoring_reference = full_reference[:z]

    initial = [f"[{(x + 1) % width},0]" for x in range(z)]
    transition = [scoring_reference[0]] + [
        f"[{(z - i) % width},0]" for i in range(1, z)
    ]
    premature = scoring_reference[:4]
    overlong = full_reference[:]
    noncanonical = scoring_reference[:]
    noncanonical[3] = f"[{int(noncanonical[3][1:-1].split(',')[0]) + width},0]"

    irregular_reference = [
        "[1,3]", "[0,3]", "[9,3]", "[9,4]", "[9,3]", "[9,2]",
        "[9,1]", "[8,1]", "[9,1]", "[8,1]", "[8,0]", "[8,7]",
        "[8,0]", "[8,7]", "[8,0]", "[8,1]", "[8,0]", "[7,0]",
    ]
    irregular_actions = [
        "west", "west", "north", "south", "south", "south", "west",
        "east", "west", "south", "south", "north", "south", "north",
        "north", "south", "west",
    ]
    irregular_prediction = [
        "[1,3]", "[10,3]", "[9,3]", "[9,4]", "[9,3]", "[9,2]",
        "[9,1]", "[8,1]", "[9,1]", "[8,1]", "[8,8]", "[8,7]",
        "[8,8]", "[8,7]", "[8,8]", "[8,7]",
    ]
    irregular_rules = {
        "east": {"dx": 1, "dy": 0},
        "north": {"dx": 0, "dy": 1},
        "south": {"dx": 0, "dy": -1},
        "west": {"dx": -1, "dy": 0},
    }
    cases = [
        ("initial_state_corruption", initial, scoring_reference, actions[:z - 1],
         rules, width, height, "Semantic Interpretation Failure",
         "initial coordinate shifted by one"),
        ("transition_rule_error", transition, scoring_reference, actions[:z - 1],
         rules, width, height, "Transition Rule Failure",
         "all requested east transitions reversed"),
        ("premature_termination", premature, scoring_reference, actions[:z - 1],
         rules, width, height, "Horizon Collapse", "output truncated after four states"),
        ("overlong_output", overlong, scoring_reference, actions[:z - 1],
         rules, width, height, "Horizon Collapse", "one extra state beyond requested z"),
        ("noncanonical_coordinates", noncanonical, scoring_reference, actions[:z - 1],
         rules, width, height, "Formatting Failure",
         "one coordinate shifted by a full grid width"),
        ("unmapped_divergence", irregular_prediction, irregular_reference[:17],
         irregular_actions[:16], irregular_rules, 10, 8,
         "State Tracking Failure",
         "irregular noncanonical divergence with later drift and truncation"),
    ]
    results = []
    for name, predicted, expected, case_actions, case_rules, gw, gh, expected_label, injected in cases:
        result = classify_from_result(
            predicted, expected, case_rules, case_actions, 0, gw, gh
        )
        actual = result.legacy_mode.value
        fallback = result.explanation == "State synchronization lost after early divergence."
        matches = result.valid and actual == expected_label
        if name == "unmapped_divergence":
            matches = matches and fallback
        results.append({
            "synthetic_case": name,
            "injected_deviation": injected,
            "valid": result.valid,
            "expected_primary_taxonomy": expected_label,
            "actual_primary_taxonomy": actual,
            "state_tracking_fallback": fallback,
            "classifier_consistency": matches,
            "explanation": result.explanation or "",
            "interpretation_scope": (
                "deterministic classifier consistency only; not human semantic validation"
            ),
        })
    return results


def write_summary(
    raw_manifest: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    grid_summary: list[dict[str, Any]],
    reconciliation: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
    synthetic: list[dict[str, Any]],
    precedence_summary: list[dict[str, Any]],
    fracture_reconciliation: list[dict[str, Any]],
    curve_reconciliation: list[dict[str, Any]],
) -> None:
    total_cells = len(PUBLICATION_MODELS) * len(GRID) * len(TIERS)
    observed_cells = sum(row["coverage_status"] != "missing" for row in coverage)
    valid_total = sum(int(row["valid_observations"]) for row in coverage)
    invalid_total = sum(int(row["invalid_observations"]) for row in coverage)
    missing_horizons = [
        row["horizon_z"] for row in grid_summary
        if row["observations"] == 0
    ]
    complete_horizons = [
        row["horizon_z"] for row in grid_summary
        if row["model_tier_cells_observed"] == row["model_tier_cells_expected"]
    ]
    fallback_total = sum(int(row["fallback_state_tracking_assignments"]) for row in fallback)
    state_total = sum(int(row["state_tracking_assignments"]) for row in fallback)
    synthetic_pass = sum(bool(row["classifier_consistency"]) for row in synthetic)

    lines = [
        "ARCUS-X reviewer-response analyses",
        "Source: preserved raw logs and existing corrected audit artifacts; no model inference or API calls.",
        f"Publication cohort: {len(PUBLICATION_MODELS)} models.",
        f"Requested horizon grid: {', '.join(map(str, GRID))}.",
        (
            f"Observed model-tier-horizon cells: {observed_cells}/{total_cells}; "
            f"missing cells: {total_cells - observed_cells}."
        ),
        f"Grid observations: {valid_total + invalid_total}; valid: {valid_total}; invalid: {invalid_total}.",
        (
            "Horizons with no observations across the cohort: "
            + (", ".join(map(str, missing_horizons)) if missing_horizons else "none")
            + "."
        ),
        (
            "Horizons with all model-tier cells observed: "
            + (", ".join(map(str, complete_horizons)) if complete_horizons else "none")
            + "."
        ),
        (
            "Step Accuracy convention: score the first z coordinates of the complete z+1-state "
            "reference, exclude the supplied initial state, and average valid responses only."
        ),
        (
            "Uncertainty: percentile 95% cluster-bootstrap intervals, resampling master-seed "
            f"blocks ({BOOTSTRAP_REPLICATES} replicates) and retaining repeated task/gravity rows."
        ),
        (
            "Tier-contrast bootstrap uses the union of seed blocks when tiers share seeds, "
            "including blocks observed in only one tier; tiers without shared seeds are resampled separately."
        ),
        (
            "Resume: rerun reviewer_response_analysis.py; completed raw-file and bootstrap units "
            "are reused only when input and code fingerprints match. "
            "Use --no-resume for a fresh run or --checkpoint PATH to select a database."
        ),
        (
            f"State Tracking assignments: {state_total}; explicit-condition assignments: "
            f"{state_total - fallback_total}; fallback assignments: {fallback_total}."
        ),
        (
            f"Synthetic classifier checks: {synthetic_pass}/{len(synthetic)} matched their "
            "deterministic expected observable signature; this is not human validation."
        ),
        (
            "Valid None / full canonical Exact Match reconciliations passed for all "
            f"{len(reconciliation)} models."
        ),
        "Raw source checksums and row-alignment counts:",
    ]
    lines.extend(
        f"  {item['raw_file']}: {item['corrected_rows_verified']} rows; SHA-256 {item['sha256']}"
        for item in raw_manifest
    )
    sensitivity_policies = sorted({
        row["policy"] for row in precedence_summary
        if row["policy"] != "A_corrected_original_precedence"
    })
    lines.append("Existing precedence sensitivity (pooled valid observations):")
    for policy in sensitivity_policies:
        policy_rows = [
            row for row in precedence_summary if row["policy"] == policy
        ]
        changed = policy_rows[0]
        shifted = sorted(
            (
                (abs(float(row["difference_vs_corrected_primary"])), row["taxonomy"],
                 float(row["difference_vs_corrected_primary"]))
                for row in policy_rows
            ),
            reverse=True,
        )
        max_shift = shifted[0] if shifted else (0.0, "", 0.0)
        lines.append(
            f"  {policy}: {100 * float(changed['proportion_reclassified']):.3f}% "
            f"of valid classifications changed; largest pooled category-share shift "
            f"{100 * max_shift[2]:+.3f} percentage points ({max_shift[1]})."
        )
    mai_rows = [
        row for row in fracture_reconciliation
        if row["model"] == "MAI-Thinking-1" and row["horizon_z"] in (20, 30)
    ]
    lines.append("MAI-Thinking-1 valid-only stored curve at z=20/30:")
    for row in mai_rows:
        lines.append(
            "  Tier {tier}, z={horizon_z}: {valid_observations} valid / "
            "{invalid_observations} invalid; stored full-transition Step Accuracy "
            "{valid_only_stored_step_accuracy_full_transition_metric}; "
            "requested-z Step Accuracy "
            "{extended_analysis_step_accuracy_z_coordinate_metric}; "
            "valid-only fracture {valid_only_fracture_depth} ({valid_only_fracture_status}).".format(**row)
        )
    mai_legacy_tier0 = {
        row["horizon_z"]: row for row in fracture_reconciliation
        if row["model"] == "MAI-Thinking-1" and row["tier"] == 0
        and row["horizon_z"] in RECONCILIATION_HORIZONS
    }
    mai_early_cells = [
        row for row in fracture_reconciliation
        if row["model"] == "MAI-Thinking-1"
        and row["horizon_z"] in (1, 2, 5, 10)
    ]
    if len(mai_early_cells) == 16 and all(
        row["valid_observations"] == 0 and row["invalid_observations"] == 45
        for row in mai_early_cells
    ):
        lines.append(
            "Across all four MAI-Thinking-1 tiers, each of z=1,2,5,10 has "
            "0/45 valid responses; the stored invalid-as-zero curve assigns "
            "0.000 at these points. At z=1, the next two sampled points (z=2,5) "
            "remain below the 0.5 floor, producing the legacy depth-1 decision."
        )
    if all(z in mai_legacy_tier0 for z in (1, 2, 5, 20, 30)):
        lines.extend([
            (
                "MAI-Thinking-1 Tier 0 trace: the stored all_probes_invalid_as_zero "
                "curve is 0.000 at z=1,2,5, then remains below the 0.5 floor at "
                f"z=20 ({float(mai_legacy_tier0[20]['historical_all_probes_invalid_as_zero_step_accuracy']):.3f}); "
                "the two-point look-ahead therefore returns z=1."
            ),
            (
                "At z=1, the raw tier-0 batch has "
                f"{mai_legacy_tier0[1]['valid_observations']} valid and "
                f"{mai_legacy_tier0[1]['invalid_observations']} invalid responses. "
                "On the corrected valid-only curve, z=20 has "
                f"{mai_legacy_tier0[20]['valid_observations']}/"
                f"{mai_legacy_tier0[20]['observations']} valid responses with "
                f"conditional accuracy "
                f"{float(mai_legacy_tier0[20]['valid_only_stored_step_accuracy_full_transition_metric']):.3f}, "
                "and z=30 is "
                f"{float(mai_legacy_tier0[30]['valid_only_stored_step_accuracy_full_transition_metric']):.3f}; "
                "the valid-only tier-0 curve has no sampled fracture."
            ),
        ])
    lines.extend([
        (
            "The saved valid-only fracture curves include calibration-anchor rows at "
            "z=3 for some conditions; the publication-cohort fracture table is "
            "recomputed after removing those anchors and matches the stored curves "
            "at all remaining observed horizons."
        ),
        (
            "The historical MAI-Thinking-1 depth-1 value appears only in the stored "
            "all_probes_invalid_as_zero condition. The corrected valid-only curve "
            "does not score invalid-only horizons as zero."
        ),
        "",
        "Interpretation limits: horizons absent from the source remain missing; the observed AFS "
        "support is not a balanced fixed-grid experiment. Horizon matching alone is not task pairing.",
        "Invalid-only cells have blank accuracy/taxonomy estimates; invalid responses are never scored as zero.",
    ])
    (OUT / "reviewer_response_analysis_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    global CHECKPOINTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-resume", action="store_true",
        help="recompute without reading or writing checkpoints",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=OUT / ".reviewer_response_cache" / "checkpoints.sqlite3",
        help="checkpoint database path (default: audit correction cache)",
    )
    args = parser.parse_args()
    start = time.monotonic()
    progress("[1/5] Loading corrected probe data")
    probe_rows = read_csv(PROBE_FILE)
    cohort_rows = [
        row for row in probe_rows if row["model"] in PUBLICATION_MODELS
    ]
    if {row["model"] for row in cohort_rows} != set(PUBLICATION_MODELS):
        raise ValueError("Corrected probe artifact does not contain the full 13-model cohort")

    if not args.no_resume:
        progress("[2/5] Fingerprinting inputs and opening resume checkpoints")
        namespace = checkpoint_namespace(sorted({row["source_file"] for row in cohort_rows}))
        CHECKPOINTS = CheckpointStore(args.checkpoint, namespace)
    else:
        progress("[2/5] Resume disabled; recomputing from raw artifacts")
    try:
        return run_analysis(cohort_rows, start)
    finally:
        if CHECKPOINTS is not None:
            CHECKPOINTS.close()
            CHECKPOINTS = None


def run_analysis(cohort_rows: list[dict[str, str]], start: float) -> int:
    progress("[3/5] Verifying preserved raw observations")
    records, raw_manifest = load_and_verify_raw(cohort_rows)
    progress("[4/5] Computing coverage, seed-block uncertainty and tier comparisons")
    coverage, metrics, taxonomy, grid_summary, groups = coverage_and_metrics(records)
    selected_profiles = selected_horizon_profiles(records)
    write_selected_horizon_plot(selected_profiles)
    comparisons = tier_comparisons(groups)
    cohort_tiers = cohort_tier_summary(records)
    progress("[5/5] Reconciling taxonomy and fracture results, then writing outputs")
    cross_tab, fallback, reconciliation = taxonomy_reconciliation(records)
    write_taxonomy_profile_figure(cross_tab)
    verify_state_tracking_breakdown(fallback)
    sensitivity = existing_precedence_sensitivity()
    verify_precedence_baseline(cross_tab, sensitivity)
    precedence_summary = summarize_precedence_sensitivity(sensitivity)
    synthetic = synthetic_cases()
    if not all(row["classifier_consistency"] for row in synthetic):
        raise AssertionError("A synthetic classifier consistency check failed")

    (
        fracture_model_tier,
        fracture_reconciliation,
        fracture_curve_reconciliation,
        _fracture_data,
    ) = summarize_fracture_artifacts(
        cohort_rows, records, groups
    )
    write_csv(OUT / "reviewer_response_horizon_coverage.csv", coverage)
    write_csv(OUT / "reviewer_response_horizon_metrics.csv", metrics)
    write_csv(OUT / "reviewer_response_horizon_taxonomy.csv", taxonomy)
    write_csv(OUT / "reviewer_response_tier_comparisons.csv", comparisons)
    write_csv(OUT / "reviewer_response_matched_horizon_tiers.csv", cohort_tiers)
    write_csv(OUT / "reviewer_response_grid_summary.csv", grid_summary)
    write_csv(
        OUT / "reviewer_response_selected_horizon_profiles.csv",
        selected_profiles,
    )
    write_csv(OUT / "reviewer_response_probe_level.csv", [
        {
            **{
                key: row[key]
                for key in (
                    "model", "source_file", "source_row", "seed", "task_hash",
                    "z", "tier", "gravity", "valid", "taxonomy",
                    "exact_match_full_reference", "requested_length_compliant",
                    "exact_match_requested_output", "step_accuracy",
                    "early_accuracy", "middle_accuracy", "late_accuracy",
                    "full_reference_states", "scored_reference_states",
                    "scored_transitions",
                )
            },
            "predicted_path": json.dumps(row["predicted_path"], separators=(",", ":")),
            "ground_truth_path_full": json.dumps(
                row["ground_truth_path_full"], separators=(",", ":")
            ),
        }
        for row in records if row["z"] in GRID
    ])
    write_csv(OUT / "reviewer_response_taxonomy_reconciliation.csv", cross_tab)
    write_csv(OUT / "reviewer_response_model_reconciliation.csv", reconciliation)
    write_csv(OUT / "reviewer_response_state_tracking_fallback.csv", fallback)
    write_csv(OUT / "reviewer_response_precedence_sensitivity.csv", sensitivity)
    write_csv(
        OUT / "reviewer_response_precedence_policy_summary.csv",
        precedence_summary,
    )
    write_csv(OUT / "reviewer_response_synthetic_checks.csv", synthetic)
    write_csv(OUT / "reviewer_response_fracture_model_tier.csv", fracture_model_tier)
    write_csv(OUT / "reviewer_response_afs_reconciliation.csv", fracture_reconciliation)
    write_csv(
        OUT / "reviewer_response_fracture_curve_reconciliation.csv",
        fracture_curve_reconciliation,
    )
    write_summary(
        raw_manifest, coverage, grid_summary, reconciliation,
        fallback, synthetic,
        precedence_summary, fracture_reconciliation, fracture_curve_reconciliation,
    )
    print(
        f"Verified {sum(row['corrected_rows_verified'] for row in raw_manifest):,} "
        f"raw entries across {len(raw_manifest)} publication-model logs; wrote "
        f"{len(coverage)} coverage cells and reviewer-response audit tables "
        f"in {time.monotonic() - start:.1f} seconds."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
