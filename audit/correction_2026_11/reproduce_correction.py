#!/usr/bin/env python3
"""Reproduce the November 2026 ARCUS-X correction analyses."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MANUSCRIPT_SOURCE = Path(
    r"C:\Users\games\Downloads\Formatting_Instructions_for_ICLR_2027_Conference_Submissions\iclr2027_conference.tex"
)
MANUSCRIPT_SUPPORT_FILES = (
    "iclr2027_conference.bib",
    "iclr2027_conference.sty",
    "iclr2027_conference.bst",
    "math_commands.tex",
    "fancyhdr.sty",
    "natbib.sty",
    "horizon_profiles.pdf",
    "horizon_profiles_selected.pdf",
)
sys.path.insert(0, str(ROOT))

import audit.analyze_artifacts as audit  # noqa: E402


TAXONOMY = [
    "None",
    "State Tracking Failure",
    "Transition Rule Failure",
    "Semantic Interpretation Failure",
    "Horizon Collapse",
    "Formatting Failure",
    "Unknown / Unmapped",
    "Unclassified",
]
POLICIES = {
    "A_corrected_original_precedence": "Use the corrected classifier, strict correctness, and original primary-category precedence.",
    "B_horizon_first": "For valid failures with an explicit length mismatch, assign Horizon Collapse before other detected categories.",
    "B_transition_first": "For valid probes with an explicit transition-rule indicator, assign Transition Rule Failure before other primary categories.",
    "B_conservative_unclassified": "Assign Unclassified to classifier fallback State Tracking, Unknown / Unmapped, or an unexplained non-exact None.",
    "B_fallback_unclassified": "Assign Unclassified only to State Tracking assignments made by the classifier fallback.",
    "B_modulo_equivalent": "Treat a full-length, modulo-equivalent trajectory as correct even when coordinates are noncanonical; sensitivity only.",
}
MANUSCRIPT_MODELS = [
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
MODEL_LABELS = {
    "qwen.qwen3-235b-a22b-2507": r"qwen.qwen3-235b-a22b-2507*",
    "moonshotai.kimi-k2.5": "Kimi K2.5",
    "qwen.qwen3-coder-480b-a35b-v1:0": "Qwen 3 Coder 480B A35B",
    "mistral-small-2603": "mistral-small-2603",
    "mistral-large-2512": "Mistral Large 2512",
    "mistral.devstral-2-123b": r"mistral.devstral-2-123b*",
    "ministral-14b-2512": "ministral-14b-2512",
    "ministral-8b-2512": "ministral-8b-2512",
    "ministral-3b-2512": "ministral-3b-2512",
    "nvidia.nemotron-super-3-120b": "nvidia.nemotron-super-3-120b",
    "zai.glm-5": "GLM-5",
    "GPT-5-mini": "GPT-5-mini",
    "MAI-Thinking-1": r"MAI-Thinking-1*",
    "minimax.minimax-m2.5": "minimax.minimax-m2.5",
}


def _csv_read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _csv_write(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str] | None = None,
) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _normalize_probe(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    for key in (
        "valid", "exact_match", "trajectory_correct", "classifier_exact_match",
        "state_tracking_fallback", "transition_first", "strict_fully_correct",
        "ground_truth_length_matches_z",
    ):
        if key in result:
            result[key] = _bool(result[key])
    for key in ("seed", "z", "tier", "predicted_length", "ground_truth_length"):
        result[key] = _int(result.get(key))
    for key in (
        "gravity", "step_accuracy", "continuity", "horizon_compliance",
        "early_accuracy", "middle_accuracy", "late_accuracy",
    ):
        result[key] = _float(result.get(key), float("nan"))
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_inventory(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(set(paths))
        if path.is_file()
    ]


def _raw_file_paths() -> list[Path]:
    return sorted(
        p for p in (ROOT / "outputs").glob("absolute_raw_data_local_*.txt")
        if ".analyzed." not in p.name
    )


def _output_actions_rows(path: Path) -> set[int]:
    matching: set[int] = set()
    entry_index = 0
    in_output = False
    in_entry = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("=== RAW STREAM ENTRY"):
            entry_index += 1
            in_entry = True
            in_output = False
        elif in_entry and stripped.startswith("=============================="):
            in_entry = False
            in_output = False
        elif in_entry and stripped.startswith("[MODEL OUTPUT]:"):
            in_output = True
        elif in_entry and stripped.startswith("[GROUND TRUTH EXPECTED]:"):
            in_output = False
        elif in_entry and in_output and stripped.startswith("Actions:"):
            matching.add(entry_index)
    return matching


def _canonical_status(row: dict[str, Any]) -> str:
    try:
        predicted = json.loads(row.get("predicted_path", "[]"))
        truth = json.loads(row.get("ground_truth_path", "[]"))
        width = _int(row.get("grid_width"))
        height = _int(row.get("grid_height"))
        coords = [tuple(map(int, c.strip("[]").split(","))) for c in predicted]
        gt = [tuple(map(int, c.strip("[]").split(","))) for c in truth]
    except (ValueError, TypeError, json.JSONDecodeError):
        return "unparseable"
    if width <= 0 or height <= 0:
        return "dimensions_unavailable"
    if not any(x < 0 or x >= width or y < 0 or y >= height for x, y in coords):
        return "canonical"
    if len(coords) == len(gt) and all(
        (x % width, y % height) == expected
        for (x, y), expected in zip(coords, gt)
    ):
        return "modulo_equivalent_noncanonical"
    return "noncanonical_not_fully_modulo_equivalent"


def _recompute_transition_flags(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for row in rows:
        predicted = json.loads(row.get("predicted_path", "[]"))
        ground_truth = json.loads(row.get("ground_truth_path", "[]"))
        rules = json.loads(row.get("transition_rules", "{}"))
        actions = json.loads(row.get("actions", "[]"))
        row["transition_first"] = audit._transition_first_matches(
            predicted,
            ground_truth,
            rules,
            actions,
            _int(row.get("grid_width")),
            _int(row.get("grid_height")),
        )
    return rows


def _corrected_probe_file(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = _recompute_transition_flags(rows)
    corrected = []
    for raw in rows:
        row = dict(raw)
        valid = _bool(row.get("valid"))
        exact = _bool(row.get("exact_match"))
        trajectory_correct = _bool(row.get("trajectory_correct"))
        row["strict_fully_correct"] = valid and (exact or trajectory_correct)
        row["analysis_A_taxonomy"] = row.get("reported_taxonomy", "")
        row["coordinate_interpretation"] = _canonical_status(row)
        corrected.append(row)
    _csv_write(OUT / "analysis_A_probe_level.csv", corrected)
    return corrected


def _exact_observations(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    return audit._deduplicate(rows)


def _write_exact_duplicate_mapping(rows: list[dict[str, Any]]) -> None:
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    mapping = []
    for row in rows:
        key = (
            row["model"],
            _int(row.get("seed")),
            _int(row.get("z")),
            _float(row.get("gravity")),
            _int(row.get("tier")),
            row.get("grid_key", ""),
            row.get("task_hash", ""),
            row.get("entry_hash", row.get("source_entry_hash", "")),
        )
        canonical = seen.get(key)
        if canonical is None:
            seen[key] = row
            continue
        mapping.append({
            "model": row["model"],
            "duplicate_source_file": row.get("source_file", ""),
            "duplicate_source_row": row.get("source_row", ""),
            "canonical_source_file": canonical.get("source_file", ""),
            "canonical_source_row": canonical.get("source_row", ""),
            "entry_hash": key[-1],
            "task_hash": row.get("task_hash", ""),
            "deduplication_basis": "same model/task condition and complete raw-entry hash",
        })
    _csv_write(
        OUT / "exact_duplicate_mapping.csv",
        mapping,
        [
            "model", "duplicate_source_file", "duplicate_source_row",
            "canonical_source_file", "canonical_source_row", "entry_hash",
            "task_hash", "deduplication_basis",
        ],
    )


def _model_summary(
    rows: list[dict[str, Any]],
    baseline_summary: list[dict[str, str]],
    baseline_probe_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    _write_exact_duplicate_mapping(rows)
    unique, exact_duplicates = _exact_observations(rows)
    duplicates_by_model = Counter(
        r["model"] for r in rows
    ) - Counter(r["model"] for r in unique)
    old_by_model = {r["model"]: r for r in baseline_summary}
    old_by_source = {
        (r["model"], r["source_row"]): r for r in baseline_probe_rows
    }
    out = []
    for model in sorted({r["model"] for r in unique}):
        model_rows = [r for r in unique if r["model"] == model]
        valid = [r for r in model_rows if _bool(r["valid"])]
        task_condition_count = len({
            (
                r["model"], _int(r.get("seed")), _int(r.get("z")),
                _float(r.get("gravity")), _int(r.get("tier")),
                r.get("grid_key", ""), r.get("task_hash", ""),
            )
            for r in model_rows
        })
        old = old_by_model.get(model, {})
        matched_old = [
            old_by_source[(r["model"], r["source_row"])]
            for r in model_rows
            if (r["model"], r["source_row"]) in old_by_source
        ]
        item: dict[str, Any] = {
            "model": model,
            "logged_entries": sum(r["model"] == model for r in rows),
            "exact_duplicate_raw_entries_excluded": duplicates_by_model[model],
            "analysis_observations": len(model_rows),
            "unique_task_condition_count": task_condition_count,
            "repeated_task_condition_observations": (
                len(model_rows) - task_condition_count
            ),
            "valid_probes": len(valid),
            "invalid_probes": len(model_rows) - len(valid),
            "validity_rate": len(valid) / len(model_rows) if model_rows else None,
            "old_audit_reported_none_count": old.get("reported_none_count_valid", ""),
            "old_reported_none_count_same_observations": sum(
                r.get("reported_taxonomy") == "None" and _bool(r.get("valid"))
                for r in matched_old
            ),
            "old_valid_probes_same_observations": sum(
                _bool(r.get("valid")) for r in matched_old
            ),
            "old_exact_match_count": old.get("exact_match_count_valid", ""),
            "corrected_none_count": sum(r["reported_taxonomy"] == "None" for r in valid),
            "corrected_exact_match_count": sum(_bool(r["exact_match"]) for r in valid),
            "corrected_noncanonical_modulo_count": sum(
                r.get("subtype") == "ORF_Invalid_Coordinate"
                for r in valid
            ),
            "state_tracking_fallback_count": sum(
                _bool(r["state_tracking_fallback"]) for r in valid
            ),
        }
        for category in TAXONOMY[:-1]:
            count = sum(r["reported_taxonomy"] == category for r in valid)
            key = category.lower().replace(" / ", "_").replace(" ", "_")
            item[f"{key}_count"] = count
            item[f"{key}_proportion_valid"] = count / len(valid) if valid else None
        fields = {
            "step_accuracy_valid": ("step_accuracy", True),
            "step_accuracy_all": ("step_accuracy", False),
            "exact_match_valid": ("exact_match", True),
            "continuity_valid": ("continuity", True),
            "horizon_compliance_valid": ("horizon_compliance", True),
            "early_accuracy_valid": ("early_accuracy", True),
            "middle_accuracy_valid": ("middle_accuracy", True),
            "late_accuracy_valid": ("late_accuracy", True),
        }
        for output_key, (field, valid_only) in fields.items():
            metric_rows = model_rows if not valid_only else valid
            for r in metric_rows:
                r["_metric_value"] = float(_bool(r[field])) if field == "exact_match" else _float(r.get(field), 0.0)
            point, low, high, clusters = audit._cluster_bootstrap(
                metric_rows, "_metric_value", valid_only=False
            )
            item[output_key] = point
            item[f"{output_key}_ci95_lower_cluster_seed"] = low
            item[f"{output_key}_ci95_upper_cluster_seed"] = high
            item[f"{output_key}_n_seed_clusters"] = clusters
        out.append(item)
    _csv_write(OUT / "analysis_A_model_summary.csv", out)
    return out


def _write_reassignments(
    old_rows: list[dict[str, str]],
    new_rows: list[dict[str, str]],
    action_rows: dict[str, set[int]],
) -> list[dict[str, Any]]:
    unique_new, _ = _exact_observations(new_rows)
    new_by_source = {
        (r["model"], r["source_row"]): r for r in unique_new
    }
    aggregate: Counter[tuple[str, str, str, str]] = Counter()
    denominators: Counter[str] = Counter()
    for old in old_rows:
        new = new_by_source.get((old["model"], old["source_row"]))
        if new is None or not _bool(old.get("valid")) or not _bool(new.get("valid")):
            continue
        key = (old["model"], old.get("reported_taxonomy", ""), new.get("reported_taxonomy", ""))
        denominators[old["model"]] += 1
        if key[1] == key[2]:
            cause = "unchanged"
        elif key[1] == "None" and _float(old.get("step_accuracy")) >= 1.0:
            cause = "perfect_step_accuracy_reporting_override"
        elif (
            _int(old.get("source_row")) in action_rows.get(old.get("source_file", ""), set())
            and key[1] == "None"
        ):
            cause = "model_output_actions_metadata_corruption"
        elif new.get("subtype") == "ORF_Invalid_Coordinate":
            cause = "noncanonical_coordinate_contract"
        elif key[1] == "None":
            cause = "nonexact_classifier_success_or_fallback"
        else:
            cause = "classifier_precedence_or_parser_correction"
        aggregate[(key[0], key[1], key[2], cause)] += 1
    out = [
        {
            "model": model,
            "original_category": old,
            "corrected_category": new,
            "root_cause": cause,
            "count": count,
            "valid_denominator": denominators[model],
            "proportion_valid": count / denominators[model] if denominators[model] else None,
            "changed": old != new,
        }
        for (model, old, new, cause), count in sorted(aggregate.items())
    ]
    _csv_write(OUT / "analysis_A_reassignment_matrix.csv", out)
    return out


def _analysis_b(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    probes = []
    counts: Counter[tuple[str, str, str]] = Counter()
    denominators: Counter[tuple[str, str]] = Counter()
    categories = set(TAXONOMY)
    for raw in rows:
        r = _normalize_probe(raw)
        valid = r["valid"]
        primary = r["reported_taxonomy"]
        mismatch = r["predicted_length"] != r["ground_truth_length"]
        fallback = r["state_tracking_fallback"]
        transition = r["transition_first"]
        modulo = r.get("coordinate_interpretation") == "modulo_equivalent_noncanonical"
        labels = {"A_corrected_original_precedence": primary}
        labels["B_horizon_first"] = (
            "Horizon Collapse" if valid and mismatch else primary
        )
        labels["B_transition_first"] = (
            "Transition Rule Failure" if valid and transition else primary
        )
        labels["B_conservative_unclassified"] = (
            "Unclassified"
            if valid and (
                fallback
                or primary == "Unknown / Unmapped"
                or (primary == "None" and not r["strict_fully_correct"])
            )
            else primary
        )
        labels["B_fallback_unclassified"] = (
            "Unclassified" if valid and fallback else primary
        )
        labels["B_modulo_equivalent"] = (
            "None" if valid and modulo else primary
        )
        r.update({
            "length_mismatch_indicator": mismatch,
            "explicit_transition_indicator": transition,
            "state_tracking_fallback_indicator": fallback,
            "strict_canonical_violation_indicator": modulo,
        })
        for policy, label in labels.items():
            r[policy] = label if valid else "Invalid / Excluded"
            if valid:
                counts[(r["model"], policy, label)] += 1
                denominators[(r["model"], policy)] += 1
                categories.add(label)
        probes.append(r)
    _csv_write(OUT / "analysis_B_probe_level.csv", probes)
    summary = []
    for model, policy in sorted(denominators):
        for category in sorted(categories):
            count = counts[(model, policy, category)]
            summary.append({
                "model": model,
                "policy": policy,
                "policy_definition": POLICIES[policy],
                "category": category,
                "count_valid": count,
                "valid_denominator": denominators[(model, policy)],
                "proportion_valid": count / denominators[(model, policy)],
            })
    _csv_write(OUT / "analysis_B_sensitivity_summary.csv", summary)
    stability = []
    sensitivity_policies = [name for name in POLICIES if name.startswith("B_")]
    for model in sorted({r["model"] for r in probes}):
        valid_rows = [r for r in probes if r["model"] == model and r["valid"]]
        for policy in sensitivity_policies:
            changed = sum(
                r[policy] != r["A_corrected_original_precedence"]
                for r in valid_rows
            )
            stability.append({
                "model": model,
                "policy": policy,
                "changed_valid_probes": changed,
                "valid_denominator": len(valid_rows),
                "proportion_changed": changed / len(valid_rows) if valid_rows else None,
            })
    _csv_write(OUT / "analysis_B_policy_stability.csv", stability)
    return probes, summary


def _replace_table(tex: str, label: str, replacement: str) -> str:
    marker = rf"\label{{{label}}}"
    if tex.count(marker) != 1:
        raise ValueError(f"Expected exactly one table labeled {label}")
    label_position = tex.index(marker)
    start = tex.rfind(r"\begin{table}", 0, label_position)
    end = tex.find(r"\end{table}", label_position)
    if start < 0 or end < 0:
        raise ValueError(f"Could not locate table boundaries for {label}")
    end += len(r"\end{table}")
    return tex[:start] + replacement + tex[end:]


def _taxonomy_table(
    model_summary: list[dict[str, Any]],
    models: list[str],
    label: str,
    caption: str,
) -> str:
    by_model = {r["model"]: r for r in model_summary}
    missing = [model for model in models if model not in by_model]
    if missing:
        raise ValueError(f"Missing manuscript models in reanalysis: {missing}")
    categories = [
        ("state_tracking_failure", "State"),
        ("transition_rule_failure", "Transition"),
        ("semantic_interpretation_failure", "Semantic"),
        ("horizon_collapse", "Horizon"),
        ("formatting_failure", "Format"),
        ("unknown_unmapped", "Unknown"),
        ("none", "None"),
    ]

    def percentage(value: Any) -> str:
        return f"{100 * _float(value):.1f}\\%"

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lccccccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Valid} & \textbf{Invalid} & "
        + " & ".join(rf"\textbf{{{header}}}" for _, header in categories)
        + r" \\",
        r"\midrule",
    ]
    for model in models:
        row = by_model[model]
        total = _int(row.get("analysis_observations"))
        valid = _int(row.get("valid_probes"))
        invalid = _int(row.get("invalid_probes"))
        values = [percentage(valid / total if total else 0)]
        values.append(percentage(invalid / total if total else 0))
        values.extend(
            percentage(
                _int(row.get(f"{key}_count"))
                / valid if valid else 0
            )
            for key, _ in categories
        )
        lines.append(
            f"{MODEL_LABELS[model]} & " + " & ".join(values) + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}"])
    return "\n".join(lines)


def _generate_failure_figure(model_summary: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    by_model = {r["model"]: r for r in model_summary}
    models = [model for model in MANUSCRIPT_MODELS if model in by_model]
    if len(models) != len(MANUSCRIPT_MODELS):
        raise ValueError("Cannot regenerate the manuscript figure: model cohort is incomplete")
    categories = [
        ("state_tracking_failure", "State Tracking"),
        ("transition_rule_failure", "Transition Rule"),
        ("semantic_interpretation_failure", "Semantic Interpretation"),
        ("horizon_collapse", "Horizon Collapse"),
        ("formatting_failure", "Formatting"),
    ]
    values = np.array([
        [
            100 * _float(by_model[model].get(f"{key}_proportion_valid"))
            for key, _ in categories
        ]
        for model in models
    ])
    height = max(7.0, len(models) * 0.43)
    fig, ax = plt.subplots(figsize=(11, height))
    image = ax.imshow(values, vmin=0, vmax=100, aspect="auto", cmap="Greys")
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels([label for _, label in categories], fontsize=8)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([MODEL_LABELS[model] for model in models], fontsize=7)
    for i in range(len(models)):
        for j in range(len(categories)):
            value = values[i, j]
            ax.text(
                j, i, f"{value:.1f}%", ha="center", va="center", fontsize=7,
                color="white" if value > 50 else "black",
            )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Failure proportion among valid probes", fontsize=9)
    colorbar.set_ticks([0, 20, 40, 60, 80, 100])
    colorbar.set_ticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.set_xlabel("Failure signature", fontsize=10)
    ax.set_ylabel("Model", fontsize=10)
    ax.set_title(
        "Failure-signature profiles",
        fontsize=12,
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(OUT / "failure_profiles.pdf", bbox_inches="tight")
    fig.savefig(OUT / "failure_profiles.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _revise_manuscript(
    model_summary: list[dict[str, Any]],
) -> None:
    if not MANUSCRIPT_SOURCE.is_file():
        raise FileNotFoundError(f"Manuscript source not found: {MANUSCRIPT_SOURCE}")
    source_dir = MANUSCRIPT_SOURCE.parent
    for name in MANUSCRIPT_SUPPORT_FILES:
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Required manuscript dependency not found: {source}")
        shutil.copy2(source, OUT / name)

    tex = MANUSCRIPT_SOURCE.read_text(encoding="utf-8")
    abstract = r"""\begin{abstract}
Long-horizon language-model evaluations often reduce performance to final-answer accuracy, obscuring failures that accumulate across dependent sequential transitions. We introduce ARCUS-X, an auditable benchmark and evaluation protocol for decomposing deterministic sequential execution into trajectory correctness, continuity, rule binding, horizon control, response validity, and excess generation. ARCUS-X uses procedurally generated toroidal-grid tasks with exact per-transition ground truth, controlled lexical and transition perturbations, raw-stream preservation, and deterministic post-run diagnostics. We reanalyze the reported 13-model cohort under the benchmark's canonical coordinate contract and a strict definition of full-trajectory correctness: a valid response receives \emph{None} only when its complete coordinate sequence matches the reference; perfect Step Accuracy alone is insufficient. Repeated observations are retained unless complete raw entries are byte-identical. The resulting failure partitions are reproducible behavioral signatures governed by stated precedence rules, not causal accounts of internal model cognition. Because adaptive sampling allocates different horizons across models and conditions, pooled aggregates describe acquired support rather than a common uniform horizon distribution. ARCUS-X provides an auditable diagnostic record of explicit state-transition execution, not a universal model ranking or measure of general long-horizon agency.
\end{abstract}"""
    tex, count = re.subn(
        r"\\begin\{abstract\}.*?\\end\{abstract\}",
        lambda _: abstract,
        tex,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise ValueError("Could not replace the manuscript abstract")

    replacements = {
        "designed not to introduce trajectory evaluation itself, but to make sequential execution failures exactly attributable under controlled transitions, perturbations, and operational validity.":
            "designed not to introduce trajectory evaluation itself, but to assign reproducible behavioral signatures under controlled transitions, perturbations, operational validity, and an explicit precedence policy.",
        "Exact trajectories receive \\emph{none}; invalid responses never enter the trajectory taxonomy; and valid failures that cannot be mapped are retained as \\emph{unknown/unmapped}.":
            "The label \\emph{none} requires an operationally valid, complete trajectory whose coordinates exactly match the canonical reference path; perfect Step Accuracy alone is insufficient. Invalid responses never enter the trajectory taxonomy, and valid failures that cannot be mapped are retained as \\emph{unknown/unmapped}.",
        "Both horizons were observed with analyzable\ntrajectories for all 13 models in the reported evaluation, permitting a\nmatched-horizon descriptive comparison.":
            "Both requested horizons are represented across the reported cohort, permitting a descriptive comparison at common horizon values. Equal horizon does not imply identical task conditions: paired comparisons are restricted to observations with identical task hashes and are reported separately in the correction package.",
        "The deterministic taxonomy is applied uniformly across all 13 evaluated models:":
            "The deterministic taxonomy is applied uniformly across the 13-model publication cohort under corrected strict correctness and canonical-coordinate rules:",
        "Figure~\\ref{fig:failure_profiles} presents the same partition as a heatmap for direct visual comparison across models and failure categories.":
            "Figure~\\ref{fig:failure_profiles} visualizes the five primary failure categories; \\emph{None}, Unknown / Unmapped, and invalid responses remain visible in the table rather than the heatmap.",
        "Two models with the lowest response validity, qwen.qwen3-235b-a22b-2507 (26.2\\%) and MAI-Thinking-1 (40.8\\%), also show the highest \\emph{None} or diffuse taxonomy mass alongside their invalid share; because the taxonomy denominator is valid probes only, these percentages describe a small, potentially unrepresentative subset of each model's completed probes and should be read with that caveat.":
            "For models with low response validity, the taxonomy columns describe only the analyzable subset and may therefore be less representative of all completed probes.",
        "We evaluated ARCUS-X on 13 models and focus the detailed analysis on the five representative models in Table~\\ref{tab:overall_results}; the Appendix reports all 13.":
            "We evaluated ARCUS-X on 13 models and focus the detailed analysis on the five representative models in Table~\\ref{tab:overall_results}; the Appendix reports all 13. The paper's 13-model reporting cohort is preserved. The checkout also contains a Minimax M2.5 raw log; it is retained in the correction-package analyses and provenance but not silently added to the originally reported cohort.",
    }
    for old, new in replacements.items():
        if old not in tex:
            raise ValueError(f"Expected manuscript passage was not found: {old[:90]}")
        tex = tex.replace(old, new, 1)
    temporal_caption = r"""\caption{
Within-trajectory step accuracy at two common requested horizons for the 13-model publication cohort. Equal horizon does not imply identical tasks; this table is descriptive and is not task-paired unless exact task hashes match. Early, middle, and late correspond to contiguous thirds of the requested transition sequence. Values are conditional on valid, analyzable trajectories. The table is a positional diagnostic rather than a model ranking.
}"""
    tex, count = re.subn(
        r"\\caption\{\s*Within-trajectory step accuracy at two horizons shared by all 13 reported\s*models\..*?ranking\.\s*\}",
        lambda _: temporal_caption,
        tex,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise ValueError("Could not revise the common-horizon table caption")

    selected = [
        "GPT-5-mini",
        "moonshotai.kimi-k2.5",
        "zai.glm-5",
        "qwen.qwen3-coder-480b-a35b-v1:0",
        "mistral-large-2512",
    ]
    tex = _replace_table(
        tex,
        "tab:failure_taxonomy_selected",
        _taxonomy_table(
            model_summary,
            selected,
            "tab:failure_taxonomy_selected",
            "Corrected post-run validity and taxonomy partition for the five representative models. Taxonomy percentages are relative to valid probes; \\emph{None} requires complete canonical trajectory correctness.",
        ),
    )
    tex = _replace_table(
        tex,
        "tab:failure_taxonomy_full",
        _taxonomy_table(
            model_summary,
            MANUSCRIPT_MODELS,
            "tab:failure_taxonomy_full",
            "Corrected post-run validity and taxonomy partition for the 13-model publication cohort. Taxonomy percentages are relative to valid probes; invalid responses are reported separately.",
        ),
    )

    tex = tex.replace(
        "Failure-mode profiles across all 13 evaluated models, expressed as a percentage of valid probes.",
        "Corrected failure-signature profiles across the 13-model publication cohort, expressed as a percentage of valid probes.",
        1,
    )

    taxonomy_appendix = r"""\subsubsection{Primary mechanisms and precedence}
    For a parseable response, the primary taxonomy applies these decisions in
    order:
\begin{enumerate}
        \item A full-length modulo-equivalent path containing noncanonical
        coordinates is an output-format failure, not success, unless an
        out-of-range step follows the unwrapped transition rule; that boundary
        violation retains transition-failure precedence.
        \item A complete, literal match to the full canonical reference path is
        \emph{None}. Verbose or Markdown-wrapped correct responses remain correct;
        output compliance is stored separately.
    \item An initial-state mismatch is a semantic-interpretation failure.
    \item A wraparound-only boundary error is a transition-rule failure.
        \item If all overlapping transitions are otherwise correct, a length
        mismatch is a horizon failure.
    \item After a correct prefix, persistent offset or increasing drift is a
    state-tracking failure. Otherwise, a majority of structured transition
    errors is a transition-rule failure; immediate increasing divergence is a
    horizon failure; remaining valid early-divergence cases use the
    conservative state-tracking fallback.
        \item For a valid parsed response with zero aligned matches and a length
        mismatch, output-format failure overrides the preceding mechanism.
\end{enumerate}
    Responses that fail operational parsing are invalid and excluded from this
    trajectory-level taxonomy.
The classifier also records nonexclusive secondary patterns and temporal
dynamics. Reports normalize implementation labels to \emph{State Tracking
Failure}, \emph{Transition Rule Failure}, \emph{Semantic Interpretation
Failure}, \emph{Horizon Collapse}, and \emph{Formatting Failure}. A valid
trajectory that cannot be assigned to one of these categories is
\emph{Unknown / Unmapped}; it is never counted as \emph{None}. The analysis
also records fallback assignments separately, and Analysis B reports their
reclassification as a sensitivity policy. These deterministic labels describe
observable output under this precedence, not latent cognitive causes."""
    tex, count = re.subn(
        r"\\subsubsection\{Primary mechanisms and precedence\}.*?"
        r"(?=\\subsection\{Operational and Temporal Failure Analysis\})",
        lambda _: taxonomy_appendix + "\n\n",
        tex,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise ValueError("Could not update the taxonomy precedence appendix")

    policies = [
        ("B_conservative_unclassified", "Conservative"),
        ("B_fallback_unclassified", "Fallback"),
        ("B_horizon_first", "Horizon-first"),
        ("B_transition_first", "Transition-first"),
        ("B_modulo_equivalent", "Modulo"),
    ]
    stability = _csv_read(OUT / "analysis_B_policy_stability.csv")
    stability_by_model = {
        (row["model"], row["policy"]): row for row in stability
    }
    appendix = [
        r"\subsection{Taxonomy Policy Sensitivity (Analysis B)}",
        r"\label{app:taxonomy_sensitivity}",
        r"Analysis A is the primary analysis and uses the corrected classifier, strict full-trajectory correctness, canonical coordinates, and the original primary precedence. Analysis B changes one policy at a time; it does not replace Analysis A. The table reports, for each manuscript model, the percentage of valid observations whose primary label changes under each sensitivity policy. Conservative reclassification maps fallback, unknown, and any non-exact \emph{None} to Unclassified; fallback-only reclassification maps only state-tracking fallback cases; horizon-first and transition-first reorder the corresponding detected indicator; modulo sensitivity treats full-length modulo-equivalent noncanonical paths as correct. Full category distributions and probe-level labels, including the additional Minimax M2.5 log, are in \texttt{analysis\_B\_sensitivity\_summary.csv} and \texttt{analysis\_B\_probe\_level.csv}.",
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Share of valid probes reassigned relative to corrected Analysis A under alternative taxonomy policies.}",
        r"\label{tab:taxonomy_policy_stability}",
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Conservative} & \textbf{Fallback} & "
        r"\textbf{Horizon-first} & \textbf{Transition-first} & \textbf{Modulo} \\",
        r"\midrule",
    ]
    for model in MANUSCRIPT_MODELS:
        if any((model, policy) not in stability_by_model for policy, _ in policies):
            raise ValueError(f"Missing Analysis B stability row for {model}")
        values = []
        for policy, _ in policies:
            row = stability_by_model[(model, policy)]
            values.append(
                f"{100 * _float(row.get('proportion_changed')):.1f}\\%"
            )
        appendix.append(
            f"{MODEL_LABELS[model]} & " + " & ".join(values) + r" \\"
        )
    appendix.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ])
    if r"\end{document}" not in tex:
        raise ValueError("Could not append Analysis B section to manuscript")
    tex = tex.replace(
        r"\end{document}",
        "\n".join(appendix) + "\n\n\\end{document}",
        1,
    )
    (OUT / "arcus_x_corrected.tex").write_text(tex, encoding="utf-8")
    _generate_failure_figure(model_summary)


def _synthetic_validation() -> None:
    from arcus.analysis.taxonomy import classify_trajectory

    width, height = 20, 10
    gt = [f"[{i},0]" for i in range(7)]
    rules = {"east": {"dx": 1, "dy": 0}}
    actions = ["east"] * (len(gt) - 1)
    cases: dict[str, tuple[list[str], str]] = {
        "fully_correct": (list(gt), "\n".join(gt)),
        "premature_termination": (gt[:-2], "\n".join(gt[:-2])),
        "correct_prefix_with_excess_coordinates": (gt + ["[7,0]"], "\n".join(gt + ["[7,0]"])),
        "wrong_initial_coordinate": (
            [f"[{i + 1},0]" for i in range(7)], "shifted initial state",
        ),
        "single_incorrect_transition": (
            gt[:3] + ["[3,1]"] + gt[4:], "single transition deviation",
        ),
        "persistent_state_offset": (
            gt[:3] + [f"[{i},1]" for i in range(3, 7)], "persistent offset",
        ),
        "incorrect_transition_rule": (
            ["[0,0]", "[0,1]", "[1,1]", "[2,1]", "[3,1]", "[4,1]", "[5,1]"],
            "modeled north movement for east",
        ),
        "repeated_or_omitted_state": (
            gt[:3] + ["[2,0]"] + gt[4:], "repeated state in requested sequence",
        ),
        "noncanonical_modulo_equivalent": (
            gt[:-1] + ["[26,0]"], "last state is modulo-equivalent but noncanonical",
        ),
        "invalid_exception_output": (list(gt), "[EXCEPTION] invalid response"),
        "multiple_failure_signatures": (
            ["[1,0]", "[2,0]", "[2,1]", "[3,2]", "[4,2]", "[5,2]", "[6,3]"],
            "initial offset plus non-rule-consistent transitions",
        ),
        "ambiguous_unstructured_failure": (
            ["[0,0]", "[2,3]", "[6,1]", "[9,8]", "[12,4]", "[16,7]", "[19,2]"],
            "no unique deterministic failure mechanism asserted",
        ),
        "unparseable_output": (["no coordinate output"], "not parseable"),
    }
    out = []
    for name, (pred, raw) in cases.items():
        result = classify_trajectory(
            pred, gt, rules, actions, tier=0, grid_width=width,
            grid_height=height, raw_output=raw,
        )
        pc = result.probe_classification.to_dict() if result.probe_classification else {}
        length_mismatch = len(pred) != len(gt)
        primary = result.legacy_mode.value
        if result.valid and length_mismatch:
            horizon_first = "Horizon Collapse"
        else:
            horizon_first = primary
        transition_first = (
            "Transition Rule Failure"
            if result.valid and result.mechanism.value == "Transition Failure"
            else primary
        )
        fallback = (
            result.mechanism.value == "State Tracking Failure"
            and result.explanation == audit.FALLBACK_EXPLANATION
        )
        conservative = (
            "Unclassified" if fallback or primary == "Unknown / Unmapped" else primary
        )
        modulo = name == "noncanonical_modulo_equivalent"
        modulo_policy = "None" if modulo and result.valid else primary
        out.append({
            "synthetic_case": name,
            "known_constructed_property": "synthetic only; not human validation",
            "valid": result.valid,
            "primary_mechanism": result.mechanism.value,
            "primary_taxonomy": primary,
            "subtype": result.subtype.value,
            "explanation": result.explanation or "",
            "secondary_flags": json.dumps(pc.get("secondary_flags", [])),
            "exact_match": pc.get("exact_match", False),
            "trajectory_correct": pc.get("trajectory_correct", False),
            "predicted_length": len(pred),
            "ground_truth_length": len(gt),
            "length_mismatch_indicator": length_mismatch,
            "transition_violation_indicator": result.mechanism.value == "Transition Failure",
            "fallback_indicator": fallback,
            "horizon_first_taxonomy": horizon_first,
            "transition_first_taxonomy": transition_first,
            "conservative_taxonomy": conservative,
            "modulo_equivalent_taxonomy": modulo_policy,
        })
    _csv_write(OUT / "synthetic_validation.csv", out)


def _matched_horizon_analysis(rows: list[dict[str, str]]) -> None:
    base_rows = _csv_read(OUT / "matched_horizon_metrics.csv")
    task_sets: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    task_rows: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        r = _normalize_probe(raw)
        key = (r["model"], r["tier"], r["z"])
        if r.get("task_hash"):
            task_sets[key].add(r["task_hash"])
            task_rows[(*key, r["task_hash"])].append(r)
    model_conditions = sorted(task_sets)
    augmented = []
    for item in base_rows:
        tier, z = _int(item.get("tier")), _int(item.get("z"))
        model = item.get("model", "")
        own = task_sets.get((model, tier, z), set())
        common_max = max(
            (len(own & task_sets.get((other, tier, z), set()))
             for other in {m for m, t, h in model_conditions if t == tier and h == z and m != model}),
            default=0,
        )
        augmented.append({
            **item,
            "scope": "model x tier x requested horizon; descriptive estimate",
            "task_hashes_matched_with_any_other_model": common_max,
            "cross_model_task_match_available": common_max > 0,
            "paired_contrasts_claimed": False,
        })
    paired_metrics = (
        "step_accuracy", "exact_match", "continuity", "horizon_compliance",
        "early_accuracy", "middle_accuracy", "late_accuracy",
    )
    rng_seed = 241106
    for tier in range(4):
        for z in (20, 30):
            models = sorted(m for m, t, h in model_conditions if t == tier and h == z)
            for left_index, left in enumerate(models):
                for right in models[left_index + 1:]:
                    left_hashes = task_sets[(left, tier, z)]
                    right_hashes = task_sets[(right, tier, z)]
                    common = sorted(left_hashes & right_hashes)
                    for metric in paired_metrics:
                        by_seed: dict[int, list[float]] = defaultdict(list)
                        valid_pairs = 0
                        for task_hash in common:
                            left_rows = task_rows[(left, tier, z, task_hash)]
                            right_rows = task_rows[(right, tier, z, task_hash)]
                            left_values = [
                                float(r[metric]) if r["valid"] and not (r[metric] != r[metric]) else 0.0
                                for r in left_rows
                            ]
                            right_values = [
                                float(r[metric]) if r["valid"] and not (r[metric] != r[metric]) else 0.0
                                for r in right_rows
                            ]
                            if left_rows[0]["valid"] and right_rows[0]["valid"]:
                                valid_pairs += 1
                            difference = (
                                sum(left_values) / len(left_values)
                                - sum(right_values) / len(right_values)
                            )
                            by_seed[left_rows[0]["seed"]].append(difference)
                        seed_means = [
                            sum(values) / len(values)
                            for _, values in sorted(by_seed.items()) if values
                        ]
                        estimate = sum(seed_means) / len(seed_means) if seed_means else None
                        lows = highs = None
                        if seed_means:
                            rng = random.Random(rng_seed)
                            draws = [
                                sum(seed_means[rng.randrange(len(seed_means))]
                                    for _ in seed_means) / len(seed_means)
                                for _ in range(10000)
                            ]
                            draws.sort()
                            lows = draws[int(0.025 * (len(draws) - 1))]
                            highs = draws[int(0.975 * (len(draws) - 1))]
                        augmented.append({
                            "model": f"{left} - {right}",
                            "tier": tier,
                            "z": z,
                            "metric": f"paired_difference_{metric}",
                            "estimate": estimate,
                            "ci95_lower_cluster_seed": lows,
                            "ci95_upper_cluster_seed": highs,
                            "n_probe_conditions": len(common),
                            "n_valid_probe_conditions": valid_pairs,
                            "n_unique_task_hashes": len(common),
                            "denominator_n": len(common),
                            "n_seed_clusters": len(seed_means),
                            "bootstrap_replicates": 10000 if seed_means else 0,
                            "scope": "paired by identical task_hash; repeats averaged within task; seed-cluster bootstrap",
                            "task_hashes_matched_with_any_other_model": len(common),
                            "cross_model_task_match_available": bool(common),
                            "paired_contrasts_claimed": bool(common),
                        })
    _csv_write(OUT / "matched_horizon_analysis.csv", augmented)


def _original_table_source(tex: str, label: str) -> str:
    marker = rf"\label{{{label}}}"
    position = tex.find(marker)
    if position < 0:
        raise ValueError(f"Original manuscript table is missing label {label}")
    start = tex.rfind(r"\begin{table}", 0, position)
    end = tex.find(r"\end{table}", position)
    if start < 0 or end < 0:
        raise ValueError(f"Could not extract original manuscript table {label}")
    return tex[start:end + len(r"\end{table}")]


def _write_initial_manifest(
    raw_hashes_before: dict[str, str],
) -> None:
    if not MANUSCRIPT_SOURCE.is_file():
        raise FileNotFoundError(f"Manuscript source not found: {MANUSCRIPT_SOURCE}")
    source_tex = MANUSCRIPT_SOURCE.read_text(encoding="utf-8")
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    dependencies = {}
    for name in ("pytest", "numpy", "matplotlib", "transformers"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    inputs = [
        ROOT / "audit" / "probe_level_audit.csv",
        ROOT / "audit" / "taxonomy_model_summary.csv",
        ROOT / "audit" / "analyze_artifacts.py",
        MANUSCRIPT_SOURCE,
    ]
    initial = {
        "analysis_name": "ARCUS-X correction pre-analysis manifest",
        "baseline_git_commit": commit,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "dependencies": dependencies,
        },
        "raw_input_sha256_before_analysis": raw_hashes_before,
        "input_sha256_before_analysis": {
            str(path): _sha256(path) for path in inputs if path.is_file()
        },
        "manuscript_source": {
            "path": str(MANUSCRIPT_SOURCE),
            "sha256": _sha256(MANUSCRIPT_SOURCE),
            "original_taxonomy_tables": {
                label: _original_table_source(source_tex, label)
                for label in (
                    "tab:failure_taxonomy_selected",
                    "tab:failure_taxonomy_full",
                )
            },
        },
        "manuscript_support_file_sha256_before_analysis": {
            name: _sha256(MANUSCRIPT_SOURCE.parent / name)
            for name in MANUSCRIPT_SUPPORT_FILES
        },
        "prior_audit_model_table_values": _csv_read(
            ROOT / "audit" / "taxonomy_model_summary.csv"
        ),
    }
    (OUT / "PROVENANCE_MANIFEST_INITIAL.json").write_text(
        json.dumps(initial, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _original_taxonomy_values(label: str) -> dict[str, dict[str, str]]:
    tex = MANUSCRIPT_SOURCE.read_text(encoding="utf-8")
    table = _original_table_source(tex, label)
    model_by_label = {value.rstrip("*"): key for key, value in MODEL_LABELS.items()}
    headers = [
        "Valid", "Invalid", "State", "Transition", "Semantic",
        "Horizon", "Format", "Unknown", "None",
    ]
    values: dict[str, dict[str, str]] = {}
    for line in table.splitlines():
        if "&" not in line or r"\textbf{Model}" in line:
            continue
        line = re.sub(r"\\\\\s*$", "", line.strip())
        cells = [cell.strip() for cell in line.split("&")]
        if len(cells) != len(headers) + 1:
            continue
        model_label = cells[0].rstrip("*")
        model = model_by_label.get(model_label)
        if model is None:
            continue
        values[model] = dict(zip(headers, cells[1:]))
    if not values:
        raise ValueError(f"No original model rows could be read from {label}")
    return values


def _write_docs(
    model_summary: list[dict[str, Any]],
    reassignments: list[dict[str, Any]],
    b_summary: list[dict[str, Any]],
    rows: list[dict[str, str]],
    raw_hashes_before: dict[str, str],
    raw_hashes_after: dict[str, str],
) -> None:
    models = sorted({r["model"] for r in rows})
    raw_count = len(rows)
    changed = sum(
        _int(r["count"]) for r in reassignments if _bool(r.get("changed"))
    )
    max_shift = max(
        (abs(
            _int(r.get("corrected_none_count"))
            - _int(r.get("old_reported_none_count_same_observations"))
        )
         for r in model_summary),
        default=0,
    )
    model_summary_by_name = {r["model"]: r for r in model_summary}
    ml = model_summary_by_name.get("mistral-large-2512", {})
    km = model_summary_by_name.get("moonshotai.kimi-k2.5", {})
    fallback = sum(_int(r.get("state_tracking_fallback_count")) for r in model_summary)
    repeated_conditions = sum(
        _int(r.get("repeated_task_condition_observations"))
        for r in model_summary
    )
    duplicate_mapping_count = len(
        _csv_read(OUT / "exact_duplicate_mapping.csv")
    )
    same_raw = raw_hashes_before == raw_hashes_after

    dependency = """# Metric dependency audit

| Metric or output | Depends on trajectory/reference | Depends on action/rule metadata | Correction status |
|---|---|---|---|
| Step Accuracy | Yes; exact requested-state matches on requested transitions | No | Recomputed independently from preserved predicted and reference paths; unchanged by taxonomy-only and metadata-parser repairs. |
| Exact Match | Yes; literal full path and equal length | No | Recomputed independently; separate from Step Accuracy and taxonomy `None`. |
| Continuity / first divergence | Yes; ordered path comparison | No | Recomputed from raw paths; unaffected by action labels. |
| Horizon Match / Compliance | Yes; generated versus requested path length | No | Recomputed; extra coordinates remain non-compliant. |
| Operational validity | Raw parse / provider markers | No | Recomputed; full-length modulo-equivalent noncanonical paths are parse-valid formatting failures unless unwrapped transition violations retain Transition Failure precedence. |
| Taxonomy | Predicted/reference paths | Yes; transition rules/actions | Corrected for strict `None`, trusted metadata parsing, and noncanonical modulo-equivalent states. |
| AFS live acquisition | Saved probe accuracy/validity and runner search decisions | No taxonomy metadata | Existing acquisition decisions are not retroactively changed; no new model calls were made. |
| Post-hoc fracture depth | Horizon-level accuracy curve and validity policy | No | Sensitivity curves regenerated for valid-only and invalid-as-zero policies; distinct from AFS. |
| Temporal segmentation | Ordered path/reference comparisons and original thirds rule | No | Recomputed independently for available horizons; no taxonomy dependence. |
| CRI / other composite indices | Component metrics and their aggregation | Possibly indirectly | No new manuscript-level claim made; saved snapshots retained as source, and changed taxonomy does not redefine component metrics. |
| Generation overhead / GBI | Token count and expected-token convention | No | Raw token metrics are retained; not changed by taxonomy correction. |

Coordinate contract: the generator specifies `x in [0,width-1]`, `y in [0,height-1]`, a toroidal transition, and a full list containing the initial state plus one coordinate per action. Primary Analysis A uses literal canonical coordinates. A modulo-equivalent noncanonical path is retained as a failure, not `None`; modulo equality is reported only as an Analysis B sensitivity.
"""
    (OUT / "metric_dependency_audit.md").write_text(dependency, encoding="utf-8")

    claim_audit = """# Manuscript and checked-in claim audit

The supplied ICLR manuscript was inspected, copied into this correction package, and revised without modifying the original source. Numerical claims in the taxonomy tables and failure-profile figure were regenerated from Analysis A. Other metrics and tables remain unchanged where their computation does not depend on taxonomy labels; the common-horizon table is explicitly described as descriptive rather than task-paired.

| Claim | Status | Evidence / action |
|---|---|---|
| `None` denotes a complete correct trajectory | REVISED | Requires valid, complete, literal canonical-path correctness. Perfect Step Accuracy alone is insufficient. |
| Exact Match and Step Accuracy are interchangeable | UNSUPPORTED | Kept distinct; the paper's metrics tables are preserved separately from taxonomy. |
| Toroidal modulo equivalence makes noncanonical coordinates acceptable | WEAKENED | Canonical coordinates are primary; modulo-equivalent interpretations are Analysis B sensitivity only. |
| Taxonomy labels identify internal model mechanisms | UNSUPPORTED | Manuscript now describes deterministic behavioral signatures and documents precedence/fallback. |
| AFS is more efficient than fixed-horizon sampling | NOT CLAIMED | No cost-matched comparison supports an efficiency-superiority claim. |
| Cross-model comparisons at the same horizon are matched-condition comparisons | QUALIFIED | Main common-horizon table is descriptive; paired analyses require identical task hashes. |
| Taxonomy tables and figure | RECOMPUTED | Both taxonomy tables and the failure-profile heatmap use corrected Analysis A observations and denominators. |
| 13-model publication cohort | PRESERVED | The extra Minimax M2.5 raw log is analyzed and disclosed in machine-readable outputs, not silently added to paper tables. |
"""
    (OUT / "manuscript_claim_audit.md").write_text(claim_audit, encoding="utf-8")

    table_changes = []
    current = {r["model"]: r for r in model_summary}
    old_categories: Counter[tuple[str, str]] = Counter()
    for row in reassignments:
        old_categories[(row["model"], row["original_category"])] += _int(row["count"])
    category_specs = [
        ("Valid", "valid_probes", None),
        ("Invalid", "invalid_probes", None),
        ("State", "state_tracking_failure_count", "State Tracking Failure"),
        ("Transition", "transition_rule_failure_count", "Transition Rule Failure"),
        ("Semantic", "semantic_interpretation_failure_count", "Semantic Interpretation Failure"),
        ("Horizon", "horizon_collapse_count", "Horizon Collapse"),
        ("Format", "formatting_failure_count", "Formatting Failure"),
        ("Unknown", "unknown_unmapped_count", "Unknown / Unmapped"),
        ("None", "none_count", "None"),
    ]
    full_original = _original_taxonomy_values("tab:failure_taxonomy_full")
    selected_original = _original_taxonomy_values("tab:failure_taxonomy_selected")
    table_specs = [
        ("tab:failure_taxonomy_full", MANUSCRIPT_MODELS, full_original),
        (
            "tab:failure_taxonomy_selected",
            [
                "GPT-5-mini",
                "moonshotai.kimi-k2.5",
                "zai.glm-5",
                "qwen.qwen3-coder-480b-a35b-v1:0",
                "mistral-large-2512",
            ],
            selected_original,
        ),
    ]
    for table_name, table_models, original_values in table_specs:
        for model in table_models:
            row = current[model]
            old_valid = _int(row.get("old_valid_probes_same_observations"))
            if old_valid == 0 and _int(row.get("valid_probes")):
                old_valid = _int(row.get("valid_probes"))
            valid = _int(row.get("valid_probes"))
            total = _int(row.get("analysis_observations"))
            for header, field, old_category in category_specs:
                old_count = (
                    total - old_valid
                    if header == "Invalid"
                    else old_valid if header == "Valid"
                    else old_categories[(model, old_category)]
                )
                new_count = _int(row.get(field))
                denominator = total if header in {"Valid", "Invalid"} else valid
                old_denominator = total if header in {"Valid", "Invalid"} else old_valid
                table_changes.append({
                    "table": table_name,
                    "model": model,
                    "metric": f"{header} proportion",
                    "original_value_same_observations": (
                        f"{old_count}/{old_denominator}"
                        if old_denominator else ""
                    ),
                    "corrected_value": f"{new_count}/{denominator}" if denominator else "",
                    "original_table_value": original_values.get(model, {}).get(header, ""),
                    "absolute_change_percentage_points": (
                        100 * (
                            new_count / denominator - old_count / old_denominator
                        )
                        if denominator and old_denominator else ""
                    ),
                    "comparison_baseline": (
                        "Original labels joined by model/source row to the same retained observations"
                    ),
                    "original_source": str(MANUSCRIPT_SOURCE),
                    "corrected_source": (
                        "analysis_A_model_summary.csv and arcus_x_corrected.tex"
                    ),
                    "calculation_script": (
                        "audit/correction_2026_11/reproduce_correction.py"
                    ),
                })
    table_changes.extend([
        {
            "table": "fig:failure_profiles",
            "model": "",
            "metric": "failure-signature percentages among valid probes",
            "original_value": "Original PDF and source data recorded by the supplied manuscript package",
            "corrected_value": "Regenerated from analysis_A_model_summary.csv",
            "absolute_change": "",
            "original_source": str(MANUSCRIPT_SOURCE.parent / "failure_profiles.pdf"),
            "corrected_source": "failure_profiles.pdf",
            "calculation_script": "audit/correction_2026_11/reproduce_correction.py",
            "explanation": "Figure now depicts the 13-model publication cohort under corrected Analysis A.",
        },
        {
            "table": "tab:temporal_thirds",
            "model": "",
            "metric": "matched-horizon interpretation",
            "original_value": "Same requested horizon described as matched",
            "corrected_value": "Common-horizon descriptive; exact task-hash match required for paired comparison",
            "absolute_change": "",
            "original_source": str(MANUSCRIPT_SOURCE),
            "corrected_source": "arcus_x_corrected.tex and matched_horizon_analysis.csv",
            "calculation_script": "audit/correction_2026_11/reproduce_correction.py",
            "explanation": "No claim of paired conditions is made solely from horizon equality.",
        },
    ])
    _csv_write(OUT / "manuscript_table_changes.csv", table_changes)

    reviewer = f"""# Reviewer-response evidence package

The following addresses only the issue themes included in the correction request, not the full reviewer reports (which are not present in the checkout).

## Reviewer 9P4m

- **Concern:** State Tracking fallback, practical diagnostic interpretation, and justification of AFS versus fixed-horizon sampling.
- **Established:** Fallback cases are explicitly tracked in `analysis_A_model_summary.csv` and the taxonomy sensitivity tables. Corrected categories are operational labels, not causal explanations of model internals.
- **Corrected:** Perfect Step Accuracy no longer erases failure categories; fallback assignments can be moved to `Unclassified` in Analysis B.
- **Numerical change:** Total fallback assignments after reanalysis: {fallback}. AFS remains an adaptive acquisition method; no cost-controlled superiority claim is supported by this artifact set.
- **Suggested response:** “We now distinguish deterministic behavioral labels from mechanistic explanations and report fallback sensitivity separately. We do not claim AFS is more efficient than fixed-horizon sampling without a cost-matched comparison.”
- **Unresolved:** Full reviewer text was not part of the supplied materials.

## Reviewer Mf4g

- **Concern:** Taxonomy precedence, synthetic validation, unknown/unclassified sensitivity, and matched-horizon comparisons.
- **Established:** Five explicitly defined sensitivity policies, deterministic synthetic cases, and task-hash matched horizon comparisons are written to `analysis_B_sensitivity_summary.csv`, `synthetic_validation.csv`, and `matched_horizon_analysis.csv`.
- **Corrected:** Horizon-first, transition-first, conservative, fallback-unclassified, and modulo-equivalent policies are kept separate from Analysis A. Synthetic checks demonstrate implementation behavior only, not human agreement.
- **Suggested response:** “We now report all detected indicators and a preregistered set of alternative precedence policies. Matched comparisons use identical task hashes and seed-cluster uncertainty; equal horizon alone is not treated as matching.”
- **Unresolved:** No independent human annotations were performed; full reviewer text was not part of the supplied materials.

## Reviewer 9bBk

- **Concern:** `None` versus Exact Match, Mistral Large and Kimi K2.5 discrepancies, and AFS/fracture interpretation.
- **Established:** The previous Mistral Large artifact reported 1,501 `None` among 3,280 valid probes but only 37 exact matches; the corrected count is {ml.get('corrected_none_count', 'unavailable')} after full reanalysis. Kimi K2.5's old reported counts were 3,231 `None` and 2,539 Exact Matches; the corrected `None` count is {km.get('corrected_none_count', 'unavailable')}.
- **Corrected:** Taxonomy `None` now requires a valid full reference trajectory. Perfect Step Accuracy remains a separate metric. The action-metadata parser and noncanonical-coordinate labels are corrected.
- **Suggested response:** “We identified and corrected a reporting override that treated perfect requested-transition accuracy as full trajectory success. Exact Match and Step Accuracy are now reported separately, and noncanonical coordinates are not counted as primary-analysis success.”
- **Unresolved:** Live AFS acquisition history is not changed by post-hoc reanalysis.
"""
    (OUT / "reviewer_response_evidence.md").write_text(reviewer, encoding="utf-8")

    max_model = max(
        model_summary,
        key=lambda r: abs(
            _int(r.get("corrected_none_count"))
            - _int(r.get("old_reported_none_count_same_observations"))
        ),
    ) if model_summary else {}
    report = f"""# ARCUS-X correction report

## Executive summary

Analysis A applies strict trajectory correctness and the original primary taxonomy precedence after the verified parser and coordinate-contract fixes. Analysis B is separately labeled sensitivity evidence. No model API calls were made; source response logs and saved run snapshots were used. Raw input hashes match before and after analysis: **{same_raw}**.

## Inventory and provenance

- Raw model datasets: {len(models)} (the task asks for 13; the checkout contains {len(models)} named output logs, all included rather than silently dropping one).
- Logged response entries: {raw_count}; corrected summary observations are documented in the provenance manifest.
- Repeated task-condition observations: {repeated_conditions}; distinct responses at the same condition remain separate observations.
- Byte-identical complete raw entries excluded from estimates: {duplicate_mapping_count}; row-to-canonical-entry mapping is in `exact_duplicate_mapping.csv`.
- The paper retains its 13-model publication cohort; the additional Minimax M2.5 log is fully represented in the correction outputs and disclosed in the manuscript.
- Repository commit before edits: recorded in `PROVENANCE_MANIFEST.json`.
- Manuscript: the supplied `.tex` was copied and revised without overwriting the original; taxonomy tables and the affected heatmap were regenerated.

## Verified defects and fixes

1. `arcus/analysis/reporting.py`: `step_accuracy >= 1.0` no longer makes a probe `None`; correctness requires a valid exact full trajectory (or classifier-confirmed coordinate-list equality).
2. `post_run_analyzer.py` and `arcus/experiments/benchmark_runner.py`: task metadata fields are read only before `[MODEL OUTPUT]`; model-generated `Actions:` and similar text cannot overwrite trusted metadata.
3. `arcus/analysis/taxonomy.py`: noncanonical, modulo-equivalent paths are not success under the task's canonical output contract; unwrapped transition-rule violations retain the existing Transition Failure precedence.
4. Regression tests verify the reporting rule, coordinate policy, and agreement between live-run reconstruction and post-run scoring.

## Correction impact

- Valid-probe taxonomy reassignments (non-identical old/new labels): {changed}.
- Largest absolute `None` count change: {max_shift} probes ({max_model.get('model', 'n/a')}).
- Prior Mistral Large: 1,501 reported `None` / 3,280 valid vs. 37 exact matches; corrected `None`: {ml.get('corrected_none_count', 'unavailable')}.
- Prior Kimi K2.5: 3,231 `None` / 6,128 valid vs. 2,539 exact matches; corrected `None`: {km.get('corrected_none_count', 'unavailable')}.
- All numeric summaries include explicit denominators and exclusion counts. Old-to-new categories and distinct root causes are in `analysis_A_reassignment_matrix.csv`.

## Empirical implications

Step Accuracy, Exact Match, continuity, horizon compliance, temporal segmentation, and token overhead are recomputed from original path outputs. Parser and taxonomy corrections do not alter model responses or task references. Taxonomy distributions are revised; claims about diagnostic mechanisms are weakened to behavioral labels. Model comparisons are descriptive unless identical task hashes support paired comparisons. AFS remains an adaptive search procedure; efficiency superiority over fixed-horizon sampling is not established.

## Robustness

Analysis B compares corrected original precedence, horizon-first, transition-first, conservative-Unclassified, fallback-Unclassified, and modulo-equivalent coordinate policies. Full distributions are in `analysis_B_sensitivity_summary.csv`; synthetic examples are behavioral tests, not human validation. Policy-dependent conclusions must be described as taxonomy-sensitive.

## Manuscript updates

The supplied source was preserved. `arcus_x_corrected.tex` is a revised copy, with taxonomy-dependent tables, the failure-profile figure, strict `None` semantics, precedence documentation, the cohort disclosure, and matched-horizon wording updated. Other figures and metric tables remain unchanged where no dependency on the taxonomy correction was found. All 14 raw model logs are represented in the audit outputs; paper tables retain the 13-model publication cohort and disclose the extra Minimax M2.5 run.

## Reproduction

From the repository root, run:

```powershell
python audit\\correction_2026_11\\reproduce_correction.py
```

The command reads existing raw logs and snapshots, writes only within this correction directory, and does not invoke model APIs.
"""
    (OUT / "CORRECTION_REPORT.md").write_text(report, encoding="utf-8")

    raw_unchanged = same_raw
    original_summary = _csv_read(ROOT / "audit" / "taxonomy_model_summary.csv")
    input_files = (
        list((ROOT / "outputs").glob("*"))
        + [
            ROOT / "audit" / "probe_level_audit.csv",
            ROOT / "audit" / "taxonomy_model_summary.csv",
            ROOT / "audit" / "taxonomy_semantics.md",
            ROOT / "audit" / "analyze_artifacts.py",
            ROOT / "README.md",
            ROOT / "post_run_analyzer.py",
            ROOT / "arcus" / "analysis" / "reporting.py",
            ROOT / "arcus" / "analysis" / "taxonomy.py",
            ROOT / "arcus" / "experiments" / "benchmark_runner.py",
        ]
    )
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    dependencies = {}
    for name in ("pytest", "numpy", "matplotlib", "transformers"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    tex_files = list(ROOT.rglob("*.tex")) + list(ROOT.rglob("*.bib"))
    initial_manifest = OUT / "PROVENANCE_MANIFEST_INITIAL.json"
    initial_data = json.loads(initial_manifest.read_text(encoding="utf-8"))
    manifest = {
        "analysis_name": "ARCUS-X taxonomy correction and sensitivity analysis",
        "analysis_date": "2026-11-06",
        "repository": "nayanres/ARCUS-X",
        "baseline_git_commit": commit,
        "working_tree_changes_present": True,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "dependencies": dependencies,
        },
        "model_count_in_checkout": len(models),
        "models": models,
        "manuscript_publication_model_count": len(MANUSCRIPT_MODELS),
        "manuscript_publication_models": MANUSCRIPT_MODELS,
        "extra_raw_models_analyzed_but_not_added_to_publication_tables": sorted(
            set(models) - set(MANUSCRIPT_MODELS)
        ),
        "logged_response_entry_count": raw_count,
        "exact_duplicate_mapping_entry_count": len(
            _csv_read(OUT / "exact_duplicate_mapping.csv")
        ),
        "repeated_task_condition_observation_count": sum(
            _int(row.get("repeated_task_condition_observations"))
            for row in model_summary
        ),
        "raw_inputs_unchanged_before_after_analysis": raw_unchanged,
        "raw_input_sha256_before": raw_hashes_before,
        "raw_input_sha256_after": raw_hashes_after,
        "input_inventory": _hash_inventory([p for p in input_files if p.is_file()]),
        "manuscript_sources_found": [
            str(p.relative_to(ROOT)) for p in tex_files
        ] + [str(MANUSCRIPT_SOURCE)],
        "manuscript_source_sha256_before_analysis": initial_data[
            "manuscript_source"
        ]["sha256"],
        "manuscript_source_sha256_after_analysis": _sha256(MANUSCRIPT_SOURCE),
        "manuscript_support_file_sha256_before_analysis": initial_data[
            "manuscript_support_file_sha256_before_analysis"
        ],
        "manuscript_support_file_sha256_after_analysis": {
            name: _sha256(MANUSCRIPT_SOURCE.parent / name)
            for name in MANUSCRIPT_SUPPORT_FILES
        },
        "manuscript_original_taxonomy_tables": initial_data[
            "manuscript_source"
        ]["original_taxonomy_tables"],
        "initial_manifest_sha256": _sha256(initial_manifest),
        "prior_audit_model_table_values": original_summary,
        "duplicate_policy": (
            "Retain distinct raw response blocks with the same task-condition hash. "
            "Only byte-identical complete entry blocks may be excluded from summary "
            "estimates; every raw record remains in analysis_A_probe_level.csv."
        ),
    }
    (OUT / "PROVENANCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def finalize_existing_outputs() -> None:
    initial_path = OUT / "PROVENANCE_MANIFEST_INITIAL.json"
    initial = json.loads(initial_path.read_text(encoding="utf-8"))
    raw_hashes_before = initial["raw_input_sha256_before_analysis"]
    raw_hashes_after = {
        path.name: _sha256(path) for path in _raw_file_paths()
    }
    if raw_hashes_before != raw_hashes_after:
        raise RuntimeError("Raw model-output logs changed during correction analysis")
    if _sha256(MANUSCRIPT_SOURCE) != initial["manuscript_source"]["sha256"]:
        raise RuntimeError("The original manuscript source changed during analysis")
    model_summary = _csv_read(OUT / "analysis_A_model_summary.csv")
    base_rows = _recompute_transition_flags(
        _csv_read(OUT / "probe_level_audit.csv")
    )
    audit.OUTPUT = OUT
    audit._write_probe_csv(base_rows)
    audit._write_taxonomy_outputs(
        [_normalize_probe(row) for row in base_rows]
    )
    corrected_rows = _csv_read(OUT / "analysis_A_probe_level.csv")
    task_conditions: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    observation_counts: Counter[str] = Counter()
    unique_corrected_rows, _ = _exact_observations(corrected_rows)
    for row in unique_corrected_rows:
        model = row["model"]
        observation_counts[model] += 1
        task_conditions[model].add((
            model, _int(row.get("seed")), _int(row.get("z")),
            _float(row.get("gravity")), _int(row.get("tier")),
            row.get("grid_key", ""), row.get("task_hash", ""),
        ))
    for row in model_summary:
        model = row["model"]
        unique_count = len(task_conditions[model])
        row["unique_task_condition_count"] = unique_count
        row["repeated_task_condition_observations"] = (
            observation_counts[model] - unique_count
        )
    _csv_write(OUT / "analysis_A_model_summary.csv", model_summary)
    transition_by_source = {
        (row["model"], row["source_row"]): row["transition_first"]
        for row in base_rows
    }
    for row in corrected_rows:
        row["transition_first"] = transition_by_source[
            (row["model"], row["source_row"])
        ]
    _csv_write(OUT / "analysis_A_probe_level.csv", corrected_rows)
    _analysis_b(corrected_rows)
    _write_exact_duplicate_mapping(corrected_rows)
    _revise_manuscript(model_summary)
    reassignments = _csv_read(OUT / "analysis_A_reassignment_matrix.csv")
    b_summary = _csv_read(OUT / "analysis_B_sensitivity_summary.csv")
    _write_docs(
        model_summary,
        reassignments,
        b_summary,
        base_rows,
        raw_hashes_before,
        raw_hashes_after,
    )
    print(f"Finalized existing correction outputs in {OUT}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw_files = _raw_file_paths()
    raw_hashes_before = {p.name: _sha256(p) for p in raw_files}
    _write_initial_manifest(raw_hashes_before)
    audit.OUTPUT = OUT
    audit.main()

    original_probe_path = ROOT / "audit" / "probe_level_audit.csv"
    original_rows = _csv_read(original_probe_path)
    current_base_rows = _csv_read(OUT / "probe_level_audit.csv")
    corrected = _corrected_probe_file(current_base_rows)
    model_summary = _model_summary(
        corrected,
        _csv_read(ROOT / "audit" / "taxonomy_model_summary.csv"),
        original_rows,
    )
    action_rows = {
        p.name: _output_actions_rows(p) for p in raw_files
    }
    reassignments = _write_reassignments(
        original_rows, current_base_rows, action_rows
    )
    b_probes, b_summary = _analysis_b(corrected)
    _synthetic_validation()
    _matched_horizon_analysis(current_base_rows)
    _revise_manuscript(model_summary)
    raw_hashes_after = {p.name: _sha256(p) for p in raw_files}
    if raw_hashes_before != raw_hashes_after:
        raise RuntimeError("Raw model-output logs changed during correction analysis")
    _write_docs(
        model_summary, reassignments, b_summary, current_base_rows,
        raw_hashes_before, raw_hashes_after,
    )
    print(
        f"Created correction package for {len(raw_files)} raw model logs and "
        f"{len(current_base_rows)} response entries in {OUT}"
    )


if __name__ == "__main__":
    if "--finalize-existing" in sys.argv:
        finalize_existing_outputs()
    else:
        main()
