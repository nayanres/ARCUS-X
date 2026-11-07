#!/usr/bin/env python3
"""Reconcile requested-output and full-reference correctness against raw logs."""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from audit.correction_2026_11 import reviewer_response_analysis as audit

OUTPUT = audit.OUT / "reference_convention_reconciliation.csv"


def reconcile(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        if audit.is_calibration_anchor({
            "z": str(row["z"]), "tier": str(row["tier"]),
            "gravity": str(row["gravity"]),
        }):
            continue
        c = counters[row["model"]]
        c["completed"] += 1
        if not row["valid"]:
            c["invalid"] += 1
            continue
        c["valid"] += 1
        predicted, reference, z = row["predicted_path"], row["ground_truth_path_full"], row["z"]
        if len(reference) != z + 1:
            raise ValueError(f"{row['model']}: incomplete reference at z={z}")
        requested = predicted == reference[:z]
        full = predicted == reference
        none = row["taxonomy"] == "None"
        if none != full or (none and row["coordinate_interpretation"] != "canonical"):
            raise ValueError(f"{row['model']}: None/full-reference reconciliation failed")
        if row["taxonomy"] not in audit.TAXONOMY:
            raise ValueError(f"Unknown taxonomy label: {row['taxonomy']}")
        c["requested_exact"] += requested
        c["full_exact"] += full
        c["none"] += none
        c["none_length_z"] += none and len(predicted) == z
        c["none_length_z_plus_1"] += none and len(predicted) == z + 1
        if requested:
            c["requested_taxonomy_" + row["taxonomy"]] += 1

    output = []
    for model in audit.PUBLICATION_MODELS:
        c = counters[model]
        if not c["valid"] or c["completed"] != c["valid"] + c["invalid"]:
            raise ValueError(f"{model}: invalid observation partition")
        result = {
            "model": model, "completed": c["completed"], "valid": c["valid"], "invalid": c["invalid"],
            "requested_exact_valid_count": c["requested_exact"],
            "requested_exact_pct_completed": 100 * c["requested_exact"] / c["completed"],
            "requested_exact_pct_valid": 100 * c["requested_exact"] / c["valid"],
            "full_reference_exact_valid_count": c["full_exact"],
            "full_reference_exact_pct_valid": 100 * c["full_exact"] / c["valid"],
            "none_valid_count": c["none"], "none_length_z": c["none_length_z"],
            "none_length_z_plus_1": c["none_length_z_plus_1"],
        }
        for label in audit.TAXONOMY:
            result["requested_exact_label_" + label] = c["requested_taxonomy_" + label]
        if sum(result["requested_exact_label_" + label] for label in audit.TAXONOMY) != c["requested_exact"]:
            raise ValueError(f"{model}: requested-output label partition failed")
        output.append(result)
    return output


def main() -> None:
    with audit.PROBE_FILE.open(newline="", encoding="utf-8-sig") as handle:
        source = [r for r in csv.DictReader(handle) if r["model"] in audit.PUBLICATION_MODELS]
    namespace = audit.checkpoint_namespace(sorted({r["source_file"] for r in source}))
    audit.CHECKPOINTS = audit.CheckpointStore(
        audit.OUT / ".reviewer_response_cache" / "checkpoints.sqlite3", namespace
    )
    try:
        verified, _ = audit.load_and_verify_raw(source)
        output = reconcile(verified)
        audit.write_csv(OUTPUT, output)
        print(f"Verified {len(verified)} raw rows; wrote {OUTPUT}")
        for row in output:
            print(f"{row['model']}: requested={row['requested_exact_valid_count']}, "
                  f"full=None={row['none_valid_count']}, valid={row['valid']}")
    finally:
        audit.CHECKPOINTS.close()
        audit.CHECKPOINTS = None


if __name__ == "__main__":
    main()
