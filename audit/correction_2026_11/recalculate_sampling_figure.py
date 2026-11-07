#!/usr/bin/env python3
"""Recalculate the Appendix A.4 sampling plot from preserved observations."""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from audit.correction_2026_11 import reviewer_response_analysis as audit

MODEL = "ministral-14b-2512"
OUTPUT = audit.OUT / "ministral_14b_sampling_figure.csv"
MANUSCRIPT = audit.PACKAGE / "iclr2027_conference.tex"


def calculate(records):
    groups = defaultdict(list)
    for row in records:
        if row["model"] == MODEL:
            groups[row["z"]].append(row)
    if not groups:
        raise ValueError(f"No records for {MODEL}")
    total = sum(len(rows) for rows in groups.values())
    output = []
    for z, rows in sorted(groups.items()):
        valid = [row for row in rows if row["valid"]]
        if not valid:
            raise ValueError(f"Cannot plot undefined conditional accuracy at z={z}")
        values = [row["step_accuracy"] for row in valid]
        if any(value is None for value in values):
            raise ValueError(f"Missing valid-response accuracy at z={z}")
        calibration = sum(
            audit.is_calibration_anchor({
                "z": str(row["z"]), "tier": str(row["tier"]),
                "gravity": str(row["gravity"]),
            }) for row in rows
        )
        output.append({
            "horizon_z": z, "observations": len(rows),
            "valid_observations": len(valid), "invalid_observations": len(rows) - len(valid),
            "calibration_observations": calibration,
            "step_accuracy_pct_valid": 100 * sum(values) / len(values),
            "probe_density_pct_all_observations": 100 * len(rows) / total,
        })
    return output


def verify_manuscript(rows):
    text = MANUSCRIPT.read_text(encoding="utf-8")
    match = re.search(
        r"\\label\{app:afs_sampling\}.*?\] table\[meta=density\] \{\s*"
        r"z acc density\s*\n(.*?)\n\};",
        text, re.DOTALL,
    )
    if not match:
        raise ValueError("Could not locate Appendix A.4 sampling data")
    expected = [
        f'{r["horizon_z"]} {r["step_accuracy_pct_valid"]:.2f} '
        f'{r["probe_density_pct_all_observations"]:.3f}'
        for r in rows
    ]
    actual = [line.strip() for line in match.group(1).splitlines() if line.strip()]
    if actual != expected:
        raise ValueError("Manuscript plot values do not match recalculated observations")


def main():
    with audit.PROBE_FILE.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["model"] == MODEL]
    verified, _ = audit.load_and_verify_raw(rows)
    summary = calculate(verified)
    audit.write_csv(OUTPUT, summary)
    print(f"Verified {len(verified)} raw observations; wrote {len(summary)} horizon rows to {OUTPUT}")
    for row in summary:
        print(f'{row["horizon_z"]} {row["step_accuracy_pct_valid"]:.2f} '
              f'{row["probe_density_pct_all_observations"]:.3f}')
    if "--verify-manuscript" in sys.argv[1:]:
        verify_manuscript(summary)
        print("Manuscript plot values match recalculated observations.")


if __name__ == "__main__":
    main()
