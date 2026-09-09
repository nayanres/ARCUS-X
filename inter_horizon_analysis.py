#!/usr/bin/env python3
"""Post-hoc inter-horizon trajectory analysis for ARCUS-X raw output logs.

Taxonomy is reported at trajectory level because the canonical classifier does
not provide step-local taxonomy labels. Step accuracy, however, is computed
for each trajectory third.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from arcus.analysis.taxonomy import classify_from_result
from arcus.evaluation.parser import extract_model_path


TAXONOMY_BUCKETS = (
    "State Tracking Failure",
    "Transition Rule Failure",
    "Semantic Interpretation Failure",
    "Horizon Collapse",
    "Formatting Failure",
    "Unknown / Unmapped",
    "None",
)
TAXONOMY_MAP = {
    "State Tracking Failure": "State Tracking Failure",
    "Transition Failure": "Transition Rule Failure",
    "Transition Rule Failure": "Transition Rule Failure",
    "Semantic Failure": "Semantic Interpretation Failure",
    "Semantic Interpretation Failure": "Semantic Interpretation Failure",
    "Horizon Failure": "Horizon Collapse",
    "Horizon Collapse": "Horizon Collapse",
    "Output Format Failure": "Formatting Failure",
    "Formatting Failure": "Formatting Failure",
    "Unknown Failure": "Unknown / Unmapped",
    "Unknown / Unmapped": "Unknown / Unmapped",
    "None": "None",
}
INVALID_CODES = {
    "E100_EXCEPTION_TOKEN",
    "E101_EMPTY_OUTPUT",
    "E102_PARSE_FAILURE",
    "E103_PROVIDER_ERROR",
    "E104_UNKNOWN_INVALID",
}


@dataclass(frozen=True)
class RawEntry:
    z: int
    grid: str
    actions: list[str]
    transition_rules: dict[str, dict[str, int]]
    raw_output: str
    ground_truth: str


@dataclass(frozen=True)
class AnalyzedEntry:
    entry: RawEntry
    predicted: list[str]
    expected: list[str]
    valid: bool
    taxonomy: str | None


def split_horizon(z: int) -> dict[str, int]:
    """Return the required early/middle/end sizes, assigning all remainder to end."""
    if z < 0:
        raise ValueError("horizon must be non-negative")
    base, remainder = divmod(z, 3)
    return {"early": base, "middle": base, "end": base + remainder}


def _parse_grid(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", value, re.IGNORECASE)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def _literal(value: str, expected_type: type) -> Any:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return expected_type()
    return parsed if isinstance(parsed, expected_type) else expected_type()


def parse_raw_file(path: str | Path) -> list[RawEntry]:
    """Parse structured raw-output entries without invoking benchmark code."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"raw output file does not exist: {source}")
    text = source.read_text(encoding="utf-8")
    blocks = re.split(r"(?m)^=== RAW STREAM ENTRY.*$", text)
    entries: list[RawEntry] = []
    for block in blocks[1:]:
        block = block.split("==============================", 1)[0]
        params = re.search(r"(?m)^Parameters:\s*(.*)$", block)
        output = re.search(r"(?ms)^\[MODEL OUTPUT\]:\s*(.*?)(?=^\[GROUND TRUTH EXPECTED\]:)", block)
        truth = re.search(r"(?ms)^\[GROUND TRUTH EXPECTED\]:\s*(.*)$", block)
        if not (params and output and truth):
            continue
        fields = dict(re.findall(r"(\w+)\s*=\s*([^,]+)", params.group(1)))
        try:
            z = int(float(fields["z"]))
        except (KeyError, ValueError) as exc:
            raise ValueError("raw output contains an entry with invalid horizon metadata") from exc
        rules_match = re.search(r"(?m)^TransitionRules:\s*(.*)$", block)
        actions_match = re.search(r"(?m)^Actions:\s*(.*)$", block)
        grid_match = re.search(r"(?m)^Grid:\s*(.*)$", block)
        if not (rules_match and actions_match and grid_match):
            raise ValueError("raw output entry is missing Grid, Actions, or TransitionRules")
        entries.append(
            RawEntry(
                z=z,
                grid=grid_match.group(1).strip(),
                actions=[str(v) for v in _literal(actions_match.group(1).strip(), list)],
                transition_rules=_literal(rules_match.group(1).strip(), dict),
                raw_output=output.group(1),
                ground_truth=truth.group(1).strip(),
            )
        )
    if not entries:
        raise ValueError("no parseable RAW STREAM ENTRY blocks were found")
    return entries


def _taxonomy(entry: RawEntry, predicted: list[str], expected: list[str]) -> tuple[bool, str | None]:
    width, height = _parse_grid(entry.grid)
    result = classify_from_result(
        predicted, expected, entry.transition_rules, entry.actions, None, width, height, entry.raw_output
    )
    classification = result.probe_classification
    if not result.valid or (classification and classification.exception_code in INVALID_CODES):
        return False, None
    mode = result.legacy_mode.value
    if mode == "None" and predicted != expected:
        mode = "Unknown / Unmapped"
    return True, TAXONOMY_MAP.get(mode, "Unknown / Unmapped")


def analyze_entries(entries: Iterable[RawEntry]) -> list[AnalyzedEntry]:
    analyzed: list[AnalyzedEntry] = []
    for entry in entries:
        predicted = extract_model_path(entry.raw_output)
        expected = extract_model_path(entry.ground_truth)
        valid, taxonomy = _taxonomy(entry, predicted, expected)
        analyzed.append(AnalyzedEntry(entry, predicted, expected, valid, taxonomy))
    return analyzed


def _region_for_step(step: int, sizes: dict[str, int]) -> str | None:
    for region in ("early", "middle", "end"):
        if step <= sizes[region]:
            return region
        step -= sizes[region]
    return None


def _percent(correct: int, total: int) -> str:
    return f"{(correct / total) * 100:.1f}%" if total else "N/A"


def render_report(analyzed: list[AnalyzedEntry], all_tax: bool = False) -> str:
    accuracy: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    taxonomy: dict[str, Counter[str]] = defaultdict(Counter)
    per_horizon_tax: dict[int, Counter[str]] = defaultdict(Counter)

    for item in analyzed:
        for region in ("early", "middle", "end"):
            accuracy[item.entry.z][region]
        if not item.valid:
            continue
        sizes = split_horizon(item.entry.z)
        for step in range(1, item.entry.z + 1):
            region = _region_for_step(step, sizes)
            if region is None or step >= len(item.expected) or step >= len(item.predicted):
                continue
            accuracy[item.entry.z][region][1] += 1
            accuracy[item.entry.z][region][0] += item.predicted[step] == item.expected[step]
        label = item.taxonomy or "Unknown / Unmapped"
        taxonomy["trajectory"][label] += 1
        per_horizon_tax[item.entry.z][label] += 1

    lines = [
        "=" * 60,
        "ARCUS-X Inter-horizon trajectory analysis.",
        "=" * 60,
        "",
        "This tool aims to see where in the trajectory the model fails most often over horizons.",
        "",
        "Step Accuracy Per Third:",
    ]
    for z in sorted(accuracy):
        lines.append(f"\nz={z}")
        for region in ("early", "middle", "end"):
            correct, total = accuracy[z][region]
            lines.append(f"{region + ':':<8}{_percent(correct, total)}")
    lines.extend(["", "=" * 60, "Trajectory-level Taxonomy Distribution", "=" * 60])
    lines.append("")
    lines.append("Taxonomy labels are trajectory-level; they are not localized to a third.")
    counts = taxonomy["trajectory"]
    for bucket in TAXONOMY_BUCKETS:
        lines.append(f"{bucket}: {_percent(counts[bucket], sum(counts.values()))}")
    if all_tax:
        lines.extend(["", "=" * 60, "Trajectory-level Taxonomy By Horizon", "=" * 60])
        for z in sorted(per_horizon_tax):
            lines.extend(["", f"z={z}"])
            counts = per_horizon_tax[z]
            for bucket in TAXONOMY_BUCKETS:
                lines.append(f"{bucket}: {_percent(counts[bucket], sum(counts.values()))}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_file", help="ARCUS-X absolute raw-output text file")
    parser.add_argument("--all_tax", action="store_true", help="include taxonomy for every horizon and third")
    args = parser.parse_args(argv)
    try:
        report = render_report(analyze_entries(parse_raw_file(args.raw_file)), args.all_tax)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
