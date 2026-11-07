#!/usr/bin/env python3
"""Verify publication taxonomy against preserved probe and metric artifacts."""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from arcus.analysis.taxonomy import classify_requested_output
from audit.correction_2026_11 import reviewer_response_analysis as base
from audit.correction_2026_11 import horizon_local_analysis as local


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def synthetic_checks():
    reference = ["[0,0]", "[1,0]", "[2,0]", "[3,0]", "[4,0]"]
    cases = [
        ("requested exact", reference[:4], "raw", "None", True, {"R": {"dx": 1, "dy": 0}}),
        ("full reference overlong", reference, "raw", "Horizon Collapse", True, {"R": {"dx": 1, "dy": 0}}),
        ("short correct prefix", reference[:2], "raw", "Horizon Collapse", True, {"R": {"dx": 1, "dy": 0}}),
        ("extra incorrect coordinate", reference[:4] + ["[99,99]"], "raw", "Horizon Collapse", True, {"R": {"dx": 1, "dy": 0}}),
        ("overlong initial corruption", ["[9,9]"] + reference[1:], "raw", "Semantic Interpretation Failure", True, {"R": {"dx": 1, "dy": 0}}),
        ("initial corruption", ["[1,1]", "[2,1]", "[3,1]", "[4,1]"], "raw", "Semantic Interpretation Failure", True, {"R": {"dx": 1, "dy": 0}}),
        ("transition rule error", ["[0,0]", "[9,0]", "[8,0]", "[7,0]"], "raw", "Transition Rule Failure", True, {"R": {"dx": 1, "dy": 0}}),
        ("state tracking error", ["[0,0]", "[1,0]", "[3,0]", "[4,0]"], "raw", "State Tracking Failure", True, {"R": {"dx": 1, "dy": 0}}),
        ("noncanonical equivalent", ["[10,0]", "[11,0]", "[12,0]", "[13,0]"], "raw", "Formatting Failure", True, {"R": {"dx": 1, "dy": 0}}),
        ("unmapped divergence", ["[0,0]", "[9,0]", "[8,0]", "[7,0]"], "raw", "Unknown / Unmapped", True, {}),
        ("unparseable", ["not-a-coordinate"], "nonsense", "Invalid / Excluded", False, {}),
        ("empty", [], "", "Invalid / Excluded", False, {}),
        ("provider exception", [], "EXCEPTION: provider runtime failure", "Invalid / Excluded", False, {}),
    ]
    output = []
    for name, predicted, raw, expected, expected_valid, rules in cases:
        result = classify_requested_output(predicted, reference, 4, rules, ["R"] * 4, 0, 10, 10, raw)
        actual = result.legacy_mode.value if result.valid else "Invalid / Excluded"
        if actual != expected or result.valid != expected_valid:
            raise AssertionError(f"{name}: {actual}, valid={result.valid}; expected {expected}")
        output.append({"case": name, "expected": expected, "actual": actual, "valid": result.valid,
                       "matched": True, "explanation": result.explanation})
    base.write_csv(base.OUT / "requested_taxonomy_synthetic_checks.csv", output)


def main():
    crosswalk = read(base.OUT / "requested_taxonomy_probe_crosswalk.csv")
    previous = read(base.OUT / "requested_taxonomy_v1_archive" / "requested_taxonomy_probe_crosswalk.csv")
    if len(previous) != len(crosswalk):
        raise AssertionError("Prior taxonomy crosswalk size mismatch")
    fallback_count = 0
    for old, new in zip(previous, crosswalk):
        if (old["source_file"], old["source_row"]) != (new["source_file"], new["source_row"]):
            raise AssertionError("Prior taxonomy crosswalk row order mismatch")
        was_fallback = base.truth(old["fallback"])
        if was_fallback:
            fallback_count += 1
            if old["corrected_label"] != "State Tracking Failure" or new["corrected_label"] != "Unknown / Unmapped":
                raise AssertionError("Former state fallback was not mapped to Unknown")
        elif old["corrected_label"] != new["corrected_label"]:
            raise AssertionError("Non-fallback label changed")
        if new["taxonomy_version"] != "requested-output-v2":
            raise AssertionError("Unexpected taxonomy version")
    if fallback_count != 3:
        raise AssertionError("Unexpected number of previous fallback cases")
    source = {
        (r["source_file"], r["source_row"]): r
        for r in read(base.PROBE_FILE)
        if r["model"] in base.PUBLICATION_MODELS and not base.is_calibration_anchor(r)
    }
    if len(source) != len(crosswalk):
        raise AssertionError("Cohort row count mismatch")
    counts = defaultdict(Counter)
    observed = defaultdict(list)
    for row in crosswalk:
        original = source.pop((row["source_file"], row["source_row"]))
        z = int(row["z"])
        pred, ref = json.loads(original["predicted_path"]), json.loads(original["ground_truth_path"])
        valid = base.truth(original["valid"])
        requested = valid and pred == ref[:z]
        full = valid and pred == ref
        if (len(ref) != z + 1 or len(pred) != int(row["predicted_length"])
                or valid != base.truth(row["valid"])
                or requested != base.truth(row["requested_exact"])
                or full != base.truth(row["full_reference_exact"])
                or (row["corrected_label"] == "None") != requested
                or (full and row["corrected_label"] != "Horizon Collapse")
                or (not valid and row["corrected_label"] != "Invalid / Excluded")
                or (valid and row["historical_label"] != original["analysis_A_taxonomy"])):
            raise AssertionError(f'Probe invariant failed: {row["source_file"]}:{row["source_row"]}')
        c = counts[row["model"]]
        c["completed"] += 1
        c["valid"] += valid
        c["requested"] += requested
        c[row["corrected_label"]] += valid
        observed[(row["model"], int(row["tier"]), z)].append({
            "model": row["model"], "tier": int(row["tier"]), "z": z,
            "seed": row["seed"], "task_hash": row["task_hash"], "valid": valid,
            "taxonomy": row["corrected_label"],
        })
    if source or set(counts) != set(base.PUBLICATION_MODELS):
        raise AssertionError("Missing or unexpected cohort observations")
    metric_path = ROOT / "audit" / "prompt_coordinate_reanalysis" / "post_run_table_metrics_publication.csv"
    metrics = {r["model"]: r for r in read(metric_path)}
    for model, c in counts.items():
        m = metrics[model]
        if (c["completed"] != int(m["analysis_observations"]) or c["valid"] != int(m["valid"])
                or abs(100 * c["requested"] / c["completed"] - float(m["exact_match_all_pct"])) > 0.051
                or sum(c[label] for label in base.TAXONOMY) != c["valid"]):
            raise AssertionError(f"Model denominator or exact-match mismatch: {model}")
    crosstab = read(base.OUT / "requested_taxonomy_cross_tab.csv")
    if sum(int(r["count"]) for r in crosstab) != len(crosswalk):
        raise AssertionError("Cross-tab does not include all completed observations")
    cells = defaultdict(list)
    for row in read(base.OUT / "requested_taxonomy_horizon_taxonomy.csv"):
        cells[(row["scope"], row["target_horizon"], row["model"], row["tier"])].append(row)
    if len(cells) != 1664:
        raise AssertionError("Horizon coverage cell count mismatch")
    checked_bootstrap = set()
    for key, rows in cells.items():
        valid = int(rows[0]["valid"])
        scope, target, model, tier = key
        target, tier = int(target), int(tier)
        bandwidth = int(rows[0]["bandwidth"])
        group = [r for z in range(target - bandwidth, target + bandwidth + 1)
                 for r in observed[(model, tier, z)]]
        expected = Counter(r["taxonomy"] for r in group if r["valid"])
        if (local.support(group) != rows[0]["status"]
                or any(int(r["count"]) != expected[r["taxonomy"]] for r in rows)):
            raise AssertionError(f"Horizon inclusion or coverage mismatch: {key}")
        if len(rows) != len(base.TAXONOMY) or sum(int(r["count"]) for r in rows) != valid:
            raise AssertionError(f"Horizon partition mismatch: {key}")
        if valid and abs(sum(float(r["proportion_valid"]) for r in rows) - 1) > 1e-10:
            raise AssertionError(f"Horizon proportions mismatch: {key}")
        for r in rows:
            if r["status"] == "OK":
                if not (0 <= float(r["ci95_lower"]) <= float(r["ci95_upper"]) <= 1):
                    raise AssertionError(f"Invalid seed-block confidence interval: {key}")
            elif r["ci95_lower"] or r["ci95_upper"]:
                raise AssertionError(f"Unsupported confidence interval: {key}")
        if rows[0]["status"] == "OK" and scope not in checked_bootstrap:
            valid_rows = [r for r in group if r["valid"]]
            _, ci = base.cluster_bootstrap(valid_rows, f"requested-taxonomy-v1|{scope}|{target}|{model}|{tier}")
            if any((float(r["ci95_lower"]), float(r["ci95_upper"])) != ci[r["taxonomy"]] for r in rows):
                raise AssertionError(f"Seed-block bootstrap mismatch: {key}")
            checked_bootstrap.add(scope)
    if len(checked_bootstrap) != 3:
        raise AssertionError("Bootstrap not checked for every horizon scope")
    synthetic_checks()
    print(f"Verified {len(crosswalk)} cohort observations, {len(cells)} horizon cells, "
          f"{len(crosstab)} cross-tab rows, {fallback_count} reclassified fallbacks, "
          "13 synthetic cases, and all 13 model exact-match denominators.")


if __name__ == "__main__":
    main()
