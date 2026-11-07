#!/usr/bin/env python3
"""Independently reconcile local summaries with corrected rows and raw metadata."""

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from audit.analyze_artifacts import _parse_raw_context
from audit.correction_2026_11 import horizon_local_analysis as local


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def assert_close(actual, expected):
    if expected is None:
        assert actual == ""
    else:
        assert math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=1e-10), (actual, expected)


def main():
    source = [r for r in read(local.base.PROBE_FILE) if r["model"] in local.base.PUBLICATION_MODELS]
    grouped = defaultdict(list)
    for r in source:
        grouped[r["source_file"]].append(r)
    signatures = {}
    for filename, rows in grouped.items():
        contexts = _parse_raw_context(local.base.RAW_DIR / filename)
        assert len(contexts) == len(rows)
        for index, (r, raw) in enumerate(zip(rows, contexts), 1):
            assert int(r["source_row"]) == index
            for name in ("seed", "tier", "z"):
                assert r[name] == raw[name], (filename, index, name)
            assert r["task_hash"] == raw.get("experiment_hash", "")
            assert float(r["gravity"]) == float(raw["gravity"])
            assert json.loads(r["actions"]) == raw["actions"]
            assert json.loads(r["transition_rules"]) == raw["transition_rules"]
            assert int(r["grid_width"]) == raw["grid_width"]
            assert int(r["grid_height"]) == raw["grid_height"]
            if r["task_hash"]:
                identity = (
                    r["seed"], r["tier"], r["z"], r["grid_width"], r["grid_height"],
                    r["actions"], r["transition_rules"], r["ground_truth_path"],
                )
                previous = signatures.setdefault(r["task_hash"], identity)
                assert previous == identity, (filename, index, "conflicting task hash")
        print(f"[raw metadata] {filename}: {len(rows)} rows", flush=True)

    cells = read(local.base.OUT / "horizon_local_cells.csv")
    comparisons = read(local.base.OUT / "horizon_local_comparisons.csv")
    groups = defaultdict(list)
    for r in source:
        if local.base.is_calibration_anchor(r):
            continue
        z = int(r["z"])
        targets = [("exact", z)] if z in (20, 30) else []
        targets += [(scope, h) for scope, bandwidth, hs in local.SCOPES if bandwidth
                    for h in hs if abs(z - h) <= bandwidth]
        if not targets:
            continue
        pred, truth = json.loads(r["predicted_path"]), json.loads(r["ground_truth_path"])
        assert len(truth) == z + 1
        valid = local.base.truth(r["valid"])
        if valid:
            accuracy = [int(i < len(pred) and pred[i] == truth[i]) for i in range(1, z)]
            third = len(accuracy) // 3
            regions = (accuracy[:third], accuracy[third:2 * third], accuracy[2 * third:])
            metrics = [sum(accuracy) / len(accuracy)] + [sum(part) / len(part) for part in regions]
            taxonomy = r["analysis_A_taxonomy"]
            assert taxonomy in local.base.TAXONOMY
            if taxonomy == "None":
                assert pred == truth and r["coordinate_interpretation"] == "canonical"
        else:
            metrics, taxonomy = [], ""
        simplified = {
            "z": z, "valid": valid, "seed": r["seed"], "task_hash": r["task_hash"],
            "gravity": r["gravity"], "taxonomy": taxonomy, "metrics": metrics,
        }
        for scope, h in targets:
            groups[(scope, h, r["model"], int(r["tier"]))].append(simplified)
    assert len(cells) == 52 * 32
    labels = {
        "none": "None", "horizon": "Horizon Collapse",
        "state_tracking": "State Tracking Failure", "transition": "Transition Rule Failure",
        "semantic": "Semantic Interpretation Failure", "format": "Formatting Failure",
        "unknown": "Unknown / Unmapped",
    }
    for c in cells:
        key = (c["scope"], int(c["target_horizon"]), c["model"], int(c["tier"]))
        rows = groups[key]
        valid = [r for r in rows if r["valid"]]
        seeds = {r["seed"] for r in valid if r["seed"]}
        tasks = {r["task_hash"] for r in valid if r["task_hash"]}
        expected_status = ("N/A-U" if not rows else "N/A-N" if not valid else
                           "N/A-I" if len(valid) < 10 or len(seeds) < 3 or len(tasks) < 5 else "OK")
        assert c["status"] == expected_status
        assert (int(c["observations"]), int(c["valid"]), int(c["invalid"])) == (
            len(rows), len(valid), len(rows) - len(valid))
        assert int(c["master_seeds_valid"]) == len(seeds)
        assert int(c["task_hashes_valid"]) == len(tasks)
        hz = defaultdict(Counter)
        for r in rows:
            hz[str(r["z"])]["valid" if r["valid"] else "invalid"] += 1
        expected_hz = {z: {"valid": counts["valid"], "invalid": counts["invalid"]} for z, counts in hz.items()}
        assert json.loads(c["horizon_frequencies"]) == expected_hz
        for i, metric in enumerate(local.base.METRICS):
            assert_close(c[f"mean_{metric}"], sum(r["metrics"][i] for r in valid) / len(valid) if valid else None)
            if expected_status != "OK":
                assert c[f"{metric}_ci95_lower"] == c[f"{metric}_ci95_upper"] == ""
        for name, label in labels.items():
            count = sum(r["taxonomy"] == label for r in valid)
            assert int(c[f"taxonomy_{name}_count"]) == count
            assert_close(c[f"taxonomy_{name}_proportion"], count / len(valid) if valid else None)
        if valid:
            assert sum(int(c[f"taxonomy_{name}_count"]) for name in labels) == len(valid)
    print(f"[cells] reconciled {len(cells)} cells, inclusion, scoring and taxonomy", flush=True)

    supported = 0
    for c in comparisons:
        h = int(c["target_horizon"])
        lkey = (c["scope"], h, c["model_left"], int(c["tier_left"]))
        rkey = (c["scope"], h, c["model_right"], int(c["tier_right"]))
        left, right = groups[lkey], groups[rkey]
        observed = sorted({r["z"] for r in left} & {r["z"] for r in right})
        assert c["observed_horizon_intersection"] == ";".join(map(str, observed))
        usable = []
        pairs = set()
        for z in observed:
            lv = [r for r in left if r["z"] == z and r["valid"]]
            rv = [r for r in right if r["z"] == z and r["valid"]]
            def sufficient(v):
                return len(v) >= 10 and len({r["seed"] for r in v if r["seed"]}) >= 3 and len({r["task_hash"] for r in v if r["task_hash"]}) >= 5
            if sufficient(lv) and sufficient(rv):
                usable.append((z, lv, rv))
            pairs.update((z, *p) for p in
                         {(r["seed"], r["task_hash"], float(r["gravity"])) for r in lv if r["task_hash"]}
                         & {(r["seed"], r["task_hash"], float(r["gravity"])) for r in rv if r["task_hash"]})
        assert c["supported_horizon_intersection"] == ";".join(str(z) for z, _, _ in usable)
        assert int(c["matched_seed_task_pairs"]) == len(pairs)
        if c["comparison_status"] == "OK":
            assert usable and c["left_status"] == c["right_status"] == "OK"
            lm = sum(sum(r["metrics"][0] for r in lv) / len(lv) for _, lv, _ in usable) / len(usable)
            rm = sum(sum(r["metrics"][0] for r in rv) / len(rv) for _, _, rv in usable) / len(usable)
            assert_close(c["standardized_left"], lm)
            assert_close(c["standardized_right"], rm)
            assert_close(c["difference"], lm - rm)
            supported += 1
        else:
            assert c["difference"] == ""
            if c["comparison_status"] in ("N/A-U", "N/A-N"):
                assert c["paired_status"] == c["comparison_status"]
    print(f"[comparisons] reconciled {len(comparisons)} intersections and task matches; {supported} supported", flush=True)
    print(f"VALIDATION PASSED: {len(source)} cohort observations; {len(signatures)} task hashes", flush=True)


if __name__ == "__main__":
    main()
