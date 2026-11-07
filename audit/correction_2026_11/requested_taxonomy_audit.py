#!/usr/bin/env python3
"""Reclassify preserved responses under the requested-coordinate contract."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from arcus.analysis.reporting import canonicalize_taxonomy
from arcus.analysis.taxonomy import (
    REQUESTED_OUTPUT_TAXONOMY_VERSION, classify_requested_output, classify_trajectory,
)
from arcus.evaluation.parser import extract_model_path
from audit.correction_2026_11 import reviewer_response_analysis as base
from audit.correction_2026_11 import horizon_local_analysis as local
from audit.analyze_artifacts import _transition_first_matches
from inter_horizon_analysis import parse_raw_file

PREFIX = "requested_taxonomy_"


def report_label(result):
    if not result.valid:
        return "Invalid / Excluded"
    return canonicalize_taxonomy(
        result.legacy_mode.value, bool(result.probe_classification.trajectory_correct)
    )


def classify_file(filename, rows):
    raw = parse_raw_file(base.RAW_DIR / filename)
    if len(raw) != len(rows):
        raise ValueError(f"{filename}: source row count mismatch")
    output = []
    for i, (row, entry) in enumerate(zip(rows, raw), 1):
        if int(row["source_row"]) != i or int(row["z"]) != entry.z:
            raise ValueError(f"{filename}:{i}: source identity mismatch")
        predicted = extract_model_path(entry.raw_output)
        reference = extract_model_path(entry.ground_truth)
        if predicted != json.loads(row["predicted_path"]) or reference != json.loads(row["ground_truth_path"]):
            raise ValueError(f"{filename}:{i}: raw trajectory mismatch")
        z = entry.z
        if len(reference) != z + 1 or len(entry.actions) != z:
            raise ValueError(f"{filename}:{i}: incomplete generated reference/actions")
        width, height = int(row["grid_width"]), int(row["grid_height"])
        historical = classify_trajectory(
            predicted, reference, entry.transition_rules, entry.actions,
            int(row["tier"]), width, height, entry.raw_output,
        )
        corrected = classify_requested_output(
            predicted, reference, z, entry.transition_rules, entry.actions,
            int(row["tier"]), width, height, entry.raw_output,
        )
        valid = base.truth(row["valid"])
        if historical.valid != valid or corrected.valid != valid:
            raise ValueError(f"{filename}:{i}: validity changed")
        old = row["analysis_A_taxonomy"] if valid else "Invalid / Excluded"
        if valid and report_label(historical) != old:
            raise ValueError(f"{filename}:{i}: historical label mismatch ({report_label(historical)} != {old})")
        new = report_label(corrected)
        requested = predicted == reference[:z] if valid else False
        full = predicted == reference if valid else False
        if valid and ((new == "None") != requested):
            raise ValueError(f"{filename}:{i}: None/requested exact invariant failed")
        if full and new != "Horizon Collapse":
            raise ValueError(f"{filename}:{i}: full exact not overlong horizon")
        pc = corrected.probe_classification
        flags = list(pc.secondary_flags)
        old_flags = list(historical.probe_classification.secondary_flags)
        mismatch = len(predicted) != z
        transition = _transition_first_matches(
            predicted[:z], reference[:z], entry.transition_rules, entry.actions[:z - 1], width, height
        ) if valid else False
        fallback = valid and corrected.explanation == "Early divergence has no explicitly mapped mechanism."
        modulo = (
            valid and len(predicted) == z and width > 0 and height > 0
            and any(int(x.strip("[]").split(",")[0]) < 0
                    or int(x.strip("[]").split(",")[0]) >= width
                    or int(x.strip("[]").split(",")[1]) < 0
                    or int(x.strip("[]").split(",")[1]) >= height for x in predicted)
            and all(
                tuple(int(v) % dim for v, dim in zip(p.strip("[]").split(","), (width, height)))
                == tuple(int(v) for v in q.strip("[]").split(","))
                for p, q in zip(predicted, reference[:z])
            )
        )
        reasons = (
            "unchanged" if old == new else
            "requested_exact" if requested else
            "full_reference_overlong" if full else
            "requested_reference_reclassification"
        )
        output.append({
            "model": row["model"], "source_file": filename, "source_row": i,
            "seed": row["seed"], "tier": int(row["tier"]), "gravity": row["gravity"],
            "z": z, "task_hash": row["task_hash"], "valid": valid,
            "requested_exact": requested, "full_reference_exact": full,
            "requested_length": z, "predicted_length": len(predicted),
            "historical_label": old, "corrected_label": new,
            "historical_secondary_flags": json.dumps(old_flags),
            "corrected_secondary_flags": json.dumps(flags),
            "label_changed": old != new, "change_reason": reasons,
            "corrected_mechanism": pc.mechanism.value,
            "corrected_subtype": corrected.subtype.value,
            "corrected_explanation": corrected.explanation,
            "fallback": fallback, "transition_first": transition,
            "length_mismatch": mismatch, "modulo_equivalent_noncanonical": modulo,
            "taxonomy_version": REQUESTED_OUTPUT_TAXONOMY_VERSION,
        })
    return output


def aggregate(records):
    noncal = [r for r in records if not base.is_calibration_anchor({
        "z": str(r["z"]), "tier": str(r["tier"]), "gravity": str(r["gravity"])
    })]
    matrix = Counter((r["historical_label"], r["corrected_label"]) for r in noncal if r["valid"])
    base.write_csv(base.OUT / f"{PREFIX}transition_matrix.csv", [
        {"historical_label": a, "corrected_label": b, "count": matrix[(a, b)]}
        for a in base.TAXONOMY for b in base.TAXONOMY
    ])
    summaries = []
    crosstab = Counter()
    sensitivity = Counter()
    sensitivity_changed = Counter()
    grouped = defaultdict(list)
    for r in noncal:
        grouped[(r["model"], r["tier"])].append(r)
        if r["label_changed"] and r["change_reason"] == "requested_reference_reclassification":
            r["change_reason"] = (
                f'{r["historical_label"]} -> {r["corrected_label"]}: '
                f'{r["corrected_explanation"]}'
            )
        compliance_numerator = min(r["predicted_length"], r["requested_length"])
        compliance_denominator = max(r["predicted_length"], r["requested_length"])
        divisor = math.gcd(compliance_numerator, compliance_denominator)
        compliance = f"{compliance_numerator // divisor}/{compliance_denominator // divisor}"
        crosstab[(r["model"], r["tier"], r["valid"], r["requested_exact"] if r["valid"] else "",
                  r["full_reference_exact"] if r["valid"] else "", not r["length_mismatch"],
                  compliance, r["corrected_label"])] += 1
        if not r["valid"]:
            continue
        p = r["corrected_label"]
        policies = {
            "primary": p,
            "horizon_first": "Horizon Collapse" if r["length_mismatch"] else p,
            "transition_first": "Transition Rule Failure" if r["transition_first"] else p,
            "conservative_unclassified": "Unclassified" if r["fallback"] or p == "Unknown / Unmapped" else p,
            "fallback_state_tracking": "State Tracking Failure" if r["fallback"] else p,
            "modulo_equivalent": "None" if r["modulo_equivalent_noncanonical"] else p,
        }
        for policy, label in policies.items():
            sensitivity[(r["model"], policy, label)] += 1
            sensitivity_changed[(r["model"], policy)] += label != p
    for model in base.PUBLICATION_MODELS:
        for tier in base.TIERS:
            rows = grouped[(model, tier)]
            valid = [r for r in rows if r["valid"]]
            old, new = Counter(r["historical_label"] for r in valid), Counter(r["corrected_label"] for r in valid)
            count = len(valid)
            if sum(old.values()) != count or sum(new.values()) != count:
                raise ValueError("taxonomy partition mismatch")
            summary = {
                "model": model, "tier": tier, "completed": len(rows),
                "valid": count, "invalid": len(rows) - count,
                "labels_changed": sum(r["label_changed"] for r in valid),
                "changed_pct_valid": 100 * sum(r["label_changed"] for r in valid) / count if count else "",
                "requested_exact_none": sum(r["requested_exact"] and r["corrected_label"] == "None" for r in valid),
                "full_exact_overlong": sum(r["full_reference_exact"] and r["corrected_label"] == "Horizon Collapse" for r in valid),
                "former_state_fallback_unknown": sum(r["fallback"] for r in valid),
                "state_explicit": new["State Tracking Failure"],
                "unknown_unmapped": new["Unknown / Unmapped"],
            }
            for label in base.TAXONOMY:
                summary[f"historical_{label}_count"] = old[label]
                summary[f"corrected_{label}_count"] = new[label]
                summary[f"historical_{label}_pct_valid"] = 100 * old[label] / count if count else ""
                summary[f"corrected_{label}_pct_valid"] = 100 * new[label] / count if count else ""
            summaries.append(summary)
    base.write_csv(base.OUT / f"{PREFIX}model_tier_summary.csv", summaries)
    base.write_csv(base.OUT / f"{PREFIX}cross_tab.csv", [
        {"model": m, "tier": t, "valid": valid, "requested_exact": req, "full_reference_exact": full,
         "requested_length_compliant": length, "horizon_compliance": compliance,
         "taxonomy": label, "count": n}
        for (m, t, valid, req, full, length, compliance, label), n in sorted(crosstab.items(), key=lambda item: tuple(map(str, item[0])))
    ])
    policies = []
    for model in base.PUBLICATION_MODELS:
        denominator = sum(1 for r in noncal if r["model"] == model and r["valid"])
        for policy in ("primary", "horizon_first", "transition_first", "conservative_unclassified", "fallback_state_tracking", "modulo_equivalent"):
            for label in (*base.TAXONOMY, "Unclassified"):
                policies.append({"model": model, "policy": policy, "taxonomy": label,
                                 "count": sensitivity[(model, policy, label)], "valid": denominator,
                                 "changed": sensitivity_changed[(model, policy)]})
    base.write_csv(base.OUT / f"{PREFIX}precedence_sensitivity.csv", policies)
    return noncal, summaries


def horizon_results(records, prior_horizon=None, changed=None):
    groups = defaultdict(list)
    for row in records:
        groups[(row["model"], row["tier"], row["z"])].append(row)
    previous = defaultdict(list)
    for row in prior_horizon or []:
        previous[(row["scope"], int(row["target_horizon"]), row["model"], int(row["tier"]))].append(row)
    output = []
    for scope, bandwidth, targets in local.SCOPES:
        for h in targets:
            print(f"[taxonomy {scope} h={h}]", flush=True)
            for model in base.PUBLICATION_MODELS:
                for tier in base.TIERS:
                    key = (scope, h, model, tier)
                    if prior_horizon is not None and not any(
                        m == model and t == tier and abs(z - h) <= bandwidth
                        for m, t, z in changed
                    ):
                        if len(previous[key]) != len(base.TAXONOMY):
                            raise ValueError(f"Missing prior horizon cell: {key}")
                        output.extend(previous[key])
                        continue
                    group = [r for z in range(h - bandwidth, h + bandwidth + 1)
                             for r in groups[(model, tier, z)]]
                    valid = [r for r in group if r["valid"]]
                    status = local.support(group)
                    counts = Counter(r["corrected_label"] for r in valid)
                    bootrows = [{"seed": r["seed"], "valid": True, "taxonomy": r["corrected_label"]}
                                for r in valid]
                    _, ci = base.cluster_bootstrap(bootrows, f"requested-taxonomy-v1|{scope}|{h}|{model}|{tier}") if status == "OK" else ({}, {})
                    for label in base.TAXONOMY:
                        low, high = ci.get(label, (None, None))
                        output.append({"scope": scope, "bandwidth": bandwidth, "target_horizon": h,
                                       "model": model, "tier": tier, "status": status, "valid": len(valid),
                                       "taxonomy": label, "count": counts[label],
                                       "proportion_valid": counts[label] / len(valid) if valid else "",
                                       "ci95_lower": low if low is not None else "",
                                       "ci95_upper": high if high is not None else ""})
    base.write_csv(base.OUT / f"{PREFIX}horizon_taxonomy.csv", output)


def upgrade_v1():
    archive = base.OUT / "requested_taxonomy_v1_archive"
    with (archive / f"{PREFIX}probe_crosswalk.csv").open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    with (archive / f"{PREFIX}horizon_taxonomy.csv").open(encoding="utf-8-sig", newline="") as f:
        prior_horizon = list(csv.DictReader(f))
    for row in records:
        if row["taxonomy_version"] != "requested-output-v1":
            raise ValueError("Upgrade requires a complete v1 crosswalk")
        for key in ("tier", "z", "requested_length", "predicted_length"):
            row[key] = int(row[key])
        for key in ("valid", "requested_exact", "full_reference_exact", "label_changed", "fallback",
                    "transition_first", "length_mismatch", "modulo_equivalent_noncanonical"):
            row[key] = base.truth(row[key])
    selected = [r for r in records if r["fallback"]]
    if not selected or any(r["corrected_label"] != "State Tracking Failure" for r in selected):
        raise ValueError("Unexpected v1 fallback records")
    source = {
        (r["source_file"], int(r["source_row"])): r
        for r in csv.DictReader(base.PROBE_FILE.open(encoding="utf-8-sig", newline=""))
        if (r["source_file"], int(r["source_row"])) in
        {(s["source_file"], int(s["source_row"])) for s in selected}
    }
    raw_files = {name: parse_raw_file(base.RAW_DIR / name) for name in {r["source_file"] for r in selected}}
    changed = set()
    for row in selected:
        entry = raw_files[row["source_file"]][int(row["source_row"]) - 1]
        original = source[(row["source_file"], int(row["source_row"]))]
        predicted, reference = extract_model_path(entry.raw_output), extract_model_path(entry.ground_truth)
        if (predicted != json.loads(original["predicted_path"])
                or reference != json.loads(original["ground_truth_path"]) or entry.z != row["z"]):
            raise ValueError("Fallback source record mismatch")
        result = classify_requested_output(
            predicted, reference, entry.z, entry.transition_rules, entry.actions, row["tier"],
            int(original["grid_width"]), int(original["grid_height"]), entry.raw_output,
        )
        if (report_label(result) != "Unknown / Unmapped"
                or result.explanation != "Early divergence has no explicitly mapped mechanism."):
            raise ValueError("Unexpected v2 fallback classification")
        row.update({
            "corrected_label": report_label(result),
            "corrected_secondary_flags": json.dumps(list(result.probe_classification.secondary_flags)),
            "corrected_mechanism": result.probe_classification.mechanism.value,
            "corrected_subtype": result.subtype.value,
            "corrected_explanation": result.explanation,
            "label_changed": row["historical_label"] != report_label(result),
            "change_reason": "unmapped_fallback_instead_of_state_tracking",
        })
        changed.add((row["model"], row["tier"], row["z"]))
    for row in records:
        row["taxonomy_version"] = REQUESTED_OUTPUT_TAXONOMY_VERSION
    noncal, _ = aggregate(records)
    base.write_csv(base.OUT / f"{PREFIX}probe_crosswalk.csv", noncal)
    horizon_results(noncal, prior_horizon, changed)
    print(f"Upgraded {len(records)} v1 classifications; reclassified {len(selected)} fallback records from raw evidence.")


def main():
    if sys.argv[1:] == ["--upgrade-v1"]:
        upgrade_v1()
        return
    if sys.argv[1:] == ["--summarize-existing"]:
        with (base.OUT / f"{PREFIX}probe_crosswalk.csv").open(encoding="utf-8-sig", newline="") as f:
            records = list(csv.DictReader(f))
        for row in records:
            for key in ("tier", "z", "requested_length", "predicted_length"):
                row[key] = int(row[key])
            for key in ("valid", "requested_exact", "full_reference_exact", "label_changed", "fallback",
                        "transition_first", "length_mismatch", "modulo_equivalent_noncanonical"):
                row[key] = base.truth(row[key])
        noncal, _ = aggregate(records)
        base.write_csv(base.OUT / f"{PREFIX}probe_crosswalk.csv", noncal)
        print(f"Summarized {len(noncal)} previously classified non-calibration observations", flush=True)
        return
    if sys.argv[1:]:
        raise ValueError("Usage: requested_taxonomy_audit.py [--summarize-existing|--upgrade-v1]")
    with base.PROBE_FILE.open(encoding="utf-8-sig", newline="") as f:
        source = [r for r in csv.DictReader(f) if r["model"] in base.PUBLICATION_MODELS]
    by_file = defaultdict(list)
    for row in source:
        by_file[row["source_file"]].append(row)
    namespace = base.checkpoint_namespace(sorted(by_file)) + "|" + base.file_sha256(Path(__file__))
    store = base.CheckpointStore(base.OUT / ".reviewer_response_cache" / "requested_taxonomy.sqlite3", namespace)
    try:
        records = []
        for index, (filename, rows) in enumerate(sorted(by_file.items()), 1):
            print(f"[classify {index}/{len(by_file)}] {filename}", flush=True)
            cached = store.get(filename)
            if cached is None:
                cached = classify_file(filename, rows)
                store.put(filename, cached)
                store.flush()
            else:
                print(f"  resumed {len(cached)} classifications", flush=True)
            records.extend(cached)
        noncal, _ = aggregate(records)
        base.write_csv(base.OUT / f"{PREFIX}probe_crosswalk.csv", noncal)
        horizon_results(noncal)
        print(f"Completed {len(noncal)} non-calibration observations", flush=True)
    finally:
        store.close()


if __name__ == "__main__":
    main()
