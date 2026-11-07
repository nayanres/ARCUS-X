"""Counterfactual post-hoc scoring against z coordinates, including the start.

The original raw responses and references are not changed. Historical prompts
were not preserved; application of this policy to them is an assumption.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "audit" / "correction_2026_11"
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SOURCE))

import post_run_analyzer as post_run
from arcus.evaluation.metrics import (
    ARCUSRobustnessIndex, GenerationBloatIndex, GenerationEfficiency, mean,
)
from arcus.evaluation.parser import estimated_optimal_path_length
from audit import analyze_artifacts as audit
import finalize_forensic_audit as final
import reproduce_correction as correction


def rescore_entry(entry):
    z = int(entry["z"])
    if z < 1:
        raise ValueError(f"Coordinate request must be positive, got {z}")
    full_reference = post_run._parse_coord_block(entry.get("ground_truth", ""))
    actions = post_run._parse_actions(entry.get("actions", ""))
    if len(full_reference) != z + 1 or len(actions) != z:
        raise ValueError(f"Unexpected reference/action lengths at z={z}: "
                         f"{len(full_reference)}, {len(actions)}")
    adjusted = dict(entry)
    adjusted["ground_truth"] = "\n".join(full_reference[:z])
    adjusted["actions"] = repr(actions[:z - 1])
    captured = []
    classify = post_run.classify_from_result

    def capture(*args, **kwargs):
        result = classify(*args, **kwargs)
        captured.append(result)
        return result

    post_run.classify_from_result = capture
    try:
        scored = ORIGINAL_SCORE(adjusted)
    finally:
        post_run.classify_from_result = classify
    if len(captured) != 1:
        raise RuntimeError("Expected one classifier result")

    predicted = scored["predicted_path"]
    reference = full_reference[:z]
    independent = audit._independent_metrics(predicted, reference, z - 1)
    independent["horizon_compliance"] = min(len(predicted), z) / max(len(predicted), z)
    independent["ground_truth_length_matches_z"] = len(reference) == z
    scored["horizon_compliance"] = independent["horizon_compliance"]
    allowance = estimated_optimal_path_length(z - 1)
    scored["generation_bloat_index"] = GenerationBloatIndex.calculate(
        scored["output_tokens"], allowance
    )
    scored["generation_efficiency"] = GenerationEfficiency.calculate(
        scored["generation_bloat_index"]
    )
    scored["_rules"] = post_run._parse_transition_rules(adjusted.get("transition_rules", ""))
    scored["_actions"] = actions[:z - 1]
    detail = audit._row_values(scored, captured[0], independent)
    detail.pop("probe_classification", None)
    detail.update(independent)
    detail["predicted_length"] = len(predicted)
    detail["ground_truth_length"] = len(reference)
    detail["ground_truth_path"] = json.dumps(reference)
    detail["actions"] = json.dumps(actions[:z - 1])
    detail["strict_fully_correct"] = detail["valid"] and independent["exact_match"]
    detail["analysis_A_taxonomy"] = detail["reported_taxonomy"]
    detail["coordinate_interpretation"] = correction._canonical_status({
        **scored, "predicted_path": json.dumps(predicted),
        "ground_truth_path": json.dumps(reference),
    })
    return scored, detail


ORIGINAL_SCORE = post_run._score_probe


def write(path, rows):
    if not rows:
        raise ValueError(f"No records to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main():
    source_rows = final.read_csv(SOURCE / "analysis_A_probe_level.csv")
    by_file = defaultdict(list)
    for row in source_rows:
        by_file[row["source_file"]].append(row)
    all_rows = []
    metrics_rows = []
    seed_rows = []
    input_hashes = []
    for filename, original_rows in sorted(by_file.items()):
        raw_path = ROOT / "outputs" / filename
        details = []

        def score(entry):
            scored, detail = rescore_entry(entry)
            details.append(detail)
            return scored

        post_run._score_probe = score
        try:
            data = post_run.process_txt_file(str(raw_path))
        finally:
            post_run._score_probe = ORIGINAL_SCORE
        matrix = data["results_matrix"]
        if not len(matrix) == len(details) == len(original_rows):
            raise RuntimeError(f"Entry count mismatch for {filename}")
        for index, ((key, scored), old, detail) in enumerate(
                zip(matrix.items(), original_rows, details), start=1):
            if old["matrix_key"] != key or int(old["source_row"]) != index:
                raise RuntimeError(f"Probe identity mismatch: {filename} row {index}")
            if json.loads(old["predicted_path"]) != scored["predicted_path"]:
                raise RuntimeError(f"Parsed response mismatch: {filename} row {index}")
            if bool(detail["valid"]) != final.truth(old["valid"]):
                raise RuntimeError(f"Validity changed unexpectedly: {filename} row {index}")
            all_rows.append({**old, **detail})
        filtered = {key: value for key, value in matrix.items()
                    if not final.is_calibration_anchor(value)}
        aggregate = post_run._compute_aggregated_metrics(filtered, data["seeds"], None)
        cri = aggregate["cri"]
        valid = sum(detail["valid"] for old, detail in zip(original_rows, details)
                    if not final.is_calibration_anchor(old))
        row = {
            "model": data["model"], "raw_log": filename,
            "logged_observations": len(matrix),
            "calibration_observations_excluded": len(matrix) - len(filtered),
            "analysis_observations": len(filtered), "valid": valid,
            "invalid": len(filtered) - valid, "validity": valid / len(filtered),
            "step_accuracy_all_pct": aggregate["step_accuracy"],
            "exact_match_all_pct": aggregate["exact_match"],
            "continuity_all_pct": aggregate["continuity"],
            "horizon_compliance_all_pct": aggregate["horizon_compliance"],
            "generation_bloat_index": aggregate["generation_bloat_index"],
            "generation_efficiency_pct": aggregate["efficiency"],
            "correct_steps_per_second": aggregate["latency_diagnostics"].get("mean_correct_steps_per_sec"),
            "posthoc_fracture_depth": aggregate["fracture_depth"],
            "cri": cri["cri"], "trajectory_fidelity": cri["trajectory_fidelity"],
            "horizon_robustness": cri["horizon_robustness"],
            "semantic_robustness": cri["semantic_robustness"],
            "cri_generation_efficiency": cri["generation_efficiency"],
        }
        for tier, values in sorted(aggregate["per_tier"].items()):
            row[f"tier_{tier}_step_accuracy_pct"] = round(values["accuracy"] * 100, 1)
            row[f"tier_{tier}_fracture_depth"] = values["fracture_depth"]
        metrics_rows.append(row)
        for seed in data["seeds"]:
            group = {key: value for key, value in filtered.items() if value["seed"] == seed}
            if group:
                seed_cri = post_run._compute_cri(
                    group, ARCUSRobustnessIndex, mean
                )
                seed_rows.append({"model": data["model"], "seed": seed,
                                  "observations": len(group), "cri": seed_cri["cri"]})
        input_hashes.append({"file": filename, "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest()})
        print(f"{data['model']}: {len(matrix)} probes rescored", flush=True)

    write(OUT / "analysis_A_probe_level.csv", all_rows)
    write(OUT / "post_run_table_metrics_publication.csv", metrics_rows)
    write(OUT / "manuscript_seed_cri.csv", seed_rows)
    write(OUT / "raw_input_sha256.csv", input_hashes)
    final.OUT = OUT
    final.output_model_summaries(all_rows)
    final.output_probe_crosswalk(all_rows, source_rows)
    final.output_taxonomy_cross_tabs(all_rows, source_rows)
    final.output_state_tracking(all_rows)
    correction.OUT = OUT
    correction._analysis_b(all_rows)
    final.output_taxonomy_sensitivity(all_rows)
    final.output_horizon_distribution(all_rows)
    final.output_tier_task_matching(all_rows)
    final.output_matched_horizon_metrics(all_rows)
    final.output_matched_pairs(all_rows)
    print(f"Completed {len(all_rows)} probes.", flush=True)


if __name__ == "__main__":
    main()
