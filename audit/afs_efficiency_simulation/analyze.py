#!/usr/bin/env python3
"""Audit and summarize the frozen synthetic AFS experiment."""

import csv
import json
import math
import sys
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from audit.afs_efficiency_simulation import simulation as sim

OUT = sim.HERE / "test"


def load(name):
    with (OUT / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def value(row, key):
    return float(row[key]) if row[key] not in ("", None) else None


def observations(row):
    truth = value(row, "first_truth")
    estimate = value(row, "first_estimate")
    crossing = truth is not None
    detected = estimate is not None
    cost3 = value(row, "cost_to_first_within_3")
    cost5 = value(row, "cost_to_first_within_5")
    error = value(row, "first_abs_error")
    return {
        "cost": value(row, "cost"),
        "first_abs_error_detected": value(row, "first_abs_error"),
        "persistent_abs_error_detected": value(row, "persistent_abs_error"),
        "detection_crossing": float(detected) if crossing else None,
        "false_positive_censored": float(detected) if not crossing else None,
        "correct_censor_censored": float(not detected) if not crossing else None,
        "no_estimate": float(not detected),
        "within_3_crossing": float(error is not None and error <= 3) if crossing else None,
        "within_5_crossing": float(error is not None and error <= 5) if crossing else None,
        "ever_within_3_crossing": float(cost3 is not None) if crossing else None,
        "ever_within_5_crossing": float(cost5 is not None) if crossing else None,
        "cost_to_3_success": cost3,
        "cost_to_5_success": cost5,
        "region_coverage": value(row, "region_coverage"),
        "region_precision": value(row, "region_precision"),
        "invalid_only_batches": value(row, "invalid_only_batches"),
        "sampled_horizons": value(row, "distinct_horizons"),
    }


def bootstrap_ratio(numerator, denominator, picks):
    n = numerator[picks].sum(axis=1)
    d = denominator[picks].sum(axis=1)
    return np.divide(n, d, out=np.full_like(n, np.nan, dtype=float), where=d > 0)


def percentile(data):
    data = data[np.isfinite(data)]
    return (float(np.quantile(data, 0.025)), float(np.quantile(data, 0.975))) if len(data) else ("", "")


def summarize(rows, filename, group_key="budget"):
    lookup = {(r["curve_id"], r["replicate"], r["policy"], r[group_key]): r for r in rows}
    groups = defaultdict(list)
    for r in rows:
        if r["policy"] == "production_afs":
            groups[(r["family"], r[group_key])].append(r)
            groups[("pooled", r[group_key])].append(r)
    summary = []
    for (family, budget), baseline_rows in groups.items():
        ids = sorted({r["curve_id"] for r in baseline_rows})
        index = {x: i for i, x in enumerate(ids)}
        rng = np.random.default_rng(sim.stable_seed(sim.CONFIG["bootstrap_seed"], family, budget, filename))
        picks = rng.integers(0, len(ids), size=(sim.CONFIG["bootstrap_replicates"], len(ids)))
        for policy in sim.POLICIES:
            data = [(r, lookup[(r["curve_id"], r["replicate"], policy, r[group_key])]) for r in baseline_rows]
            fields = observations(data[0][1])
            for metric in fields:
                num, den, paired_base, paired_other, paired_den = (np.zeros(len(ids)) for _ in range(5))
                for base, row in data:
                    i = index[row["curve_id"]]
                    b, x = observations(base)[metric], observations(row)[metric]
                    if x is not None:
                        num[i] += x
                        den[i] += 1
                    if x is not None and b is not None:
                        paired_base[i] += b
                        paired_other[i] += x
                        paired_den[i] += 1
                estimate = num.sum() / den.sum() if den.sum() else None
                ci = percentile(bootstrap_ratio(num, den, picks))
                pair_base = bootstrap_ratio(paired_base, paired_den, picks)
                pair_other = bootstrap_ratio(paired_other, paired_den, picks)
                diff = ((paired_other.sum() - paired_base.sum()) / paired_den.sum()) if paired_den.sum() else None
                dci = percentile(pair_other - pair_base)
                summary.append({
                    "family": family, group_key: budget, "policy": policy, "metric": metric,
                    "value": estimate, "ci95_lower": ci[0], "ci95_upper": ci[1],
                    "value_sum": float(num.sum()), "observations": int(den.sum()),
                    "curves": int((den > 0).sum()),
                    "paired_difference_vs_afs": diff, "paired_ci95_lower": dci[0],
                    "paired_ci95_upper": dci[1], "paired_observations": int(paired_den.sum()),
                    "paired_curves": int((paired_den > 0).sum()),
                })
    sim.write_rows(OUT / filename, summary)
    return summary


def audit(rows, curves, traces):
    expected = len(sim.CONFIG["families"]) * sim.CONFIG["test_curves_per_family"]
    if len(curves) != expected or len(rows) != expected * sim.CONFIG["replicates_per_curve"] * len(sim.POLICIES) * len(sim.CONFIG["budgets_responses"]):
        raise ValueError("Unexpected final experiment size")
    p = {r["curve_id"]: np.array(json.loads(r["accuracy_curve"])) for r in curves}
    by_run = defaultdict(list)
    for r in traces:
        by_run[(r["curve_id"], r["replicate"], r["policy"])].append({
            "z": int(r["z"]), "mean": value(r, "mean"), "valid": int(r["valid"]), "cost": int(r["cost"])
        })
    for key, trace in by_run.items():
        if len({r["z"] for r in trace}) != len(trace):
            raise ValueError(f"Unexpected repeated charged horizon: {key}")
        if any(r["cost"] != 1 + 5 * (i + 1) or not 1 <= r["z"] <= 150 for i, r in enumerate(trace)):
            raise ValueError(f"Invalid cost or horizon: {key}")
    for r in rows:
        budget, cost = int(r["budget"]), int(r["cost"])
        if cost > budget or int(r["unused_budget"]) != budget - cost:
            raise ValueError("Budget accounting failed")
        key = (r["curve_id"], r["replicate"], r["policy"])
        selected = [x for x in by_run[key] if x["cost"] <= budget]
        expected_result = sim.evaluate(selected, p[r["curve_id"]], budget, False, None)
        for field in ("first_truth", "first_estimate", "persistent_truth", "persistent_estimate",
                      "first_abs_error", "persistent_abs_error", "region_coverage", "region_precision",
                      "invalid_only_batches", "distinct_horizons", "cost_to_first_within_3", "cost_to_first_within_5"):
            actual, expected_value = value(r, field), expected_result[field]
            if actual != expected_value:
                raise ValueError(f"Recomputed result mismatch {key} {field}: {actual}, {expected_value}")
    return p, by_run


def actual_cost_rows(rows, p, traces):
    full = [r for r in rows if r["budget"] == str(max(sim.CONFIG["budgets_responses"]))]
    afs = {(r["curve_id"], r["replicate"]): r for r in full if r["policy"] == "production_afs"}
    output = []
    for r in full:
        base = afs[(r["curve_id"], r["replicate"])]
        budget = int(base["cost"])
        key = (r["curve_id"], r["replicate"], r["policy"])
        selected = [x for x in traces[key] if x["cost"] <= budget]
        cost = selected[-1]["cost"] if selected else 1
        result = {"curve_id": r["curve_id"], "family": r["family"], "replicate": r["replicate"],
                  "policy": r["policy"], "comparison": "afs_consumed_cost", "budget": budget,
                  "cost": cost, **sim.evaluate(selected, p[r["curve_id"]], budget, False, None)}
        output.append(result)
    sim.write_rows(OUT / "actual_cost_results.csv", output)
    return output


def sensitivity(rows, curves):
    params = {r["curve_id"]: json.loads(r["parameters"]) for r in curves}
    output = []
    groups = defaultdict(list)
    bounds = {
        "scale": [5, 10, 20, 30], "slope": [-0.0002, 0, 0.0002, 0.001, 0.002, 0.003],
        "center": [30, 60, 90, 120], "fracture": [30, 60, 90, 120],
        "crossing": [30, 60, 90, 120], "period": [20, 40, 60, 100],
        "validity_center": [30, 60, 90, 120],
    }
    for r in rows:
        if r["budget"] != str(max(sim.CONFIG["budgets_responses"])):
            continue
        par = params[r["curve_id"]]
        metrics = observations(r)
        groups[(r["family"], r["policy"], "valid_responses", min(int(r["valid_responses"]) // 10 * 10, 50))].append(metrics)
        for field in ("scale", "slope", "center", "fracture", "crossing", "period", "validity_center"):
            if field in par:
                edges = [0.45, 0.5, 0.55] if field == "center" and r["family"] in ("oscillatory", "near_threshold") else bounds[field]
                index = bisect_right(edges, par[field])
                lo = str(edges[index - 1]) if index else "-inf"
                hi = str(edges[index]) if index < len(edges) else "inf"
                groups[(r["family"], r["policy"], field, f"[{lo},{hi})")].append(metrics)
    for (family, policy, factor, level), samples in groups.items():
        numeric = [x["first_abs_error_detected"] for x in samples if x["first_abs_error_detected"] is not None]
        hit = [x["within_5_crossing"] for x in samples if x["within_5_crossing"] is not None]
        output.append({"family": family, "policy": policy, "factor": factor, "level": level,
                       "replicates": len(samples), "detected_mae": float(np.mean(numeric)) if numeric else "",
                       "within5_rate": float(np.mean(hit)) if hit else ""})
    sim.write_rows(OUT / "stress_factors.csv", output)


def threshold_and_persistence(rows):
    output = []
    for family in (*sim.CONFIG["families"], "pooled"):
        for policy in sim.POLICIES:
            for budget in sim.CONFIG["budgets_responses"]:
                subset = [r for r in rows if (family == "pooled" or r["family"] == family)
                          and r["policy"] == policy and int(r["budget"]) == budget]
                for window in (3, 10):
                    eligible = [r for r in subset if value(r, f"truth_persistent_{window}") is not None]
                    detected = [r for r in eligible if value(r, f"estimate_persistent_{window}") is not None]
                    output.append({"family": family, "policy": policy, "budget": budget,
                                   "scenario": f"persistence_{window}", "eligible": len(eligible),
                                   "detected": len(detected), "detection_rate": len(detected) / len(eligible) if eligible else "",
                                   "mean_abs_error": np.mean([abs(value(r, f"truth_persistent_{window}") - value(r, f"estimate_persistent_{window}")) for r in detected]) if detected else ""})
    sim.write_rows(OUT / "persistence_sensitivity.csv", output)
    threshold = load("threshold_stress.csv")
    threshold_output = []
    groups = defaultdict(list)
    for r in threshold:
        groups[(r["family"], r["threshold"], r["budget"])].append(r)
        groups[("pooled", r["threshold"], r["budget"])].append(r)
    for (family, t, budget), group in groups.items():
        eligible = [r for r in group if value(r, "truth") is not None]
        detected = [r for r in eligible if value(r, "estimate") is not None]
        no_crossing = [r for r in group if value(r, "truth") is None]
        threshold_output.append({
            "family": family, "threshold": t, "budget": budget, "crossing_count": len(eligible),
            "detection_rate": len(detected) / len(eligible) if eligible else "",
            "detected_mean_abs_error": np.mean([value(r, "absolute_error") for r in detected]) if detected else "",
            "censored_count": len(no_crossing),
            "false_positive_rate": sum(value(r, "estimate") is not None for r in no_crossing) / len(no_crossing) if no_crossing else "",
        })
    sim.write_rows(OUT / "threshold_stress_summary.csv", threshold_output)


def noise_summary():
    rows = load("noise_stress.csv")
    lookup = {(r["curve_id"], r["replicate"], r["concentration"], r["budget"], r["policy"]): r for r in rows}
    groups = defaultdict(list)
    for r in rows:
        if r["policy"] == "production_afs":
            groups[(r["family"], r["concentration"], r["budget"])].append(r)
            groups[("pooled", r["concentration"], r["budget"])].append(r)
    output = []
    for (family, concentration, budget), group in groups.items():
        ids = sorted({r["curve_id"] for r in group})
        index = {x: i for i, x in enumerate(ids)}
        rng = np.random.default_rng(sim.stable_seed(sim.CONFIG["bootstrap_seed"], "noise", family, concentration, budget))
        picks = rng.integers(0, len(ids), size=(sim.CONFIG["bootstrap_replicates"], len(ids)))
        for policy in sim.POLICIES:
            bnum = np.zeros(len(ids))
            pnum = np.zeros(len(ids))
            den = np.zeros(len(ids))
            for base in group:
                row = lookup[(base["curve_id"], base["replicate"], concentration, budget, policy)]
                b = (float(value(base, "first_abs_error") is not None and value(base, "first_abs_error") <= 5)
                     if value(base, "first_truth") is not None else None)
                other = (float(value(row, "first_abs_error") is not None and value(row, "first_abs_error") <= 5)
                         if value(row, "first_truth") is not None else None)
                if b is not None and other is not None:
                    i = index[row["curve_id"]]
                    bnum[i] += b
                    pnum[i] += other
                    den[i] += 1
            rate = pnum.sum() / den.sum() if den.sum() else None
            delta = (pnum.sum() - bnum.sum()) / den.sum() if den.sum() else None
            interval = percentile(bootstrap_ratio(pnum, den, picks) - bootstrap_ratio(bnum, den, picks))
            output.append({"family": family, "concentration": concentration, "budget": budget,
                           "policy": policy, "within_5_rate": rate, "paired_difference_vs_afs": delta,
                           "paired_ci95_lower": interval[0], "paired_ci95_upper": interval[1],
                           "observations": int(den.sum()), "curves": int((den > 0).sum())})
    sim.write_rows(OUT / "noise_stress_summary.csv", output)


def operational_comparison(rows, traces):
    output = []
    for r in rows:
        selected = [x for x in traces[(r["curve_id"], r["replicate"], r["policy"])]
                    if x["cost"] <= int(r["budget"])]
        if r["policy"] == "production_afs":
            native = value(r, "native_output")
            state = "not_terminated" if native is None else (
                "right_censored_ceiling" if native == 150 and not any(
                    x["mean"] is not None and x["mean"] < sim.THRESHOLD for x in selected
                ) else ("no_candidate" if native == 0 else "candidate")
            )
            estimator = native if state == "candidate" else None
        else:
            estimator = value(r, "first_estimate")
            state = "observed_candidate" if estimator is not None else "no_observed_candidate"
        truth = value(r, "first_truth")
        output.append({
            "curve_id": r["curve_id"], "family": r["family"], "replicate": r["replicate"],
            "policy": r["policy"], "budget": r["budget"], "state": state,
            "native_output": r["native_output"], "operational_estimate": estimator,
            "first_crossing_truth": truth,
            "absolute_error": abs(estimator - truth) if estimator is not None and truth is not None else "",
        })
    sim.write_rows(OUT / "operational_comparison.csv", output)
    lookup = {(r["curve_id"], r["replicate"], r["policy"], r["budget"]): r for r in output}
    groups = defaultdict(list)
    for r in output:
        if r["policy"] == "production_afs":
            groups[(r["family"], r["budget"])].append(r)
            groups[("pooled", r["budget"])].append(r)
    summary = []
    for (family, budget), group in groups.items():
        ids = sorted({r["curve_id"] for r in group})
        index = {x: i for i, x in enumerate(ids)}
        rng = np.random.default_rng(sim.stable_seed(sim.CONFIG["bootstrap_seed"], "operational", family, budget))
        picks = rng.integers(0, len(ids), size=(sim.CONFIG["bootstrap_replicates"], len(ids)))

        def metric(row, name):
            truth, est = value(row, "first_crossing_truth"), value(row, "operational_estimate")
            if name == "detected_mae":
                return abs(truth - est) if truth is not None and est is not None else None
            if name == "within_5":
                return float(est is not None and abs(truth - est) <= 5) if truth is not None else None
            if name == "detection":
                return float(est is not None) if truth is not None else None
            return float(est is not None) if truth is None else None

        for policy in sim.POLICIES:
            for name in ("detected_mae", "within_5", "detection", "false_positive"):
                num, den, paired_base, paired_other, paired_den = (np.zeros(len(ids)) for _ in range(5))
                for base in group:
                    row = lookup[(base["curve_id"], base["replicate"], policy, budget)]
                    i = index[row["curve_id"]]
                    b, x = metric(base, name), metric(row, name)
                    if x is not None:
                        num[i] += x
                        den[i] += 1
                    if x is not None and b is not None:
                        paired_base[i] += b
                        paired_other[i] += x
                        paired_den[i] += 1
                ci = percentile(bootstrap_ratio(num, den, picks))
                delta_ci = percentile(bootstrap_ratio(paired_other, paired_den, picks)
                                      - bootstrap_ratio(paired_base, paired_den, picks))
                summary.append({
                    "family": family, "budget": budget, "policy": policy, "metric": name,
                    "value": num.sum() / den.sum() if den.sum() else "",
                    "ci95_lower": ci[0], "ci95_upper": ci[1], "observations": int(den.sum()),
                    "curves": int((den > 0).sum()),
                    "paired_difference_vs_afs": (paired_other.sum() - paired_base.sum()) / paired_den.sum() if paired_den.sum() else "",
                    "paired_ci95_lower": delta_ci[0], "paired_ci95_upper": delta_ci[1],
                    "paired_observations": int(paired_den.sum()), "paired_curves": int((paired_den > 0).sum()),
                })
    sim.write_rows(OUT / "operational_summary.csv", summary)


def figures(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric, ylabel, name in (
        ("first_abs_error_detected", "Mean absolute localization error (detected only)", "cost_localization.pdf"),
        ("within_5_crossing", "Crossings localized within 5 (fraction)", "cost_precision.pdf"),
        ("detection_crossing", "Crossings detected (fraction)", "detection.pdf"),
        ("false_positive_censored", "False positives among no-crossing curves", "false_positive.pdf"),
    ):
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for policy in sim.POLICIES:
            subset = [r for r in summary if r["family"] == "pooled" and r["policy"] == policy and r["metric"] == metric]
            subset.sort(key=lambda r: int(r["budget"]))
            ax.plot([int(r["budget"]) for r in subset], [r["value"] for r in subset], marker="o", label=policy)
        ax.set(xlabel="Response budget (including calibration)", ylabel=ylabel)
        if metric != "first_abs_error_detected":
            ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / name)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for policy in sim.POLICIES:
        values = [next(r["value"] for r in summary if r["family"] == family and r["policy"] == policy
                       and r["metric"] == "within_5_crossing" and int(r["budget"]) == 81)
                  for family in sim.CONFIG["families"]]
        ax.plot(range(len(values)), values, marker="o", label=policy)
    ax.set_xticks(range(len(sim.CONFIG["families"])), sim.CONFIG["families"], rotation=40, ha="right")
    ax.set(ylabel="Fraction localized within 5", ylim=(0, 1), title="By synthetic family; 81-response budget")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "family_precision.pdf")
    plt.close(fig)


def main():
    rows, curves, traces = load("results.csv"), load("curves.csv"), load("traces.csv")
    p, by_run = audit(rows, curves, traces)
    summary = summarize(rows, "statistical_summary.csv")
    actual = actual_cost_rows(rows, p, by_run)
    summarize(actual, "actual_cost_summary.csv", group_key="comparison")
    sensitivity(rows, curves)
    threshold_and_persistence(rows)
    operational_comparison(rows, by_run)
    noise_summary()
    figures(summary)
    print(f"Audited {len(curves)} curves, {len(rows)} budget outcomes, {len(traces)} batches; wrote paired cluster-bootstrap summaries.")


if __name__ == "__main__":
    main()
