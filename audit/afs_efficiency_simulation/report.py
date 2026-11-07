#!/usr/bin/env python3
"""Generate a concise report and appendix from audited frozen outputs."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from audit.afs_efficiency_simulation import simulation as sim

OUT = sim.HERE / "test"


def load(name):
    with (OUT / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def pct(value):
    return "N/A" if value in ("", None) else f"{100 * float(value):.1f}%"


def number(value):
    return "N/A" if value in ("", None) else f"{float(value):.2f}"


def tex_pct(value):
    return pct(value).replace("%", r"\%")


def row(summary, family, budget, policy, metric):
    return next(r for r in summary if r["family"] == family and r["budget"] == str(budget)
                and r["policy"] == policy and r["metric"] == metric)


def main():
    summary, actual, noise = load("statistical_summary.csv"), load("actual_cost_summary.csv"), load("noise_stress_summary.csv")
    thresholds = load("threshold_stress_summary.csv")
    budget_rows = [r for r in load("results.csv") if r["budget"] == "81"]
    historical_path = sim.HERE.parent / "correction_2026_11" / "reviewer_response_horizon_metrics.csv"
    with historical_path.open(encoding="utf-8", newline="") as stream:
        historical = list(csv.DictReader(stream))
    historical_example = [
        next(r for r in historical if r["model"] == "mistral.devstral-2-123b"
             and r["tier"] == "3" and r["horizon_z"] == str(z))
        for z in (30, 40)
    ]
    policies = sim.POLICIES
    names = {"production_afs": "Production AFS", "fixed_grid": "Fixed grid",
             "sequential": "Sequential", "uniform_random": "Uniform random", "adaptive": "Adaptive alternative"}
    lines = [
        "Synthetic AFS efficiency benchmark",
        "==================================",
        "Independent final set: 1,800 curves (200 in each of nine families), two noise realizations per curve.",
        "Every policy pays for one ignored calibration response and five per new batch.",
        "The fixed ten-step grid has 16 points and stops at 81 responses, even at the 161-response budget.",
        "The test set contains equal numbers of curves per family; conditioning on crossing or detection",
        "changes their effective weights. No conclusion is an empirical LLM efficiency result.",
        "A no-observed-crossing outcome is called correctly censored only when the generating curve has no crossing;",
        "it is not evidence that a policy established nonexistence of a crossing.",
        "Detected-only MAE and cost-to-target averages exclude failures; precision rates include failures.",
        "All pairwise differences below are policy minus production AFS; CIs resample curve blocks (500 replicates).",
        "",
        "Pooled results by budget",
    ]
    for budget in sim.CONFIG["budgets_responses"]:
        lines.append(f"\nBudget {budget} responses")
        for policy in policies:
            hit = row(summary, "pooled", budget, policy, "within_5_crossing")
            detection = row(summary, "pooled", budget, policy, "detection_crossing")
            mae = row(summary, "pooled", budget, policy, "first_abs_error_detected")
            false = row(summary, "pooled", budget, policy, "false_positive_censored")
            cost = row(summary, "pooled", budget, policy, "cost")
            cost5 = row(summary, "pooled", budget, policy, "cost_to_5_success")
            ever5 = row(summary, "pooled", budget, policy, "ever_within_5_crossing")
            lines.append(
                f"  {names[policy]}: within5={pct(hit['value'])}, detected={pct(detection['value'])}, "
                f"detected MAE={number(mae['value'])}, false positive={pct(false['value'])}, "
                f"mean cost={number(cost['value'])}, ever within5={pct(ever5['value'])}, "
                f"cost-to-5 among successes={number(cost5['value'])}, "
                f"within5 difference vs AFS={pct(hit['paired_difference_vs_afs'])} "
                f"[{pct(hit['paired_ci95_lower'])}, {pct(hit['paired_ci95_upper'])}]; "
                f"paired curves={hit['paired_curves']}."
            )
    lines.extend(["", "Family-level within-five rates and paired differences at 81 responses"])
    for family in sim.CONFIG["families"]:
        lines.append(f"  {family}:")
        for policy in policies:
            hit = row(summary, family, 81, policy, "within_5_crossing")
            lines.append(f"    {names[policy]} {pct(hit['value'])}; difference {pct(hit['paired_difference_vs_afs'])} "
                         f"[{pct(hit['paired_ci95_lower'])}, {pct(hit['paired_ci95_upper'])}], "
                         f"paired curves={hit['paired_curves']}.")
    lines.extend(["", "Five-consecutive-horizon persistent-target detection at 81 responses"])
    for family in ("recovery", "oscillatory"):
        for policy in policies:
            eligible = [r for r in budget_rows if r["family"] == family and r["policy"] == policy
                        and r["persistent_truth"]]
            detected = sum(bool(r["persistent_estimate"]) for r in eligible)
            lines.append(f"  {family}, {names[policy]}: {detected}/{len(eligible)} detected "
                         "(estimator requires five consecutive sampled valid below-threshold horizons).")
    lines.extend(["", "Baselines compared at the actual cost consumed by production AFS with a 161-response cap"])
    for policy in policies:
        match = next(r for r in actual if r["family"] == "pooled" and r["comparison"] == "afs_consumed_cost"
                     and r["policy"] == policy and r["metric"] == "within_5_crossing")
        lines.append(f"  {names[policy]}: {pct(match['value'])}; difference vs AFS {pct(match['paired_difference_vs_afs'])} "
                     f"[{pct(match['paired_ci95_lower'])}, {pct(match['paired_ci95_upper'])}].")
    lines.extend(["", "Invalid-only search batches at 81 responses (descriptive, excluding calibration)"])
    for policy in policies:
        selected = [r for r in budget_rows if r["family"] == "validity_degradation" and r["policy"] == policy]
        invalid = sum(int(r["invalid_only_batches"]) for r in selected)
        total = sum(int(r["distinct_horizons"]) for r in selected)
        lines.append(f"  {names[policy]}, validity-degradation family: {invalid}/{total} "
                     f"({pct(invalid / total if total else None)}).")
    afs_validity = [r for r in budget_rows if r["family"] == "validity_degradation"
                    and r["policy"] == "production_afs"]
    for label, selected in (
        ("0-9", [r for r in afs_validity if int(r["valid_responses"]) < 10]),
        ("50+", [r for r in afs_validity if int(r["valid_responses"]) >= 50]),
    ):
        detected = sum(bool(r["first_estimate"]) for r in selected)
        lines.append(f"  AFS with {label} valid search responses: {detected}/{len(selected)} "
                     f"observed a crossing. This is a descriptive, policy-selected association.")
    lines.extend(["", "Correlation stress: within-five rates at 81 responses (independent observation stream)"])
    for concentration in (5, 30, 200):
        vals = [next(r for r in noise if r["family"] == "pooled" and r["concentration"] == str(concentration)
                     and r["budget"] == "81" and r["policy"] == p) for p in policies]
        lines.append(f"  beta concentration {concentration}: " + "; ".join(
            f"{names[p]} {pct(v['within_5_rate'])}" for p, v in zip(policies, vals)))
        grid = vals[policies.index("fixed_grid")]
        lines.append(f"    Fixed-grid minus AFS: {pct(grid['paired_difference_vs_afs'])} "
                     f"[{pct(grid['paired_ci95_lower'])}, {pct(grid['paired_ci95_upper'])}]; "
                     f"{grid['curves']} crossing curves.")
    lines.extend(["", "Altered-threshold AFS stress at 81 responses (descriptive; not primary policy comparisons)"])
    for threshold in (0.4, 0.6):
        r = next(r for r in thresholds if r["family"] == "pooled" and r["budget"] == "81"
                 and float(r["threshold"]) == threshold)
        lines.append(f"  threshold {threshold}: detection {pct(r['detection_rate'])} among "
                     f"{r['crossing_count']} crossing realizations; detected-only MAE "
                     f"{number(r['detected_mean_abs_error'])}; false positive "
                     f"{pct(r['false_positive_rate'])} among {r['censored_count']} no-crossing realizations.")
    lines.extend([
        "",
        "Historical support sanity check (not simulation calibration or policy replay)",
        "  Existing valid-only requested-output summaries for mistral.devstral-2-123b tier 3:",
    ])
    for r in historical_example:
        lines.append(
            f"  z={r['horizon_z']}: {r['valid_observations']} valid, {r['invalid_observations']} invalid, "
            f"mean Step Accuracy {pct(r['mean_step_accuracy'])} ({r['accuracy_denominator']})."
        )
    lines.append(
        "  These are observed, potentially unpaired task mixtures and use z-1 scored transitions; "
        "they do not establish an individual-task recovery, calibrate the simulator, or predict unobserved horizons."
    )
    pooled_afs = row(summary, "pooled", 81, "production_afs", "within_5_crossing")
    pooled_grid = row(summary, "pooled", 81, "fixed_grid", "within_5_crossing")
    pooled_alternative = row(summary, "pooled", 81, "adaptive", "within_5_crossing")
    recovery_afs = row(summary, "recovery", 81, "production_afs", "within_5_crossing")
    recovery_grid = row(summary, "recovery", 81, "fixed_grid", "within_5_crossing")
    oscillatory_afs = row(summary, "oscillatory", 81, "production_afs", "within_5_crossing")
    oscillatory_grid = row(summary, "oscillatory", 81, "fixed_grid", "within_5_crossing")
    persistent_afs = [r for r in budget_rows if r["policy"] == "production_afs" and r["persistent_truth"]]
    persistent_detected = sum(bool(r["persistent_estimate"]) for r in persistent_afs)
    lines.extend([
        "",
        "Interpretation",
        f"  At 81 responses the pooled AFS within-five rate is {pct(pooled_afs['value'])}, versus "
        f"{pct(pooled_grid['value'])} for the fixed grid and {pct(pooled_alternative['value'])} "
        "for the adaptive alternative. The paired intervals above quantify the differences.",
        f"  At 41 responses, the alternative is {pct(row(summary, 'pooled', 41, 'adaptive', 'within_5_crossing')['value'])} "
        f"versus AFS {pct(row(summary, 'pooled', 41, 'production_afs', 'within_5_crossing')['value'])}; "
        "consult the paired interval rather than extrapolating from one budget.",
        f"  Nonmonotonic results differ by family: AFS versus fixed grid is "
        f"{pct(recovery_afs['value'])} versus {pct(recovery_grid['value'])} for recoveries, "
        f"but {pct(oscillatory_afs['value'])} versus {pct(oscillatory_grid['value'])} for oscillations.",
        f"  At 81 responses the AFS persistent estimator detected {persistent_detected}/"
        f"{len(persistent_afs)} realizations with a true persistent crossing; it requires five consecutive sampled valid "
        "below-threshold horizons. Valid-response-bin comparisons are descriptive.",
        "  Budget dependence, the equally sized generating families, the fixed grid's finite schedule, early AFS "
        "termination, the response correlation model and the separate actual-cost results limit "
        "generalization. No controlled simulation result establishes empirical LLM efficiency.",
    ])
    lines.extend([
        "",
        "Full family/budget estimates, paired confidence intervals, missing counts, actual-cost comparisons,",
        "noise/validity/parameter factors, persistence and threshold stress results are in the CSV artifacts.",
        "The beta-binomial model does not reproduce complete state trajectories, causal model failures,",
        "provider behavior, adaptive task selection, or natural-language responses. The simulation assesses",
        "allocation under its stated response-generating assumptions, not empirical LLM sample efficiency.",
        "Detection and precision can be misleading on nonmonotone curves; examine first versus persistent",
        "targets and their failure rates. Differing valid sample counts condition MAE and its paired interval.",
    ])
    (sim.HERE / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    table = []
    for policy in policies:
        hit = row(summary, "pooled", 81, policy, "within_5_crossing")
        detect = row(summary, "pooled", 81, policy, "detection_crossing")
        false = row(summary, "pooled", 81, policy, "false_positive_censored")
        mae = row(summary, "pooled", 81, policy, "first_abs_error_detected")
        interval = f"{tex_pct(hit['value'])} [{tex_pct(hit['ci95_lower'])}, {tex_pct(hit['ci95_upper'])}]"
        table.append(f"{names[policy]} & {interval} & "
                     f"{tex_pct(detect['value'])} & {number(mae['value'])} & "
                     f"{tex_pct(false['value'])} \\\\")
    appendix = r"""\subsection{Synthetic Sampling-Efficiency Evaluation}
\label{app:afs_simulation}
We compared the production AFS controller with fixed-grid, sequential, uniform-random, and threshold-directed adaptive sampling on synthetic accuracy and validity curves, without model inference. The test set contains 200 independently parameterized curves in each of nine families and two noise realizations per curve. Each policy pays for one calibration response and five responses per sampled horizon; invalid responses consume budget but are excluded from batch accuracy. Valid probe scores follow a beta-binomial model of the $z$ transition positions. The estimand is the first integer horizon with conditional accuracy below 50\%; a common estimator uses the earliest sampled valid batch below that threshold. A five-consecutive-horizon target and alternative thresholds are examined separately. The complete generating curve is available only to the evaluator.

\begin{table}[H]
\centering
\caption{Synthetic first-crossing performance at an 81-response budget. Within-five and detection rates are conditional on a true crossing; brackets give curve-clustered 95\% intervals. Mean absolute error (MAE) is conditional on detection. False-positive rates are conditional on no true crossing.}
\label{tab:afs_simulation}
\scriptsize
\begin{tabular}{lrrrr}
\toprule
Policy & Within five & Detected & MAE & False positive \\
\midrule
""" + "\n".join(table) + r"""
\bottomrule
\end{tabular}
\end{table}

""" + (
        f"At this budget, AFS localized {tex_pct(pooled_afs['value'])} of pooled crossings within five "
        f"horizons, compared with {tex_pct(pooled_grid['value'])} for the fixed grid "
        f"(paired difference {100 * -float(pooled_grid['paired_difference_vs_afs']):.1f} percentage points, "
        f"95\\% CI [{100 * -float(pooled_grid['paired_ci95_upper']):.1f}, "
        f"{100 * -float(pooled_grid['paired_ci95_lower']):.1f}]). In the recovery "
        f"family the respective rates were {tex_pct(recovery_afs['value'])} and {tex_pct(recovery_grid['value'])}; "
        f"in the oscillatory family they were {tex_pct(oscillatory_afs['value'])} and {tex_pct(oscillatory_grid['value'])}. "
        f"The persistent estimator detected {persistent_detected} of {len(persistent_afs)} realizations with a true "
        r"persistent crossing under AFS at this budget; it requires five consecutively sampled valid below-threshold horizons. "
    ) + r"""These are synthetic allocation results, not estimates of empirical LLM sample efficiency. The pooled rates depend on the chosen family mixture and response model; AFS may terminate with unused budget. Curve-clustered paired intervals, actual-cost comparisons, persistent-crossing results, validity and correlation stress tests, and family-specific outcomes are reported in the supplementary reproducibility artifacts. Lack of an observed crossing is not proof of no fracture.
"""
    (sim.HERE / "suggested_appendix.tex").write_text(appendix, encoding="utf-8")
    print("Wrote report.txt and suggested_appendix.tex.")


if __name__ == "__main__":
    main()
