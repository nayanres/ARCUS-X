"""Verify probe alignment and produce a policy-comparison report."""
from __future__ import annotations

import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / "audit" / "correction_2026_11"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(ROOT))

import finalize_forensic_audit as final


def main():
    rows = final.read_csv(OUT / "analysis_A_probe_level.csv")
    old = final.read_csv(SOURCE / "analysis_A_probe_level.csv")
    old_by_key = {(r["source_file"], r["source_row"]): r for r in old}
    if len(old_by_key) != len(old) or len(rows) != len(old):
        raise ValueError("Source probe identities are not unique or counts differ")
    transitions = Counter()
    examples = {}
    violations = []
    for r in rows:
        before = old_by_key[(r["source_file"], r["source_row"])]
        if final.truth(r["valid"]) != final.truth(before["valid"]):
            raise ValueError("Validity changed between policies")
        if final.is_calibration_anchor(r) or not final.truth(r["valid"]):
            continue
        exact = final.truth(r["exact_match"])
        if (r["reported_taxonomy"] == "None") != exact:
            violations.append((r["source_file"], r["source_row"]))
        transitions[(r["model"], before["reported_taxonomy"],
                     r["reported_taxonomy"])] += 1
        if (final.truth(before["exact_match"]) and not exact
                and r["model"] not in examples):
            examples[r["model"]] = r
    if violations:
        raise ValueError(f"None/Exact Match probe-level violations: {violations[:10]}")

    final.OUT = OUT
    final.output_probe_crosswalk(rows, old)
    final.output_taxonomy_cross_tabs(rows, old)
    breakdown = final.output_state_tracking(rows)
    final.write_csv(OUT / "taxonomy_policy_transition_counts.csv", [
        {"model": model, "old_full_transition_label": before,
         "new_coordinate_count_label": after, "valid_probe_count": count}
        for (model, before, after), count in sorted(transitions.items())
    ])
    before_metrics = {r["model"]: r for r in final.read_csv(
        SOURCE / "post_run_table_metrics_publication.csv")}
    after_metrics = {r["model"]: r for r in final.read_csv(
        OUT / "post_run_table_metrics_publication.csv")}
    comparison = []
    for model in sorted(after_metrics):
        if model not in before_metrics:
            raise ValueError(f"Missing prior metrics for {model}")
        a, b = before_metrics[model], after_metrics[model]
        result = {"model": model, "observations": b["analysis_observations"],
                  "valid": b["valid"], "invalid": b["invalid"]}
        for metric in ("step_accuracy_all_pct", "exact_match_all_pct",
                       "continuity_all_pct", "horizon_compliance_all_pct",
                       "generation_bloat_index", "generation_efficiency_pct",
                       "cri"):
            result[f"old_{metric}"] = a[metric]
            result[f"new_{metric}"] = b[metric]
        comparison.append(result)
    final.write_csv(OUT / "model_policy_comparison.csv", comparison)
    lines = [
        "# Prompt-coordinate post-hoc reanalysis",
        "",
        "The selected interpretation scores z coordinates: the initial state plus "
        "the first z-1 action results. The supplied final action is not scored. "
        "The old interpretation scored all z actions and z+1 coordinates. "
        "A z+1 output is therefore overlong under the selected interpretation.",
        "",
        "This is an interpretation of an ambiguous instruction, not a discovery "
        "of the model's intent. Current and historical prompt-builder versions "
        "request z coordinates, but the historical raw logs do not preserve exact "
        "prompt text or prove the generator revision for every run. The minitest "
        "transcripts do preserve their actual prompts. No API responses were "
        "generated; the original logs and reference trajectories were not edited.",
        "",
        f"Probe records checked: {len(rows):,}, across {len(after_metrics)} raw logs; "
        "the publication cohort contains 13 models and excludes the extra "
        "Minimax run and fixed calibration anchors. Validity is unchanged.",
        "Every valid non-calibration probe was checked individually: "
        f"`None` disagreed with Exact Match in {len(violations)} cases.",
        "",
        "## Overall scoring changes",
        "",
        "Percentages below are pooled over all non-calibration observations, "
        "including invalid responses. Taxonomy percentages use valid probes; "
        "the two denominators must not be interchanged.",
        "",
        "| Model | N | Valid | Step old → new | Exact old → new | HC old → new |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in comparison:
        lines.append(
            f"| {r['model']} | {r['observations']} | {r['valid']} | "
            f"{r['old_step_accuracy_all_pct']} → {r['new_step_accuracy_all_pct']} | "
            f"{r['old_exact_match_all_pct']} → {r['new_exact_match_all_pct']} | "
            f"{r['old_horizon_compliance_all_pct']} → {r['new_horizon_compliance_all_pct']} |"
        )
    lines += [
        "",
        "## Individual examples",
        "",
        "These valid probes matched the full z+1 reference under the old policy. "
        "Their additional coordinate makes them non-exact under the new policy; "
        "the label also depends on the classifier's precedence.",
        "",
        "| Model | Source row | Tier | z | Emitted | New label |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for model in ("moonshotai.kimi-k2.5", "GPT-5-mini", "zai.glm-5",
                  "mistral-large-2512"):
        if model in examples:
            r = examples[model]
            lines.append(f"| {model} | {r['source_row']} | {r['tier']} | {r['z']} | "
                         f"{r['predicted_length']} | {r['reported_taxonomy']} |")
    lines += [
        "",
        "## State Tracking fallback",
        "",
        "The following counts are among valid non-calibration probes. Explicit "
        "means an assigned State Tracking case not flagged as the fallback; "
        "this does not validate any attribution to cognition.",
        "",
        "| Model | Valid | State Tracking | Explicit | Fallback | Fallback / valid |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in breakdown:
        lines.append(
            f"| {r['model']} | {r['valid_probe_denominator']} | "
            f"{r['state_tracking_total']} | {r['state_tracking_explicit_rule']} | "
            f"{r['state_tracking_fallback']} | "
            f"{100 * r['fallback_fraction_of_valid_probes']:.3f}% |"
        )
    def thirds(model):
        matching = [r for r in rows if r["model"] == model
                    and int(r["z"]) == 30 and final.truth(r["valid"])
                    and not final.is_calibration_anchor(r)]
        return [100 * statistics.mean(float(r[k]) for r in matching)
                for k in ("early_accuracy", "middle_accuracy", "late_accuracy")]
    glm = thirds("zai.glm-5")
    mistral = thirds("mistral-large-2512")
    lines += [
        "",
        "## Central finding that remains",
        "",
        "The results retain observable positional differences not captured by a "
        "single pooled accuracy. GLM-5 and Mistral Large have pooled step "
        f"accuracies of {after_metrics['zai.glm-5']['step_accuracy_all_pct']}% and "
        f"{after_metrics['mistral-large-2512']['step_accuracy_all_pct']}%, respectively. "
        "Among valid z=30 responses pooled across tiers, their early/middle/late "
        f"accuracies are {glm[0]:.1f}/{glm[1]:.1f}/{glm[2]:.1f}% and "
        f"{mistral[0]:.1f}/{mistral[1]:.1f}/{mistral[2]:.1f}%. "
        "These descriptive comparisons use different horizon and tier mixtures "
        "for pooled accuracy and do not by themselves establish identical tasks, "
        "a causal tier effect, or the validity of cognitive taxonomy labels.",
        "",
        "## Interpretation and limitations",
        "",
        "- The live adaptive search was not rerun: its original sampled horizons "
        "and decisions used z-transition scores. Recomputed fracture and CRI "
        "describe the acquired support, not a counterfactual adaptive run.",
        "- At z=1 no transitions are scored; the existing implementation records "
        "zero Step Accuracy and continuity. These zeros are not measurements of "
        "transition failure and can affect summaries. The MAI z=1 records are invalid.",
        "- Seed-cluster intervals and fixed-z summaries are in "
        "`matched_horizon_metrics_publication.csv`; horizon matching does not "
        "itself imply identical tasks. `matched_horizon_pair_contrasts.csv` "
        "uses the script's exact task matching criteria. The intervals are coarse "
        "because there are usually five master-seed clusters or fewer.",
        "- Taxonomy rules are precedence-dependent behavioral labels; fallback "
        "and alternative-precedence analyses are in the sensitivity CSVs. "
        "Synthetic cases are not a substitute for independent human annotations.",
        "- The prior full forensic-audit deliverables and complete reviewer reports "
        "are not all available in this package; do not treat this report as a "
        "completed literature review or a full point-by-point reviewer response.",
        "",
        "## Reproduction",
        "",
        "From the repository root, run `python audit\\prompt_coordinate_reanalysis\\rescore.py`, "
        "then `python audit\\prompt_coordinate_reanalysis\\finish_reanalysis.py` "
        "and `python audit\\prompt_coordinate_reanalysis\\update_manuscript.py`. "
        "Compile the resulting TeX with pdflatex, bibtex, and two further pdflatex passes. "
        "The opt-in production CLI mode is `--coordinate-count`; the default "
        "retains original scoring, and JSON snapshots cannot be rescored with that option.",
        "Raw-input SHA-256 values are in `raw_input_sha256.csv`.",
        "",
    ]
    (OUT / "REANALYSIS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Verified {len(rows)} rows; wrote policy comparison and crosswalk.", flush=True)


if __name__ == "__main__":
    main()
