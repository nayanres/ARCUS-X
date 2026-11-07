"""Update manuscript tables and figures from the calibration-excluded audit.

Reads stored responses only; does not invoke models or alter raw logs.
"""
from __future__ import annotations

import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import post_run_analyzer as post_run
from arcus.evaluation.metrics import ARCUSRobustnessIndex, mean
from finalize_forensic_audit import is_calibration_anchor
from reproduce_correction import MANUSCRIPT_MODELS, MODEL_LABELS, _generate_failure_figure

REUSE_SEED_ROWS = False
PRESERVE_APPENDIX = False


def read(name):
    with (HERE / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(name, rows):
    with (HERE / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replace_rows(text, label, rows):
    anchor = text.index("\\label{" + label + "}")
    start = text.index("\\midrule", anchor) + len("\\midrule")
    end = text.index("\\bottomrule", start)
    original = text[start:end]
    original_names = re.findall(r"(?m)^\s*([^&\n]+?)\s*(?=\s*&)", original)
    if len(original_names) != len(rows):
        raise ValueError(f"{label}: expected {len(rows)} rows, got {len(original_names)}")
    if original_names != list(rows):
        raise ValueError(f"{label}: unexpected model order: {original_names}")
    return text[:start] + "\n" + "\n".join(rows.values()) + "\n" + text[end:]


def row(name, cells):
    return name + " & " + " & ".join(cells) + r" \\"


def percent(value):
    return f"{float(value):.1f}" + r"\%"


def num(value, digits):
    return f"{float(value):.{digits}f}"


def no_anchor(r):
    return not is_calibration_anchor(r)


def main():
    metrics = {r["model"]: r for r in read("post_run_table_metrics_publication.csv")}
    summary = {r["model"]: r for r in read("analysis_A_publication_model_summary.csv")}
    probes = [r for r in read("analysis_A_probe_level.csv")
              if r["model"] in MANUSCRIPT_MODELS and no_anchor(r)]
    sensitivity = [r for r in read("analysis_B_probe_level.csv")
                   if r["model"] in MANUSCRIPT_MODELS and no_anchor(r)
                   and r["valid"] == "True"]
    assert all(m in metrics and m in summary for m in MANUSCRIPT_MODELS)
    grouped = defaultdict(list)
    for r in probes:
        grouped[r["model"]].append(r)
    for m in MANUSCRIPT_MODELS:
        assert len(grouped[m]) == int(metrics[m]["analysis_observations"])

    # The post-run parser supplies token efficiency for per-seed CRI.
    if REUSE_SEED_ROWS:
        seed_rows = read("manuscript_seed_cri.csv")
        for m in MANUSCRIPT_MODELS:
            assert sum(int(r["observations"]) for r in seed_rows if r["model"] == m) == len(grouped[m]), m
    else:
        seed_rows = []
        for m in MANUSCRIPT_MODELS:
            data = post_run.process_txt_file(str(ROOT / "outputs" / metrics[m]["raw_log"]))
            filtered = {k: v for k, v in data["results_matrix"].items() if no_anchor(v)}
            assert len(filtered) == len(grouped[m]), m
            by_seed = defaultdict(dict)
            for key, result in filtered.items():
                by_seed[result["seed"]][key] = result
            for seed, matrix in sorted(by_seed.items()):
                seed_rows.append({"model": m, "seed": seed, "observations": len(matrix),
                                  "cri": post_run._compute_cri(
                                      matrix, ARCUSRobustnessIndex, mean
                                  )["cri"]})
            print(f"Processed {m}: {len(filtered)} observations, {len(by_seed)} seeds", flush=True)
        write("manuscript_seed_cri.csv", seed_rows)

    horizon_rows = []
    temporal = {}
    for m in MANUSCRIPT_MODELS:
        by_horizon = defaultdict(list)
        for r in grouped[m]:
            by_horizon[int(r["z"])].append(float(r["step_accuracy"]))
        for z, scores in sorted(by_horizon.items()):
            horizon_rows.append({"model": m, "horizon": z, "n": len(scores),
                                 "step_accuracy_pct": 100 * statistics.mean(scores)})
        for z in (20, 30):
            subset = [r for r in grouped[m] if int(r["z"]) == z and r["valid"] == "True"]
            if not subset:
                raise ValueError(f"No valid trajectories for {m}, z={z}")
            temporal[(m, z)] = [100 * statistics.mean(float(r[k]) for r in subset)
                                for k in ("early_accuracy", "middle_accuracy", "late_accuracy")]
    write("manuscript_horizon_curves.csv", horizon_rows)

    full = MANUSCRIPT_MODELS
    selected = ["GPT-5-mini", "moonshotai.kimi-k2.5", "zai.glm-5",
                "qwen.qwen3-coder-480b-a35b-v1:0", "mistral-large-2512"]
    temporal_order = [selected[0], selected[1], selected[2], selected[3], full[0],
                      full[4], full[3], full[5], full[6], full[7], full[8], full[9], full[12]]
    temporal_names = ["GPT-5-mini", "Kimi K2.5", "GLM-5", "Qwen3-Coder-480B-A35B",
                      "Qwen3-235B-A22B-2507", "Mistral Large 2512", "Mistral Small 2603",
                      "Devstral 2 123B", "Ministral 14B 2512", "Ministral 8B 2512",
                      "Ministral 3B 2512", "Nemotron Super 3 120B", "MAI-Thinking-1"]
    label = lambda m: MODEL_LABELS[m]
    values = {}
    for m in full:
        v = metrics[m]
        s = summary[m]
        name = label(m)
        values[m] = {
            "overall": [percent(v[k]) for k in ("step_accuracy_all_pct", "exact_match_all_pct",
                                                 "continuity_all_pct", "horizon_compliance_all_pct")]
                       + [percent(100 * float(v["validity"]))],
            "fracture": [str(int(float(v[f"tier_tier_{t}_fracture_depth"])))
                         if v[f"tier_tier_{t}_fracture_depth"] else "N/A" for t in range(4)],
            "tier": [percent(v[f"tier_tier_{t}_step_accuracy_pct"]) for t in range(4)],
            "efficiency": [percent(v["step_accuracy_all_pct"]), num(v["generation_bloat_index"], 2),
                           percent(v["generation_efficiency_pct"]), num(v["correct_steps_per_second"], 3)],
            "cri": [num(v[k], 3) for k in ("trajectory_fidelity", "horizon_robustness",
                                            "semantic_robustness", "cri_generation_efficiency", "cri")],
            "taxonomy": [percent(100 * float(s["validity"])), percent(100 * (1 - float(s["validity"])))]
                        + [percent(100 * float(s[k + "_proportion_valid"]))
                           for k in ("state_tracking_failure", "transition_rule_failure",
                                     "semantic_interpretation_failure", "horizon_collapse",
                                     "formatting_failure", "unknown_unmapped", "none")],
        }
        assert int(s["none_count"]) == int(s["exact_match_valid_count"]), name
        valid_seeds = [float(r["cri"]) for r in seed_rows if r["model"] == m]
        values[m]["seed"] = [f"${statistics.mean(valid_seeds):.3f} \\pm {statistics.pstdev(valid_seeds):.3f}$"]

    horizon_stats = {}
    for m in full:
        scores = [round(r["step_accuracy_pct"], 1) for r in horizon_rows if r["model"] == m]
        horizon_stats[m] = (len(scores), statistics.mean(scores), statistics.pstdev(scores))
    for m in full:
        n, average, std = horizon_stats[m]
        values[m]["fluctuation"] = [str(n), f"{average:.2f}" + r"\%", f"{std:.2f}" + r"\%"]
        rows = [r for r in sensitivity if r["model"] == m]
        policies = ("B_conservative_unclassified", "B_fallback_unclassified",
                    "B_horizon_first", "B_transition_first", "B_modulo_equivalent")
        values[m]["sensitivity"] = [percent(100 * sum(r[p] != r["A_corrected_original_precedence"]
                                                      for r in rows) / len(rows)) for p in policies]

    text_path = HERE / "arcus_x_corrected.tex"
    text = text_path.read_text(encoding="utf-8-sig")
    for table, models, key in [
        ("tab:overall_results", selected, "overall"),
        ("tab:overall_results_full", full, "overall"),
        ("tab:fracture_selected", selected, "fracture"),
        ("tab:fracture_full", full, "fracture"),
        ("tab:tier_selected", selected, "tier"),
        ("tab:tier_full", full, "tier"),
        ("tab:generation_efficiency_selected", selected, "efficiency"),
        ("tab:generation_efficiency", full, "efficiency"),
        ("tab:failure_taxonomy_selected", selected, "taxonomy"),
        ("tab:failure_taxonomy_full", full, "taxonomy"),
        ("tab:cri_decomposition", full, "cri"),
        ("tab:cri_seed_variance", full, "seed"),
        ("tab:horizon_fluctuation", full, "fluctuation"),
        ("tab:taxonomy_policy_stability", full, "sensitivity"),
    ]:
        rows = {}
        for m in models:
            cells = values[m][key][:]
            if table == "tab:overall_results_full":
                cells = cells[:4]
            rows[label(m)] = row(label(m), cells)
        text = replace_rows(text, table, rows)

    temporal_rows = {name: row(name, [percent(x) for z in (20, 30)
                                     for x in temporal[(m, z)]])
                     for m, name in zip(temporal_order, temporal_names)}
    text = replace_rows(text, "tab:temporal_thirds", temporal_rows)
    fluct = [horizon_stats[m][2] for m in full]
    correlations = {}
    for label_text, key in [("Horizon Compliance", "horizon_compliance_all_pct"),
                            ("Exact Match", "exact_match_all_pct"),
                            ("Step Accuracy", "step_accuracy_all_pct"),
                            ("Continuity", "continuity_all_pct")]:
        correlations[label_text] = float(np.corrcoef(fluct, [float(metrics[m][key]) for m in full])[0, 1])
    text = replace_rows(text, "tab:fluctuation_correlations",
                        {name: row(name, [f"${v:.3f}$"]) for name, v in correlations.items()})

    density_rows = [r for r in horizon_rows if r["model"] == "ministral-14b-2512"]
    density_total = sum(r["n"] for r in density_rows)
    density_table = "z acc density\n" + "\n".join(
        f'{r["horizon"]} {r["step_accuracy_pct"]:.1f} {100 * r["n"] / density_total:.2f}'
        for r in density_rows
    )
    text, replacements = re.subn(
        r"(?<=\] table\[meta=density\] \{\n)z acc density\n.*?(?=\n\};)",
        lambda _: density_table, text, count=1, flags=re.DOTALL,
    )
    if replacements != 1:
        raise ValueError("Could not locate the sampling-density plot")

    text = text.replace("exact match (42.5\\%)", "exact match (42.4\\%)")
    text = text.replace(r"\usepackage{fancyvrb}", r"\usepackage{fvextra}")
    if not PRESERVE_APPENDIX:
        marker = r"\section{Minitest transcripts (z=50, tiers 0--3)}"
        start = text.index(marker)
        text = text[:start] + r"""\section{Minitest transcripts}
These are the recorded prompts, model responses, and reference trajectories for
Ministral-14B-2512 at $z=50$, seed 3157, and gravity 0.5 across all four tiers.
The prompts are reproduced as sent, including their request for 50 coordinates
despite specifying an initial position followed by 50 transitions.
\VerbatimInput[breaklines,breakanywhere,fontsize=\scriptsize]{minitests_ministral-14b-2512_transcripts.txt}

\end{document}
"""
    text_path.write_text(text, encoding="utf-8")

    _generate_failure_figure(list(summary.values()))
    for models, filename, size in [(full, "horizon_profiles.pdf", (10, 7)),
                                    (selected, "horizon_profiles_selected.pdf", (9, 4.5))]:
        fig, ax = plt.subplots(figsize=size)
        colors = ["#CC79A7", "#0072B2", "#D55E00", "#009E73", "#E69F00"]
        markers = ["o", "s", "^", "D", "*"]
        for i, m in enumerate(models):
            series = [r for r in horizon_rows if r["model"] == m]
            kwargs = {"color": colors[i], "marker": markers[i]} if models is selected else {}
            ax.plot([r["horizon"] for r in series], [r["step_accuracy_pct"] for r in series],
                    linewidth=0.8, markersize=3, label=label(m).replace("*", ""), **kwargs)
        ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
        ax.set(xlabel="Requested horizon", ylabel="Step accuracy (%)", ylim=(0, 105))
        if models is full:
            ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(HERE / filename, bbox_inches="tight")
        plt.close(fig)

    raw = (ROOT / "outputs" / "minitests_ministral-14b-2512.txt").read_text(encoding="utf-8")
    entries = raw.split("=== RAW STREAM ENTRY")[1:]
    if len(entries) != 4 or "seed=3157" not in raw.splitlines()[1]:
        raise ValueError("Expected exactly one four-tier minitest at seed 3157")
    transcripts = []
    for tier, entry in enumerate(entries):
        if f"tier={tier}," not in entry:
            raise ValueError(f"Unexpected minitest tier {tier}")
        start = entry.index("[EXACT SENT PROMPT]:")
        end = entry.index("\n==============================", start)
        transcript = entry[start:end]
        assert "[MODEL OUTPUT]:" in transcript and "[GROUND TRUTH EXPECTED]:" in transcript
        transcripts.append(f"=== TIER {tier} ===\n\n{transcript}\n")
    (HERE / "minitests_ministral-14b-2512_transcripts.txt").write_text(
        "\n".join(transcripts), encoding="utf-8")
    print("Temporal values:", temporal, flush=True)
    print("Correlations:", correlations, flush=True)


if __name__ == "__main__":
    main()
