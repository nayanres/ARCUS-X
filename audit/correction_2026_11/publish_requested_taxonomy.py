#!/usr/bin/env python3
"""Publish verified requested-contract taxonomy tables and figure."""

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from audit.correction_2026_11 import reviewer_response_analysis as base
from audit.correction_2026_11 import reproduce_correction as correction

MANUSCRIPT = base.PACKAGE / "iclr2027_conference.tex"
LABELS = base.TAXONOMY


def read(name):
    with (base.OUT / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def replace_table(text, label, rows):
    pos = text.index("\\label{" + label + "}")
    start = text.index("\\midrule", pos) + len("\\midrule")
    end = text.index("\\bottomrule", start)
    previous = text[start:end]
    names = [line.split("&", 1)[0].strip() for line in previous.splitlines() if "&" in line]
    if names != list(rows):
        raise ValueError(f"{label}: unexpected model/order {names}")
    return text[:start] + "\n" + "\n".join(rows.values()) + "\n" + text[end:]


def main():
    rows = read("requested_taxonomy_model_tier_summary.csv")
    sensitivity = read("requested_taxonomy_precedence_sensitivity.csv")
    horizon = read("requested_taxonomy_horizon_taxonomy.csv")
    models = {}
    for model in base.PUBLICATION_MODELS:
        items = [r for r in rows if r["model"] == model]
        if len(items) != 4:
            raise ValueError(f"Missing tier rows for {model}")
        values = {label: sum(int(r[f"corrected_{label}_count"]) for r in items) for label in LABELS}
        valid = sum(int(r["valid"]) for r in items)
        invalid = sum(int(r["invalid"]) for r in items)
        if sum(values.values()) != valid:
            raise ValueError(f"{model}: taxonomy partition mismatch")
        models[model] = {"model": model, "valid": valid, "invalid": invalid, **values}
    out = []
    for model in base.PUBLICATION_MODELS:
        x = models[model]
        item = {"model": model, "valid": x["valid"], "invalid": x["invalid"],
                "valid_pct": 100 * x["valid"] / (x["valid"] + x["invalid"])}
        for label in LABELS:
            item[f"{label}_count"] = x[label]
            item[f"{label}_pct_valid"] = 100 * x[label] / x["valid"]
        out.append(item)
    base.write_csv(base.OUT / "requested_taxonomy_model_summary.csv", out)

    text = MANUSCRIPT.read_text(encoding="utf-8")
    named = {correction.MODEL_LABELS[m]: m for m in base.PUBLICATION_MODELS}
    for table in ("tab:failure_taxonomy_selected", "tab:failure_taxonomy_full"):
        pos = text.index("\\label{" + table + "}")
        section = text[pos:].split("\\midrule", 1)[1].split("\\bottomrule", 1)[0]
        names = [line.split("&", 1)[0].strip() for line in section.splitlines() if "&" in line]
        replacements = {}
        for name in names:
            item = models[named[name]]
            valid, invalid = item["valid"], item["invalid"]
            values = [100 * valid / (valid + invalid), 100 * invalid / (valid + invalid)]
            values += [100 * item[label] / valid for label in (
                "State Tracking Failure", "Transition Rule Failure", "Semantic Interpretation Failure",
                "Horizon Collapse", "Formatting Failure", "Unknown / Unmapped", "None",
            )]
            replacements[name] = name + " & " + " & ".join(f"{v:.1f}\\%" for v in values) + r" \\"
        text = replace_table(text, table, replacements)

    names = [correction.MODEL_LABELS[m] for m in base.PUBLICATION_MODELS]
    lookup = {(r["model"], r["policy"]): int(r["changed"]) for r in sensitivity if r["taxonomy"] == "None"}
    rows = {}
    for name in names:
        model = named[name]
        denominator = models[model]["valid"]
        values = [100 * lookup[(model, policy)] / denominator for policy in (
            "conservative_unclassified", "fallback_state_tracking", "horizon_first",
            "transition_first", "modulo_equivalent",
        )]
        rows[name] = name + " & " + " & ".join(f"{v:.1f}\\%" for v in values) + r" \\"
    text = replace_table(text, "tab:taxonomy_policy_stability", rows)
    table_rows = {}
    for z in (20, 30):
        subset = [r for r in horizon if r["scope"] == "exact" and int(r["target_horizon"]) == z]
        counts = Counter()
        for r in subset:
            counts[r["taxonomy"]] += int(r["count"])
        denominator = sum(counts.values())
        expected = 3604 if z == 20 else 2992
        if denominator != expected:
            raise ValueError(f"Unexpected z={z} valid denominator")
        percentages = [100 * counts[label] / denominator for label in LABELS]
        table_rows[str(z)] = f"{z} & {denominator:,} & " + " & ".join(f"{p:.1f}\\%" for p in percentages) + r" \\"
    text = replace_table(text, "tab:matched_horizon_taxonomy", table_rows)
    MANUSCRIPT.write_text(text, encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    display = ["State Tracking", "Transition Rule", "Semantic Interpretation", "Horizon Collapse", "Formatting"]
    categories = ["State Tracking Failure", "Transition Rule Failure", "Semantic Interpretation Failure",
                  "Horizon Collapse", "Formatting Failure"]
    order = correction.MANUSCRIPT_MODELS
    values = np.array([[100 * models[m][c] / models[m]["valid"] for c in categories] for m in order])
    fig, ax = plt.subplots(figsize=(11, max(7.0, len(order) * 0.43)))
    image = ax.imshow(values, vmin=0, vmax=100, aspect="auto", cmap="Greys")
    ax.set_xticks(np.arange(len(display)), display, fontsize=8)
    ax.set_yticks(np.arange(len(order)), [correction.MODEL_LABELS[m] for m in order], fontsize=7)
    for i in range(len(order)):
        for j in range(len(display)):
            ax.text(j, i, f"{values[i, j]:.1f}%", ha="center", va="center", fontsize=7,
                    color="white" if values[i, j] > 50 else "black")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Failure proportion among valid probes", fontsize=9)
    colorbar.set_ticks([0, 20, 40, 60, 80, 100])
    colorbar.set_ticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.set_xlabel("Failure signature", fontsize=10)
    ax.set_ylabel("Model", fontsize=10)
    ax.set_title("Failure-signature profiles", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(base.PACKAGE / "failure_profiles_requested.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Updated taxonomy tables and wrote failure_profiles_requested.pdf")


if __name__ == "__main__":
    main()
