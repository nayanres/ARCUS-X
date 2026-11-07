"""Verify manuscript tables, original inputs, and verbatim minitest evidence."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / "audit" / "correction_2026_11"
sys.path.insert(0, str(SOURCE))
import reproduce_correction as correction


def read(name):
    with (OUT / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def table_rows(text, label):
    section = text[text.index("\\label{" + label + "}"):]
    section = section[section.index("\\midrule") + len("\\midrule"):section.index("\\bottomrule")]
    return {
        parts[0].strip(): [part.strip().replace(r"\%", "") for part in parts[1:]]
        for line in section.splitlines()
        if "&" in line
        for parts in [line.strip().removesuffix(r"\\").split("&")]
    }


def main():
    text = (OUT / "arcus_x_corrected.tex").read_text(encoding="utf-8")
    metrics = {r["model"]: r for r in read("post_run_table_metrics_publication.csv")}
    summary = {r["model"]: r for r in read("analysis_A_publication_model_summary.csv")}
    models = correction.MANUSCRIPT_MODELS
    table = table_rows(text, "tab:overall_results_full")
    taxonomy = table_rows(text, "tab:failure_taxonomy_full")
    tier = table_rows(text, "tab:tier_full")
    for model in models:
        name = correction.MODEL_LABELS[model]
        m, s = metrics[model], summary[model]
        expected = [float(m[k]) for k in ("step_accuracy_all_pct", "exact_match_all_pct",
                                          "continuity_all_pct", "horizon_compliance_all_pct")]
        expected_tax = [100 * float(s["validity"]), 100 * (1 - float(s["validity"]))]
        expected_tax += [100 * float(s[k + "_proportion_valid"])
                         for k in ("state_tracking_failure", "transition_rule_failure",
                                   "semantic_interpretation_failure", "horizon_collapse",
                                   "formatting_failure", "unknown_unmapped", "none")]
        expected_tier = [float(m[f"tier_tier_{t}_step_accuracy_pct"]) for t in range(4)]
        for actual, wanted in ((table[name], expected), (taxonomy[name], expected_tax),
                               (tier[name], expected_tier)):
            if len(actual) != len(wanted) or any(
                abs(float(a) - b) > 0.051 for a, b in zip(actual, wanted)
            ):
                raise ValueError(f"Table mismatch for {model}: {actual} vs {wanted}")
        if s["none_count"] != s["exact_match_valid_count"]:
            raise ValueError(f"None/Exact Match mismatch for {model}")

    raw = (ROOT / "outputs" / "minitests_ministral-14b-2512.txt").read_text(encoding="utf-8")
    entries = raw.split("=== RAW STREAM ENTRY")[1:]
    if len(entries) != 4:
        raise ValueError("Expected four raw minitest entries")
    blocks = []
    for tier_number, entry in enumerate(entries):
        begin = entry.index("[EXACT SENT PROMPT]:")
        end = entry.index("\n==============================", begin)
        blocks.append(f"=== TIER {tier_number} ===\n\n{entry[begin:end]}\n")
    expected_transcript = "\n".join(blocks)
    if (OUT / "minitests_ministral-14b-2512_transcripts.txt").read_text(
            encoding="utf-8") != expected_transcript:
        raise ValueError("Transcript differs from the original raw minitest")

    manifest = json.loads((SOURCE / "PROVENANCE_MANIFEST.json").read_text(encoding="utf-8"))
    original = {Path(item["path"]).name: item["sha256"]
                for item in manifest["input_inventory"] if item["path"].startswith("outputs\\")}
    hashes = read("raw_input_sha256.csv")
    for entry in hashes:
        actual = hashlib.sha256((ROOT / "outputs" / entry["file"]).read_bytes()).hexdigest()
        if actual != entry["sha256"] or original.get(entry["file"]) != actual:
            raise ValueError(f"Raw-log hash changed or is absent: {entry['file']}")
    if len(hashes) != 14:
        raise ValueError("Expected 14 raw model logs")

    log = (OUT / "arcus_x_corrected.log").read_text(encoding="utf-8", errors="replace")
    for warning in ("There were undefined references", "Citation `", "Overfull \\hbox"):
        if warning in log:
            raise ValueError(f"Check LaTeX warning: {warning}")
    if not (OUT / "arcus_x_corrected.pdf").is_file():
        raise ValueError("Compiled PDF is missing")
    print("Verified 13-model tables, four verbatim transcripts, 14 raw-log hashes, and LaTeX log.")


if __name__ == "__main__":
    main()
