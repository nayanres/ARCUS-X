"""Rebuild an isolated manuscript from the prompt-coordinate reanalysis."""
from __future__ import annotations

import csv
import shutil
import statistics
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / "audit" / "correction_2026_11"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(ROOT))

import reproduce_correction as correction
import update_manuscript_evidence as builder


def read(name):
    with (OUT / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one occurrence: {old[:100]!r}")
    return text.replace(old, new, 1)


def replace_between(text, start, end, new):
    first = text.index(start)
    last = text.index(end, first)
    return text[:first] + new + "\n\n" + text[last:]


def main():
    assets = ("arcus_x_corrected.tex", "math_commands.tex", "fancyhdr.sty",
              "natbib.sty", "iclr2027_conference.sty", "iclr2027_conference.bst",
              "iclr2027_conference.bib")
    for name in assets:
        shutil.copy2(SOURCE / name, OUT / name)
    builder.HERE = OUT
    builder.REUSE_SEED_ROWS = True
    builder.PRESERVE_APPENDIX = True
    correction.OUT = OUT
    builder.main()

    path = OUT / "arcus_x_corrected.tex"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "Sinha et al.~\\cite{sinha2026execution} isolate long-horizon execution by supplying the knowledge and plan needed for a task, and study how reliability changes with execution length. Wu et al.~\\cite{wu2024counterfactual} evaluate counterfactual task variants, including altered spatial conventions. ARCUS-X builds on these directions rather than claiming that execution-depth effects or rule perturbations are new in isolation.",
        "\\textit{The Illusion of Diminishing Returns} \\cite{sinha2026execution} isolates long-horizon execution by supplying the knowledge and plan needed for a task, and studies how reliability changes with execution length. \\textit{Reasoning or Reciting?} \\cite{wu2024counterfactual} evaluates counterfactual task variants, including altered spatial conventions. ARCUS-X builds on these directions rather than claiming that execution-depth effects or rule perturbations are new in isolation.\n\n"
        "RULER \\cite{hsieh2024ruler} evaluates long-context capabilities through retrieval, multi-hop tracing, and aggregation tasks. LongMemEval \\cite{wu2025longmemeval} evaluates long-term conversational memory, including information extraction, multi-session and temporal reasoning, knowledge updates, and abstention. LoCoMo \\cite{maharana2024locomo} examines long-term conversational memory through multi-session interactions and associated reasoning tasks.",
    )
    text = replace_once(
        text,
        "ARCUS-X instead isolates sequential state-transition reliability in a deterministic environment. Exact per-transition ground truth supports direct measurement of step correctness, continuity, exact agreement, divergence, horizon compliance, and sampled fracture behavior while controlling horizon and the model-facing rule system. Its contribution is narrower than general trajectory evaluation: exact trajectory attribution, controlled rule perturbations, an explicit validity split, deterministic failure signatures, and adaptive horizon allocation in one auditable protocol.",
        "ARCUS-X addresses a narrower evaluation objective than these long-context and memory benchmarks: the reliability of executing explicitly specified deterministic state transitions over dependent sequences. Rather than primarily testing retrieval or memory across extended contexts, ARCUS-X evaluates intermediate states against a computer-generated reference trajectory. Its contribution lies in jointly characterizing transition correctness, temporal continuity, exact trajectory agreement, horizon compliance, operational validity, and generation overhead under controlled rule perturbations. Long-horizon degradation and trajectory evaluation are not claimed as novel in themselves; the benchmark provides an auditable decomposition of observable sequential execution behavior.",
    )
    text = replace_once(text, "\\bibliography{iclr2027_conference}",
                        "\\bibliography{iclr2027_conference,long_context_references}")
    if "horizon (\\textit{z}), the number of generated actions;" in text:
        path.write_text(text, encoding="utf-8")
        print(f"Updated {path}", flush=True)
        return
    original = (SOURCE / "iclr2027_conference.tex").read_text(encoding="utf-8-sig")
    original_abstract = original.split("\\begin{abstract}\n", 1)[1].split("\n\\end{abstract}", 1)[0]
    original_abstract = replace_once(original_abstract,
        "84.8\\% step accuracy, 42.5\\% exact match, and 84.5\\% continuity, "
        "but 25.9\\% generation efficiency",
        "85.3\\% step accuracy, 36.0\\% exact match, and 85.0\\% continuity, "
        "but 25.6\\% generation efficiency",
    )
    original_abstract = replace_once(original_abstract,
        "93.8\\% of Qwen 3 Coder 480B A35B probes and 84.3\\% of GLM-5 probes, "
        "transition-rule signatures are largest for Mistral Large 2512 (23.6\\%), "
        "and semantic-interpretation signatures are largest for Kimi K2.5 (23.2\\%).",
        "94.0\\% of valid Qwen 3 Coder 480B A35B probes and 84.3\\% of valid GLM-5 probes, "
        "while Mistral Large 2512 has 24.8\\% transition-rule signatures and "
        "Kimi K2.5 has 23.2\\% semantic-interpretation signatures. "
        "Horizon signatures, including overlong outputs, account for 49.4\\% and "
        "56.1\\% of valid Mistral Large and Kimi probes, respectively.",
    )
    original_abstract = replace_once(original_abstract, "The valid-error partitions",
                                     "The valid-probe partitions")
    text = replace_between(text, "\\begin{abstract}\n", "\\end{abstract}",
                           "\\begin{abstract}\n" + original_abstract)
    text = replace_once(
        text,
        "horizon (\\textit{z}), the number of sequential state transitions required.",
        "horizon (\\textit{z}), the number of generated actions; the evaluated "
        "trajectory contains $z$ coordinates including the initial state.",
    )
    text = replace_once(
        text,
        "The parsed trajectory is compared against ground truth to calculate transition\n"
        "accuracy, exact trajectory match, and transition continuity. The expected\n"
        "trajectory contains the provided initial state plus one state per requested\n"
        "transition: $L_{\\mathrm{GT}} = z + 1$.",
        "The parsed trajectory is compared against ground truth to calculate transition\n"
        "accuracy, exact trajectory match, and transition continuity. The evaluated\n"
        "reference contains the provided initial state and the first $z-1$ action\n"
        "results: $L_{\\mathrm{GT}} = z$. The final action remains in the prompt\n"
        "but its result is not scored. A response with $z+1$ coordinates is\n"
        "overlong under this criterion even if it correctly executes every action.",
    )
    text = replace_once(text, "{number of correctly reproduced post-initial states}}\n    {z}.",
                        "{number of correctly reproduced post-initial states}}\n    {z-1}.")
    text = replace_once(text, "{longest correct post-initial prefix}}\n    {z}.",
                        "{longest correct post-initial prefix}}\n    {z-1}.")
    text = replace_once(
        text,
        "Comparison is performed at aligned post-initial positions over the\n"
        "    overlapping region. The initial state is not counted.",
        "Comparison is performed at aligned post-initial positions over the\n"
        "    overlapping region. The initial state is not counted. At $z=1$,\n"
        "    no transitions are scored; the implementation records zero rather\n"
        "    than interpreting this as evidence about transition execution.",
    )
    text = replace_once(
        text,
        "z_{\\mathrm{generated}}-z_{\\mathrm{requested}}",
        "L_{\\mathrm{pred}}-z",
    )
    text = replace_once(text, "\\max(z_{\\mathrm{generated}},z_{\\mathrm{requested}})",
                        "\\max(L_{\\mathrm{pred}},z)")
    text = replace_once(text, "$T_{\\mathrm{expected}}=10(z+1)+10$: ten tokens for each of the $z+1$ coordinate pairs",
                        "$T_{\\mathrm{expected}}=10z+10$: ten tokens for each of the $z$ coordinate pairs")
    text = replace_once(
        text,
        "divided by the requested horizon $z$; exact match requires identical\n"
        "coordinates and length; and continuity is the longest correct post-initial\n"
        "prefix divided by $z$.",
        "divided by $z-1$ scored transitions; exact match requires identical\n"
        "coordinates and length; and continuity is the longest correct post-initial\n"
        "prefix divided by $z-1$ (the zero-transition edge case is recorded as zero).",
    )
    text = replace_once(
        text,
        "At each horizon $z$, cognitive step accuracy is computed over operationally valid trajectories:",
        "The search uses accuracy over all $z$ action transitions, whereas the "
        "reported trajectory metrics evaluate the first $z-1$ transitions. "
        "At each horizon $z$, cognitive step accuracy is computed over operationally valid trajectories:",
    )
    text = replace_once(
        text,
        "Thus, AFS operates on the same scores used for the benchmark and for resume reconstruction.",
        "Thus, live AFS and resume reconstruction score all $z$ action transitions; "
        "the reported trajectory tables score the requested $z$ coordinates.",
    )
    text = replace_once(
        text,
        "The reported runs additionally retain their manifests and raw\n"
        "streams, so the exact evaluated conditions can be inspected rather than\n"
        "inferred from aggregate tables.",
        "The available raw streams retain task parameters and responses, but do\n"
        "not establish the exact prompt text and source revision for every run.",
    )
    text = replace_once(
        text,
        "A stable SHA-256 experiment signature is implemented in the code release; when\n"
        "present, it binds the configuration and source revision to the resulting\n"
        "artifacts.",
        "A stable SHA-256 experiment signature in the code release identifies\n"
        "the generated task inputs; it is not itself a source-revision identifier.",
    )
    text = replace_once(
        text,
        "while avoiding unnecessary probing at horizons that are clearly reliable.",
        "without exhaustively evaluating every horizon.",
    )
    text = replace_once(
        text,
        "If any point in that window returns to at least $\\tau$, the candidate dip is suppressed;",
        "The post-run tables use a two-point window. If any point in the selected "
        "window returns to at least $\\tau$, the candidate dip is suppressed;",
    )
    text = replace_once(
        text,
        "The prompts are reproduced as sent, including their request for 50 coordinates\n"
        "despite specifying an initial position followed by 50 transitions.",
        "The prompts are reproduced as sent. The recorded reference contains the initial\n"
        "state and all 50 action results; reported metrics score its first 50\n"
        "coordinates, including the initial state. The benchmark logs do not\n"
        "preserve exact prompts, so their wording is inferred from available\n"
        "prompt-builder versions rather than verified for each run.",
    )
    text = replace_once(
        text,
        "Early, middle, and late correspond to contiguous thirds of the requested transition sequence.",
        "Early, middle, and late correspond to contiguous thirds of the scored transition sequence.",
    )
    text = replace_once(text,
                        "\\path[fill=red!7] (axis cs:70,0) rectangle (axis cs:107,80);\n",
                        "")
    text = replace_between(text,
        "\\draw[red!60!black, dashed, line width=.5pt] (axis cs:70,0)",
        "\\end{axis}", "")
    text = replace_once(
        text,
        " The shaded and dashed interval highlights pooled degradation from "
        "$z=70$ through the maximum observed horizon $z=107$.",
        "",
    )
    text = replace_between(
        text, "The pooled degradation is not caused",
        "\\subsection{CRI}",
        "The density varies across sampled horizons. This descriptive plot "
        "does not establish the cause of the observed accuracy pattern.",
    )

    metrics = {r["model"]: r for r in read("post_run_table_metrics_publication.csv")}
    summary = {r["model"]: r for r in read("analysis_A_publication_model_summary.csv")}
    abstract_values = {
        "85.3": float(metrics["GPT-5-mini"]["step_accuracy_all_pct"]),
        "36.0": float(metrics["GPT-5-mini"]["exact_match_all_pct"]),
        "85.0": float(metrics["GPT-5-mini"]["continuity_all_pct"]),
        "25.6": float(metrics["GPT-5-mini"]["generation_efficiency_pct"]),
        "11.2": 100 * (1 - float(metrics["GPT-5-mini"]["validity"])),
        "94.0": 100 * float(summary["qwen.qwen3-coder-480b-a35b-v1:0"]["state_tracking_failure_proportion_valid"]),
        "84.3": 100 * float(summary["zai.glm-5"]["state_tracking_failure_proportion_valid"]),
        "24.8": 100 * float(summary["mistral-large-2512"]["transition_rule_failure_proportion_valid"]),
        "23.2": 100 * float(summary["moonshotai.kimi-k2.5"]["semantic_interpretation_failure_proportion_valid"]),
        "49.4": 100 * float(summary["mistral-large-2512"]["horizon_collapse_proportion_valid"]),
        "56.1": 100 * float(summary["moonshotai.kimi-k2.5"]["horizon_collapse_proportion_valid"]),
    }
    if any(abs(float(displayed) - actual) > 0.051 for displayed, actual in abstract_values.items()):
        raise ValueError("An abstract value no longer agrees with the analysis")
    selected = ["GPT-5-mini", "moonshotai.kimi-k2.5", "zai.glm-5",
                "qwen.qwen3-coder-480b-a35b-v1:0", "mistral-large-2512"]
    a = metrics[selected[0]]
    others = [metrics[m] for m in selected[1:]]
    text = replace_between(
        text, "Within the observed AFS-supported pools,", "\\subsection{Horizon-Dependent",
        "Within the observed AFS-supported pools, GPT-5-mini has the highest "
        "step accuracy, exact match, and continuity of the five representative runs: "
        f"{float(a['step_accuracy_all_pct']):.1f}\\% step accuracy, "
        f"{float(a['exact_match_all_pct']):.1f}\\% exact match, and "
        f"{float(a['continuity_all_pct']):.1f}\\% continuity. The other four show "
        f"{min(float(r['step_accuracy_all_pct']) for r in others):.1f}--"
        f"{max(float(r['step_accuracy_all_pct']) for r in others):.1f}\\% step accuracy and "
        f"{min(float(r['exact_match_all_pct']) for r in others):.1f}--"
        f"{max(float(r['exact_match_all_pct']) for r in others):.1f}\\% exact match. "
        "GLM-5 and Qwen 3 Coder 480B A35B also show that high horizon compliance "
        "can coexist with lower trajectory correctness. These are run-level descriptive "
        "comparisons, not a controlled ranking over matched horizons.",
    )
    text = replace_between(
        text, "Table~\\ref{tab:fracture_selected} shows", "\\subsection{Semantic Robustness}",
        "Table~\\ref{tab:fracture_selected} shows that sampled structural fracture "
        "is strongly condition-dependent rather than an invariant model property. "
        "GPT-5-mini has no post-hoc fracture in the sampled curves, whereas Kimi "
        "K2.5 has late sampled fractures in Tier 0 and Tier 2 but an earlier one "
        "in Tier 3. Mistral Large 2512 makes the non-monotonicity especially clear: "
        "its Tier-2 sampled fracture is late despite much earlier values in the "
        "other tiers. The recovery rule operates on the next two \\emph{sampled} "
        "horizons, whose spacing is not uniform, so these values describe the "
        "acquired curves rather than a fixed-width horizon neighborhood. "
        "The live AFS window was capped at $H=150$; reaching that cap without a "
        "below-floor midpoint is recorded separately as a saturated, right-censored result.",
    )
    def tier_values(model):
        return [float(metrics[model][f"tier_tier_{t}_step_accuracy_pct"])
                for t in range(4)]
    gpt, kimi, glm, coder, mistral = map(tier_values, selected)
    if not (kimi[3] < kimi[0] and glm[3] < glm[0] and coder[3] < coder[0]
            and mistral.index(min(mistral)) == 1 and mistral.index(max(mistral)) == 3
            and kimi[1] > kimi[0]):
        raise ValueError("The original qualitative tier discussion is no longer supported")
    text = replace_between(
        text, "The observed tier profiles are not ordered", "\\subsection{Generation Efficiency",
        "The observed tier profiles are not ordered uniformly across models. "
        f"GPT-5-mini spans {max(gpt)-min(gpt):.1f} points between its highest and "
        "lowest pooled tier values. Kimi K2.5 and GLM-5 are substantially lower "
        f"in Tier 3 than Tier 0 ({kimi[0]-kimi[3]:.1f} and {glm[0]-glm[3]:.1f} "
        f"points), while Qwen 3 Coder 480B A35B differs by {coder[0]-coder[3]:.1f} "
        "points. Mistral Large 2512 instead has its lowest pooled accuracy in "
        "Tier 1 and its highest in Tier 3, and Kimi K2.5 is higher in Tier 1 "
        "than Tier 0. These results argue against treating perturbation robustness "
        "as a single ordinal scalar: lexical relabeling and transition-altering "
        "transformations can produce different signatures for the same model. "
        "Because horizons are not matched by construction, we describe these "
        "as pooled tier associations rather than attributing the differences "
        "uniquely to the perturbation itself. "
        "Appendix~\\ref{app:full_model} reports all 13 models and the CRI decomposition.",
    )
    probes = [r for r in read("analysis_A_probe_level.csv")
              if r["model"] in correction.MANUSCRIPT_MODELS
              and not builder.is_calibration_anchor(r) and r["valid"] == "True"]
    def thirds(model, z):
        subset = [r for r in probes if r["model"] == model and int(r["z"]) == z]
        if not subset:
            raise ValueError(f"No valid trajectories for {model}, z={z}")
        return tuple(100 * statistics.mean(float(r[k]) for r in subset)
                     for k in ("early_accuracy", "middle_accuracy", "late_accuracy"))
    def fmt(values):
        return ", ".join(f"{v:.1f}\\%" for v in values)
    counts = [sum((lambda v: v[0] > v[1] > v[2])(thirds(m, z))
                  for m in correction.MANUSCRIPT_MODELS) for z in (20, 30)]
    temporal_text = (
        "A within-trajectory accuracy gradient is visible in a majority of the observed "
        "profiles. "
        f"At $z=20$, {counts[0]} of the 13 models have strictly decreasing accuracy "
        f"from the early to middle to late region. At $z=30$, this pattern occurs "
        f"for {counts[1]} of 13 models. The magnitude, however, differs substantially "
        "across models.\n\n"
        "Several models preserve high initial-step accuracy while degrading sharply "
        "later in the same trajectory. At $z=30$, the early, middle, and "
        f"late values are {fmt(thirds('nvidia.nemotron-super-3-120b', 30))} for "
        "Nemotron Super 3, "
        f"{fmt(thirds('qwen.qwen3-coder-480b-a35b-v1:0', 30))} for Qwen3-Coder, and "
        f"{fmt(thirds('zai.glm-5', 30))} for GLM-5. These profiles would be "
        "substantially compressed by a single trajectory-level mean: the model "
        "can execute a large fraction of early transitions correctly even when "
        "the terminal portion is already unreliable.\n\n"
        "The pattern is not universal. At the same $z=30$ horizon, Kimi K2.5 has "
        f"{fmt(thirds('moonshotai.kimi-k2.5', 30))}, Mistral Large has "
        f"{fmt(thirds('mistral-large-2512', 30))}, and MAI-Thinking-1 has "
        f"{fmt(thirds('MAI-Thinking-1', 30))}. Devstral 2 is approximately flat "
        f"with {fmt(thirds('mistral.devstral-2-123b', 30))}. We therefore do not "
        "interpret late-region degradation as a universal monotonic law of "
        "sequential execution.\n\n"
        "\\paragraph{Development at longer horizons.}\n"
        "The onset of temporal degradation is itself horizon dependent. "
        "GPT-5-mini, for example, remains nearly uniform through moderate horizons: "
        f"at $z=80$ its early, middle, and late accuracies are {fmt(thirds('GPT-5-mini', 80))}, "
        "respectively. The separation becomes substantially larger among the "
        "longest sampled trajectories. "
        f"At $z=115$, the corresponding values are {fmt(thirds('GPT-5-mini', 115))}; "
        f"at $z=133$, they are {fmt(thirds('GPT-5-mini', 133))}; and "
        f"at $z=149$, they are {fmt(thirds('GPT-5-mini', 149))}. Thus, aggregate "
        "long-horizon degradation can arise even when the beginning of the "
        "requested trajectory remains almost perfectly executed.\n\n"
        "Other models exhibit the same qualitative separation at much shorter "
        "horizons. GLM-5 has early, middle, and late values of "
        f"{fmt(thirds('zai.glm-5', 20))} at $z=20$, and "
        f"{fmt(thirds('zai.glm-5', 60))} at $z=60$. Qwen3-Coder and Nemotron "
        "Super 3 show sharper early-to-late separation by $z=30$. The horizon "
        "at which a temporal gradient becomes visible is therefore model "
        "dependent and cannot be recovered from a single pooled accuracy statistic.\n\n"
        "Adjacent horizons can also recover or fluctuate. We consequently do not "
        "fit a monotonic temporal-decay model or interpolate unobserved horizons. "
        "As elsewhere in ARCUS-X, the reported values describe only sampled, observed support."
    )
    text = replace_between(text, "A within-trajectory accuracy gradient",
                           "\\paragraph{Relation to the trajectory-level taxonomy.}",
                           temporal_text)
    text = text.replace("the regenerated population mean", "the population mean")
    text = text.replace("all regenerated\nfailure reports", "all\nfailure reports")
    text = text.replace("regenerated run.", "run.")
    text = text.replace("five independently regenerated CRI values", "five independently computed CRI values")
    text = text.replace("Corrected post-run validity", "Post-run validity")
    text = text.replace("Corrected failure-signature profiles", "Failure-signature profiles")
    text = text.replace("under corrected strict correctness and canonical-coordinate rules",
                        "under strict correctness and canonical-coordinate rules")
    text = text.replace("uses the corrected classifier, strict full-trajectory correctness, "
                        "canonical coordinates, and the original primary precedence",
                        "uses the classifier with strict trajectory correctness, "
                        "canonical coordinates, and the primary precedence")
    text = text.replace("relative to corrected Analysis A", "relative to Analysis A")
    path.write_text(text, encoding="utf-8")
    print(f"Updated {path}", flush=True)


if __name__ == "__main__":
    main()
