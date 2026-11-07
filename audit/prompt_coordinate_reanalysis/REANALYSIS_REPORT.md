# Prompt-coordinate post-hoc reanalysis

The selected interpretation scores z coordinates: the initial state plus the first z-1 action results. The supplied final action is not scored. The old interpretation scored all z actions and z+1 coordinates. A z+1 output is therefore overlong under the selected interpretation.

This is an interpretation of an ambiguous instruction, not a discovery of the model's intent. Current and historical prompt-builder versions request z coordinates, but the historical raw logs do not preserve exact prompt text or prove the generator revision for every run. The minitest transcripts do preserve their actual prompts. No API responses were generated; the original logs and reference trajectories were not edited.

Probe records checked: 55,033, across 14 raw logs; the publication cohort contains 13 models and excludes the extra Minimax run and fixed calibration anchors. Validity is unchanged.
Every valid non-calibration probe was checked individually: `None` disagreed with Exact Match in 0 cases.

## Overall scoring changes

Percentages below are pooled over all non-calibration observations, including invalid responses. Taxonomy percentages use valid probes; the two denominators must not be interchanged.

| Model | N | Valid | Step old → new | Exact old → new | HC old → new |
|---|---:|---:|---:|---:|---:|
| GPT-5-mini | 5700 | 5061 | 84.8 → 85.3 | 42.4 → 36.0 | 88.1 → 87.7 |
| MAI-Thinking-1 | 7622 | 3110 | 39.9 → 40.0 | 33.0 → 6.4 | 40.7 → 40.1 |
| minimax.minimax-m2.5 | 5700 | 218 | 3.7 → 3.7 | 3.5 → 0.1 | 3.8 → 3.7 |
| ministral-14b-2512 | 4520 | 4121 | 54.9 → 55.5 | 15.5 → 5.9 | 74.5 → 74.5 |
| ministral-3b-2512 | 2220 | 1820 | 42.4 → 43.7 | 10.2 → 8.4 | 71.5 → 72.7 |
| ministral-8b-2512 | 2460 | 2452 | 47.3 → 47.7 | 7.0 → 0.9 | 66.4 → 64.2 |
| mistral-large-2512 | 3400 | 3295 | 55.8 → 56.0 | 1.1 → 0.4 | 46.0 → 45.2 |
| mistral-small-2603 | 2180 | 2180 | 49.4 → 50.3 | 4.8 → 0.6 | 80.4 → 80.1 |
| mistral.devstral-2-123b | 5220 | 3793 | 44.5 → 44.7 | 0.1 → 0.1 | 42.7 → 42.0 |
| moonshotai.kimi-k2.5 | 6580 | 6240 | 59.2 → 59.2 | 39.1 → 0.1 | 78.0 → 77.0 |
| nvidia.nemotron-super-3-120b | 2260 | 2258 | 45.7 → 47.7 | 4.8 → 0.6 | 96.7 → 96.8 |
| qwen.qwen3-235b-a22b-2507 | 2300 | 599 | 14.0 → 14.1 | 6.1 → 0.3 | 18.0 → 17.3 |
| qwen.qwen3-coder-480b-a35b-v1:0 | 2160 | 2160 | 48.0 → 49.7 | 3.5 → 1.2 | 97.0 → 98.3 |
| zai.glm-5 | 2640 | 2639 | 55.1 → 56.2 | 13.9 → 1.0 | 98.3 → 98.4 |

## Individual examples

These valid probes matched the full z+1 reference under the old policy. Their additional coordinate makes them non-exact under the new policy; the label also depends on the classifier's precedence.

| Model | Source row | Tier | z | Emitted | New label |
|---|---:|---:|---:|---:|---|
| moonshotai.kimi-k2.5 | 5 | 3 | 20 | 21 | Horizon Collapse |
| GPT-5-mini | 2 | 0 | 20 | 21 | Horizon Collapse |
| zai.glm-5 | 3 | 1 | 20 | 21 | Horizon Collapse |
| mistral-large-2512 | 155 | 1 | 70 | 71 | Horizon Collapse |

## State Tracking fallback

The following counts are among valid non-calibration probes. Explicit means an assigned State Tracking case not flagged as the fallback; this does not validate any attribution to cognition.

| Model | Valid | State Tracking | Explicit | Fallback | Fallback / valid |
|---|---:|---:|---:|---:|---:|
| GPT-5-mini | 5061 | 528 | 528 | 0 | 0.000% |
| MAI-Thinking-1 | 3110 | 94 | 94 | 0 | 0.000% |
| minimax.minimax-m2.5 | 218 | 1 | 1 | 0 | 0.000% |
| ministral-14b-2512 | 4121 | 2086 | 2086 | 0 | 0.000% |
| ministral-3b-2512 | 1820 | 1036 | 1035 | 1 | 0.055% |
| ministral-8b-2512 | 2452 | 520 | 520 | 0 | 0.000% |
| mistral-large-2512 | 3295 | 365 | 365 | 0 | 0.000% |
| mistral-small-2603 | 2180 | 1043 | 1043 | 0 | 0.000% |
| mistral.devstral-2-123b | 3793 | 898 | 898 | 0 | 0.000% |
| moonshotai.kimi-k2.5 | 6240 | 670 | 668 | 2 | 0.032% |
| nvidia.nemotron-super-3-120b | 2258 | 2092 | 2092 | 0 | 0.000% |
| qwen.qwen3-235b-a22b-2507 | 599 | 49 | 49 | 0 | 0.000% |
| qwen.qwen3-coder-480b-a35b-v1:0 | 2160 | 2031 | 2031 | 0 | 0.000% |
| zai.glm-5 | 2639 | 2224 | 2224 | 0 | 0.000% |

## Central finding that remains

The results retain observable positional differences not captured by a single pooled accuracy. GLM-5 and Mistral Large have pooled step accuracies of 56.2% and 56.0%, respectively. Among valid z=30 responses pooled across tiers, their early/middle/late accuracies are 93.6/67.3/37.3% and 77.7/76.9/76.6%. These descriptive comparisons use different horizon and tier mixtures for pooled accuracy and do not by themselves establish identical tasks, a causal tier effect, or the validity of cognitive taxonomy labels.

## Interpretation and limitations

- The live adaptive search was not rerun: its original sampled horizons and decisions used z-transition scores. Recomputed fracture and CRI describe the acquired support, not a counterfactual adaptive run.
- At z=1 no transitions are scored; the existing implementation records zero Step Accuracy and continuity. These zeros are not measurements of transition failure and can affect summaries. The MAI z=1 records are invalid.
- Seed-cluster intervals and fixed-z summaries are in `matched_horizon_metrics_publication.csv`; horizon matching does not itself imply identical tasks. `matched_horizon_pair_contrasts.csv` uses the script's exact task matching criteria. The intervals are coarse because there are usually five master-seed clusters or fewer.
- Taxonomy rules are precedence-dependent behavioral labels; fallback and alternative-precedence analyses are in the sensitivity CSVs. Synthetic cases are not a substitute for independent human annotations.
- The prior full forensic-audit deliverables and complete reviewer reports are not all available in this package; do not treat this report as a completed literature review or a full point-by-point reviewer response.

## Reproduction

From the repository root, run `python audit\prompt_coordinate_reanalysis\rescore.py`, then `python audit\prompt_coordinate_reanalysis\finish_reanalysis.py` and `python audit\prompt_coordinate_reanalysis\update_manuscript.py`. Compile the resulting TeX with pdflatex, bibtex, and two further pdflatex passes. The opt-in production CLI mode is `--coordinate-count`; the default retains original scoring, and JSON snapshots cannot be rescored with that option.
Raw-input SHA-256 values are in `raw_input_sha256.csv`.
