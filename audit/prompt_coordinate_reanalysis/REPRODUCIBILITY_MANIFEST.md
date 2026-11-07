# Reproducibility manifest

## Scope and provenance

- The publication tables use 13 models; the additional Minimax M2.5 raw log is
  included in the machine-readable analyses but not the manuscript tables.
- All 55,033 records in the 14 raw logs were rescored. Fixed calibration
  anchors are excluded from manuscript summaries. Repeated responses are
  retained, not collapsed merely because their tasks match.
- [raw_input_sha256.csv](raw_input_sha256.csv) records the source-log hashes;
  `verify_outputs.py` checks them against both the actual files and the earlier
  [provenance manifest](../correction_2026_11/PROVENANCE_MANIFEST.json).
- The earlier manifest records baseline commit
  `a1d3c950ccdc634d05b5d93d22863f53e587d11c`. The working tree includes
  staged and unstaged modifications; that commit alone is not the code used
  for this analysis. Exact historical prompt text and source revision per
  individual run were not established from the original logs.
- The recorded four-tier minitest is the separate file
  `outputs\minitests_ministral-14b-2512.txt`. Its extracted transcript is
  verified verbatim. It was not included in the publication cohort.
- The run used Windows, Python 3.11.9, NumPy 2.2.6, Matplotlib 3.11.1,
  pytest 9.1.1 and MiKTeX 26.5. The tokenizer library was unavailable; the
  stored output-token fields were used for this analysis.

## Commands

Run from the repository root:

```powershell
python audit\prompt_coordinate_reanalysis\rescore.py
python audit\prompt_coordinate_reanalysis\finish_reanalysis.py
python audit\prompt_coordinate_reanalysis\update_manuscript.py
python -m pytest tests\test_prompt_coordinate_reanalysis.py tests\test_post_run_analyzer.py -q
```

In this directory, compile `arcus_x_corrected.tex` with `pdflatex`, `bibtex`,
and two more `pdflatex` passes. Then, from the repository root, run:

```powershell
python audit\prompt_coordinate_reanalysis\verify_outputs.py
```

The scripts read stored data and do not call model APIs. The original source
manuscript, logs, references, and earlier corrected manuscript are not
overwritten. The `rescore.py` analysis separately recalculates transition
accuracy, continuity, Exact Match and positional thirds; targeted tests compare
the new production scoring option with this independent analysis.

## Statistical and scoring limitations

The acquired support and live AFS decisions reflect the original scoring,
not the alternative coordinate-count policy. Cluster-bootstrap intervals
resample master seeds, typically only five, and some paired subsets contain
fewer clusters; they are consequently coarse and should not be read as
strong evidence of significance. Valid-conditioned analyses are conditional
on responses being analyzable. Task-hash matching and equal-horizon matching
are distinct; task hashes come from the stored experiment signatures.

At z=1 the selected reference has no transitions. The existing scorer records
zero for both step accuracy and continuity rather than a mathematically
defined transition rate. Every MAI-Thinking-1 z=1 observation is invalid, so
the z=1 post-hoc fracture is not evidence of cognitive failure.
