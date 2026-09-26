# ARCUS-X: Deterministic Benchmark for Long-Horizon Sequential Reasoning

ARCUS-X evaluates how well language models maintain state and follow procedural
rules over long horizons. The benchmark generates **deterministic, procedural
grid-trajectory tasks** where the ground truth is computed from explicit
transition rules — never from model output.

## Overview

ARCUS-X is designed to separate sequential state tracking from semantic
robustness. Every task is generated from a seed, scored against an internally
computed trajectory, and reported with both headline metrics and diagnostic
failure categories. The repository includes API evaluation, optional local GPU
inference, model-free baselines, reproducibility controls, and post-run audits.

### Contents

- [Benchmark design](#benchmark-design)
- [Metrics and reporting](#metrics-and-reporting)
- [Validation suite](#validation-suite)
- [Architecture](#architecture-single-source-of-truth)
- [Quick start](#quick-start)
- [Inter-horizon trajectory analysis](#inter-horizon-trajectory-analysis)
- [Reproducibility](#reproducibility)
- [Dependency management](#dependency-management)
- [File manifest](#file-manifest)
- [License](#license)

## Benchmark Design

- A discrete toroidal grid (width × height) with an initial coordinate.
- A sequence of **actions** (e.g., `north`, `east`, `south`, `west`).
- **Transition rules** mapping each action to a `(dx, dy)` delta.
- The **ground-truth trajectory** is the result of applying the rules step by
  step with modulo wrapping.
- The model is given the rules and action sequence and must emit the full
  coordinate path.

## Difficulty Tiers (Gravity)

| Tier | Name | Transformation | Ground Truth |
|------|------|----------------|--------------|
| 0 | Baseline | None | Base rules |
| 1 | Semantic Disruption | 1:1 label → pseudoword mapping | **Unchanged** (only labels change) |
| 2 | Attribute Inversion | Invert y-axis deltas `(dx, -dy)` | **Recomputed** from inverted rules |
| 3 | Axiomatic Contradiction | 90° rotation `(-dy, dx)` | **Recomputed** from rotated rules |

Tier 1 preserves the exact trajectory; Tiers 2/3 change the transition
semantics and the ground truth is **recomputed** from the mutated rules.

**Tier Classification Summary**

- **Tier 0 (Baseline)** – No perturbation; the task uses the original transition rules.
- **Tier 1 (Semantic Disruption)** – Only the action labels are remapped to pseudowords; the underlying transition dynamics and ground‑truth trajectory remain unchanged. This is a *semantic robustness* probe.
- **Tier 2 (Attribute Inversion)** – The sign of the y‑axis delta is flipped; the ground truth is recomputed from the inverted rules.
- **Tier 3 (Coordinate Transformation)** – Deltas are rotated by 90°; the ground truth is recomputed from the rotated rules.

> **Tier 1 is a semantic‑invariance probe, not a reasoning‑difficulty probe.**
> The action sequence, grid, initial state, transition rules, and ground‑truth
> trajectory are *identical* to Tier 0 — only the surface vocabulary (labels) is
> swapped for pseudowords. A model that solves Tier 0 but fails Tier 1 therefore
> reveals a *lexical/semantic binding* failure (it cannot map the new label to the
> same underlying rule), **not** a deficit in sequential reasoning. This is the
> explicit scientific purpose of Tier 1: it isolates semantic robustness from
> reasoning difficulty. Do not interpret Tier 1 accuracy drops as evidence of
> reduced reasoning capacity.

### Semantic Gravity: Definition and Scope

**Semantic gravity** is a *deterministic benchmark difficulty parameter*, not a
measure of human cognition or model "understanding". It controls perturbation
intensity, while **tier** selects the qualitative perturbation family. They are
crossed experimental dimensions in the evaluation matrix: every requested
gravity level is evaluated independently at each requested tier. For example,
the paper's gravity levels can include `0.5`, `3.0`, and `15.0`, and are not
limited to the tier numbers `0`–`3`.

`DifficultyConfig.tier_for(gravity)` remains a deterministic convenience
mapping for code paths that need a default tier from a scalar gravity value
(and for backward-compatible utilities); it does **not** describe the main
runner's experimental design. The `disruption_ratio` is
`min(1.0, gravity / 3.0)` and is used only for internal logging/calibration —
it is **never** leaked to the model prompt (see
[Reproducibility](#reproducibility) and the validation suite).

**Methodology note:** Difficulty parameters (tier, gravity_target, seed, task_id, experiment_hash) are controlled internally and are not exposed to evaluated models. They are used only by the generation and evaluation code paths; the model-facing prompt contains solely the initial state, grid dimensions, transition rules, action sequence, task question, and required output format.

**Why this matters :** Semantic gravity is a *controlled
perturbation of the evaluation environment*, analogous to changing the rules of
a game. It is not a proxy for "reasoning depth" in the cognitive sense. All
tiers share the same grid dimensions, initial state, and action sequence; only
the transition function (and, for Tier 1, the label vocabulary) differs. This
ensures that any observed accuracy drop is attributable to the model's ability
to follow the *provided* rules, not to a change in task length or structure.

### Axis Separation: Tier vs. Gravity vs. Horizon vs. Fracture

To avoid confusion, the benchmark separates four distinct axes:

| Axis | What it controls | Values | Effect on Ground Truth |
|------|------------------|--------|------------------------|
| **Tier** | *Qualitative perturbation family* (type of rule change) | 0, 1, 2, 3 | Tier 0/1: unchanged; Tier 2/3: recomputed |
| **Gravity** | *Perturbation intensity* (difficulty parameter) | Any configured numeric levels (for example, 0.5, 3.0, 15.0) | Crossed with Tier; does not select the qualitative tier in the main runner |
| **Horizon / Depth** | *Long-horizon state tracking difficulty* (number of steps) | 1, 2, 3, ... | Increases trajectory length; GT scales with horizon |
| **Fracture Depth** | *Observed failure point* (where accuracy drops below threshold) | Integer horizon index | Derived from model performance, not a task parameter |

**distinction:** Tier and Gravity are *task construction* parameters (set before evaluation). Horizon is a *task configuration* parameter. Fracture Depth is a *measurement* derived from model outputs. 

### Grid Scaling (optional axis)

**Grid scaling is a fully implemented, optional evaluation axis — not future
extensibility.** It is disabled by default; when enabled, grid dimensions become
an independent sweep axis alongside gravity / tier / depth.

- Controlled by the `grid_size_levels` parameter (a list of `(width, height)`
  tuples, e.g. `[(8, 8), (16, 16), (32, 32)]`).
- When `None` (default), grid dimensions are sampled per-instance exactly as
  before — the default benchmark behavior is unchanged.
- When set, the runner yields each `(width, height)` as an independent axis; the
  results-matrix key becomes a 5-tuple `(z, gravity, tier, probe_idx, grid_key)`
  so aggregation is uniform across both modes.
- The generator consumes `grid_width` / `grid_height` directly
  ([`arcus/tasks/generator.py`](arcus/tasks/generator.py)); the ground-truth
  trajectory is recomputed via the single source of truth `compute_trajectory`,
  so larger grids produce longer, valid toroidal paths with no code fork.

**Why optional:** The default benchmark already varies grid size per-instance
(stochastically), which is sufficient for the headline results. The explicit
`grid_size_levels` sweep is provided for studies that need to *isolate* grid-size
effects from other axes (e.g. controlled scaling curves). It is a working
capability, not a placeholder.

## Metrics and Reporting (all on a 0.0–1.0 scale)

**Primary metrics** (directly measure trajectory compliance — these are the
headline results):

- **Step Accuracy** — fraction of requested transitions matching ground truth;
  the provided initial state is excluded from this metric.
- **Exact Match** — entire predicted path equals ground truth.
- **Continuity** — longest correct transition prefix / number of requested
  transitions; exact-match and structural length checks still include the
  initial state.

**Secondary metrics** (diagnostic / efficiency context — reported alongside
primary metrics but never used as the sole success criterion):

- **First Divergence** — 1-based index of first mismatch.
- **Horizon Compliance** — did the model generate the requested number of steps?
- **Generation Bloat Index (GBI)** — `max(0, actual_tokens - expected_tokens) / expected_tokens`, where `expected_tokens = estimated_optimal_path_length(z)`. Range ≥ 0.
- **Generation Efficiency** — `1 / (1 + GBI)`. Range 0–1.
- **Token Density** (auxiliary, raw) — `total_tokens / depth`, reported per tier as `token_density`. This is a raw ratio (not a normalized 0–1 score) used for generation-cost accounting; it is *not* a primary metric.
- **Error Taxonomy** — classifies *why* a trajectory failed:
  `State Tracking Failure`, `Transition Rule Failure`,
  `Semantic Interpretation Failure`, `Horizon Collapse`, `Formatting Failure`.
- **Fracture Depth** — the canonical reporting detector returns the first
  *sampled* horizon whose step accuracy is strictly below `0.5` and has no
  recovery within the next three sampled points. This look-ahead belongs to
  reporting; it is not the AFS search algorithm.
- **Adaptive Fracture Search (AFS)** — the active search uses deterministic
  midpoint bisection over the horizon bounds. Each midpoint is classified from
  the observed batch accuracy: operationally invalid probes are filtered out
  before the mean is computed, and a batch with no valid probes yields
  `None` rather than `0.0`. AFS does not require three consecutive failures,
  does not treat exactly `50%` accuracy as fracture, and does not claim
  unbiasedness or unique horizon evaluations. Resume/cached observations may
  therefore be reused. AFS and the canonical fracture detector are separate
  layers; the search loop is not implemented by `FracturePointFinder`.
- **ARCUS Robustness Index (CRI)** — *secondary aggregate* robustness indicator
  (NOT a primary metric). It summarises correctness retention across horizons,
  semantic perturbations, and efficiency:

      CRI = 0.45·TrajectoryFidelity + 0.30·HorizonRobustness
          + 0.20·SemanticRobustness + 0.05·GenerationEfficiency

  where `TrajectoryFidelity = 0.5·step_accuracy + 0.3·continuity + 0.2·exact_match`,
  `HorizonRobustness` is a normalised trapezoidal area over sampled depths, and
  `SemanticRobustness` measures preservation of Tier-0 performance under tier
  perturbation (Tier 1 is a label-only perturbation expected to preserve the
  trajectory; Tiers 2/3 change transition semantics and are expected to
  degrade). CRI is reported alongside the primary trajectory metrics but never
  replaces them.

### Interpretable Reporting

ARCUS-X now emits **per-tier** and **per-gravity** fracture diagnostics alongside
the aggregate CRI, so one can localise *where* and *why* a model breaks:

- **Per-tier fracture depth** — `metrics.per_tier[tier].fracture_depth` reports
  the mean sampled horizon at which step accuracy first falls strictly below
  the fracture floor without recovering within the next three sampled points,
  for each tier (0–3). This isolates the effect of each perturbation family.
- **Per-gravity fracture curves** — `metrics.per_gravity_fracture_curve[gravity]`
  contains the sorted `(depth, step_accuracy)` pairs for each gravity level,
  enabling a direct plot of accuracy vs. horizon per gravity.
- **CRI component breakdown** — the `cri` block in the aggregated metrics now
  exposes each weighted component (`trajectory_fidelity`, `horizon_robustness`,
  `semantic_robustness`, `generation_efficiency`) alongside the final `cri` score,
  so the contribution of each axis is auditable rather than opaque.

All three are written to `outputs/benchmark_results_<timestamp>.json` and printed
in the structured debug log (`outputs/benchmark_debug.log`).

> **Note on `final_state_correct`:** A model can reach the correct final
> coordinate via a wrong intermediate path. ARCUS-X therefore reports
> `final_state_correct` as a *secondary* signal only; the primary metrics
> (step accuracy, exact match, continuity) require the entire path to be correct.

> **CRI is a secondary indicator.** ARCUS-X reports trajectory metrics (step
> accuracy, exact match, continuity, fracture depth) as the *primary* evaluation
> signals. CRI is a *secondary* aggregate robustness indicator that summarises
> **trajectory robustness** and **sequential state-tracking robustness** — i.e.
> correctness retention across horizons, semantic perturbations, and efficiency.
> It is not a measure of intelligence or cognition, and it never replaces the
> primary trajectory metrics.

### Multi-Seed Variance Reporting

When `--master-seeds` is supplied (or `all`), the aggregated metrics block
(`metrics` in the JSON report) includes a self-documenting seed list and a
per-seed breakdown alongside the aggregate, so every report is reproducible and
directly analysable without cross-referencing hardcoded constants:

```json
{
  "master_seeds": [42, 12345, 6734, 9878, 2026, 9001, 1337, 31415],
  "seed_variance": {
    "n_seeds": 8,
    "cri":            { "mean": 0.655, "std": 0.018 },
    "fracture_depth": { "mean": 11.0, "std": 0.7 }
  },
  "per_seed": {
    "42":    { "cri": { "cri": 0.661 }, "cri_value": 0.661, "fracture_depth": 11 },
    "12345": { "cri": { "cri": 0.640 }, "cri_value": 0.640, "fracture_depth": 10 },
    "6734":  { "cri": { "cri": 0.658 }, "cri_value": 0.658, "fracture_depth": 12 }
  }
}
```

- **`master_seeds`** — the exact list of master seeds used for this run (the
  actual expanded list, not a reference to a constant).
- **`seed_variance`** — aggregate `mean ± std` of CRI and fracture depth across
  the master seeds (population std; `std = 0.0` when only one seed is used).
- **`per_seed`** — a dict keyed by seed (as a string) holding each seed's full
  CRI bundle (`cri`), the scalar `cri_value`, and `fracture_depth`. This lets
  downstream statistical analysis access every seed's results directly rather
  than reconstructing them from the aggregate.

The terminal `ARCUS-X RUN SUMMARY` also prints a `Master Seeds:` line, and the
report's `parameters.seeds` field records the same list for archival
reproducibility.

## Parallel Master-Seed Execution & Resume

ARCUS-X can run multiple master seeds **concurrently** and can **resume** an
interrupted run from its raw stream. Both are runtime/scheduling features only —
they do **not** change benchmark semantics, metrics, probe ordering, or scoring.

### Parallel execution (`--parallel-seeds`)

```bash
# Serial (default) — one seed at a time, in order
python -m arcus.experiments.quickstart --model ... --master-seeds 42 123 456

# Explicit worker count
python -m arcus.experiments.quickstart --model ... --master-seeds 42 123 456 \
    --parallel-seeds 3

# AUTO — min(cpu_count, n_seeds, user_max_workers)
python -m arcus.experiments.quickstart --model ... --master-seeds 42 123 456 \
    --parallel-seeds auto
```

- **`"1"` (default)** — fully serial; behavior is byte-for-byte identical to the
  original single-threaded path.
- **`N`** — up to `N` worker threads, one per master seed.
- **`"auto"`** — `workers = min(cpu_count(), len(master_seeds), user_max_workers)`.
  It **never** creates more workers than there are master seeds, so a 3-seed run
  uses at most 3 workers regardless of core count.

**Determinism guarantee.** Each worker owns exactly **one** master seed and runs
on its **own isolated `BenchmarkRunner`** instance (it shares only the evaluator
and the raw-stream log file). Probes *within* a seed are never parallelized, and
each seed instantiates its own `GridTaskGenerator(master_seed)`. As a result,
parallel execution produces **identical results** to a serial run: the
results-matrix signature and the multiset of raw-stream probe blocks are the
same. The only non-deterministic field is the wall-clock `End Time:` stamp in the
trailing `ARCUS-X RUN END` block.

**Independent outputs / crash safety.** Workers append to the *same* raw stream
file under a lock, so if one worker crashes the other seeds' completed results
remain valid and recoverable. A crashed worker's contribution is simply omitted
from the merged results; the run still reports the seeds that finished.

### Resume (`--resume`)

```bash
# Continue an interrupted run by pointing at its absolute raw stream
python -m arcus.experiments.quickstart --model ... --master-seeds 42 123 456 \
    --resume outputs/absolute_raw_stream_<model>.txt
```

- The runner parses the existing raw stream and detects **completed probes by
  identity** — `master_seed + gravity + tier + horizon (z) + probe_index +
  grid_key`. No file ordering or timestamps are used, so resume is fully
  deterministic.
- Completed probes are **re-scored from the raw stream** using the exact same
  pipeline as a live run, then loaded into the bisection search so fracture-depth
  decisions are identical to a fresh run.
- Only the **missing** probes are executed. The run **appends** to the same raw
  stream file; the final stream is identical (up to cross-seed entry ordering)
  to a fresh run.
- If **every** probe is already complete, execution is skipped entirely and only
  aggregation / reporting / post-run analysis run.
- **Resume + parallel work together:** workers divide only the *remaining* probes
  and never duplicate completed work.

### Adaptive rate-limit handling (429)

When the evaluator reports a rate-limit (`429`), the scheduler **reduces
concurrency** (`4 → 3 → 2`) and **restores** it after a cooldown. Completed work
is never lost (it is already written to the raw stream). Throttling is **not**
counted as a benchmark failure.

### Progress display & logging

The live UX shows a seed bar and worker count:

```
Seeds: [####-----] 3/8 complete
Workers: 3 active
```

and, on resume, the loaded/remaining counts:

```
Loaded: 120   Remaining: 35
Skipping completed work...
```

The logger records: worker started/completed, concurrency reduced/restored,
resume loaded, and completed/skipped/resumed probe counts.

## Architecture (Single Source of Truth)

```
arcus/
    tasks/
        generator.py      # GridTaskGenerator, GridTask, TaskConfig
        config.py
    environment/
        grid.py           # compute_trajectory, GridEnvironment, format_coord
    evaluation/
        parser.py         # extract_model_path, estimated_optimal_path_length
        metrics.py        # compare_trajectories, TrajectoryResult, ...
        evaluator.py      # ModelEvaluator (vLLM + API backends)
        token_accounting.py
    analysis/
        gravity.py        # DifficultyTier, DifficultyConfig
        taxonomy.py       # ErrorMode, classify_from_result
        fracture.py       # FracturePointFinder
    experiments/
        runner.py         # shim -> benchmark_runner
        benchmark_runner.py  # BenchmarkRunner (orchestrator)
        quickstart.py     # CLI entry point
```

**Public entry points (repo root):**
- `run_framework.sh` — shell wrapper
- `runframework.sh` — compatibility alias for `run_framework.sh`
- `quickstart_api.py` — Python wrapper (delegates to `arcus.experiments.quickstart`)

Everything else is internal.

## Quick Start

Install the runtime (API evaluation / lightweight benchmark execution) — only
`requests` is required. Local GPU inference is **optional** and ARCUS-X does
**not** require it.

```bash
# Runtime / API evaluation (no GPU / heavy ML libs required)
pip install -r requirements.txt

# Optional: only if you run local models (vLLM / transformers)
pip install -r requirements-local.txt
```

```bash
# Run via the shell wrapper (API-only mode, needs an OpenRouter-compatible key)
export OPENROUTER_API_KEY="sk-or-v1-..."
./run_framework.sh --model google/gemini-2.5-flash-lite --probes 5
```

Or directly via Python:

```bash
python -m arcus.experiments.quickstart \
    --api-key "$OPENROUTER_API_KEY" \
    --model google/gemini-2.5-flash-lite \
    --n-probes 5 \
    --gravity-levels "0.0,1.0,2.0,3.0"
```

Results are written to `outputs/benchmark_results_<timestamp>.json`.

### Post-Run Analysis (`post_run_analyzer.py`)

You can analyze completed JSON result files, directories, or raw TXT logs using
[`post_run_analyzer.py`](post_run_analyzer.py):

```bash
# Analyze all runs in a TXT log file (default)
python post_run_analyzer.py --input outputs/absolute_raw_data_local_gpt_5_mini.txt

# Analyze ONLY the most recent run in a TXT log file with multiple appended runs
python post_run_analyzer.py --input outputs/absolute_raw_data_local_gpt_5_mini.txt --lastrun

# Include the share of all probes at each horizon
python post_run_analyzer.py --input outputs/absolute_raw_data_local_gpt_5_mini.txt --density

# Include non-zero error rates and error codes at each horizon
python post_run_analyzer.py --input outputs/absolute_raw_data_local_gpt_5_mini.txt --error-density
```

- **`--lastrun`** — filters TXT logs to parse only the most recent run session (starting from the final `ARCUS-X RUN START` block).
- **`--density`** — adds the percentage of total probes represented by each horizon (for example, `z=3: 3.0%`).
- **`--error-density`** — adds non-zero operational error rates by horizon using all probes evaluated at that horizon as the denominator, with per-code counts (for example, `z=3   20 probes   E100: 2   E101: 1   error density: 15.0%`).
- **Multi-seed support** — automatically captures all seeds present in the session and reports per-seed variance and metrics correctly.

### Inter-Horizon Trajectory Analysis (`inter_horizon_analysis.py`)

[`inter_horizon_analysis.py`](inter_horizon_analysis.py) is a focused,
post-hoc diagnostic for the raw TXT stream logs produced by an ARCUS-X run.
While `post_run_analyzer.py` summarizes performance by probe and horizon, this
tool shows **where within each trajectory** errors occur. It divides every
horizon into early, middle, and end thirds (any remainder is assigned to the
end) and reports step accuracy for each region and horizon.

Run it against an absolute raw-output log:

```bash
python inter_horizon_analysis.py outputs/absolute_raw_data_local_gpt_5_mini.txt
```

On Windows PowerShell, the same command can be run directly:

```powershell
python .\inter_horizon_analysis.py .\outputs\absolute_raw_data_local_gpt_5_mini.txt
```

Use `--all_tax` when you also want the trajectory-level taxonomy broken out
for each horizon:

```bash
python inter_horizon_analysis.py outputs/absolute_raw_data_local_gpt_5_mini.txt --all_tax
```

The report contains:

- **Step Accuracy Per Third** — early/middle/end accuracy for each numeric
  horizon (`z`). A declining end-of-trajectory score can indicate accumulated
  state-tracking difficulty, while a uniform drop suggests a broader
  interpretation or formatting problem.
- **Trajectory-level Taxonomy Distribution** — the canonical error category
  for each valid trajectory, including state-tracking, transition-rule,
  semantic-interpretation, horizon-collapse, formatting, and unmapped errors.
- **Optional taxonomy by horizon** — enabled by `--all_tax`, useful for seeing
  whether the mix of failure modes changes as trajectories become longer.

This analysis is valuable because aggregate exact-match accuracy can hide the
failure pattern. Comparing the thirds separates early mistakes from errors
that emerge only after several state transitions, making it easier to
distinguish long-horizon degradation from a general inability to apply the
rules. It is also lightweight and reproducible: it reads the raw stream
format, reuses ARCUS-X's parser and taxonomy classifier, and does not invoke
the model or rerun the benchmark.

Interpret the results with two important limitations in mind. Taxonomy labels
are assigned at the **trajectory level**, not to a specific third, so the
taxonomy cannot by itself prove that an error occurred in the end region.
Also, invalid/provider-error outputs are excluded from step-accuracy totals;
they should be considered separately rather than treated as cognitive
failures.

### Run it locally (step by step)

The benchmark is **API-only by default** — you only need an OpenRouter-compatible
key and `requests`. No GPU or heavy ML libraries are required.

**1. Install the runtime**

```bash
pip install -r requirements.txt
```

**2. Provide your API key** (any of these work — the runner reads the env var
automatically if `--api-key` is omitted):

```bash
# Linux / macOS / Git Bash
export OPENROUTER_API_KEY="sk-or-v1-..."

# Windows (Command Prompt)
set OPENROUTER_API_KEY=sk-or-v1-...

# Windows (PowerShell)
$env:OPENROUTER_API_KEY = "sk-or-v1-..."
```

**3. Run it**

```bash
# Linux / macOS / Git Bash — shell wrapper
./run_framework.sh --model google/gemini-2.5-flash-lite --probes 5

# Windows (no bash needed) — call the Python entry point directly
python -m arcus.experiments.quickstart --model google/gemini-2.5-flash-lite --n-probes 5

# Or the legacy Python wrapper (delegates to the same module)
python quickstart_api.py --model google/gemini-2.5-flash-lite --probes 5
```

> **Note:** `run_framework.sh` is a bash script, so on Windows use
> `python -m arcus.experiments.quickstart` (or `python quickstart_api.py`)
> instead — they are exactly equivalent and go through the same
> `BenchmarkRunner.run()`.

**4. What you'll see**

The run prints an interactive UX on stdout:

- A **startup banner** (`ARCUS-X v<version>`, self-test result, loading messages).
- A **live progress** block (`Probe N / Total`, a percentage bar, and the
  current `Model / Tier / Gravity / Horizon` context) plus a transient
  **status line** (`Waiting for API...`, `Retrying after rate limit (n/8)`,
  `Computing CRI...`, `Saving results...`).
- A **final evaluation chart** with the headline metrics, the CRI breakdown,
  per-tier fracture depth, and the full **error taxonomy** distribution.

On a UTF-8 terminal the progress bar uses block glyphs (`█`/`░`); on a legacy
Windows console it falls back to ASCII (`#`/`-`) automatically. When the output
is piped or redirected (non-TTY), the progress degrades to throttled plain lines.

**5. Optional flags**

`--n-probes N` (default 5), `--output FILE`, `--gravity-levels "0.0,1.0,2.0,3.0"`,
`--depths "1,2,3"`, `--api-base URL` (override the default OpenRouter endpoint),
and `--baselines` (include model-free oracle/random/initial-state references in
the report without needing a model).

Two additional scheduling flags are documented in
[Parallel Master-Seed Execution & Resume](#parallel-master-seed-execution--resume):
`--parallel-seeds "1"|N|auto"` (concurrent seed execution; `auto` = `min(cpu_count,
n_seeds)`) and `--resume <path>` (continue an interrupted run from its absolute
raw stream, skipping already-completed probes by identity).

**Multi-seed variance (`--master-seeds`)**

By default the benchmark runs a single master seed. To measure run-to-run
variance (so headline results are not an artifact of one lucky/unlucky seed),
pass a space-separated list of master seeds:

```bash
# Explicit seeds
python quickstart_api.py \
    --api-key "$OPENROUTER_API_KEY" \
    --model deepseek/deepseek-v4-flash \
    --master-seeds 42 1337 2026 9001 123456

# Canonical paper seed list (expands "all")
python quickstart_api.py \
    --api-key "$OPENROUTER_API_KEY" \
    --model deepseek/deepseek-v4-flash \
    --master-seeds all
```

`--master-seeds all` expands to the canonical paper seed list
`[42, 12345, 6734, 9878, 2026, 9001, 1337, 31415]`. The benchmark is executed
once per master seed; each seed regenerates the task set via
`GridTaskGenerator(seed=master_seed)` and produces its own results matrix. The
`run_framework.sh` wrapper accepts the same flag
(`./run_framework.sh --model ... --master-seeds all`).

### Local GPU Inference (optional)

ARCUS-X does **not** require local inference — the default path is API-only.
Install [`requirements-local.txt`](requirements-local.txt) only when running
models locally (vLLM / transformers).

The underlying evaluator
[`arcus/evaluation/evaluator.py`](arcus/evaluation/evaluator.py) uses **local
vLLM automatically when `api_base` is `None`**, so no API key or network
connection is needed for this path:

```python
from arcus.evaluation.evaluator import ModelEvaluator

evaluator = ModelEvaluator(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    api_base=None,  # Triggers local vLLM
    use_vllm=True
)
```

> **API vs local switching:** If `--api-base` is provided, ARCUS-X uses the hosted API backend. If `--api-base` is omitted, ARCUS-X passes `api_base=None` to the evaluator, which automatically initializes local vLLM inference.

Requirements for local mode:
- A CUDA-capable GPU with enough VRAM for the chosen model.
- `vllm` installed (via [`requirements-local.txt`](requirements-local.txt)).
  If vLLM is missing you'll get
  `ImportError: vLLM required for local inference`.
- The model must be a vLLM-compatible Hugging Face model id.

## Dependency Management

ARCUS-X uses a layered dependency setup so you only install what you need:

| File | Role |
|------|------|
| `requirements.txt` | Flexible runtime spec (`>=`) — API-only evaluation (just `requests`) |
| `requirements-local.txt` | Flexible optional spec (`>=`) — local GPU inference (torch / vLLM / transformers) |
| `requirements-lock.txt` | **Pinned** runtime closure, generated by `pip-compile` |
| `requirements-lock-local.txt` | **Pinned** local-inference closure, generated by `pip-compile` |
| `requirements-dev.txt` | Dev-only tooling (`pip-tools`) used to regenerate the locks |

For reproducible / CI installs, install from the pinned locks instead of the
flexible specs:

```bash
pip install -r requirements-lock.txt            # runtime (API-only)
pip install -r requirements-lock-local.txt      # optional local inference
```

Regenerate both locks after changing `requirements.txt` or
`requirements-local.txt`:

```bash
pip install -r requirements-dev.txt
./scripts/lock.sh        # Linux / macOS / Git Bash
# or on Windows (cmd.exe):
scripts\lock.bat
```

> **Windows note:** `vllm`'s wheel contains very long nested paths that exceed
> Windows' `MAX_PATH` when `pip-compile` extracts metadata. `scripts/lock.bat`
> redirects `TEMP` to a short path to work around this; the Linux/macOS
> `scripts/lock.sh` is unaffected.

## Reference Environment (Container Image)

**The container image is the reference evaluation environment used to generate
this paper's results.** It is the canonical, archival artifact: pull it by
digest and run your evaluation inside it for a bit-for-bit reproducible
environment.

```bash
# Pull the digest-pinned image (replace <org> and the digest with the values
# recorded for the camera-ready / released tag):
docker pull ghcr.io/<org>/arcus-x:<tag-or-digest>@sha256:<digest>

# Verify the environment works (runs the validation suite):
docker run --rm ghcr.io/<org>/arcus-x:<tag-or-digest>@sha256:<digest> \
    python tests/test_arcus_hardening.py

# Run a benchmark instead (API-only mode needs an OpenRouter-compatible key):
docker run --rm -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
    ghcr.io/<org>/arcus-x:<tag-or-digest>@sha256:<digest> \
    python -m arcus.experiments.quickstart --model google/gemini-2.5-flash-lite --n-probes 5
```

- **The image is the archival artifact.** The `Dockerfile` is provided only as
  a convenience for rebuilding; it is *not* guaranteed to produce a
  bit-identical image forever (base-image digests and package registries can
  change). Always cite the published image **by digest**.
- **SBOM & signature.** Each release publishes a Software Bill of Materials
  (`arcus-x-sbom.spdx.json`) and a keyless `cosign` signature — not required
  for most ML papers, but they signal good engineering practice and let you
  verify exactly what is inside the image.
- **Optional GPU image.** `Dockerfile.gpu` builds a local-inference image
  (torch / vLLM / transformers on CUDA). It is *not* part of the reference
  evaluation environment; the paper's results use the API-only runtime image.
- Images are built and published by [`.github/workflows/publish.yml`](.github/workflows/publish.yml),
  which runs the validation suite **inside the container before pushing**.

## Reproducibility

- All randomness derives from a master `seed` via `random.Random`.
- Same `(seed, tier, task_index)` → identical task (grid, actions, rules, GT).
- Different `task_index` → unique task.
- Different `seed` → different task distribution.
- Experiment signature: `sha256` over the task's defining inputs.
- Prompt hashing: `sha256(prompt.encode())` (not Python's non-deterministic `hash()`).
- **Parallel execution is reproducible.** `--parallel-seeds` runs each master
  seed on its own isolated `BenchmarkRunner` (own `GridTaskGenerator`, own
  results matrix) that shares only the evaluator and the raw-stream log. Results
  are therefore identical to a serial run; the only non-deterministic artifact is
  the wall-clock `End Time:` stamp in the `ARCUS-X RUN END` block.
- **Resume is reproducible.** `--resume` re-detects completed probes purely by
  identity (`seed + gravity + tier + z + probe_index + grid_key`), never by file
  order or timestamp, and re-scores them with the same pipeline as a live run, so
  a resumed run is bit-for-bit equivalent (up to cross-seed entry ordering) to a
  fresh run.

## Validation Suite

```bash
python tests/test_arcus_hardening.py
```

Covers:
1. Same seed → same task
2. Different seeds → different tasks
3. Tier 0/1 preserve trajectory relationship
4. Tier 2/3 modify ground truth correctly (recomputed, internally consistent)
5. Parser rejects malformed paths
6. Metrics score perfect / partial / wrong-path-but-right-final
7. Fracture depth works on an accuracy curve
8. Full offline pipeline from public entry-point modules
9. **Difficulty metadata is never leaked to the model prompt** (audit)
10. **Oracle solver achieves perfect accuracy** (upper-bound sanity check)
11. **Random baseline performs near chance** (lower-bound sanity check)
12. **Initial-state baseline scores ~0 on trajectory metrics** (lower-bound)
13. **Difficulty validation experiments confirm the difficulty structure**
    (horizon scaling, tier scaling, seed variance, reproducibility)

### Baseline Evaluators

Three reference evaluators were added to anchor the score range and detect
shortcut leakage:

| Evaluator | File | Purpose |
|-----------|------|---------|
| Random baseline | [`arcus/evaluation/baselines.py`](arcus/evaluation/baselines.py) | Lower bound: random task generation + model eval |
| Oracle solver | [`arcus/evaluation/baselines.py`](arcus/evaluation/baselines.py) | Upper bound: rule-following solver (perfect by construction) |
| Initial-state baseline | [`arcus/evaluation/baselines.py`](arcus/evaluation/baselines.py) | Lower bound: outputs only the initial coordinate |

### Difficulty Validation

[`arcus/experiments/difficulty_validation.py`](arcus/experiments/difficulty_validation.py)
runs controlled experiments confirming the benchmark's difficulty structure:

- **Horizon scaling** — ground-truth length scales with `horizon`.
- **Tier scaling** — Tier 1 preserves the Tier-0 trajectory; Tiers 2/3 change it.
- **Seed variance** — different seeds yield independent tasks.
- **Reproducibility** — same `(seed, tier, task_index)` reproduces identical tasks.

### Task Shortcut Audit

[`arcus/experiments/task_shortcut_audit.py`](arcus/experiments/task_shortcut_audit.py)
scans the **prompt** portion of every generated probe (`axioms` + `question`)
for leakage of `tier`, `gravity`, `experiment_hash`, internal IDs,
output-length hints, **checksum values/formulas**, final-coordinate
constraints, or hidden-answer patterns. The model-facing prompt is built
*only* from the task axioms, transition rules, and the question — no checksum,
final-coordinate constraint, or hidden-answer pattern is ever placed in it, and
scoring is performed downstream from `ground_truth_path`. The validation suite
asserts that **no** probe in the standard sweep leaks difficulty metadata or
answer shortcuts into its prompt.

## File Manifest

| Path | Purpose |
|------|---------|
| `run_framework.sh` | Public shell entry point |
| `quickstart_api.py` | Public Python entry point (legacy compatibility) |
| `requirements.txt` | Runtime dependencies only (flexible `>=` spec) |
| `requirements-local.txt` | Optional heavy deps (torch, vLLM, transformers) — flexible `>=` spec |
| `requirements-lock.txt` | Pinned runtime dependency lock (reproducibility) |
| `requirements-lock-local.txt` | Pinned local-inference dependency lock |
| `requirements-dev.txt` | Dev tooling (`pip-tools`) for regenerating locks |
| `scripts/lock.sh` | Regenerate both lockfiles (Linux/macOS/Git Bash) |
| `scripts/lock.bat` | Regenerate both lockfiles (Windows, long-path workaround) |
| `.github/workflows/ci.yml` | CI: installs pinned lock + runs validation suite |
| `Dockerfile` | Reference evaluation image (runtime/API-only); base pinned by digest |
| `Dockerfile.gpu` | Optional local-inference image (torch/vLLM/transformers on CUDA) |
| `.dockerignore` | Keeps the Docker build context small |
| `.github/workflows/publish.yml` | Builds image, runs suite inside it, pushes to GHCR + SBOM/sign |
| `arcus/tasks/generator.py` | Procedural task generation (SSoT) |
| `arcus/tasks/config.py` | TaskConfig dataclass |
| `arcus/environment/grid.py` | Grid environment, `compute_trajectory` (SSoT) |
| `arcus/evaluation/parser.py` | Path extraction, optimal length estimation |
| `arcus/evaluation/metrics.py` | Trajectory metrics, GBI, efficiency, CRI (secondary robustness index) |
| `arcus/evaluation/evaluator.py` | Model inference (vLLM + API) |
| `arcus/evaluation/token_accounting.py` | Deterministic token counting |
| `arcus/analysis/gravity.py` | DifficultyTier, DifficultyConfig |
| `arcus/analysis/taxonomy.py` | Error taxonomy classification |
| `arcus/analysis/fracture.py` | Fracture point detection |
| `arcus/experiments/benchmark_runner.py` | Orchestrator (bisection search) |
| `arcus/experiments/quickstart.py` | CLI for API-only runs |
| `inter_horizon_analysis.py` | Within-trajectory accuracy and failure-pattern diagnostics |
| `tests/test_arcus_hardening.py` | Validation suite |
| `arcus/evaluation/baselines.py` | Model-free baseline evaluators (oracle / random / initial-state) |
| `arcus/experiments/difficulty_validation.py` | Difficulty validation experiments |
| `arcus/experiments/task_shortcut_audit.py` | Task shortcut / leakage audit |

## Archived / Removed (Legacy)

The following root-level files were **archived** because they duplicated or
conflicted with the `arcus/` package. Their root-level duplicates have since
been **physically removed** from the repository (only the `archive/` legacy
copies remain), so the single source of truth is the `arcus/` package:

- `model_evaluation.py` → `arcus/evaluation/evaluator.py`
- `benchmark_runner.py` → `arcus/experiments/benchmark_runner.py`
- `cross_model_validator.py` (unused, called non-existent `calculate_gs`)
- `fracture_finder.py` → `arcus/analysis/fracture.py`
- `error_taxonomy.py` → `arcus/analysis/taxonomy.py`
- `path_utils.py` → `arcus/evaluation/parser.py` + `arcus/environment/grid.py`
- `generator.py` (empty stub)

Archived copies are in `archive/`.

## Post-Run Analysis & Accounting Invariants

[`post_run_analyzer.py`](post_run_analyzer.py) scores benchmark runs from raw output logs with strict validation invariants:
1. **Validity Partition**: $N_{\text{valid}} + N_{\text{invalid}} = N_{\text{completed}}$
2. **Taxonomy Summation**: $\sum_{\text{taxonomy buckets}} N_i = N_{\text{valid probes}}$
3. **Strict Correctness for "None"**: Unverified or unmapped failure modes are routed to `"Unknown / Unmapped"` rather than silently misclassified as `"None"`.

## Error Code Reference

- **E100** - Exception Token: Used when model output is literally `[EXCEPTION]`. Indicates an invalid response that cannot be evaluated.
- **E101** - Empty Output: Used when model provides no output or an empty string.
- **E102** - Parse Failure: Used when model output cannot be parsed into a valid trajectory.
- **E103** - Provider Error: Used when an external provider error is returned.
- **E104** - Unknown Invalid: Used for any other invalid response type not covered by the specific categories.

## License

MIT
