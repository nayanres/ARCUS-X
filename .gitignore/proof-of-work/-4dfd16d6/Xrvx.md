# ARCUS-X: Deterministic Benchmark for Long-Horizon Sequential Reasoning

ARCUS-X evaluates how well language models maintain state and follow procedural
rules over long horizons. The benchmark generates **deterministic, procedural
grid-trajectory tasks** where the ground truth is computed from explicit
transition rules — never from model output.

## Core Idea

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

### Semantic Gravity: Definition and Scope

**Semantic gravity** is a *deterministic benchmark difficulty parameter*, not a
measure of human cognition or model "understanding". It controls how strongly
the transition semantics are perturbed relative to the Tier-0 baseline:

- `gravity = 0.0` → Tier 0 (baseline rules).
- `gravity = 1.0` → Tier 1 (semantic disruption: labels are replaced by
  pseudowords; the underlying trajectory is identical to Tier 0).
- `gravity = 2.0` → Tier 2 (attribute inversion: y-axis deltas are negated;
  ground truth is recomputed from the inverted rules).
- `gravity = 3.0` → Tier 3 (axiomatic contradiction: a 90° rotation of the
  delta vector; ground truth is recomputed from the rotated rules).

The mapping from gravity to tier is implemented in
[`arcus/analysis/gravity.py`](arcus/analysis/gravity.py) and is fully
deterministic: `tier_for(gravity)` returns the same tier for the same gravity
value across runs. The `disruption_ratio` is `min(1.0, gravity / 3.0)` and is
used only for internal logging/calibration — it is **never** leaked to the
model prompt (see [Reproducibility](#reproducibility) and the validation suite).

**Methodology note:** Difficulty parameters (tier, gravity_target, seed, task_id, experiment_hash) are controlled internally and are not exposed to evaluated models. They are used only by the generation and evaluation code paths; the model-facing prompt contains solely the initial state, grid dimensions, transition rules, action sequence, task question, and required output format.

**Why this matters for reviewers:** Semantic gravity is a *controlled
perturbation of the evaluation environment*, analogous to changing the rules of
a game. It is not a proxy for "reasoning depth" in the cognitive sense. All
tiers share the same grid dimensions, initial state, and action sequence; only
the transition function (and, for Tier 1, the label vocabulary) differs. This
ensures that any observed accuracy drop is attributable to the model's ability
to follow the *provided* rules, not to a change in task length or structure.

## Metrics (all on 0.0–1.0 scale)

**Primary metrics** (directly measure trajectory compliance — these are the
headline results):

- **Step Accuracy** — fraction of steps matching ground truth.
- **Exact Match** — entire predicted path equals ground truth.
- **Continuity** — longest correct prefix / ground-truth length.

**Secondary metrics** (diagnostic / efficiency context — reported alongside
primary metrics but never used as the sole success criterion):

- **First Divergence** — 1-based index of first mismatch.
- **Horizon Compliance** — did the model generate the requested number of steps?
- **Generation Bloat Index (GBI)** — `max(0, actual_tokens - expected_tokens) / expected_tokens`, where `expected_tokens = estimated_optimal_path_length(z)`. Range ≥ 0.
- **Generation Efficiency** — `1 / (1 + GBI)`. Range 0–1.
- **Token Efficiency** (auxiliary) — `1 / (1 + log1p(total_tokens / depth))`; the raw `total_tokens / depth` ratio (token-density) is also reported per tier as `token_density`. Range 0–1.
- **Error Taxonomy** — classifies *why* a trajectory failed:
  `State Tracking Failure`, `Transition Rule Failure`,
  `Semantic Interpretation Failure`, `Horizon Collapse`, `Formatting Failure`.
- **Fracture Depth** — horizon where step accuracy drops below 0.5 and stays
  there (with look-ahead to avoid transient dips).
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

### Local GPU Inference (optional)

ARCUS-X does **not** require local inference — the default path is API-only.
Install `requirements-local.txt` only when running models locally (vLLM /
transformers). The evaluator auto-detects vLLM when available.

```bash
pip install -r requirements-local.txt
```

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
docker pull ghcr.io/<org>/arcus-x:iclr2025@sha256:<digest>

# Verify the environment works (runs the validation suite):
docker run --rm ghcr.io/<org>/arcus-x:iclr2025@sha256:<digest> \
    python tests/test_arcus_hardening.py

# Run a benchmark instead (API-only mode needs an OpenRouter-compatible key):
docker run --rm -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
    ghcr.io/<org>/arcus-x:iclr2025@sha256:<digest> \
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

### Baseline Evaluators (new)

Three reference evaluators were added to anchor the score range and detect
shortcut leakage:

| Evaluator | File | Purpose |
|-----------|------|---------|
| Random baseline | [`arcus/evaluation/baselines.py`](arcus/evaluation/baselines.py) | Lower bound: random task generation + model eval |
| Oracle solver | [`arcus/evaluation/baselines.py`](arcus/evaluation/baselines.py) | Upper bound: rule-following solver (perfect by construction) |
| Initial-state baseline | [`arcus/evaluation/baselines.py`](arcus/evaluation/baselines.py) | Lower bound: outputs only the initial coordinate |

### Difficulty Validation (new)

[`arcus/experiments/difficulty_validation.py`](arcus/experiments/difficulty_validation.py)
runs controlled experiments confirming the benchmark's difficulty structure:

- **Horizon scaling** — ground-truth length scales with `horizon`.
- **Tier scaling** — Tier 1 preserves the Tier-0 trajectory; Tiers 2/3 change it.
- **Seed variance** — different seeds yield independent tasks.
- **Reproducibility** — same `(seed, tier, task_index)` reproduces identical tasks.

### Task Shortcut Audit (new)

[`arcus/experiments/task_shortcut_audit.py`](arcus/experiments/task_shortcut_audit.py)
scans the **prompt** portion of every generated probe (`axioms` + `question`)
for leakage of `tier`, `gravity`, `experiment_hash`, internal IDs, or
output-length hints. The `CHECKSUM` directive lives only in the held-out
`expected_answer` scoring field (it is never concatenated into the prompt), so
it is correctly excluded from the audit. The validation suite asserts that
**no** probe in the standard sweep leaks difficulty metadata into its prompt.

## File Manifest

| Path | Purpose |
|------|---------|
| `run_framework.sh` | Public shell entry point |
| `quickstart_api.py` | Public Python entry point (legacy compat) |
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

## License

MIT
