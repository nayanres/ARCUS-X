#!/usr/bin/env python3
"""Benchmark runner for ARCUS-X.

Orchestrates the full evaluation pipeline:

    TaskGenerator -> GridEnvironment (compute_trajectory) -> Ground Truth
    -> ModelEvaluator (raw output) -> metrics.compare_trajectories
    -> taxonomy.classify_from_result -> fracture.FracturePointFinder

All scoring is trajectory-based (step-level accuracy, exact match, continuity)
on a **0.0-1.0** scale. The model prompt contains only the task axioms/rules,
the action/state information, and the question/request. No checksum, final-
state constraint, or hidden-answer pattern is ever placed in the prompt; the
ground truth never depends on model output.

Latency is recorded per probe as a *systems-level diagnostic only* (it is
affected by provider infrastructure, architecture, batching, and deployment
conditions and is never treated as a model capability score). Latency
diagnostics (ALE, Correct Steps/sec, aggregate + per-horizon latency) are
reported separately and never feed into CRI, GBI, trajectory fidelity, or
fracture metrics.
"""

import os
import ast
import json
import hashlib
import logging
import math
import re
import threading
import concurrent.futures
import multiprocessing
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable

def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for use in filenames.
    
    Replaces '/' and other illegal filename characters with safe alternatives.
    """
    # Replace forward slash with hyphen
    sanitized = model_name.replace('/', '-')
    # Replace other potentially problematic characters
    sanitized = re.sub(r'[\\:*?"<>|]', '-', sanitized)
    # Remove any leading/trailing whitespace or dots
    sanitized = sanitized.strip('. ')
    return sanitized

from arcus.evaluation.metrics import (
    HorizonCompliance,
    GenerationBloatIndex,
    GenerationEfficiency,
    compare_trajectories,
    aggregate_trajectory_results,
    LatencyDiagnostics,
)
from arcus.evaluation.parser import extract_model_path, estimated_optimal_path_length
from arcus.environment.grid import compute_trajectory, format_coord
from arcus.tasks.generator import GridTaskGenerator, TIER_NAMES
from arcus.analysis.gravity import DifficultyConfig, build_gravity_context
from arcus.analysis.taxonomy import classify_from_result, ErrorMode
from arcus.analysis.fracture import FracturePointFinder
from arcus.evaluation.token_accounting import measure_tokens
from arcus.experiments.ux import print_banner, print_metrics_chart, ProgressDisplay
from post_run_analyzer import (
    _parse_coord_block,
    _parse_grid,
    _parse_actions,
    _parse_transition_rules,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Main orchestrator: generates probes, evaluates on models, aggregates results.

    Features:
    - Bisection search (Fracture Hunter) across the depth horizon z.
    - Trajectory-based compliance metrics (0-1 scale, higher is better).
    - Token-density (token_density) reporting per tier.
    - Error taxonomy classification grounded in the real task environment.
    - Latency-normalized efficiency diagnostics (systems-level only; never a
      model capability score and excluded from CRI/GBI/trajectory/fracture).
    """

    # Accuracy floor on the 0.0-1.0 scale (NOT 0-100). Below this the model is
    # considered fractured at that horizon.
    MIN_ACCURACY_THRESHOLD = 0.5
    file_write_lock = threading.Lock()

    def __init__(self,
                 model_name: str,
                 evaluator_class,
                 fracture_finder_class=None,
                 n_probes: int = 5,
                 gravity_levels: Optional[List[float]] = None,
                 seed: int = 42,
                 seeds: Optional[List[int]] = None,
                 api_base: Optional[str] = None,
                 api_key: Optional[str] = None,
                 hard_ceiling: int = 150,
                 grid_size_levels: Optional[List[tuple]] = None,
                 tiers: Optional[List[int]] = None,
                 max_horizon: Optional[int] = None,
                 adaptive_expansion: bool = True):
        """Initialize benchmark parameters.

        ``grid_size_levels`` is an OPTIONAL axis (list of ``(width, height)``
        tuples, e.g. ``[(8, 8), (16, 16), (32, 32)]``). When ``None`` (default)
        the grid dimensions are sampled per-instance exactly as before, so the
        default benchmark behaviour is unchanged. When set, the runner sweeps
        each grid size as an independent axis alongside gravity / tier / depth.

        ``seed`` is the *primary* master seed (kept for backward compatibility
        and used as the single seed when ``seeds`` is not provided). ``seeds``
        is the OPTIONAL multi-seed set used to measure run-to-run variance: the
        benchmark is executed once per seed and the headline metrics (CRI,
        fracture depth) are reported as ``mean ± std`` across seeds. A reviewer
        can therefore see that results are not an artifact of one lucky/unlucky
        seed. Defaults to ``[42, 123, 456, 789]`` (minimum 3 seeds recommended).
        """
        self.model_name = model_name
        self.n_probes = n_probes
        self.gravity_levels = gravity_levels or [0.0, 1.0, 2.0, 3.0]
        self.seed = seed
        # Multi-seed set for variance measurement. Falls back to the single
        # ``seed`` when not explicitly provided, preserving old behaviour.
        self.seeds: List[int] = list(seeds) if seeds else [seed]
        self.generator = GridTaskGenerator(seed=self.seed)
        self.api_base = api_base
        self.api_key = api_key
        self.hard_ceiling = hard_ceiling
        self.grid_size_levels = grid_size_levels
        self.tiers = tiers
        self.max_horizon = max_horizon
        self.adaptive_expansion = adaptive_expansion

        self.evaluator = evaluator_class(
            model_name, api_base=api_base, api_key=api_key
        )
        # Fracture depth is computed exclusively by FracturePointFinder (canonical,
        # 0.0-1.0 scale). All fracture analysis delegates to it -- there is no
        # second/competing definition in the active codebase.
        self.fracture_finder = (fracture_finder_class or FracturePointFinder)()

        self.results_matrix = {}
        self.token_density_matrix = {}
        self.fracture_cache = {}  # (gravity, "final") -> final_fracture_depth
        
        # Model-specific raw log file
        self.sanitized_model_name = sanitize_model_name(model_name)
        # Safer raw-log filename scheme: absolute_raw_data_{provider}_{model_slug}.txt
        # Model identifiers can get messy (provider changes, version bumps, aliases),
        # so we split the "provider/model" form into explicit parts and slugify each.
        parts = model_name.split("/", 1)
        if len(parts) == 2:
            provider_slug, model_slug = parts
        else:
            provider_slug, model_slug = "local", parts[0]
        provider_slug = sanitize_model_name(provider_slug).replace("-", "_")
        model_slug = sanitize_model_name(model_slug).replace("-", "_")
        self.raw_log_filename = (
            f"outputs/absolute_raw_data_{provider_slug}_{model_slug}.txt"
        )
        
        # Run status tracking
        self.expected_probe_count = None
        self.actual_probe_count = 0
        self.run_status = None

        # Experiment manifest (single sidecar every result points back to).
        self.experiment_id = None
        self.manifest_path = None

        # Live progress display (created in run()); None until then so that
        # direct calls to scoring/aggregation helpers stay UX-free (e.g. tests).
        self.progress = None

        logger.info(f"Initialized ARCUS-X BenchmarkRunner for {model_name}")

    # ------------------------------------------------------------------
    # UX helpers (no-ops when no live progress display is active)
    # ------------------------------------------------------------------
    def _progress_set_context(self, tier, gravity, horizon, seed=None):
        if self.progress is not None:
            self.progress.set_context(tier, gravity, horizon, seed=seed if seed is not None else self.seed)

    def _progress_set_status(self, status: str):
        if self.progress is not None:
            self.progress.set_status(status)

    def _progress_increment(self, n: int = 1):
        if self.progress is not None:
            self.progress.increment(n)
        self.actual_probe_count += n

    def _estimate_total_evaluations(self) -> int:
        """Rough upper bound on probe evaluations for the progress denominator.

        The bisection search makes the exact count dynamic, so this is an
        estimate; :class:`~arcus.experiments.ux.ProgressDisplay` grows the
        denominator if the real count exceeds it, keeping the bar honest.
        """
        probes_per_batch = self.n_probes
        tiers = 4
        grids = len(self.grid_size_levels) if self.grid_size_levels else 1
        initial_max = min(40, self.hard_ceiling)
        est_iters = max(6, int(math.log2(initial_max)) + 4)
        n_seeds = len(self.seeds)
        return max(1, n_seeds * len(self.gravity_levels) * est_iters * probes_per_batch * tiers * grids)

    def _grid_size_iter(self):
        """Yield ``(grid_width, grid_height, grid_key)`` for the active grid-size axis.

        When ``grid_size_levels`` is ``None`` (the default) a single
        ``(None, None, "default")`` entry is yielded so generation samples grid
        dimensions exactly as before -- the default benchmark behaviour is
        unchanged. When set, each ``(width, height)`` tuple becomes an independent
        axis alongside gravity / tier / depth.
        """
        if self.grid_size_levels is None:
            yield (None, None, "default")
            return
        for gw, gh in self.grid_size_levels:
            yield (gw, gh, f"{gw}x{gh}")

    @staticmethod
    def _unpack_matrix_key(key):
        """Normalise a results_matrix key to ``(z, gravity, tier, probe_idx, grid_key)``.

        Supports 4-tuple, 5-tuple, and 6-tuple keys (including multi-seed prefix).
        """
        if isinstance(key, tuple):
            if len(key) == 6:
                return (key[1], key[2], key[3], key[4], key[5])
            if len(key) == 5:
                if isinstance(key[4], str):
                    return key
                return (key[1], key[2], key[3], key[4], "default")
            if len(key) == 4:
                z, gravity, tier, probe_idx = key
                return (z, gravity, tier, probe_idx, "default")
        if isinstance(key, str):
            parts = key.split("_")
            if len(parts) == 6:
                return (parts[1], parts[2], parts[3], parts[4], parts[5])
            if len(parts) == 5:
                if "x" in parts[4] or parts[4] == "default":
                    return (parts[0], parts[1], parts[2], parts[3], parts[4])
                return (parts[1], parts[2], parts[3], parts[4], "default")
            if len(parts) == 4:
                return (parts[0], parts[1], parts[2], parts[3], "default")
        return (key[0], key[1], key[2], key[3], "default")

    def _get_expected_completion_tokens(self, z: int) -> Dict[str, Any]:
        """Helper to return expected completion tokens, with fallback mechanism."""
        measured_expected_tokens = estimated_optimal_path_length(z)
        return {
            "tokens": measured_expected_tokens,
            "source": "estimated"
        }

    # ------------------------------------------------------------------
    # Parallel / resume infrastructure
    # ------------------------------------------------------------------
    def _probe_identity(self, seed: int, gravity: float, tier: int, z: int,
                        probe_idx: int, grid_key: str = "default") -> Tuple:
        """Stable, order-independent identity for a single probe.

        Used for resume completion detection. Combines the master seed, gravity,
        tier, horizon (z), probe index and grid key. This is deterministic and
        does NOT depend on file ordering or timestamps.
        """
        return (int(seed), float(gravity), int(tier), int(z), int(probe_idx), str(grid_key))

    @staticmethod
    def parse_completed_probes(raw_stream_path: str) -> Dict[Tuple, Dict[str, Any]]:
        """Parse an existing absolute raw stream and return completed probes.

        Returns a dict keyed by :meth:`_probe_identity` -> parsed entry metadata.
        Completion is determined purely from the structured ``EnvironmentMetadata``
        line of each ``=== RAW STREAM ENTRY ===`` block, so it is deterministic
        and independent of file ordering or timestamps.

        Only entries that carry a ``seed=`` field (written by this version of the
        runner) are considered, so older logs without seed metadata are treated as
        having zero completed probes for resume purposes (safe default).
        """
        completed: Dict[Tuple, Dict[str, Any]] = {}
        if not raw_stream_path or not os.path.exists(raw_stream_path):
            return completed
        try:
            with open(raw_stream_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"[RESUME] Could not read raw stream {raw_stream_path}: {e}")
            return completed

        run_starts = [idx for idx, line in enumerate(lines) if "ARCUS-X RUN START" in line]
        detected_sessions = len(run_starts)

        def _parse_lines(target_lines):
            parsed = {}
            i = 0
            n = len(target_lines)
            combo_counts: Dict[Tuple, int] = {}
            while i < n:
                line = target_lines[i].strip()
                if not line.startswith("=== RAW STREAM ENTRY"):
                    i += 1
                    continue
                i += 1
                # Read until the closing ``====...`` separator.
                block_lines: List[str] = []
                while i < n and not target_lines[i].strip().startswith("==="):
                    block_lines.append(target_lines[i])
                    i += 1
                if i < n:
                    i += 1

                seed = None
                gravity = None
                tier = None
                z = None
                grid_key = "default"
                for bl in block_lines:
                    s = bl.strip()
                    if s.startswith("EnvironmentMetadata:"):
                        # Parse key=value pairs from the metadata line.
                        meta = s.split(":", 1)[1]
                        for kv in meta.split(","):
                            kv = kv.strip()
                            if "=" not in kv:
                                continue
                            k, v = kv.split("=", 1)
                            k = k.strip()
                            v = v.strip()
                            try:
                                if k == "seed":
                                    seed = int(v)
                                elif k == "gravity":
                                    gravity = float(v)
                                elif k == "tier":
                                    tier = int(v)
                                elif k == "z":
                                    z = int(v)
                                elif k == "grid_key":
                                    grid_key = v
                            except ValueError:
                                continue
                if seed is None or gravity is None or tier is None or z is None:
                    # Entry lacks the required identity fields; skip it.
                    continue
                combo = (seed, gravity, tier, z, grid_key)
                probe_idx = combo_counts.get(combo, 0)
                combo_counts[combo] = probe_idx + 1

                ident = (seed, gravity, tier, z, probe_idx, grid_key)
                # Probe index tracked correctly per combo occurrence, reproducing fresh run ordering.
                parsed[ident] = {"seed": seed, "gravity": gravity, "tier": tier,
                                    "z": z, "grid_key": grid_key}
            return parsed

        completed: Dict[Tuple, Dict[str, Any]] = {}
        selected_session = 0
        sessions_to_try = run_starts[::-1] if run_starts else [0]
        if not run_starts:
            sessions_to_try = [0]

        for s_idx in sessions_to_try:
            sub_lines = lines[s_idx:]
            parsed = _parse_lines(sub_lines)
            if parsed:
                completed = parsed
                selected_session = s_idx
                if s_idx != run_starts[-1] if run_starts else 0:
                    logger.info(f"[RESUME] Latest session starting at line {run_starts[-1] if run_starts else 0} has 0 completed probes; fell back to session at line {s_idx}")
                break

        if not completed and detected_sessions > 1:
            logger.info(f"[RESUME] All sessions had 0 completed probes; falling back to parse all entries in {raw_stream_path}")
            completed = _parse_lines(lines)
            selected_session = 0

        logger.info(
            f"[RESUME DIAGNOSTICS]\n"
            f"  Raw stream: {raw_stream_path}\n"
            f"  Detected sessions: {detected_sessions}\n"
            f"  Selected session: {selected_session}\n"
            f"  Parsed completed probes: {len(completed)}"
        )
        if completed:
            logger.info(f"  First 3 completed probe identities: {list(completed.keys())[:3]}")
        return completed

    @staticmethod
    def _parse_raw_stream_blocks(raw_stream_path: str) -> List[Dict[str, Any]]:
        """Parse every ``=== RAW STREAM ENTRY ===`` block into a dict.

        Returns a list of parsed entries (one per block) carrying the structured
        fields needed to re-score a probe: seed, gravity, tier, z, grid_key,
        grid, actions, transition_rules, raw_output, ground_truth, latency_ms,
        input_tokens, output_tokens. Used by resume to reconstruct the prior
        results matrix without re-running the model.
        """
        entries: List[Dict[str, Any]] = []
        if not raw_stream_path or not os.path.exists(raw_stream_path):
            return entries
        try:
            with open(raw_stream_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:  # pragma: no cover - defensive
            return entries

        # Restrict to the latest run session (from the last ARCUS-X RUN START onward)
        start_idx = 0
        for idx, line in enumerate(lines):
            if "ARCUS-X RUN START" in line:
                start_idx = idx
        lines = lines[start_idx:]

        i = 0
        n = len(lines)
        while i < n:
            line = lines[i].strip()
            if not line.startswith("=== RAW STREAM ENTRY"):
                i += 1
                continue
            i += 1
            block_lines: List[str] = []
            while i < n and not lines[i].strip().startswith("==="):
                block_lines.append(lines[i])
                i += 1
            if i < n:
                i += 1

            entry: Dict[str, Any] = {
                "seed": None, "gravity": None, "tier": None, "z": None,
                "grid_key": "default", "grid": "", "actions": "",
                "transition_rules": "", "raw_output": "", "ground_truth": "",
                "latency_ms": None, "input_tokens": None, "output_tokens": None,
            }
            section = None  # "output" | "ground_truth"
            for bl in block_lines:
                s = bl.strip()
                if s.startswith("EnvironmentMetadata:"):
                    meta = s.split(":", 1)[1]
                    for kv in meta.split(","):
                        kv = kv.strip()
                        if "=" not in kv:
                            continue
                        k, v = kv.split("=", 1)
                        k, v = k.strip(), v.strip()
                        try:
                            if k == "seed":
                                entry["seed"] = int(v)
                            elif k == "gravity":
                                entry["gravity"] = float(v)
                            elif k == "tier":
                                entry["tier"] = int(v)
                            elif k == "z":
                                entry["z"] = int(v)
                            elif k == "grid_key":
                                entry["grid_key"] = v
                        except ValueError:
                            continue
                elif s.startswith("Grid:"):
                    entry["grid"] = s.split(":", 1)[1].strip()
                elif s.startswith("Actions:"):
                    entry["actions"] = s.split(":", 1)[1].strip()
                elif s.startswith("TransitionRules:"):
                    entry["transition_rules"] = s.split(":", 1)[1].strip()
                elif s.startswith("Latency(ms):"):
                    try:
                        entry["latency_ms"] = float(s.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif s.startswith("InputTokens:"):
                    try:
                        entry["input_tokens"] = int(s.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif s.startswith("OutputTokens:"):
                    try:
                        entry["output_tokens"] = int(s.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif s.startswith("[MODEL OUTPUT]:"):
                    section = "output"
                    entry["raw_output"] = ""
                elif s.startswith("[GROUND TRUTH EXPECTED]:"):
                    section = "ground_truth"
                    entry["ground_truth"] = ""
                elif s.startswith("-" * 30) or s.startswith("=" * 30):
                    section = None
                elif section == "output":
                    entry["raw_output"] += bl
                elif section == "ground_truth":
                    entry["ground_truth"] += bl
            if entry["seed"] is None or entry["gravity"] is None \
                    or entry["tier"] is None or entry["z"] is None:
                continue
            entries.append(entry)
        return entries

    def _load_prior_results(self, resume_path: Optional[str],
                            master_seed: int) -> Dict[Tuple, Any]:
        """Re-score completed probes for ``master_seed`` from the raw stream.

        Returns a results_matrix-shaped dict keyed by
        ``(z, gravity, tier, probe_idx[, grid_key])`` whose values are the exact
        trajectory-metric bundles that a fresh run would have produced. This lets
        the bisection search (Fracture Hunter) make identical decisions during a
        resume, preserving determinism (requirement #9).

        The scoring pipeline is identical to :meth:`_evaluate_depth_batch` so the
        reconstructed results are equivalent to a live evaluation.
        """
        prior: Dict[Tuple, Any] = {}
        if not resume_path:
            return prior
        entries = self._parse_raw_stream_blocks(resume_path)
        # Recover per-combo probe index (runner emits probes in a fixed order,
        # so the Nth occurrence of a (seed, gravity, tier, z, grid_key) combo is
        # probe_idx N -- matching the fresh-run ordering exactly).
        combo_counts: Dict[Tuple, int] = {}
        for entry in entries:
            if entry["seed"] != master_seed:
                continue
            combo = (entry["seed"], entry["gravity"], entry["tier"],
                     entry["z"], entry["grid_key"])
            probe_idx = combo_counts.get(combo, 0)
            combo_counts[combo] = probe_idx + 1

            z = entry["z"]
            gravity = entry["gravity"]
            tier = entry["tier"]
            grid_key = entry["grid_key"]

            raw_output = entry.get("raw_output", "") or ""
            ground_truth_block = entry.get("ground_truth", "") or ""
            predicted_path = extract_model_path(raw_output)
            ground_truth_path = _parse_coord_block(ground_truth_block)
            traj = compare_trajectories(predicted_path, ground_truth_path)

            gt_len = len(ground_truth_path)
            overlap = min(len(predicted_path), gt_len)
            correct_transitions = sum(
                1 for i in range(overlap) if predicted_path[i] == ground_truth_path[i]
            )

            transition_rules = _parse_transition_rules(entry.get("transition_rules", "") or "")
            actions = _parse_actions(entry.get("actions", "") or "")
            grid_width, grid_height = _parse_grid(entry.get("grid", "") or "")

            tax = classify_from_result(
                predicted_path, ground_truth_path, transition_rules, actions,
                tier, grid_width, grid_height, raw_output,
            )
            error_mode = tax.legacy_mode.value if tax.valid else "Unknown"

            generated_depth = max(0, len(predicted_path) - 1)
            horizon_compliance = HorizonCompliance.calculate(z, generated_depth)

            output_tokens = int(entry.get("output_tokens", 0) or 0)
            baseline_tokens = estimated_optimal_path_length(z)
            gbi = GenerationBloatIndex.calculate(output_tokens, baseline_tokens)
            generation_efficiency = GenerationEfficiency.calculate(gbi)

            result = {
                "z": z,
                "gravity": gravity,
                "tier": tier,
                "grid_key": grid_key,
                "step_accuracy": traj.step_accuracy,
                "exact_match": traj.exact_match,
                "continuity_score": traj.continuity_score,
                "first_divergence": traj.first_divergence,
                "final_state_correct": traj.final_state_correct,
                "predicted_length": traj.predicted_length,
                "ground_truth_length": traj.ground_truth_length,
                "correct_transitions": correct_transitions,
                "error_mode": error_mode,
                "horizon_compliance": horizon_compliance,
                "generation_bloat_index": gbi,
                "generation_efficiency": generation_efficiency,
                "fol_depth": z,
                "latency_ms": entry.get("latency_ms") or 0.0,
                "output_tokens": output_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": output_tokens,
                "input_tokens": int(entry.get("input_tokens", 0) or 0),
                "finish_reason": "truncated" if "[OUTPUT_TRUNCATED]" in raw_output else "stop",
                "raw_output": raw_output,
                "fracture_depth": 0,  # filled in after fracture search completes
            }
            grid_suffix = (grid_key,) if self.grid_size_levels is not None else ()
            prior[(z, gravity, tier, probe_idx) + grid_suffix] = result
        return prior

    # ------------------------------------------------------------------
    # Adaptive concurrency controller (rate-limit aware)
    # ------------------------------------------------------------------
    class _ConcurrencyController:
        """Dynamically reduces/restores worker concurrency on rate limits.

        On a 429 (or equivalent throttling) the active worker count is reduced by
        one (never below 1). After a cooldown without further throttling the
        count is gradually restored toward the original target. Completed work is
        never lost: the controller only changes how many seeds run *concurrently*,
        not what has already been written to the raw stream.
        """

        def __init__(self, target: int, cooldown: float = 30.0):
            self.target = max(1, target)
            self.current = self.target
            self.cooldown = cooldown
            self._last_throttle = 0.0
            self._lock = threading.Lock()

        def on_throttle(self) -> int:
            with self._lock:
                if self.current > 1:
                    self.current -= 1
                    logger.warning(
                        f"[CONCURRENCY] Reduced workers to {self.current} "
                        f"after rate-limit (429)."
                    )
                self._last_throttle = time.time()
                return self.current

        def maybe_restore(self, now: Optional[float] = None) -> int:
            with self._lock:
                now = now if now is not None else time.time()
                if (self.current < self.target
                        and (now - self._last_throttle) >= self.cooldown):
                    self.current += 1
                    logger.info(
                        f"[CONCURRENCY] Restored workers to {self.current} "
                        f"after cooldown."
                    )
                return self.current

        def snapshot(self) -> int:
            with self._lock:
                return self.current

    def _resolve_worker_count(self, parallel_seeds: str, user_max: Optional[int] = None) -> int:
        """Resolve --parallel-seeds into a concrete worker count.

        ``"auto"`` -> min(cpu_count, len(master_seeds), user_max or inf).
        An integer N -> min(N, len(master_seeds)) (never more workers than seeds).
        """
        n_seeds = len(self.seeds)
        if str(parallel_seeds).strip().lower() == "auto":
            workers = min(multiprocessing.cpu_count(), n_seeds)
            if user_max is not None:
                workers = min(workers, max(1, user_max))
            return max(1, workers)
        try:
            requested = int(parallel_seeds)
        except (TypeError, ValueError):
            requested = 1
        return max(1, min(requested, n_seeds))

    def _build_resume_skip_sets(self, resume_path: Optional[str]):
        """Build per-seed skip sets from a prior raw stream.

        Returns ``(skip_sets, completed_count)`` where ``skip_sets`` is a dict
        ``master_seed -> set(probe_identity)`` of probes already completed. Probe
        identity is order-independent (seed, gravity, tier, z, probe_idx, grid_key)
        and is reconstructed deterministically from the per-combo occurrence
        count in the raw stream (matching the runner's own probe ordering).
        """
        skip_sets: Dict[int, set] = {s: set() for s in self.seeds}
        if not resume_path:
            return skip_sets, 0
        completed = self.parse_completed_probes(resume_path)
        completed_count = 0
        for ident in completed.keys():
            seed = ident[0]
            if seed in skip_sets:
                skip_sets[seed].add(ident)
                completed_count += 1

        run_starts_count = 0
        if resume_path and os.path.exists(resume_path):
            with open(resume_path, "r", encoding="utf-8") as f:
                content = f.read()
                run_starts_count = content.count("ARCUS-X RUN START")

        diag_text = (
            f"\nARCUS-X Resume Diagnostics\n"
            f"--------------------------\n"
            f"Raw stream: {resume_path}\n"
            f"Detected sessions: {run_starts_count}\n"
            f"Selected session: {run_starts_count if run_starts_count > 0 else 1}\n"
            f"Parsed completed probes: {len(completed)}\n"
            f"Generated probes: (tracked during execution)\n"
            f"Matching completed probes: {completed_count}\n"
            f"Remaining probes: 0\n"
        )
        print(diag_text)
        logger.info(diag_text)

        logger.info(
            f"[RESUME] Loaded {completed_count} completed probes from {resume_path}"
        )
        return skip_sets, completed_count

    def _execute_seed_loops(self, base_probes: List[Dict],
                            parallel_seeds: str = "1",
                            resume_path: Optional[str] = None,
                            user_max_workers: Optional[int] = None):
        """Run the per-seed benchmark loops, serially or in parallel.

        This is the single orchestration point for multi-seed execution. It
        preserves the exact serial behaviour when ``parallel_seeds == 1`` and
        otherwise dispatches one worker per master seed across a thread pool.

        Determinism guarantees:
          * Each worker instantiates its own ``GridTaskGenerator(master_seed)``
            and its own ``results_matrix`` -- no shared RNG or mutable benchmark
            state. Probes inside a seed are never parallelized.
          * Workers append to the *same* raw stream file under ``file_write_lock``,
            so a crash leaves valid, independently-recoverable results.
          * Resume skip sets are derived from probe identity only (no ordering /
            timestamps), so resumed runs are deterministic and never re-run
            completed probes.
        """
        self._per_seed_metrics: List[Dict[str, float]] = []
        self._per_seed_full: Dict[int, Dict[str, object]] = {}

        n_seeds = len(self.seeds)
        workers = self._resolve_worker_count(parallel_seeds, user_max_workers)
        skip_sets, completed_count = self._build_resume_skip_sets(resume_path)
        self._write_run_start()
        self._write_experiment_manifest()

        # --- Resume progress display (requirement #13) ---
        if resume_path:
            logger.info(
                f"[RESUME] Loaded {completed_count} completed probes; "
                f"resuming remaining work."
            )
            if self.progress is not None:
                # "Remaining" is the number of probes we will (re)run this pass;
                # it is not known until the bisection search expands, so we show
                # the loaded count and let the live probe counter track the rest.
                self.progress.set_resume_info(
                    loaded=completed_count, remaining=0
                )

        # If every probe is already complete (resume onto a finished run), skip
        # execution entirely and let aggregation/reporting run (requirement #9).
        all_done = bool(resume_path) and completed_count > 0 and n_seeds > 0
        # We still must run the bisection to know the totals; but if the resume
        # file already contains a full run we can short-circuit when the caller
        # indicates completion. We detect "fully complete" lazily: if after
        # building skip sets every seed's skip set already covers a full prior
        # run we cannot know without running. So we always run, but skip probes.

        if workers <= 1 or n_seeds <= 1:
            # --- Serial path (preserves original behaviour exactly) ---
            logger.info(f"[SCHEDULER] Serial execution ({n_seeds} seed(s)).")
            merged_matrix: Dict[Tuple, Any] = {}
            merged_fracture: Dict[Tuple, Any] = {}
            for seed_idx, master_seed in enumerate(self.seeds):
                local = self._run_single_seed(
                    seed_idx, master_seed, n_seeds, base_probes,
                    skip_sets.get(master_seed, set()),
                    resume_path=resume_path,
                )
                if local:
                    self._per_seed_metrics.append(local["metrics"])
                    self._per_seed_full[master_seed] = local["full"]
                    for k, v in local["results_matrix"].items():
                        key = (master_seed,) + k if len(self.seeds) > 1 else k
                        merged_matrix[key] = v
                    for k, v in local["fracture_cache"].items():
                        merged_fracture[k] = v
            self.results_matrix = merged_matrix
            self.fracture_cache = merged_fracture
            if self._per_seed_full:
                self.seeds = sorted(list(self._per_seed_full.keys()))
            return

        # --- Parallel path: one worker thread per master seed ---
        logger.info(
            f"[SCHEDULER] Parallel execution: {workers} worker(s) for "
            f"{n_seeds} seed(s)."
        )
        controller = self._ConcurrencyController(workers)
        # A shared, thread-safe map of seed -> (results_matrix, fracture_cache,
        # token_density_matrix, per_seed metrics). Workers populate their own
        # slot; the main thread merges after join.
        seed_results: Dict[int, Dict[str, Any]] = {}
        results_lock = threading.Lock()

        def _worker(seed_idx: int, master_seed: int):
            logger.info(
                f"[WORKER] Started seed {master_seed} "
                f"({seed_idx + 1}/{n_seeds})."
            )
            try:
                # Each worker runs on its OWN isolated runner instance so there is
                # no shared mutable benchmark state (seed/generator/matrix) across
                # threads -- this is what keeps parallel execution deterministic
                # and identical to a serial run (requirement #3 / #4).
                isolated = self._spawn_for_seed(master_seed)
                local = isolated._run_single_seed(
                    seed_idx, master_seed, n_seeds, base_probes,
                    skip_sets.get(master_seed, set()), controller=controller,
                    resume_path=resume_path,
                )
                with results_lock:
                    # The isolated runner tracks its own probe counter; carry it
                    # so the parent can report the true completed total (matching
                    # the serial path, where self.actual_probe_count spans seeds).
                    local["actual_probe_count"] = isolated.actual_probe_count
                    seed_results[master_seed] = local
                logger.info(f"[WORKER] Completed seed {master_seed}.")
            except Exception as e:  # pragma: no cover - defensive
                logger.error(f"[WORKER] Seed {master_seed} failed: {e}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(_worker, seed_idx, master_seed)
                for seed_idx, master_seed in enumerate(self.seeds)
            ]
            # Track completion for the live "Seeds: X/Y complete" display.
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                try:
                    fut.result()
                except Exception:  # pragma: no cover - defensive
                    pass
                done += 1
                if self.progress is not None:
                    self.progress.set_seed_progress(done, n_seeds, workers)

        # --- Merge worker outputs into the runner's primary structures ---
        # The first seed becomes the "primary" results_matrix used by the
        # aggregation/reporting pipeline; all seeds are also kept in
        # ``_per_seed_full`` for variance reporting.
        merged_matrix: Dict[Tuple, Any] = {}
        merged_fracture: Dict[Tuple, Any] = {}
        for master_seed in self.seeds:
            local = seed_results.get(master_seed)
            if local is None:
                # Worker crashed; skip its contribution (completed probes from
                # other seeds remain valid).
                logger.warning(
                    f"[WORKER] No results for seed {master_seed} (worker crashed)."
                )
                continue
            for k, v in local["results_matrix"].items():
                key = (master_seed,) + k if len(self.seeds) > 1 else k
                merged_matrix[key] = v
            for k, v in local["fracture_cache"].items():
                merged_fracture[k] = v
            self._per_seed_metrics.append(local["metrics"])
            self._per_seed_full[master_seed] = local["full"]
        # Primary matrix = union (aggregation reads self.results_matrix).
        self.results_matrix = merged_matrix
        self.fracture_cache = merged_fracture
        if self._per_seed_full:
            self.seeds = sorted(list(self._per_seed_full.keys()))
        self.token_density_matrix = {}
        # The parent's probe counter is not incremented by the isolated workers
        # (each tracks its own). Sum their completed counts so run-status and the
        # RUN END block report the true total and match the serial path exactly
        # (requirement #3: parallel must not change results/output). Note the
        # results_matrix key space is per-(z, gravity, tier, probe_idx) and does
        # not include the seed, so its length is not a reliable completed count
        # when multiple seeds are merged.
        self.actual_probe_count = sum(
            (local.get("actual_probe_count", 0) for local in seed_results.values()
             if local),
            start=0,
        )

    def _spawn_for_seed(self, master_seed: int) -> "BenchmarkRunner":
        """Create an isolated runner for a single master seed.

        Parallel workers must NOT share ``self`` -- doing so races on
        ``self.seed``, ``self.generator``, ``self.results_matrix`` and
        ``self.fracture_cache`` and corrupts both probe generation and the raw
        stream. Each worker therefore gets its own runner instance that:

        * owns its own ``GridTaskGenerator(master_seed)`` and results state
          (no shared RNG / mutable benchmark state -- requirement #3),
        * shares the already-initialised ``evaluator`` (no re-init cost and a
          single inference client), and
        * appends to the SAME raw stream file (lock-protected writes) so the
          final log is the union of every seed's output.

        The serial path does not use this; it runs directly on ``self``.
        """
        r = BenchmarkRunner(
            model_name=self.model_name,
            evaluator_class=self.evaluator.__class__,
            fracture_finder_class=self.fracture_finder.__class__,
            n_probes=self.n_probes,
            gravity_levels=self.gravity_levels,
            seeds=[master_seed],
            api_base=self.api_base,
            api_key=self.api_key,
            hard_ceiling=self.hard_ceiling,
            grid_size_levels=self.grid_size_levels,
        )
        # Reuse the parent's (already constructed) evaluator instead of building
        # a second one -- avoids duplicate model/API client initialisation.
        r.evaluator = self.evaluator
        # All seeds append to the one shared raw stream file.
        r.raw_log_filename = self.raw_log_filename
        return r

    def _run_single_seed(self, seed_idx: int, master_seed: int, n_seeds: int,
                         base_probes: List[Dict], skip_set: set,
                         controller: Optional["BenchmarkRunner._ConcurrencyController"] = None,
                         resume_path: Optional[str] = None
                         ) -> Dict[str, Any]:
        """Execute the full benchmark for a single master seed.

        Runs deterministically: its own ``GridTaskGenerator(master_seed)`` and its
        own ``results_matrix``. Returns a dict of local artifacts for the parallel
        merge. In serial mode the side effects on ``self`` are used directly.
        """
        # Reset generator + results for this seed (fresh, reproducible).
        self.seed = master_seed
        self.generator = GridTaskGenerator(seed=master_seed)
        local_matrix: Dict[Tuple, Any] = {}
        local_fracture: Dict[Tuple, Any] = {}
        self.token_density_matrix = {}

        # On resume, pre-load the previously completed probes for this seed so
        # the bisection search sees the full (resumed) dataset and makes
        # identical fracture-depth decisions. These are re-scored from the raw
        # stream using the same pipeline as a live run (deterministic).
        if skip_set:
            prior = self._load_prior_results(
                resume_path if resume_path else self.raw_log_filename,
                master_seed,
            )
            local_matrix.update(prior)
            logger.info(
                f"[RESUME] Seed {master_seed}: loaded {len(prior)} "
                f"prior result(s) from raw stream."
            )

        # In serial mode we operate directly on self; in parallel mode we use the
        # local structures and return them. To keep one code path, always use the
        # local structures, then copy onto self when running serially.
        self.results_matrix = local_matrix
        self.fracture_cache = local_fracture

        # Wire the evaluator's rate-limit callback to the adaptive controller so a
        # 429 temporarily reduces concurrency (requirement #11). Provider
        # throttling is NOT counted as a benchmark failure.
        if controller is not None and hasattr(self.evaluator, "status_callback"):
            original_cb = self.evaluator.status_callback

            def _throttle_aware(status: str):
                if status and "rate limit" in status.lower():
                    controller.on_throttle()
                if original_cb:
                    try:
                        original_cb(status)
                    except Exception:  # pragma: no cover - defensive
                        pass
            self.evaluator.status_callback = _throttle_aware

        if skip_set:
            logger.info(
                f"[RESUME] Seed {master_seed}: skipping {len(skip_set)} "
                f"already-completed probe(s); resuming remaining work."
            )

        self._evaluate_all_dimensions(
            base_probes, results_matrix=local_matrix, skip_set=skip_set
        )

        # Capture this seed's headline metrics for variance reporting.
        seed_cri = self.compute_cri()
        cri_val = float(seed_cri.get("cri", 0.0))
        seed_fracture = min(
            (fd for (_, label), fd in local_fracture.items()
             if label == "final" and fd > 0),
            default=0,
        )
        metrics = {"cri": cri_val, "fracture_depth": float(seed_fracture)}
        full = {
            "cri": seed_cri,
            "cri_value": cri_val,
            "fracture_depth": float(seed_fracture),
        }
        logger.info(
            f"[SEED {seed_idx + 1}/{n_seeds}] master_seed={master_seed} "
            f"CRI={cri_val:.3f}, FractureDepth={seed_fracture}"
        )
        return {
            "results_matrix": local_matrix,
            "fracture_cache": local_fracture,
            "metrics": metrics,
            "full": full,
        }

    def run(self, output_filepath: str = "benchmark_results.json",
            grid_size_levels: Optional[List[tuple]] = None,
            parallel_seeds: str = "1",
            resume_path: Optional[str] = None,
            user_max_workers: Optional[int] = None) -> Dict:
        """Execute full benchmark with bisection search and complexity scaling.

        New parameters (runtime/scheduling only -- no benchmark-semantics change):

        ``parallel_seeds``
            ``"1"`` (default, serial), an integer N, or ``"auto"``. Never creates
            more workers than master seeds.
        ``resume_path``
            Path to an existing absolute raw stream file. Completed probes are
            detected by identity and skipped; only missing probes are run. The run
            appends to the same raw stream file by default.
        ``user_max_workers``
            Optional explicit cap used by ``auto`` mode (e.g. a configured limit).
        """
        if grid_size_levels is not None:
            self.grid_size_levels = grid_size_levels
        logger.info(f"Starting benchmark execution for {self.model_name}")
        logger.info(f"Parameters: {self.n_probes} probes, gravities={self.gravity_levels}"
                    f"{', grid_sizes=' + str(self.grid_size_levels) if self.grid_size_levels else ''}")

        logger.info("[1/3] Generating base probe templates...")
        base_probes = self._generate_base_probes()

        # --- UX: startup banner (self-test result drives the PASS/FAIL line) ---
        self_test_passed = True
        try:
            self.run_self_test(base_probes)
            self.run_metric_unit_tests()
        except Exception:
            self_test_passed = False
            print_banner(self_test_passed=False, model_name=self.model_name)
            raise
        print_banner(self_test_passed=True, model_name=self.model_name)

        # --- UX: live progress display ---
        self.progress = ProgressDisplay(
            total=self._estimate_total_evaluations(),
            model_name=self.model_name,
        )
        # Route evaluator rate-limit events onto the live status line.
        if hasattr(self.evaluator, "status_callback"):
            self.evaluator.status_callback = self.progress.set_status

        # Route evaluator API-compatibility adaptations onto the raw stream so
        # every parameter adjustment (e.g. max_tokens -> max_completion_tokens)
        # is recorded with structured metadata as it happens. This preserves
        # reproducibility without modifying scoring / trajectory logic.
        if hasattr(self.evaluator, "adaptation_log"):
            self.evaluator.adaptation_log._on_adapt = self._write_compatibility_adaptation

        # Suppress noisy INFO logging during the live phase so the progress bar
        # (stdout) renders cleanly; logger writes to stderr and on a shared
        # terminal the cursor control needs stderr to stay quiet. Restored in
        # a finally block so it can never leak.
        root_logger = logging.getLogger()
        previous_level = root_logger.level
        root_logger.setLevel(logging.WARNING)
        try:
            logger.info("[2/3] Running Bisection Search (Fracture Hunter)...")
            # --- Multi-seed variance measurement ---
            # The benchmark is executed once per master seed. Each seed produces
            # its own results matrix; the headline metrics (CRI, fracture depth)
            # are aggregated as mean +/- std across seeds so a reviewer can see
            # the result is not an artifact of one lucky/unlucky seed.
            #
            # Seeds are independent experiments and may run in parallel
            # (--parallel-seeds) and/or resume from a prior raw stream
            # (--resume). The orchestration lives in _execute_seed_loops so the
            # serial path is preserved exactly when parallel_seeds == 1.
            self._execute_seed_loops(
                base_probes,
                parallel_seeds=parallel_seeds,
                resume_path=resume_path,
                user_max_workers=user_max_workers,
            )
        finally:
            root_logger.setLevel(previous_level)

        self._progress_set_status("Computing CRI...")
        logger.info("[3/3] Aggregating composite metrics...")
        aggregated = self._compute_aggregated_metrics()

        self._print_horizon_summary()
        self._print_gravity_summary()

        self._print_structured_report(aggregated)

        # The bisection search makes the exact probe count dynamic; the real
        # expected total is the number of entries actually produced in the
        # results matrix. Use it so PARTIAL detection is accurate.
        self.expected_probe_count = len(self.results_matrix)

        # Determine run status based on completed vs expected probe counts
        if self.actual_probe_count == 0:
            self.run_status = "FAILED"
        elif self.expected_probe_count is not None and self.actual_probe_count < self.expected_probe_count:
            self.run_status = "PARTIAL"
        else:
            self.run_status = "COMPLETE"

        # Generate clean terminal summary report
        self._print_summary_report(aggregated)

        # Export metrics to JSON
        self._export_metrics_json(aggregated)

        # Partial run reporting if incomplete
        if self.run_status != "COMPLETE":
            self._print_partial_run_report()

        self._progress_set_status("Saving results...")
        self.save_results(output_filepath)

        # Append an "ARCUS-X RUN END" block so the post-run analyzer can read the
        # run's completion status explicitly (no inference needed for PARTIAL /
        # crashed runs). Failures = probes with zero step accuracy or no output.
        run_failures = sum(
            1 for r in self.results_matrix.values()
            if self._coerce_float(r.get("step_accuracy", 0.0)) == 0.0
            or not r.get("raw_output")
        )
        run_reason = ""
        if self.run_status == "PARTIAL":
            run_reason = "Run stopped before completing all probes"
        elif self.run_status == "FAILED":
            run_reason = "No probes completed"

        # Refresh the manifest so any API-compatibility capabilities/adaptations
        # discovered during evaluation are persisted (the manifest is written
        # once at run start, before discovery happens).
        try:
            self._write_experiment_manifest()
        except Exception:  # pragma: no cover - defensive
            pass

        self._write_run_end(
            status=self.run_status or "UNKNOWN",
            completed=self.actual_probe_count,
            total=self.expected_probe_count or 0,
            failures=run_failures,
            reason=run_reason,
        )

        # Tear down the live display and render the final evaluation chart.
        if self.progress is not None:
            self.progress.finish()
        print_metrics_chart(aggregated)

        report = self._generate_report(aggregated)

        logger.info(
            f"Benchmark complete. Step accuracy: {report['metrics']['step_accuracy']:.1f}%, "
            f"Exact match: {report['metrics']['exact_match']:.1f}%"
        )

        logger.info("[4/4] Running Error Taxonomy Classification...")
        try:
            taxonomy_summary = []
            for gravity in self.gravity_levels:
                analysis = self.fracture_finder.parse_matrix_telemetry(
                    self.results_matrix, gravity
                )
                if analysis.get("depth_error_distribution"):
                    taxonomy_summary.append({
                        "gravity": gravity,
                        "distribution": analysis["depth_error_distribution"]
                    })

            if taxonomy_summary:
                self._append_debug_log(
                    "[TAXONOMY DIAGNOSTIC] Classification Compilation:\n"
                    + json.dumps(taxonomy_summary, indent=2)
                )
            else:
                self._append_debug_log(
                    "[TAXONOMY DIAGNOSTIC] 0 classifications compiled due to "
                    "sequence truncation."
                )

        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"[TAXONOMY DIAGNOSTIC] Pipeline crashed: {e}")
            self._append_debug_log(
                f"[TAXONOMY DIAGNOSTIC] 0 classifications compiled due to "
                f"taxonomy processing error: {e}"
            )

        return report

    def run_baselines(self, baseline_names: Optional[List[str]] = None,
                     n_tasks: int = 5, horizon: int = 8,
                     gravity_levels: Optional[List[float]] = None,
                     grid_size_levels: Optional[List[tuple]] = None) -> Dict:
        """Run model-free baselines and aggregate their trajectory performance.

        Baselines are OPTIONAL: they are never required for normal model
        evaluation and are only included in a report when explicitly enabled
        (e.g. ``run_framework.sh --baselines``). They establish reference points
        -- chance (random), trivial (initial-state) and upper bound (oracle) --
        against which a model's trajectory accuracy can be interpreted.

        Returns a dict keyed by baseline name. Each entry contains
        ``baseline_name``, ``trajectory_accuracy``, ``step_accuracy``,
        ``exact_match`` and ``fracture_depth`` (``None`` for baselines, which do
        not perform a depth sweep).
        """
        from arcus.evaluation.baselines import available_baselines
        from arcus.evaluation.metrics import mean, ARCUSRobustnessIndex

        registry = available_baselines()
        if baseline_names is None:
            baseline_names = list(registry.keys())
        gravity_levels = gravity_levels or [0.0]

        report: Dict[str, Dict[str, object]] = {}
        for name in baseline_names:
            baseline = registry.get(name)
            if baseline is None:
                logger.warning(f"Unknown baseline '{name}'; skipping.")
                continue

            traj_accs: List[float] = []
            step_accs: List[float] = []
            cont_accs: List[float] = []
            exacts: List[float] = []
            ge_accs: List[float] = []
            tier_step = {0: [], 1: [], 2: [], 3: []}
            grid_levels = grid_size_levels if grid_size_levels is not None else self.grid_size_levels
            for gravity in gravity_levels:
                tier = DifficultyConfig().tier_for(gravity)
                for i in range(n_tasks):
                    for gw, gh in (grid_levels or [(None, None)]):
                        task = self.generator.generate(
                            tier=tier, task_index=i, horizon=horizon,
                            grid_width=gw, grid_height=gh
                        )
                    res = baseline.evaluate(task)
                    traj_accs.append(float(res["trajectory_accuracy"]))
                    step_accs.append(float(res["step_accuracy"]))
                    cont_accs.append(float(res.get("continuity_score", 0.0)))
                    exacts.append(1.0 if res["exact_match"] else 0.0)
                    if tier in tier_step:
                        tier_step[tier].append(float(res["step_accuracy"]))

                    # Model-free generation efficiency: measure the token cost of
                    # the baseline's own trajectory against the optimal-path
                    # estimate, using the same accounting pipeline as the model
                    # evaluation path. This lets CRI's efficiency term reflect
                    # the oracle (which emits the minimal correct path) instead of
                    # defaulting to 0.0 for baselines.
                    try:
                        predicted_traj = baseline.generate_trajectory(task)
                        traj_text = "\n".join(predicted_traj)
                        tok = measure_tokens("", traj_text)
                        actual_tokens = int(tok.get("total_tokens") or 0)
                        expected_tokens = estimated_optimal_path_length(horizon)
                        gbi = GenerationBloatIndex.calculate(actual_tokens, expected_tokens)
                        ge_accs.append(GenerationEfficiency.calculate(gbi))
                    except Exception:  # pragma: no cover - defensive
                        pass

            # Secondary CRI summary (does NOT replace the trajectory metrics).
            step_acc = mean(step_accs) if step_accs else 0.0
            cont_acc = mean(cont_accs) if cont_accs else 0.0
            exact_acc = mean(exacts) if exacts else 0.0
            ge_acc = mean(ge_accs) if ge_accs else 0.0
            cri_summary = ARCUSRobustnessIndex.compute(
                step_accuracy=step_acc,
                continuity_score=cont_acc,
                exact_match=exact_acc,
                horizon_points=[(float(horizon), step_acc)],
                generation_efficiency=ge_acc,
            )
            report[name] = {
                "baseline_name": name,
                "trajectory_accuracy": (mean(traj_accs) * 100.0) if traj_accs else 0.0,
                "step_accuracy": step_acc,
                "exact_match": (exact_acc * 100.0),
                "fracture_depth": None,
                "cri": cri_summary.as_dict(),
            }
        return report

    def _generate_base_probes(self) -> List[Dict]:
        """Generate a small set of deterministic base probe templates.

        These templates are expanded per (tier, depth, grid) inside the
        evaluation loop; the seed distribution is recorded in the manifest for
        reproducibility.
        """
        probes = []
        for i in range(self.n_probes):
            probes.append({
                "id": f"base_probe_{i}",
                "seed": self.seed,
                "index": i,
            })
        return probes

    def _task_index_for(self, probe_id: str, tier: int, z: int) -> int:
        """Derive a stable per-instance task index from probe/tier/depth."""
        h = hashlib.sha256(f"{probe_id}|{tier}|{z}|{self.seed}".encode("utf-8")).hexdigest()[:8]
        return int(h, 16) % 100000

    def _shift_semantic_gravity(self, z: int, gravity_target: float) -> Dict[str, Any]:
        """Build the gravity-shifted semantic layer for a probe."""
        return {
            "gravity_target": gravity_target,
            "z": z,
        }

    def _apply_ood_tier(self, tier: int, gravity_target: float) -> Dict[str, Any]:
        """Build the out-of-distribution tier perturbation layer."""
        return {
            "tier": tier,
            "tier_name": TIER_NAMES[tier] if tier < len(TIER_NAMES) else f"tier_{tier}",
        }

    def _build_probe_from_task(self, task, gravity_target: float) -> Dict[str, Any]:
        """Assemble a probe payload from a generated GridTask."""
        return {
            "id": getattr(task, "task_id", "task"),
            "seed": self.seed,
            "gravity_target": gravity_target,
            "fol_depth": getattr(task, "horizon", 1),
            "tier": getattr(task, "tier", 0),
            "question": getattr(task, "question", ""),
            "axioms": getattr(task, "axioms", []),
            "ground_truth_path": list(getattr(task, "ground_truth_trajectory", [])),
            "initial_state": getattr(task, "initial_state", None),
            "actions": getattr(task, "actions", None),
            "transition_rules": getattr(task, "transition_rules", None),
            "grid_width": getattr(task, "grid_width", None),
            "grid_height": getattr(task, "grid_height", None),
            "tier_name": getattr(task, "tier_name", None),
            "experiment_hash": getattr(task, "experiment_hash", None),
        }

    def _evaluate_all_dimensions(self, base_probes: List[Dict],
                                  user_hard_ceiling: Optional[int] = None,
                                  results_matrix: Optional[Dict] = None,
                                  skip_set: Optional[set] = None):
        """
        Evaluates dimensions via a Bisection Search strategy.
        Dynamically doubles the depth horizon window if the model proves invariant.

        ``results_matrix`` and ``skip_set`` are optional so the same method can be
        driven by a parallel worker (own matrix, shared skip set for resume) or by
        the legacy serial path (defaults to ``self.results_matrix`` / no skip).
        """
        results_matrix = results_matrix if results_matrix is not None else self.results_matrix
        skip_set = skip_set if skip_set is not None else set()
        ABSOLUTE_HARD_CEILING = user_hard_ceiling if user_hard_ceiling is not None else self.hard_ceiling

        INITIAL_MAX_DEPTH = min(40, ABSOLUTE_HARD_CEILING)
        logger.info(f"Execution initialized with ABSOLUTE_HARD_CEILING = {ABSOLUTE_HARD_CEILING}")

        # --- Baseline calibration anchor ---
        logger.info(
            "\n=== [BASELINE CALIBRATION ANCHOR] Running simplified calibration "
            "at z=3, Tier 1, Gravity 0.0 ==="
        )
        try:
            anchor_probes = base_probes[:1]
            anchor_acc, _ = self._evaluate_depth_batch(
                anchor_probes, gravity_target=0.0, z=3, tiers=self.tiers if self.tiers is not None else [1],
                results_matrix=results_matrix, skip_set=skip_set,
            )
            logger.info(f"  [BASELINE CALIBRATION ANCHOR] Result Accuracy: {anchor_acc:.3f}")
        except Exception as e:
            msg = f"Benchmark Stopped - Check {e}"
            logger.error(f"  [BASELINE CALIBRATION ANCHOR] {msg}")
            raise Exception(msg)

        for gravity_target in self.gravity_levels:
            if not self.adaptive_expansion:
                target_z = self.max_horizon if self.max_horizon is not None else ABSOLUTE_HARD_CEILING
                logger.info(f"\n--- Evaluating Gravity: {gravity_target} (No adaptive expansion, horizon={target_z}) ---")
                batch_acc, context_exhausted_count = self._evaluate_depth_batch(
                    base_probes, gravity_target, target_z,
                    tiers=self.tiers,
                    results_matrix=results_matrix, skip_set=skip_set,
                )
                final_fracture_depth = target_z if batch_acc >= self.MIN_ACCURACY_THRESHOLD else 0
                self.fracture_cache[(gravity_target, "final")] = final_fracture_depth
                logger.info(
                    f"--- Final Profile for Gravity {gravity_target}: "
                    f"Fracture Depth z={final_fracture_depth} ---"
                )
                continue

            logger.info(f"\n--- Starting Bisection Search (Gravity: {gravity_target}) ---")

            low = 1
            high = INITIAL_MAX_DEPTH
            final_fracture_depth = 0
            horizon_accuracies = {}
            max_iter = max(50, int(math.log2(self.hard_ceiling)) * 5)
            iter_count = 0

            while low <= high and iter_count < max_iter:
                iter_count += 1
                mid_z = (low + high) // 2
                logger.info(
                    f"Testing depth window [{low}-{high}]. Current Target Horizon z: {mid_z}"
                )

                batch_acc, context_exhausted_count = self._evaluate_depth_batch(
                    base_probes, gravity_target, mid_z,
                    tiers=self.tiers,
                    results_matrix=results_matrix, skip_set=skip_set,
                )
                horizon_accuracies[mid_z] = batch_acc

                if batch_acc >= self.MIN_ACCURACY_THRESHOLD:
                    logger.info(
                        f"  Result: SUCCESS (Acc: {batch_acc:.3f} >= Floor). Shifting deeper."
                    )
                    low = mid_z + 1

                    if mid_z >= high and high < ABSOLUTE_HARD_CEILING:
                        old_high = high
                        high = min(high * 2, ABSOLUTE_HARD_CEILING)
                        logger.info(
                            f"  [HORIZON EXPANSION] Model invariant. Doubling search "
                            f"space: [{old_high} -> {high}]"
                        )
                else:
                    logger.info(
                        f"  Result: FRACTURE (Acc: {batch_acc:.3f} < Floor). "
                        f"Retreating search profile."
                    )
                    final_fracture_depth = mid_z
                    high = mid_z - 1

            if iter_count >= max_iter:
                logger.warning(
                    f"  [MAX ITERATION GUARD] Bisection loop terminated after "
                    f"{max_iter} iterations for gravity {gravity_target}"
                )

            if final_fracture_depth == 0 and low > ABSOLUTE_HARD_CEILING:
                final_fracture_depth = ABSOLUTE_HARD_CEILING
                logger.info(f"  Result: COMPLETE SATURATION at z={ABSOLUTE_HARD_CEILING}")

            self.fracture_cache[(gravity_target, "final")] = final_fracture_depth
            logger.info(
                f"--- Final Profile for Gravity {gravity_target}: "
                f"Fracture Depth z={final_fracture_depth} ---"
            )

    def _write_run_start(self):
        """Append an ``ARCUS-X RUN START`` block to the absolute raw stream.

        The post-run analyzer is a pure log consumer (no API, no runner state),
        so each run persists its own identity and intended probe total. This
        lets the analyzer:

          * separate multiple runs of the same model (each gets a unique Run ID
            and its own START/END block, appended after the previous run),
          * detect PARTIAL runs (completed < total) without guessing, and
          * report per-run accuracy/cost without inference.

        The block is always *appended* (never truncated), so re-running the same
        model simply adds a new run segment to the same file.
        """
        import datetime

        run_id = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        expected = self._estimate_total_evaluations()
        try:
            with open(self.raw_log_filename, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 50 + "\n")
                f.write("ARCUS-X RUN START\n")
                f.write(f"Run ID: {run_id}\n")
                f.write(f"Model: {self.model_name}\n")
                f.write(f"Total Probes: {expected}\n")
                f.write("=" * 50 + "\n")
        except Exception as e:  # pragma: no cover - defensive logging
            logger.warning(f"[RAW LOG] Could not write run start: {e}")

    def _write_compatibility_adaptation(self, adaptation) -> None:
        """Append an API compatibility adaptation block to the raw stream.

        Called by the evaluator's :class:`~arcus.evaluation.api_compat.AdaptationLog`
        whenever a provider/model requires a parameter change (name fallback or
        value omission). Recording it in the raw stream (with structured
        metadata) keeps the post-run analyzer able to report exactly what
        compatibility changes occurred, without any model-name hardcoding or
        scoring changes.
        """
        try:
            with open(self.raw_log_filename, "a", encoding="utf-8") as f:
                f.write("\n" + "-" * 30 + "\n")
                f.write("EXECUTION COMPATIBILITY\n")
                f.write(f"Provider: {adaptation.provider}\n")
                f.write(f"Model: {adaptation.model}\n")
                f.write("Adaptations:\n")
                f.write(f"- {adaptation.adjustment}\n")
                if adaptation.reason:
                    f.write(f"  (reason: {adaptation.reason})\n")
                f.write("-" * 30 + "\n")
        except Exception as e:  # pragma: no cover - defensive logging
            logger.warning(f"[RAW LOG] Could not write compatibility adaptation: {e}")

    def _write_run_end(self, status: str, completed: int, total: int,
                       failures: int, reason: str = ""):
        """Append an ``ARCUS-X RUN END`` block recording the run's outcome.

        Writing the completion status explicitly (rather than letting the
        analyzer infer it) makes PARTIAL / incomplete runs unambiguous even when
        the log was cut short by an API crash or budget exhaustion.
        """
        import datetime

        end_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        try:
            with open(self.raw_log_filename, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 50 + "\n")
                f.write("ARCUS-X RUN END\n")
                f.write(f"Status: {status}\n")
                f.write(f"Completed: {completed}/{total}\n")
                f.write(f"Failures: {failures}\n")
                if reason:
                    f.write(f"Reason: {reason}\n")
                f.write(f"End Time: {end_time}\n")
                f.write("=" * 50 + "\n")
        except Exception as e:  # pragma: no cover - defensive logging
            logger.warning(f"[RAW LOG] Could not write run end: {e}")

    def _write_experiment_manifest(self):
        """Create (or refresh) a single experiment manifest for this run.

        A manifest is a small JSON sidecar that every result (raw log, metrics
        JSON, results JSON) points back to. It records the experiment identity,
        software versions, and the exact parameters used, so a result file is
        always reproducible and attributable even if model aliases or provider
        names change later. The manifest is written once per ``run()`` call and
        overwritten on re-run (it describes the *intended* experiment, not the
        per-run outcome -- that lives in the RUN START/END blocks).
        """
        import json as _json
        import datetime as _dt
        from arcus import __version__ as arcus_version

        # Derive a stable experiment id from the model + a timestamp.
        run_stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        parts = self.model_name.split("/", 1)
        if len(parts) == 2:
            provider_slug, model_slug = parts
        else:
            provider_slug, model_slug = "local", parts[0]
        provider_slug = sanitize_model_name(provider_slug).replace("-", "_")
        model_slug = sanitize_model_name(model_slug).replace("-", "_")
        experiment_id = f"ARCUS-X-exp-{provider_slug}_{model_slug}_{run_stamp}"

        manifest = {
            "experiment_id": experiment_id,
            "benchmark_version": arcus_version,
            "models": [self.model_name],
            "parameters": {
                "probes": self.n_probes,
                "seed": self.seed,
                "seeds": list(self.seeds),
                "gravity_levels": list(self.gravity_levels),
                "horizons": self._horizon_range(),
                "grid_size_levels": self.grid_size_levels,
            },
            # Task Seed Manifest: the derived per-instance ``task_seed`` for each
            # base probe. These are the actual instance selectors fed to the
            # generator, so the seed distribution is auditable and reproducible
            # without rebuilding the generator. Larger sampling can be added
            # later by extending this manifest.
            "task_seed_manifest": self._build_task_seed_manifest(),
            "software": {
                "arcus_version": arcus_version,
                "taxonomy_version": "2.0",
            },
            # Discovered provider/model API capabilities (full capability
            # negotiation record: token parameter, temperature/top_p/penalty
            # support and value constraints) and the compatibility adaptations
            # made during the run. These are populated dynamically (no hardcoded
            # model table) and let the post-run analyzer report exactly what
            # parameter changes occurred. Empty until discovery happens during
            # evaluation.
            "api_compatibility": {
                "capabilities": (
                    self.evaluator.capability_cache.snapshot()
                    if hasattr(self.evaluator, "capability_cache") else []
                ),
                "adaptations": (
                    self.evaluator.adaptation_log.to_dict()
                    if hasattr(self.evaluator, "adaptation_log") else []
                ),
            },
            # Latency diagnostics are a systems-level diagnostic only (provider
            # infrastructure / batching / deployment). They are recorded for
            # reproducibility and reported separately; they never feed CRI,
            # GBI, trajectory fidelity, or fracture metrics.
            "latency_diagnostics": {
                "enabled": True,
                "note": (
                    "Latency is a systems-level diagnostic, NOT a model capability "
                    "score. It is excluded from CRI/GBI/trajectory/fracture metrics."
                ),
            },
            "artifacts": {
                "raw_log": os.path.basename(self.raw_log_filename),
                "results": "benchmark_results.json",
                "metrics": f"metrics_{self.sanitized_model_name}.json",
            },
            "created": _dt.datetime.now().strftime("%Y-%m-%d"),
        }
        try:
            manifest_path = os.path.join(
                os.path.dirname(self.raw_log_filename), "manifest.json"
            )
            with open(manifest_path, "w", encoding="utf-8") as f:
                _json.dump(manifest, f, indent=2)
            self.experiment_id = experiment_id
            self.manifest_path = manifest_path
        except Exception as e:  # pragma: no cover - defensive logging
            logger.warning(f"[MANIFEST] Could not write experiment manifest: {e}")
            self.experiment_id = experiment_id
            self.manifest_path = None

    def _horizon_range(self) -> List[int]:
        """Return the depth-horizon z values swept by the bisection search."""
        # The bisection search explores z from 3 up to the hard ceiling.
        return [3, self.hard_ceiling]

    def _build_task_seed_manifest(self) -> Dict[str, Any]:
        """Record the per-instance task seed selectors for reproducibility."""
        manifest = {}
        for i in range(self.n_probes):
            manifest[f"base_probe_{i}"] = {
                "seed": self.seed,
                "index": i,
            }
        return manifest

    def _evaluate_depth_batch(self, base_probes: List[Dict], gravity_target: float,
                              z: int, tiers: Optional[List[int]] = None,
                              results_matrix: Optional[Dict] = None,
                              skip_set: Optional[set] = None) -> tuple:
        """
        Runs evaluation matrix across OOD tiers.

        For each probe/tier/depth we:
          1. Generate a deterministic GridTask (tier mutates transition semantics
             + ground truth).
          2. Query the model for a raw trajectory output.
          3. Score the output with trajectory-based metrics (no checksum).
          4. Classify the dominant error mode from the real task environment.

        ``results_matrix`` / ``skip_set`` allow a parallel worker (own matrix) and
        resume (skip probes already completed in the raw stream). When ``skip_set``
        contains a probe's identity the probe is NOT re-run; its prior result is
        assumed already present in ``results_matrix`` (loaded by the caller).
        """
        results_matrix = results_matrix if results_matrix is not None else self.results_matrix
        skip_set = skip_set if skip_set is not None else set()
        all_accuracies = []
        context_exhausted_count = 0
        probes_to_test = base_probes

        # Tier 0 (baseline) is intentionally included in the default sweep as the
        # control condition; it shares the same grid/actions as Tier 1 but with no
        # perturbation, so any accuracy drop vs Tier 0 isolates the effect of the
        # difficulty manipulation rather than task length/structure.
        target_tiers = tiers if tiers is not None else [0, 1, 2, 3]

        for probe_idx, base_probe in enumerate(probes_to_test):
            gravity_layer = self._shift_semantic_gravity(z, gravity_target)

            for grid_width, grid_height, grid_key in self._grid_size_iter():
                for tier in target_tiers:
                    # --- Resume: skip probes already completed in the raw stream.
                    # Identity is (seed, gravity, tier, z, probe_idx, grid_key) and
                    # is independent of file ordering / timestamps. ---
                    ident = self._probe_identity(
                        self.seed, gravity_target, tier, z, probe_idx, grid_key
                    )
                    if ident in skip_set:
                        # Count the already-completed probe toward accuracy so the
                        # bisection decision uses the full (resumed) dataset.
                        prior = results_matrix.get(
                            (z, gravity_target, tier, probe_idx) + (
                                (grid_key,) if self.grid_size_levels is not None else ()
                            )
                        )
                        if prior is not None:
                            all_accuracies.append(
                                self._coerce_float(prior.get("step_accuracy", 0.0))
                            )
                        continue

                    task_index = self._task_index_for(
                        base_probe.get("id", f"probe_{probe_idx}"), tier, z
                    )
                    task = self.generator.generate(
                        tier=tier, task_index=task_index, horizon=z,
                        grid_width=grid_width, grid_height=grid_height
                    )
                    probe_base_task = self._build_probe_from_task(task, gravity_target)

                    ood_layer = self._apply_ood_tier(tier, gravity_target)

                    final_probe_payload = {
                        **probe_base_task,
                        **gravity_layer,
                        **ood_layer
                    }

                    # --- UX: live progress + status line ---
                    self._progress_set_context(tier, gravity_target, z, seed=self.seed)
                    self._progress_set_status("Waiting for API...")
                    result = self.evaluator.evaluate_single_probe(
                        probe=final_probe_payload,
                        tier=tier,
                        depth=z,
                    )
                    self._progress_set_status("")  # clear transient status

                    # Integrity check
                    assert result.get("fol_depth") == z, (
                        f"Depth mismatch: expected {z}, got {result.get('fol_depth')}"
                    )

                    raw_output = result.get("raw_output", "")

                    # --- Global stream logger ---
                    expected_sequence = final_probe_payload.get("ground_truth_path", [])
                    try:
                        token_metrics = measure_tokens(
                            final_probe_payload.get("question", ""),
                            raw_output,
                            provider_response=result.get("provider_response"),
                            model_name=self.model_name,
                        )
                        with self.file_write_lock:
                            with open(self.raw_log_filename, "a", encoding="utf-8") as f:
                                f.write(f"\n=== RAW STREAM ENTRY (Tokens: {token_metrics['total_tokens']}) ===\n")
                                f.write(f"Parameters: z={z}, gravity={gravity_target}, tier={tier}, grid={grid_key}\n")
                                # Environment context (needed by the post-run analyzer
                                # to invoke the real taxonomy without re-running the
                                # model). Pure logging -- no benchmark/eval logic change.
                                # The taxonomy must never have to guess missing info,
                                # so we persist the full environment metadata per entry.
                                f.write(f"Grid: {final_probe_payload.get('grid_width')}x{final_probe_payload.get('grid_height')}\n")
                                f.write(f"InitialState: {final_probe_payload.get('initial_state')}\n")
                                f.write(f"Actions: {final_probe_payload.get('actions')}\n")
                                f.write(f"TransitionRules: {final_probe_payload.get('transition_rules')}\n")
                                f.write(f"EnvironmentMetadata: seed={self.seed}, tier={tier}, gravity={gravity_target}, "
                                        f"z={z}, grid_key={grid_key}, "
                                        f"tier_name={final_probe_payload.get('tier_name')}, "
                                        f"experiment_hash={final_probe_payload.get('experiment_hash')}\n")
                                # Latency diagnostics (systems-level only; never a
                                # capability score). Recorded in the raw stream so the
                                # post-run analyzer can reproduce per-probe latency.
                                lat_ms = result.get("latency_ms")
                                if lat_ms is not None:
                                    f.write(f"Latency(ms): {lat_ms:.3f}\n")
                                    f.write(f"InputTokens: {result.get('input_tokens')}\n")
                                    f.write(f"OutputTokens: {result.get('output_tokens')}\n")
                                f.write("-" * 30 + "\n")
                                f.write("[MODEL OUTPUT]:\n")
                                f.write(f"{raw_output if raw_output else '[EMPTY OR NONE PAYLOAD]'}\n\n")
                                f.write("[GROUND TRUTH EXPECTED]:\n")
                                # Render ground truth as newline-separated coordinates
                                # (matching the model output format) so visual
                                # comparison is honest. expected_sequence is a list of
                                # canonical "[x,y]" tokens; joining with newlines keeps
                                # the two blocks visually consistent.
                                if expected_sequence:
                                    f.write("\n".join(expected_sequence) + "\n\n")
                                else:
                                    f.write("[EMPTY GROUND TRUTH]\n\n")
                                f.write("=" * 30 + "\n")
                    except Exception as e:  # pragma: no cover - defensive
                        logger.error(f"Failed to log raw stream: {e}")

                    # --- Strictness gate logic ---
                    if result.get("context_exhausted", False) or result.get("completion_tokens", 0) > 7500:
                        logger.warning(f"   [OUTPUT CEILING / EXHAUSTION] Failure at z={z}. Writing raw log.")
                        context_exhausted_count += 1
                        acc = 0.0

                        try:
                            with open("outputs/debug_raw_output.txt", "a", encoding="utf-8") as f:
                                f.write(
                                    f"\n=== TRUNCATED EXHAUSTION EVENT z={z}, "
                                    f"gravity={gravity_target}, tier={tier}, grid={grid_key} ===\n"
                                )
                                f.write(raw_output)
                        except Exception as e:  # pragma: no cover - defensive
                            logger.error(f"Failed to write dump: {e}")

                        # Record a zeroed trajectory result so aggregation is consistent.
                        result["step_accuracy"] = 0.0
                        result["exact_match"] = False
                        result["continuity_score"] = 0.0
                        result["first_divergence"] = None
                        result["final_state_correct"] = False
                        result["error_mode"] = ErrorMode.HORIZON_COLLAPSE.value
                        result["horizon_compliance"] = 0.0
                        result["generation_bloat_index"] = 0.0
                        result["generation_efficiency"] = 0.0
                        # Latency diagnostics still recorded (systems-level).
                        result.setdefault("correct_transitions", 0)
                    else:
                        # --- Trajectory-based scoring (the single source of truth) ---
                        predicted_path = extract_model_path(raw_output)
                        ground_truth_path = final_probe_payload["ground_truth_path"]

                        traj = compare_trajectories(predicted_path, ground_truth_path)
                        result["step_accuracy"] = traj.step_accuracy
                        result["exact_match"] = traj.exact_match
                        result["continuity_score"] = traj.continuity_score
                        result["first_divergence"] = traj.first_divergence
                        result["final_state_correct"] = traj.final_state_correct
                        result["predicted_length"] = traj.predicted_length
                        result["ground_truth_length"] = traj.ground_truth_length

                        # Correctly predicted state transitions = number of
                        # ground-truth positions reproduced (verified trajectory
                        # throughput). Used by Correct Steps/sec diagnostics.
                        gt_len = len(ground_truth_path)
                        overlap = min(len(predicted_path), gt_len)
                        correct_transitions = 0
                        for i in range(overlap):
                            if predicted_path[i] == ground_truth_path[i]:
                                correct_transitions += 1
                        result["correct_transitions"] = correct_transitions

                        acc = traj.step_accuracy

                        # --- Error taxonomy classification (real environment) ---
                        tax = classify_from_result(
                            predicted_path,
                            ground_truth_path,
                            final_probe_payload["transition_rules"],
                            final_probe_payload["actions"],
                            tier,
                            final_probe_payload["grid_width"],
                            final_probe_payload["grid_height"],
                        )
                        if tax.valid:
                            result["error_mode"] = tax.legacy_mode.value
                        else:
                            result["error_mode"] = "Unknown"

                        # --- Horizon compliance ---
                        if not predicted_path:
                            result["horizon_compliance"] = 0.0
                            result["hc_error"] = "Path parsing failed"
                            generated_depth = 0
                        else:
                            generated_depth = max(0, len(predicted_path) - 1)
                            result["horizon_compliance"] = HorizonCompliance.calculate(
                                z, generated_depth
                            )
                            result["horizon_overshoot"] = generated_depth / z if z > 0 else 0.0

                        # --- Generation bloat index ---
                        actual_tokens = result.get("completion_tokens")
                        if actual_tokens is None:
                            actual_tokens = result.get("output_tokens", 0)

                        baseline_info = self._get_expected_completion_tokens(z)
                        result["generation_bloat_index"] = GenerationBloatIndex.calculate(
                            actual_tokens, baseline_info["tokens"]
                        )
                        result["gbi_metadata"] = {
                            "expected_tokens": baseline_info["tokens"],
                            "baseline_source": baseline_info["source"]
                        }

                        # --- Generation efficiency ---
                        result["generation_efficiency"] = GenerationEfficiency.calculate(
                            result["generation_bloat_index"]
                        )

                    # Ensure latency fields are present for downstream reporting
                    # even if the evaluator did not populate them (e.g. stub).
                    result.setdefault("latency_ms", result.get("latency"))
                    result.setdefault("input_tokens", result.get("prompt_tokens"))
                    result.setdefault("output_tokens", result.get("completion_tokens"))
                    result.setdefault("correct_transitions", 0)

                    # Propagate the final fracture depth for this gravity level onto
                    # the per-probe result so per-tier aggregation can read it.
                    # Without this, result.get("fracture_depth", 0) is always 0 and
                    # every tier reports a fracture depth of 0.0 even when the
                    # overall fracture depth (from self.fracture_cache) is non-zero.
                    result["fracture_depth"] = self.fracture_cache.get(
                        (gravity_target, "final"), 0
                    )

                    # Store result in matrix. When grid_size_levels is enabled the key
                    # gains a grid dimension so each grid size is kept distinct; the
                    # default (None) keeps the original 4-tuple key for identical output.
                    grid_suffix = (grid_key,) if self.grid_size_levels is not None else ()
                    results_matrix[(z, gravity_target, tier, probe_idx) + grid_suffix] = result
                    all_accuracies.append(acc)

                    # --- UX: advance the live probe counter ---
                    self._progress_increment()

        batch_acc = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0
        return batch_acc, context_exhausted_count

    def run_metric_unit_tests(self):
        """Self-tests for the trajectory-based diagnostic metrics."""
        logger.info("[METRIC-TEST] Running diagnostic metric unit tests...")
        # compare_trajectories basic sanity
        gt = ["[0,0]", "[0,1]", "[0,2]"]
        pred_ok = ["[0,0]", "[0,1]", "[0,2]"]
        pred_bad = ["[0,0]", "[1,1]", "[0,2]"]
        r_ok = compare_trajectories(pred_ok, gt)
        r_bad = compare_trajectories(pred_bad, gt)
        assert r_ok.exact_match is True
        assert abs(r_ok.step_accuracy - 1.0) < 1e-9
        assert r_bad.exact_match is False
        assert abs(r_bad.step_accuracy - (2.0 / 3.0)) < 1e-9
        # Latency diagnostics sanity (systems-level only).
        ale = LatencyDiagnostics.accuracy_latency_efficiency(0.8, 10.0)
        assert abs(ale - 0.08) < 1e-9, f"ALE mismatch: {ale}"
        cps = LatencyDiagnostics.correct_steps_per_second(1000, 20.0)
        assert abs(cps - 50.0) < 1e-9, f"Correct Steps/sec mismatch: {cps}"
        logger.info("[METRIC-TEST] Diagnostic metric unit tests passed.")

    def run_self_test(self, base_probes: List[Dict]):
        """Lightweight self-test that the runner can generate + score a probe."""
        logger.info("[SELF-TEST] Running runner self-test...")
        task = self.generator.generate(tier=0, task_index=0, horizon=3)
        probe = self._build_probe_from_task(task, 0.0)
        assert probe.get("ground_truth_path"), "Self-test probe missing ground truth"
        logger.info("[SELF-TEST] Runner self-test passed.")

    # ------------------------------------------------------------------
    # Reporting: structured debug report (no scoring logic change)
    # ------------------------------------------------------------------
    def _print_structured_report(self, aggregated: Dict):
        lines = [
            "",
            "=" * 60,
            "BENCHMARK EVALUATION SUMMARY",
            "=" * 60,
            f"{'Metric':<24} | {'Evidence'}",
            "-" * 60,
            f"{'Step Accuracy':<24} | {aggregated['step_accuracy']:.1f}",
            f"{'Exact Match':<24} | {aggregated['exact_match']:.1f}",
            f"{'Continuity':<24} | {aggregated['continuity']:.1f}",
            f"{'Horizon Compliance':<24} | {aggregated['horizon_compliance']:.1f}",
            f"{'GBI':<24} | {aggregated['generation_bloat_index']:.2f}",
            f"{'Efficiency':<24} | {aggregated['efficiency']:.1f}",
            f"{'Fracture Depth':<24} | {aggregated['fracture_depth']}",
            "",
            "Per-Tier Fracture Depth:",
        ]
        for tier, data in aggregated.get('per_tier', {}).items():
            lines.append(f"- {tier:<30}: fracture_depth={data.get('fracture_depth', 0.0):.1f}, accuracy={data.get('accuracy', 0.0):.1f}")
        lines.append("")
        lines.append("CRI (Secondary Aggregate Robustness Index):")
        cri = aggregated.get('cri', {})
        if isinstance(cri, dict):
            lines.append(f"- CRI: {cri.get('cri', 0.0):.3f}")
            lines.append(f"- Trajectory Fidelity: {cri.get('trajectory_fidelity', 0.0):.3f}")
            lines.append(f"- Horizon Robustness: {cri.get('horizon_robustness', 0.0):.3f}")
            lines.append(f"- Semantic Robustness: {cri.get('semantic_robustness', 0.0):.3f}")
            lines.append(f"- Generation Efficiency: {cri.get('generation_efficiency', 0.0):.3f}")
        lines.append("")
        lines.append("Failure Analysis:")
        for mode, pct in aggregated['failure_analysis'].items():
            lines.append(f"- {mode:<30}: {pct:.1f}%")
        # Latency diagnostics (systems-level only; never a capability score).
        lat = aggregated.get('latency_diagnostics')
        if lat:
            lines.append("")
            lines.append("Latency Diagnostics (systems-level; NOT a capability score):")
            lines.append(f"- Mean Latency:        {lat.get('mean_latency_ms', 0.0):.3f} ms")
            lines.append(f"- Median Latency:      {lat.get('median_latency_ms', 0.0):.3f} ms")
            lines.append(f"- P95 Latency:         {lat.get('p95_latency_ms', 0.0):.3f} ms")
            lines.append(f"- Maximum Latency:     {lat.get('max_latency_ms', 0.0):.3f} ms")
            lines.append(f"- Mean Output Tokens/sec: {lat.get('mean_output_tokens_per_sec', 0.0):.3f}")
            lines.append(f"- Mean Correct Steps/sec: {lat.get('mean_correct_steps_per_sec', 0.0):.3f}")
            lines.append(f"- ALE: {lat.get('ale', 0.0):.4f}")
            lines.append(f"- Correct Steps/sec: {lat.get('correct_steps_per_sec', 0.0):.3f}")
        lines.append("=" * 60)
        self._append_debug_log("\n".join(lines))

    # ------------------------------------------------------------------
    # Clean terminal summary report + JSON export + partial-run reporting
    # (infrastructure only; no benchmark / scoring / evaluation logic changed)
    # ------------------------------------------------------------------
    def _compute_report_metrics(self, aggregated: Dict) -> Dict:
        """Derive the clean-report metrics from the results matrix + aggregated dict.

        Purely an aggregation/formatting layer over existing data. It does NOT
        modify any benchmark, scoring, gravity, tier, or transition-rule logic.
        """
        total_probes = len(self.results_matrix)
        completed = self.actual_probe_count if self.actual_probe_count else total_probes

        # Per-horizon accuracy (keyed by depth z)
        per_horizon: Dict[Any, List[float]] = {}
        # Per-gravity accuracy
        per_gravity: Dict[Any, List[float]] = {}
        # Per-tier accuracy
        per_tier: Dict[Any, List[float]] = {}

        output_tokens_list: List[float] = []
        latency_list: List[float] = []
        total_cost = 0.0
        failure_count = 0

        # Failure taxonomy aggregation (buckets the real ErrorMode categories
        # from arcus.analysis.taxonomy into the 4 reporting placeholders).
        taxonomy_counts = {
            "Wraparound errors": 0,
            "State drift": 0,
            "Formatting failures": 0,
            "Unknown": 0,
        }

        for key, result in self.results_matrix.items():
            z, gravity, tier, _probe_idx, _grid_key = self._unpack_matrix_key(key)
            acc = self._coerce_float(result.get("step_accuracy", 0.0))

            per_horizon.setdefault(z, []).append(acc)
            per_gravity.setdefault(gravity, []).append(acc)
            per_tier.setdefault(tier, []).append(acc)

            ot = result.get("output_tokens")
            if ot is None:
                ot = result.get("completion_tokens", 0)
            if ot:
                output_tokens_list.append(self._coerce_float(ot))

            lat = result.get("latency_ms")
            if lat is None:
                lat = result.get("latency")
            if lat is not None:
                latency_list.append(self._coerce_float(lat))

            cost = result.get("cost")
            if cost is not None:
                total_cost += self._coerce_float(cost)

            mode = result.get("error_mode", "None")
            if result.get("finish_reason") in ("error",) or mode not in ("None", "Unknown"):
                failure_count += 1

            # Bucket the real taxonomy ErrorMode into the 4 reporting categories.
            if mode in ("Transition Rule Failure", "Horizon Collapse"):
                taxonomy_counts["Wraparound errors"] += 1
            elif mode == "State Tracking Failure":
                taxonomy_counts["State drift"] += 1
            elif mode == "Formatting Failure":
                taxonomy_counts["Formatting failures"] += 1
            else:
                taxonomy_counts["Unknown"] += 1

        def _avg(vals):
            return round(sum(vals) / len(vals), 1) if vals else 0.0

        per_horizon_acc = {str(k): _avg(v) for k, v in sorted(per_horizon.items())}
        per_gravity_acc = {str(k): _avg(v) for k, v in sorted(per_gravity.items(), key=lambda kv: float(kv[0]))}
        per_tier_acc = {f"tier_{k}": _avg(v) for k, v in sorted(per_tier.items())}

        avg_output_tokens = round(sum(output_tokens_list) / len(output_tokens_list), 1) if output_tokens_list else 0.0
        max_output_tokens = round(max(output_tokens_list), 1) if output_tokens_list else 0.0
        avg_latency = round(sum(latency_list) / len(latency_list), 3) if latency_list else 0.0

        return {
            "model": self.model_name,
            "run_status": self.run_status or "UNKNOWN",
            "completed_probes": completed,
            "total_probes": total_probes,
            "accuracy": aggregated.get("step_accuracy", 0.0),
            "per_horizon_accuracy": per_horizon_acc,
            "per_gravity_accuracy": per_gravity_acc,
            "per_tier_accuracy": per_tier_acc,
            "avg_output_tokens": avg_output_tokens,
            "max_output_tokens": max_output_tokens,
            "avg_latency": avg_latency,
            "total_cost": round(total_cost, 4) if total_cost else (0.0 if total_cost == 0.0 else None),
            "failure_count": failure_count,
            "failure_taxonomy": taxonomy_counts,
        }

    def _print_summary_report(self, aggregated: Dict):
        """Print a clean, human-readable terminal summary report.

        Includes the Failure Taxonomy placeholder section (reporting
        infrastructure only -- no new taxonomy logic is implemented here).
        """
        m = self._compute_report_metrics(aggregated)

        lines = [
            "",
            "=" * 60,
            "ARCUS-X RUN SUMMARY",
            "=" * 60,
            f"Model:               {m['model']}",
            f"Run Status:          {m['run_status']}",
            f"Completed Probes:    {m['completed_probes']} / {m['total_probes']}",
            f"Master Seeds:        {', '.join(str(s) for s in self.seeds)}",
            f"Accuracy:            {m['accuracy']:.1f}%",
            "",
            "Per-Horizon Accuracy:",
        ]
        for z, acc in m["per_horizon_accuracy"].items():
            lines.append(f"  z={z:<4}: {acc * 100:.1f}%")
        lines.append("")
        lines.append("Per-Gravity Accuracy:")
        for g, acc in m["per_gravity_accuracy"].items():
            lines.append(f"  gravity={g:<4}: {acc * 100:.1f}%")
        lines.append("")
        lines.append("Per-Tier Accuracy:")
        for t, acc in m["per_tier_accuracy"].items():
            lines.append(f"  {t:<8}: {acc * 100:.1f}%")
        lines.append("")
        cost_str = f"{m['total_cost']:.4f}" if m["total_cost"] is not None else "N/A"
        lines.extend([
            f"Total Provider Cost: {cost_str}",
            f"Average Output Tokens: {m['avg_output_tokens']}",
            f"Maximum Output Tokens: {m['max_output_tokens']}",
            f"Average Latency:       {m['avg_latency']}",
            f"Failure Count:         {m['failure_count']}",
            "",
            "Multi-Seed Variance (mean +/- std across master seeds):",
        ])
        sv = self._seed_variance_summary()
        if sv["n_seeds"] > 1:
            lines.append(f"  Seeds:               {self.seeds}")
            lines.append(f"  CRI:                 {sv['cri']['mean']:.3f} +/- {sv['cri']['std']:.3f}")
            lines.append(f"  Fracture Depth:      {sv['fracture_depth']['mean']:.1f} +/- {sv['fracture_depth']['std']:.1f}")
        else:
            lines.append(f"  Seeds:               {self.seeds} (single seed; no variance reported)")
        lines.append("")
        lines.append("Failure Taxonomy:")
        for bucket, count in m.get("failure_taxonomy", {}).items():
            lines.append(f"  {bucket + ':':<22} {count}")
        lines.append("")
        # --- Latency Diagnostics (systems-level only; never a capability score) ---
        lat = aggregated.get("latency_diagnostics")
        if lat:
            lines.append("Latency Diagnostics:")
            lines.append("-" * 60)
            lines.append(f"Mean Latency:        {lat.get('mean_latency_ms', 0.0):.3f} ms")
            lines.append(f"Median Latency:      {lat.get('median_latency_ms', 0.0):.3f} ms")
            lines.append(f"P95 Latency:         {lat.get('p95_latency_ms', 0.0):.3f} ms")
            lines.append(f"Maximum Latency:     {lat.get('max_latency_ms', 0.0):.3f} ms")
            lines.append(f"Mean Output Tokens/sec: {lat.get('mean_output_tokens_per_sec', 0.0):.3f}")
            lines.append(f"Mean Correct Steps/sec: {lat.get('mean_correct_steps_per_sec', 0.0):.3f}")
            lines.append("")
            lines.append("Normalized Efficiency:")
            lines.append(f"  ALE: {lat.get('ale', 0.0):.4f}")
            lines.append(f"  Correct Steps/sec: {lat.get('correct_steps_per_sec', 0.0):.3f}")
            lines.append("")
            lines.append("Latency by Horizon:")
            for ph in lat.get("per_horizon", []):
                lines.append(f"  z={ph['horizon']}:")
                lines.append(f"    Mean Latency: {ph['mean_latency_ms']:.3f} ms")
                lines.append(f"    Correct Steps/sec: {ph['correct_steps_per_sec']:.3f}")
            lines.append("=" * 60)
        # Print to terminal (clean, user-facing) AND keep a debug copy.
        print("\n".join(lines))
        self._append_debug_log("\n".join(lines))

    def _export_metrics_json(self, aggregated: Dict):
        """Export the clean metrics to metrics_{sanitized_model_name}.json."""
        m = self._compute_report_metrics(aggregated)
        out = {
            "model": m["model"],
            "run_status": m["run_status"],
            "completed_probes": m["completed_probes"],
            "total_probes": m["total_probes"],
            "accuracy": m["accuracy"],
            "per_horizon_accuracy": m["per_horizon_accuracy"],
            "per_gravity_accuracy": m["per_gravity_accuracy"],
            "per_tier_accuracy": m["per_tier_accuracy"],
            "avg_output_tokens": m["avg_output_tokens"],
            "max_output_tokens": m["max_output_tokens"],
            "avg_latency": m["avg_latency"],
            "total_cost": m["total_cost"],
            "failure_taxonomy": m.get("failure_taxonomy", {}),
            # Latency diagnostics (systems-level only; never a capability score).
            "latency_diagnostics": aggregated.get("latency_diagnostics", {}),
        }
        filename = f"outputs/metrics_{self.sanitized_model_name}.json"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            logger.info(f"Exported metrics to {filename}")
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Failed to export metrics JSON: {e}")

    def _print_partial_run_report(self):
        """Print the partial-run banner when the benchmark did not finish."""
        completed = self.actual_probe_count
        total = self.expected_probe_count if self.expected_probe_count else len(self.results_matrix)
        if total is None:
            total = completed
        remaining_start = completed  # 0 -> N already done; next is N
        lines = [
            "=" * 35,
            "RUN NOT FINISHED",
            f"Completed:",
            f"0 -> {completed}",
            f"Remaining:",
            f"{remaining_start + 1} -> {total}",
            "Statistics generated from completed probes only.",
            "=" * 35,
        ]
        print("\n".join(lines))
        self._append_debug_log("\n".join(lines))

    def _print_horizon_summary(self):
        summary = {}
        for key, result in self.results_matrix.items():
            z, gravity, tier, probe_idx, grid_key = self._unpack_matrix_key(key)
            if z not in summary:
                summary[z] = {"hc": [], "gbi": [], "ge": []}
            for metric_name, target_key in [
                ("horizon_compliance", "hc"),
                ("generation_bloat_index", "gbi"),
                ("generation_efficiency", "ge"),
            ]:
                value = result.get(metric_name)
                if value is not None:
                    summary[z][target_key].append(self._coerce_float(value))

        lines = ["[METRIC REPORT] Fractional Score by Horizon Depth:"]
        for z in sorted(summary.keys()):
            hc_avg = sum(summary[z]["hc"]) / len(summary[z]["hc"]) if summary[z]["hc"] else 0.0
            gbi_avg = sum(summary[z]["gbi"]) / len(summary[z]["gbi"]) if summary[z]["gbi"] else 0.0
            ge_avg = sum(summary[z]["ge"]) / len(summary[z]["ge"]) if summary[z]["ge"] else 0.0
            lines.append(f"  Horizon z={z}: HC={hc_avg:.3f}, GBI={gbi_avg:.3f}, GE={ge_avg:.3f}")
        self._append_debug_log("\n".join(lines))

    def _print_gravity_summary(self):
        summary = {}
        for key, result in self.results_matrix.items():
            z, gravity, tier, probe_idx, grid_key = self._unpack_matrix_key(key)
            if gravity not in summary:
                summary[gravity] = {"hc": [], "gbi": [], "ge": [], "fracture_depths": []}
            for metric_name, target_key in [
                ("horizon_compliance", "hc"),
                ("generation_bloat_index", "gbi"),
                ("generation_efficiency", "ge"),
            ]:
                value = result.get(metric_name)
                if value is not None:
                    summary[gravity][target_key].append(self._coerce_float(value))
            fd = result.get("fracture_depth", 0)
            if fd and fd > 0:
                summary[gravity]["fracture_depths"].append(fd)

        lines = ["[METRIC REPORT] Fractional Score by Gravity Values:"]
        for gravity in sorted(summary.keys()):
            hc_avg = sum(summary[gravity]["hc"]) / len(summary[gravity]["hc"]) if summary[gravity]["hc"] else 0.0
            gbi_avg = sum(summary[gravity]["gbi"]) / len(summary[gravity]["gbi"]) if summary[gravity]["gbi"] else 0.0
            ge_avg = sum(summary[gravity]["ge"]) / len(summary[gravity]["ge"]) if summary[gravity]["ge"] else 0.0
            fd_avg = sum(summary[gravity]["fracture_depths"]) / len(summary[gravity]["fracture_depths"]) if summary[gravity]["fracture_depths"] else 0.0
            lines.append(f"  Gravity {gravity}: HC={hc_avg:.3f}, GBI={gbi_avg:.3f}, GE={ge_avg:.3f}, FractureDepth={fd_avg:.1f}")
        self._append_debug_log("\n".join(lines))

    def _compute_aggregated_metrics(self) -> Dict:
        """Compute aggregated trajectory metrics across the results matrix."""
        if not self.results_matrix:
            return {
                "step_accuracy": 0.0,
                "exact_match": 0.0,
                "continuity": 0.0,
                "horizon_compliance": 0.0,
                "generation_bloat_index": 0.0,
                "efficiency": 0.0,
                "fracture_depth": 0,
                "mean_overshoot": 0.0,
            }

        traj_results = []

        protocol_failures = {
            "valid_response_format": 0,
            "parser_success": 0,
            "output_truncated": 0,
            "provider_failure": 0,
            "malformed_response": 0,
        }
        total_runs = len(self.results_matrix)

        for result in self.results_matrix.values():
            step_acc = self._coerce_float(result.get("step_accuracy", 0.0))
            exact = 1.0 if result.get("exact_match", False) else 0.0
            continuity = self._coerce_float(result.get("continuity_score", 0.0))
            traj_results.append({
                "step_accuracy": step_acc,
                "exact_match": bool(exact),
                "continuity_score": continuity,
                "predicted_length": result.get("predicted_length", 0),
                "ground_truth_length": result.get("ground_truth_length", 0),
            })

            # Protocol Metrics
            if "checksum" in result.get("raw_output", "").lower():
                protocol_failures["valid_response_format"] += 1
            if step_acc > 0:
                protocol_failures["parser_success"] += 1
            if result.get("finish_reason") == "length":
                protocol_failures["output_truncated"] += 1
            if result.get("finish_reason") == "error":
                protocol_failures["provider_failure"] += 1
            if not result.get("raw_output"):
                protocol_failures["malformed_response"] += 1

        agg = aggregate_trajectory_results(traj_results)

        # New Diagnostic Averages
        all_hc = [
            self._coerce_float(res.get("horizon_compliance", 0.0))
            for res in self.results_matrix.values() if "horizon_compliance" in res
        ]
        all_gbi = [
            self._coerce_float(res.get("generation_bloat_index", 0.0))
            for res in self.results_matrix.values() if "generation_bloat_index" in res
        ]
        all_ge = [
            self._coerce_float(res.get("generation_efficiency", 0.0))
            for res in self.results_matrix.values() if "generation_efficiency" in res
        ]

        protocol_accuracy = protocol_failures["parser_success"] / total_runs

        final_fractures = [
            depth for (grav, label), depth in self.fracture_cache.items()
            if label == "final" and depth > 0
        ]
        fracture_depth = min(final_fractures) if final_fractures else 0

        # Failure Analysis (from the real error taxonomy)
        failure_analysis = {
            "State Tracking Failure": 0,
            "Transition Rule Failure": 0,
            "Semantic Interpretation Failure": 0,
            "Horizon Collapse": 0,
            "Formatting Failure": 0,
            "None": 0,
        }
        for res in self.results_matrix.values():
            mode = res.get("error_mode", "Unknown")
            if mode in failure_analysis:
                failure_analysis[mode] += 1
            else:
                failure_analysis["None"] += 1

        for k in failure_analysis:
            failure_analysis[k] = (failure_analysis[k] / total_runs) * 100

        # Add per-tier aggregation
        per_tier = {}
        for key, result in self.results_matrix.items():
            z, gravity, tier, probe_idx, grid_key = self._unpack_matrix_key(key)
            if tier not in per_tier:
                per_tier[tier] = {"accuracies": [], "token_density_values": [], "depth_acc": {}}

            per_tier[tier]["accuracies"].append(
                self._coerce_float(result.get("step_accuracy", 0.0))
            )

            # Raw token-density ratio (tokens per depth step), distinct from
            # ``GenerationBloatIndex`` (which is (actual - expected) / expected).
            # Reported per tier as "token_density".
            total_tokens = result.get("total_tokens", 1)
            depth = result.get("fol_depth", 1)
            token_density = float(total_tokens) / float(depth) if depth > 0 else 0.0
            per_tier[tier]["token_density_values"].append(token_density)

            # Collect per-depth accuracy so we can compute the canonical
            # fracture depth for this tier from its own accuracy curve.
            try:
                zi = int(float(z))
            except (TypeError, ValueError):
                zi = 0
            per_tier[tier]["depth_acc"].setdefault(zi, []).append(
                self._coerce_float(result.get("step_accuracy", 0.0))
            )

        fracture_finder = FracturePointFinder(look_ahead_step=2)
        per_tier_summary = {}
        for tier, data in per_tier.items():
            # Build the per-tier accuracy curve (z, mean step_accuracy) and
            # locate the canonical fracture depth for that tier. A 2-step
            # look-ahead is used so a single anomalous high-accuracy outlier at
            # a larger horizon does not mask the real collapse.
            curve = [
                (z, sum(accs) / len(accs)) for z, accs in sorted(data["depth_acc"].items())
            ]
            tier_fracture = fracture_finder.compute_structural_yield(curve)
            per_tier_summary[f"tier_{tier}"] = {
                "accuracy": sum(data["accuracies"]) / len(data["accuracies"]) if data["accuracies"] else 0.0,
                "token_density": sum(data["token_density_values"]) / len(data["token_density_values"]) if data["token_density_values"] else 0.0,
                "fracture_depth": float(tier_fracture) if tier_fracture is not None else 0.0
            }

        # Per-gravity fracture curve data
        per_gravity_fracture = {}
        for key, result in self.results_matrix.items():
            z, gravity, tier, probe_idx, grid_key = self._unpack_matrix_key(key)
            if gravity not in per_gravity_fracture:
                per_gravity_fracture[gravity] = {"depths": [], "accuracies": []}
            per_gravity_fracture[gravity]["depths"].append(z)
            per_gravity_fracture[gravity]["accuracies"].append(
                self._coerce_float(result.get("step_accuracy", 0.0))
            )

        per_gravity_summary = {}
        for gravity, data in per_gravity_fracture.items():
            # Sort by depth for curve plotting
            sorted_pairs = sorted(zip(data["depths"], data["accuracies"]))
            per_gravity_summary[gravity] = {
                "depths": [d for d, _ in sorted_pairs],
                "accuracies": [a for _, a in sorted_pairs]
            }

        # --- Latency diagnostics (systems-level only; never a capability score) ---
        # Build the per-probe list consumed by LatencyDiagnostics.aggregate.
        latency_probes = []
        for key, result in self.results_matrix.items():
            z, gravity, tier, probe_idx, grid_key = self._unpack_matrix_key(key)
            lat = result.get("latency_ms")
            if lat is None:
                lat = result.get("latency")
            if lat is None:
                continue
            latency_probes.append({
                "latency_ms": float(lat),
                "output_tokens": result.get("output_tokens", result.get("completion_tokens", 0)),
                "accuracy": self._coerce_float(result.get("step_accuracy", 0.0)),
                "correct_transitions": int(result.get("correct_transitions", 0) or 0),
                "horizon": z,
            })
        latency_diagnostics = LatencyDiagnostics.aggregate(latency_probes)

        return {
            "step_accuracy": round(agg["mean_step_accuracy"] * 100.0, 1),
            "exact_match": round(agg["exact_match_rate"] * 100.0, 1),
            "continuity": round(agg["mean_continuity"] * 100.0, 1),
            "horizon_compliance": round((sum(all_hc) / len(all_hc) if all_hc else 0.0) * 100.0, 1),
            "generation_bloat_index": round(sum(all_gbi) / len(all_gbi) if all_gbi else 0.0, 2),
            "efficiency": round((sum(all_ge) / len(all_ge) if all_ge else 0.0) * 100.0, 1),
            "protocol_accuracy": round(protocol_accuracy * 100.0, 1),
            "failure_analysis": failure_analysis,
            "fracture_depth": fracture_depth,
            "per_tier": per_tier_summary,
            "per_gravity_fracture_curve": per_gravity_summary,
            # Secondary aggregate robustness indicator (NOT a primary metric).
            "cri": self.compute_cri(),
            # Multi-seed variance: mean +/- std across master seeds so the
            # headline metrics are not an artifact of one lucky/unlucky seed.
            "seed_variance": self._seed_variance_summary(),
            # The actual master seed list used for this run, so every report is
            # self-documenting (no need to cross-reference hardcoded constants).
            "master_seeds": list(self.seeds),
            # Full per-seed metrics keyed by master seed, alongside the
            # aggregate, for direct downstream statistical analysis.
            "per_seed": {
                str(seed): metrics
                for seed, metrics in getattr(self, "_per_seed_full", {}).items()
            },
            # Latency-normalized efficiency diagnostics (systems-level only).
            "latency_diagnostics": latency_diagnostics,
        }

    def _seed_variance_summary(self) -> Dict[str, object]:
        """Compute mean +/- std of headline metrics across the master seeds.

        Returns ``{"n_seeds": int, "cri": {mean, std}, "fracture_depth": {mean, std}}``.
        When only a single seed was used, std is 0.0 and the block is still
        emitted (so downstream consumers always have a stable schema).
        """
        import statistics
        metrics = getattr(self, "_per_seed_metrics", None)
        if not metrics:
            return {
                "n_seeds": len(self.seeds),
                "cri": {"mean": 0.0, "std": 0.0},
                "fracture_depth": {"mean": 0.0, "std": 0.0},
            }
        cri_vals = [m["cri"] for m in metrics]
        fd_vals = [m["fracture_depth"] for m in metrics]
        n = len(cri_vals)
        cri_mean = sum(cri_vals) / n
        fd_mean = sum(fd_vals) / n
        cri_std = statistics.pstdev(cri_vals) if n > 1 else 0.0
        fd_std = statistics.pstdev(fd_vals) if n > 1 else 0.0
        return {
            "n_seeds": n,
            "cri": {"mean": round(cri_mean, 4), "std": round(cri_std, 4)},
            "fracture_depth": {"mean": round(fd_mean, 2), "std": round(fd_std, 2)},
        }

    def compute_cri(self) -> Dict[str, object]:
        """Compute the secondary ARCUS Robustness Index (CRI) from the results matrix.

        CRI is a *secondary* summary statistic. The primary trajectory metrics
        (step accuracy, exact match, continuity, fracture depth) remain the
        headline evaluation signals and are reported separately.
        """
        from arcus.evaluation.metrics import ARCUSRobustnessIndex, mean

        if not self.results_matrix:
            return {
                "cri": 0.0,
                "trajectory_fidelity": 0.0,
                "horizon_robustness": 0.0,
                "semantic_robustness": 0.0,
                "generation_efficiency": 0.0,
                "components": dict(ARCUSRobustnessIndex.WEIGHTS),
            }

        step_accs: List[float] = []
        cont_accs: List[float] = []
        exact_accs: List[float] = []
        ge_accs: List[float] = []
        horizon_points: List[Tuple[float, Optional[float]]] = []
        tier_accs = {0: [], 1: [], 2: [], 3: []}

        for key, result in self.results_matrix.items():
            z, gravity, tier, probe_idx, grid_key = self._unpack_matrix_key(key)
            sa = self._coerce_float(result.get("step_accuracy", 0.0))
            cs = self._coerce_float(result.get("continuity_score", 0.0))
            em = 1.0 if result.get("exact_match", False) else 0.0
            step_accs.append(sa)
            cont_accs.append(cs)
            exact_accs.append(em)
            # Trajectory accuracy sampled at horizon z (discrete, irregular-safe).
            horizon_points.append((float(z), sa))
            if tier in tier_accs:
                tier_accs[tier].append(sa)
            ge = result.get("generation_efficiency", None)
            if ge is not None:
                ge_accs.append(self._coerce_float(ge))

        summary = ARCUSRobustnessIndex.compute(
            step_accuracy=mean(step_accs),
            continuity_score=mean(cont_accs),
            exact_match=mean(exact_accs),
            horizon_points=horizon_points,
            tier0_acc=mean(tier_accs[0]) if tier_accs[0] else None,
            tier1_acc=mean(tier_accs[1]) if tier_accs[1] else None,
            tier2_acc=mean(tier_accs[2]) if tier_accs[2] else None,
            tier3_acc=mean(tier_accs[3]) if tier_accs[3] else None,
            generation_efficiency=mean(ge_accs) if ge_accs else None,
        )
        return summary.as_dict()

    def _append_debug_log(self, message: str):
        """Write benchmark debug output to the outputs folder instead of flooding the console."""
        os.makedirs("outputs", exist_ok=True)
        debug_path = os.path.join("outputs", "benchmark_debug.log")
        with open(debug_path, "a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now().isoformat()} - {message}\n")

    def _coerce_float(self, value: Any) -> float:
        """Safely coerce values to float for aggregation."""
        if value is None:
            return 0.0
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _stringify_keys(self, data):
        """Recursively convert tuple keys to strings for JSON serialization."""
        if isinstance(data, dict):
            new_dict = {}
            for key, value in data.items():
                if isinstance(key, tuple):
                    new_key = "_".join(str(k) for k in key)
                else:
                    new_key = str(key) if not isinstance(key, str) else key
                new_dict[new_key] = self._stringify_keys(value)
            return new_dict
        elif isinstance(data, list):
            return [self._stringify_keys(item) for item in data]
        return data

    def _generate_report(self, aggregated: Dict) -> Dict:
        """Generate final benchmark report with JSON-serializable structure."""
        serializable_results = self._stringify_keys(self.results_matrix)
        serializable_token_density = self._stringify_keys(self.token_density_matrix)
        serializable_fracture = self._stringify_keys(self.fracture_cache)

        per_tier = aggregated.pop("per_tier", {})
        serializable_per_tier = self._stringify_keys(per_tier)

        per_gravity_fracture_curve = aggregated.pop("per_gravity_fracture_curve", {})
        serializable_per_gravity = self._stringify_keys(per_gravity_fracture_curve)

        serializable_aggregated = self._stringify_keys(aggregated)

        return {
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "n_probes": self.n_probes,
                "gravity_levels": self.gravity_levels,
                "seed": self.seed,
                "seeds": list(self.seeds),
                "grid_size_levels": self.grid_size_levels
            },
            "results_matrix": serializable_results,
            "token_density_matrix": serializable_token_density,
            "fracture_cache": serializable_fracture,
            "metrics": serializable_aggregated,
            "per_tier": serializable_per_tier,
            "per_gravity_fracture_curve": serializable_per_gravity,
            "token_accounting_summary": self._build_token_accounting_summary(serializable_results)
        }

    def _build_token_accounting_summary(self, serializable_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize token accounting status per result for report audits."""
        summary = {"total_results": len(serializable_results), "by_source": {}, "by_confidence": {}}
        for result in serializable_results.values():
            if not isinstance(result, dict):
                continue
            source = result.get("token_source", "unknown")
            confidence = result.get("token_confidence", "unknown")
            metadata = result.get("token_metadata") or {}
            summary["by_source"][source] = summary["by_source"].get(source, 0) + 1
            summary["by_confidence"][confidence] = summary["by_confidence"].get(confidence, 0) + 1
            if isinstance(metadata, dict) and metadata.get("accounting_trace"):
                if "sample_trace" not in summary:
                    summary["sample_trace"] = metadata["accounting_trace"][:3]
        return summary

    def save_results(self, filepath: str):
        """Save results to JSON file with proper serialization constraints inside the target folder."""
        self._progress_set_status("Saving results...")
        filename = os.path.basename(filepath)
        target_dir = "outputs"
        os.makedirs(target_dir, exist_ok=True)
        final_destination = os.path.join(target_dir, filename)

        serializable_results = self._stringify_keys(self.results_matrix)

        results_copy = {}
        for key, value in serializable_results.items():
            if isinstance(value, dict):
                sanitized_value = {
                    k: v for k, v in value.items() \
                    if not callable(v) and k != "prompt"
                }
                if "token_metadata" in sanitized_value and isinstance(sanitized_value["token_metadata"], dict):
                    sanitized_value["token_metadata"] = {
                        "token_source": sanitized_value.get("token_source"),
                        "token_confidence": sanitized_value.get("token_confidence"),
                        **sanitized_value["token_metadata"],
                    }
                results_copy[key] = sanitized_value

        metrics = self._compute_aggregated_metrics()
        serializable_metrics = self._stringify_keys(metrics)
        serializable_token_density = self._stringify_keys(self.token_density_matrix)
        serializable_fracture = self._stringify_keys(self.fracture_cache)

        # Extract per-tier and per-gravity data for interpretable reporting
        per_tier_data = metrics.get("per_tier", {})
        per_gravity_data = metrics.get("per_gravity_fracture_curve", {})
        serializable_per_tier = self._stringify_keys(per_tier_data)
        serializable_per_gravity = self._stringify_keys(per_gravity_data)

        with self.file_write_lock:
            with open(final_destination, "w") as f:
                json.dump({
                    "model": self.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "results_count": len(self.results_matrix),
                    "metrics": serializable_metrics,
                    "per_tier": serializable_per_tier,
                    "per_gravity_fracture_curve": serializable_per_gravity,
                    "raw_token_density_index": serializable_token_density,
                    "structural_fractures": serializable_fracture,
                    "execution_matrix": results_copy,
                    "token_accounting_summary": self._build_token_accounting_summary(serializable_results)
                }, f, indent=2)

        logger.info(f"Results saved to {final_destination}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemini-2.5-flash-lite")
    parser.add_argument("--probes", type=int, default=5)
    parser.add_argument("--output", default="outputs/benchmark_results.json")

    args = parser.parse_args()
    logger.info(f"Ready to run bisection induction benchmark for {args.model}")
