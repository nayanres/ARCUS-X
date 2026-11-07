#!/usr/bin/env python3
"""
Quick Start: Run your first API-only benchmark on Windows
Updated to run within the arcus package structure.

Usage:
    # Default (uses env vars or hardcoded defaults)
    python -m arcus.experiments.quickstart

    # Custom API endpoint, model, and key via command line
    python -m arcus.experiments.quickstart ^
        --api-key sk-or-v1-your-key ^
        --model deepseek/deepseek-r1 ^
        --api-base https://openrouter.ai/api/v1

    # Run the model-free baselines only (no API key required)
    python -m arcus.experiments.quickstart --baselines

    # Run the model benchmark AND include baselines in the report
    python -m arcus.experiments.quickstart --api-key ... --baselines

    # Multi-seed variance measurement (explicit seeds)
    python -m arcus.experiments.quickstart ^
        --api-key ... --model ... ^
        --master-seeds 42 1337 2026 9001 123456

    # Canonical paper seed list (expands "all" to the published seeds)
    python -m arcus.experiments.quickstart ^
        --api-key ... --model ... ^
        --master-seeds all
"""
# debugging was not easy :(
import os
import sys
import logging
import argparse
import re
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional

# Canonical paper seed list. ``--master-seeds all`` expands to this set so that
# published results are reproducible across runs and across models. These are
# the master seeds used for the variance (mean +/- std) reporting in the paper.
PAPER_SEEDS: List[int] = [42, 12345, 6734, 9878, 2026, 9001, 1337, 31415]
MINITEST_SEED = 3157

# Corrected imports for the new arcus package structure
from arcus.experiments.benchmark_runner import BenchmarkRunner
from arcus.evaluation.evaluator import ModelEvaluator
from arcus.analysis.fracture import FracturePointFinder
from arcus.analysis.taxonomy import classify_trajectory
from arcus.evaluation.metrics import HorizonCompliance, compare_trajectories
from arcus.evaluation.parser import extract_model_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _normalize_windows_continuations(argv: List[str]) -> List[str]:
    """Split an option accidentally joined to the preceding seed by ``^``."""
    normalized: List[str] = []
    for arg in argv:
        match = re.match(r"^(-?\d+)(--(?:resume|offline|parallel-seeds|tiers|max-horizon)(?:=.*)?)$", arg)
        if match:
            normalized.extend(match.groups())
        else:
            normalized.append(arg)
    return normalized


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run AI evaluation benchmark with custom API configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--api-key', type=str, default=None, help='API key')
    parser.add_argument('--model', type=str, default=None, help='Model name')
    parser.add_argument('--api-base', type=str, default=None, help='API base URL')
    parser.add_argument('--n-probes', type=int, default=5, help='Number of probes')
    parser.add_argument('--depths', type=str, default='1,2,3', help='Depths to test')
    parser.add_argument('--gravity-levels', type=str, default='0.0,1.0,2.0,3.0', help='Gravity levels')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--baselines', action='store_true', default=False,
                        help='Run model-free baseline evaluators and include them in the report')
    parser.add_argument('--master-seeds', type=str, nargs='*', default=None,
                        help=('Master seeds for variance measurement. Space-separated '
                              'integers (e.g. "42 1337 2026"), or "all" to use the '
                              'canonical paper seed list.'))
    parser.add_argument('--parallel-seeds', type=str, default="1",
                        help=('Number of master seeds to execute in parallel. '
                              'An integer (e.g. "4") or "auto" (min(cpu_count, '
                              'num_seeds)). Default: 1 (serial).'))
    parser.add_argument('--resume', type=str, default=None,
                        help=('Path to an existing absolute raw stream file '
                              '(e.g. absolute_raw_stream_xxx.txt). Completed probes '
                              'are detected by identity and skipped; only missing '
                              'probes are run. Appends to the same file.'))
    parser.add_argument('--offline', action='store_true', default=False,
                        help='Reuse exact observations from --resume and acquire only missing probes')
    parser.add_argument('--dev', action='store_true', default=False,
                        help='Developer mode: seeds=1, gravity=[0.5], tier=1, max horizon=3, no adaptive expansion, 1 probe')
    parser.add_argument('--tiers', type=str, default=None, help='Target tiers (comma separated)')
    parser.add_argument('--max-horizon', type=int, default=None, help='Maximum depth horizon z')
    parser.add_argument('--no-adaptive-expansion', action='store_true', default=False, help='Disable adaptive expansion')
    parser.add_argument(
        '--minitest',
        action='store_true',
        default=False,
        help=(
            'Run one illustrative z=50 probe for each tier 0-3 '
            f'(seed {MINITEST_SEED}, gravity 0.5); saves prompts, outputs, and chain analysis '
            'to outputs/minitests_<model>.txt. Not a statistical evaluation.'
        ),
    )

    return parser.parse_args(_normalize_windows_continuations(sys.argv[1:]))


def resolve_master_seeds(raw) -> Optional[List[int]]:
    """Resolve the --master-seeds argument into a concrete list of ints.

    Accepts:
      * None            -> None (caller uses BenchmarkRunner default)
      * ["all"]         -> the canonical PAPER_SEEDS list
      * ["42","1337"]   -> [42, 1337]
    """
    if raw is None or len(raw) == 0:
        return None
    # Normalise: a single comma/space separated token is also accepted.
    tokens: List[str] = []
    for tok in raw:
        tokens.extend(t.strip() for t in str(tok).replace(',', ' ').split() if t.strip())
    if not tokens:
        return None
    if set(t.lower() for t in tokens) == {'all'}:
        return list(PAPER_SEEDS)
    try:
        return [int(t) for t in tokens]
    except ValueError as e:
        raise ValueError(f"--master-seeds must be integers or 'all': {e}")


def _run_baselines(n_probes: int, gravity_levels: List[float]) -> Dict:
    """Run the optional baselines and return their aggregated report."""
    from arcus.evaluation.baselines import available_baselines
    # Baselines are model-free; the evaluator is never invoked, but
    # BenchmarkRunner.__init__ still constructs one. Use API-only mode with a
    # placeholder key so no local vLLM/GPU dependency is required.
    runner = BenchmarkRunner(
        model_name="baseline-mode",
        evaluator_class=ModelEvaluator,
        api_base="https://openrouter.ai/api/v1",
        api_key="baseline-mode-no-model",
        n_probes=n_probes,
        gravity_levels=list(gravity_levels),
    )
    return runner.run_baselines(
        n_tasks=max(1, n_probes),
        horizon=8,
        gravity_levels=list(gravity_levels),
    )


def _append_minitest_analysis(
    output_path: Path,
    model_name: str,
    records: List[Dict],
) -> None:
    """Append exact prompts, responses, and deterministic chain comparisons."""
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + "=" * 78 + "\n")
        handle.write(
            f"MINITEST CHAIN ANALYSIS | model={model_name} | "
            f"created={datetime.now().isoformat(timespec='seconds')}\n"
        )
        handle.write(
            "Illustrative single-probe examples only; not a statistical evaluation.\n"
        )
        handle.write("=" * 78 + "\n")

        for record in records:
            prompt = record.get("prompt")
            raw_output = record.get("raw_output")
            if not isinstance(prompt, str):
                raise ValueError("Minitest result is missing its exact prompt.")
            if raw_output is not None and not isinstance(raw_output, str):
                raise ValueError("Minitest raw output must be text or None.")

            raw_output_available = isinstance(raw_output, str)
            raw_output_text = raw_output if raw_output_available else ""
            expected = record["ground_truth_path"]
            predicted = extract_model_path(raw_output_text)
            comparison = compare_trajectories(predicted, expected)
            taxonomy = classify_trajectory(
                predicted,
                expected,
                transition_rules=record["transition_rules"],
                actions=record["actions"],
                tier=record["tier"],
                grid_width=record["grid_width"],
                grid_height=record["grid_height"],
                raw_output=raw_output_text,
            )
            generated_depth = max(0, len(predicted) - 1)
            horizon_compliance = HorizonCompliance.calculate(
                record["z"], generated_depth
            )
            valid = bool(taxonomy.probe_classification and taxonomy.probe_classification.valid)

            handle.write("\n" + "-" * 78 + "\n")
            handle.write(
                f"tier={record['tier']} | seed={record['seed']} | "
                f"z={record['z']} | gravity={record['gravity']}\n"
            )
            handle.write(f"task_id={record['task_id']}\n")
            handle.write(f"experiment_hash={record['experiment_hash']}\n")
            handle.write("\n[EXACT SENT PROMPT]\n")
            handle.write(prompt)
            if not prompt.endswith("\n"):
                handle.write("\n")
            handle.write("\n[EXACT MODEL OUTPUT]\n")
            handle.write(
                raw_output_text if raw_output_available
                else "[NO RAW OUTPUT RETURNED]"
            )
            if not raw_output_text.endswith("\n"):
                handle.write("\n")
            handle.write("\n[GROUND-TRUTH CHAIN]\n")
            handle.write("\n".join(expected) + "\n")
            handle.write("\n[PARSED MODEL CHAIN]\n")
            handle.write("\n".join(predicted) + "\n" if predicted else "[NO COORDINATES PARSED]\n")
            handle.write("\n[CHAIN COMPARISON]\n")
            for index in range(max(len(expected), len(predicted))):
                expected_coord = expected[index] if index < len(expected) else "<missing>"
                predicted_coord = predicted[index] if index < len(predicted) else "<missing>"
                position = "initial" if index == 0 else f"transition {index}"
                status = "MATCH" if expected_coord == predicted_coord else "MISMATCH"
                handle.write(
                    f"{index:03d} {position}: expected={expected_coord} "
                    f"actual={predicted_coord} [{status}]\n"
                )
            handle.write("\n[ANALYSIS]\n")
            handle.write(f"raw_output_available={raw_output_available}\n")
            handle.write(f"parseable_coordinates={bool(predicted)}\n")
            handle.write(f"response_valid={valid}\n")
            handle.write(f"exact_match={comparison.exact_match}\n")
            handle.write(f"step_accuracy={comparison.step_accuracy:.6f}\n")
            handle.write(f"continuity={comparison.continuity_score:.6f}\n")
            handle.write(f"horizon_compliance={horizon_compliance:.6f}\n")
            handle.write(f"taxonomy={taxonomy.legacy_mode.value}\n")
            handle.write(f"taxonomy_mechanism={taxonomy.mechanism.value}\n")
            handle.write(f"taxonomy_explanation={taxonomy.explanation}\n")
        handle.write("\n" + "=" * 78 + "\n")


def _run_minitest(model_name: str, api_base: Optional[str], api_key: Optional[str]) -> int:
    """Run one fixed-horizon example per ARCUS-X tier and preserve its evidence."""
    runner = BenchmarkRunner(
        model_name=model_name,
        evaluator_class=ModelEvaluator,
        fracture_finder_class=FracturePointFinder,
        api_base=api_base,
        api_key=api_key,
        n_probes=1,
        gravity_levels=[0.5],
        seed=MINITEST_SEED,
        tiers=[0, 1, 2, 3],
        max_horizon=50,
        adaptive_expansion=False,
    )
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._-") or "model"
    output_path = Path("outputs") / f"minitests_{model_slug}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\nMINITEST RUN | model={model_name} | seed={MINITEST_SEED} | "
            f"gravity=0.5 | z=50 | tiers=0,1,2,3\n"
        )
    runner.raw_log_filename = str(output_path)

    base_probes = runner._generate_base_probes()
    results: Dict = {}
    runner._evaluate_depth_batch(
        base_probes,
        gravity_target=0.5,
        z=50,
        tiers=[0, 1, 2, 3],
        results_matrix=results,
        skip_set=set(),
        log_prompt=True,
    )

    records = []
    for tier in range(4):
        task_index = runner._task_index_for("base_probe_0", tier, 50)
        task = runner.generator.generate(tier=tier, task_index=task_index, horizon=50)
        result = results.get((50, 0.5, tier, 0))
        if result is None:
            raise RuntimeError(f"Minitest did not produce a result for tier {tier}.")
        records.append({
            "tier": tier,
            "seed": MINITEST_SEED,
            "z": 50,
            "gravity": 0.5,
            "task_id": task.task_id,
            "experiment_hash": task.experiment_hash,
            "prompt": result.get("prompt"),
            "raw_output": result.get("raw_output"),
            "ground_truth_path": list(task.ground_truth_trajectory),
            "transition_rules": task.transition_rules,
            "actions": task.actions,
            "grid_width": task.grid_width,
            "grid_height": task.grid_height,
        })

    _append_minitest_analysis(output_path, model_name, records)
    logger.info("Minitest examples saved to %s", output_path)
    return 0


def main():
    args = parse_args()

    if args.minitest:
        incompatible = []
        if args.baselines:
            incompatible.append("--baselines")
        if args.dev:
            incompatible.append("--dev")
        if args.resume or args.offline:
            incompatible.append("--resume/--offline")
        if args.output:
            incompatible.append("--output (minitest uses its own text file)")
        if args.parallel_seeds != "1":
            incompatible.append("--parallel-seeds")
        if args.tiers:
            try:
                configured_tiers = {int(value.strip()) for value in args.tiers.split(",")}
            except ValueError:
                logger.error("--tiers must contain comma-separated integers.")
                return 2
            if configured_tiers != {0, 1, 2, 3}:
                incompatible.append("--tiers (minitest always runs tiers 0,1,2,3)")
        if args.max_horizon not in (None, 50):
            incompatible.append("--max-horizon (minitest uses z=50)")
        if args.gravity_levels not in ("0.0,1.0,2.0,3.0", "0.5"):
            incompatible.append("--gravity-levels (minitest uses gravity 0.5)")
        try:
            configured_seeds = resolve_master_seeds(args.master_seeds)
        except ValueError as exc:
            logger.error("%s", exc)
            return 2
        if configured_seeds is not None and configured_seeds != [MINITEST_SEED]:
            incompatible.append(f"--master-seeds (minitest uses only seed {MINITEST_SEED})")
        if incompatible:
            logger.error("--minitest cannot be combined with: %s", ", ".join(incompatible))
            return 2

    if args.offline and not args.resume:
        logger.error("âŒ --offline requires --resume FILE")
        return 2

    if args.dev:
        args.n_probes = 1
        args.gravity_levels = '0.5'
        if not args.master_seeds:
            args.master_seeds = ['42']
        if not args.tiers:
            args.tiers = '1'
        if not args.max_horizon:
            args.max_horizon = 3
        args.no_adaptive_expansion = True

    try:
        depths = [int(x.strip()) for x in args.depths.split(',')]
        gravity_levels = [float(x.strip()) for x in args.gravity_levels.split(',')]
        master_seeds = resolve_master_seeds(args.master_seeds)
        tiers = [int(x.strip()) for x in args.tiers.split(',')] if args.tiers else None
        adaptive_expansion = not args.no_adaptive_expansion
    except ValueError as e:
        logger.error(f"âŒ Invalid format: {e}")
        return 1

    os.makedirs("outputs", exist_ok=True)
    output_file = args.output or f"outputs/quickstart_results_{datetime.now():%Y%m%d_%H%M%S}.json"

    baseline_report = None
    if args.baselines:
        logger.info("Running model-free baseline evaluators (no model required)...")
        baseline_report = _run_baselines(args.n_probes, gravity_levels)
        logger.info(f"âœ“ Baselines complete: {list(baseline_report.keys())}")

    api_base = args.api_base

    model_name = args.model or ("google/gemini-2.5-flash-lite" if api_base else "Qwen/Qwen2.5-7B-Instruct")

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if api_base is not None and not api_key:
        if baseline_report is not None:
            # Baselines-only run: persist and exit successfully.
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({"baselines": baseline_report}, f, indent=2)
            logger.info(f"âœ“ Baseline-only report saved to {output_file}")
            return 0
        logger.error("âŒ No API key provided for API mode!")
        return 1

    if args.minitest:
        return _run_minitest(model_name, api_base, api_key)

    logger.info(f"Starting Benchmark for {model_name}")
    if master_seeds is not None:
        logger.info(f"Master seeds (variance set): {master_seeds}")

    try:
        runner = BenchmarkRunner(
            model_name=model_name,
            evaluator_class=ModelEvaluator,
            fracture_finder_class=FracturePointFinder,
            api_base=api_base,
            api_key=api_key,
            n_probes=args.n_probes,
            gravity_levels=gravity_levels,
            seeds=master_seeds,
            tiers=tiers,
            max_horizon=args.max_horizon,
            adaptive_expansion=adaptive_expansion,
        )

        results = runner.run(
            output_filepath=output_file,
            parallel_seeds=args.parallel_seeds,
            resume_path=args.resume,
            offline=args.offline,
        )

        # Merge baselines into the model report when both were requested.
        if baseline_report is not None:
            results["baselines"] = baseline_report

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info(f"âœ“ SUCCESS! Framework is working. Results: {output_file}")
        return 0

    except Exception as e:
        logger.error(f"âŒ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
