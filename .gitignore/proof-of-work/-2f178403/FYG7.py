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

import os
import sys
import logging
import argparse
from datetime import datetime
import json
from typing import Dict, List, Optional

# Canonical paper seed list. ``--master-seeds all`` expands to this set so that
# published results are reproducible across runs and across models. These are
# the master seeds used for the variance (mean +/- std) reporting in the paper.
PAPER_SEEDS: List[int] = [42, 12345, 6734, 9878, 2026, 9001, 1337, 31415]

# Corrected imports for the new arcus package structure
from arcus.experiments.benchmark_runner import BenchmarkRunner
from arcus.evaluation.evaluator import ModelEvaluator
from arcus.analysis.fracture import FracturePointFinder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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

    return parser.parse_args()


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


def main():
    args = parse_args()

    try:
        depths = [int(x.strip()) for x in args.depths.split(',')]
        gravity_levels = [float(x.strip()) for x in args.gravity_levels.split(',')]
        master_seeds = resolve_master_seeds(args.master_seeds)
    except ValueError as e:
        logger.error(f"❌ Invalid format: {e}")
        return 1

    os.makedirs("outputs", exist_ok=True)
    output_file = args.output or f"outputs/quickstart_results_{datetime.now():%Y%m%d_%H%M%S}.json"

    # --- Optional baselines (never require a model) ---
    baseline_report = None
    if args.baselines:
        logger.info("Running model-free baseline evaluators (no model required)...")
        baseline_report = _run_baselines(args.n_probes, gravity_levels)
        logger.info(f"✓ Baselines complete: {list(baseline_report.keys())}")

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        if baseline_report is not None:
            # Baselines-only run: persist and exit successfully.
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({"baselines": baseline_report}, f, indent=2)
            logger.info(f"✓ Baseline-only report saved to {output_file}")
            return 0
        logger.error("❌ No API key provided!")
        return 1

    model_name = args.model or "google/gemini-2.5-flash-lite"
    api_base = args.api_base or "https://ai.hackclub.com/proxy/v1/chat/completions"

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
        )

        results = runner.run(output_filepath=output_file)

        # Merge baselines into the model report when both were requested.
        if baseline_report is not None:
            results["baselines"] = baseline_report

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info(f"✓ SUCCESS! Framework is working. Results: {output_file}")
        return 0

    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
