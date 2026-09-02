"""ARCUS-X: a deterministic benchmark for long-horizon sequential reasoning.

Public surface (everything else is internal):

    arcus.tasks            -> GridTaskGenerator, GridTask, TaskConfig
    arcus.environment      -> compute_trajectory, GridEnvironment, format_coord
    arcus.evaluation       -> ModelEvaluator, compare_trajectories, metrics
    arcus.analysis         -> DifficultyConfig, ErrorMode, FracturePointFinder
    arcus.experiments      -> BenchmarkRunner

Entry points: ``run_framework.sh`` and ``quickstart_api.py`` at the repo root.
"""

__version__ = "1.0.0"
