"""ARCUS-X evaluation package: parsing, metrics, and model inference."""

from arcus.evaluation.parser import (
    extract_model_path,
    estimated_optimal_path_length,
    parse_first_coord,
)
from arcus.evaluation.metrics import (
    compare_trajectories,
    TrajectoryResult,
    HorizonCompliance,
    GenerationBloatIndex,
    GenerationEfficiency,
    aggregate_trajectory_results,
    ARCUSRobustnessIndex,
    CRISummary,
)
from arcus.evaluation.evaluator import ModelEvaluator
from arcus.evaluation.token_accounting import measure_tokens

__all__ = [
    "extract_model_path",
    "estimated_optimal_path_length",
    "parse_first_coord",
    "compare_trajectories",
    "TrajectoryResult",
    "HorizonCompliance",
    "GenerationBloatIndex",
    "GenerationEfficiency",
    "aggregate_trajectory_results",
    "ARCUSRobustnessIndex",
    "CRISummary",
    "ModelEvaluator",
    "measure_tokens",
]
from .observations import ObservationKey, ObservationRecord, ResumeObservationStore

__all__ = ["ObservationKey", "ObservationRecord", "ResumeObservationStore"]
