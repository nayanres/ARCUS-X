"""ARCUS-X analysis package: difficulty, error taxonomy, and fracture detection."""

from arcus.analysis.gravity import (
    DifficultyTier,
    DifficultyConfig,
    build_gravity_context,
)
from arcus.analysis.taxonomy import (
    ErrorMode,
    classify_trajectory_error,
    classify_from_result,
    classify_trajectory,
    Outcome,
    FailureMechanism,
    FailureSubtype,
    ErrorPattern,
    DivergenceEvolution,
    TaxonomyResult,
)
from arcus.analysis.fracture import FracturePointFinder

__all__ = [
    "DifficultyTier",
    "DifficultyConfig",
    "build_gravity_context",
    "ErrorMode",
    "classify_trajectory_error",
    "classify_from_result",
    "classify_trajectory",
    "Outcome",
    "FailureMechanism",
    "FailureSubtype",
    "ErrorPattern",
    "DivergenceEvolution",
    "TaxonomyResult",
    "FracturePointFinder",
]

