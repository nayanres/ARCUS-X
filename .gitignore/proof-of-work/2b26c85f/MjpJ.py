"""Deterministic, evidence-based failure taxonomy for ARCUS-X.

This module replaces the previous coarse, tier-driven taxonomy with a layered,
*evidence-based* classification pipeline. The taxonomy explains **WHY** a model
failed a trajectory task, not merely that it failed.

Pipeline (each layer feeds the next):

    Layer 0  Outcome Classification      SUCCESS / FAILURE / UNCLASSIFIABLE
        |
        v
    Layer 1  Failure Mechanism           ERF / TEF / SRF / HDF / ORF / COMPOSITE
        |
        v
    Layer 2  Error Pattern Detection      deterministic detectors (wraparound,
                                          axis swap, sign inversion, ...)
        |
        v
    Layer 3  Failure Dynamics             first divergence, persistence, recovery,
                                          divergence evolution
        |
        v
    Layer 4  Trajectory Impact Metrics    mean / max / terminal divergence,
                                          persistence score, recovery rate

Design principles
-----------------
* **Tier is metadata only.** Tier labels never determine the classification.
  The taxonomy is derived purely from the trajectory evidence and the real task
  environment (transition rules, action sequence, grid dimensions).
* **Deterministic.** Given the same inputs the result is always identical.
* **Backwards compatible.** The legacy :class:`ErrorMode` enum values are
  preserved as aliases so existing callers (and the runner's ``error_mode``
  field) keep working, but new code should use the richer
  :class:`FailureMechanism` / :class:`ErrorPattern` enums.
* **Never guesses.** When required metadata is missing the result is
  ``UNCLASSIFIABLE`` rather than a fabricated category.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from arcus.environment.grid import parse_coord

logger = logging.getLogger(__name__)


# ===========================================================================
# Layer 0: Outcome Classification
# ===========================================================================
class Outcome(Enum):
    """Top-level outcome of a single trajectory comparison."""

    SUCCESS = "Success"
    FAILURE = "Failure"
    UNCLASSIFIABLE = "Unclassifiable"


# ===========================================================================
# Layer 1: Failure Mechanism Classification
# ===========================================================================
class FailureMechanism(Enum):
    """Primary failure mechanism (the *why* behind a divergence)."""

    # Environment Reconstruction Failure: wrong representation of the env.
    ERF = "Environment Reconstruction Failure"
    # Transition Execution Failure: right env, wrong transition applied.
    TEF = "Transition Execution Failure"
    # State Retention Failure: follows then loses state synchronization.
    SRF = "State Retention Failure"
    # Horizon Degradation Failure: behaviour degrades with reasoning horizon.
    HDF = "Horizon Degradation Failure"
    # Output Reliability Failure: output cannot be reliably evaluated.
    ORF = "Output Reliability Failure"
    # Composite: multiple mechanisms present.
    COMPOSITE = "Composite Failure"
    # No failure (success path).
    NONE = "None"


class FailureSubtype(Enum):
    """Fine-grained subtypes for each failure mechanism."""

    # ERF subtypes
    ERF_GRID_SIZE = "ERF_Grid_Size"
    ERF_WRAPAROUND = "ERF_Wraparound"
    ERF_COORDINATE_SYSTEM = "ERF_Coordinate_System"
    ERF_RULE_RECONSTRUCTION = "ERF_Rule_Reconstruction"

    # TEF subtypes
    TEF_DIRECTION_INVERSION = "TEF_Direction_Inversion"
    TEF_AXIS_SWAP = "TEF_Axis_Swap"
    TEF_SIGN_ERROR = "TEF_Sign_Error"
    TEF_MAGNITUDE_ERROR = "TEF_Magnitude_Error"
    TEF_PARITY_ERROR = "TEF_Parity_Error"
    TEF_GRAVITY_APPLICATION_ERROR = "TEF_Gravity_Application_Error"

    # SRF subtypes
    SRF_FIRST_DIVERGENCE = "SRF_First_Divergence"
    SRF_PERSISTENT_OFFSET = "SRF_Persistent_Offset"
    SRF_RECURSIVE_DRIFT = "SRF_Recursive_Drift"
    SRF_PARTIAL_RECOVERY = "SRF_Partial_Recovery"

    # HDF subtypes
    HDF_GRADUAL_DECAY = "HDF_Gradual_Decay"
    HDF_CLIFF_COLLAPSE = "HDF_Cliff_Collapse"
    HDF_MEMORY_DEGRADATION = "HDF_Memory_Degradation"

    # ORF subtypes
    ORF_EMPTY_RESPONSE = "ORF_Empty_Response"
    ORF_MALFORMED_STRUCTURE = "ORF_Malformed_Structure"
    ORF_TRUNCATION = "ORF_Truncation"
    ORF_INVALID_COORDINATE = "ORF_Invalid_Coordinate"
    ORF_EXTRA_TEXT = "ORF_Extra_Text"

    # Composite / none
    COMPOSITE = "Composite"
    NONE = "None"


# ===========================================================================
# Layer 2: Error Pattern Detection
# ===========================================================================
class ErrorPattern(Enum):
    """Deterministic, detectable error patterns in a trajectory."""

    WRAPAROUND = "Wraparound Error"
    OFF_BY_ONE = "Off-by-One Error"
    SIGN_INVERSION = "Sign Inversion"
    AXIS_SWAP = "Axis Swap"
    COORDINATE_CLAMP = "Coordinate Clamp"
    OSCILLATION = "Repeated State / Oscillation"
    FROZEN_STATE = "Frozen State"
    RANDOM_DIVERGENCE = "Random Divergence"
    NONE = "None"


# ===========================================================================
# Layer 3: Failure Dynamics
# ===========================================================================
class DivergenceEvolution(Enum):
    """Temporal character of the divergence after first detection."""

    TEMPORARY = "temporary"
    PERSISTENT = "persistent"
    INCREASING = "increasing"
    OSCILLATORY = "oscillatory"
    NONE = "none"


# ===========================================================================
# Backwards-compatible legacy aliases
# ===========================================================================
class ErrorMode(Enum):
    """Legacy enum preserved for backwards compatibility.

    New code should use :class:`FailureMechanism` / :class:`FailureSubtype`.
    The values here mirror the previous taxonomy so the runner's ``error_mode``
    field and existing reports keep rendering.
    """

    NONE = "None"
    STATE_TRACKING_FAILURE = "State Tracking Failure"
    TRANSITION_RULE_FAILURE = "Transition Rule Failure"
    SEMANTIC_INTERPRETATION_FAILURE = "Semantic Interpretation Failure"
    HORIZON_COLLAPSE = "Horizon Collapse"
    FORMATTING_FAILURE = "Formatting Failure"


# Mapping from the new mechanism/subtype to the closest legacy ErrorMode value
# (used only to keep ``error_mode`` strings stable for old reports).
_LEGACY_BY_MECHANISM = {
    FailureMechanism.ERF: ErrorMode.TRANSITION_RULE_FAILURE,
    FailureMechanism.TEF: ErrorMode.TRANSITION_RULE_FAILURE,
    FailureMechanism.SRF: ErrorMode.STATE_TRACKING_FAILURE,
    FailureMechanism.HDF: ErrorMode.HORIZON_COLLAPSE,
    FailureMechanism.ORF: ErrorMode.FORMATTING_FAILURE,
    FailureMechanism.COMPOSITE: ErrorMode.STATE_TRACKING_FAILURE,
    FailureMechanism.NONE: ErrorMode.NONE,
}


# ===========================================================================
# Data containers
# ===========================================================================
@dataclass(frozen=True)
class TrajectoryImpactMetrics:
    """Layer 4: measurable trajectory impact (no subjective severity)."""

    mean_divergence: float
    max_divergence: float
    terminal_divergence: float
    error_persistence_score: int
    recovery_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_divergence": self.mean_divergence,
            "max_divergence": self.max_divergence,
            "terminal_divergence": self.terminal_divergence,
            "error_persistence_score": self.error_persistence_score,
            "recovery_rate": self.recovery_rate,
        }


@dataclass(frozen=True)
class FailureDynamics:
    """Layer 3: temporal analysis of the failure."""

    first_divergence_step: Optional[int]
    error_persistence: int
    recovered: bool
    recovery_latency: Optional[int]
    divergence_evolution: DivergenceEvolution

    def to_dict(self) -> Dict[str, Any]:
        return {
            "first_divergence_step": self.first_divergence_step,
            "error_persistence": self.error_persistence,
            "recovered": self.recovered,
            "recovery_latency": self.recovery_latency,
            "divergence_evolution": self.divergence_evolution.value,
        }


@dataclass(frozen=True)
class TaxonomyResult:
    """Full layered classification result for a single trajectory."""

    outcome: Outcome
    mechanism: FailureMechanism
    subtype: FailureSubtype
    patterns: List[ErrorPattern] = field(default_factory=list)
    dynamics: Optional[FailureDynamics] = None
    impact: Optional[TrajectoryImpactMetrics] = None
    # Legacy single-mode value (kept for backwards compatibility with the
    # runner's ``error_mode`` field and old reports).
    legacy_mode: ErrorMode = ErrorMode.NONE
    valid: bool = True
    error: Optional[str] = None
    # Human-readable explanation of *why* (evidence-based).
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "mechanism": self.mechanism.value,
            "subtype": self.subtype.value,
            "patterns": [p.value for p in self.patterns],
            "dynamics": self.dynamics.to_dict() if self.dynamics else None,
            "impact": self.impact.to_dict() if self.impact else None,
            "legacy_mode": self.legacy_mode.value,
            "valid": self.valid,
            "error": self.error,
            "explanation": self.explanation,
        }


# ===========================================================================
# Geometry helpers (toroidal)
# ===========================================================================
def toroidal_distance(
    a: Tuple[int, int], b: Tuple[int, int], gw: int, gh: int
) -> float:
    """Wrapped (toroidal) Euclidean distance between two coordinates.

    dx = min(|x1-x2|, width - |x1-x2|)
    dy = min(|y1-y2|, height - |y1-y2|)
    distance = sqrt(dx^2 + dy^2)
    """
    if gw <= 0 or gh <= 0:
        # Degenerate grid: fall back to plain Euclidean distance.
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return math.sqrt(dx * dx + dy * dy)
    dx = min(abs(a[0] - b[0]), gw - abs(a[0] - b[0]))
    dy = min(abs(a[1] - b[1]), gh - abs(a[1] - b[1]))
    return math.sqrt(dx * dx + dy * dy)


def _wrapped_delta(
    a: Tuple[int, int], b: Tuple[int, int], gw: int, gh: int
) -> Tuple[int, int]:
    """Wrapped delta ``(a - b) mod (gw, gh)`` -- the rule the model applied."""
    return ((a[0] - b[0]) % gw, (a[1] - b[1]) % gh)


def _parse_coords(tokens: List[str]) -> Optional[List[Tuple[int, int]]]:
    """Parse canonical ``[x,y]`` tokens; return None on any parse failure."""
    coords: List[Tuple[int, int]] = []
    for t in tokens:
        try:
            coords.append(parse_coord(t))
        except (ValueError, TypeError):
            return None
    return coords


# ===========================================================================
# Layer 2: deterministic error-pattern detectors
# ===========================================================================
def _detect_wraparound(
    pred: List[Tuple[int, int]],
    gt: List[Tuple[int, int]],
    gw: int,
    gh: int,
) -> bool:
    """Detect incorrect toroidal boundary handling.

    The model's wrapped delta disagrees with the expected wrapped delta *and*
    the disagreement is consistent with a non-wrapping (clamped) interpretation
    of the boundary. Example: expected (0,0)->(11,0) but predicted (0,0)->(-1,0)
    -- the model moved off the grid instead of wrapping.
    """
    n = min(len(pred), len(gt))
    for i in range(1, n):
        exp = _wrapped_delta(gt[i], gt[i - 1], gw, gh)
        act = _wrapped_delta(pred[i], pred[i - 1], gw, gh)
        if act != exp:
            # Did the model fail to wrap? Compare raw (unwrapped) deltas.
            raw_exp = (gt[i][0] - gt[i - 1][0], gt[i][1] - gt[i - 1][1])
            raw_act = (pred[i][0] - pred[i - 1][0], pred[i][1] - pred[i - 1][1])
            if raw_act != raw_exp and act == raw_exp:
                # Model applied the correct *unwrapped* delta but the expected
                # delta wraps -> model ignored toroidal topology.
                return True
    return False


def _detect_off_by_one(
    pred: List[Tuple[int, int]], gt: List[Tuple[int, int]]
) -> bool:
    """Detect transitions where direction is correct but magnitude differs by 1."""
    n = min(len(pred), len(gt))
    for i in range(1, n):
        de = (gt[i][0] - gt[i - 1][0], gt[i][1] - gt[i - 1][1])
        da = (pred[i][0] - pred[i - 1][0], pred[i][1] - pred[i - 1][1])
        if de == (0, 0):
            continue
        # Same axis, same sign, magnitude off by exactly 1.
        if de[0] == 0 and da[0] == 0 and de[1] != 0 and da[1] != 0:
            if (da[1] > 0) == (de[1] > 0) and abs(abs(da[1]) - abs(de[1])) == 1:
                return True
        if de[1] == 0 and da[1] == 0 and de[0] != 0 and da[0] != 0:
            if (da[0] > 0) == (de[0] > 0) and abs(abs(da[0]) - abs(de[0])) == 1:
                return True
    return False


def _detect_sign_inversion(
    pred: List[Tuple[int, int]], gt: List[Tuple[int, int]]
) -> bool:
    """Detect opposite movement direction (delta negated)."""
    n = min(len(pred), len(gt))
    for i in range(1, n):
        de = (gt[i][0] - gt[i - 1][0], gt[i][1] - gt[i - 1][1])
        da = (pred[i][0] - pred[i - 1][0], pred[i][1] - pred[i - 1][1])
        if de == (0, 0):
            continue
        if da == (-de[0], -de[1]) and da != (0, 0):
            return True
    return False


def _detect_axis_swap(
    pred: List[Tuple[int, int]], gt: List[Tuple[int, int]]
) -> bool:
    """Detect dx/dy interchange."""
    n = min(len(pred), len(gt))
    for i in range(1, n):
        de = (gt[i][0] - gt[i - 1][0], gt[i][1] - gt[i - 1][1])
        da = (pred[i][0] - pred[i - 1][0], pred[i][1] - pred[i - 1][1])
        if de == (0, 0):
            continue
        if da == (de[1], de[0]) and da != de and de[0] != de[1]:
            return True
    return False


def _detect_coordinate_clamp(
    pred: List[Tuple[int, int]], gt: List[Tuple[int, int]], gw: int, gh: int
) -> bool:
    """Detect failure to wrap / move beyond expected bounds (stuck at edge)."""
    n = min(len(pred), len(gt))
    for i in range(1, n):
        exp = _wrapped_delta(gt[i], gt[i - 1], gw, gh)
        act = _wrapped_delta(pred[i], pred[i - 1], gw, gh)
        if act != exp:
            # Model coordinate pinned at a boundary while expected moved on.
            if (pred[i - 1][0] in (0, gw - 1) and pred[i][0] == pred[i - 1][0]
                    and gt[i][0] != gt[i - 1][0]):
                return True
            if (pred[i - 1][1] in (0, gh - 1) and pred[i][1] == pred[i - 1][1]
                    and gt[i][1] != gt[i - 1][1]):
                return True
    return False


def _detect_oscillation(
    pred: List[Tuple[int, int]], gt: List[Tuple[int, int]]
) -> bool:
    """Detect A -> B -> A -> B oscillation patterns."""
    n = min(len(pred), len(gt))
    if n < 4:
        return False
    for i in range(1, n - 2):
        a, b, c, d = pred[i - 1], pred[i], pred[i + 1], pred[i + 2]
        if a == c and b == d and a != b:
            return True
    return False


def _detect_frozen_state(
    pred: List[Tuple[int, int]], gt: List[Tuple[int, int]]
) -> bool:
    """Detect repeated identical coordinates (frozen state)."""
    n = min(len(pred), len(gt))
    if n < 2:
        return False
    for i in range(1, n):
        if pred[i] == pred[i - 1] and gt[i] != gt[i - 1]:
            return True
    return False


def _detect_random_divergence(
    pred: List[Tuple[int, int]],
    gt: List[Tuple[int, int]],
    gw: int,
    gh: int,
) -> bool:
