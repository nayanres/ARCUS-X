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

    The model produces coordinates outside the valid grid range
    ``[0, gw-1] x [0, gh-1]`` (e.g. ``(12, 0)`` on a 12-wide grid) instead of
    wrapping. Example: expected ``(11,0) -> (0,0)`` but predicted
    ``(11,0) -> (12,0)`` -- the model moved off the grid instead of wrapping.
    """
    if gw <= 0 or gh <= 0:
        return False
    for c in pred:
        if c[0] < 0 or c[0] >= gw or c[1] < 0 or c[1] >= gh:
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
    """Detect trajectories that no longer correlate with expected behaviour.

    Heuristic: the model's per-step deltas share no consistent relationship
    (sign/axis/magnitude) with the expected deltas across the divergent region.
    """
    n = min(len(pred), len(gt))
    mismatched = 0
    correlated = 0
    for i in range(1, n):
        de = (gt[i][0] - gt[i - 1][0], gt[i][1] - gt[i - 1][1])
        da = (pred[i][0] - pred[i - 1][0], pred[i][1] - pred[i - 1][1])
        if da == de:
            continue
        mismatched += 1
        # Correlated if sign-inverted, axis-swapped, or off-by-one on same axis.
        if da == (-de[0], -de[1]) or da == (de[1], de[0]):
            correlated += 1
    if mismatched == 0:
        return False
    # If fewer than 25% of mismatches show any structured relationship, treat
    # the divergence as random / uncorrelated.
    return (correlated / mismatched) < 0.25


def _run_pattern_detectors(
    pred: List[Tuple[int, int]],
    gt: List[Tuple[int, int]],
    gw: int,
    gh: int,
) -> List[ErrorPattern]:
    patterns: List[ErrorPattern] = []
    if _detect_wraparound(pred, gt, gw, gh):
        patterns.append(ErrorPattern.WRAPAROUND)
    if _detect_sign_inversion(pred, gt):
        patterns.append(ErrorPattern.SIGN_INVERSION)
    if _detect_axis_swap(pred, gt):
        patterns.append(ErrorPattern.AXIS_SWAP)
    if _detect_off_by_one(pred, gt):
        patterns.append(ErrorPattern.OFF_BY_ONE)
    if _detect_coordinate_clamp(pred, gt, gw, gh):
        patterns.append(ErrorPattern.COORDINATE_CLAMP)
    if _detect_oscillation(pred, gt):
        patterns.append(ErrorPattern.OSCILLATION)
    if _detect_frozen_state(pred, gt):
        patterns.append(ErrorPattern.FROZEN_STATE)
    if _detect_random_divergence(pred, gt, gw, gh):
        patterns.append(ErrorPattern.RANDOM_DIVERGENCE)
    if not patterns:
        patterns.append(ErrorPattern.NONE)
    return patterns


# ===========================================================================
# Layer 3: failure dynamics
# ===========================================================================
def _compute_dynamics(
    pred: List[Tuple[int, int]], gt: List[Tuple[int, int]]
) -> FailureDynamics:
    n = min(len(pred), len(gt))
    first_div: Optional[int] = None
    for i in range(n):
        if pred[i] != gt[i]:
            first_div = i  # 0-based index of first divergence
            break

    if first_div is None:
        return FailureDynamics(
            first_divergence_step=None,
            error_persistence=0,
            recovered=True,
            recovery_latency=0,
            divergence_evolution=DivergenceEvolution.NONE,
        )

    # Steps after first divergence until recovery (or end).
    mismatched_after = 0
    recovered = False
    recovery_latency: Optional[int] = None
    for i in range(first_div, n):
        if pred[i] == gt[i]:
            if i > first_div:
                recovered = True
                recovery_latency = i - first_div
            break
        mismatched_after += 1
    else:
        # Loop completed without recovery -> persistent to the end.
        mismatched_after = n - first_div

    # Divergence evolution: inspect the divergent window.
    window = [pred[i] != gt[i] for i in range(first_div, n)]
    if not window or all(not w for w in window):
        evolution = DivergenceEvolution.NONE
    elif recovered and recovery_latency is not None and recovery_latency <= 2:
        evolution = DivergenceEvolution.TEMPORARY
    elif _detect_oscillation(pred, gt):
        evolution = DivergenceEvolution.OSCILLATORY
    else:
        # Increasing if the toroidal distance trend grows; else persistent.
        dists = [
            toroidal_distance(pred[i], gt[i], _safe_gw(gt), _safe_gh(gt))
            for i in range(first_div, n)
        ]
        if len(dists) >= 3 and dists[-1] > dists[0] + 1e-9:
            evolution = DivergenceEvolution.INCREASING
        else:
            evolution = DivergenceEvolution.PERSISTENT

    return FailureDynamics(
        first_divergence_step=first_div,
        error_persistence=mismatched_after,
        recovered=recovered,
        recovery_latency=recovery_latency,
        divergence_evolution=evolution,
    )


def _safe_gw(gt: List[Tuple[int, int]]) -> int:
    if not gt:
        return 1
    return max(c[0] for c in gt) + 1


def _safe_gh(gt: List[Tuple[int, int]]) -> int:
    if not gt:
        return 1
    return max(c[1] for c in gt) + 1


# ===========================================================================
# Layer 4: trajectory impact metrics
# ===========================================================================
def _compute_impact(
    pred: List[Tuple[int, int]],
    gt: List[Tuple[int, int]],
    gw: int,
    gh: int,
    dynamics: FailureDynamics,
) -> TrajectoryImpactMetrics:
    n = min(len(pred), len(gt))
    if n == 0:
        return TrajectoryImpactMetrics(0.0, 0.0, 0.0, 0, 0.0)

    dists = [toroidal_distance(pred[i], gt[i], gw, gh) for i in range(n)]
    mean_div = sum(dists) / len(dists)
    max_div = max(dists) if dists else 0.0
    terminal_div = dists[-1] if dists else 0.0

    # Error persistence score: number of steps the trajectory remains incorrect.
    incorrect = sum(1 for i in range(n) if pred[i] != gt[i])
    error_persistence_score = incorrect

    # Recovery rate: 1.0 if recovered else 0.0 (single-trajectory view).
    recovery_rate = 1.0 if dynamics.recovered else 0.0

    return TrajectoryImpactMetrics(
        mean_divergence=round(mean_div, 4),
        max_divergence=round(max_div, 4),
        terminal_divergence=round(terminal_div, 4),
        error_persistence_score=error_persistence_score,
        recovery_rate=recovery_rate,
    )


# ===========================================================================
# Layer 1: mechanism + subtype attribution (evidence-based, tier-independent)
# ===========================================================================
def _attribute_mechanism(
    pred: List[Tuple[int, int]],
    gt: List[Tuple[int, int]],
    transition_rules: Dict[str, Dict[str, int]],
    actions: List[str],
    gw: int,
    gh: int,
    patterns: List[ErrorPattern],
    dynamics: FailureDynamics,
) -> Tuple[FailureMechanism, FailureSubtype, str]:
    """Determine the primary failure mechanism and subtype from evidence.

    Returns ``(mechanism, subtype, explanation)``. Tier is intentionally NOT
    used here.
    """
    n = min(len(pred), len(gt))

    # --- Initial-state mismatch => Environment Reconstruction Failure -------
    if pred and gt and pred[0] != gt[0]:
        # Distinguish grid-size / coordinate-system from rule reconstruction by
        # checking whether the *deltas* are still correct (just offset).
        if n >= 2:
            exp0 = _wrapped_delta(gt[1], gt[0], gw, gh)
            act0 = _wrapped_delta(pred[1], pred[0], gw, gh)
            if act0 == exp0:
                return (
                    FailureMechanism.ERF,
                    FailureSubtype.ERF_COORDINATE_SYSTEM,
                    "Initial coordinate offset but subsequent deltas are correct: "
                    "the model reconstructed the coordinate system with a wrong "
                    "origin.",
                )
        return (
            FailureMechanism.ERF,
            FailureSubtype.ERF_RULE_RECONSTRUCTION,
            "Initial state does not match ground truth: the model reconstructed "
            "the environment incorrectly from the start.",
        )

    # --- Per-step delta analysis against the real transition rules ----------
    mismatched: List[int] = []
    sign_inversions = 0
    axis_swaps = 0
    magnitude_errors = 0
    parity_errors = 0
    gravity_errors = 0
    wraparound_errors = 0
    for i in range(1, n):
        action = actions[i - 1] if i - 1 < len(actions) else None
        rule = transition_rules.get(action) if action else None
        if rule is None:
            continue
        exp = (int(rule["dx"]) % gw, int(rule["dy"]) % gh)
        act = _wrapped_delta(pred[i], pred[i - 1], gw, gh)
        # Wraparound misunderstanding: the model applied the *correct unwrapped*
        # delta but failed to wrap at the toroidal boundary. Detected when the
        # predicted coordinate equals the unwrapped correct position
        # (gt[i-1] + expected delta) yet lies outside the valid grid range.
        # This is distinct from a sign-inversion error, which yields negative
        # coordinates (the model moved the wrong way, not off the grid).
        unwrapped_correct = (gt[i - 1][0] + exp[0], gt[i - 1][1] + exp[1])
        if pred[i] == unwrapped_correct and (
            pred[i][0] < 0 or pred[i][0] >= gw or pred[i][1] < 0 or pred[i][1] >= gh
        ):
            wraparound_errors += 1
            mismatched.append(i)
            continue
        if act == exp:
            continue
        mismatched.append(i)
        de = (gt[i][0] - gt[i - 1][0], gt[i][1] - gt[i - 1][1])
        da = (pred[i][0] - pred[i - 1][0], pred[i][1] - pred[i - 1][1])
        # Wraparound misunderstanding: the model applied the *correct unwrapped*
        # delta but the expected delta wraps around the toroidal boundary.
        if da == de and act == de and act != exp:
            wraparound_errors += 1
            continue
        if da == (-de[0], -de[1]) and da != (0, 0):
            sign_inversions += 1
        elif da == (de[1], de[0]) and da != de and de[0] != de[1]:
            axis_swaps += 1
        else:
            # Magnitude / parity / gravity: compare against the canonical rule.
            if abs(act[0]) != abs(exp[0]) or abs(act[1]) != abs(exp[1]):
                magnitude_errors += 1
            if (act[0] % 2) != (exp[0] % 2) or (act[1] % 2) != (exp[1] % 2):
                parity_errors += 1
            if act != (int(rule["dx"]), int(rule["dy"])) and act == exp:
                # Applied the wrapped rule but not the raw gravity rule.
                gravity_errors += 1

    # Wraparound misunderstanding (ERF) takes precedence: out-of-bounds
    # coordinates are unambiguous evidence of a failed toroidal wrap, even when
    # the wrapped delta coincides with the expected transition.
    if wraparound_errors > 0 and sign_inversions == 0 and axis_swaps == 0 \
            and magnitude_errors == 0 and parity_errors == 0:
        return (
            FailureMechanism.ERF,
            FailureSubtype.ERF_WRAPAROUND,
            f"{wraparound_errors} step(s) produced out-of-bounds coordinates: "
            "the model failed to wrap at the toroidal boundary (wraparound error).",
        )

    if not mismatched:
        return (
            FailureMechanism.NONE,
            FailureSubtype.NONE,
            "Predicted trajectory exactly matches ground truth.",
        )

    total_mismatch = len(mismatched)

    # Count steps where the *position* diverges (state desync) vs steps where
    # the *delta* (transition) is wrong. This separates State Retention Failures
    # (deltas correct, positions drift) from Transition Execution Failures
    # (deltas consistently wrong).
    wrong_position_steps = sum(1 for i in range(n) if pred[i] != gt[i])
    wrong_delta_steps = total_mismatch

    # --- State Retention Failure --------------------------------------------
    # The model follows correctly at first (correct deltas), then loses state
    # synchronization. Detected when the model's deltas are mostly correct but
    # positions diverge (wrong_position_steps >> wrong_delta_steps), i.e. a
    # single offset is introduced and subsequent deltas maintain it.
    if dynamics.first_divergence_step is not None and dynamics.first_divergence_step > 0:
        # The model followed correctly at first, then lost state
        # synchronization (SRF). A Transition Execution Failure, by contrast,
        # applies a wrong transition from the *very first* step
        # (first_divergence_step == 0). Because the model demonstrably followed
        # the correct transition initially here, the divergence is attributed to
        # state desynchronization rather than a systematic transition error.
        if dynamics.recovered and dynamics.recovery_latency is not None:
            return (
                FailureMechanism.SRF,
                FailureSubtype.SRF_PARTIAL_RECOVERY,
                f"Model followed correctly until step "
                f"{dynamics.first_divergence_step}, diverged, then partially "
                f"recovered after {dynamics.recovery_latency} steps.",
            )
        if dynamics.divergence_evolution == DivergenceEvolution.INCREASING:
            return (
                FailureMechanism.SRF,
                FailureSubtype.SRF_RECURSIVE_DRIFT,
                f"Model followed correctly until step "
                f"{dynamics.first_divergence_step}, then drift increased "
                "monotonically (recursive drift).",
            )
        if dynamics.divergence_evolution == DivergenceEvolution.OSCILLATORY:
            return (
                FailureMechanism.SRF,
                FailureSubtype.SRF_PERSISTENT_OFFSET,
                f"Model followed correctly until step "
                f"{dynamics.first_divergence_step}, then entered an "
                "oscillatory offset.",
            )
        return (
            FailureMechanism.SRF,
            FailureSubtype.SRF_PERSISTENT_OFFSET,
            f"Model followed correctly until step "
            f"{dynamics.first_divergence_step}, then held a persistent offset "
            "(state retention failure).",
        )

    # --- Transition Execution Failure ---------------------------------------
    # TEF requires the model to *consistently* apply a wrong transition. We
    # only attribute TEF when the majority of mismatched steps show a structured
    # wrong-delta pattern (sign inversion / axis swap / magnitude / parity).
    structured = sign_inversions + axis_swaps + magnitude_errors + parity_errors
    if structured >= total_mismatch / 2:
        if sign_inversions >= total_mismatch / 2:
            return (
                FailureMechanism.TEF,
                FailureSubtype.TEF_DIRECTION_INVERSION,
                f"{sign_inversions}/{total_mismatch} step deltas are sign-inverted "
                "relative to the expected transition.",
            )
        if axis_swaps >= total_mismatch / 2:
            return (
                FailureMechanism.TEF,
                FailureSubtype.TEF_AXIS_SWAP,
                f"{axis_swaps}/{total_mismatch} step deltas show dx/dy interchange.",
            )
        if magnitude_errors >= total_mismatch / 2:
            return (
                FailureMechanism.TEF,
                FailureSubtype.TEF_MAGNITUDE_ERROR,
                f"{magnitude_errors}/{total_mismatch} step deltas have wrong magnitude.",
            )
        if parity_errors >= total_mismatch / 2:
            return (
                FailureMechanism.TEF,
                FailureSubtype.TEF_PARITY_ERROR,
                f"{parity_errors}/{total_mismatch} step deltas have wrong parity.",
            )
        if gravity_errors >= total_mismatch / 2:
            return (
                FailureMechanism.TEF,
                FailureSubtype.TEF_GRAVITY_APPLICATION_ERROR,
                "Model applied wrapped deltas instead of the raw gravity rule.",
            )

    # --- Horizon Degradation Failure ----------------------------------------
    # Divergence at the very first step but correlated, growing with horizon.
    if dynamics.first_divergence_step == 0 and \
            dynamics.divergence_evolution == DivergenceEvolution.INCREASING:
        return (
            FailureMechanism.HDF,
            FailureSubtype.HDF_GRADUAL_DECAY,
            "Divergence begins immediately and grows with horizon (gradual decay).",
        )

    # --- Composite (multiple strong signals) --------------------------------
    strong_signals = sum(
        1 for c in (sign_inversions, axis_swaps, magnitude_errors, parity_errors)
        if c >= total_mismatch / 2
    )
    if strong_signals >= 2:
        return (
            FailureMechanism.COMPOSITE,
            FailureSubtype.COMPOSITE,
            "Multiple distinct failure mechanisms detected simultaneously.",
        )

    # --- Default: state retention loss --------------------------------------
    # When the model's deltas are mostly correct but positions diverge, the
    # failure is a loss of state synchronization, not a transition error.
    return (
        FailureMechanism.SRF,
        FailureSubtype.SRF_PERSISTENT_OFFSET,
        "State synchronization lost after an early divergence (persistent offset).",
    )


# ===========================================================================
# Public entry points
# ===========================================================================
def classify_trajectory(
    predicted: List[str],
    ground_truth: List[str],
    transition_rules: Optional[Dict[str, Dict[str, int]]] = None,
    actions: Optional[List[str]] = None,
    tier: Optional[int] = None,
    grid_width: Optional[int] = None,
    grid_height: Optional[int] = None,
    raw_output: Optional[str] = None,
) -> TaxonomyResult:
    """Full layered, evidence-based classification of a trajectory.

    Parameters
    ----------
    predicted, ground_truth:
        Canonical ``[x,y]`` token lists.
    transition_rules:
        The task's ``action -> {dx, dy}`` mapping (the real environment).
    actions:
        The action sequence that produced ``ground_truth``.
    tier:
        Difficulty tier (0-3). **Metadata only** -- never drives classification.
    grid_width, grid_height:
        Toroidal grid dimensions (needed for wrapped distance / deltas).
    raw_output:
        The raw model text (used to detect ORF patterns like extra text).

    Returns
    -------
    TaxonomyResult
    """
    gw = int(grid_width) if grid_width else 0
    gh = int(grid_height) if grid_height else 0
    rules = transition_rules or {}
    acts = actions or []

    # ---- Layer 0: Outcome Classification -----------------------------------
    # UNCLASSIFIABLE: missing metadata / parser failure / invalid format.
    if gw <= 0 or gh <= 0 or not rules or not acts:
        # We can still attempt a structural comparison if both parse, but the
        # taxonomy cannot be evidence-based without the environment. Mark the
        # mechanism as UNCLASSIFIABLE only when parsing also fails.
        pass

    pred_coords = _parse_coords(predicted)
    gt_coords = _parse_coords(ground_truth)

    # Parser failure / invalid trajectory format => ORF / UNCLASSIFIABLE.
    if pred_coords is None:
        # Try to detect *why* the output is unreliable.
        subtype = FailureSubtype.ORF_MALFORMED_STRUCTURE
        if raw_output is not None and not raw_output.strip():
            subtype = FailureSubtype.ORF_EMPTY_RESPONSE
        elif raw_output is not None and len(predicted) == 0:
            subtype = FailureSubtype.ORF_EMPTY_RESPONSE
        return TaxonomyResult(
            outcome=Outcome.UNCLASSIFIABLE,
            mechanism=FailureMechanism.ORF,
            subtype=subtype,
            patterns=[ErrorPattern.NONE],
            dynamics=None,
            impact=None,
            legacy_mode=ErrorMode.FORMATTING_FAILURE,
            valid=True,
            explanation="Model output could not be parsed into a valid "
                        "coordinate trajectory (output reliability failure).",
        )

    if gt_coords is None:
        return TaxonomyResult(
            outcome=Outcome.UNCLASSIFIABLE,
            mechanism=FailureMechanism.ORF,
            subtype=FailureSubtype.ORF_INVALID_COORDINATE,
            patterns=[ErrorPattern.NONE],
            dynamics=None,
            impact=None,
            legacy_mode=ErrorMode.FORMATTING_FAILURE,
            valid=True,
            explanation="Ground-truth trajectory is in an invalid coordinate "
                        "format; cannot classify.",
        )

    # Empty predicted trajectory.
    if not pred_coords:
        return TaxonomyResult(
            outcome=Outcome.UNCLASSIFIABLE,
            mechanism=FailureMechanism.ORF,
            subtype=FailureSubtype.ORF_EMPTY_RESPONSE,
            patterns=[ErrorPattern.NONE],
            dynamics=None,
            impact=None,
            legacy_mode=ErrorMode.FORMATTING_FAILURE,
            valid=True,
            explanation="Model produced no parseable coordinates (empty response).",
        )

    # SUCCESS: exact match.
    if pred_coords == gt_coords:
        return TaxonomyResult(
            outcome=Outcome.SUCCESS,
            mechanism=FailureMechanism.NONE,
            subtype=FailureSubtype.NONE,
            patterns=[ErrorPattern.NONE],
            dynamics=FailureDynamics(
                first_divergence_step=None, error_persistence=0,
                recovered=True, recovery_latency=0,
                divergence_evolution=DivergenceEvolution.NONE,
            ),
            impact=TrajectoryImpactMetrics(0.0, 0.0, 0.0, 0, 1.0),
            legacy_mode=ErrorMode.NONE,
            valid=True,
            explanation="Predicted trajectory exactly matches ground truth.",
        )

    # ---- FAILURE path ------------------------------------------------------
    # Extra-text detection (ORF): raw output carries coordinates but also a
    # large amount of non-coordinate prose. We still classify the trajectory,
    # but record the ORF pattern as a secondary signal.
    extra_text = False
    if raw_output is not None:
        coord_chars = sum(len(t) for t in predicted)
        non_coord = max(0, len(raw_output) - coord_chars)
        if non_coord > max(40, 2 * coord_chars):
            extra_text = True

    patterns = _run_pattern_detectors(pred_coords, gt_coords, gw, gh)
    if extra_text:
        patterns = [p for p in patterns if p != ErrorPattern.NONE]
        patterns.append(ErrorPattern.NONE) if not patterns else None
    dynamics = _compute_dynamics(pred_coords, gt_coords)
    impact = _compute_impact(pred_coords, gt_coords, gw, gh, dynamics)

    mechanism, subtype, explanation = _attribute_mechanism(
        pred_coords, gt_coords, rules, acts, gw, gh, patterns, dynamics
    )

    if extra_text and mechanism != FailureMechanism.ORF:
        # Keep the primary mechanism but note the reliability issue.
        explanation += " (model output also contained extraneous prose)"

    legacy = _LEGACY_BY_MECHANISM.get(mechanism, ErrorMode.STATE_TRACKING_FAILURE)

    return TaxonomyResult(
        outcome=Outcome.FAILURE,
        mechanism=mechanism,
        subtype=subtype,
        patterns=patterns,
        dynamics=dynamics,
        impact=impact,
        legacy_mode=legacy,
        valid=True,
        explanation=explanation,
    )


def classify_from_result(
    predicted: List[str],
    ground_truth: List[str],
    transition_rules: Optional[Dict[str, Dict[str, int]]] = None,
    actions: Optional[List[str]] = None,
    tier: Optional[int] = None,
    grid_width: Optional[int] = None,
    grid_height: Optional[int] = None,
    raw_output: Optional[str] = None,
) -> TaxonomyResult:
    """Backwards-compatible wrapper around :func:`classify_trajectory`.

    The previous signature passed ``tier`` positionally; it is now metadata-only
    and accepted for compatibility but never drives classification.
    """
    try:
        return classify_trajectory(
            predicted, ground_truth, transition_rules, actions, tier,
            grid_width, grid_height, raw_output,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(f"[TAXONOMY] classification failed: {exc}")
        return TaxonomyResult(
            outcome=Outcome.UNCLASSIFIABLE,
            mechanism=FailureMechanism.ORF,
            subtype=FailureSubtype.ORF_MALFORMED_STRUCTURE,
            patterns=[ErrorPattern.NONE],
            dynamics=None,
            impact=None,
            legacy_mode=ErrorMode.FORMATTING_FAILURE,
            valid=False,
            error=str(exc),
        )


# Backwards-compatible alias kept so old imports keep working.
def classify_trajectory_error(
    predicted: List[str],
    ground_truth: List[str],
    transition_rules: Dict[str, Dict[str, int]],
    actions: List[str],
    tier: int,
    grid_width: int,
    grid_height: int,
):
    """Deprecated shim returning a legacy ``ErrorModeClassification``-like dict.

    Retained only so any external caller referencing the old function signature
    continues to import. New code should use :func:`classify_trajectory`.
    """
    result = classify_trajectory(
        predicted, ground_truth, transition_rules, actions, tier,
        grid_width, grid_height,
    )
    from dataclasses import replace
    # Return a minimal object with the legacy attributes for compatibility.
    class _Legacy:
        mode = result.legacy_mode
        confidence = 1.0 if result.outcome != Outcome.SUCCESS else 1.0
        score = float(result.impact.error_persistence_score) if result.impact else 0.0
        evidence_indices = (
            [result.dynamics.first_divergence_step]
            if result.dynamics and result.dynamics.first_divergence_step is not None
            else []
        )

        def to_dict(self):
            return {
                "mode": self.mode.value,
                "confidence": self.confidence,
                "score": self.score,
                "evidence_indices": self.evidence_indices,
            }
    return _Legacy()
