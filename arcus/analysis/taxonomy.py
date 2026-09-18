"""Deterministic, evidence-based failure taxonomy for ARCUS-X with separated Output Validity
and Broad Cognitive Failure Taxonomy.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from arcus.environment.grid import parse_coord

logger = logging.getLogger(__name__)


# Invalid Error Codes (Output Validity Layer)
E100_EXCEPTION_TOKEN = "E100_EXCEPTION_TOKEN"
E101_EMPTY_OUTPUT = "E101_EMPTY_OUTPUT"
E102_PARSE_FAILURE = "E102_PARSE_FAILURE"
E103_PROVIDER_ERROR = "E103_PROVIDER_ERROR"
E104_UNKNOWN_INVALID = "E104_UNKNOWN_INVALID"


# Output Compliance Layer (Formatting / Protocol Evaluation)
class OutputCompliance(Enum):
    """Output protocol compliance classification.

    Separates formatting/protocol issues from cognitive correctness.
    """

    PERFECT_FORMAT = "Perfect Format"
    VERBOSE_CORRECT = "Verbose Correct"
    MALFORMED_OUTPUT = "Malformed Output"
    NO_TRAJECTORY = "No Trajectory"
    PARSE_FAILURE = "Parse Failure"


# Formatting Subcodes (deterministic, small taxonomy)
F001_VERBOSE_CORRECT = "F001_VERBOSE_CORRECT"
F002_MARKDOWN_WRAPPER = "F002_MARKDOWN_WRAPPER"
F003_INVALID_DELIMITERS = "F003_INVALID_DELIMITERS"
F004_MISSING_TRAJECTORY = "F004_MISSING_TRAJECTORY"
F005_MALFORMED_TRAJECTORY = "F005_MALFORMED_TRAJECTORY"
F006_PARSE_FAILURE = "F006_PARSE_FAILURE"


# Layer 0: Outcome Classification
class Outcome(Enum):
    """Top-level outcome of a single trajectory comparison."""

    SUCCESS = "Success"
    FAILURE = "Failure"
    UNCLASSIFIABLE = "Unclassifiable"


# Cognitive Failure Buckets (Broad & Scientifically Defensible)
class FailureMechanism(Enum):
    """Primary broad cognitive failure mechanism.

    OUTPUT_FORMAT_FAILURE applies only when:
    - trajectory is incorrect because formatting prevented extraction, OR
    - required structure is violated and no correct trajectory can be recovered.
    """

    NONE = "None"
    STATE_TRACKING_FAILURE = "State Tracking Failure"
    TRANSITION_FAILURE = "Transition Failure"
    SEMANTIC_FAILURE = "Semantic Failure"
    OUTPUT_FORMAT_FAILURE = "Output Format Failure"
    HORIZON_FAILURE = "Horizon Failure"
    UNKNOWN_FAILURE = "Unknown Failure"


# Fine-grained subtypes (retained for evidence mapping)
class FailureSubtype(Enum):
    """Fine-grained subtypes for analysis."""

    ERF_GRID_SIZE = "ERF_Grid_Size"
    ERF_WRAPAROUND = "ERF_Wraparound"
    ERF_COORDINATE_SYSTEM = "ERF_Coordinate_System"
    ERF_RULE_RECONSTRUCTION = "ERF_Rule_Reconstruction"

    TEF_DIRECTION_INVERSION = "TEF_Direction_Inversion"
    TEF_AXIS_SWAP = "TEF_Axis_Swap"
    TEF_SIGN_ERROR = "TEF_Sign_Error"
    TEF_MAGNITUDE_ERROR = "TEF_Magnitude_Error"
    TEF_PARITY_ERROR = "TEF_Parity_Error"
    TEF_GRAVITY_APPLICATION_ERROR = "TEF_Gravity_Application_Error"

    SRF_FIRST_DIVERGENCE = "SRF_First_Divergence"
    SRF_PERSISTENT_OFFSET = "SRF_Persistent_Offset"
    SRF_RECURSIVE_DRIFT = "SRF_Recursive_Drift"
    SRF_PARTIAL_RECOVERY = "SRF_Partial_Recovery"

    HDF_GRADUAL_DECAY = "HDF_Gradual_Decay"
    HDF_CLIFF_COLLAPSE = "HDF_Cliff_Collapse"
    HDF_MEMORY_DEGRADATION = "HDF_Memory_Degradation"

    ORF_EMPTY_RESPONSE = "ORF_Empty_Response"
    ORF_MALFORMED_STRUCTURE = "ORF_Malformed_Structure"
    ORF_TRUNCATION = "ORF_Truncation"
    ORF_INVALID_COORDINATE = "ORF_Invalid_Coordinate"
    ORF_EXTRA_TEXT = "ORF_Extra_Text"

    COMPOSITE = "Composite"
    NONE = "None"


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


class DivergenceEvolution(Enum):
    """Temporal character of the divergence after first detection."""

    TEMPORARY = "temporary"
    PERSISTENT = "persistent"
    INCREASING = "increasing"
    OSCILLATORY = "oscillatory"
    NONE = "none"


# Backwards-compatible legacy aliases
class ErrorMode(Enum):
    """Legacy enum preserved for backwards compatibility."""

    NONE = "None"
    STATE_TRACKING_FAILURE = "State Tracking Failure"
    TRANSITION_RULE_FAILURE = "Transition Rule Failure"
    SEMANTIC_INTERPRETATION_FAILURE = "Semantic Interpretation Failure"
    HORIZON_COLLAPSE = "Horizon Collapse"
    FORMATTING_FAILURE = "Formatting Failure"


# Canonical Classification Objects
@dataclass(frozen=True)
class ProbeClassification:
    """Single source of truth classification object separating validity and cognitive failure."""
    valid: bool
    exception_code: Optional[str]
    mechanism: FailureMechanism
    first_failure_step: Optional[int]
    secondary_flags: tuple[str, ...] = field(default_factory=tuple)
    trajectory_correct: bool = False
    step_accuracy: float = 0.0
    exact_match: bool = False
    output_compliance: OutputCompliance = OutputCompliance.VERBOSE_CORRECT
    formatting_subcode: str = F001_VERBOSE_CORRECT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "exception_code": self.exception_code,
            "mechanism": self.mechanism.value,
            "first_failure_step": self.first_failure_step,
            "secondary_flags": list(self.secondary_flags),
            "trajectory_correct": self.trajectory_correct,
            "step_accuracy": self.step_accuracy,
            "exact_match": self.exact_match,
            "output_compliance": self.output_compliance.value,
            "formatting_subcode": self.formatting_subcode,
        }



@dataclass(frozen=True)
class TaxonomyResult:
    outcome: Outcome
    mechanism: FailureMechanism
    subtype: FailureSubtype
    patterns: List[ErrorPattern]
    dynamics: Optional[FailureDynamics]
    impact: Optional[TrajectoryImpactMetrics]
    legacy_mode: ErrorMode
    valid: bool
    error: Optional[str] = None
    explanation: Optional[str] = None
    probe_classification: Optional[ProbeClassification] = None

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
            "probe_classification": self.probe_classification.to_dict() if self.probe_classification else None,
        }


@dataclass(frozen=True)
class TrajectoryImpactMetrics:
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





# Geometry helpers (toroidal)
def toroidal_distance(
    a: Tuple[int, int], b: Tuple[int, int], gw: int, gh: int
) -> float:
    if gw <= 0 or gh <= 0:
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return math.sqrt(dx * dx + dy * dy)
    dx = min(abs(a[0] - b[0]), gw - abs(a[0] - b[0]))
    dy = min(abs(a[1] - b[1]), gh - abs(a[1] - b[1]))
    return math.sqrt(dx * dx + dy * dy)


def _wrapped_delta(
    a: Tuple[int, int], b: Tuple[int, int], gw: int, gh: int
) -> Tuple[int, int]:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    if gw > 0:
        dx = dx % gw
    if gh > 0:
        dy = dy % gh
    return (dx, dy)


def _parse_coords(tokens: List[str]) -> Optional[List[Tuple[int, int]]]:
    coords: List[Tuple[int, int]] = []
    for t in tokens:
        try:
            coords.append(parse_coord(t))
        except (ValueError, TypeError):
            return None
    return coords


def _detect_wraparound(
    pred: List[Tuple[int, int]],
    gt: List[Tuple[int, int]],
    gw: int,
    gh: int,
) -> bool:
    if gw <= 0 or gh <= 0:
        return False
    for c in pred:
        if c[0] < 0 or c[0] >= gw or c[1] < 0 or c[1] >= gh:
            return True
    return False


def _detect_off_by_one(
    pred: List[Tuple[int, int]], gt: List[Tuple[int, int]]
) -> bool:
    n = min(len(pred), len(gt))
    for i in range(1, n):
        de = (gt[i][0] - gt[i - 1][0], gt[i][1] - gt[i - 1][1])
        da = (pred[i][0] - pred[i - 1][0], pred[i][1] - pred[i - 1][1])
        if de == (0, 0):
            continue
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
    n = min(len(pred), len(gt))
    for i in range(1, n):
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
    n = min(len(pred), len(gt))
    mismatched = 0
    correlated = 0
    for i in range(1, n):
        de = (gt[i][0] - gt[i - 1][0], gt[i][1] - gt[i - 1][1])
        da = (pred[i][0] - pred[i - 1][0], pred[i][1] - pred[i - 1][1])
        if da == de:
            continue
        mismatched += 1
        if da == (-de[0], -de[1]) or da == (de[1], de[0]):
            correlated += 1
    if mismatched == 0:
        return False
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


def _compute_dynamics(
    pred: List[Tuple[int, int]], gt: List[Tuple[int, int]]
) -> FailureDynamics:
    n = min(len(pred), len(gt))
    first_div: Optional[int] = None
    for i in range(n):
        if pred[i] != gt[i]:
            first_div = i
            break

    if first_div is None:
        return FailureDynamics(
            first_divergence_step=None,
            error_persistence=0,
            recovered=True,
            recovery_latency=0,
            divergence_evolution=DivergenceEvolution.NONE,
        )

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
        mismatched_after = n - first_div

    window = [pred[i] != gt[i] for i in range(first_div, n)]
    if not window or all(not w for w in window):
        evolution = DivergenceEvolution.NONE
    elif recovered and recovery_latency is not None and recovery_latency <= 2:
        evolution = DivergenceEvolution.TEMPORARY
    elif _detect_oscillation(pred, gt):
        evolution = DivergenceEvolution.OSCILLATORY
    else:
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
    incorrect = sum(1 for i in range(n) if pred[i] != gt[i])
    recovery_rate = 1.0 if dynamics.recovered else 0.0

    return TrajectoryImpactMetrics(
        mean_divergence=round(mean_div, 4),
        max_divergence=round(max_div, 4),
        terminal_divergence=round(terminal_div, 4),
        error_persistence_score=incorrect,
        recovery_rate=recovery_rate,
    )


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
    """Determine the primary failure mechanism and map to the new broad buckets."""
    n = min(len(pred), len(gt))

    # Initial-state mismatch -> Semantic or Transition Failure
    if pred and gt and pred[0] != gt[0]:
        if n >= 2:
            exp0 = _wrapped_delta(gt[1], gt[0], gw, gh)
            act0 = _wrapped_delta(pred[1], pred[0], gw, gh)
            if act0 == exp0:
                return (
                    FailureMechanism.SEMANTIC_FAILURE,
                    FailureSubtype.ERF_COORDINATE_SYSTEM,
                    "Initial coordinate offset but subsequent deltas are correct (semantic coordinate offset).",
                )
        return (
            FailureMechanism.SEMANTIC_FAILURE,
            FailureSubtype.ERF_RULE_RECONSTRUCTION,
            "Initial state does not match ground truth (semantic rule/environment reconstruction failure).",
        )

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
        dx_rule = int(rule["dx"])
        dy_rule = int(rule["dy"])
        if gw > 0:
            dx_rule = dx_rule % gw
        if gh > 0:
            dy_rule = dy_rule % gh
        exp = (dx_rule, dy_rule)
        act = _wrapped_delta(pred[i], pred[i - 1], gw, gh)
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
        if da == de and act == de and act != exp:
            wraparound_errors += 1
            continue
        if da == (-de[0], -de[1]) and da != (0, 0):
            sign_inversions += 1
        elif da == (de[1], de[0]) and da != de and de[0] != de[1]:
            axis_swaps += 1
        else:
            if abs(act[0]) != abs(exp[0]) or abs(act[1]) != abs(exp[1]):
                magnitude_errors += 1
            if (act[0] % 2) != (exp[0] % 2) or (act[1] % 2) != (exp[1] % 2):
                parity_errors += 1
            if act != (int(rule["dx"]), int(rule["dy"])) and act == exp:
                gravity_errors += 1

    if wraparound_errors > 0 and sign_inversions == 0 and axis_swaps == 0 \
            and magnitude_errors == 0 and parity_errors == 0:
        return (
            FailureMechanism.TRANSITION_FAILURE,
            FailureSubtype.ERF_WRAPAROUND,
            f"{wraparound_errors} step(s) produced out-of-bounds coordinates (transition/boundary failure).",
        )

    if not mismatched:
        if len(pred) != len(gt):
            return (
                FailureMechanism.HORIZON_FAILURE,
                FailureSubtype.ORF_EMPTY_RESPONSE,
                "Predicted trajectory length differs from ground truth (horizon/truncation mismatch).",
            )
        return (
            FailureMechanism.NONE,
            FailureSubtype.NONE,
            "Predicted trajectory exactly matches ground truth.",
        )

    total_mismatch = len(mismatched)

    # State Tracking Failure
    if dynamics.first_divergence_step is not None and dynamics.first_divergence_step >= 2:
        if dynamics.divergence_evolution == DivergenceEvolution.INCREASING:
            return (
                FailureMechanism.STATE_TRACKING_FAILURE,
                FailureSubtype.SRF_RECURSIVE_DRIFT,
                f"Model followed correctly until step {dynamics.first_divergence_step}, then drift increased (state tracking failure / recursive drift).",
            )
        return (
            FailureMechanism.STATE_TRACKING_FAILURE,
            FailureSubtype.SRF_PERSISTENT_OFFSET,
            f"Model followed correctly until step {dynamics.first_divergence_step}, then held persistent offset (state tracking failure).",
        )

    # Transition Failure
    structured = sign_inversions + axis_swaps + magnitude_errors + parity_errors
    if structured >= total_mismatch / 2:
        if sign_inversions >= total_mismatch / 2:
            return (
                FailureMechanism.TRANSITION_FAILURE,
                FailureSubtype.TEF_DIRECTION_INVERSION,
                f"{sign_inversions}/{total_mismatch} step deltas are sign-inverted (transition failure).",
            )
        if axis_swaps >= total_mismatch / 2:
            return (
                FailureMechanism.TRANSITION_FAILURE,
                FailureSubtype.TEF_AXIS_SWAP,
                f"{axis_swaps}/{total_mismatch} step deltas show axis swap (transition failure).",
            )
        if magnitude_errors >= total_mismatch / 2:
            return (
                FailureMechanism.TRANSITION_FAILURE,
                FailureSubtype.TEF_MAGNITUDE_ERROR,
                f"{magnitude_errors}/{total_mismatch} step deltas have wrong magnitude (transition failure).",
            )
        return (
            FailureMechanism.TRANSITION_FAILURE,
            FailureSubtype.TEF_PARITY_ERROR,
            "Transition failure detected.",
        )

    # Horizon Failure
    if dynamics.first_divergence_step == 0 and \
            dynamics.divergence_evolution == DivergenceEvolution.INCREASING:
        return (
            FailureMechanism.HORIZON_FAILURE,
            FailureSubtype.HDF_GRADUAL_DECAY,
            "Divergence begins immediately and grows with horizon (horizon failure).",
        )

    return (
        FailureMechanism.STATE_TRACKING_FAILURE,
        FailureSubtype.SRF_PERSISTENT_OFFSET,
        "State synchronization lost after early divergence.",
    )


# Public Entry Points
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
    """Full classification pipeline with binary Output Validity layer and broad Cognitive Failure taxonomy."""
    gw = int(grid_width) if grid_width else 0
    gh = int(grid_height) if grid_height else 0
    rules = transition_rules or {}
    acts = actions or []

    valid = True
    exception_code: Optional[str] = None

    raw_str = raw_output or ""
    if "[EXCEPTION]" in raw_str.upper() or "EXCEPTION:" in raw_str.upper():
        valid = False
        exception_code = E100_EXCEPTION_TOKEN
    elif not raw_str.strip() and (not predicted or len(predicted) == 0):
        valid = False
        exception_code = E101_EMPTY_OUTPUT

    pred_coords = _parse_coords(predicted)
    gt_coords = _parse_coords(ground_truth)

    if valid and pred_coords is None:
        valid = False
        if not predicted or len(predicted) == 0:
            exception_code = E101_EMPTY_OUTPUT
        else:
            exception_code = E102_PARSE_FAILURE

    if not valid:
        # Invalid response: must not enter cognitive failure taxonomy
        pc = ProbeClassification(
            valid=False,
            exception_code=exception_code or E104_UNKNOWN_INVALID,
            mechanism=FailureMechanism.NONE,
            first_failure_step=None,
            secondary_flags=(),
        )
        subtype = FailureSubtype.ORF_EMPTY_RESPONSE if exception_code == E101_EMPTY_OUTPUT else FailureSubtype.ORF_MALFORMED_STRUCTURE
        return TaxonomyResult(
            outcome=Outcome.UNCLASSIFIABLE,
            mechanism=FailureMechanism.OUTPUT_FORMAT_FAILURE,
            subtype=subtype,
            patterns=[ErrorPattern.NONE],
            dynamics=None,
            impact=None,
            legacy_mode=ErrorMode.FORMATTING_FAILURE,
            valid=False,
            error=exception_code,
            explanation=f"Invalid response encountered (error code: {exception_code}).",
            probe_classification=pc,
        )

    if gt_coords is None:
        pc = ProbeClassification(
            valid=False,
            exception_code=E102_PARSE_FAILURE,
            mechanism=FailureMechanism.NONE,
            first_failure_step=None,
            secondary_flags=(),
        )
        return TaxonomyResult(
            outcome=Outcome.UNCLASSIFIABLE,
            mechanism=FailureMechanism.OUTPUT_FORMAT_FAILURE,
            subtype=FailureSubtype.ORF_INVALID_COORDINATE,
            patterns=[ErrorPattern.NONE],
            dynamics=None,
            impact=None,
            legacy_mode=ErrorMode.FORMATTING_FAILURE,
            valid=False,
            error="Ground-truth parse failure",
            explanation="Ground-truth trajectory is in an invalid coordinate format.",
            probe_classification=pc,
        )

    if gw <= 0 and gt_coords:
        gw = _safe_gw(gt_coords)
    if gh <= 0 and gt_coords:
        gh = _safe_gh(gt_coords)

    # Empty predicted trajectory
    if not pred_coords:
        pc = ProbeClassification(
            valid=False,
            exception_code=E101_EMPTY_OUTPUT,
            mechanism=FailureMechanism.NONE,
            first_failure_step=None,
            secondary_flags=(),
        )
        return TaxonomyResult(
            outcome=Outcome.UNCLASSIFIABLE,
            mechanism=FailureMechanism.OUTPUT_FORMAT_FAILURE,
            subtype=FailureSubtype.ORF_EMPTY_RESPONSE,
            patterns=[ErrorPattern.NONE],
            dynamics=None,
            impact=None,
            legacy_mode=ErrorMode.FORMATTING_FAILURE,
            valid=False,
            error=E101_EMPTY_OUTPUT,
            explanation="Model produced no parseable coordinates (empty response).",
            probe_classification=pc,
        )

    # SUCCESS: exact match or verbose correct trajectory
    if pred_coords == gt_coords:
        raw_str = (raw_output or "").strip()
        # Check if output has extra text/explanation or markdown wrapper
        has_extra_text = len(pred_coords) > 0 and len(raw_str) > sum(len(str(c)) for c in pred_coords) + 10
        is_wrapped = "```" in raw_str
        
        if has_extra_text:
            out_comp = OutputCompliance.VERBOSE_CORRECT
            f_code = F001_VERBOSE_CORRECT
            exact_m = False
        elif is_wrapped:
            out_comp = OutputCompliance.PERFECT_FORMAT
            f_code = F002_MARKDOWN_WRAPPER
            exact_m = False
        else:
            out_comp = OutputCompliance.PERFECT_FORMAT
            f_code = ""
            exact_m = True

        pc = ProbeClassification(
            valid=True,
            exception_code=None,
            mechanism=FailureMechanism.NONE,
            first_failure_step=None,
            secondary_flags=(),
            trajectory_correct=True,
            step_accuracy=1.0,
            exact_match=exact_m,
            output_compliance=out_comp,
            formatting_subcode=f_code,
        )
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
            explanation="Predicted trajectory correctly matches ground truth (or verbose correct).",
            probe_classification=pc,
        )

    # FAILURE path
    patterns = _run_pattern_detectors(pred_coords, gt_coords, gw, gh)
    dynamics = _compute_dynamics(pred_coords, gt_coords)
    impact = _compute_impact(pred_coords, gt_coords, gw, gh, dynamics)

    mechanism, subtype, explanation = _attribute_mechanism(
        pred_coords, gt_coords, rules, acts, gw, gh, patterns, dynamics
    )

    # Check if formatting prevented extraction / no correct trajectory recovered
    n_gt = len(gt_coords)
    step_acc = sum(1 for i in range(min(len(pred_coords), n_gt)) if pred_coords[i] == gt_coords[i]) / n_gt if n_gt > 0 else 0.0
    exact_m = (pred_coords == gt_coords)

    if not pred_coords or len(pred_coords) == 0:
        out_comp = OutputCompliance.NO_TRAJECTORY
        f_code = F004_MISSING_TRAJECTORY
        mechanism = FailureMechanism.OUTPUT_FORMAT_FAILURE
    elif step_acc == 0.0 and len(pred_coords) != n_gt:
        out_comp = OutputCompliance.MALFORMED_OUTPUT
        f_code = F005_MALFORMED_TRAJECTORY
        mechanism = FailureMechanism.OUTPUT_FORMAT_FAILURE
    else:
        out_comp = OutputCompliance.VERBOSE_CORRECT if (raw_output and len(raw_output.strip()) > sum(len(str(c)) for c in pred_coords) + 10) else OutputCompliance.PERFECT_FORMAT
        f_code = F001_VERBOSE_CORRECT if out_comp == OutputCompliance.VERBOSE_CORRECT else ""

    # Map legacy mode for backwards compatibility
    legacy_map = {
        FailureMechanism.NONE: ErrorMode.NONE,
        FailureMechanism.STATE_TRACKING_FAILURE: ErrorMode.STATE_TRACKING_FAILURE,
        FailureMechanism.TRANSITION_FAILURE: ErrorMode.TRANSITION_RULE_FAILURE,
        FailureMechanism.SEMANTIC_FAILURE: ErrorMode.SEMANTIC_INTERPRETATION_FAILURE,
        FailureMechanism.OUTPUT_FORMAT_FAILURE: ErrorMode.FORMATTING_FAILURE,
        FailureMechanism.HORIZON_FAILURE: ErrorMode.HORIZON_COLLAPSE,
        FailureMechanism.UNKNOWN_FAILURE: ErrorMode.STATE_TRACKING_FAILURE,
    }
    legacy = legacy_map.get(mechanism, ErrorMode.STATE_TRACKING_FAILURE)

    secondary_flags = tuple(p.value for p in patterns if p != ErrorPattern.NONE)
    pc = ProbeClassification(
        valid=True,
        exception_code=None,
        mechanism=mechanism,
        first_failure_step=dynamics.first_divergence_step,
        secondary_flags=secondary_flags,
        trajectory_correct=False,
        step_accuracy=round(step_acc, 4),
        exact_match=exact_m,
        output_compliance=out_comp,
        formatting_subcode=f_code,
    )
    return TaxonomyResult(
        outcome=Outcome.FAILURE,
        mechanism=mechanism,
        subtype=subtype,
        patterns=patterns,
        dynamics=dynamics,
        impact=impact,
        legacy_mode=legacy,
        valid=True,
        error=None,
        explanation=explanation,
        probe_classification=pc,
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
    try:
        return classify_trajectory(
            predicted, ground_truth, transition_rules, actions, tier,
            grid_width, grid_height, raw_output,
        )
    except Exception as exc:
        logger.error(f"[TAXONOMY] classification failed: {exc}")
        pc = ProbeClassification(
            valid=False,
            exception_code=E103_PROVIDER_ERROR,
            mechanism=FailureMechanism.NONE,
            first_failure_step=None,
            secondary_flags=(),
        )
        return TaxonomyResult(
            outcome=Outcome.UNCLASSIFIABLE,
            mechanism=FailureMechanism.OUTPUT_FORMAT_FAILURE,
            subtype=FailureSubtype.ORF_MALFORMED_STRUCTURE,
            patterns=[ErrorPattern.NONE],
            dynamics=None,
            impact=None,
            legacy_mode=ErrorMode.FORMATTING_FAILURE,
            valid=False,
            error=str(exc),
            explanation=f"Classification exception: {exc}",
            probe_classification=pc,
        )


def classify_trajectory_error(
    predicted: List[str],
    ground_truth: List[str],
    transition_rules: Dict[str, Dict[str, int]],
    actions: List[str],
    tier: int,
    grid_width: int,
    grid_height: int,
):
    result = classify_trajectory(
        predicted, ground_truth, transition_rules, actions, tier,
        grid_width, grid_height,
    )
    class _Legacy:
        mode = result.legacy_mode
        confidence = 1.0
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
