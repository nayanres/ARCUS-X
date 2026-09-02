from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum
import math

@dataclass
class TaxonomyResult:
    classifications: List[Any]
    valid: bool
    error: Optional[str] = None


class ErrorMode(Enum):
    NONE = "None"
    
    # State Tracking Failure
    STATE_TRACKING_ARITHMETIC_DRIFT = "State Tracking Failure: Arithmetic Drift"
    STATE_TRACKING_PARITY_VIOLATION = "State Tracking Failure: Parity Rule Violation"
    
    # Termination Failure
    TERMINATION_HORIZON_OVERSHOOT = "Termination Failure: Horizon Overshoot"
    TERMINATION_RUNAWAY_GENERATION = "Termination Failure: Runaway Generation"
    
    # Terminal Failure
    TERMINAL_CHECKSUM_FAILURE = "Terminal Failure: Wrong Checksum"
    TERMINAL_EXTRACTION_FAILURE = "Terminal Failure: Extraction/parsing failure"
    
    # Protocol Failure
    PROTOCOL_INVALID_FORMAT = "Protocol Failure: Invalid format"
    PROTOCOL_MISSING_FIELDS = "Protocol Failure: Missing required fields"
    
    # Provider / Legacy
    PROVIDER_FAILURE = "Provider / Evaluation Failure"


@dataclass(frozen=True)
class TrajectoryStep:
    x: int
    y: int
    timestamp: int
    delta_x: float
    delta_y: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "timestamp": self.timestamp,
            "delta_x": self.delta_x,
            "delta_y": self.delta_y
        }


@dataclass(frozen=True)
class TrajectoryRecord:
    steps: List[TrajectoryStep]
    depth: int
    gravity_target: float
    rule_even: Tuple[float, float]
    rule_odd: Tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "depth": self.depth,
            "gravity_target": self.gravity_target,
            "rule_even": self.rule_even,
            "rule_odd": self.rule_odd
        }


@dataclass(frozen=True)
class ErrorModeClassification:
    mode: ErrorMode
    confidence: float = 0.0
    score: float = 0.0
    evidence_indices: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "confidence": self.confidence,
            "score": self.score,
            "evidence_indices": self.evidence_indices
        }


@dataclass(frozen=True)
class DepthErrorModeStore:
    depth: int
    mode: str
    count: int
    percentage: float


def extract_deltas(trajectory: List[TrajectoryStep]) -> List[Tuple[float, float]]:
    """
    IMPORTANT FIX:
    - DO NOT apply modulo here
    - preserve raw signal structure for classification
    """
    deltas = []
    for i in range(1, len(trajectory)):
        dx = trajectory[i].x - trajectory[i - 1].x
        dy = trajectory[i].y - trajectory[i - 1].y
        deltas.append((float(dx), float(dy)))
    return deltas


def detect_memory_failure(deltas: List[Tuple[float, float]]) -> ErrorModeClassification:
    if len(deltas) < 3:
        return ErrorModeClassification(ErrorMode.NONE, 0.0, 0.0, [])

    magnitudes = [math.sqrt(dx**2 + dy**2) for dx, dy in deltas]
    mean = sum(magnitudes) / len(magnitudes)
    variance = sum((m - mean) ** 2 for m in magnitudes) / len(magnitudes)
    std = math.sqrt(variance)

    if std < 1e-6:
        return ErrorModeClassification(ErrorMode.NONE, 0.0, 0.0, [])

    flagged = []
    max_z = 0.0

    for i, m in enumerate(magnitudes):
        z = (m - mean) / std
        if abs(z) > 3.0:
            flagged.append(i + 1)
            max_z = max(max_z, abs(z))

    if flagged:
        score = min(1.0, max_z / 6.0)
        return ErrorModeClassification(
            ErrorMode.MEMORY_FAILURE,
            confidence=score,
            score=max_z,
            evidence_indices=flagged
        )

    return ErrorModeClassification(ErrorMode.NONE, 0.0, 0.0, [])


def detect_algorithm_collapse(
    deltas: List[Tuple[float, float]],
    rule_even: Tuple[float, float],
    rule_odd: Tuple[float, float],
    threshold: int = 2
) -> ErrorModeClassification:

    if not deltas:
        return ErrorModeClassification(ErrorMode.NONE, 0.0, 0.0, [])

    consecutive = 0
    max_consecutive = 0
    flagged = []

    for i, (dx, dy) in enumerate(deltas):

        ex, ey = rule_even if i % 2 == 0 else rule_odd

        mag_a = math.sqrt(dx**2 + dy**2)
        mag_e = math.sqrt(ex**2 + ey**2)

        if mag_a == 0 or mag_e == 0:
            cos_sim = 0.0
        else:
            cos_sim = (dx * ex + dy * ey) / (mag_a * mag_e)

        if cos_sim < 0.9:
            consecutive += 1
            flagged.append(i + 1)
        else:
            consecutive = 0

        max_consecutive = max(max_consecutive, consecutive)

    if max_consecutive > threshold:
        score = max_consecutive / (threshold + 3.0)
        return ErrorModeClassification(
            ErrorMode.STATE_TRACKING_PARITY_VIOLATION,
            confidence=min(1.0, score),
            score=max_consecutive,
            evidence_indices=flagged
        )

    return ErrorModeClassification(ErrorMode.NONE, 0.0, 0.0, [])


def detect_heuristic_substitution(deltas: List[Tuple[float, float]]) -> ErrorModeClassification:
    if len(deltas) < 3:
        return ErrorModeClassification(ErrorMode.NONE, 0.0, 0.0, [])

    flagged = set()

    for i in range(len(deltas) - 2):
        w = deltas[i:i + 3]

        # variance check
        mx = sum(d[0] for d in w) / 3.0
        my = sum(d[1] for d in w) / 3.0

        var = sum((d[0] - mx) ** 2 + (d[1] - my) ** 2 for d in w) / 3.0

        if var < 1e-4 and w[0] == w[1] == w[2]:
            flagged.update([i + 1, i + 2, i + 3])

    if flagged:
        score = len(flagged) / len(deltas)
        return ErrorModeClassification(
            ErrorMode.HEURISTIC_SUBSTITUTION,
            confidence=min(1.0, score),
            score=score,
            evidence_indices=sorted(flagged)
        )

    return ErrorModeClassification(ErrorMode.NONE, 0.0, 0.0, [])


# =========================
# DETECTOR 4: ARITHMETIC SLIPS
# =========================

def detect_arithmetic_slips(
    deltas: List[Tuple[float, float]],
    rule_even: Tuple[float, float],
    rule_odd: Tuple[float, float]
) -> ErrorModeClassification:

    if not deltas:
        return ErrorModeClassification(ErrorMode.NONE, 0.0, 0.0, [])

    slips = []

    for i, (dx, dy) in enumerate(deltas):

        if dx == 0 and dy == 0:
            continue

        ex, ey = rule_even if i % 2 == 0 else rule_odd

        err_x = dx - ex
        err_y = dy - ey

        if (err_x != 0 or err_y != 0) and abs(err_x) <= 2 and abs(err_y) <= 2:
            slips.append(i + 1)

    if slips:
        score = len(slips) / len(deltas)
        return ErrorModeClassification(
            ErrorMode.STATE_TRACKING_ARITHMETIC_DRIFT,
            confidence=min(1.0, score),
            score=score,
            evidence_indices=slips
        )

    return ErrorModeClassification(ErrorMode.NONE, 0.0, [])


# =========================
# FINAL CLASSIFIER (HIERARCHICAL)
# =========================

def classify_error(
    trajectory: List[TrajectoryStep],
    rule_even: Tuple[float, float],
    rule_odd: Tuple[float, float]
) -> ErrorModeClassification:

    deltas = extract_deltas(trajectory)

    candidates = [
        detect_algorithm_collapse(deltas, rule_even, rule_odd),
        detect_heuristic_substitution(deltas),
        detect_arithmetic_slips(deltas, rule_even, rule_odd),
        detect_memory_failure(deltas)
    ]

    # pick highest confidence non-NONE
    best = max(candidates, key=lambda c: c.confidence)

    return best