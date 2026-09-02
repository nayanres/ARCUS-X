"""ARCUS-X evaluation metrics.

This module measures *trajectory compliance*, not chain-of-thought or final
answer substring matching. The primary signal is whether the model produced
the exact ground-truth path; partial credit is derived from step-level
accuracy, continuity (longest correct prefix) and the first divergence point.

All scores are on a **0.0-1.0** scale unless explicitly noted. The reporting
layer is responsible for any 0-100 presentation scaling, so there is a single
canonical range throughout the codebase (this fixes the previous 0-1 vs 0-100
threshold bugs).

Removed during the hardening pass (they did not measure ARCUS-X):
``PathTraceExtractor``, ``FOLGraphValidator``, ``PathTraceValidityMetric``,
``RiemannSumAggregator``/CRI, ``GenerationInefficiencyIndex`` (FOL GII, a
legacy token-density shorthand — distinct from the current ``token_density``
ratio and from ``GenerationBloatIndex``), ``SyllogisticLeakageScore``,
``StructuralYieldPoint`` and ``EvaluationMetricsFactory``.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trajectory compliance metrics
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryResult:
    """Per-trajectory evaluation result (all ratios on a 0.0-1.0 scale)."""

    parsed: bool
    exact_match: bool
    step_accuracy: float
    continuity_score: float
    first_divergence: Optional[int]
    horizon_match: bool
    final_state_correct: bool
    predicted_length: int
    ground_truth_length: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "parsed": self.parsed,
            "exact_match": self.exact_match,
            "step_accuracy": self.step_accuracy,
            "continuity_score": self.continuity_score,
            "first_divergence": self.first_divergence,
            "horizon_match": self.horizon_match,
            "final_state_correct": self.final_state_correct,
            "predicted_length": self.predicted_length,
            "ground_truth_length": self.ground_truth_length,
        }


def compare_trajectories(
    predicted: List[str],
    ground_truth: List[str],
) -> TrajectoryResult:
    """Compare a predicted path to the ground-truth trajectory.

    Parameters
    ----------
    predicted:
        Canonical ``[x,y]`` tokens extracted from the model output (may be
        empty if parsing failed).
    ground_truth:
        Canonical ``[x,y]`` tokens of the authoritative trajectory.

    Returns
    -------
    TrajectoryResult
        Structural comparison. ``step_accuracy`` is the fraction of
        ground-truth positions reproduced; ``continuity_score`` is the longest
        correct prefix divided by the ground-truth length; ``first_divergence``
        is the 1-based step index of the first mismatch (or ``None`` on an
        exact match).
    """
    gt_len = len(ground_truth)

    if not predicted:
        return TrajectoryResult(
            parsed=False,
            exact_match=False,
            step_accuracy=0.0,
            continuity_score=0.0,
            first_divergence=1 if gt_len > 0 else None,
            horizon_match=False,
            final_state_correct=False,
            predicted_length=0,
            ground_truth_length=gt_len,
        )

    overlap = min(len(predicted), gt_len)
    correct = 0
    first_div: Optional[int] = None
    for i in range(overlap):
        if predicted[i] == ground_truth[i]:
            correct += 1
        elif first_div is None:
            first_div = i + 1  # 1-based step index

    # If the overlapping region matched perfectly but lengths differ, the
    # divergence is the first step beyond the overlap.
    if first_div is None and len(predicted) != gt_len:
        first_div = overlap + 1

    exact = (len(predicted) == gt_len) and (first_div is None)
    step_accuracy = (correct / gt_len) if gt_len > 0 else 0.0
    # Longest correct prefix length == (first_div - 1) when a divergence exists.
    prefix_len = (first_div - 1) if first_div is not None else gt_len
    continuity = (prefix_len / gt_len) if gt_len > 0 else 0.0

    return TrajectoryResult(
        parsed=True,
        exact_match=exact,
        step_accuracy=step_accuracy,
        continuity_score=continuity,
        first_divergence=first_div,
        horizon_match=(len(predicted) == gt_len),
        final_state_correct=(predicted[-1] == ground_truth[-1]) if gt_len > 0 else False,
        predicted_length=len(predicted),
        ground_truth_length=gt_len,
    )


# ---------------------------------------------------------------------------
# Efficiency / horizon metrics (0.0-1.0 unless noted)
# ---------------------------------------------------------------------------
class HorizonCompliance:
    """Measures whether the model generated the requested number of transitions."""

    @staticmethod
    def calculate(requested_depth: int, generated_depth: int) -> float:
        """``HC = 1 - |generated - z| / max(generated, z)`` (0.0-1.0)."""
        if requested_depth <= 0 or generated_depth <= 0:
            return 0.0
        hc = 1.0 - abs(generated_depth - requested_depth) / max(generated_depth, requested_depth)
        return float(max(0.0, hc))


class GenerationBloatIndex:
    """Quantifies excessive output generation relative to expected output size."""

    @staticmethod
    def calculate(actual_tokens: int, expected_tokens: int) -> float:
        """``GBI = max(0, actual - expected) / expected`` (>= 0.0)."""
        if expected_tokens <= 0:
            return 0.0
        gbi = max(0, actual_tokens - expected_tokens) / expected_tokens
        return float(gbi)


class GenerationEfficiency:
    """Quantifies generation efficiency from the bloat index."""

    @staticmethod
    def calculate(gbi: float) -> float:
        """``GE = 1 / (1 + GBI)`` (0.0-1.0)."""
        ge = 1.0 / (1.0 + gbi)
        return float(ge)


def _token_density(total_tokens: int, depth: int) -> float:
    """Raw token-density ratio (tokens per horizon step), 0.0 when depth <= 0."""
    if depth <= 0:
        return 0.0
    return float(total_tokens) / float(depth)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def mean(values: List[float]) -> float:
    """Arithmetic mean, returning 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0


def aggregate_trajectory_results(results: List[TrajectoryResult]) -> Dict[str, float]:
    """Aggregate a list of per-trajectory results into summary statistics.

    Metric reporting hierarchy (per reviewer guidance):
    - PRIMARY metrics (directly measure trajectory compliance):
        * exact_match_rate      — fraction of trajectories exactly matching GT
        * mean_step_accuracy    — fraction of GT positions reproduced
        * mean_continuity       — longest correct prefix / GT length
    - SECONDARY metrics (diagnostic / efficiency context):
        * mean_final_state      — final coordinate correctness (necessary but
                                   not sufficient; a wrong path can end correctly)
        * horizon_compliance_rate — did the model emit the requested depth?
        * parse_failure_rate    — fraction of outputs that failed path parsing

    Accepts either ``TrajectoryResult`` objects or plain dicts carrying the
    same field names. This tolerance keeps the function usable from the
    benchmark aggregation path, which builds lightweight dicts from the
    results matrix without constructing full ``TrajectoryResult`` instances.
    """
    if not results:
        return {
            "exact_match_rate": 0.0,
            "mean_step_accuracy": 0.0,
            "mean_continuity": 0.0,
            "mean_final_state": 0.0,
            "horizon_compliance_rate": 0.0,
            "parse_failure_rate": 0.0,
        }

    def _field(r, attr, default=0.0):
        if isinstance(r, dict):
            return r.get(attr, default)
        return getattr(r, attr, default)

    return {
        # PRIMARY
        "exact_match_rate": mean([1.0 if _field(r, "exact_match") else 0.0 for r in results]),
        "mean_step_accuracy": mean([_field(r, "step_accuracy") for r in results]),
        "mean_continuity": mean([_field(r, "continuity_score") for r in results]),
        # SECONDARY
        "mean_final_state": mean([1.0 if _field(r, "final_state_correct") else 0.0 for r in results]),
        "horizon_compliance_rate": mean([1.0 if _field(r, "horizon_match") else 0.0 for r in results]),
        "parse_failure_rate": mean([0.0 if _field(r, "parsed") else 1.0 for r in results]),
    }


# Metric classification for reporting / reviewer transparency
PRIMARY_METRICS = ("exact_match_rate", "mean_step_accuracy", "mean_continuity")
SECONDARY_METRICS = ("mean_final_state", "horizon_compliance_rate", "parse_failure_rate")


def metric_tier(metric_name: str) -> str:
    """Return 'primary' or 'secondary' for a given aggregated metric name."""
    if metric_name in PRIMARY_METRICS:
        return "primary"
    if metric_name in SECONDARY_METRICS:
        return "secondary"
    return "uncategorized"


# ---------------------------------------------------------------------------
# ARCUS Robustness Index (CRI) -- SECONDARY aggregate summary statistic
# ---------------------------------------------------------------------------
@dataclass
class CRISummary:
    """Result of the ARCUS Robustness Index computation.

    CRI is a *secondary* summary statistic. It is NOT a primary benchmark
    metric and must never replace the trajectory metrics (step accuracy,
    exact match, continuity, fracture depth). It summarises correctness
    retention across horizons, semantic perturbations, and efficiency.
    """

    cri: float
    trajectory_fidelity: float
    horizon_robustness: float
    semantic_robustness: float
    generation_efficiency: float
    components: Dict[str, float]

    def as_dict(self) -> Dict[str, object]:
        return {
            "cri": self.cri,
            "trajectory_fidelity": self.trajectory_fidelity,
            "horizon_robustness": self.horizon_robustness,
            "semantic_robustness": self.semantic_robustness,
            "generation_efficiency": self.generation_efficiency,
            "components": self.components,
        }


class ARCUSRobustnessIndex:
    """Secondary aggregate robustness indicator for ARCUS-X.

    CRI = 0.45 * TrajectoryFidelity
        + 0.30 * HorizonRobustness
        + 0.20 * SemanticRobustness
        + 0.05 * GenerationEfficiency

    All component scores are on a 0.0-1.0 scale, so CRI is also on 0.0-1.0.
    This is a *summary* statistic only; the primary evaluation signals remain
    the trajectory metrics (step accuracy, exact match, continuity, fracture
    depth).
    """

    WEIGHTS = {
        "trajectory_fidelity": 0.45,
        "horizon_robustness": 0.30,
        "semantic_robustness": 0.20,
        "generation_efficiency": 0.05,
    }

    # --- Trajectory Fidelity (TF) -----------------------------------------
    @staticmethod
    def trajectory_fidelity(
        step_accuracy: float,
        continuity_score: float,
        exact_match: float,
    ) -> float:
        """Discrete trajectory correctness.

        TF = 0.5 * step_accuracy + 0.3 * continuity_score + 0.2 * exact_match
        """
        return 0.5 * step_accuracy + 0.3 * continuity_score + 0.2 * exact_match

    # --- Horizon Robustness (HR) ------------------------------------------
    @staticmethod
    def horizon_robustness(
        horizon_points: List[Tuple[float, Optional[float]]],
    ) -> float:
        """Normalised trapezoidal area over sampled depths (discrete).

        HR = Σ ((A_i + A_(i+1)) / 2) * (z_(i+1) - z_i) / (z_max - z_min)

        where A_i is the trajectory accuracy at sampled horizon z_i.

        Handles:
        * irregular depth sampling (trapezoid widths use the actual z gaps),
        * missing points (entries with ``None`` accuracy are skipped),
        * a single sampled depth (returns the mean available accuracy).

        ``horizon_points`` is an iterable of ``(z, accuracy)`` tuples; the
        accuracy may be ``None`` to denote a missing sample.
        """
        valid = [
            (float(z), float(a))
            for z, a in horizon_points
            if a is not None
        ]
        if not valid:
            return 0.0
        valid.sort(key=lambda t: t[0])

        z_min = valid[0][0]
        z_max = valid[-1][0]
        if z_max == z_min:
            # Single sampled depth: HR is the mean accuracy at that depth.
            return mean([a for _, a in valid])

        area = 0.0
        for i in range(len(valid) - 1):
            z0, a0 = valid[i]
            z1, a1 = valid[i + 1]
            dz = z1 - z0
            if dz <= 0:
                # Skip non-increasing / duplicate depths.
                continue
            area += (a0 + a1) / 2.0 * dz
        return area / (z_max - z_min)

    # --- Semantic Robustness (SR) -----------------------------------------
    @staticmethod
    def semantic_robustness(
        tier0_acc: Optional[float],
        tier1_acc: Optional[float],
        tier2_acc: Optional[float],
        tier3_acc: Optional[float],
    ) -> float:
        """Preservation of performance under tier perturbation (Tier 0 baseline).

        Tier 0 is the baseline. Tiers are NOT treated as equally difficult:

        * Tier 1 is a *label-only* perturbation (the trajectory is unchanged),
          so it is expected to *preserve* Tier-0 performance. It is weighted as
          a preservation signal (0.5).
        * Tiers 2/3 change the transition semantics, so they are expected to
          *degrade* relative to Tier 0. They are weighted as degradation
          signals (0.25 each).

        Each tier contributes its performance ratio relative to Tier 0,
        clamped to [0, 1]. Missing tiers are omitted from the weighted average.
        """
        if tier0_acc is None or tier0_acc <= 0:
            # Without a positive baseline we cannot measure preservation; fall
            # back to the mean of any available tier accuracies.
            vals = [v for v in (tier1_acc, tier2_acc, tier3_acc) if v is not None]
            return mean(vals) if vals else 0.0

        def ratio(t: Optional[float]) -> Optional[float]:
            if t is None:
                return None
            return max(0.0, min(1.0, float(t) / float(tier0_acc)))

        parts: List[float] = []
        weights: List[float] = []
        r1 = ratio(tier1_acc)
        if r1 is not None:
            parts.append(r1)
            weights.append(0.5)
        r2 = ratio(tier2_acc)
        if r2 is not None:
            parts.append(r2)
            weights.append(0.25)
        r3 = ratio(tier3_acc)
        if r3 is not None:
            parts.append(r3)
            weights.append(0.25)

        if not parts:
            return 0.0
        wsum = sum(weights)
        return sum(p * w for p, w in zip(parts, weights)) / wsum

    # --- Composite --------------------------------------------------------
    @classmethod
    def compute(
        cls,
        *,
        step_accuracy: float,
        continuity_score: float,
        exact_match: float,
        horizon_points: List[Tuple[float, Optional[float]]],
        tier0_acc: Optional[float] = None,
        tier1_acc: Optional[float] = None,
        tier2_acc: Optional[float] = None,
        tier3_acc: Optional[float] = None,
        generation_efficiency: Optional[float] = None,
    ) -> CRISummary:
        """Compute the full CRI summary.

        ``generation_efficiency`` is the canonical ``GenerationEfficiency``
        value (already on 0.0-1.0). If missing, it defaults to 0.0 so the
        remaining primary components still drive the score.
        """
        tf = cls.trajectory_fidelity(step_accuracy, continuity_score, exact_match)
        hr = cls.horizon_robustness(horizon_points)
        sr = cls.semantic_robustness(tier0_acc, tier1_acc, tier2_acc, tier3_acc)
        ge = generation_efficiency if generation_efficiency is not None else 0.0
        ge = max(0.0, min(1.0, float(ge)))

        cri = (
            cls.WEIGHTS["trajectory_fidelity"] * tf
            + cls.WEIGHTS["horizon_robustness"] * hr
            + cls.WEIGHTS["semantic_robustness"] * sr
            + cls.WEIGHTS["generation_efficiency"] * ge
        )
        cri = max(0.0, min(1.0, cri))

        return CRISummary(
            cri=cri,
            trajectory_fidelity=tf,
            horizon_robustness=hr,
            semantic_robustness=sr,
            generation_efficiency=ge,
            components={
                "trajectory_fidelity": tf,
                "horizon_robustness": hr,
                "semantic_robustness": sr,
                "generation_efficiency": ge,
            },
        )


# ---------------------------------------------------------------------------
# Latency-normalized efficiency diagnostics (SYSTEMS-LEVEL ONLY)
# ---------------------------------------------------------------------------
# IMPORTANT: latency is a *systems-level diagnostic*. It is affected by provider
# infrastructure, architecture, batching, and deployment conditions, and must
# NEVER be treated as a model capability score. These metrics measure how much
# *verified benchmark performance* is achieved per unit time; they do NOT feed
# into CRI, GBI, trajectory fidelity, or any fracture metric, and they must not
# be used to classify models as "fast" or "slow" or to create latency tiers.
class LatencyDiagnostics:
    """Latency-normalized efficiency diagnostics for ARCUS-X.

    These are *secondary / systems-level* diagnostics only. They complement
    accuracy, trajectory fidelity, generation efficiency, and the failure
    taxonomy; they never replace or modify them.

    Definitions
    -----------
    ALE (Accuracy Latency Efficiency):
        ALE = Accuracy / Latency_seconds
        where Accuracy is the probe/step accuracy (0.0-1.0) and Latency_seconds
        is the total API response latency in seconds. Higher ALE means more
        verified benchmark accuracy achieved per second of model response time.

    Correct Steps Per Second:
        Correct Steps/sec = Correct Transition Count / Latency_seconds
        where Correct Transition Count is the number of correctly predicted
        state transitions in the probe. This measures verified trajectory
        throughput.
    """

    @staticmethod
    def accuracy_latency_efficiency(accuracy: float, latency_seconds: float) -> float:
        """ALE = Accuracy / Latency_seconds (0.0 when latency <= 0)."""
        if latency_seconds is None or latency_seconds <= 0:
            return 0.0
        return float(accuracy) / float(latency_seconds)

    @staticmethod
    def correct_steps_per_second(correct_transition_count: int, latency_seconds: float) -> float:
        """Correct Steps/sec = Correct Transition Count / Latency_seconds."""
        if latency_seconds is None or latency_seconds <= 0:
            return 0.0
        return float(correct_transition_count) / float(latency_seconds)

    @staticmethod
    def _percentile(values: List[float], pct: float) -> float:
        """Linear-interpolation percentile (pct in [0, 100]); 0.0 if empty."""
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return float(ordered[0])
        k = (len(ordered) - 1) * (pct / 100.0)
        lo = int(math.floor(k))
        hi = int(math.ceil(k))
        if lo == hi:
            return float(ordered[lo])
        frac = k - lo
        return ordered[lo] * (1.0 - frac) + ordered[hi] * frac

    @classmethod
    def aggregate(cls, probes: List[Dict[str, object]]) -> Dict[str, object]:
        """Aggregate per-probe latency diagnostics into a report block.

        Parameters
        ----------
        probes:
            A list of per-probe dicts. Each dict may carry any of:
              * ``latency_ms`` (float, milliseconds)
              * ``output_tokens`` / ``completion_tokens`` (int)
              * ``accuracy`` (float, 0.0-1.0) — probe/step accuracy
              * ``correct_transitions`` (int) — correctly predicted transitions
              * ``horizon`` / ``z`` (int) — depth horizon for per-horizon split

            Missing fields are tolerated (the corresponding diagnostic is
            omitted or zeroed) so partial runs and stub evaluators still
            produce a stable schema.

        Returns
        -------
        dict with the aggregate latency report (see the spec):
          mean_latency_ms, median_latency_ms, p95_latency_ms, max_latency_ms,
          mean_output_tokens_per_sec, mean_correct_steps_per_sec,
          ale (aggregate), correct_steps_per_sec (aggregate),
          per_horizon (list of {horizon, mean_latency_ms, correct_steps_per_sec}).
        """
        latency_list: List[float] = []
        output_tps_list: List[float] = []
        correct_sps_list: List[float] = []
        ale_list: List[float] = []
        per_horizon: Dict[Any, Dict[str, List[float]]] = {}

        for p in probes:
            if not isinstance(p, dict):
                continue
            lat = p.get("latency_ms")
            if lat is None:
                lat = p.get("latency")
            if lat is None:
                continue
            lat = float(lat)
            latency_list.append(lat)
            latency_seconds = lat / 1000.0

            # Output tokens / sec.
            ot = p.get("output_tokens")
            if ot is None:
                ot = p.get("completion_tokens")
            if ot is not None and latency_seconds > 0:
                output_tps_list.append(float(ot) / latency_seconds)

            # Correct steps / sec.
            ct = p.get("correct_transitions")
            if ct is not None and latency_seconds > 0:
                cps = cls.correct_steps_per_second(int(ct), latency_seconds)
                correct_sps_list.append(cps)

            # ALE.
            acc = p.get("accuracy")
            if acc is not None and latency_seconds > 0:
                ale_list.append(cls.accuracy_latency_efficiency(float(acc), latency_seconds))

            # Per-horizon grouping.
            hz = p.get("horizon")
            if hz is None:
                hz = p.get("z")
            if hz is not None and latency_seconds > 0:
                bucket = per_horizon.setdefault(str(hz), {"lat": [], "cps": []})
                bucket["lat"].append(lat)
                if ct is not None:
                    bucket["cps"].append(cls.correct_steps_per_second(int(ct), latency_seconds))

        def _mean(v):
            return round(sum(v) / len(v), 4) if v else 0.0

        per_horizon_out = []
        for hz in sorted(per_horizon.keys(), key=lambda k: float(k)):
            bucket = per_horizon[hz]
            per_horizon_out.append({
                "horizon": hz,
                "mean_latency_ms": _mean(bucket["lat"]),
                "correct_steps_per_sec": _mean(bucket["cps"]),
            })

        return {
            "mean_latency_ms": _mean(latency_list),
            "median_latency_ms": round(cls._percentile(latency_list, 50.0), 4),
            "p95_latency_ms": round(cls._percentile(latency_list, 95.0), 4),
            "max_latency_ms": round(max(latency_list), 4) if latency_list else 0.0,
            "mean_output_tokens_per_sec": _mean(output_tps_list),
            "mean_correct_steps_per_sec": _mean(correct_sps_list),
            "ale": _mean(ale_list),
            "correct_steps_per_sec": _mean(correct_sps_list),
            "per_horizon": per_horizon_out,
            # Explicitly document that latency is NOT a capability score.
            "diagnostic_note": (
                "Latency is a systems-level diagnostic (provider infrastructure, "
                "architecture, batching, deployment). It is NOT a model capability "
                "score and is excluded from CRI/GBI/trajectory/fracture metrics."
            ),
        }
