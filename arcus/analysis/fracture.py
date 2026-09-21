"""Fracture analysis: locate the horizon depth where trajectory compliance collapses.

CANONICAL DEFINITION
--------------------
Fracture depth is the *smallest sampled* horizon ``z`` at which the primary
trajectory metric -- **step accuracy** (0.0-1.0) -- falls strictly below the
fracture floor (``accuracy_threshold``, default 0.5) with no recovery within
the next three sampled points. The look-ahead window filters transient local
dips without claiming permanent behavior between sampled horizons.

This is the single, authoritative fracture definition used everywhere in
ARCUS-X. The runner's bisection search locates the same point via binary
search; :meth:`FracturePointFinder.compute_structural_yield` locates it by a
linear scan over the accuracy curve and is the canonical reference function.

This module is intentionally free of FOL/ILP assumptions. It consumes the
per-result ``step_accuracy`` (0-1) and ``error_mode`` fields produced by the
runner, so the fracture depth is derived from the real trajectory metrics and
the real error taxonomy.
"""

from __future__ import annotations

import logging
import math
import sys
import threading
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FracturePointFinder:
    """Locates the structural fracture depth along the horizon axis."""

    def __init__(
        self,
        accuracy_threshold: float = 0.5,
        consecutive_threshold: int = 3,
        look_ahead_step: int = 3,
    ):
        """
        Parameters
        ----------
        accuracy_threshold:
            Fracture floor on the 0.0-1.0 scale (NOT 0-100). Below this the
            model is considered fractured.
        consecutive_threshold:
            Number of consecutive depths required to confirm a stable boundary.
        look_ahead_step:
            Forward depth offset used to verify a drop is a true fracture.
        """
        self.accuracy_threshold = float(accuracy_threshold)
        self.consecutive_threshold = int(consecutive_threshold)
        self.look_ahead_step = int(look_ahead_step)
        self.depth_error_distribution: Dict[int, Dict[str, float]] = {}
        logger.info(
            f"FracturePointFinder loaded. Yield floor = {self.accuracy_threshold:.3f} "
            f"(0-1 scale)."
        )

    # Progress bar (terminal only, main thread only)
    def print_search_envelope_bar(self, low: int, mid: int, high: int, max_depth: int = 200) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        bar_length = 40
        low_idx = min(int((low / max_depth) * bar_length), bar_length - 1)
        mid_idx = min(int((mid / max_depth) * bar_length), bar_length - 1)
        high_idx = min(int((high / max_depth) * bar_length), bar_length - 1)
        bar = ["-"] * bar_length
        for i in range(low_idx, min(high_idx + 1, bar_length)):
            bar[i] = "="
        bar[low_idx] = "L"
        bar[mid_idx] = "M"
        bar[high_idx] = "H"
        line = f"\r Matrix Tracking: [{''.join(bar)}] (Testing z={mid:<3} | {low}-{high})"
        sys.stdout.write(line.ljust(85))
        sys.stdout.flush()

    # Core fracture detection
    def compute_structural_yield(
        self, accuracy_curve: List[Tuple[int, float]]
    ) -> Optional[int]:
        """Return the first sampled depth with an unrecovered below-floor drop.

        The next ``look_ahead_step`` sampled points are checked for recovery.
        Returns ``None`` if no fracture is found.
        """
        if not accuracy_curve:
            return None

        for idx, (depth, accuracy) in enumerate(accuracy_curve):
            if accuracy < self.accuracy_threshold:
                window = accuracy_curve[idx + 1: idx + 1 + self.look_ahead_step]
                recovered = next(
                    (p for p in window if p[1] >= self.accuracy_threshold), None
                )
                if recovered:
                    logger.info(
                        f"[LOOK-AHEAD RECOVERY] Local dip at z={depth} "
                        f"({accuracy:.3f}) bypassed; recovered to {recovered[1]:.3f} "
                        f"at z={recovered[0]}."
                    )
                    continue
                logger.info(
                    f"[FRACTURE DETECTED] Hard breach at z={depth} "
                    f"(acc {accuracy:.3f} < {self.accuracy_threshold:.3f})."
                )
                return depth
        return None

    def compute_fracture_depth(
        self,
        accuracy_curve: List[Tuple[int, float]],
        taxonomy_valid: bool = True,
    ) -> Optional[int]:
        """Fracture depth, gated on taxonomy validity."""
        if not taxonomy_valid:
            logger.error("[TAXONOMY] invalid -> fracture depth not computed")
            return None
        return self.compute_structural_yield(accuracy_curve)

    def compute_structural_yield_robust(
        self, accuracy_curve: List[Tuple[int, float]]
    ) -> Optional[int]:
        """Fracture depth tolerant of sparse, non-monotonic accuracy curves.

        Unlike :meth:`compute_structural_yield` (which treats any single
        look-ahead point above the threshold as a recovery), this variant
        requires the drop to be *sustained*: once accuracy falls below the
        fracture floor, the majority of the remaining sampled depths must also
        be below the floor. A single anomalous high point (e.g. a sparse
        outlier at a larger horizon) does not cancel the fracture.

        Returns the smallest depth ``z`` at which accuracy drops below the
        threshold and stays predominantly below it, or ``None`` if no such
        point exists.
        """
        if not accuracy_curve:
            return None

        sorted_curve = sorted(accuracy_curve, key=lambda p: p[0])
        n = len(sorted_curve)
        for idx, (depth, accuracy) in enumerate(sorted_curve):
            if accuracy < self.accuracy_threshold:
                # Examine the remaining points after this drop.
                remaining = sorted_curve[idx + 1:]
                if not remaining:
                    return depth
                below = sum(1 for _, a in remaining if a < self.accuracy_threshold)
                # Sustained fracture: majority of subsequent depths stay below.
                if below >= (len(remaining) - below):
                    return depth
        return None

    # Telemetry aggregation
    def parse_matrix_telemetry(
        self,
        results_matrix: Dict,
        gravity_target: float,
    ) -> Dict:
        """Aggregate a results matrix for one gravity level.

        Expects each result dict to contain ``step_accuracy`` (0-1, the
        primary trajectory metric) and ``error_mode`` (string category). Builds
        the accuracy curve from step accuracy, computes the canonical fracture
        depth, and tallies the error taxonomy per depth.
        """
        depth_acc: Dict[int, List[float]] = {}
        for key, result in results_matrix.items():
            if isinstance(key, tuple) and len(key) == 4:
                z, g, tier, probe_idx = key
            elif isinstance(result, dict) and "depth" in result:
                z = result.get("depth")
                g = result.get("gravity", result.get("gravity_target"))
            else:
                continue
            if g is None or not math.isclose(float(g), float(gravity_target), abs_tol=1e-5):
                continue
            if z is None:
                continue
            z = int(z)
            acc = result.get("step_accuracy", 0.0)
            depth_acc.setdefault(z, []).append(float(acc))

        accuracy_curve: List[Tuple[int, float]] = []
        for z, accs in depth_acc.items():
            accuracy_curve.append((z, sum(accs) / len(accs)))
        accuracy_curve.sort(key=lambda x: x[0])

        # Taxonomy distribution per depth.
        self.depth_error_distribution = {}
        depth_mode_counts: Dict[int, Dict[str, int]] = {}
        for key, result in results_matrix.items():
            if not isinstance(result, dict):
                continue
            # Derive (z, g) the same way as the accuracy-curve loop above: prefer
            # the 4-tuple key, fall back to fields on the result dict. This keeps
            # the taxonomy tally consistent with the accuracy curve regardless of
            # which matrix encoding the caller uses.
            if isinstance(key, tuple) and len(key) == 4:
                z, g, _tier, _probe_idx = key
            elif isinstance(result, dict) and "depth" in result:
                z = result.get("depth")
                g = result.get("gravity", result.get("gravity_target"))
            else:
                continue
            if g is None or not math.isclose(float(g), float(gravity_target), abs_tol=1e-5):
                continue
            if z is None:
                continue
            z = int(z)
            mode = result.get("error_mode")
            if not mode:
                continue
            d = depth_mode_counts.setdefault(z, {})
            d[mode] = d.get(mode, 0) + 1

        for z, counts in depth_mode_counts.items():
            total = sum(counts.values())
            dist = {m: (c / total) * 100.0 for m, c in counts.items()}
            self.depth_error_distribution[z] = dist

        yield_point = self.compute_structural_yield(accuracy_curve)
        is_invariant = self.check_asymptotic_invariance(accuracy_curve)

        return {
            "accuracy_curve": accuracy_curve,
            "fracture_depth": yield_point if yield_point is not None else 0,
            "D_horizon": yield_point or (accuracy_curve[-1][0] if accuracy_curve else 1),
            "is_asymptotically_invariant": is_invariant,
            "compliance_status": "COMPLIANT" if yield_point is None else "FRACTURED",
            "depth_error_distribution": self.depth_error_distribution,
        }

    def check_asymptotic_invariance(
        self, accuracy_curve: List[Tuple[int, float]]
    ) -> bool:
        """True if the trailing window is flat (converged) below threshold."""
        if len(accuracy_curve) < self.consecutive_threshold:
            return False
        recent = [p[1] for p in accuracy_curve[-self.consecutive_threshold:]]
        is_flat = all(math.isclose(a, recent[0], abs_tol=1e-4) for a in recent)
        if is_flat and recent[0] < self.accuracy_threshold:
            logger.warning(
                f"[ASYMPTOTIC INVARIANCE] Flatline at {recent[0]:.3f} accuracy."
            )
            return True
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    finder = FracturePointFinder(accuracy_threshold=0.5, look_ahead_step=2)
    finder.print_search_envelope_bar(low=1, mid=100, high=200)
    import time
    time.sleep(0.3)
    finder.print_search_envelope_bar(low=1, mid=50, high=100)
    time.sleep(0.2)
    print()

    mock_matrix = {
        (10, 1.2, 1, 0): {"continuity_score": 0.95, "error_mode": "None"},
        (20, 1.2, 1, 0): {"continuity_score": 0.80, "error_mode": "None"},
        (30, 1.2, 1, 0): {"continuity_score": 0.53, "error_mode": "State Tracking Failure"},
        (40, 1.2, 1, 0): {"continuity_score": 0.70, "error_mode": "State Tracking Failure"},
        (50, 1.2, 1, 0): {"continuity_score": 0.30, "error_mode": "Transition Rule Failure"},
        (60, 1.2, 1, 0): {"continuity_score": 0.10, "error_mode": "Transition Rule Failure"},
    }
    analysis = finder.parse_matrix_telemetry(mock_matrix, gravity_target=1.2)
    print(f"Accuracy curve: {analysis['accuracy_curve']}")
    print(f"Fracture depth: z={analysis['fracture_depth']}")
    print(f"Status: {analysis['compliance_status']}")
