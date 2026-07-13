#!/usr/bin/env python3
"""Fracture analysis utilities."""

import logging
import sys
import threading
import math
import re
from typing import Dict, List, Tuple, Optional, Any

import error_taxonomy
from error_taxonomy import TrajectoryStep, TrajectoryRecord, ErrorModeClassification, DepthErrorModeStore, TaxonomyResult

logger = logging.getLogger(__name__)


class FracturePointFinder:
    """
    Analyzes evaluation result arrays to locate mathematical fracture boundaries.
    
    Programmatic Tasks:
    1. Locates the Structural Yield Point where a model drops below the accuracy floor.
    2. Identifies Asymptotic Invariance vectors to short-circuit redundant evaluation.
    3. Traverses non-monotonic delta anomalies using look-ahead step confirmations.
    4. Computes Cumulative Cognitive Volatility scores across the horizontal tracking spectrum.
    5. Renders a terminal-dynamic envelope tracking map during search runtime.
    """
    
    def __init__(self, accuracy_threshold: float = 50.0, consecutive_threshold: int = 3, look_ahead_step: int = 3):
        """
        Initialize finder with scaled whole-number thresholds.
        
        Args:
            accuracy_threshold: Floor value below which a model is fractured (0-100 scale).
            consecutive_threshold: Continuous step sequences required to confirm stability boundaries.
            look_ahead_step: The forward depth offset (+z) used to verify if a drop is a true fracture.
        """
        self.accuracy_threshold = accuracy_threshold
        self.consecutive_threshold = consecutive_threshold
        self.look_ahead_step = look_ahead_step
        
        self.volatility_scores: Dict[float, float] = {}
        
        self.error_mode_history: Dict[int, List[ErrorModeClassification]] = {}
        self.depth_error_stats: List[DepthErrorModeStore] = []
        self.depth_error_distribution: Dict[int, Dict[str, float]] = {}

        logger.info(f"v2.4 FracturePointFinder loaded. Yield Floor locked at {self.accuracy_threshold}% Accuracy.")

    def print_search_envelope_bar(self, low: int, mid: int, high: int, max_depth: int = 200):
        """
        Renders an interactive, text-based progress bar showing the bisection search boundaries.
        Format: [----L====M====H-------]
        Optimized with trailing character space-clearing pads to eliminate terminal truncation artifacts.
        """
        if threading.current_thread() is not threading.main_thread():
            return
            
        bar_length = 40
        
        low_idx = min(int((low / max_depth) * bar_length), bar_length - 1)
        mid_idx = min(int((mid / max_depth) * bar_length), bar_length - 1)
        high_idx = min(int((high / max_depth) * bar_length), bar_length - 1)
        
        bar = ["-"] * bar_length
        for i in range(low_idx, min(high_idx + 1, bar_length)):
            bar[i] = "="
            
        # Stamp search markers
        bar[low_idx] = "L"
        bar[mid_idx] = "M"
        bar[high_idx] = "H"
        
        visual_track = "".join(bar)
        
        # Construct base raw tracking string
        output_line = f"\r Matrix Tracking: [{visual_track}] (Testing z={mid:<3} | Active Domain: {low}-{high})"
        
        # Use fixed-width space padding to cleanly wipe trace history and avoid line wrapping issues
        sys.stdout.write(output_line.ljust(85))
        sys.stdout.flush()

    def compute_structural_yield(self, accuracy_curve: List[Tuple[int, float]]) -> Optional[int]:
        """
        Scans a sorted sequence along the depth axis z to flag the onset of matrix fractures.
        Applying a Look-Ahead Buffer verification rule (Option A) to bypass anomalous local minimum traps.
        """
        if not accuracy_curve:
            return None
            
        for idx, (depth, accuracy) in enumerate(accuracy_curve):
            if accuracy < self.accuracy_threshold:
                # OPTION A: Look ahead in the curve to check if the drop is persistent
                look_ahead_window = accuracy_curve[idx + 1 : idx + 1 + self.look_ahead_step]
                
                if look_ahead_window:
                    # If any subsequent depth within our look-ahead window recovers, bypass the halt
                    recovered_node = next((p for p in look_ahead_window if p[1] >= self.accuracy_threshold), None)
                    if recovered_node:
                        # Clear line buffer room for logging safely without messing up progress bar
                        print()
                        logger.info(
                            f"  [LOOK-AHEAD RECOVERY DETECTED] Local dip at z={depth} ({accuracy:.1f}%) bypassed. "
                            f"Model recovered to {recovered_node[1]:.1f}% accuracy at deeper depth z={recovered_node[0]}."
                        )
                        continue
                
                # If there are no future nodes or all future nodes fail, confirm the hard fracture
                print()
                logger.info(f"  [MATRIX FRACTURE DETECTED] Hard breach verified at depth z={depth} (Acc: {accuracy:.1f}% < {self.accuracy_threshold}%)")
                return depth
                
        return None

    def compute_fracture_depth(self, accuracy_curve: List[Tuple[int, float]], taxonomy_result: TaxonomyResult) -> Optional[int]:
        """
        Hard-gated fracture depth computation.
        Fracture depth can only be computed from successfully validated taxonomy outputs.
        """
        if not taxonomy_result.valid:
            logger.error("[TAXONOMY STATUS] invalid → fracture depth not computed")
            return None
            
        # Existing logic to compute fracture depth, now gated
        return self.compute_structural_yield(accuracy_curve)

    def check_asymptotic_invariance(self, accuracy_curve: List[Tuple[int, float]], raw_matrix: Dict) -> bool:
        """
        Evaluates long-range slope behaviors to confirm performance convergence limits.
        Protects optimization routines against flatlining loops.
        """
        if len(accuracy_curve) < self.consecutive_threshold:
            return False
            
        # Extract the trailing window across the current search envelope boundary
        recent_window = accuracy_curve[-self.consecutive_threshold:]
        accuracies = [point[1] for point in recent_window]
        
        # Invariance condition met if variance across trailing midpoints drops towards zero
        is_flat = all(math.isclose(acc, accuracies[0], abs_tol=1e-4) for acc in accuracies)
        
        if is_flat and accuracies[0] < self.accuracy_threshold:
            print()
            logger.warning(f"  [ASYMPTOTIC INVARIANCE COHERENCE] Stable flatline verified at {accuracies[0]:.1f}% accuracy.")
            return True
            
        return False

    def parse_matrix_telemetry(self, results_matrix: Dict, gravity_target: float) -> Dict:
        """
        Parses multi-dimensional benchmark matrices to isolate gravity-specific curves.
        Robustly extracts data parameters via regex to support clean values (10_1.2_1_0) 
        as well as label-injected string dictionary structures (z_10_gravity_1.2_tier_1_probe_0).
        """
        depth_data = {}
        taxonomy_results = []
        
        for key, result in results_matrix.items():
            if isinstance(key, str):
                # Hardened regex handles raw underscore groups or descriptive label prefixes
                match = re.search(r'(?:z_)?(\d+)_(?:gravity_)?([\d.]+)+_(?:tier_)?(\d+)_(?:probe_)?(\d+)', key, re.IGNORECASE)
                if match:
                    try:
                        depth = int(match.group(1))
                        gravity = float(match.group(2))
                        tier = int(match.group(3))
                        probe_idx = int(match.group(4))
                    except ValueError:
                        continue
                else:
                    # Alternative absolute fallback for raw segment components
                    parts = key.split('_')
                    if len(parts) == 4:
                        try:
                            depth = int(parts[0])
                            gravity = float(parts[1])
                            tier = int(parts[2])
                            probe_idx = int(parts[3])
                        except ValueError:
                            continue
                    else:
                        continue
                        
            # Fallback for raw native in-memory 4-tuples
            elif isinstance(key, tuple) and len(key) == 4:
                depth, gravity, tier, probe_idx = key
            else:
                continue
                
            # Filter matches inside the target delta window
            if not math.isclose(gravity, gravity_target, abs_tol=1e-5):
                continue
                
            if depth not in depth_data:
                depth_data[depth] = []
                
            # Safely capture the mapped accuracy value
            if isinstance(result, dict):
                depth_data[depth].append(result.get("terminal_accuracy", 0.0))
                
                # Capture Trajectory
                raw_output = result.get("raw_output", "")
                coords = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', raw_output)
                if coords:
                    coords = [(int(x), int(y)) for x, y in coords]
                    steps = []
                    for idx, (x, y) in enumerate(coords):
                        if idx == 0:
                            dx, dy = 0.0, 0.0
                        else:
                            dx = float(x - coords[idx-1][0])
                            dy = float(y - coords[idx-1][1])
                        steps.append(TrajectoryStep(x=x, y=y, timestamp=idx, delta_x=dx, delta_y=dy))
                    
                    taxonomy_results.append(self.analyze_error_modes(depth, steps, grid_dim=max(5, depth + 5)))

            elif isinstance(result, (int, float)):
                depth_data[depth].append(float(result))

        # Build sorted matrix arrays
        accuracy_curve = []
        for depth, accuracies in depth_data.items():
            avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
            accuracy_curve.append((depth, avg_acc))
            
        accuracy_curve.sort(key=lambda x: x[0])
        
        # --- OPTION B: IMPLEMENT COGNITIVE VOLATILITY EQUATION ---
        # Sum of max(0, A_z - A_z-1) for sequential measurements across this specific gravity horizon
        self.volatility_scores[gravity_target] = 0.0
        for i in range(1, len(accuracy_curve)):
            prev_depth, prev_acc = accuracy_curve[i - 1]
            curr_depth, curr_acc = accuracy_curve[i]
            
            if curr_acc > prev_acc:
                delta = curr_acc - prev_acc
                self.volatility_scores[gravity_target] += delta
                print()  # Break carriage return before logging
                logger.info(
                    f"  [VOLATILITY DELTA] Gravity {gravity_target}: Accuracy rose by +{delta:.1f}% "
                    f"from z={prev_depth} to z={curr_depth}."
                )
        
        # Compute tracking features over the cleanly reconstructed dataset
        overall_valid = all(tr.valid for tr in taxonomy_results)
        overall_taxonomy_result = TaxonomyResult(
            classifications=[c for tr in taxonomy_results for c in tr.classifications],
            valid=overall_valid,
            error=None if overall_valid else "Some taxonomy steps failed"
        )
        yield_point = self.compute_fracture_depth(accuracy_curve, overall_taxonomy_result)
        is_invariant = self.check_asymptotic_invariance(accuracy_curve, results_matrix)
        
        # Compute depth-stratified percentage tracking
        self.depth_error_stats = []
        self.depth_error_distribution = {}
        for depth, classifications in self.error_mode_history.items():
            total_count = len(classifications)
            if total_count == 0:
                continue
            counts = {}
            for c in classifications:
                counts[c.mode.value] = counts.get(c.mode.value, 0) + 1
            
            self.depth_error_distribution[depth] = {}
            for mode_val, count in counts.items():
                percentage = (count / total_count) * 100.0
                self.depth_error_distribution[depth][mode_val] = percentage
                self.depth_error_stats.append(DepthErrorModeStore(depth, mode_val, count, percentage))

        return {
            "accuracy_curve": accuracy_curve,
            "fracture_depth": yield_point if yield_point is not None else 0,
            "D_horizon": yield_point or (accuracy_curve[-1][0] if accuracy_curve else 1),
            "is_asymptotically_invariant": is_invariant,
            "cognitive_volatility_score": self.volatility_scores[gravity_target],
            "compliance_status": "COMPLIANT" if yield_point is None else "FRACTURED",
            "depth_error_distribution": self.depth_error_distribution
        }

    def analyze_error_modes(self, depth: int, trajectory: List[TrajectoryStep], grid_dim: int) -> TaxonomyResult:
        """
        Orchestrates error detectors, resolves classification, and aggregates stats.
        Wrapped in TaxonomyResult for validation.
        """
        try:
            # 1. Extract deltas
            deltas = error_taxonomy.extract_deltas(trajectory)
            
            # 2. Run detectors
            results = [
                error_taxonomy.detect_memory_failure(deltas),
                error_taxonomy.detect_algorithm_collapse(deltas, (0, 1), (1, 0)), # Assuming dummy rules or needing proper rules
                error_taxonomy.detect_heuristic_substitution(deltas),
                error_taxonomy.detect_arithmetic_slips(deltas, (0, 1), (1, 0))
            ]
            
            # 3. Resolve by highest confidence
            valid_results = [r for r in results if r.mode != error_taxonomy.ErrorMode.NONE]
            if valid_results:
                best_class = max(valid_results, key=lambda x: x.confidence)
            else:
                best_class = error_taxonomy.ErrorModeClassification(mode=error_taxonomy.ErrorMode.NONE, confidence=0.0, score=0.0, evidence_indices=[])
                
            # 4. Store result
            if depth not in self.error_mode_history:
                self.error_mode_history[depth] = []
            self.error_mode_history[depth].append(best_class)
            
            return TaxonomyResult(classifications=[best_class], valid=True)

        except Exception as e:
            logger.error(f"[TAXONOMY ERROR] Failed at depth {depth}: {e}")
            return TaxonomyResult(classifications=[], valid=False, error=str(e))


if __name__ == "__main__":
    # Rapid verification verification routine
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    finder = FracturePointFinder(accuracy_threshold=60.0, look_ahead_step=2)
    
    print("--- Simulating Live Progress Bar Sequence ---")
    finder.print_search_envelope_bar(low=1, mid=100, high=200)
    import time; time.sleep(0.4)
    finder.print_search_envelope_bar(low=1, mid=50, high=100)
    time.sleep(0.4)
    finder.print_search_envelope_bar(low=25, mid=37, high=50)
    time.sleep(0.2)
    print("\n-------------------------------------------")

    # Mock volatile degradation curve payload testing label integration structures
    mock_matrix = {
        "z_10_gravity_1.2_tier_1_probe_0": {"accuracy": 95.0},
        "z_20_gravity_1.2_tier_1_probe_0": {"accuracy": 80.0},
        "z_30_gravity_1.2_tier_1_probe_0": {"accuracy": 53.3},  # Local dip
        "z_40_gravity_1.2_tier_1_probe_0": {"accuracy": 70.0},  # Volatility recovery
        "z_50_gravity_1.2_tier_1_probe_0": {"accuracy": 30.0},  # Hard collapse
        "z_60_gravity_1.2_tier_1_probe_0": {"accuracy": 10.0}
    }
    
    analysis = finder.parse_matrix_telemetry(mock_matrix, gravity_target=1.2)
    print(f"\nExecution Verification Summary:")
    print(f"Isolated Curve Vector: {analysis['accuracy_curve']}")
    print(f"Calculated Cognitive Volatility Score (V): {analysis['cognitive_volatility_score']:.1f}%")
    print(f"Identified Fracture Point Depth (Bypassing Anomaly): z={analysis['fracture_depth']}")
    print(f"Framework Status Metric: {analysis['compliance_status']}")