"""Benchmark metrics."""

import logging
import re
import ast
from typing import Optional, Dict, List, Tuple, Any, Set
from dataclasses import dataclass
import math

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PathTraceResult:
    """Result of path trace validation"""
    is_valid: bool
    path: List[str]
    broken_at_step: Optional[int] = None
    error_message: Optional[str] = None
    score: float = 0.0


class PathTraceExtractor:
    """Extracts variable path traces from model outputs"""
    
    # Pattern to match comma-separated variable paths
    COMMA_SEPARATED_PATTERN = r'([a-zA-Z_]\w*(?:\s*,\s*[a-zA-Z_]\w*)*)'
    # Pattern for arrow-separated paths
    ARROW_PATTERN = r'([a-zA-Z_]\w*(?:\s*->\s*[a-zA-Z_]\w*)+)'
    # Pattern for dot-separated paths (like graph notation)
    DOT_PATTERN = r'([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)'
    
    @staticmethod
    def extract_from_output(output_text: str) -> List[List[str]]:
        """
        Extract variable path traces from model output.
        Tries multiple pattern formats and returns all found paths.
        
        Args:
            output_text: Raw output from model
        
        Returns:
            List of paths, where each path is a list of variable names
        """
        paths = []
        
        if not output_text or not isinstance(output_text, str):
            return paths
        
        arrow_matches = re.finditer(PathTraceExtractor.ARROW_PATTERN, output_text)
        for match in arrow_matches:
            path_str = match.group(1)
            path = [var.strip() for var in re.split(r'\s*->\s*', path_str)]
            if path and len(path) > 1:
                paths.append(path)
        
        if paths:
            return paths
        
        dot_matches = re.finditer(PathTraceExtractor.DOT_PATTERN, output_text)
        for match in dot_matches:
            path_str = match.group(1)
            path = [var.strip() for var in path_str.split('.')]
            if path and len(path) > 1:
                paths.append(path)
        
        if paths:
            return paths
        
        comma_matches = re.finditer(PathTraceExtractor.COMMA_SEPARATED_PATTERN, output_text)
        for match in comma_matches:
            path_str = match.group(1)
            path = [var.strip() for var in path_str.split(',') if var.strip()]
            if path and len(path) > 1 and all(var.isidentifier() for var in path):
                paths.append(path)
        
        return paths


class FOLGraphValidator:
    """Validates paths against FOL graph structure"""
    
    def __init__(self, predicate_graph: Dict[str, Set[str]], facts: List[str]):
        """
        Initialize validator with FOL graph structure.
        
        Args:
            predicate_graph: Dict mapping predicates to their dependencies
            facts: List of ground facts in the graph
        """
        self.predicate_graph = predicate_graph or {}
        self.facts = set(facts) if facts else set()
    
    def validate_path(self, path: List[str]) -> PathTraceResult:
        """
        Validate a path against the FOL graph.
        
        Args:
            path: List of variable/predicate names to validate
        
        Returns:
            PathTraceResult with validation status
        """
        if not path or len(path) < 2:
            return PathTraceResult(
                is_valid=False,
                path=path,
                error_message="Path must contain at least 2 nodes"
            )
        
        current_node = path[0]
        valid_path = [current_node]
        
        for step_idx, next_node in enumerate(path[1:], 1):
            is_reachable = self._is_reachable(current_node, next_node)
            
            if not is_reachable:
                return PathTraceResult(
                    is_valid=False,
                    path=valid_path,
                    broken_at_step=step_idx,
                    error_message=f"No edge from '{current_node}' to '{next_node}'",
                    score=0.0
                )
            
            valid_path.append(next_node)
            current_node = next_node
        
        return PathTraceResult(
            is_valid=True,
            path=valid_path,
            score=1.0
        )
    
    def _is_reachable(self, source: str, target: str, max_depth: int = 5) -> bool:
        """Check if target is reachable from source via graph edges"""
        visited = set()
        queue = [(source, 0)]
        
        while queue:
            current, depth = queue.pop(0)
            
            if depth > max_depth:
                continue
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == target:
                return True
            
            if current in self.predicate_graph:
                for next_pred in self.predicate_graph[current]:
                    if next_pred not in visited:
                        queue.append((next_pred, depth + 1))
        
        return False


class PathTraceValidityMetric:
    """
    Custom metric that validates variable path traces from model outputs.
    
    Integrates with deepeval BaseMetric pattern while independently
    validating FOL path traces.
    """
    
    def __init__(
        self,
        predicate_graph: Optional[Dict[str, Set[str]]] = None,
        facts: Optional[List[str]] = None,
        threshold: float = 0.7
    ):
        """
        Initialize PathTraceValidityMetric.
        
        Args:
            predicate_graph: FOL graph predicate dependencies
            facts: Ground facts in the graph
            threshold: Score threshold for success
        """
        self.predicate_graph = predicate_graph or {}
        self.facts = facts or []
        self.threshold = threshold
        
        self.validator = FOLGraphValidator(self.predicate_graph, self.facts)
        self.score = None
        self.success = False
        self.reason = ""
    
    def measure(self, model_output: str) -> float:
        """
        Measure path trace validity from model output.
        
        Args:
            model_output: Raw output from evaluated model
        
        Returns:
            Score between 0.0 and 1.0
        """
        paths = PathTraceExtractor.extract_from_output(model_output)
        
        if not paths:
            self.score = 0.0
            self.success = False
            self.reason = "No valid path traces found in output"
            return self.score
        
        path_results = [self.validator.validate_path(path) for path in paths]
        
        valid_paths = [r for r in path_results if r.is_valid]
        
        if valid_paths:
            self.score = len(valid_paths) / len(path_results)
            self.success = self.score >= self.threshold
            self.reason = f"Validated {len(valid_paths)}/{len(path_results)} paths"
        else:
            self.score = 0.0
            self.success = False
            self.reason = f"All {len(path_results)} paths are invalid"
        
        return self.score
    
    def is_successful(self) -> bool:
        """Check if metric evaluation was successful"""
        return self.success
    
    def __name__(self) -> str:
        return "PathTraceValidityMetric"


class RiemannSumAggregator:
    """
    Aggregates evaluation results using Riemann sum over depth and semantic gravity.
    
    Implements discrete double-nested Riemann summation for Composite Robustness Index (CRI):
      CRI = Σ_{d=1}^{D_horizon} Σ_{G_s ∈ G} [ Accuracy(G_s, d) / (1 + ln(1 + GII(G_s, d))) ] * ΔG_s
    """
    
    @staticmethod
    def calculate_riemann_cri(
        results_matrix: Dict[Tuple[float, float], float],
        delta_d: float = 1.0,
        delta_g_s: float = 1.0,
        apply_constraint_penalty: bool = True,
        gii_matrix: Dict[Tuple[float, float], float] = None
    ) -> float:
        """
        Calculate Composite Robustness Index (CRI) using Riemann sum.
        
        Implements discrete double-nested loop Riemann sum over:
        - Depth (d): structural complexity dimension
        - Semantic Gravity (G_s): model confidence dimension
        
        Updated constraint penalty (research paper):
          point_value = accuracy / (1 + ln(1 + GII))
        
        Args:
            results_matrix: Dict mapping (depth, gravity) tuples to accuracy values
            delta_d: Depth step size for Riemann integration
            delta_g_s: Gravity step size for Riemann integration
            apply_constraint_penalty: Whether to apply GII penalty
            gii_matrix: Optional dict mapping (depth, gravity) to GII values
        
        Returns:
            CRI value (sum of normalized accuracies across all depth/gravity points)
        """
        if not results_matrix:
            logger.warning("Empty results matrix provided")
            return 0.0
        
        depths = sorted(set(d for d, g in results_matrix.keys()))
        gravities = sorted(set(g for d, g in results_matrix.keys()))
        
        if not depths or not gravities:
            return 0.0
        
        min_depth = min(depths)
        max_depth = max(depths)
        min_gravity = min(gravities)
        max_gravity = max(gravities)
        
        accuracies = []
        efficiencies = []
        
        d = min_depth
        while d <= max_depth:
            g_s = min_gravity
            while g_s <= max_gravity:
                if (d, g_s) in results_matrix:
                    accuracy = results_matrix[(d, g_s)]
                    accuracies.append(accuracy)
                    
                    if apply_constraint_penalty:
                        if gii_matrix and (d, g_s) in gii_matrix:
                            gii = gii_matrix[(d, g_s)]
                        else:
                            gii = RiemannSumAggregator._compute_generalization_inconsistency(
                                results_matrix, d, g_s
                            )
                        efficiency = 1.0 / (1.0 + math.log1p(max(gii, 0.0)))
                    else:
                        efficiency = 1.0
                    efficiencies.append(efficiency)
                
                g_s += delta_g_s
            
            d += delta_d
        
        if not accuracies:
            return {"mean_accuracy": 0.0, "mean_efficiency_coefficient": 0.0, "composite_robustness_index": 0.0}
        
        mean_accuracy = sum(accuracies) / len(accuracies)
        mean_efficiency = sum(efficiencies) / len(efficiencies)
        cri = mean_accuracy * mean_efficiency
        
        return {
            "mean_accuracy": float(mean_accuracy),
            "mean_efficiency_coefficient": float(mean_efficiency),
            "composite_robustness_index": float(max(0.0, min(1.0, cri)))
        }
    
    @staticmethod
    def _compute_generalization_inconsistency(
        results_matrix: Dict[Tuple[float, float], float],
        target_d: float,
        target_g_s: float,
        window_size: int = 3
    ) -> float:
        """
        Compute Generalization Inconsistency Index (GII).
        
        Measures how much model accuracy varies in a neighborhood of (d, G_s).
        Higher GII indicates less robust generalization.
        
        Args:
            results_matrix: Full evaluation matrix
            target_d: Target depth coordinate
            target_g_s: Target gravity coordinate
            window_size: Size of neighborhood window
        
        Returns:
            GII value (higher = less consistent)
        """
        depths = sorted(set(d for d, g in results_matrix.keys()))
        gravities = sorted(set(g for d, g in results_matrix.keys()))
        
        if not depths or not gravities:
            return 0.0
        
        depth_idx = depths.index(target_d) if target_d in depths else 0
        gravity_idx = gravities.index(target_g_s) if target_g_s in gravities else 0
        
        half_window = window_size // 2
        depth_start = max(0, depth_idx - half_window)
        depth_end = min(len(depths), depth_idx + half_window + 1)
        gravity_start = max(0, gravity_idx - half_window)
        gravity_end = min(len(gravities), gravity_idx + half_window + 1)
        
        neighborhood_accuracies = []
        for d_idx in range(depth_start, depth_end):
            for g_idx in range(gravity_start, gravity_end):
                key = (depths[d_idx], gravities[g_idx])
                if key in results_matrix:
                    neighborhood_accuracies.append(results_matrix[key])
        
        if not neighborhood_accuracies:
            return 0.0
        
        mean_accuracy = sum(neighborhood_accuracies) / len(neighborhood_accuracies)
        variance = sum((x - mean_accuracy) ** 2 for x in neighborhood_accuracies) / len(neighborhood_accuracies)
        std_dev = math.sqrt(variance)
        
        gii = std_dev / (mean_accuracy + 1e-8)
        
        return float(gii)


def calculate_riemann_cri(
    results_matrix: Dict[Tuple[float, float], float],
    delta_d: float = 1.0,
    delta_g_s: float = 1.0
) -> Dict[str, float]:
    """
    Convenience function to calculate Composite Robustness Index.
    
    Args:
        results_matrix: Dictionary mapping (depth, gravity) to accuracy
        delta_d: Depth step size
        delta_g_s: Gravity step size
    
    Returns:
        Dictionary with CRI metrics
    """
    return RiemannSumAggregator.calculate_riemann_cri(
        results_matrix,
        delta_d=delta_d,
        delta_g_s=delta_g_s,
        apply_constraint_penalty=True
    )


class GenerationInefficiencyIndex:
    """
    Generation Inefficiency Index (GII): Quantifies token overhead under semantic displacement.
    
    Formula: GII = (|T_think| + |T_out|) * (1 + L) / D_FOL
    
    Where:
      - |T_think|: Internal reasoning/search tokens
      - |T_out|: Terminal generation tokens
      - L: Syllogistic Leakage Score
      - D_FOL: Discrete FOL graph depth
    """
    
    @staticmethod
    def calculate(
        thinking_tokens: int,
        output_tokens: int,
        leakage_score: float,
        fol_depth: int
    ) -> float:
        """Calculate GII given components"""
        if fol_depth == 0:
            return float('inf')
        
        numerator = (thinking_tokens + output_tokens) * (1.0 + leakage_score)
        gii = numerator / fol_depth
        return float(gii)


class HorizonCompliance:
    """Measures whether the model generated the requested number of transitions."""
    
    @staticmethod
    def calculate(requested_depth: int, generated_depth: int) -> float:
        """
        HC = 1 - abs(generated_depth - z) / max(generated_depth, z)
        """
        if requested_depth <= 0 or generated_depth <= 0:
            return 0.0
        
        hc = 1.0 - abs(generated_depth - requested_depth) / max(generated_depth, requested_depth)
        return float(max(0.0, hc))


class GenerationBloatIndex:
    """Quantifies excessive output generation relative to expected output size."""
    
    @staticmethod
    def calculate(actual_tokens: int, expected_tokens: int) -> float:
        """
        GBI = max(0, actual_tokens - expected_tokens) / expected_tokens
        """
        if expected_tokens <= 0:
            return 0.0
        
        gbi = max(0, actual_tokens - expected_tokens) / expected_tokens
        return float(gbi)


class GenerationEfficiency:
    """Quantifies generation efficiency."""
    
    @staticmethod
    def calculate(gbi: float) -> float:
        """
        GE = 1 / (1 + GBI)
        """
        ge = 1.0 / (1.0 + gbi)
        return float(ge)


class SyllogisticLeakageScore:
    """
    Syllogistic Leakage Score (L): Captures semantic failures without length bias.
    
    Formula: L = (sum(Contradictions) + ω*sum(Affirmations of Priors)) / |A|
    
    Where:
      - Contradictions: Outputs contradicting prompt rules
      - Affirmations of Priors: Affirmations of training priors contradicting prompt (ω=2.0)
      - |A|: Total number of assertions in output
    """
    
    AFFIRMATION_PENALTY = 2.0
    
    @staticmethod
    def calculate(
        contradictions: int,
        prior_affirmations: int,
        total_assertions: int
    ) -> float:
        """Calculate Syllogistic Leakage Score"""
        if total_assertions == 0:
            return 0.0
        
        leakage = (contradictions + SyllogisticLeakageScore.AFFIRMATION_PENALTY * prior_affirmations) / total_assertions
        return float(leakage)


class StructuralYieldPoint:
    """
    Structural Yield Point (E_yield): Critical G_s threshold where reasoning fractures.
    
    Identifies the exact semantic gravity value at which the model's internal reasoning
    engine completely fractures, causing accuracy to drop precipitously below random chance.
    """
    
    @staticmethod
    def calculate(
        gravity_values: List[float],
        accuracy_values: List[float],
        random_baseline: float = 0.5
    ) -> Tuple[float, float]:
        """
        Calculate structural yield point.
        
        Returns:
            (yield_point_gravity, yield_point_accuracy): The G_s and accuracy at fracture
        """
        if len(gravity_values) != len(accuracy_values):
            return float('inf'), random_baseline
        
        for g, acc in zip(gravity_values, accuracy_values):
            if acc < random_baseline:
                return g, acc
        
        return float('inf'), 1.0
    
    @staticmethod
    def detect_fracture(accuracy_sequence: List[float]) -> bool:
        """Detect if accuracy drops below random chance for 3 consecutive measurements"""
        if len(accuracy_sequence) < 3:
            return False
        
        for i in range(len(accuracy_sequence) - 2):
            if all(acc < 0.5 for acc in accuracy_sequence[i:i+3]):
                return True
        
        return False


class EvaluationMetricsFactory:
    """Factory for creating evaluation metrics"""
    
    @staticmethod
    def create_path_trace_metric(
        fol_graph_dict: Dict[str, Any],
        threshold: float = 0.7
    ) -> PathTraceValidityMetric:
        """
        Create PathTraceValidityMetric from FOL graph dictionary.
        
        Args:
            fol_graph_dict: Dictionary with 'facts', 'rules', 'predicate_graph'
            threshold: Success threshold
        
        Returns:
            Initialized PathTraceValidityMetric
        """
        predicate_graph = {}
        if 'predicate_graph' in fol_graph_dict:
            for source, targets in fol_graph_dict['predicate_graph'].items():
                predicate_graph[source] = set(targets)
        
        facts = fol_graph_dict.get('facts', [])
        
        return PathTraceValidityMetric(
            predicate_graph=predicate_graph,
            facts=facts,
            threshold=threshold
        )


if __name__ == "__main__":
    example_graph = {
        "facts": ["person(alice)", "person(bob)", "tall(alice)"],
        "rules": ["smart(X) :- person(X), tall(X)"],
        "predicate_graph": {
            "person": {"smart"},
            "tall": {"smart"}
        }
    }
    
    metric = EvaluationMetricsFactory.create_path_trace_metric(example_graph)
    
    model_output = "The path trace is: person -> tall -> smart"
    score = metric.measure(model_output)
    logger.info(f"Path trace validity score: {score}")
    logger.info(f"Metric success: {metric.is_successful()}")
    logger.info(f"Reason: {metric.reason}")
    
    results_matrix = {
        (1.0, 0.5): 0.85,
        (1.0, 1.0): 0.78,
        (2.0, 0.5): 0.92,
        (2.0, 1.0): 0.88,
        (3.0, 0.5): 0.81,
        (3.0, 1.0): 0.75,
    }
    
    cri_data = calculate_riemann_cri(results_matrix, delta_d=1.0, delta_g_s=0.5)
    logger.info(f"Composite Robustness Index: {cri_data['composite_robustness_index']:.4f}")
