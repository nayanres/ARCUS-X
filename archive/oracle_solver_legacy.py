#!/usr/bin/env python3

# Oracle Solver for ARCUS-X
# Rule-following solver that achieves perfect accuracy by design

import random
import re
from typing import Dict, List, Optional
from arcus.evaluation.evaluator import ModelEvaluator
from arcus.tasks.generator import GridTaskGenerator
from arcus.analysis.gravity import DifficultyConfig
from arcus.environment.grid import compute_trajectory

class OracleSolver:
    def __init__(self, model_name: str, api_base: Optional[str] = None, api_key: Optional[str] = None):
        self.evaluator = ModelEvaluator(model_name, api_base=api_base, api_key=api_key)
        self.generator = GridTaskGenerator(seed=42)  # Fixed seed for reproducibility

    def evaluate_oracle_probes(self, n_probes: int = 5, gravity_levels: List[float] = [0.0, 1.0, 2.0, 3.0]) -> Dict:
        results = {}
        for gravity in gravity_levels:
            for _ in range(n_probes):
                tier = DifficultyConfig().tier_for(gravity)
                task = self.generator.generate(tier=tier)
                probe = self._build_probe_from_task(task, gravity)
                result = self.evaluator.evaluate_single_probe(probe, tier=tier, depth=task.horizon)
                results.setdefault(gravity, []).append(result)
        return results

    def _build_probe_from_task(self, task, gravity: float) -> Dict:
        # Calculate final coordinates from ground truth trajectory
        last_coords = [int(n) for n in re.findall(r'\d+', task.ground_truth_trajectory[-1])]
        final_x, final_y = last_coords[0], last_coords[1]
        
        return {
            "id": f"oracle_{random.randint(1000, 9999)}",
            "fol_depth": task.horizon,
            "grid_width": task.grid_width,
            "grid_height": task.grid_height,
            "initial_state": list(task.initial_state),
            "actions": list(task.actions),
            "transition_rules": {k: dict(v) for k, v in task.transition_rules.items()},
            "label_mapping": dict(task.label_mapping) if task.label_mapping else None,
            "variables": task.ground_truth_trajectory,
            "axioms": [
                f"Grid Environment: {task.grid_width}x{task.grid_height} coordinate matrix.",
                f"Initial state: {task.initial_state}",
                "Action rules: " + "; ".join([f"{k}: Δx = {v['dx']}, Δy = {v['dy']}" for k, v in task.transition_rules.items()]),
                "Action sequence: " + ", ".join(task.actions)
            ],
            "question": f"Calculate the full path from {task.initial_state} using the rules above.",
            "expected_answer": f"CHECKSUM: {(final_x * 13) + (final_y * 7)}",
            "gravity_target": gravity
        }

    def solve_perfect(self, probe: Dict) -> str:
        """Generate perfect solution by following rules exactly."""
        initial_state = tuple(probe["initial_state"])
        actions = probe["actions"]
        transition_rules = probe["transition_rules"]
        grid_width = probe["grid_width"]
        grid_height = probe["grid_height"]
        
        # Compute trajectory using the same function as ground truth
        trajectory = compute_trajectory(initial_state, actions, transition_rules, grid_width, grid_height)
        
        # Format as required by the benchmark
        path_str = "->".join(trajectory)
        
        # Calculate checksum
        last_coords = [int(n) for n in re.findall(r'\d+', trajectory[-1])]
        final_x, final_y = last_coords[0], last_coords[1]
        checksum = (final_x * 13) + (final_y * 7)
        
        return f"{path_str}->CHECKSUM: {checksum}"