#!/usr/bin/env python3

# Difficulty Validation Experiments for ARCUS-X
# Supports horizon scaling, tier scaling, and seed variance experiments

import random
from typing import Dict, List, Optional, Tuple
from arcus.tasks.generator import GridTaskGenerator
from arcus.analysis.gravity import DifficultyConfig
from arcus.environment.grid import compute_trajectory

class DifficultyValidation:
    """Runs controlled experiments to validate benchmark difficulty structure."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.generator = GridTaskGenerator(seed=seed)
    
    def horizon_scaling(self, tiers: List[int] = [0, 1, 2, 3], 
                        horizons: List[int] = [4, 8, 12, 16, 20],
                        task_index: int = 0) -> Dict:
        """Validate that task difficulty scales with horizon (sequence length)."""
        results = {}
        for tier in tiers:
            results[tier] = {}
            for h in horizons:
                task = self.generator.generate(tier=tier, task_index=task_index, horizon=h)
                # Ground truth length is horizon + 1 (initial state + steps)
                results[tier][h] = {
                    "ground_truth_length": len(task.ground_truth_trajectory),
                    "expected_length": h + 1,
                    "valid": task.validate_ground_truth(),
                    "experiment_hash": task.experiment_hash,
                }
        return results
    
    def tier_scaling(self, gravity_levels: List[float] = [0.0, 1.0, 2.0, 3.0],
                     task_index: int = 0, horizon: int = 8) -> Dict:
        """Validate that tiers produce the expected trajectory relationships."""
        results = {}
        baseline_task = self.generator.generate(tier=0, task_index=task_index, horizon=horizon)
        baseline_traj = baseline_task.ground_truth_trajectory
        
        for gravity in gravity_levels:
            tier = DifficultyConfig().tier_for(gravity)
            task = self.generator.generate(tier=tier, task_index=task_index, horizon=horizon)
            traj = task.ground_truth_trajectory
            
            # ``trajectory_preserved`` means the tier's trajectory is identical to
            # the Tier-0 baseline. Tier 1 (label-only mutation) preserves it;
            # Tiers 2/3 change the transition semantics and therefore do NOT.
            trajectory_preserved = (traj == baseline_traj)

            results[gravity] = {
                "tier": tier,
                "trajectory_preserved": trajectory_preserved,
                "trajectory_modified": not trajectory_preserved,
                "valid": task.validate_ground_truth(),
                "experiment_hash": task.experiment_hash,
            }
        return results
    
    def seed_variance(self, seeds: List[int] = [42, 100, 200, 300],
                     tier: int = 0, task_index: int = 0, horizon: int = 8) -> Dict:
        """Validate that different seeds produce independent (different) tasks."""
        results = {}
        trajectories = {}
        for seed in seeds:
            gen = GridTaskGenerator(seed=seed)
            task = gen.generate(tier=tier, task_index=task_index, horizon=horizon)
            results[seed] = {
                "valid": task.validate_ground_truth(),
                "experiment_hash": task.experiment_hash,
                "initial_state": list(task.initial_state),
                "actions": list(task.actions),
            }
            trajectories[seed] = task.ground_truth_trajectory
        
        # Check pairwise independence
        unique_trajectories = set(tuple(t) for t in trajectories.values())
        results["summary"] = {
            "num_seeds": len(seeds),
            "unique_trajectories": len(unique_trajectories),
            "all_independent": len(unique_trajectories) == len(seeds),
        }
        return results
    
    def reproducibility_check(self, seed: int = 42, tier: int = 1, 
                             task_index: int = 5, horizon: int = 10) -> Dict:
        """Validate that same (seed, tier, task_index) reproduces identical task."""
        gen1 = GridTaskGenerator(seed=seed)
        gen2 = GridTaskGenerator(seed=seed)
        t1 = gen1.generate(tier=tier, task_index=task_index, horizon=horizon)
        t2 = gen2.generate(tier=tier, task_index=task_index, horizon=horizon)
        
        return {
            "identical_trajectory": t1.ground_truth_trajectory == t2.ground_truth_trajectory,
            "identical_initial_state": t1.initial_state == t2.initial_state,
            "identical_actions": t1.actions == t2.actions,
            "identical_rules": t1.transition_rules == t2.transition_rules,
            "identical_hash": t1.experiment_hash == t2.experiment_hash,
            "both_valid": t1.validate_ground_truth() and t2.validate_ground_truth(),
        }


if __name__ == "__main__":
    dv = DifficultyValidation(seed=42)
    
    print("=== Horizon Scaling ===")
    hs = dv.horizon_scaling()
    for tier, data in hs.items():
        print(f"  Tier {tier}: {data}")
    
    print("\n=== Tier Scaling ===")
    ts = dv.tier_scaling()
    for gravity, data in ts.items():
        print(f"  Gravity {gravity}: {data}")
    
    print("\n=== Seed Variance ===")
    sv = dv.seed_variance()
    for seed, data in sv.items():
        print(f"  Seed {seed}: {data}")
    
    print("\n=== Reproducibility ===")
    rc = dv.reproducibility_check()
    print(f"  {rc}")
