"""Unified, model-free baseline evaluators for ARCUS-X.

Baselines establish reference performance points **without** invoking a model:

* :class:`OracleBaseline`      -- applies the transition rules exactly
  (upper bound; should reach 100% trajectory accuracy).
* :class:`RandomBaseline`      -- emits a random *valid* trajectory
  (chance performance).
* :class:`InitialStateBaseline` -- repeats only the initial state
  (trivial strategy; should score poorly on trajectory metrics).

Every baseline implements :class:`BaselineEvaluator`, which exposes
``generate_trajectory(task)`` and ``evaluate(task)``. The runner aggregates
these into a report containing ``baseline_name``, ``trajectory_accuracy``,
``step_accuracy``, ``exact_match`` and (where meaningful) ``fracture_depth``.

These baselines are optional: they are never required for normal model
evaluation and are only included in a report when explicitly enabled.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from arcus.environment.grid import compute_trajectory, format_coord
from arcus.evaluation.metrics import compare_trajectories, mean
from arcus.tasks.generator import GridTask


class BaselineEvaluator:
    """Unified interface for model-free baseline evaluators.

    Subclasses must implement :meth:`generate_trajectory`, returning the
    baseline's predicted trajectory as a list of canonical ``[x,y]`` tokens.
    """

    #: Machine-readable baseline identifier used in reports.
    name: str = "baseline"

    def generate_trajectory(self, task: GridTask) -> List[str]:
        """Return the baseline's predicted trajectory (canonical ``[x,y]`` tokens)."""
        raise NotImplementedError

    def evaluate(self, task: GridTask) -> Dict[str, object]:
        """Score this baseline on a single task against its ground truth.

        Returns a per-task record with ``baseline_name``, ``trajectory_accuracy``
        (1.0 iff the full path is an exact match), ``step_accuracy``,
        ``exact_match`` and ``continuity_score``.
        """
        predicted = self.generate_trajectory(task)
        traj = compare_trajectories(predicted, list(task.ground_truth_trajectory))
        return {
            "baseline_name": self.name,
            "trajectory_accuracy": 1.0 if traj.exact_match else 0.0,
            "step_accuracy": traj.step_accuracy,
            "exact_match": traj.exact_match,
            "continuity_score": traj.continuity_score,
            "predicted_length": traj.predicted_length,
            "ground_truth_length": traj.ground_truth_length,
        }


class OracleBaseline(BaselineEvaluator):
    """Upper-bound baseline: follows the transition rules exactly."""

    name = "oracle"

    def generate_trajectory(self, task: GridTask) -> List[str]:
        return compute_trajectory(
            task.initial_state,
            list(task.actions),
            {k: dict(v) for k, v in task.transition_rules.items()},
            task.grid_width,
            task.grid_height,
        )


class InitialStateBaseline(BaselineEvaluator):
    """Trivial baseline: emits only the initial-state coordinate."""

    name = "initial_state"

    def generate_trajectory(self, task: GridTask) -> List[str]:
        return [format_coord(int(task.initial_state[0]), int(task.initial_state[1]))]


class RandomBaseline(BaselineEvaluator):
    """Chance baseline: a random *valid* walk through the grid.

    At each step a random action is sampled from the task's transition rules and
    applied, producing a trajectory that is always a legal sequence of grid
    transitions (so it is "valid" in the environment sense) but carries no
    task-solving signal.

    Reproducibility: this baseline uses a FIXED, hard-coded seed (``12345``) and
    is therefore a *deterministic* lower-bound reference, NOT a sample from the
    benchmark's master seed. It is intentionally independent of the master seed
    so that the chance floor is stable and comparable across runs and across
    models. The seed may be overridden for ad-hoc experiments, but the published
    lower bound is always the fixed-seed instance.
    """

    name = "random"

    def __init__(self, seed: int = 12345):
        self.rng = random.Random(seed)

    def generate_trajectory(self, task: GridTask) -> List[str]:
        rules = {k: (int(v["dx"]), int(v["dy"])) for k, v in task.transition_rules.items()}
        action_labels = list(rules.keys())
        x, y = int(task.initial_state[0]), int(task.initial_state[1])
        gw, gh = int(task.grid_width), int(task.grid_height)
        traj = [format_coord(x, y)]
        for _ in range(int(task.horizon)):
            action = self.rng.choice(action_labels)
            dx, dy = rules[action]
            x = (x + dx) % gw
            y = (y + dy) % gh
            traj.append(format_coord(x, y))
        return traj


def available_baselines() -> Dict[str, BaselineEvaluator]:
    """Return the registry of built-in baseline evaluators keyed by name."""
    return {
        "oracle": OracleBaseline(),
        "random": RandomBaseline(),
        "initial_state": InitialStateBaseline(),
    }
