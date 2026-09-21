"""Procedural grid task generation for the ARCUS-X trajectory benchmark.

Pipeline (faithful to the required seed flow)::

    seed
      -> random.Random(seed)            (master deterministic generator)
      -> GridTaskGenerator.generate()   (per-instance derived RNG)
      -> GridTask                       (serialized task metadata)

Every generated instance is reproducible from ``(seed, tier, task_index)``
while still being unique and different from its siblings.

Single source of truth
-----------------------
The ground-truth trajectory is computed by
:func:`arcus.environment.grid.compute_trajectory` -- the one and only
transition function in the project. This module never re-implements the
transition logic; it only supplies the *inputs* (initial state, action
sequence, tier-mutated transition rules) and delegates the rollout.
"""

from __future__ import annotations

import hashlib
import random
import string
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

from arcus.environment.grid import compute_trajectory, format_coord


# ---------------------------------------------------------------------------
# Base directional vocabulary (Tier 0 "normal" semantics)
# ---------------------------------------------------------------------------
BASE_DIRECTIONS: Dict[str, Tuple[int, int]] = {
    "north": (0, 1),
    "south": (0, -1),
    "east": (1, 0),
    "west": (-1, 0),
}

TIER_NAMES: Dict[int, str] = {
    0: "baseline",
    1: "semantic_disruption",
    2: "attribute_inversion",
    3: "axiomatic_contradiction",
}


def _canonical_signature(parts: List[Any]) -> str:
    """Deterministic, order-stable hash of the given parts (sha256 hex)."""
    payload = repr(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class GridTask:
    """A single procedurally generated ARCUS-X grid trajectory task."""

    task_id: str
    seed: int
    tier: int
    grid_width: int
    grid_height: int
    initial_state: Tuple[int, int]
    horizon: int
    actions: List[str]
    transition_rules: Dict[str, Dict[str, int]]
    label_mapping: Optional[Dict[str, str]] = None
    ground_truth_trajectory: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    experiment_hash: str = ""
    # Model-facing prompt content (procedurally derived from the task's own
    # environment; never contains difficulty metadata, checksums, or answers).
    axioms: List[str] = field(default_factory=list)
    question: str = ""

    # ----- serialization -----
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "tier": self.tier,
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "initial_state": list(self.initial_state),
            "horizon": self.horizon,
            "actions": list(self.actions),
            "transition_rules": {k: dict(v) for k, v in self.transition_rules.items()},
            "label_mapping": dict(self.label_mapping) if self.label_mapping else None,
            "ground_truth_trajectory": list(self.ground_truth_trajectory),
            "metadata": dict(self.metadata),
            "experiment_hash": self.experiment_hash,
            "axioms": list(self.axioms),
            "question": self.question,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GridTask":
        return cls(
            task_id=data["task_id"],
            seed=data["seed"],
            tier=data["tier"],
            grid_width=data["grid_width"],
            grid_height=data["grid_height"],
            initial_state=tuple(data["initial_state"]),
            horizon=data["horizon"],
            actions=list(data["actions"]),
            transition_rules={
                k: {"dx": int(v["dx"]), "dy": int(v["dy"])}
                for k, v in data["transition_rules"].items()
            },
            label_mapping=dict(data["label_mapping"]) if data.get("label_mapping") else None,
            ground_truth_trajectory=list(data.get("ground_truth_trajectory", [])),
            metadata=dict(data.get("metadata", {})),
            experiment_hash=data.get("experiment_hash", ""),
            axioms=list(data.get("axioms", [])),
            question=data.get("question", ""),
        )

    def validate_ground_truth(self) -> bool:
        """Recompute the trajectory from the rules and compare to stored GT.

        Returns ``True`` iff the stored ``ground_truth_trajectory`` is exactly
        what :func:`arcus.environment.grid.compute_trajectory` produces from
        this task's own inputs. This is the reproducibility guarantee required
        by the benchmark: ground truth never depends on model output, parsing,
        or evaluation code.
        """
        recomputed = compute_trajectory(
            self.initial_state, self.actions, self.transition_rules,
            self.grid_width, self.grid_height,
        )
        return recomputed == self.ground_truth_trajectory


class GridTaskGenerator:
    """Procedurally generates unique, reproducible grid trajectory tasks.

    The generator accepts a master ``seed`` and builds a deterministic
    :class:`random.Random` from it. Each task instance derives its own
    reproducible RNG from ``(seed, tier, task_index)`` so that:

    * the same ``(seed, tier, task_index)`` always yields the same task, and
    * different ``task_index`` values yield different, unique instances.
    """

    def __init__(self, seed: int = 42,
                 base_directions: Optional[Dict[str, Tuple[int, int]]] = None):
        self.seed = seed
        self.base_directions = dict(base_directions) if base_directions else dict(BASE_DIRECTIONS)
        # Per-instance cache so sweeps over (tier, task_index, horizon) do not
        # regenerate identical tasks repeatedly.
        self._cache: Dict[Tuple[int, int, Optional[int], Optional[int], Optional[int]], GridTask] = {}

    # ------------------------------------------------------------------
    # Deterministic per-instance RNG
    # ------------------------------------------------------------------
    def _base_rng(self, task_index: int) -> random.Random:
        """Reproducible RNG for the shared base instance of a task.

        The base (grid dimensions, initial state, action sequence) is derived
        ONLY from ``(seed, task_index)`` so that every difficulty tier of the
        same instance shares the identical environment. Tiers then mutate the
        transition semantics on top of this shared base.
        """
        return random.Random(f"arcus-x:{self.seed}:base:{task_index}")

    def _mapping_rng(self, task_index: int) -> random.Random:
        """Reproducible RNG used only for the Tier 1 label (semantic) mutation."""
        return random.Random(f"arcus-x:{self.seed}:semantic:{task_index}")

    # ------------------------------------------------------------------
    # Tier mutation primitives (grid-adapted)
    # ------------------------------------------------------------------
    def _generate_pseudoword(self, rng: random.Random, existing: set) -> str:
        """Generate a unique, token-safe lowercase pseudoword."""
        while True:
            length = rng.randint(4, 8)
            word = "".join(rng.choice(string.ascii_lowercase) for _ in range(length))
            if word not in existing:
                existing.add(word)
                return word

    def _build_label_mapping(self, rng: random.Random, labels: List[str]) -> Dict[str, str]:
        """Tier 1: deterministic 1:1 mapping of action labels to pseudowords."""
        existing: set = set()
        mapping: Dict[str, str] = {}
        for label in labels:
            if label not in mapping:
                mapping[label] = self._generate_pseudoword(rng, existing)
        return mapping

    @staticmethod
    def _invert_directions(directions: Dict[str, Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
        """Tier 2: invert the y-axis semantics.

        Example: normal ``north`` increases ``y``; inverted ``north`` decreases
        ``y``. The generated environment AND ground truth reflect this.
        """
        return {label: (dx, -dy) for label, (dx, dy) in directions.items()}

    @staticmethod
    def _contradictory_directions(directions: Dict[str, Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
        """Tier 3: internally consistent but counterintuitive 90-degree rotation.

        ``north`` no longer moves 'up' (``y+``) but 'right' (``x+``). The system
        is fully self-consistent; the model must follow the provided rules
        rather than its pretrained directional priors.
        """
        return {label: (-dy, dx) for label, (dx, dy) in directions.items()}

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------
    def generate(self, tier: int = 0, task_index: int = 0,
                 grid_width: Optional[int] = None,
                 grid_height: Optional[int] = None,
                 horizon: Optional[int] = None) -> GridTask:
        """Generate one grid trajectory task at the given difficulty ``tier``.

        Parameters
        ----------
        tier:
            0 = baseline, 1 = semantic disruption, 2 = attribute inversion,
            3 = axiomatic contradiction.
        task_index:
            Stable identifier used to derive a unique, reproducible instance.
        grid_width, grid_height:
            Optional overrides; otherwise sampled deterministically.
        horizon:
            Number of transition steps; otherwise sampled deterministically.
        """
        cache_key = (tier, task_index, grid_width, grid_height, horizon)
        if cache_key in self._cache:
            return self._cache[cache_key]

        rng = self._base_rng(task_index)

        gw = int(grid_width) if grid_width else rng.randint(5, 12)
        gh = int(grid_height) if grid_height else rng.randint(5, 12)

        # Initial state is NOT fixed to [0,0]; it is sampled per instance.
        ix = rng.randint(0, gw - 1)
        iy = rng.randint(0, gh - 1)
        initial_state = (ix, iy)

        h = int(horizon) if horizon else rng.randint(4, 12)

        # Base action sequence is drawn once and shared across tiers so that
        # Tier 1 (semantic mutation) preserves the exact same environment and
        # ground-truth positions as Tier 0, while Tier 2/3 change semantics.
        base_labels = list(self.base_directions.keys())
        actions = [rng.choice(base_labels) for _ in range(h)]

        # Apply tier mutation to the transition semantics only.
        directions = dict(self.base_directions)
        label_mapping: Optional[Dict[str, str]] = None
        if tier == 1:
            # SCIENTIFIC FRAMING: Tier 1 is a *semantic invariance* probe, NOT a
            # reasoning-difficulty probe. Only the surface vocabulary (action
            # labels) is remapped to pseudowords; the transition rules, grid,
            # initial state, action sequence, and ground-truth trajectory are
            # byte-for-byte identical to Tier 0. A model that solves Tier 0 but
            # fails Tier 1 reveals a *lexical/semantic binding* failure (it cannot
            # map the new label to the same underlying rule), not a deficit in
            # sequential reasoning. This isolates semantic robustness from
            # reasoning difficulty, which is the explicit purpose of this tier.
            label_mapping = self._build_label_mapping(self._mapping_rng(task_index), base_labels)
        elif tier == 2:
            directions = self._invert_directions(directions)
        elif tier == 3:
            directions = self._contradictory_directions(directions)

        transition_rules = {
            label: {"dx": int(dx), "dy": int(dy)}
            for label, (dx, dy) in directions.items()
        }

        ground_truth = compute_trajectory(
            initial_state, actions, transition_rules, gw, gh
        )

        # Deterministic experiment signature over the task's defining inputs.
        experiment_hash = _canonical_signature([
            self.seed, tier, gw, gh, list(initial_state), h,
            list(actions), {k: [v["dx"], v["dy"]] for k, v in transition_rules.items()},
            dict(label_mapping) if label_mapping else None,
        ])

        task = GridTask(
            task_id=f"arcus_x_{self.seed}_t{tier}_i{task_index}",
            seed=self.seed,
            tier=tier,
            grid_width=gw,
            grid_height=gh,
            initial_state=initial_state,
            horizon=h,
            actions=actions,
            transition_rules=transition_rules,
            label_mapping=label_mapping,
            ground_truth_trajectory=ground_truth,
            metadata={
                "tier_name": TIER_NAMES.get(tier, "unknown"),
                "base_directions": {k: list(v) for k, v in self.base_directions.items()},
            },
            experiment_hash=experiment_hash,
        )
        # Procedurally derive the model-facing prompt content (axioms +
        # question) from the task's own environment. This is the single source
        # of truth for what the model sees; it contains ONLY the grid, initial
        # state, transition rules, action sequence, and the request -- never
        # tier/gravity/experiment_hash, checksums, or the answer.
        axioms, question = self._build_prompt_fields(
            gw, gh, initial_state, actions, transition_rules, label_mapping, h
        )
        task.axioms = axioms
        task.question = question
        self._cache[cache_key] = task
        return task

    # ------------------------------------------------------------------
    # Procedural prompt construction (model-facing content only)
    # ------------------------------------------------------------------
    def _build_prompt_fields(
        self,
        gw: int,
        gh: int,
        initial_state: Tuple[int, int],
        actions: List[str],
        transition_rules: Dict[str, Dict[str, int]],
        label_mapping: Optional[Dict[str, str]],
        horizon: int,
    ) -> Tuple[List[str], str]:
        """Build the ``axioms`` and ``question`` strings the model receives.

        Everything here is derived strictly from the task's own environment so
        the prompt is fully procedural and reproducible. No difficulty metadata,
        checksum, final-coordinate constraint, or ground-truth answer is ever
        included (the shortcut audit enforces this downstream).
        """
        # The labels the model actually sees: Tier 1 remaps base labels to
        # pseudowords via ``label_mapping``; other tiers use the base labels.
        def surface_label(label: str) -> str:
            if label_mapping and label in label_mapping:
                return label_mapping[label]
            return label

        axioms: List[str] = []
        axioms.append(
            f"The world is a grid with width {gw} and height {gh}. "
            f"Coordinates are (x, y) with x in [0, {gw - 1}] and "
            f"y in [0, {gh - 1}]."
        )
        axioms.append("The grid is toroidal.")
        axioms.append("If movement leaves one edge, the agent re-enters from the opposite edge.")
        axioms.append("Coordinates are always wrapped into the valid grid dimensions.")
        axioms.append(f"The agent starts at position {tuple(initial_state)}.")

        rule_lines = []
        for label in actions:
            surf = surface_label(label)
            rule = transition_rules[label]
            rule_lines.append(
                f"When the action is '{surf}', the agent moves by "
                f"(dx={rule['dx']}, dy={rule['dy']})."
            )
        axioms.append("Transition rules:")
        axioms.extend(f"  - {line}" for line in rule_lines)

        action_seq = " -> ".join(surface_label(a) for a in actions)
        axioms.append(f"The agent performs this exact sequence of actions: {action_seq}.")

        question = (
            f"Apply the transition rules step by step starting from the initial "
            f"position, following the action sequence in order. Output the full "
            f"trajectory as a list of {horizon} coordinates "
            f"(initial position followed by each subsequent position), one per "
            f"line, in the format '[x,y]'."
        )
        return axioms, question


if __name__ == "__main__":
    # Quick demonstration of Tier 0-3 generation from a single seed.
    gen = GridTaskGenerator(seed=42)
    for tier in (0, 1, 2, 3):
        task = gen.generate(tier=tier, task_index=0, horizon=6)
        print(f"--- Tier {tier} ({task.metadata['tier_name']}) ---")
        print(f"  grid: {task.grid_width}x{task.grid_height}  initial: {task.initial_state}")
        print(f"  actions: {task.actions}")
        if task.label_mapping:
            print(f"  label_mapping: {task.label_mapping}")
        print(f"  transition_rules: {task.transition_rules}")
        print(f"  ground_truth: {task.ground_truth_trajectory}")
        print(f"  valid_gt: {task.validate_ground_truth()}")
        print()
