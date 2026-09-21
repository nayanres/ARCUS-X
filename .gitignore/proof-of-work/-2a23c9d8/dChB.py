"""Task-generation configuration for ARCUS-X.

Holds the *generation* parameters (grid sizes, horizon ranges, default tiers
and seed). Difficulty/gravity parameters live in
:mod:`arcus.analysis.gravity` to keep the two concerns separate and avoid the
duplicate ``DifficultyConfig`` that previously lived here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class TaskConfig:
    """Deterministic parameters that control procedural task generation."""

    seed: int = 42
    # Multi-seed set for variance measurement. The benchmark is executed once
    # per seed and headline metrics (CRI, fracture depth) are reported as
    # mean +/- std across seeds, so results are not an artifact of one
    # lucky/unlucky seed. Minimum 3 seeds recommended; defaults to 4.
    seeds: Tuple[int, ...] = (42, 123, 456, 789)
    # Inclusive ranges sampled (deterministically) when not overridden.
    grid_width_range: Tuple[int, int] = (5, 12)
    grid_height_range: Tuple[int, int] = (5, 12)
    horizon_range: Tuple[int, int] = (4, 12)
    # The four difficulty tiers that must be producible.
    tiers: Tuple[int, ...] = (0, 1, 2, 3)
    # Base directional vocabulary (Tier 0 "normal" semantics).
    base_directions: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {
            "north": (0, 1),
            "south": (0, -1),
            "east": (1, 0),
            "west": (-1, 0),
        }
    )

    def as_dict(self) -> Dict[str, object]:
        """JSON-serialisable snapshot of the configuration."""
        return {
            "seed": self.seed,
            "grid_width_range": list(self.grid_width_range),
            "grid_height_range": list(self.grid_height_range),
            "horizon_range": list(self.horizon_range),
            "tiers": list(self.tiers),
            "base_directions": {k: list(v) for k, v in self.base_directions.items()},
        }
