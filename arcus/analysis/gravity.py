"""Deterministic difficulty layer for ARCUS-X."""

from __future__ import annotations

from typing import Dict, List, Optional


class DifficultyTier:
    """Deterministic benchmark difficulty tiers.
    
    Tier represents the **type** of perturbation applied to the evaluation
    environment. Each tier changes different aspects of the transition semantics
    while preserving the underlying task structure (grid size, initial state,
    action sequence).
    
    Tier 0 (Baseline):
        - No perturbation; uses the original transition rules.
        - Serves as the control condition for all difficulty experiments.
    
    Tier 1 (Semantic Disruption):
        - Only action labels are remapped to pseudowords.
        - The underlying transition dynamics and ground‑truth trajectory remain
          **identical** to Tier 0.
        - This is a *semantic robustness* probe: models must follow the provided
          rules despite label changes.
    
    Tier 2 (Attribute Inversion):
        - The sign of the y‑axis delta is flipped (north ↔ south).
        - Ground truth is recomputed from the inverted rules.
        - Tests robustness to altered transition assumptions.
    
    Tier 3 (Axiomatic Contradiction):
        - Deltas are rotated by 90° (north → east, east → south, etc.).
        - Ground truth is recomputed from the rotated rules.
        - Tests robustness to fundamentally different transition semantics.
    """
    BASELINE = 0
    SEMANTIC = 1
    INVERSION = 2
    CONTRADICTION = 3


class DifficultyConfig:
    """Config wrapper for deterministic task perturbation strength."""

    def __init__(self, gravity_levels: Optional[List[float]] = None, seed: int = 42):
        self.gravity_levels = gravity_levels or [0.0, 1.0, 2.0, 3.0]
        self.seed = seed

    def tier_for(self, gravity: float) -> int:
        if gravity >= 3.0:
            return DifficultyTier.CONTRADICTION
        if gravity >= 2.0:
            return DifficultyTier.INVERSION
        if gravity >= 1.0:
            return DifficultyTier.SEMANTIC
        return DifficultyTier.BASELINE

    def disruption_ratio(self, gravity: float) -> float:
        return min(1.0, max(0.0, float(gravity) / 3.0))

    def as_dict(self) -> Dict[str, object]:
        return {"gravity_levels": self.gravity_levels, "seed": self.seed}


def build_gravity_context(gravity: float) -> Dict[str, object]:
    """Return deterministic difficulty metadata for prompts and logging."""
    return {
        "gravity_target": float(gravity),
        "tier": DifficultyConfig().tier_for(gravity),
        "disruption_ratio": DifficultyConfig().disruption_ratio(gravity),
    }
