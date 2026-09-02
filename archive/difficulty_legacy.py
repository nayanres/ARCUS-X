"""Difficulty helpers for ARCUS-X."""

from __future__ import annotations

from typing import Dict, Optional


def difficulty_context(gravity_target: Optional[float] = None) -> Dict[str, object]:
    """Return a compact difficulty payload for prompt and logging use."""
    if gravity_target is None:
        gravity_target = 1.0
    return {
        "gravity_target": float(gravity_target),
        "difficulty_scale": min(1.0, max(0.001, float(gravity_target) / 5.0)),
    }
