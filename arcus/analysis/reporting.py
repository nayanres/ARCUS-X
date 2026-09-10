"""Shared reporting taxonomy helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


TAXONOMY_ALIAS_MAP = {
    "state drift": "State Tracking Failure",
    "state tracking failure": "State Tracking Failure",
    "state tracking": "State Tracking Failure",
    "wraparound errors": "Transition Rule Failure",
    "transition rule failure": "Transition Rule Failure",
    "transition failure": "Transition Rule Failure",
    "transition": "Transition Rule Failure",
    "semantic interpretation failure": "Semantic Interpretation Failure",
    "semantic failure": "Semantic Interpretation Failure",
    "semantic": "Semantic Interpretation Failure",
    "horizon collapse": "Horizon Collapse",
    "horizon failure": "Horizon Collapse",
    "horizon": "Horizon Collapse",
    "formatting failures": "Formatting Failure",
    "formatting failure": "Formatting Failure",
    "output format failure": "Formatting Failure",
    "malformed output": "Formatting Failure",
    "output format": "Formatting Failure",
    "transition rule": "Transition Rule Failure",
    "semantic interpretation": "Semantic Interpretation Failure",
    "none": "None",
    "no failure": "None",
    "unknown": "Unknown / Unmapped",
    "unknown / unmapped": "Unknown / Unmapped",
    "unmapped": "Unknown / Unmapped",
}


def normalize_taxonomy_label(value: Any) -> str:
    """Normalize taxonomy strings to a canonical lookup form."""
    cleaned = str(value or "").strip().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", cleaned).lower()


def is_fully_correct(
    result: Dict[str, Any],
    probe_classification: Optional[Dict[str, Any]] = None,
) -> bool:
    """Use the benchmark trajectory signals as the correctness SSOT."""
    if not isinstance(result, dict):
        return False
    pc = probe_classification or {}
    try:
        step_accuracy = float(result.get("step_accuracy", 0.0) or 0.0)
    except (TypeError, ValueError):
        step_accuracy = 0.0
    return bool(
        result.get("exact_match", False)
        or pc.get("exact_match", False)
        or pc.get("trajectory_correct", False)
        or step_accuracy >= 1.0
    )


def canonicalize_taxonomy(raw_mode: Any, fully_correct: bool) -> str:
    """Map a raw taxonomy label into the report's canonical buckets."""
    if fully_correct:
        return "None"
    normalized = normalize_taxonomy_label(raw_mode)
    return TAXONOMY_ALIAS_MAP.get(normalized, "Unknown / Unmapped")
