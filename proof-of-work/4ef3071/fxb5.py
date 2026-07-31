"""Parsing utilities for model outputs.

This module is the single place that turns a free-form model response into a
list of canonical ``[x,y]`` coordinate tokens. It replaces the legacy
``arcus/environment/path_utils.py`` so that parsing logic lives next to the
rest of the evaluation surface.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# Matches ``[x,y]`` with optional internal whitespace and optional sign.
_COORD_RE = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")


def extract_model_path(raw_output: str) -> List[str]:
    """Extract coordinate tokens ``[x,y]`` from a model response.

    The regex is tolerant of whitespace and signs so that ``[ 3 , -2 ]`` is
    accepted. Coordinates are returned in the order they appear; no assumption
    is made about grid bounds (out-of-range values are caught later by the
    metrics layer, not here).

    Parameters
    ----------
    raw_output:
        The raw text produced by the model.

    Returns
    -------
    list[str]
        Canonical ``[x,y]`` tokens in appearance order. Empty if nothing
        parseable was found.
    """
    if not raw_output or not isinstance(raw_output, str):
        return []
    return [f"[{m[0]},{m[1]}]" for m in _COORD_RE.findall(raw_output)]


def estimated_optimal_path_length(z: int) -> int:
    """Deterministic estimate of the token cost of a correct path of depth ``z``.

    Used as the efficiency baseline (expected completion tokens) so that the
    Generation Bloat Index has a stable reference. Pure function of ``z``.
    """
    # (z + 1) coordinate pairs at ~10 tokens each, plus a small formatting tail.
    return 10 * (max(0, int(z)) + 1) + 10


def parse_first_coord(token: str) -> Optional[Tuple[int, int]]:
    """Parse a single ``[x,y]`` token, returning ``None`` if malformed."""
    m = _COORD_RE.fullmatch(token.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))
