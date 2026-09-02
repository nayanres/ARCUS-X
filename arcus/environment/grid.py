"""Grid-based state-transition environment: the single source of truth.

This module is the canonical implementation of the ARCUS-X discrete toroidal
grid world. It owns:

* the coordinate formatting helper (:func:`format_coord`),
* the deterministic transition function (:func:`compute_trajectory`), and
* the :class:`GridEnvironment` wrapper used by both task generation and
  evaluation verification.

There is exactly ONE transition function and ONE trajectory generator in the
whole project: :func:`compute_trajectory`. Every other module (task generator,
evaluator, metrics, taxonomy) must import it from here rather than re-implement
it. Legacy FOL/ILP-style helpers (``generate_transition_rule``,
``generate_expected_path``) were removed during the hardening pass because they
competed with this implementation and were not deterministic w.r.t. the task
environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


def format_coord(x: int, y: int) -> str:
    """Format a grid coordinate as a canonical token ``[x,y]``."""
    return f"[{int(x)},{int(y)}]"


def parse_coord(token: str) -> Tuple[int, int]:
    """Parse a canonical ``[x,y]`` token back into an ``(x, y)`` integer pair.

    Raises
    ------
    ValueError
        If ``token`` is not a well-formed ``[int,int]`` coordinate.
    """
    s = token.strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise ValueError(f"Not a coordinate token: {token!r}")
    inner = s[1:-1]
    parts = inner.split(",")
    if len(parts) != 2:
        raise ValueError(f"Coordinate token must have exactly two components: {token!r}")
    return int(parts[0].strip()), int(parts[1].strip())


def compute_trajectory(
    initial_state: Tuple[int, int],
    actions: List[str],
    transition_rules: Dict[str, Dict[str, int]],
    grid_width: int,
    grid_height: int,
) -> List[str]:
    """Deterministically compute the ground-truth trajectory for a task.

    This is THE transition function of ARCUS-X. It applies each action's
    ``(dx, dy)`` delta to the running state, wrapping coordinates modulo the
    grid dimensions (toroidal topology). The result is the single,
    authoritative ground-truth trajectory used by both task generation and
    evaluation verification, so the ground truth always reflects the
    (possibly tier-mutated) environment.

    Parameters
    ----------
    initial_state:
        ``(x, y)`` starting coordinate.
    actions:
        Ordered list of action labels; each must be a key of ``transition_rules``.
    transition_rules:
        Mapping ``action -> {"dx": int, "dy": int}``.
    grid_width, grid_height:
        Toroidal grid dimensions used for modulo wrapping.

    Returns
    -------
    list[str]
        Canonical coordinate tokens, beginning with the initial state and
        containing one token per applied action (so ``len == 1 + len(actions)``).
    """
    x, y = int(initial_state[0]), int(initial_state[1])
    gw, gh = int(grid_width), int(grid_height)
    traj: List[str] = [format_coord(x, y)]
    for action in actions:
        rule = transition_rules[action]
        dx, dy = int(rule["dx"]), int(rule["dy"])
        x = (x + dx) % gw
        y = (y + dy) % gh
        traj.append(format_coord(x, y))
    return traj


@dataclass
class GridEnvironment:
    """The discrete toroidal grid state-transition environment.

    A thin, explicit wrapper around :func:`compute_trajectory` that maintains
    the grid dimensions and transition rules so callers can step through a
    trajectory one action at a time and reason about expected deltas.
    """

    grid_width: int
    grid_height: int
    transition_rules: Dict[str, Dict[str, int]]
    initial_state: Tuple[int, int]

    def step(self, state: Tuple[int, int], action: str) -> Tuple[int, int]:
        """Return the next state after applying ``action`` to ``state``."""
        rule = self.transition_rules[action]
        dx, dy = int(rule["dx"]), int(rule["dy"])
        nx = (int(state[0]) + dx) % self.grid_width
        ny = (int(state[1]) + dy) % self.grid_height
        return (nx, ny)

    def trajectory(self, actions: List[str]) -> List[str]:
        """Compute the full trajectory for ``actions`` (delegates to the SSoT)."""
        return compute_trajectory(
            self.initial_state, actions, self.transition_rules,
            self.grid_width, self.grid_height,
        )

    def expected_delta(self, action: str) -> Tuple[int, int]:
        """Return the canonical ``(dx, dy)`` delta for ``action``."""
        rule = self.transition_rules[action]
        return int(rule["dx"]), int(rule["dy"])
