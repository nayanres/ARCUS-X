"""Unit tests for the upgraded deterministic, evidence-based failure taxonomy.

Covers the required known-failure examples:

  * wraparound failure
  * axis swap
  * sign inversion
  * persistent drift
  * recovery case
  * formatting failure

Run with:  python tests/test_taxonomy_upgrade.py
(or:        python -m pytest tests/test_taxonomy_upgrade.py)
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from arcus.analysis.taxonomy import (
    classify_trajectory,
    Outcome,
    FailureMechanism,
    FailureSubtype,
    ErrorPattern,
    DivergenceEvolution,
    TaxonomyResult,
    toroidal_distance,
)


# A simple 12x12 toroidal grid with a single "east" action (dx=1, dy=0).
GRID_W = 12
GRID_H = 12
RULES = {"east": {"dx": 1, "dy": 0}}
ACTIONS = ["east"] * 10


def _tokens(coords):
    return [f"[{x},{y}]" for x, y in coords]


def test_wraparound_failure():
    """Model moves off the grid instead of wrapping (ERF_Wraparound)."""
    # Ground truth wraps: (11,0) -> (0,0) on a 12-wide grid.
    gt = [(10, 0), (11, 0), (0, 0), (1, 0)]
    # Model fails to wrap: (11,0) -> (12,0) (off-grid, no wrap).
    pred = [(10, 0), (11, 0), (12, 0), (13, 0)]
    res = classify_trajectory(
        _tokens(pred), _tokens(gt), RULES, ACTIONS[:3],
        grid_width=GRID_W, grid_height=GRID_H,
    )
    assert res.outcome == Outcome.FAILURE
    assert res.mechanism == FailureMechanism.ERF
    assert res.subtype == FailureSubtype.ERF_WRAPAROUND
    assert ErrorPattern.WRAPAROUND in res.patterns


def test_axis_swap():
    """Model interchanges dx/dy (TEF_Axis_Swap)."""
    # Ground truth: east moves +x.
    gt = [(0, 0), (1, 0), (2, 0), (3, 0)]
    # Model swaps axes: moves +y instead of +x.
    pred = [(0, 0), (0, 1), (0, 2), (0, 3)]
    res = classify_trajectory(
        _tokens(pred), _tokens(gt), RULES, ACTIONS[:3],
        grid_width=GRID_W, grid_height=GRID_H,
    )
    assert res.outcome == Outcome.FAILURE
    assert res.mechanism == FailureMechanism.TEF
    assert res.subtype == FailureSubtype.TEF_AXIS_SWAP
    assert ErrorPattern.AXIS_SWAP in res.patterns


def test_sign_inversion():
    """Model moves in the opposite direction (TEF_Direction_Inversion)."""
    gt = [(0, 0), (1, 0), (2, 0), (3, 0)]
    # Model moves west (-x) instead of east (+x).
    pred = [(0, 0), (-1, 0), (-2, 0), (-3, 0)]
    res = classify_trajectory(
        _tokens(pred), _tokens(gt), RULES, ACTIONS[:3],
        grid_width=GRID_W, grid_height=GRID_H,
    )
    assert res.outcome == Outcome.FAILURE
    assert res.mechanism == FailureMechanism.TEF
    assert res.subtype == FailureSubtype.TEF_DIRECTION_INVERSION
    assert ErrorPattern.SIGN_INVERSION in res.patterns


def test_persistent_drift():
    """Model follows then loses sync, holding a persistent offset (SRF)."""
    # Follows correctly for 2 steps, then drifts by +1 in y and stays.
    gt = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
    pred = [(0, 0), (1, 0), (2, 1), (3, 1), (4, 1), (5, 1)]
    res = classify_trajectory(
        _tokens(pred), _tokens(gt), RULES, ACTIONS[:5],
        grid_width=GRID_W, grid_height=GRID_H,
    )
    assert res.outcome == Outcome.FAILURE
    assert res.mechanism == FailureMechanism.SRF
    assert res.subtype == FailureSubtype.SRF_PERSISTENT_OFFSET
    # First divergence is at index 2 (0-based) -> step 3 (1-based).
    assert res.dynamics.first_divergence_step == 2
    assert res.dynamics.recovered is False
    assert res.dynamics.divergence_evolution == DivergenceEvolution.PERSISTENT


def test_recovery_case():
    """Model diverges then returns to the correct trajectory (SRF partial)."""
    gt = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
    # Diverges at step 2 (off by +1 y) but recovers at step 4.
    pred = [(0, 0), (1, 0), (2, 1), (3, 1), (4, 0), (5, 0)]
    res = classify_trajectory(
        _tokens(pred), _tokens(gt), RULES, ACTIONS[:5],
        grid_width=GRID_W, grid_height=GRID_H,
    )
    assert res.outcome == Outcome.FAILURE
    assert res.mechanism == FailureMechanism.SRF
    assert res.subtype == FailureSubtype.SRF_PARTIAL_RECOVERY
    assert res.dynamics.recovered is True
    assert res.dynamics.recovery_latency is not None
    assert res.impact.recovery_rate == 1.0


def test_formatting_failure():
    """Empty / unparseable output -> ORF (Unclassifiable)."""
    gt = [(0, 0), (1, 0), (2, 0)]
    # No parseable coordinates.
    res = classify_trajectory(
        [], _tokens(gt), RULES, ACTIONS[:2],
        grid_width=GRID_W, grid_height=GRID_H,
        raw_output="",
    )
    assert res.outcome == Outcome.UNCLASSIFIABLE
    assert res.mechanism == FailureMechanism.ORF
    assert res.subtype == FailureSubtype.ORF_EMPTY_RESPONSE

    # Malformed structure (non-coordinate text).
    res2 = classify_trajectory(
        ["not a coordinate"], _tokens(gt), RULES, ACTIONS[:2],
        grid_width=GRID_W, grid_height=GRID_H,
        raw_output="the model refused to answer",
    )
    assert res2.outcome == Outcome.UNCLASSIFIABLE
    assert res2.mechanism == FailureMechanism.ORF


def test_success_exact_match():
    """Exact match -> SUCCESS, no failure mechanism."""
    gt = [(0, 0), (1, 0), (2, 0), (3, 0)]
    res = classify_trajectory(
        _tokens(gt), _tokens(gt), RULES, ACTIONS[:3],
        grid_width=GRID_W, grid_height=GRID_H,
    )
    assert res.outcome == Outcome.SUCCESS
    assert res.mechanism == FailureMechanism.NONE
    assert res.impact.mean_divergence == 0.0
    assert res.impact.terminal_divergence == 0.0


def test_toroidal_distance():
    """Toroidal distance wraps correctly across the boundary."""
