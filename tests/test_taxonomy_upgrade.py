"""Unit tests for the refactored deterministic taxonomy with separated Output Validity
and Broad Cognitive Failure Taxonomy.
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
    ProbeClassification,
    toroidal_distance,
    E100_EXCEPTION_TOKEN,
    E101_EMPTY_OUTPUT,
    E102_PARSE_FAILURE,
)


GRID_W = 12
GRID_H = 12
RULES = {"east": {"dx": 1, "dy": 0}}
ACTIONS = ["east"] * 10


def _tokens(coords):
    return [f"[{x},{y}]" for x, y in coords]


def test_wraparound_failure():
    """Model moves off the grid instead of wrapping (Transition Failure)."""
    gt = [(10, 0), (11, 0), (0, 0), (1, 0)]
    pred = [(10, 0), (11, 0), (12, 0), (13, 0)]
    res = classify_trajectory(
        _tokens(pred), _tokens(gt), RULES, ACTIONS[:3],
        grid_width=GRID_W, grid_height=GRID_H,
    )
    assert res.outcome == Outcome.FAILURE
    assert res.mechanism == FailureMechanism.TRANSITION_FAILURE
    assert res.probe_classification.valid is True


def test_axis_swap():
    """Model interchanges dx/dy (Transition Failure)."""
    gt = [(0, 0), (1, 0), (2, 0), (3, 0)]
    pred = [(0, 0), (0, 1), (0, 2), (0, 3)]
    res = classify_trajectory(
        _tokens(pred), _tokens(gt), RULES, ACTIONS[:3],
        grid_width=GRID_W, grid_height=GRID_H,
    )
    assert res.outcome == Outcome.FAILURE
    assert res.mechanism == FailureMechanism.TRANSITION_FAILURE
    assert ErrorPattern.AXIS_SWAP in res.patterns


def test_sign_inversion():
    """Model moves in the opposite direction (Transition Failure)."""
    gt = [(0, 0), (1, 0), (2, 0), (3, 0)]
    pred = [(0, 0), (-1, 0), (-2, 0), (-3, 0)]
    res = classify_trajectory(
        _tokens(pred), _tokens(gt), RULES, ACTIONS[:3],
        grid_width=GRID_W, grid_height=GRID_H,
    )
    assert res.outcome == Outcome.FAILURE
    assert res.mechanism == FailureMechanism.TRANSITION_FAILURE
    assert ErrorPattern.SIGN_INVERSION in res.patterns


def test_persistent_drift():
    """Model follows then loses sync, holding a persistent offset (State Tracking Failure)."""
    gt = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
    pred = [(0, 0), (1, 0), (2, 1), (3, 1), (4, 1), (5, 1)]
    res = classify_trajectory(
        _tokens(pred), _tokens(gt), RULES, ACTIONS[:5],
        grid_width=GRID_W, grid_height=GRID_H,
    )
    assert res.outcome == Outcome.FAILURE
    assert res.mechanism == FailureMechanism.STATE_TRACKING_FAILURE
    assert res.dynamics.first_divergence_step == 2
    assert res.dynamics.recovered is False


def test_recovery_case():
    """Model diverges then returns to correct trajectory (State Tracking Failure partial)."""
    gt = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
    pred = [(0, 0), (1, 0), (2, 1), (3, 1), (4, 0), (5, 0)]
    res = classify_trajectory(
        _tokens(pred), _tokens(gt), RULES, ACTIONS[:5],
        grid_width=GRID_W, grid_height=GRID_H,
    )
    assert res.outcome == Outcome.FAILURE
    assert res.mechanism == FailureMechanism.STATE_TRACKING_FAILURE
    assert res.dynamics.recovered is True


def test_formatting_failure_and_validity():
    """Empty / unparseable output -> Invalid Output with error codes."""
    gt = [(0, 0), (1, 0), (2, 0)]
    res = classify_trajectory(
        [], _tokens(gt), RULES, ACTIONS[:2],
        grid_width=GRID_W, grid_height=GRID_H,
        raw_output="",
    )
    assert res.probe_classification.valid is False
    assert res.probe_classification.exception_code == E101_EMPTY_OUTPUT

    res2 = classify_trajectory(
        ["not a coordinate"], _tokens(gt), RULES, ACTIONS[:2],
        grid_width=GRID_W, grid_height=GRID_H,
        raw_output="[EXCEPTION] runtime error",
    )
    assert res2.probe_classification.valid is False
    assert res2.probe_classification.exception_code == E100_EXCEPTION_TOKEN


def test_success_exact_match():
    """Exact match -> SUCCESS, no failure mechanism."""
    gt = [(0, 0), (1, 0), (2, 0), (3, 0)]
    res = classify_trajectory(
        _tokens(gt), _tokens(gt), RULES, ACTIONS[:3],
        grid_width=GRID_W, grid_height=GRID_H,
    )
    assert res.outcome == Outcome.SUCCESS
    assert res.mechanism == FailureMechanism.NONE
    assert res.probe_classification.valid is True


def test_toroidal_distance():
    d = toroidal_distance((0, 0), (11, 0), 12, 12)
    assert abs(d - 1.0) < 1e-9


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
