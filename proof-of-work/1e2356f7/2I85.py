"""Verification of grid-size scaling as a working optional axis (Task 5)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcus.experiments.benchmark_runner import BenchmarkRunner
from arcus.tasks.generator import GridTaskGenerator


class _StubEvaluator:
    def __init__(self, *a, **k):
        pass


def test_grid_size_iter_default_yields_single():
    runner = BenchmarkRunner(
        model_name="stub",
        evaluator_class=_StubEvaluator,
        n_probes=1,
        grid_size_levels=None,
    )
    levels = list(runner._grid_size_iter())
    assert levels == [(None, None, "default")]


def test_grid_size_iter_explicit_yields_all():
    runner = BenchmarkRunner(
        model_name="stub",
        evaluator_class=_StubEvaluator,
        n_probes=1,
        grid_size_levels=[(8, 8), (16, 16)],
    )
    levels = list(runner._grid_size_iter())
    assert levels == [(8, 8, "8x8"), (16, 16, "16x16")]


def test_unpack_matrix_key_5tuple():
    key = (5, 1.0, 2, 0, "16x16")
    unpacked = BenchmarkRunner._unpack_matrix_key(key)
    assert unpacked == key


def test_unpack_matrix_key_4tuple_default_grid():
    key = (5, 1.0, 2, 0)
    unpacked = BenchmarkRunner._unpack_matrix_key(key)
    assert unpacked == (5, 1.0, 2, 0, "default")


def test_generator_respects_grid_size():
    gen = GridTaskGenerator(seed=42)
    task_default = gen.generate(tier=0, task_index=0, horizon=5)
    task_16 = gen.generate(tier=0, task_index=0, grid_width=16, grid_height=16, horizon=5)
    assert task_default.grid_width != 16 or task_default.grid_height != 16
    assert task_16.grid_width == 16 and task_16.grid_height == 16
    # Ground truth recomputed for the larger grid
    assert len(task_16.ground_truth_trajectory) == 6  # initial + 5 steps


if __name__ == "__main__":
    test_grid_size_iter_default_yields_single()
    test_grid_size_iter_explicit_yields_all()
    test_unpack_matrix_key_5tuple()
    test_unpack_matrix_key_4tuple_default_grid()
    test_generator_respects_grid_size()
    print("Grid scaling audit: all checks passed")
