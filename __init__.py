"""ARCUS-X benchmark (repository root shim).

The canonical, importable package is :mod:`arcus`. All public symbols resolve
through it, e.g.::

    import arcus
    from arcus.tasks.generator import GridTaskGenerator
    from arcus.evaluation.evaluator import ModelEvaluator
    from arcus.experiments.benchmark_runner import BenchmarkRunner

This root ``__init__.py`` intentionally performs **no** imports of obsolete
top-level modules (the old ``metrics``, ``model_evaluation``,
``benchmark_runner``, ``fracture_finder``, ``cross_model_validator`` and
``statistics`` implementations were folded into the ``arcus`` package during the
hardening pass). Keeping it import-free guarantees that importing the root
package, collecting tests from the repository root, and running
``python -m arcus.experiments.quickstart`` never fail on a stale symbol.
"""

__version__ = "1.0.0"
