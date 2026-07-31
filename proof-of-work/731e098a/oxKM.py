"""Validation suite for the ARCUS-X scientific hardening pass.

Covers the required scenarios:

  1. Same seed -> same task (determinism)
  2. Different seeds -> different tasks
  3. Tier 0/1 preserve trajectory relationship (Tier 1 = label mutation only)
  4. Tier 2/3 modify ground truth correctly (recomputed, internally consistent)
  5. Parser rejects malformed paths
  6. Metrics score perfect / partial / wrong-path-but-right-final
  7. Fracture depth works on an accuracy curve
  8. Full offline pipeline from the public entry-point modules
  9. Difficulty metadata is never leaked to the model prompt (audit)
 10. Oracle solver achieves perfect accuracy (upper-bound sanity check)
 11. Random baseline performs near chance (lower-bound sanity check)
 12. Initial-state baseline scores ~0 on trajectory metrics (lower-bound)

Run with:  python tests/test_arcus_hardening.py
(or:        python -m pytest tests/test_arcus_hardening.py)
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from arcus.tasks.generator import GridTaskGenerator
from arcus.environment.grid import compute_trajectory
from arcus.evaluation.parser import extract_model_path
from arcus.evaluation.metrics import compare_trajectories, HorizonCompliance, ARCUSRobustnessIndex
from arcus.analysis.taxonomy import classify_from_result, ErrorMode
from arcus.analysis.fracture import FracturePointFinder
from arcus.experiments.benchmark_runner import BenchmarkRunner
from arcus.experiments.task_shortcut_audit import TaskShortcutAudit
from arcus.experiments.difficulty_validation import DifficultyValidation


def test_same_seed_same_task():
    g1 = GridTaskGenerator(seed=42)
    g2 = GridTaskGenerator(seed=42)
    t1 = g1.generate(tier=0, task_index=3, horizon=6)
    t2 = g2.generate(tier=0, task_index=3, horizon=6)
    assert t1.ground_truth_trajectory == t2.ground_truth_trajectory
    assert t1.experiment_hash == t2.experiment_hash
    assert t1.to_dict() == t2.to_dict()


def test_different_seeds_different_tasks():
    g1 = GridTaskGenerator(seed=42)
    g2 = GridTaskGenerator(seed=7)
    t1 = g1.generate(tier=0, task_index=3, horizon=6)
    t2 = g2.generate(tier=0, task_index=3, horizon=6)
    assert t1.ground_truth_trajectory != t2.ground_truth_trajectory


def test_different_task_index_unique():
    g = GridTaskGenerator(seed=42)
    seen = set()
    for i in range(20):
        t = g.generate(tier=0, task_index=i, horizon=6)
        key = tuple(t.ground_truth_trajectory)
        assert key not in seen, f"collision at task_index={i}"
        seen.add(key)


def test_tier0_tier1_preserve_trajectory():
    g = GridTaskGenerator(seed=42)
    t0 = g.generate(tier=0, task_index=5, horizon=6)
    t1 = g.generate(tier=1, task_index=5, horizon=6)
    # Tier 1 only mutates labels; coords / transition semantics unchanged.
    assert t0.ground_truth_trajectory == t1.ground_truth_trajectory
    assert t0.transition_rules == t1.transition_rules
    assert t1.label_mapping is not None  # semantic disruption present


def test_tier2_tier3_modify_ground_truth():
    g = GridTaskGenerator(seed=42)
    t0 = g.generate(tier=0, task_index=5, horizon=6)
    t2 = g.generate(tier=2, task_index=5, horizon=6)
    t3 = g.generate(tier=3, task_index=5, horizon=6)
    # Tier 2/3 change transition semantics -> different trajectory.
    assert t2.ground_truth_trajectory != t0.ground_truth_trajectory
    assert t3.ground_truth_trajectory != t0.ground_truth_trajectory
    # Ground truth must be internally consistent (recomputable from rules).
    assert t2.validate_ground_truth()
    assert t3.validate_ground_truth()
    # Recompute Tier 2 from its own (inverted) rules and confirm equality.
    recomputed = compute_trajectory(
        t2.initial_state, t2.actions, t2.transition_rules,
        t2.grid_width, t2.grid_height,
    )
    assert recomputed == t2.ground_truth_trajectory


def test_parser_rejects_malformed():
    good = extract_model_path("[0,0] -> [1,0] -> [1,1]")
    assert good == ["[0,0]", "[1,0]", "[1,1]"]
    # No coordinates at all -> empty path.
    bad = extract_model_path("the model moved north then east and stopped")
    assert bad == []
    # Partial coordinate -> only the well-formed token is captured.
    partial = extract_model_path("[0,0] [1,0")
    assert partial == ["[0,0]"]


def test_metrics_perfect_partial_wrongfinal():
    gt = ["[0,0]", "[1,0]", "[1,1]", "[2,1]"]
    # Perfect
    r = compare_trajectories(gt, gt)
    assert r.exact_match and r.step_accuracy == 1.0
    # Partial (first two correct, then diverges)
    pred_partial = ["[0,0]", "[1,0]", "[0,0]", "[0,1]"]
    r2 = compare_trajectories(pred_partial, gt)
    assert 0.0 < r2.step_accuracy < 1.0
    assert not r2.exact_match
    # Wrong path but right final state must NOT score 1.0
    pred_wrongfinal = ["[0,0]", "[0,1]", "[1,1]", "[2,1]"]
    r3 = compare_trajectories(pred_wrongfinal, gt)
    assert r3.final_state_correct  # final coord matches
    assert r3.step_accuracy < 1.0  # but intermediate steps wrong
    assert not r3.exact_match


def test_fracture_depth():
    finder = FracturePointFinder(accuracy_threshold=0.5, look_ahead_step=2)
    # Accuracy stays high then collapses and stays low.
    curve = [(1, 1.0), (2, 0.95), (3, 0.9), (4, 0.4), (5, 0.2), (6, 0.1)]
    assert finder.compute_structural_yield(curve) == 4
    # No fracture: always above threshold
    curve_ok = [(1, 1.0), (2, 0.9), (3, 0.8), (4, 0.7)]
    assert finder.compute_structural_yield(curve_ok) is None


def test_full_offline_pipeline():
    """Exercise the runner's offline self-tests (no model required)."""
    class StubEvaluator:
        def __init__(self, *a, **k):
            pass

        def evaluate_single_probe(self, probe, tier, depth=1):
            # Return a perfect trajectory so the runner's scoring path is exercised.
            gt = probe.get("ground_truth_path", [])
            return {
                "raw_output": "->".join(gt),
                "fol_depth": depth,
                "context_exhausted": False,
                "completion_tokens": 0,
                "total_tokens": 0,
                "provider_response": None,
                "finish_reason": "stop",
            }

    runner = BenchmarkRunner(
        model_name="stub",
        evaluator_class=StubEvaluator,
        n_probes=2,
        gravity_levels=[0.0, 1.0],
    )
    base_probes = runner._generate_base_probes()
    # Ground-truth correctness proof + metric consistency (no network).
    runner.run_self_test(base_probes)
    runner.run_metric_unit_tests()
    # Fracture telemetry aggregation from a synthetic matrix.
    finder = FracturePointFinder()
    runner.results_matrix = {
        (3, 0.0, 1, 0): {"step_accuracy": 1.0, "error_mode": ErrorMode.NONE.value},
        (4, 0.0, 1, 0): {
            "step_accuracy": 0.3,
            "error_mode": ErrorMode.STATE_TRACKING_FAILURE.value,
        },
    }
    analysis = finder.parse_matrix_telemetry(runner.results_matrix, 0.0)
    assert analysis["fracture_depth"] == 4
    assert "State Tracking Failure" in analysis["depth_error_distribution"].get(4, {})


def test_full_offline_pipeline_minimal():
    """Minimal offline self-test path (no model, no bisection search)."""
    class StubEvaluator:
        def __init__(self, *a, **k):
            pass

    runner = BenchmarkRunner(
        model_name="stub",
        evaluator_class=StubEvaluator,
        n_probes=1,
        gravity_levels=[0.0],
    )
    base_probes = runner._generate_base_probes()
    # This must not raise; it proves ground truth regenerates from axioms.
    runner.run_self_test(base_probes)
    runner.run_metric_unit_tests()



def test_no_difficulty_metadata_leakage():
    """Audit probes across the standard sweep for tier/gravity/checksum leakage."""
    audit = TaskShortcutAudit(seed=42)
    probes = audit.generate_audit_probes(n_probes=5)
    result = audit.audit_batch(probes)
    assert result["all_clean"], (
        f"Leakage detected in {result['leaky']}/{result['total']} probes: "
        f"{[d for d in result['details'] if not d['clean']]}"
    )


def test_no_difficulty_metadata_leakage_direct():
    """Directly audit probes built by the runner without the DummyEval hack."""
    from arcus.experiments.benchmark_runner import BenchmarkRunner
    runner = BenchmarkRunner(
        model_name="audit-stub",
        evaluator_class=type("NullEval", (), {
            "__init__": lambda self, *a, **k: None,
            "evaluate_single_probe": lambda self, *a, **k: {},
        }),
        n_probes=3,
        gravity_levels=[0.0, 1.0, 2.0, 3.0],
        seed=42,
    )
    probes = []
    for gravity in runner.gravity_levels:
        for i in range(3):
            tier = runner.generator  # placeholder
            from arcus.analysis.gravity import DifficultyConfig
            t = DifficultyConfig().tier_for(gravity)
            task = runner.generator.generate(tier=t, task_index=i, horizon=8)
            probe = runner._build_probe_from_task(task, gravity)
            probes.append(probe)
    audit = TaskShortcutAudit(seed=42)
    result = audit.audit_batch(probes)
    assert result["all_clean"], (
        f"Leakage detected in {result['leaky']}/{result['total']} probes: "
        f"{[d for d in result['details'] if not d['clean']]}"
    )





def test_difficulty_validation_structure():
    """Difficulty validation experiments confirm the difficulty structure.

    Covers requirement #5 / #9: horizon scaling, tier scaling, seed variance,
    and reproducibility are all internally consistent and deterministic.
    """
    dv = DifficultyValidation(seed=42)

    # Horizon scaling: ground-truth length must equal horizon + 1 and stay valid.
    hs = dv.horizon_scaling(tiers=[0, 1, 2, 3], horizons=[4, 8, 12, 16], task_index=0)
    for _tier, by_h in hs.items():
        for _h, data in by_h.items():
            assert data["ground_truth_length"] == data["expected_length"] == _h + 1
            assert data["valid"] is True

    # Tier scaling: Tier 1 preserves the baseline trajectory; Tiers 2/3 differ.
    ts = dv.tier_scaling(gravity_levels=[0.0, 1.0, 2.0, 3.0], task_index=0, horizon=8)
    assert ts[1.0]["trajectory_preserved"] is True
    assert ts[2.0]["trajectory_preserved"] is False
    assert ts[3.0]["trajectory_preserved"] is False
    for _g, data in ts.items():
        assert data["valid"] is True

    # Seed variance: distinct seeds must yield independent tasks.
    sv = dv.seed_variance(seeds=[42, 100, 200, 300], tier=0, task_index=0, horizon=8)
    assert sv["summary"]["all_independent"] is True
    assert sv["summary"]["unique_trajectories"] == sv["summary"]["num_seeds"]

    # Reproducibility: same (seed, tier, task_index) reproduces an identical task.
    rc = dv.reproducibility_check(seed=42, tier=1, task_index=5, horizon=10)
    assert rc["identical_trajectory"] is True
    assert rc["identical_hash"] is True
    assert rc["both_valid"] is True




# ---------------------------------------------------------------------------
# Issue 1: root package import safety (no obsolete top-level imports)
# ---------------------------------------------------------------------------
def test_root_init_executes_without_error():
    """The repository-root __init__.py must import cleanly.

    Regression guard: an earlier version imported obsolete top-level modules
    (``metrics``, ``model_evaluation``, ``benchmark_runner``, ...) that no
    longer exist, which broke ``import`` of the root package and pytest
    collection. The root __init__ is now import-free, so executing it must not
    raise.
    """
    import importlib.util
    root_init = os.path.join(ROOT, "__init__.py")
    spec = importlib.util.spec_from_file_location("__arcus_root_init__", root_init)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # must not raise


def test_arcus_package_imports_resolve_through_arcus():
    """All public symbols resolve through the arcus package."""
    import importlib
    for mod in [
        "arcus",
        "arcus.tasks",
        "arcus.environment",
        "arcus.evaluation",
        "arcus.evaluation.metrics",
        "arcus.evaluation.evaluator",
        "arcus.evaluation.baselines",
        "arcus.analysis",
        "arcus.experiments",
        "arcus.experiments.benchmark_runner",
    ]:
        importlib.import_module(mod)


def test_no_archive_imports_in_active_code():
    """Active code must never import from the legacy ``archive/`` directory.

    Regression guard implemented with the **AST** (not text grep), so comments,
    docstrings and string literals that merely mention "archive" cannot cause
    false positives. Only real ``import archive...`` / ``from archive import ...``
    statements (i.e. ``ast.Import`` / ``ast.ImportFrom`` nodes whose top-level
    package is ``archive``) fail the build.
    """
    import ast

    # Directories that are NOT active ARCUS-X code and must be skipped.
    SKIP_DIRS = {
        "archive", "tests", "outputs", ".pytest_cache", ".vscode",
        "deepeval-main", "FaithEval-master", "lm-evaluation-harness-main",
        "Logic-LLM-main", "__pycache__",
    }

    violations = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # Prune skipped directories so os.walk never descends into them.
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                src = f.read()
            try:
                tree = ast.parse(src, filename=path)
            except SyntaxError:
                # A stray/legacy file must not break the guard; skip it.
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split(".")[0] == "archive":
                        violations.append(f"{path}: from {node.module} import ...")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] == "archive":
                            violations.append(f"{path}: import {alias.name}")

    assert not violations, (
        "Active code must not import from archive/:\n  "
        + "\n  ".join(violations)
    )


# ---------------------------------------------------------------------------
# Issue 2: difficulty metadata must not leak into the model prompt
# ---------------------------------------------------------------------------
FORBIDDEN_PROMPT_TOKENS = [
    "Tier",
    "Gravity",
    "Difficulty",
    "Evaluation Abstraction Rigor Index",
]


def _sample_probe_for_prompt_audit():
    runner = BenchmarkRunner(
        model_name="audit-stub",
        evaluator_class=type("NullEval", (), {
            "__init__": lambda self, *a, **k: None,
            "evaluate_single_probe": lambda self, *a, **k: {},
        }),
        n_probes=1,
        gravity_levels=[2.0],
        seed=42,
    )
    task = runner.generator.generate(tier=2, task_index=0, horizon=6)
    return runner._build_probe_from_task(task, 2.0)


def test_prompt_contains_no_difficulty_metadata():
    """Generated prompts must not expose tier/gravity/difficulty/hashes."""
    probe = _sample_probe_for_prompt_audit()

    # The model-facing prompt is built only from axioms + question.
    from arcus.evaluation.evaluator import ModelEvaluator as EvalEvaluator
    ev = EvalEvaluator.__new__(EvalEvaluator)
    prompt = ev.build_prompt(probe)
    for token in FORBIDDEN_PROMPT_TOKENS:
        assert token not in prompt, f"Leakage: {token!r} found in prompt"
    # No numeric gravity value should appear in the prompt text.
    assert "2.0" not in prompt, "Gravity value leaked into prompt"
    assert "gravity" not in prompt.lower(), "Gravity leaked into prompt"



def test_evaluation_retains_tier_gravity_internally():
    """Internal evaluator metadata (tier/gravity/seed/hash) is preserved."""
    probe = _sample_probe_for_prompt_audit()
    assert probe.get("tier") == 2
    assert probe.get("gravity_target") == 2.0
    assert "seed" in probe
    assert "id" in probe  # task_id is carried internally as "id"
    assert "experiment_hash" in probe


# ---------------------------------------------------------------------------
# Issue 3: baseline evaluator correctness
# ---------------------------------------------------------------------------
def test_oracle_baseline_perfect_accuracy():
    """Oracle baseline reproduces the exact ground-truth trajectory (100%)."""
    from arcus.evaluation.baselines import OracleBaseline
    baseline = OracleBaseline()
    gen = GridTaskGenerator(seed=42)
    for i in range(5):
        task = gen.generate(tier=1, task_index=i, horizon=6)
        res = baseline.evaluate(task)
        assert res["exact_match"] is True
        assert res["step_accuracy"] == 1.0
        assert res["trajectory_accuracy"] == 1.0


def test_random_baseline_near_chance():
    """Random baseline performs near chance (low trajectory accuracy)."""
    from arcus.evaluation.baselines import RandomBaseline
    baseline = RandomBaseline(seed=12345)
    gen = GridTaskGenerator(seed=7)
    traj_accs = []
    for i in range(10):
        task = gen.generate(tier=0, task_index=i, horizon=8)
        res = baseline.evaluate(task)
        traj_accs.append(res["trajectory_accuracy"])
    # Exact-match rate should be ~0 (chance of matching a full trajectory).
    assert sum(traj_accs) / len(traj_accs) < 0.5


def test_initial_state_baseline_poor_score():
    """Initial-state baseline scores poorly on trajectory metrics."""
    from arcus.evaluation.baselines import InitialStateBaseline
    baseline = InitialStateBaseline()
    gen = GridTaskGenerator(seed=42)
    task = gen.generate(tier=0, task_index=0, horizon=8)
    res = baseline.evaluate(task)
    assert res["exact_match"] is False
    assert res["trajectory_accuracy"] == 0.0
    assert res["step_accuracy"] < 0.5


def test_benchmark_runner_run_baselines():
    """BenchmarkRunner.run_baselines aggregates the three baselines correctly."""
    runner = BenchmarkRunner(
        model_name="baseline-stub",
        evaluator_class=type("NullEval", (), {
            "__init__": lambda self, *a, **k: None,
            "evaluate_single_probe": lambda self, *a, **k: {},
        }),
        n_probes=3,
        gravity_levels=[0.0],
        seed=42,
    )
    report = runner.run_baselines(n_tasks=4, horizon=6, gravity_levels=[0.0])
    assert set(report.keys()) == {"oracle", "random", "initial_state"}
    for name, entry in report.items():
        assert entry["baseline_name"] == name
        assert "trajectory_accuracy" in entry
        assert "step_accuracy" in entry
        assert "exact_match" in entry
        assert entry["fracture_depth"] is None
    # Oracle must be at the upper bound; initial_state at the lower bound.
    assert report["oracle"]["trajectory_accuracy"] == 100.0
    assert report["oracle"]["exact_match"] == 100.0
    assert report["initial_state"]["exact_match"] == 0.0


# ---------------------------------------------------------------------------
# CRI (ARCUS Robustness Index) -- secondary aggregate robustness indicator
# ---------------------------------------------------------------------------
def test_cri_perfect_oracle():
    """Perfect oracle: every component is 1.0 -> CRI ~= 1.0."""
    summary = ARCUSRobustnessIndex.compute(
        step_accuracy=1.0,
        continuity_score=1.0,
        exact_match=1.0,
        horizon_points=[(1, 1.0), (2, 1.0), (3, 1.0)],
        tier0_acc=1.0, tier1_acc=1.0, tier2_acc=1.0, tier3_acc=1.0,
        generation_efficiency=1.0,
    )
    assert summary.cri == 1.0
    assert summary.trajectory_fidelity == 1.0
    assert summary.horizon_robustness == 1.0
    assert summary.semantic_robustness == 1.0
    assert summary.generation_efficiency == 1.0


def test_cri_random_baseline_lower():
    """Random baseline: CRI significantly lower than the perfect oracle."""
    perfect = ARCUSRobustnessIndex.compute(
        step_accuracy=1.0, continuity_score=1.0, exact_match=1.0,
        horizon_points=[(1, 1.0), (2, 1.0), (3, 1.0)],
        tier0_acc=1.0, tier1_acc=1.0, tier2_acc=1.0, tier3_acc=1.0,
        generation_efficiency=1.0,
    )
    random_cri = ARCUSRobustnessIndex.compute(
        step_accuracy=0.1, continuity_score=0.1, exact_match=0.0,
        horizon_points=[(1, 0.1), (2, 0.1), (3, 0.1)],
        tier0_acc=0.1, tier1_acc=0.1, tier2_acc=0.1, tier3_acc=0.1,
        generation_efficiency=0.5,
    )
    assert random_cri.cri < 0.5 * perfect.cri


def test_cri_wrong_trajectory_right_final_penalized():
    """Wrong intermediate path but correct final state is penalized.

    A trajectory that reaches the right final coordinate via wrong steps has
    low step accuracy / continuity, so CRI must be lower than a correct one.
    """
    correct = ARCUSRobustnessIndex.compute(
        step_accuracy=1.0, continuity_score=1.0, exact_match=1.0,
        horizon_points=[(1, 1.0), (2, 1.0), (3, 1.0)],
        tier0_acc=1.0, tier1_acc=1.0, tier2_acc=1.0, tier3_acc=1.0,
        generation_efficiency=1.0,
    )
    # Wrong path but right final: low step accuracy / continuity, no exact match.
    wrong_final = ARCUSRobustnessIndex.compute(
        step_accuracy=0.25, continuity_score=0.25, exact_match=0.0,
        horizon_points=[(1, 0.25), (2, 0.25), (3, 0.25)],
        tier0_acc=0.25, tier1_acc=0.25, tier2_acc=0.25, tier3_acc=0.25,
        generation_efficiency=1.0,
    )
    assert wrong_final.cri < correct.cri


def test_cri_missing_efficiency_still_computes():
    """Missing generation-efficiency data: CRI still computes from primaries."""
    summary = ARCUSRobustnessIndex.compute(
        step_accuracy=0.8, continuity_score=0.7, exact_match=0.6,
        horizon_points=[(1, 0.8), (2, 0.7), (3, 0.6)],
        tier0_acc=0.8, tier1_acc=0.8, tier2_acc=0.5, tier3_acc=0.4,
        generation_efficiency=None,
    )
    # GE defaults to 0.0; CRI is driven by the primary components.
    assert 0.0 <= summary.cri <= 1.0
    assert summary.generation_efficiency == 0.0
    # With strong TF/HR/SR the CRI should be meaningfully above 0.
    assert summary.cri > 0.0


def test_cri_discrete_horizon_nonuniform():
    """Discrete horizon robustness works with non-uniform depth spacing."""
    # Non-uniform depths: z = 1, 3, 7 with accuracies 1.0, 0.5, 0.0.
    # Trapezoid: (1+0.5)/2*(3-1) + (0.5+0)/2*(7-3) = 1.5 + 1.0 = 2.5
    # Normalized by (7-1) = 6 -> HR = 2.5/6.
    hr = ARCUSRobustnessIndex.horizon_robustness([(1, 1.0), (3, 0.5), (7, 0.0)])
    assert abs(hr - (2.5 / 6.0)) < 1e-9

    # Missing point (None) is skipped safely: only (1, 1.0) and (7, 0.0) remain,
    # giving a single trapezoid of area (1.0+0.0)/2 * (7-1) = 3.0 -> HR = 0.5.
    hr2 = ARCUSRobustnessIndex.horizon_robustness([(1, 1.0), (3, None), (7, 0.0)])
    assert abs(hr2 - 0.5) < 1e-9

    # Single depth returns the mean accuracy.
    hr3 = ARCUSRobustnessIndex.horizon_robustness([(5, 0.4)])
    assert abs(hr3 - 0.4) < 1e-9


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # pragma: no cover
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

