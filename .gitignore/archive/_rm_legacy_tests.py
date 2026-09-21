import io, os

p = os.path.join("tests", "test_arcus_hardening.py")
with io.open(p, "r", encoding="utf-8") as f:
    src = f.read()

# 1) Remove the three legacy import lines.
old_imports = (
    "from arcus.experiments.task_shortcut_audit import TaskShortcutAudit\n"
    "from arcus.evaluation.oracle_solver import OracleSolver\n"
    "from arcus.evaluation.random_baseline_evaluator import RandomBaselineEvaluator\n"
    "from arcus.evaluation.initial_state_baseline import InitialStateBaseline\n"
    "from arcus.experiments.difficulty_validation import DifficultyValidation\n"
)
new_imports = (
    "from arcus.experiments.task_shortcut_audit import TaskShortcutAudit\n"
    "from arcus.experiments.difficulty_validation import DifficultyValidation\n"
)
assert old_imports in src, "import block not found"
src = src.replace(old_imports, new_imports)

# 2) Remove legacy test_oracle_solver_perfect_accuracy (uses OracleSolver).
old_oracle = (
    "def test_oracle_solver_perfect_accuracy():\n"
    "    \"\"\"The oracle solver must reproduce the exact ground-truth trajectory.\"\"\"\n"
    "    solver = OracleSolver(model_name=\"oracle-stub\", api_base=\"https://openrouter.ai/api/v1\",\n"
    "                          api_key=\"dummy\")\n"
    "    # Build a probe directly from a generated task (no model call needed).\n"
    "    gen = GridTaskGenerator(seed=42)\n"
    "    task = gen.generate(tier=1, task_index=0, horizon=6)\n"
    "    probe = {\n"
    "        \"initial_state\": list(task.initial_state),\n"
    "        \"actions\": list(task.actions),\n"
    "        \"transition_rules\": {k: dict(v) for k, v in task.transition_rules.items()},\n"
    "        \"grid_width\": task.grid_width,\n"
    "        \"grid_height\": task.grid_height,\n"
    "    }\n"
    "    solution = solver.solve_perfect(probe)\n"
    "    # The solution must contain the full ground-truth path joined by '->'.\n"
    "    gt_path = \"->\".join(task.ground_truth_trajectory)\n"
    "    assert gt_path in solution\n"
    "    # And the checksum must match the GT final coordinate.\n"
    "    import re\n"
    "    last = [int(n) for n in re.findall(r\"\\d+\", task.ground_truth_trajectory[-1])]\n"
    "    expected_checksum = last[0] * 13 + last[1] * 7\n"
    "    assert f\"CHECKSUM: {expected_checksum}\" in solution\n"
    "\n"
    "\n"
    "def test_random_baseline_near_chance():\n"
)
new_oracle = "def test_random_baseline_near_chance():\n"
assert old_oracle in src, "oracle test block not found"
src = src.replace(old_oracle, new_oracle)

# 3) Remove legacy test_initial_state_baseline_low_score (uses InitialStateBaseline).
old_init = (
    "def test_initial_state_baseline_low_score():\n"
    "    \"\"\"Initial-state-only output must score ~0 on trajectory metrics.\"\"\"\n"
    "    gen = GridTaskGenerator(seed=42)\n"
    "    task = gen.generate(tier=0, task_index=0, horizon=8)\n"
    "    baseline = InitialStateBaseline(model_name=\"init-stub\", api_base=\"https://openrouter.ai/api/v1\",\n"
    "                                     api_key=\"dummy\")\n"
    "    probe = {\n"
    "        \"initial_state\": list(task.initial_state),\n"
    "    }\n"
    "    solution = baseline.solve_initial_only(probe)\n"
    "    # The solution only contains the initial coordinate, not the full path.\n"
    "    assert solution.startswith(f\"[{task.initial_state[0]},{task.initial_state[1]}]\")\n"
    "    # It cannot match the full ground-truth trajectory.\n"
    "    r = compare_trajectories([f\"[{task.initial_state[0]},{task.initial_state[1]}]\"],\n"
    "                              task.ground_truth_trajectory)\n"
    "    assert r.step_accuracy < 0.5\n"
    "    assert not r.exact_match\n"
    "\n"
    "\n"
    "def test_difficulty_validation_structure():\n"
)
new_init = "def test_difficulty_validation_structure():\n"
assert old_init in src, "initial-state test block not found"
src = src.replace(old_init, new_init)

with io.open(p, "w", encoding="utf-8") as f:
    f.write(src)
print("patched test file")
