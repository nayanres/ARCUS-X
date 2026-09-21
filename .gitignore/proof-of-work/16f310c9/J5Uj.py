import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcus.tasks.generator import GridTaskGenerator
from arcus.evaluation.evaluator import ModelEvaluator

ev = ModelEvaluator(model_name="stub", api_base="https://openrouter.ai/api/v1",
                    api_key="dummy", use_vllm=False)

def prompt_of(gen, tier, idx, h):
    t = gen.generate(tier=tier, task_index=idx, horizon=h)
    return t.question, t.axioms

# 1) Determinism: same seed/tier/index -> identical prompt
g1 = GridTaskGenerator(seed=42)
g2 = GridTaskGenerator(seed=42)
q1, a1 = prompt_of(g1, 2, 3, 6)
q2, a2 = prompt_of(g2, 2, 3, 6)
print("DETERMINISTIC:", q1 == q2 and a1 == a2)

# 2) Uniqueness across task_index (different instances -> different prompts)
prompts = set()
for i in range(8):
    q, a = prompt_of(g1, 0, i, 6)
    prompts.add((tuple(a), q))
print("UNIQUE_ACROSS_INDEX (8 instances -> distinct prompts):", len(prompts) == 8)

# 3) Data-driven: axioms reflect the task's own grid/initial/actions, not a constant
t = g1.generate(tier=0, task_index=5, horizon=6)
print("AXIOM_GRID_LINE:", t.axioms[0])
print("AXIOM_INIT_LINE:", t.axioms[1])
print("AXIOM_ACTIONSEQ_LINE:", [x for x in t.axioms if x.startswith("The agent performs")][0])
print("QUESTION_MENTIONS_HORIZON:", f"{t.horizon} coordinates" in t.question)

# 4) Tier 1 uses pseudowords (semantic disruption) distinct from base labels
t0 = g1.generate(tier=0, task_index=5, horizon=6)
t1 = g1.generate(tier=1, task_index=5, horizon=6)
print("TIER1_PSEUDOWORDS_DIFFER:", t0.axioms != t1.axioms)
print("TIER1_RULE_USES_PSEUDOWORD:", any("'" in line and "moves by" in line for line in t1.axioms))
