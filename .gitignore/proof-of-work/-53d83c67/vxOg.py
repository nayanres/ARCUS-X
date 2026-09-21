import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcus.tasks.generator import GridTaskGenerator
from arcus.evaluation.evaluator import ModelEvaluator

gen = GridTaskGenerator(seed=42)
ev = ModelEvaluator(model_name="stub", api_base="https://openrouter.ai/api/v1",
                    api_key="dummy", use_vllm=False)

t = gen.generate(tier=1, task_index=0, horizon=6)
probe = {"id": t.task_id, "tier": t.tier, "question": t.question, "axioms": t.axioms}
print("=== Tier 1 prompt (pseudoword labels) ===")
print(ev.build_prompt(probe))
