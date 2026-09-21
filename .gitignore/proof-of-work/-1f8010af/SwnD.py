import sys
from arcus.tasks.generator import GridTaskGenerator
from arcus.evaluation.evaluator import ModelEvaluator
from arcus.experiments.task_shortcut_audit import TaskShortcutAudit

gen = GridTaskGenerator(seed=42)
ev = ModelEvaluator(model_name="stub", api_base="https://openrouter.ai/api/v1",
                    api_key="dummy", use_vllm=False)
audit = TaskShortcutAudit(seed=42)

out = []
for tier in (0, 1, 2, 3):
    t = gen.generate(tier=tier, task_index=0, horizon=6)
    probe = {
        "id": t.task_id,
        "tier": t.tier,
        "question": t.question,
        "axioms": t.axioms,
        "tier_name": t.metadata.get("tier_name"),
        "experiment_hash": t.experiment_hash,
    }
    a = audit.audit_probe(probe)
    out.append(
        "Tier %d: axioms=%d q=%s p=%s clean=%s issues=%s"
        % (tier, len(t.axioms), bool(t.question),
           bool(ev.build_prompt(probe).strip()), a["clean"], a["issues"])
    )

with open("outputs/_verify_prompt.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))

print("WROTE", len(out))
sys.stdout.flush()
