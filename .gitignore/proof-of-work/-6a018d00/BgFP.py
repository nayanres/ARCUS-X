#!/usr/bin/env python3

# Task Shortcut Audit for ARCUS-X
# Audits generated tasks for potential shortcut leakage that could inflate scores

import re
from typing import Dict, List, Any
from arcus.tasks.generator import GridTaskGenerator
from arcus.analysis.gravity import DifficultyConfig

class TaskShortcutAudit:
    """Audits tasks for metadata leakage and shortcut vulnerabilities.
    
    IMPORTANT: The audit only inspects the *prompt content* (axioms + question)
    that is actually sent to the model. The model prompt must contain ONLY the
    task axioms/rules, action/state information, and the question/request. It
    must NEVER contain a checksum, a final-coordinate constraint, a hidden-answer
    pattern, or any difficulty metadata. The runner's `_build_probe_from_task`
    builds the prompt from those elements alone; scoring is performed downstream
    by the trajectory metrics from `ground_truth_path` and is never concatenated
    into the prompt.

    As a regression guard, this audit explicitly scans the prompt for checksum
    values, checksum formulas (e.g. ``(X_final * 13) + (Y_final * 7)``), final
    coordinate constraints, and hidden-answer patterns, and fails the probe if
    any are present.
    """

    # Patterns that should NEVER appear in the model prompt (axioms + question).
    # Group 1: difficulty / internal metadata leakage.
    LEAKAGE_PATTERNS = [
        r"tier\s*=\s*\d+",
        r"gravity\s*=\s*[\d.]+",
        r"semantic\s+disruption",
        r"attribute\s+inversion",
        r"axiomatic\s+contradiction",
        r"difficulty\s*[:=]",
        r"expected\s+answer",
        r"ground\s*truth",
        r"experiment_hash",
        r"task_id",
    ]

    # Group 2: hidden-answer / final-state constraint leakage. These are
    # regression patterns that must never appear in a model-facing prompt.
    HIDDEN_ANSWER_PATTERNS = [
        r"checksum",
        r"x_final\s*\*\s*13",
        r"y_final\s*\*\s*7",
        r"\(\s*x[^)]*\)\s*\+\s*\(\s*y[^)]*\)",
        r"terminal spatial verification",
        r"final coordinate",
        r"terminal coordinate",
        r"anomaly equation",
        r"CHECKSUM:",
        r"\bINVALID\b",
    ]
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.generator = GridTaskGenerator(seed=seed)
    
    def audit_probe(self, probe: Dict[str, Any]) -> Dict[str, Any]:
        """Audit a single probe for shortcut leakage in the *prompt* content."""
        issues = []
        
        # Check ONLY the prompt content (axioms + question) that goes to the model.
        prompt_text = "\n".join([
            str(probe.get("axioms", "")),
            str(probe.get("question", "")),
        ])
        
        for pattern in self.LEAKAGE_PATTERNS:
            if re.search(pattern, prompt_text, re.IGNORECASE):
                issues.append(f"Leakage pattern found: {pattern}")

        # Hidden-answer / final-state constraint regression scan.
        for pattern in self.HIDDEN_ANSWER_PATTERNS:
            if re.search(pattern, prompt_text, re.IGNORECASE):
                issues.append(f"Hidden-answer/checksum pattern found in prompt: {pattern}")
        
        # Check that tier name is not in the prompt
        tier_name = probe.get("tier_name", "")
        if tier_name and tier_name.lower() in prompt_text.lower():
            issues.append(f"Tier name leaked in prompt: {tier_name}")
        
        # Check that experiment hash is not in the prompt
        exp_hash = probe.get("experiment_hash", "")
        if exp_hash and exp_hash in prompt_text:
            issues.append(f"Experiment hash leaked in prompt: {exp_hash}")
        
        # Check that internal IDs are not in the prompt
        task_id = probe.get("id", "")
        if task_id and task_id in prompt_text:
            issues.append(f"Internal task ID leaked in prompt: {task_id}")
        
        # Check output length is not specified in prompt
        if re.search(r"\d+\s*(steps|coordinates|outputs)", prompt_text, re.IGNORECASE):
            # This is allowed if it's part of the action sequence description
            # but flag if it looks like a length hint
            if re.search(r"generate\s+(exactly\s+)?\d+", prompt_text, re.IGNORECASE):
                issues.append("Output length hint found in prompt")
        
        return {
            "probe_id": probe.get("id"),
            "tier": probe.get("tier"),
            "has_issues": len(issues) > 0,
            "issues": issues,
            "clean": len(issues) == 0,
        }
    
    def audit_batch(self, probes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Audit a batch of probes."""
        results = [self.audit_probe(p) for p in probes]
        clean_count = sum(1 for r in results if r["clean"])
        return {
            "total": len(results),
            "clean": clean_count,
            "leaky": len(results) - clean_count,
            "all_clean": clean_count == len(results),
            "details": results,
        }
    
    def generate_audit_probes(self, n_probes: int = 10,
                              gravity_levels: List[float] = [0.0, 1.0, 2.0, 3.0]) -> List[Dict]:
        """Generate a set of probes for auditing."""
        from arcus.experiments.benchmark_runner import BenchmarkRunner
        # Use a null evaluator that never calls a model; we only need the probe
        # construction logic from the runner.
        NullEval = type("NullEval", (), {
            "__init__": lambda self, *a, **k: None,
            "evaluate_single_probe": lambda self, *a, **k: {},
        })
        runner = BenchmarkRunner(
            model_name="audit",
            evaluator_class=NullEval,
            n_probes=n_probes,
            gravity_levels=gravity_levels,
            seed=self.seed,
        )
        
        probes = []
        for gravity in gravity_levels:
            for i in range(n_probes):
                tier = DifficultyConfig().tier_for(gravity)
                task = self.generator.generate(tier=tier, task_index=i, horizon=8)
                probe = runner._build_probe_from_task(task, gravity, "TRUE")
                probes.append(probe)
        return probes


if __name__ == "__main__":
    audit = TaskShortcutAudit(seed=42)
    probes = audit.generate_audit_probes(n_probes=5)
    result = audit.audit_batch(probes)
    print(f"Audit result: {result['clean']}/{result['total']} clean")
    if not result["all_clean"]:
        for detail in result["details"]:
            if not detail["clean"]:
                print(f"  Leaky probe {detail['probe_id']}: {detail['issues']}")
