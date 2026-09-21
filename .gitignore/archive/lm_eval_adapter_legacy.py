from lm_eval.api.task import ConfigurableTask
from lm_eval.api.registry import register_task
try:
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.api.registry import register_task
except ImportError:
    ConfigurableTask = object
    def register_task(name): return lambda cls: cls

from benchmark_runner import BenchmarkRunner

@register_task("framework_v24_sequence_endurance")
class FrameworkV24Task(ConfigurableTask):
    """
    Plugs Framework v2.4 directly into the EleutherAI LM Evaluation Harness.
    Exposes E_yield and TSA bounds tracking via a single CLI execution command.
    """
    VERSION = "2.4"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize your hardened benchmark engine
        self.runner = BenchmarkRunner(hard_ceiling=150)

    def has_training_docs(self): return False
    def has_validation_docs(self): return True
    def has_test_docs(self): return False

    def validation_docs(self):
        # Generate the test instances using your OOD generator
        from generator import GeneratorCore, OODTier
        generator = GeneratorCore()
        return generator.process_faitheval_batch(batch_size=5, max_examples=50)

    def doc_to_text(self, doc):
        return f"Execute State Path Tracking for Seed {doc['structural_isomorphic_hash']}. Begin sequence extraction."

    def process_results(self, doc, results):
        # Tie into the model evaluation parser
        raw_output = results[0]
        # Run your dynamic bisection search logic to calculate yield limits
        metrics = self.runner._evaluate_depth_batch(raw_output, doc)
        return {
            "E_yield": metrics.get("accuracy", 0.0),
            "tsa_violations": 0.0 if metrics.get("accuracy", 0.0) > 0.0 else 1.0
        }

    def aggregation(self):
        return {
            "E_yield": lambda x: sum(x) / len(x),
            "tsa_violations": sum
        }

    def higher_is_better(self):
        return {"E_yield": True, "tsa_violations": False}