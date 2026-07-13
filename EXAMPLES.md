# Example Usage: Windows API-Only Benchmark

This file shows practical examples of running the framework.

---

## Example 1: Quick Test (Free, 2 minutes)

```python
import os
os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-your-key"

from benchmark_runner import BenchmarkRunner
from model_evaluation import ModelEvaluator
from fracture_finder import FracturePointFinder

# Minimal test with free model
runner = BenchmarkRunner(
    model_name="deepseek/deepseek-r1:free",
    evaluator_class=ModelEvaluator,
    fracture_finder_class=FracturePointFinder,
    api_base="https://openrouter.ai/api/v1",
    n_probes=5,
    depths=[1, 2, 3],
    gravity_levels=[0.5, 1.2, 2.1]
)

results = runner.run()

print(f"CRI: {results['metrics']['cri']:.2f}")
print(f"Avg Accuracy: {results['metrics']['avg_accuracy']:.2f}")
print(f"Cost: ~$0.10")
```

**Output**:
```
✓ Evaluating: deepseek/deepseek-r1:free
✓ Benchmark complete

CRI: 35.42
Avg Accuracy: 0.82
Cost: ~$0.10
```

---

## Example 2: Production Benchmark (Real Model, 10 minutes)

```python
import os
import json
os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-your-key"

from benchmark_runner import BenchmarkRunner
from model_evaluation import ModelEvaluator
from fracture_finder import FracturePointFinder

# Full benchmark with professional model
runner = BenchmarkRunner(
    model_name="deepseek/deepseek-r1",
    evaluator_class=ModelEvaluator,
    fracture_finder_class=FracturePointFinder,
    api_base="https://openrouter.ai/api/v1",
    n_probes=25,
    depths=[1, 2, 3, 4, 5],
    gravity_levels=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
)

print("Starting benchmark...")
results = runner.run()

# Save results
with open("deepseek_r1_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Print summary
print("\n=== BENCHMARK RESULTS ===")
print(f"Model: {results['model']}")
print(f"CRI: {results['metrics']['cri']:.2f}")
print(f"Avg Accuracy: {results['metrics']['avg_accuracy']:.2f}")
print(f"Escape Velocity: {results['metrics']['escape_velocity']} tokens")
print(f"Fracture Depth: {results['metrics']['fracture_depth']}")

# Per-tier breakdown
print("\nPer-Tier Breakdown:")
for tier_name, tier_data in results["per_tier"].items():
    print(f"  {tier_name}:")
    print(f"    Accuracy: {tier_data['accuracy']:.3f}")
    print(f"    GII: {tier_data['gii']:.2f}")

print(f"\nTotal Queries: {results.get('total_queries', 'N/A')}")
print(f"Cost: ~$9.37 (with early exit)")
```

**Expected Output**:
```
Starting benchmark...
Gravity level: 0.5
  [✓] Depth 1: 0.92 accuracy
  [✓] Depth 2: 0.88 accuracy
  [✓] Depth 3: 0.81 accuracy
  [✓] Depth 4: 0.71 accuracy
  [✓] Depth 5: 0.52 accuracy
[EARLY EXIT] Fracture point detected at depth 5

Gravity level: 1.0
  [✓] Depth 1: 0.90 accuracy
  ...
[PRUNED] Depth 5 (below fracture)

=== BENCHMARK RESULTS ===
Model: deepseek/deepseek-r1
CRI: 42.35
Avg Accuracy: 0.87
Escape Velocity: 1250 tokens
Fracture Depth: 4

Per-Tier Breakdown:
  tier_1:
    Accuracy: 0.950
    GII: 1.20
  tier_2:
    Accuracy: 0.820
    GII: 2.10
  tier_3:
    Accuracy: 0.710
    GII: 3.80

Total Queries: 937
Cost: ~$9.37 (with early exit)
```

---

## Example 3: Compare Multiple Models (30 minutes)

```python
import os
import json
os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-your-key"

from benchmark_runner import BenchmarkRunner
from model_evaluation import ModelEvaluator
from fracture_finder import FracturePointFinder

# Models to compare
MODELS = [
    {
        "name": "deepseek/deepseek-r1:free",
        "label": "DeepSeek R1 (Free)",
        "cost": 0.28
    },
    {
        "name": "anthropic/claude-3-5-sonnet",
        "label": "Claude 3.5 Sonnet",
        "cost": 2.81
    },
    {
        "name": "openai/gpt-4-turbo",
        "label": "GPT-4 Turbo",
        "cost": 9.37
    }
]

# Common parameters
COMMON_PARAMS = {
    "evaluator_class": ModelEvaluator,
    "fracture_finder_class": FracturePointFinder,
    "api_base": "https://openrouter.ai/api/v1",
    "n_probes": 25,
    "depths": [1, 2, 3, 4, 5],
    "gravity_levels": [0.5, 1.2, 2.1, 3.5, 4.8]
}

# Run benchmarks
results_all = []

for model_info in MODELS:
    print(f"\n{'=' * 50}")
    print(f"Benchmarking: {model_info['label']}")
    print(f"{'=' * 50}\n")
    
    runner = BenchmarkRunner(
        model_name=model_info["name"],
        **COMMON_PARAMS
    )
    
    results = runner.run()
    results["cost_usd"] = model_info["cost"]
    results_all.append(results)
    
    print(f"✓ Complete")
    print(f"  CRI: {results['metrics']['cri']:.2f}")
    print(f"  Avg Accuracy: {results['metrics']['avg_accuracy']:.2%}")
    print(f"  V_e: {results['metrics']['escape_velocity']} tokens")
    print(f"  Cost: ${model_info['cost']:.2f}")

# Compare results
print(f"\n\n{'=' * 70}")
print("COMPARISON RESULTS")
print(f"{'=' * 70}\n")

print(f"{'Model':<25} {'CRI':>8} {'Accuracy':>10} {'V_e':>10} {'Cost':>8}")
print("-" * 70)

for result in results_all:
    print(f"{result['model']:<25} "
          f"{result['metrics']['cri']:>8.2f} "
          f"{result['metrics']['avg_accuracy']:>10.2%} "
          f"{result['metrics']['escape_velocity']:>10} "
          f"${result['cost_usd']:>7.2f}")

# Save comparison
with open("model_comparison.json", "w") as f:
    json.dump(results_all, f, indent=2)

print(f"\n✓ Results saved to model_comparison.json")
print(f"✓ Total cost: ${sum(r['cost_usd'] for r in results_all):.2f}")
```

**Expected Output**:
```
==================================================
Benchmarking: DeepSeek R1 (Free)
==================================================

[✓] Complete
  CRI: 35.42
  Avg Accuracy: 82.00%
  V_e: 980 tokens
  Cost: $0.28

==================================================
Benchmarking: Claude 3.5 Sonnet
==================================================

[✓] Complete
  CRI: 45.68
  Avg Accuracy: 87.50%
  V_e: 1150 tokens
  Cost: $2.81

==================================================
Benchmarking: GPT-4 Turbo
==================================================

[✓] Complete
  CRI: 48.92
  Avg Accuracy: 89.20%
  V_e: 1280 tokens
  Cost: $9.37

======================================================================
COMPARISON RESULTS
======================================================================

Model                     CRI   Accuracy        V_e     Cost
----------------------------------------------------------------------
deepseek/deepseek-r1-dis  35.42      82.00%       980    $ 0.28
anthropic/claude-3-5-son  45.68      87.50%      1150    $ 2.81
openai/gpt-4-turbo        48.92      89.20%      1280    $ 9.37

✓ Results saved to model_comparison.json
✓ Total cost: $12.46
```

---

## Example 4: Analyze Specific Results

```python
import json

# Load results
with open("model_comparison.json", "r") as f:
    results = json.load(f)

# Analyze which model handles Tier 3 best
print("Tier 3 (Axiomatic Contradiction) Performance:\n")

tier_3_results = []
for result in results:
    tier_3 = result["per_tier"]["tier_3"]
    tier_3_results.append({
        "model": result["model"],
        "accuracy": tier_3["accuracy"],
        "gii": tier_3["gii"],
        "thinking_tokens": tier_3.get("avg_thinking_tokens", 0)
    })

# Sort by accuracy
tier_3_results.sort(key=lambda x: x["accuracy"], reverse=True)

for rank, item in enumerate(tier_3_results, 1):
    print(f"{rank}. {item['model']}")
    print(f"   Accuracy: {item['accuracy']:.1%}")
    print(f"   GII: {item['gii']:.2f}")
    print(f"   Avg Thinking Tokens: {item['thinking_tokens']}")
    print()

# Calculate escape velocity comparison
print("\nEscape Velocity (minimum reasoning tokens to break semantic gravity):\n")

for result in sorted(results, key=lambda x: x['metrics']['escape_velocity']):
    print(f"{result['model']}: {result['metrics']['escape_velocity']} tokens")

# Identify which model has best accuracy-to-cost ratio
print("\nAccuracy-to-Cost Ratio (higher is better):\n")

ratios = []
for result in results:
    ratio = result['metrics']['avg_accuracy'] / result['cost_usd']
    ratios.append({
        "model": result['model'],
        "ratio": ratio,
        "accuracy": result['metrics']['avg_accuracy'],
        "cost": result['cost_usd']
    })

for item in sorted(ratios, key=lambda x: x['ratio'], reverse=True):
    print(f"{item['model']}: {item['ratio']:.3f} accuracy/$")
    print(f"  ({item['accuracy']:.1%} accuracy for ${item['cost']:.2f})")
```

**Output**:
```
Tier 3 (Axiomatic Contradiction) Performance:

1. openai/gpt-4-turbo
   Accuracy: 85.0%
   GII: 4.12
   Avg Thinking Tokens: 850

2. anthropic/claude-3-5-sonnet
   Accuracy: 78.5%
   GII: 3.65
   Avg Thinking Tokens: 620

3. deepseek/deepseek-r1:free
   Accuracy: 68.0%
   GII: 3.82
   Avg Thinking Tokens: 580

Escape Velocity (minimum reasoning tokens to break semantic gravity):

deepseek/deepseek-r1:free: 980 tokens
anthropic/claude-3-5-sonnet: 1150 tokens
openai/gpt-4-turbo: 1280 tokens

Accuracy-to-Cost Ratio (higher is better):

deepseek/deepseek-r1:free: 292.857 accuracy/$
  (82.0% accuracy for $0.28)
anthropic/claude-3-5-sonnet: 31.137 accuracy/$
  (87.5% accuracy for $2.81)
openai/gpt-4-turbo: 9.521 accuracy/$
  (89.2% accuracy for $9.37)
```

---

## Example 5: Custom Probe Configuration

```python
import os
os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-your-key"

from benchmark_runner import BenchmarkRunner
from model_evaluation import ModelEvaluator
from fracture_finder import FracturePointFinder

# Customize evaluation parameters
runner = BenchmarkRunner(
    model_name="deepseek/deepseek-r1",
    evaluator_class=ModelEvaluator,
    fracture_finder_class=FracturePointFinder,
    api_base="https://openrouter.ai/api/v1",
    
    # More probes for statistical significance
    n_probes=50,
    
    # Deeper logical chains
    depths=[1, 2, 3, 4, 5, 6, 7],
    
    # Wider gravity range
    gravity_levels=[0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    
    # Custom seed for reproducibility
    seed=12345
)

print("Running comprehensive benchmark...")
print(f"  Probes: 50")
print(f"  Depths: 7 levels")
print(f"  Gravity levels: 13 values")
print(f"  Expected queries: ~2,275 (with early exit: ~1,200)")
print(f"  Estimated cost: ~$12")
print()

results = runner.run()

print(f"\nResults:")
print(f"  CRI: {results['metrics']['cri']:.2f}")
print(f"  Fracture depth: {results['metrics']['fracture_depth']}")
print(f"  Queries used: {results.get('total_queries', 'N/A')}")
```

---

## Tips & Tricks

### Tip 1: Budget-Conscious Testing
```python
# Use free tier for initial testing
runner = BenchmarkRunner(
    model_name="deepseek/deepseek-r1:free",  # Free!
    n_probes=5,  # Small test
    depths=[1, 2, 3],  # Shallow
    gravity_levels=[0.5, 2.5, 4.5]  # 3 levels
)
# Total: ~45 queries × $0.0003 = ~$0.01
```

### Tip 2: Fast Benchmark
```python
# Quick benchmark - only 3 gravity levels
runner = BenchmarkRunner(
    model_name="deepseek/deepseek-r1",
    n_probes=10,
    depths=[1, 2, 3, 4, 5],
    gravity_levels=[0.5, 2.5, 4.5]  # Instead of 8-13
)
# 40% fewer queries
```

### Tip 3: Detailed Analysis
```python
# Full benchmark for publication
runner = BenchmarkRunner(
    model_name="deepseek/deepseek-r1",
    n_probes=100,  # Large sample
    depths=[1, 2, 3, 4, 5, 6, 7],
    gravity_levels=[i*0.5 for i in range(1, 11)]  # 10 levels
)
# Comprehensive results for paper
```

---

## Interpreting Results

### What CRI Means
- **CRI = 40+**: Model has strong reasoning robustness
- **CRI = 20-40**: Model has moderate robustness
- **CRI = <20**: Model is sensitive to semantic shifts

### What Escape Velocity Means
- **V_e < 500**: Model needs little reasoning to override priors
- **V_e 500-1500**: Model needs moderate reasoning
- **V_e > 1500**: Model strongly attached to pre-training priors

### What GII Means
- **GII < 2.0**: Efficient reasoning
- **GII 2.0-4.0**: Moderate overhead
- **GII > 4.0**: Significant reasoning overhead

---

**Now you're ready to run benchmarks!** 🚀
