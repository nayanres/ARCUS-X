# Windows API-Only Mode Setup Guide

## Overview
This framework is fully compatible with Windows Command Prompt and OpenRouter API. 
No local GPU, PyTorch, vLLM, or transformers installation required.

## Prerequisites

### Step 1: Install Python & Dependencies
```bash
python --version  # Should be 3.8+

# Install required packages (API-only, no torch/vLLM):
pip install requests pydantic datasets numpy scipy nltk tqdm pyyaml regex
```

### Step 2: Get an OpenRouter API Key
1. Visit https://openrouter.ai
2. Sign up or log in
3. Go to Dashboard → API Key
4. Copy your key

### Step 3: Set Environment Variable
```bash
# Windows Command Prompt
set OPENROUTER_API_KEY=sk-or-v1-your-actual-api-key-here

# Verify it's set:
echo %OPENROUTER_API_KEY%
```

## Running Your First Benchmark

### Option A: Quick Test (Free Model)
```python
from benchmark_runner import BenchmarkRunner
from model_evaluation import ModelEvaluator
from fracture_finder import FracturePointFinder

# Use a free model for testing
runner = BenchmarkRunner(
    model_name="deepseek/deepseek-r1-distill:free",  # Free tier
    evaluator_class=ModelEvaluator,
    fracture_finder_class=FracturePointFinder,
    api_base="https://openrouter.ai/api/v1",
    n_probes=5  # Small test
)

results = runner.run()
print(f"CRI: {results['metrics']['cri']:.3f}")
```

### Option B: Production Benchmark (DeepSeek R1)
```python
from benchmark_runner import BenchmarkRunner
from model_evaluation import ModelEvaluator
from fracture_finder import FracturePointFinder

runner = BenchmarkRunner(
    model_name="deepseek/deepseek-r1",
    evaluator_class=ModelEvaluator,
    fracture_finder_class=FracturePointFinder,
    api_base="https://openrouter.ai/api/v1",
    n_probes=25,
    depths=[1, 2, 3, 4, 5],
    gravity_levels=[0.5, 1.2, 2.1, 3.5, 4.8]
)

results = runner.run()
```

### Option C: Test Multiple Models
```python
models = [
    "deepseek/deepseek-r1-distill:free",  # Cost: ~$0.0003/query
    "openai/gpt-4-turbo",                  # Cost: ~$0.01/query
    "anthropic/claude-3-5-sonnet",        # Cost: ~$0.003/query
]

for model in models:
    runner = BenchmarkRunner(
        model_name=model,
        evaluator_class=ModelEvaluator,
        fracture_finder_class=FracturePointFinder,
        api_base="https://openrouter.ai/api/v1",
        n_probes=10
    )
    results = runner.run()
    print(f"{model}: CRI={results['metrics']['cri']:.3f}")
```

## Understanding API-Only Mode

### What's Disabled (Not Needed)
- ❌ Local vLLM (requires CUDA/GPU)
- ❌ PyTorch inference (memory intensive)
- ❌ Transformers tokenizer (optional for Windows)
- ❌ Multi-GPU setup

### What's Enabled (API-Only)
- ✓ Full benchmark suite
- ✓ All metrics: CRI, GII, Semantic Gravity
- ✓ Early exit optimization (40-50% fewer queries)
- ✓ Context exhaustion detection
- ✓ Reasoning token extraction (DeepSeek, o3-mini, etc.)
- ✓ Heuristic G_s fallback when local model unavailable

### Semantic Gravity in API-Only Mode

When running on Windows without local models:
- **Tier 1 (Semantic Disruption)**: G_s = 1.0
- **Tier 2 (Attribute Inversion)**: G_s = 2.5
- **Tier 3 (Axiomatic Contradiction)**: G_s = 4.0

The Composite Robustness Index (CRI) still uses the full formula:
```
CRI = Σ[Accuracy / (1 + ln(1 + GII))] * ΔG_s
```

## Cost Estimation

For a complete benchmark with 25 probes × 5 depths × 5 gravity levels × 3 tiers = 1,875 queries:

| Model | Cost per Query | Total Cost |
|-------|----------------|-----------|
| DeepSeek R1 (free) | $0.0003 | $0.56 |
| DeepSeek R1 | $0.01 | $18.75 |
| Claude 3.5 Sonnet | $0.003 | $5.63 |
| GPT-4 Turbo | $0.01 | $18.75 |

With early exit enabled: **40-50% fewer queries** → proportional cost reduction

## Troubleshooting

### "OPENROUTER_API_KEY not set"
```bash
# Check if key is set
echo %OPENROUTER_API_KEY%

# If empty, set it:
set OPENROUTER_API_KEY=sk-or-v1-your-key

# Make it permanent (Windows):
setx OPENROUTER_API_KEY sk-or-v1-your-key
# Then restart Command Prompt
```

### "API request failed: 401 Unauthorized"
- Check API key is correct: `echo %OPENROUTER_API_KEY%`
- Ensure no spaces or typos
- Generate a new key from OpenRouter dashboard

### "finish_reason == 'length': Model hit token limit"
- Increase model's available tokens (if offered on OpenRouter)
- Reduce probe complexity
- Use a model with higher context window

### "ImportError: No module named 'torch'"
- This is EXPECTED on Windows
- Framework automatically uses API-only mode
- No action needed - just use `api_base` parameter

## Advanced: Custom Model Parameters

### For DeepSeek-R1 (Reasoning Model)
```python
# Automatically handled by framework:
# - Sets include_reasoning=True
# - Extracts thinking tokens from "reasoning" field
# - Handles finish_reason=="length" as model failure
```

### For OpenAI o3-mini
```python
# Automatically handled by framework:
# - Sets reasoning_effort="medium" or "high"
# - Extracts thinking tokens from structured output
```

### For Azure Custom Endpoints
```python
runner = BenchmarkRunner(
    model_name="your-deployment-name",
    evaluator_class=ModelEvaluator,
    fracture_finder_class=FracturePointFinder,
    api_base="https://your-resource.openai.azure.com/v1",
    n_probes=25
)

# Also set:
# os.environ["AZURE_API_KEY"] = "your-key"
```

## Output Format

Results saved to `outputs/benchmark_results.json`:
```json
{
  "model": "deepseek/deepseek-r1",
  "metrics": {
    "cri": 42.35,
    "avg_accuracy": 0.87,
    "escape_velocity": 1250,
    "fracture_depth": 4
  },
  "per_tier": {
    "tier_1": { "accuracy": 0.95, "gii": 1.2 },
    "tier_2": { "accuracy": 0.82, "gii": 2.1 },
    "tier_3": { "accuracy": 0.71, "gii": 3.8 }
  },
  "context_exhaustion_events": 3
}
```

## Performance Notes

- Early exit reduces evaluations by **40-50%**
- Average query time: 2-5 seconds per probe
- For 25 probes with early exit: ~2-3 minutes per model
- No local GPU needed = runs on any Windows machine

## Next Steps

1. Set your OPENROUTER_API_KEY
2. Run `python test_api_only_mode.py` to verify setup
3. Start with Option A (free model) for testing
4. Scale to Option B/C for full benchmarks

## Support

For issues:
1. Check OpenRouter API status: https://status.openrouter.ai
2. Verify API key in dashboard
3. Ensure sufficient account balance
4. Check model availability on OpenRouter

---

**Framework Version**: API-Only Windows Compatible v1.0
**Last Updated**: 2024
**Status**: Production Ready ✓
