#!/usr/bin/env python3
"""
Quick Start: Run your first API-only benchmark on Windows

Usage:
    # Default (uses env vars or hardcoded defaults)
    python quickstart_api.py

    # Custom API endpoint, model, and key via command line
    python quickstart_api.py ^
        --api-key sk-or-v1-your-key ^
        --model deepseek/deepseek-r1 ^
        --api-base https://openrouter.ai/api/v1

    # Azure example
    python quickstart_api.py ^
        --api-key your-azure-key ^
        --model gpt-4 ^
        --api-base https://your-org.openai.azure.com/openai/deployments/gpt-4/chat/completions

This script runs a minimal benchmark to verify everything is working.
Different researchers can use different API registrations via CLI args.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments for flexibility across research teams."""
    parser = argparse.ArgumentParser(
        description="Run AI evaluation benchmark with custom API configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (uses environment variables)
  python quickstart_api.py

  # OpenRouter with DeepSeek
  python quickstart_api.py \\
    --api-key sk-or-v1-xxxx \\
    --model deepseek/deepseek-r1 \\
    --api-base https://openrouter.ai/api/v1

  # Azure OpenAI
  python quickstart_api.py \\
    --api-key your-azure-key \\
    --model gpt-4 \\
    --api-base https://yourorg.openai.azure.com/openai/deployments/gpt-4/chat/completions

  # Custom local proxy
  python quickstart_api.py \\
    --api-key sk-local \\
    --model llama-3.1-70b \\
    --api-base https://local-proxy:8000/v1
        """)
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='API key (default: OPENROUTER_API_KEY environment variable)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name (default: google/gemini-3.1-flash-lite for free testing)'
    )
    
    parser.add_argument(
        '--api-base',
        type=str,
        default=None,
        help='API base URL (default: https://ai.hackclub.com/proxy/v1/chat/completions for free tier)'
    )
    
    parser.add_argument(
        '--n-probes',
        type=int,
        default=5,
        help='Number of probes (default: 5)'
    )
    
    parser.add_argument(
        '--depths',
        type=str,
        default='1,2,3',
        help='Comma-separated depths to test (default: 1,2,3)'
    )
    
    parser.add_argument(
        '--gravity-levels',
        type=str,
        default='0.5,1.2,2.1',
        help='Comma-separated semantic gravity levels (default: 0.5,1.2,2.1)'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("=" * 70)
    logger.info("AI Evaluation Benchmark - Windows API-Only Quick Start")
    logger.info("=" * 70)
    
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("❌ No API key provided!")
        logger.info("\nProvide API key via:")
        logger.info("  1. Command line:       python quickstart_api.py --api-key sk-or-v1-xxxx")
        logger.info("  2. Environment variable: set OPENROUTER_API_KEY=sk-or-v1-xxxx")
        logger.info("  3. Permanent (Windows):  setx OPENROUTER_API_KEY sk-or-v1-xxxx")
        return 1
    
    logger.info(f"✓ API Key: {api_key[:20]}..." if len(api_key) > 20 else f"✓ API Key: {api_key}")
    
    model_name = args.model or "google/gemini-3.1-flash-lite"
    logger.info(f"✓ Model: {model_name}")
    
    api_base = args.api_base or "https://ai.hackclub.com/proxy/v1/chat/completions"
    logger.info(f"✓ API Base: {api_base}")
    
    try:
        depths = [int(x.strip()) for x in args.depths.split(',')]
        gravity_levels = [float(x.strip()) for x in args.gravity_levels.split(',')]
        logger.info(f"✓ Depths: {depths}")
        logger.info(f"✓ Gravity Levels: {gravity_levels}")
    except ValueError as e:
        logger.error(f"❌ Invalid depth or gravity format: {e}")
        logger.info("  Depths and gravity levels should be comma-separated numbers")
        logger.info("  Example: --depths 1,2,3 --gravity-levels 0.5,1.2,2.1")
        return 1
    
    logger.info("\nImporting framework components...")
    try:
        from benchmark_runner import BenchmarkRunner
        from model_evaluation import ModelEvaluator
        from fracture_finder import FracturePointFinder
        logger.info("✓ All imports successful")
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        logger.info("\nInstall dependencies:")
        logger.info("  pip install requests pydantic datasets numpy scipy nltk tqdm pyyaml regex")
        return 1
    
    os.makedirs("outputs", exist_ok=True)
    logger.info("✓ Output directory ready")
    
    logger.info("\n" + "=" * 70)
    logger.info(f"Starting Benchmark ({args.n_probes} probes, {len(depths)} depths, {len(gravity_levels)} gravity levels)")
    logger.info("=" * 70)
    
    try:
        runner = BenchmarkRunner(
            model_name=model_name,
            evaluator_class=ModelEvaluator,
            fracture_finder_class=FracturePointFinder,
            api_base=api_base,
            api_key=api_key,
            n_probes=args.n_probes,
            gravity_levels=gravity_levels
        )
        
        logger.info(f"\nEvaluating: {runner.model_name}")
        
        results = runner.run()
        
        logger.info("\n" + "=" * 70)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 70)
        
        metrics = results.get("metrics", {})
        logger.info(f"✓ Terminal CRI: {metrics.get('cri', 0):.3f}")
        logger.info(f"✓ Average Accuracy: {metrics.get('avg_accuracy', 0):.3f}")
        logger.info(f"✓ Escape Velocity: {metrics.get('escape_velocity', 0)} tokens")
        logger.info(f"✓ Fracture Depth: {metrics.get('fracture_depth', 0)}")
        
        per_tier = results.get("per_tier", {})
        logger.info("\nPer-Tier Breakdown:")
        for tier_name, tier_data in per_tier.items():
            acc = tier_data.get("accuracy", 0)
            gii = tier_data.get("gii", 0)
            logger.info(f"  {tier_name:12}: Accuracy={acc:.3f}, GII={gii:.3f}")
        
        import json
        
        def stringify_keys(data):
            """
            Recursively convert tuple keys to strings for JSON serialization.
            Handles nested dictionaries and lists containing dictionaries.
            """
            if isinstance(data, dict):
                new_dict = {}
                for key, value in data.items():
                    # Convert tuple keys to string representation
                    if isinstance(key, tuple):
                        new_key = "_".join(str(k) for k in key)
                    else:
                        new_key = str(key) if not isinstance(key, str) else key
                    
                    # Recursively process values
                    new_dict[new_key] = stringify_keys(value)
                return new_dict
            
            elif isinstance(data, list):
                return [stringify_keys(item) for item in data]
            
            elif isinstance(data, (int, float, str, bool, type(None))):
                return data
            
            else:
                # For other types, attempt string conversion
                return str(data)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"outputs/quickstart_results_{timestamp}.json"
        
        # Stringify all tuple keys before JSON serialization
        serializable_results = stringify_keys(results)
        
        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"\n✓ Results saved to: {results_file}")
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ SUCCESS! Framework is working in API-Only mode.")
        logger.info("=" * 70)
        logger.info("\nNext steps:")
        logger.info("  1. Read WINDOWS_API_ONLY_GUIDE.md for full documentation")
        logger.info("  2. Modify n-probes, depths, gravity-levels for larger benchmarks")
        logger.info("  3. Try other models: claude-3.5-sonnet, gpt-4-turbo, etc.")
        logger.info("\nUseful commands:")
        logger.info("  # Use your OpenRouter account")
        logger.info("  python quickstart_api.py --api-key sk-or-v1-xxxx \\")
        logger.info("    --model deepseek/deepseek-r1 \\")
        logger.info("    --api-base https://openrouter.ai/api/v1")
        logger.info("\n  # Use Azure OpenAI")
        logger.info("  python quickstart_api.py --api-key your-key \\")
        logger.info("    --model gpt-4 \\")
        logger.info("    --api-base https://yourorg.openai.azure.com/openai/deployments/gpt-4/chat/completions")
        logger.info("\nEstimated costs for full benchmark (1,875 queries):")
        logger.info("  - DeepSeek R1 (free): ~$0.56")
        logger.info("  - DeepSeek R1: ~$18.75")
        logger.info("  - Claude 3.5 Sonnet: ~$5.63")
        logger.info("\nWith early exit enabled: 40-50% fewer queries\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except KeyboardInterrupt:
        logger.warning("\n⚠ Benchmark interrupted by user")
        return 1


if __name__ == "__main__":
    sys.exit(main())
