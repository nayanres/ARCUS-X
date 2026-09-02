"""Setup and environment checks."""

import os
import sys
import subprocess
import logging
from typing import Tuple, List

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def check_python_version() -> Tuple[bool, str]:
    """Check if Python 3.8+ is available."""
    version = sys.version_info
    msg = f"Python {version.major}.{version.minor}.{version.micro}"
    
    if version.major >= 3 and version.minor >= 8:
        return True, msg
    else:
        return False, f"{msg} (requires 3.8+)"


def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True, f"{package_name} ✓"
    except ImportError:
        return False, f"{package_name} ✗"


def check_required_packages() -> Tuple[bool, List[str]]:
    """Check all required packages for API-only mode."""
    required = [
        ("requests", "requests"),
        ("pydantic", "pydantic"),
        ("datasets", "datasets"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("nltk", "nltk"),
        ("tqdm", "tqdm"),
        ("pyyaml", "yaml"),
        ("regex", "regex"),
    ]
    
    results = []
    all_installed = True
    
    for package_name, import_name in required:
        installed, msg = check_package(package_name, import_name)
        results.append(msg)
        if not installed:
            all_installed = False
    
    return all_installed, results


def check_optional_packages() -> Tuple[bool, List[str]]:
    """Check optional packages (torch, transformers, vLLM)."""
    optional = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("vllm", "vllm"),
    ]
    
    results = []
    all_installed = True
    
    for package_name, import_name in optional:
        installed, msg = check_package(package_name, import_name)
        results.append(msg)
        if not installed:
            all_installed = False
    
    return all_installed, results


def check_api_key() -> Tuple[bool, str]:
    """Check if OpenRouter API key is configured."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if api_key:
        masked = f"{api_key[:15]}...{api_key[-5:]}"
        return True, f"API Key configured: {masked}"
    else:
        return False, "API Key not configured"


def check_api_connectivity() -> Tuple[bool, str]:
    """Test connectivity to OpenRouter API."""
    try:
        import requests
        
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            timeout=5
        )
        
        if response.status_code == 200:
            return True, "OpenRouter API reachable ✓"
        else:
            return False, f"OpenRouter API error: {response.status_code}"
    
    except Exception as e:
        return False, f"Cannot reach OpenRouter: {str(e)}"


def check_local_models_installed() -> bool:
    """Check if torch/transformers are installed."""
    try:
        import torch
        import transformers
        return True
    except ImportError:
        return False


def print_status(test_name: str, success: bool, message: str):
    """Print test result with formatting."""
    symbol = "✓" if success else "✗"
    status = "PASS" if success else "FAIL"
    logger.info(f"  [{symbol}] {test_name:30} {message:50} [{status}]")


def print_section(title: str):
    """Print section header."""
    logger.info("\n" + "=" * 90)
    logger.info(f"  {title}")
    logger.info("=" * 90)


def main():
    logger.info("\n")
    logger.info(" " * 20 + "AI EVALUATION BENCHMARK FRAMEWORK")
    logger.info(" " * 15 + "Setup Verification & Diagnostics")
    logger.info("")
    
    # System Check
    print_section("SYSTEM ENVIRONMENT")
    
    python_ok, python_msg = check_python_version()
    print_status("Python Version", python_ok, python_msg)
    
    if not python_ok:
        logger.warning("\n⚠️  Python 3.8+ required. Please upgrade.")
        return 1
    
    # Required Packages
    print_section("REQUIRED PACKAGES (API-Only Mode)")
    
    required_ok, required_msgs = check_required_packages()
    for msg in required_msgs:
        ok = "✓" in msg
        print_status(msg.split()[0], ok, msg)
    
    if not required_ok:
        logger.info("\n❌ Missing required packages. Install with:")
        logger.info("  pip install requests pydantic datasets numpy scipy nltk tqdm pyyaml regex")
        return 1
    
    # Optional Packages
    print_section("OPTIONAL PACKAGES (Local GPU Mode - Not Required on Windows)")
    
    optional_ok, optional_msgs = check_optional_packages()
    for msg in optional_msgs:
        ok = "✓" in msg
        status = "available" if ok else "not installed (OK for API mode)"
        logger.info(f"  [{'✓' if ok else '·'}] {msg:30} ({status})")
    
    if optional_ok:
        logger.info("\n  ℹ️  Local model support available. Can use vLLM or Transformers.")
    else:
        logger.info("\n  ℹ️  Using API-Only Mode. Will query OpenRouter/Azure.")
    
    # API Configuration
    print_section("OPENROUTER API CONFIGURATION")
    
    api_key_ok, api_key_msg = check_api_key()
    print_status("API Key", api_key_ok, api_key_msg)
    
    if not api_key_ok:
        logger.info("\n❌ API Key not configured. Set with:")
        logger.info("  Windows Command Prompt:")
        logger.info("    set OPENROUTER_API_KEY=sk-or-v1-your-key")
        logger.info("\n  Make it permanent:")
        logger.info("    setx OPENROUTER_API_KEY sk-or-v1-your-key")
        logger.info("    (then restart Command Prompt)")
        return 1
    
    connectivity_ok, connectivity_msg = check_api_connectivity()
    print_status("API Connectivity", connectivity_ok, connectivity_msg)
    
    if not connectivity_ok:
        logger.warning(f"\n⚠️  {connectivity_msg}")
        logger.info("  Check your internet connection or OpenRouter status.")
    
    # Framework Module Check
    print_section("FRAMEWORK MODULES")
    
    modules = [
        ("generator", "generator.py"),
        ("gravity", "gravity.py"),
        ("metrics", "metrics.py"),
        ("ilp", "ilp.py"),
        ("model_evaluation", "model_evaluation.py"),
        ("benchmark_runner", "benchmark_runner.py"),
        ("fracture_finder", "fracture_finder.py"),
    ]
    
    all_modules_ok = True
    for module_name, file_path in modules:
        try:
            __import__(module_name)
            print_status(f"Module: {module_name}", True, f"{file_path} ✓")
        except ImportError as e:
            print_status(f"Module: {module_name}", False, f"Import error: {str(e)[:40]}")
            all_modules_ok = False
    
    if not all_modules_ok:
        logger.warning("\n⚠️  Some framework modules failed to import.")
        logger.info("  Ensure you're in the framework directory: c:\\Users\\games\\python\\llmframeworkcline")
        return 1
    
    # Summary
    print_section("SUMMARY & RECOMMENDATIONS")
    
    local_models = check_local_models_installed()
    
    if required_ok and api_key_ok and connectivity_ok and all_modules_ok:
        logger.info("\n✅ FRAMEWORK IS READY TO USE!\n")
        
        if local_models:
            logger.info("You can use:")
            logger.info("  Option A: API-Only Mode (recommended for Windows)")
            logger.info("  Option B: Local vLLM/Transformers (requires GPU)")
            logger.info("\nStart with:")
            logger.info("  python quickstart_api.py")
        else:
            logger.info("✓ Ready for API-Only benchmarking on Windows")
            logger.info("\nStart with:")
            logger.info("  python quickstart_api.py")
        
        logger.info("\nFull documentation:")
        logger.info("  - WINDOWS_API_ONLY_GUIDE.md")
        logger.info("  - README.md")
        logger.info("  - HOW_TO_OPENROUTER.txt")
        
        logger.info("\nExample usage:")
        logger.info("  from benchmark_runner import BenchmarkRunner")
        logger.info("  from model_evaluation import ModelEvaluator")
        logger.info("  from fracture_finder import FracturePointFinder")
        logger.info("")
        logger.info("  runner = BenchmarkRunner(")
        logger.info("      model_name='deepseek/deepseek-r1',")
        logger.info("      evaluator_class=ModelEvaluator,")
        logger.info("      fracture_finder_class=FracturePointFinder,")
        logger.info("      api_base='https://ai.hackclub.com/proxy/v1/chat/completions',")
        logger.info("      n_probes=25")
        logger.info("  )")
        logger.info("  results = runner.run()")
        
        return 0
    
    else:
        logger.error("\n❌ SETUP INCOMPLETE")
        logger.info("\nMissing:")
        if not required_ok:
            logger.info("  - Required Python packages")
        if not api_key_ok:
            logger.info("  - OpenRouter API key")
        if not connectivity_ok:
            logger.info("  - API connectivity")
        if not all_modules_ok:
            logger.info("  - Framework modules")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
