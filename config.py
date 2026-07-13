"""Framework configuration helpers."""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class FrameworkConfig:
    """Configuration management for the evaluation framework"""
    
    DEFAULT_CONFIG = {
        "model": {
            "name": "meta-llama/Meta-Llama-3-8B",
            "device": "auto",
            "dtype": "float16",
            "cache_dir": "./cache",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        },
        "dataset": {
            "name": "Salesforce/FaithEval-counterfactual-v1.0",
            "split": "test",
            "cache_dir": "./cache",
        },
        "evaluation": {
            "batch_size": 5,
            "max_examples": None,
            "output_dir": "./outputs",
            "seed": 42,
        },
        "paths": {
            "data": "./data",
            "cache": "./cache",
            "outputs": "./outputs",
            "logs": "./logs",
        },
        "ood_tiers": {
            "tier_1_disruption_ratio": 0.3,
            "tier_2_inversion_ratio": 0.3,
            "tier_3_contradiction_count": 3,
        }
    }
    
    def __init__(self, config_file: str = None):
        """Initialize configuration from file or defaults"""
        self.config = self.DEFAULT_CONFIG.copy()
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file: str) -> None:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                loaded = json.load(f)
                self._deep_update(self.config, loaded)
            logger.info(f"Configuration loaded from {config_file}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_file}: {e}")
    
    def save_config(self, output_file: str) -> None:
        """Save current configuration to JSON file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    @staticmethod
    def _deep_update(base: Dict, updates: Dict) -> Dict:
        """Deep merge updates into base dictionary"""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base:
                FrameworkConfig._deep_update(base[key], value)
            else:
                base[key] = value
        return base
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
        return value if value is not None else default


class EnvironmentValidator:
    """Validates and checks the environment"""
    
    REQUIRED_PACKAGES = [
        'torch',
        'transformers',
        'datasets',
        'nltk',
        'numpy',
        'deepeval',
    ]
    
    OPTIONAL_PACKAGES = [
        'vllm',
        'accelerate',
    ]
    
    @staticmethod
    def check_python_version() -> Tuple[bool, str]:
        """Check Python version (3.8+)"""
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            return True, f"Python {version.major}.{version.minor}.{version.micro} ✓"
        return False, f"Python {version.major}.{version.minor} (requires 3.8+)"
    
    @staticmethod
    def check_packages() -> Tuple[List[str], List[str]]:
        """Check installed packages"""
        available = []
        missing = []
        
        for package in EnvironmentValidator.REQUIRED_PACKAGES:
            try:
                __import__(package)
                available.append(f"{package} ✓")
            except ImportError:
                missing.append(package)
        
        optional_status = []
        for package in EnvironmentValidator.OPTIONAL_PACKAGES:
            try:
                __import__(package)
                optional_status.append(f"{package} ✓ (optional)")
            except ImportError:
                optional_status.append(f"{package} ✗ (optional)")
        
        return available + optional_status, missing
    
    @staticmethod
    def check_gpu() -> Tuple[bool, str]:
        """Check GPU availability"""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                return True, f"GPU available: {device_name} ({device_count} device(s)), {memory:.1f}GB VRAM"
            else:
                return False, "No GPU detected, will use CPU (slower)"
        except Exception as e:
            return False, f"Could not detect GPU: {e}"
    
    @staticmethod
    def check_disk_space() -> Tuple[bool, str]:
        """Check available disk space"""
        try:
            import shutil
            total, used, free = shutil.disk_usage("./")
            free_gb = free / 1e9
            if free_gb > 50:
                return True, f"Sufficient disk space: {free_gb:.1f}GB available"
            elif free_gb > 10:
                return True, f"Limited disk space: {free_gb:.1f}GB available (16GB recommended for models)"
            else:
                return False, f"Insufficient disk space: {free_gb:.1f}GB available (need 16GB+)"
        except Exception as e:
            return False, f"Could not check disk space: {e}"
    
    @classmethod
    def validate_all(cls) -> Dict[str, any]:
        """Run all validation checks"""
        results = {
            "python": cls.check_python_version(),
            "packages": cls.check_packages(),
            "gpu": cls.check_gpu(),
            "disk": cls.check_disk_space(),
        }
        return results
    
    @classmethod
    def print_report(cls):
        """Print validation report"""
        results = cls.validate_all()
        
        logger.info("\n" + "="*60)
        logger.info("Environment Validation Report")
        logger.info("="*60)
        
        ok, msg = results["python"]
        logger.info(f"Python: {msg}")
        
        packages, missing = results["packages"]
        for pkg in packages:
            logger.info(f"  {pkg}")
        if missing:
            logger.warning(f"  Missing required: {', '.join(missing)}")
        
        ok, msg = results["gpu"]
        level = logger.info if ok else logger.warning
        level(f"GPU: {msg}")
        
        ok, msg = results["disk"]
        level = logger.info if ok else logger.warning
        level(f"Disk: {msg}")
        
        logger.info("="*60 + "\n")
        
        return not missing


class DirectorySetup:
    """Sets up required directory structure"""
    
    REQUIRED_DIRS = [
        "./data",
        "./cache",
        "./outputs",
        "./logs",
    ]
    
    @staticmethod
    def create_directories() -> List[str]:
        """Create all required directories"""
        created = []
        for directory in DirectorySetup.REQUIRED_DIRS:
            try:
                os.makedirs(directory, exist_ok=True)
                created.append(directory)
            except Exception as e:
                logger.error(f"Failed to create {directory}: {e}")
        return created
    
    @staticmethod
    def create_sample_config() -> None:
        """Create sample configuration file"""
        config = FrameworkConfig()
        config.save_config("config_template.json")
        logger.info("Sample configuration saved to config_template.json")


def initialize_framework():
    """Complete framework initialization"""
    logger.info("Initializing AI Evaluation Benchmark Framework\n")
    
    EnvironmentValidator.print_report()
    
    logger.info("Setting up directories...")
    created_dirs = DirectorySetup.create_directories()
    for directory in created_dirs:
        logger.info(f"  Created: {directory}")
    
    logger.info("\nCreating configuration files...")
    DirectorySetup.create_sample_config()
    
    logger.info("\n" + "="*60)
    logger.info("Framework Initialization Complete!")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("1. Review config_template.json and customize if needed")
    logger.info("2. Install GPU drivers if using CUDA")
    logger.info("3. Run: python integration_example.py")
    logger.info("4. Check README.md for detailed documentation\n")


if __name__ == "__main__":
    initialize_framework()
