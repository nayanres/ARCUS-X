#!/usr/bin/env python3
"""
quickstart_api.py - Public API entry point for ARCUS-X (delegates to arcus.experiments.quickstart)

This script exists for backward compatibility with legacy documentation.
New users should use:  ./run_framework.sh  or  python -m arcus.experiments.quickstart
"""

import sys
import os

# Add the current directory to Python path so arcus can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Delegate to the actual implementation
from arcus.experiments.quickstart import main

if __name__ == "__main__":
    sys.exit(main())