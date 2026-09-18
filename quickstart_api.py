#!/usr/bin/env python3
"""
quickstart_api.py - Public API entry point for ARCUS-X (delegates to arcus.experiments.quickstart)

This script exists for backward compatibility with legacy documentation.
New users should use:  ./run_framework.sh  or  python -m arcus.experiments.quickstart
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arcus.experiments.quickstart import main

if __name__ == "__main__":
    sys.exit(main())