#!/usr/bin/env python3
"""
Post-run analyzer for ARCUS-X benchmark results.

This script now handles both JSON result files and TXT log files. For TXT files, it parses structured log entries into a format compatible with the existing metrics computation.
"""

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr.encoding != "utf-8":
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Re-implement the small helpers used by benchmark_runner
# ---------------------------------------------------------------------------

# [Existing helper functions remain unchanged]

# ---------------------------------------------------------------------------
# TXT file parsing logic
# ---------------------------------------------------------------------------

def parse_txt_entry(line: str) -> Dict[str, Any]:
    """Parse a single log line from TXT file into structured data.
    """
    data = {}
    if line.startswith("Parameters:"):
        params = line.split(": ")[1].split(", ")
        data["z"] = int(params[0].split()[1])
        data["gravity"] = float(params[1].split()[1])
        data["tier"] = int(params[2].split()[1])
        data["grid_key"] = params[3].split()[1]
    elif line.startswith("Actions:"):
        data["actions"] = line.split(": ")[1].strip("[]").split(", ")
    elif line.startswith("TransitionRules:"):
        data["transition_rules"] = line.split(": ")[1].strip()
    elif line.startswith("InitialState:"):
        data["initial_state"] = line.split(": ")[1].strip("[]").split(", ")
    elif line.startswith("Latency\("ms\):"):
        data["latency_ms"] = float(line.split(": ")[1])
    elif line.startswith("OutputTokens:"):
        data["output_tokens"] = int(line.split(": ")[1])
    return data


def process_txt_file(filepath: str) -> Dict[str, Any]:
    """Parse entire TXT file into results matrix format.
    """
    results_matrix = {}
    current_key = None
    for line in open(filepath, "r", encoding="utf-8"):
        line = line.strip()
        if line.startswith("Parameters:"):
            # Parse parameters to create matrix key
            params = line.split(": ")[1].split(", ")
            z = int(params[0].split()[1])
            gravity = float(params[1].split()[1])
            tier = int(params[2].split()[1])
            grid_key = params[3].split()[1]
            key = f"{z}_{gravity}_{tier}_{grid_key}"
            current_key = key
            results_matrix[key] = {
                "z": z,
                "gravity": gravity,
                "tier": tier,
                "grid_key": grid_key,
                "probes": []
            }
        elif current_key:
            # Parse probe data
            if line.startswith("Latency\("ms\):"):
                results_matrix[current_key]["latency_ms"] = float(line.split(": ")[1])
            elif line.startswith("OutputTokens:"):
                results_matrix[current_key]["output_tokens"] = int(line.split(": ")[1])
            elif line.startswith("[MODEL OUTPUT]:"):
                # Parse model output (simplified)
                results_matrix[current_key]["raw_output"] = line.split("[MODEL OUTPUT]:")[1].strip()
            elif line.startswith("[GROUND TRUTH EXPECTED]:"):
                results_matrix[current_key]["ground_truth"] = line.split("[GROUND TRUTH EXPECTED]:")[1].strip()
    return results_matrix

# ---------------------------------------------------------------------------
# Main function with file type detection
# ---------------------------------------------------------------------------
def main():
    """Main entry point.
    """
    parser = argparse.ArgumentParser(
        description="Reproduce the ARCUS-X completed-run report from quickstart JSON files or TXT log files."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="A quickstart JSON file OR a directory containing quickstart_results_*.json files OR a TXT log file"
    )
    parser.add_argument(
        "--pattern",
        default="quickstart_results_*.json",
        help="Glob pattern when --input is a directory (default: quickstart_results_*.json)"
    )

    args = parser.parse_args()
    input_path = args.input
    pattern = args.pattern

    files: List[str] = []
    if os.path.isdir(input_path):
        search = os.path.join(input_path, pattern)
        files = sorted(glob.glob(search))
    elif os.path.isfile(input_path):
        files = [input_path]
    else:
        print(f"Input path not found: {input_path}")
        sys.exit(1)

    if not files:
        print(f"No files found matching pattern '{pattern}' in '{input_path}'")
        sys.exit(1)

    for filepath in files:
        print(f"
{'=' * 60}"
        f"Processing: {filepath}"
        f"{'=' * 60}"
    )
        try:
            if filepath.endswith(".json"):
                data = json.load(open(filepath, "r", encoding="utf-8"))
            elif filepath.endswith(".txt"):
                data = process_txt_file(filepath)
            else:
                raise ValueError(f"Unsupported file type: {filepath}")
            report = _render_report(data)
        except Exception as exc:
            print(f"Failed to analyze {filepath}: {exc}")
            continue
        print(report)

if __name__ == "__main__":
    main()
