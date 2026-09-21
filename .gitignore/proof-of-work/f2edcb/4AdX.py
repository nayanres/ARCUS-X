#!/usr/bin/env python3
"""
Post-run analyzer for ARCUS-X benchmark results.

This script reads quickstart JSON files and absolute_raw_stream text files,
generating a comprehensive metrics report in the exact format requested by the user.
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, Any, List

# Force UTF-8 output on Windows (cp1252 cannot encode the block glyphs).
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

# Force UTF-8 output on Windows (cp1252 cannot encode the block glyphs).
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load and parse a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_raw_stream_file(filepath: str) -> str:
    """Load an absolute_raw_stream text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def format_percentage(value: float) -> str:
    """Format a percentage value with one decimal place."""
    return f"{value:.1f}%"


def format_float(value: float, decimals: int = 3) -> str:
    """Format a float with specified decimal places."""
    return f"{value:.{decimals}f}"


def create_progress_bar(percentage: float, width: int = 40) -> str:
    """Create a progress bar string."""
    filled = int(round(percentage / 100 * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def analyze_quickstart(filepath: str) -> str:
    """Analyze a quickstart JSON file and generate the metrics report."""
    data = load_json_file(filepath)
    
    # Extract key metrics
    model = data.get("model", "unknown")
    run_status = data.get("run_status", "UNKNOWN")
    completed_probes = data.get("completed_probes", 0)
    total_probes = data.get("total_probes", 0)
    accuracy = data.get("accuracy", 0.0)
    
    # Per-Horizon Accuracy
    per_horizon = data.get("per_horizon_accuracy", {})
    
    # Per-Gravity Accuracy  
    per_gravity = data.get("per_gravity_accuracy", {})
    
    # Per-Tier Accuracy
    per_tier = data.get("per_tier_accuracy", {})
    
    # Latency diagnostics
    latency_data = data.get("latency_diagnostics", {})
    
    # Failure analysis
    failure_analysis = data.get("failure_analysis", {})
    
    # Build the report
    lines = []
    
    # Header
    lines.append("============================================================")
    lines.append("ARCUS-X RUN SUMMARY")
    lines.append("============================================================")
    
    # Basic info
    lines.append(f"Model:               {model}")
    lines.append(f"Run Status:          {run_status}")
    lines.append(f"Completed Probes:    {completed_probes} / {total_probes}")
    lines.append(f"Accuracy:            {format_percentage(accuracy)}")
    lines.append("")
    
    # Per-Horizon Accuracy
    lines.append("Per-Horizon Accuracy:")
    for z in sorted(per_horizon.keys(), key=lambda x: int(x)):
        acc = per_horizon[z]
        lines.append(f"  z={z:<2} : {format_percentage(acc)}")
    lines.append("")
    
    # Per-Gravity Accuracy
    lines.append("Per-Gravity Accuracy:")
    for gravity in sorted(per_gravity.keys(), key=lambda x: float(x)):
        acc = per_gravity[gravity]
        lines.append(f"  gravity={gravity:<4} : {format_percentage(acc)}")
    lines.append("")
    
    # Per-Tier Accuracy
    lines.append("Per-Tier Accuracy:")
    for tier in sorted(per_tier.keys()):
        acc = per_tier[tier]
        lines.append(f"  {tier:<6} : {format_percentage(acc)}")
    lines.append("")
    
    # Token and latency stats
    avg_output_tokens = data.get("avg_output_tokens", 0)
    max_output_tokens = data.get("max_output_tokens", 0)
    avg_latency = data.get("avg_latency", 0)
    failure_count = data.get("failure_count", 0)
    
    lines.append(f"Total Provider Cost: 0.0000")
    lines.append(f"Average Output Tokens: {avg_output_tokens:.1f}")
    lines.append(f"Maximum Output Tokens: {max_output_tokens:.0f}")
    lines.append(f"Average Latency:       {format_float(avg_latency / 1000, 3)} ms")
    lines.append(f"Failure Count:         {failure_count}")
    lines.append("")
    
    # Multi-Seed Variance
    lines.append("Multi-Seed Variance (mean +/- std across master seeds):")
    lines.append(f"  Seeds:               {data.get('seeds', [42])} (single seed; no variance reported)")
    lines.append("")
    
    # Failure Taxonomy
    lines.append("Failure Taxonomy:")
    for mode, pct in failure_analysis.items():
        lines.append(f"  {mode:<30} {format_percentage(pct)}")
    lines.append("")
    
    # Latency Diagnostics
    lines.append("Latency Diagnostics:")
    lines.append("------------------------------------------------------------")
    lines.append(f"Mean Latency:        {format_float(latency_data.get('mean_latency_ms', 0) / 1000, 3)} ms")
    lines.append(f"Median Latency:      {format_float(latency_data.get('median_latency_ms', 0) / 1000, 3)} ms")
    lines.append(f"P95 Latency:         {format_float(latency_data.get('p95_latency_ms', 0) / 1000, 3)} ms")
    lines.append(f"Maximum Latency:     {format_float(latency_data.get('max_latency_ms', 0) / 1000, 3)} ms")
    lines.append(f"Mean Output Tokens/sec: {latency_data.get('mean_output_tokens_per_sec', 0):.3f}")
    lines.append(f"Mean Correct Steps/sec: {latency_data.get('mean_correct_steps_per_sec', 0):.3f}")
    lines.append("")
    
    # Normalized Efficiency
    lines.append("Normalized Efficiency:")
    lines.append(f"  ALE: {latency_data.get('ale', 0):.4f}")
    lines.append(f"  Correct Steps/sec: {latency_data.get('correct_steps_per_sec', 0):.3f}")
    lines.append("")
    
    # Latency by Horizon
    per_horizon_latency = latency_data.get('per_horizon', [])
    if per_horizon_latency:
        lines.append("Latency by Horizon:")
        for entry in per_horizon_latency:
            z = entry.get('horizon', 'unknown')
            mean_latency = entry.get('mean_latency_ms', 0) / 1000
            correct_steps = entry.get('correct_steps_per_sec', 0)
            lines.append(f"  z={z}:")
            lines.append(f"    Mean Latency: {format_float(mean_latency, 3)} ms")
            lines.append(f"    Correct Steps/sec: {correct_steps:.3f}")
        lines.append("")
    
    # ARCUS-X EVALUATION CHART
    lines.append("================================================================")
    lines.append("ARCUS-X EVALUATION CHART")
    lines.append("================================================================")
    
    # Headline Metrics
    lines.append("\nHeadline Metrics:")
    headline_metrics = [
        ("Step Accuracy", data.get("step_accuracy", 0.0)),
        ("Exact Match", data.get("exact_match", 0.0)),
        ("Continuity", data.get("continuity", 0.0)),
        ("Horizon Compliance", data.get("horizon_compliance", 0.0)),
        ("Efficiency", data.get("efficiency", 0.0)),
    ]
    
    for name, val in headline_metrics:
        bar = create_progress_bar(val)
        lines.append(f"  {name:<20} {bar:<32} {format_percentage(val)}")
    
    # Additional metrics
    lines.append(f"\n  {'GBI':<20} {data.get('generation_bloat_index', 0.0):>6.2f}  (lower is better)")
    
    cri = data.get("cri", {})
    if isinstance(cri, dict):
        lines.append(f"  {'CRI':<20} {cri.get('cri', 0.0):>6.3f}")
        lines.append(f"  {'Trajectory Fidelity':<20} {cri.get('trajectory_fidelity', 0.0):>6.3f}")
        lines.append(f"  {'Horizon Robustness':<20} {cri.get('horizon_robustness', 0.0):>6.3f}")
        lines.append(f"  {'Semantic Robustness':<20} {cri.get('semantic_robustness', 0.0):>6.3f}")
        lines.append(f"  {'Generation Efficiency':<20} {cri.get('generation_efficiency', 0.0):>6.3f}")
    
    # Fracture depth
    fracture_depth = data.get("fracture_depth", 0)
    lines.append(f"\n  Fracture Depth: {fracture_depth}")
    
    # Per-tier fracture depth
    per_tier_data = data.get("per_tier", {})
    if per_tier_data:
        lines.append("\nPer-Tier Fracture Depth:")
        # Compute max fracture depth for scaling
        max_fd = max([float(d.get("fracture_depth", 0.0)) for d in per_tier_data.values()]) if per_tier_data else 1.0
        for tier, tier_data in sorted(per_tier_data.items()):
            fdv = float(tier_data.get("fracture_depth", 0.0))
            bar = create_progress_bar((fdv / max(1.0, max_fd)) * 100)
            lines.append(f"  {tier:<20} {bar:<32} {fdv:>6.1f}")
    
    # Error taxonomy (failure distribution)
    failure_analysis = data.get("failure_analysis", {})
    if failure_analysis:
        lines.append("\nError Taxonomy (failure distribution):")
        for mode, pct in failure_analysis.items():
            bar = create_progress_bar(pct)
            lines.append(f"  {mode:<30} {bar:<32} {format_percentage(pct)}")
    
    lines.append("=" * 64)
    
    return "\n".join(lines)


def process_file(filepath: str) -> str:
    """Process a single quickstart JSON file."""
    if filepath.endswith('.json'):
        return analyze_quickstart(filepath)
    elif filepath.endswith('.txt'):
        # Handle absolute_raw_stream files - currently just pass through
        # Could extract additional metadata if needed
        return analyze_quickstart({"model": "raw_stream", "run_status": "N/A"}) + "\n[Raw stream content omitted from summary]"
    else:
        return ""


def main():
    """Main function to run the analyzer."""
    parser = argparse.ArgumentParser(
        description="Analyze ARCUS-X quickstart JSON and raw stream files."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing quickstart JSON and raw stream files"
    )
    parser.add_argument(
        "--pattern",
        default="quickstart_results_*.json",
        help="Glob pattern for input files (default: quickstart_results_*.json)"
    )
    
    args = parser.parse_args()
    
    input_dir = args.input
    pattern = args.pattern
    
    # Find all matching files
    search_path = os.path.join(input_dir, pattern)
    input_files = glob.glob(search_path)
    
    if not input_files:
        print(f"No files found matching pattern '{pattern}' in directory '{input_dir}'")
        sys.exit(1)
    
    # Process each file and print its report
    for filepath in sorted(input_files):
        print(f"\n{'='*60}")
        print(f"Processing: {filepath}")
        print('='*60)
        report = process_file(filepath)
        if report:
            print(report)
        else:
            print(f"No data to report for {filepath}")


if __name__ == "__main__":
    main()
