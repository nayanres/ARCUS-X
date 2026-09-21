#!/usr/bin/env bash
# run_framework.sh - Public entry point for the ARCUS-X benchmark
#
# Usage:
#   ./run_framework.sh [--model MODEL] [--probes N] [--output FILE] [--baselines] [--master-seeds S...] [--dev]
#
# Examples:
#   ./run_framework.sh --model google/gemini-2.5-flash-lite --probes 5
#   ./run_framework.sh --baselines                 # model-free baselines only
#   ./run_framework.sh --model ... --baselines     # model + baselines in report
#   ./run_framework.sh --model ... --master-seeds all
#   ./run_framework.sh --model ... --master-seeds 42 1337 2026 9001 123456

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
MODEL="google/gemini-2.5-flash-lite"
PROBES=5
OUTPUT="outputs/benchmark_results.json"
BASELINES=""
MASTER_SEEDS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --probes)
            PROBES="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --baselines)
            BASELINES="--baselines"
            shift
            ;;
        --master-seeds)
            # Collect all following non-flag tokens as seeds.
            shift
            MASTER_SEEDS="--master-seeds"
            while [[ $# -gt 0 && "$1" != --* ]]; do
                MASTER_SEEDS="$MASTER_SEEDS $1"
                shift
            done
            ;;
        --help|-h)
            echo "Usage: $0 [--model MODEL] [--probes N] [--output FILE] [--baselines] [--master-seeds S...]"
            echo ""
            echo "Runs the ARCUS-X benchmark with the specified model."
            echo ""
            echo "Options:"
            echo "  --model MODEL        Model name (default: google/gemini-2.5-flash-lite)"
            echo "  --probes N           Number of probes (default: 5)"
            echo "  --output FILE        Output JSON file (default: outputs/benchmark_results.json)"
            echo "  --baselines          Also run model-free baseline evaluators (oracle /"
            echo "                       random / initial-state) and include them in the report."
            echo "  --master-seeds S...  Master seeds for variance measurement. Space-separated"
            echo "                       integers, or 'all' for the canonical paper seed list."
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Ensure outputs directory exists
mkdir -p outputs

# The interactive UX (startup banner, live progress bar, transient status line,
# and the final metrics/taxonomy chart) is rendered by BenchmarkRunner.run(),
# which the quickstart module invokes below. No extra wiring is required here —
# just launch it and the runner prints the banner, drives the progress display,
# and renders the chart on stdout (ANSI cursor control when run in a TTY,
# throttled plain lines when piped/redirected).
echo "Configuration: model=$MODEL probes=$PROBES output=$OUTPUT${BASELINES:+ baselines=on}${MASTER_SEEDS:+ master-seeds=on}"
echo ""

python -m arcus.experiments.quickstart \
    --model "$MODEL" \
    --n-probes "$PROBES" \
    --output "$OUTPUT" \
    $BASELINES \
    $MASTER_SEEDS

echo ""
echo "Benchmark completed. Results saved to: $OUTPUT"
