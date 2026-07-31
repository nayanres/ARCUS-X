#!/usr/bin/env bash
# runframework.sh - Public entry point for the ARCUS-X benchmark (developer mode alias)
#
# Usage:
#   ./runframework.sh [--model MODEL] [--probes N] [--output FILE] [--baselines] [--master-seeds S...] [--dev]
#
# Examples:
#   ./runframework.sh --dev                        # runs dedicated developer mode

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
MODEL="google/gemini-2.5-flash-lite"
PROBES=5
OUTPUT="outputs/benchmark_results.json"
BASELINES=""
MASTER_SEEDS=""
DEV=""
API_BASE=""

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
        --api-base)
            API_BASE="--api-base $2"
            shift 2
            ;;
        --baselines)
            BASELINES="--baselines"
            shift
            ;;
        --dev)
            DEV="--dev"
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
            echo "Usage: $0 [--model MODEL] [--probes N] [--output FILE] [--baselines] [--master-seeds S...] [--dev]"
            echo ""
            echo "Runs the ARCUS-X benchmark with the specified model."
            echo ""
            echo "Options:"
            echo "  --model MODEL        Model name (default: google/gemini-2.5-flash-lite)"
            echo "  --probes N           Number of probes (default: 5)"
            echo "  --output FILE        Output JSON file (default: outputs/benchmark_results.json)"
            echo "  --baselines          Also run model-free baseline evaluators"
            echo "  --dev                Dedicated developer mode (seeds=1, gravity=0.5, tier=1, horizon=3, no adaptive expansion, 1 probe)"
            echo "  --master-seeds S...  Master seeds for variance measurement"
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

echo "Configuration: model=$MODEL probes=$PROBES output=$OUTPUT${API_BASE:+ api-base=on}${DEV:+ dev=on}${BASELINES:+ baselines=on}${MASTER_SEEDS:+ master-seeds=on}"
echo ""

python -m arcus.experiments.quickstart \
    --model "$MODEL" \
    --n-probes "$PROBES" \
    --output "$OUTPUT" \
    $API_BASE \
    $DEV \
    $BASELINES \
    $MASTER_SEEDS

echo ""
echo "Benchmark completed. Results saved to: $OUTPUT"
