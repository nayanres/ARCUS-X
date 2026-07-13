#!/usr/bin/env bash

# Framework v1 - Automated Linux Execution Plug & Verification Pipeline
set -e

# ANSI escape codes for clean terminal logging
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}[*] Launching Framework v1 Linux Integration Engine...${NC}"

# Initialize Directories
mkdir -p outputs logs

# 2. Check Dependencies
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[×] FATAL: python3 could not be found. Please install Python 3.8+.${NC}"
    exit 1
fi

# handle Python Environment Paths
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Execute Data Generation & Model Benchmarking Matrix
echo -e "${GREEN}[*] Executing Isomorphic Logic Probe evaluation suites...${NC}"
python3 generator.py

# Locate the newest generated evaluation trace artifact
LATEST_TRACE=$(ls -t outputs/trace_seed_*.json 2>/dev/null | head -n 1)

if [ -z "$LATEST_TRACE" ]; then
    echo -e "${RED}[×] SYSTEM FAILURE: No cryptographic execution trace log emitted by the generator engine.${NC}"
    exit 1
fi

echo -e "${GREEN}[✓] Target trace log located: ${LATEST_TRACE}${NC}"

# 6. Automated Verification Pipeline (The Decentralized Referee)
echo -e "${YELLOW}[*] Routing trace payload directly to verify_trace.py referee...${NC}"
python3 verify_trace.py --trace "$LATEST_TRACE"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}================================================================${NC}"
    echo -e "${GREEN}[✓] SUCCESS: Framework v1 successfully executed and validated on Linux.${NC}"
    echo -e "${GREEN}================================================================${NC}"
else
    echo -e "${RED}[×] CRITICAL ERROR: Execution trace failed mathematical verification.${NC}"
    exit 1
fi