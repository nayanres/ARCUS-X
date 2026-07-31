import sys
sys.path.insert(0, '.')
import post_run_analyzer as p
import re

# Test the regexes directly
RUN_START_RE = p.RUN_START_RE
RUN_END_RE = p.RUN_END_RE
SEP_RE = re.compile(r"^=+\s*$")

test_lines = [
    "ARCUS-X RUN START",
    "Run ID: 2026-07-16T02:31:44",
    "Model: deepseek/deepseek-v4-pro",
    "Total Probes: 2",
    "==================================================",  # 50 =
    "=== RAW STREAM ENTRY (Tokens: 10) ===",
    "Parameters: z=3, gravity=0.0, tier=0, grid=default",
    "------------------------------",
    "[MODEL OUTPUT]:",
    "[0,0]->[1,0]->[2,0]",
    "[GROUND TRUTH EXPECTED]:",
    "['[0,0]', '[1,0]', '[2,0]']",
    "==============================",  # 30 =
    "==================================================",  # 50 =
    "ARCUS-X RUN END",
    "Status: COMPLETE",
    "Completed: 2/2",
    "Failures: 0",
    "End Time: 2026-07-16T04:12:22",
    "==================================================",  # 50 =
]

with open('outputs/_dbg_regex.txt', 'w', encoding='utf-8') as f:
    for line in test_lines:
        stripped = line.strip()
        f.write(f"line={stripped!r:50} START={bool(RUN_START_RE.match(stripped))} END={bool(RUN_END_RE.match(stripped))} SEP={bool(SEP_RE.match(stripped))}\n")
