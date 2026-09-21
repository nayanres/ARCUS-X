import sys
sys.path.insert(0, '.')
import post_run_analyzer as p

SEP = "=" * 50
print(f"SEP len: {len(SEP)}")

log = (
    "\n" + SEP + "\n"
    "ARCUS-X RUN START\n"
    "Run ID: test\n"
    "Model: m\n"
    "Total Probes: 2\n"
    + SEP + "\n"
    "=== RAW STREAM ENTRY (Tokens: 10) ===\n"
    "Parameters: z=3, gravity=0.0, tier=0, grid=default\n"
    "==============================\n"
    "\n" + SEP + "\n"
    "ARCUS-X RUN END\n"
    "Status: COMPLETE\n"
    "Completed: 2/2\n"
    "Failures: 0\n"
    "End Time: x\n"
    + SEP + "\n"
)

lines = log.splitlines(keepends=True)
print(f"num lines: {len(lines)}")
for i, line in enumerate(lines[:8]):
    print(f"  line {i}: {line!r} (len={len(line)})")

segs = p._split_runs(log)
print(f"\nnum segments: {len(segs)}")

