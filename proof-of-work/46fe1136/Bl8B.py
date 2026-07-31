import sys
sys.path.insert(0, '.')
import post_run_analyzer as p

# Minimal test: what does splitlines do to "=" * 50?
s = "\n" + "=" * 50 + "\nARCUS-X RUN START\n"
print("splitlines of s:", [repr(x) for x in s.splitlines(keepends=True)])

# Now test _split_runs on a string with =" * 50
log = (
    "\n" + "=" * 50 + "\n"
    "ARCUS-X RUN START\n"
    "Run ID: test\n"
    "Model: m\n"
    "Total Probes: 2\n"
    "=" * 50 + "\n"
    "=== RAW STREAM ENTRY (Tokens: 10) ===\n"
    "Parameters: z=3, gravity=0.0, tier=0, grid=default\n"
    "==============================\n"
    "\n" + "=" * 50 + "\n"
    "ARCUS-X RUN END\n"
    "Status: COMPLETE\n"
    "Completed: 2/2\n"
    "Failures: 0\n"
    "End Time: x\n"
    "=" * 50 + "\n"
)

print("\nlog repr (first 120 chars):", repr(log[:120]))
lines = log.splitlines(keepends=True)
print(f"\nnum lines: {len(lines)}")
for i, line in enumerate(lines[:8]):
    print(f"  line {i}: {line!r}")

segs = p._split_runs(log)
print(f"\nnum segments: {len(segs)}")
