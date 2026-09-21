import sys
sys.path.insert(0, '.')
import post_run_analyzer as p

print("RUN_END_RE pattern:", repr(p.RUN_END_RE.pattern))
print("RUN_START_RE pattern:", repr(p.RUN_START_RE.pattern))

# The exact separator from the test
sep50 = "=" * 50
print(f"sep50={sep50!r} len={len(sep50)}")
print("RUN_END_RE.match(sep50):", bool(p.RUN_END_RE.match(sep50.strip())))
print("RUN_START_RE.match(sep50):", bool(p.RUN_START_RE.match(sep50.strip())))

# Check if there's something weird with the = character
# Maybe it's not ASCII = but a different char
for i, c in enumerate(sep50[:5]):
    print(f"char {i}: {c!r} ord={ord(c)}")

# Test the full _split_runs on a minimal 1-run log
def _run_block():
    return (
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

log = _run_block()
segs = p._split_runs(log)
print(f"\n1-run log -> {len(segs)} segments")
for meta, text in segs:
    print(f"  meta={meta}, text_len={len(text)}")
    print("  TEXT:", repr(text[:120]))

