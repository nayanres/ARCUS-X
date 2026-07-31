 actualyimport sys, tempfile, os
sys.path.insert(0, '.')
import post_run_analyzer as p

# Exact replica of test _run_block and _entry
def _entry(z, gravity, tier, model_path, truth_path, tokens=10):
    return (
        f"\n=== RAW STREAM ENTRY (Tokens: {tokens}) ===\n"
        f"Parameters: z={z}, gravity={gravity}, tier={tier}, grid=default\n"
        "------------------------------\n"
        "[MODEL OUTPUT]:\n"
        f"{model_path}\n\n"
        "[GROUND TRUTH EXPECTED]:\n"
        f"{truth_path}\n\n"
        "==============================\n"
    )

def _run_block(run_id, model, total, status, completed, failures, entries, reason=""):
    block = (
        "\n" + "=" * 50 + "\n"
        "ARCUS-X RUN START\n"
        f"Run ID: {run_id}\n"
        f"Model: {model}\n"
        f"Total Probes: {total}\n"
        "=" * 50 + "\n"
    )
    block += "".join(entries)
    block += (
        "\n" + "=" * 50 + "\n"
        "ARCUS-X RUN END\n"
        f"Status: {status}\n"
        f"Completed: {completed}/{total}\n"
        f"Failures: {failures}\n"
    )
    if reason:
        block += f"Reason: {reason}\n"
    block += (
        "End Time: 2026-07-16T04:12:22\n"
        "=" * 50 + "\n"
    )
    return block

def _make_log():
    run1 = _run_block(
        run_id="2026-07-16T02:31:44",
        model="deepseek/deepseek-v4-pro",
        total=2,
        status="COMPLETE",
        completed=2,
        failures=0,
        entries=[
            _entry(3, 0.0, 0, "[0,0]->[1,0]->[2,0]", "['[0,0]', '[1,0]', '[2,0]']"),
            _entry(5, 0.5, 1, "[0,0]->[1,1]->[2,2]", "['[0,0]', '[1,1]', '[2,2]']"),
        ],
    )
    run2 = _run_block(
        run_id="2026-07-16T02:31:46",
        model="deepseek/deepseek-v4-pro",
        total=4,
        status="PARTIAL",
        completed=2,
        failures=1,
        entries=[
            _entry(3, 0.0, 0, "[0,0]->[1,0]->[2,0]", "['[0,0]', '[1,0]', '[2,0]']"),
            _entry(5, 0.5, 1, "[0,0]->[1,1]->[2,2]", "['[0,0]', '[1,1]', '[2,2]']"),
        ],
    )
    return run1 + run2

log = _make_log()

# Test 1: direct _split_runs
segs = p._split_runs(log)
print(f"DIRECT _split_runs: {len(segs)} segments")

# Test 2: via file (like the test does)
fd, path = tempfile.mkstemp(suffix=".txt", prefix="arcus_test_")
with os.fdopen(fd, "w", encoding="utf-8") as f:
    f.write(log)
try:
    r = p.analyze_raw_log(path)
    print(f"VIA FILE analyze_raw_log: {len(r['runs'])} runs")
finally:
    os.remove(path)

# Test 3: check what's in the file
fd, path = tempfile.mkstemp(suffix=".txt", prefix="arcus_test_")
with os.fdopen(fd, "w", encoding="utf-8") as f:
    f.write(log)
try:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    lines = content.splitlines(keepends=True)
    print(f"FILE lines: {len(lines)}")
    for i, line in enumerate(lines[:8]):
        print(f"  line {i}: {line!r} (len={len(line)})")
finally:
    os.remove(path)
