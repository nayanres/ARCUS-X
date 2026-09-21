import sys
sys.path.insert(0, '.')
import post_run_analyzer as p

# Recreate the test log format
def _entry(z, gravity, tier, model, truth):
    return (
        f"\n=== RAW STREAM ENTRY (Tokens: 10) ===\n"
        f"Parameters: z={z}, gravity={gravity}, tier={tier}, grid=default\n"
        f"[MODEL OUTPUT]:\n{model}\n"
        f"[GROUND TRUTH EXPECTED]:\n{truth}\n"
    )

def _run_block(run_id, model, total, status, completed, failures, entries):
    sep = "=" * 50
    return (
        f"\n{sep}\nARCUS-X RUN START\nRun ID: {run_id}\nModel: {model}\n"
        f"Total Probes: {total}\n{sep}\n"
        + "".join(entries)
        + f"\n{sep}\nARCUS-X RUN END\nStatus: {status}\nCompleted: {completed}/{total}\n"
        f"Failures: {failures}\nEnd Time: 2026-07-16T02:31:45\n{sep}\n"
    )

log = _run_block(
    run_id="2026-07-16T02:31:44",
    model="deepseek/deepseek-v4-pro",
    total=2,
    status="COMPLETE",
    completed=2,
    failures=0,
    entries=[
        _entry(3, 0.0, 0, "[0,0]->[1,0]", "['[0,0]', '[1,0]']"),
        _entry(3, 0.0, 0, "[0,0]->[1,0]", "['[0,0]', '[1,0]']"),
    ],
) + _run_block(
    run_id="2026-07-16T02:31:46",
    model="deepseek/deepseek-v4-pro",
    total=4,
    status="PARTIAL",
    completed=2,
    failures=1,
    entries=[
        _entry(3, 0.0, 0, "[0,0]->[1,0]", "['[0,0]', '[1,0]']"),
        _entry(3, 0.0, 0, "[0,0]->[1,0]", "['[0,0]', '[1,0]']"),
    ],
)

segs = p._split_runs(log)
with open('outputs/_dbg_split.txt', 'w', encoding='utf-8') as f:
    f.write(f"num segments: {len(segs)}\n")
    for i, (meta, text) in enumerate(segs[:3]):
        f.write(f"\n--- segment {i} ---\n")
        f.write(f"meta: {meta}\n")
        f.write(f"text len: {len(text)}\n")
        f.write(text[:200])
        f.write("\n")
