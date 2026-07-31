import post_run_analyzer as p

# Replicate _split_runs inline with debug prints
RUN_START_RE = p.RUN_START_RE
RUN_END_RE = p.RUN_END_RE

def _entry(z, gravity, tier, model_path, truth_path, tokens=10):
    return (
        f"\n=== RAW STREAM ENTRY (Tokens: {tokens}) ===\n"
        f"Parameters: z={z}, gravity={gravity}, tier={tier}, grid=8x8\n"
        f"Grid: 8x8\nInitialState: [0,0]\nActions: [[0,0],[1,0]]\n"
        f"TransitionRules: step=1\nEnvironmentMetadata: gravity={gravity}\n"
        f"[MODEL OUTPUT]\n{model_path}\n[GROUND TRUTH EXPECTED]\n{truth_path}\n"
    )

def _run_block(run_id, model, total, status, completed, failures, entries):
    block = ("\n" + "=" * 50 + "\nARCUS-X RUN START\n" f"Run ID: {run_id}\n"
             f"Model: {model}\n" f"Total Probes: {total}\n" + "=" * 50 + "\n")
    block += "".join(entries)
    block += ("\n" + "=" * 50 + "\nARCUS-X RUN END\n" f"Status: {status}\n"
              f"Completed: {completed}/{total}\nFailures: {failures}\n"
              "End Time: 2026-07-16T04:12:22\n" + "=" * 50 + "\n")
    return block

log = _run_block("2026-07-16T02:31:44", "deepseek/deepseek-v4-pro", 2, "COMPLETE", 2, 0,
    [_entry(3, 1.0, 0, "[[0,0],[1,0],[2,0]]", "[[0,0],[1,0],[2,0]]"),
     _entry(5, 2.0, 1, "[[0,0],[1,1],[2,2]]", "[[0,0],[1,1],[2,2]]")])

lines = log.splitlines(keepends=True)
runs = []
current = None
for line in lines:
    stripped = line.strip()
    is_start = bool(RUN_START_RE.match(stripped))
    is_end = bool(RUN_END_RE.match(stripped))
    if is_start:
        print(f"START@{len(runs)}: {stripped!r}")
        if current is not None:
            runs.append(current)
        current = [line]
    elif is_end:
        print(f"END@{len(runs)}: {stripped!r}")
        if current is not None:
            current.append(line)
            runs.append(current)
            current = None
    else:
        if current is not None:
            current.append(line)
if current is not None:
    runs.append(current)
print("TOTAL RUNS:", len(runs))
