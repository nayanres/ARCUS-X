import post_run_analyzer as p

def _entry(z, gravity, tier, model_path, truth_path, tokens=10):
    return (
        f"\n=== RAW STREAM ENTRY (Tokens: {tokens}) ===\n"
        f"Parameters: z={z}, gravity={gravity}, tier={tier}, grid=8x8\n"
        f"Grid: 8x8\n"
        f"InitialState: [0,0]\n"
        f"Actions: [[0,0],[1,0]]\n"
        f"TransitionRules: step=1\n"
        f"EnvironmentMetadata: gravity={gravity}\n"
        f"[MODEL OUTPUT]\n{model_path}\n"
        f"[GROUND TRUTH EXPECTED]\n{truth_path}\n"
    )

def _run_block(run_id, model, total, status, completed, failures, entries):
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
        f"End Time: 2026-07-16T04:12:22\n"
        "=" * 50 + "\n"
    )
    return block

log = _run_block("2026-07-16T02:31:44", "deepseek/deepseek-v4-pro", 2, "COMPLETE", 2, 0,
    [_entry(3, 1.0, 0, "[[0,0],[1,0],[2,0]]", "[[0,0],[1,0],[2,0]]"),
     _entry(5, 2.0, 1, "[[0,0],[1,1],[2,2]]", "[[0,0],[1,1],[2,2]]")])
log += _run_block("2026-07-16T05:00:00", "deepseek/deepseek-v4-pro", 4, "PARTIAL", 2, 1,
    [_entry(3, 1.0, 0, "[[0,0],[1,0],[2,0]]", "[[0,0],[1,0],[2,0]]"),
     _entry(5, 2.0, 1, "[[0,0],[9,9],[9,9]]", "[[0,0],[1,1],[2,2]]")])

runs = p._split_runs(log)
print("num runs:", len(runs))
for i, (m, t) in enumerate(runs):
    parsed = p.parse_raw_log_text(t)
    print(f"run {i}: meta={m} records={len(parsed['records'])} errors={parsed['parse_errors']}")
