import post_run_analyzer as p

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

log = _run_block("2026-07-16T02:31:44", "deepseek/deepseek-v4-pro", 2, "COMPLETE", 2, 0,
    [_entry(3, 0.0, 0, "[0,0]->[1,0]->[2,0]", "['[0,0]', '[1,0]', '[2,0]']"),
     _entry(5, 0.5, 1, "[0,0]->[1,1]->[2,2]", "['[0,0]', '[1,1]', '[2,2]']")])
log += _run_block("2026-07-16T05:00:00", "deepseek/deepseek-v4-pro", 4, "PARTIAL", 2, 1,
    [_entry(3, 0.0, 0, "[0,0]->[1,0]->[2,0]", "['[0,0]', '[1,0]', '[2,0]']"),
     _entry(5, 0.5, 1, "[0,0]->[9,9]->[9,9]", "['[0,0]', '[1,1]', '[2,2]']")])

segs = p._split_runs(log)
with open("outputs/_dbg_split.txt", "w", encoding="utf-8") as f:
    f.write(f"NUM SEGS: {len(segs)}\n")
    for i, (meta, text) in enumerate(segs):
        f.write(f"\n--- SEG {i} meta={meta} len={len(text)} ---\n")
        f.write(text[:200])
        f.write("\n")
