import tempfile, os
from post_run_analyzer import analyze_raw_log

def _entry(z, gravity, tier, model_path, truth_path, tokens=10):
    return (f"\n=== RAW STREAM ENTRY (Tokens: {tokens}) ===\n"
            f"Parameters: z={z}, gravity={gravity}, tier={tier}, grid=8x8\n"
            f"Grid: 8x8\nInitialState: [0,0]\nActions: [[0,0],[1,0]]\n"
            f"TransitionRules: step=1\nEnvironmentMetadata: gravity={gravity}\n"
            f"[MODEL OUTPUT]\n{model_path}\n[GROUND TRUTH EXPECTED]\n{truth_path}\n")

def _run_block(run_id, model, total, status, completed, failures, entries):
    block = ("\n" + "=" * 50 + "\nARCUS-X RUN START\n" f"Run ID: {run_id}\n"
             f"Model: {model}\n" f"Total Probes: {total}\n" + "=" * 50 + "\n")
    block += "".join(entries)
    block += ("\n" + "=" * 50 + "\nARCUS-X RUN END\n" f"Status: {status}\n"
              f"Completed: {completed}/{total}\nFailures: {failures}\n"
              "End Time: t\n" + "=" * 50 + "\n")
    return block

def test_dbg():
    log = _run_block("R1", "m", 2, "COMPLETE", 2, 0,
        [_entry(3, 1.0, 0, "[[0,0],[1,0],[2,0]]", "[[0,0],[1,0],[2,0]]")])
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="arcus_test_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(log)
    try:
        result = analyze_raw_log(path)
    finally:
        os.remove(path)
    print("NUM RUNS:", len(result["runs"]))
    print("FIRST RUN:", result["runs"][0] if result["runs"] else None)
    assert len(result["runs"]) == 1
