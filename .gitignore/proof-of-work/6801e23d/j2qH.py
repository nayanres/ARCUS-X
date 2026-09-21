import post_run_analyzer as p
import tempfile, os

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
              "End Time: 2026-07-16T04:12:22\n" + "=" * 50 + "\n")
    return block

log = _run_block("2026-07-16T02:31:44", "deepseek/deepseek-v4-pro", 2, "COMPLETE", 2, 0,
    [_entry(3, 1.0, 0, "[[0,0],[1,0],[2,0]]", "[[0,0],[1,0],[2,0]]"),
     _entry(5, 2.0, 1, "[[0,0],[1,1],[2,2]]", "[[0,0],[1,1],[2,2]]")])
log += _run_block("2026-07-16T05:00:00", "deepseek/deepseek-v4-pro", 4, "PARTIAL", 2, 1,
    [_entry(3, 1.0, 0, "[[0,0],[1,0],[2,0]]", "[[0,0],[1,0],[2,0]]"),
     _entry(5, 2.0, 1, "[[0,0],[9,9],[9,9]]", "[[0,0],[1,1],[2,2]]")])

fd, path = tempfile.mkstemp(suffix=".txt")
with os.fdopen(fd, "w", encoding="utf-8") as f:
    f.write(log)
result = p.analyze_raw_log(path)
os.remove(path)
print("RUNS:", len(result["runs"]))
for r in result["runs"]:
    print("  ", r["run_id"], r["status"], "acc=", r["metrics"]["accuracy"], "completed=", r["completed"])
print("AGG:", result["metrics"]["accuracy"], result["metrics"]["run_status"], "num_runs=", result["metrics"]["num_runs"])
