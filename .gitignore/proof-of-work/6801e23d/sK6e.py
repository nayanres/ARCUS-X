import post_run_analyzer as p

def _entry(z, g, t, mp, tp, tok=10):
    return (f"\n=== RAW STREAM ENTRY (Tokens: {tok}) ===\n"
            f"Parameters: z={z}, gravity={g}, tier={t}, grid=8x8\n"
            f"Grid: 8x8\nInitialState: [0,0]\nActions: [[0,0],[1,0]]\n"
            f"TransitionRules: step=1\nEnvironmentMetadata: gravity={g}\n"
            f"[MODEL OUTPUT]\n{mp}\n[GROUND TRUTH EXPECTED]\n{tp}\n")

def _rb(rid, model, total, status, completed, failures, entries):
    b = ("\n" + "=" * 50 + "\nARCUS-X RUN START\nRun ID: " + rid + "\nModel: " + model +
         "\nTotal Probes: " + str(total) + "\n" + "=" * 50 + "\n")
    b += "".join(entries)
    b += ("\n" + "=" * 50 + "\nARCUS-X RUN END\nStatus: " + status + "\nCompleted: " +
          str(completed) + "/" + str(total) + "\nFailures: " + str(failures) +
          "\nEnd Time: t\n" + "=" * 50 + "\n")
    return b

log = _rb("2026-07-16T02:31:44", "deepseek/deepseek-v4-pro", 2, "COMPLETE", 2, 0,
          [_entry(3, 1.0, 0, "[[0,0],[1,0],[2,0]]", "[[0,0],[1,0],[2,0]]"),
           _entry(5, 2.0, 1, "[[0,0],[1,1],[2,2]]", "[[0,0],[1,1],[2,2]]")])
log += _rb("2026-07-16T05:00:00", "deepseek/deepseek-v4-pro", 4, "PARTIAL", 2, 1,
           [_entry(3, 1.0, 0, "[[0,0],[1,0],[2,0]]", "[[0,0],[1,0],[2,0]]"),
            _entry(5, 2.0, 1, "[[0,0],[9,9],[9,9]]", "[[0,0],[1,1],[2,2]]")])

print("before parse_raw_log_text, segs:", len(p._split_runs(log)))
parsed = p.parse_raw_log_text(log)
print("after parse_raw_log_text, segs:", len(p._split_runs(log)))
print("parsed records:", len(parsed["records"]))
