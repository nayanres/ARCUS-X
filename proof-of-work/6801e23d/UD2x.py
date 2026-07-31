import post_run_analyzer as p

print("START flags:", p.RUN_START_RE.flags)
print("START pattern:", repr(p.RUN_START_RE.pattern))

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

log = _rb("R1", "m", 2, "COMPLETE", 2, 0,
          [_entry(3, 1.0, 0, "[[0,0],[1,0],[2,0]]", "[[0,0],[1,0],[2,0]]")])
log += _rb("R2", "m", 4, "PARTIAL", 2, 1,
           [_entry(3, 1.0, 0, "[[0,0],[1,0],[2,0]]", "[[0,0],[1,0],[2,0]]"),
            _entry(5, 2.0, 1, "[[0,0],[9,9],[9,9]]", "[[0,0],[1,1],[2,2]]")])

segs = p._split_runs(log)
print("NUM SEGS:", len(segs))
for m, s in segs:
    print("  meta:", m, "len:", len(s))
