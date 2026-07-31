import post_run_analyzer as p
import tempfile, os

block = ("\n" + "=" * 50 + "\nARCUS-X RUN START\nRun ID: R1\nModel: m\nTotal Probes: 2\n" + "=" * 50 + "\n"
         "\n=== RAW STREAM ENTRY (Tokens: 10) ===\n"
         "Parameters: z=3, gravity=1.0, tier=0, grid=8x8\n"
         "Grid: 8x8\nInitialState: [0,0]\nActions: [[0,0],[1,0]]\n"
         "TransitionRules: step=1\nEnvironmentMetadata: gravity=1.0\n"
         "[MODEL OUTPUT]\n[[0,0],[1,0],[2,0]]\n[GROUND TRUTH EXPECTED]\n[[0,0],[1,0],[2,0]]\n"
         "\n" + "=" * 50 + "\nARCUS-X RUN END\nStatus: COMPLETE\nCompleted: 2/2\nFailures: 0\nEnd Time: t\n" + "=" * 50 + "\n")

fd, path = tempfile.mkstemp(suffix=".txt", prefix="arcus_test_")
with os.fdopen(fd, "w", encoding="utf-8") as f:
    f.write(block)
result = p.analyze_raw_log(path)
os.remove(path)
r = result["runs"][0]
print("status:", r["status"], "completed:", r["completed"], "total:", r["total"])
segs = p._split_runs(block)
seg_text = segs[0][1]
print("'Status: COMPLETE' in seg:", "Status: COMPLETE" in seg_text)
print("seg tail:", repr(seg_text[-80:]))
