import post_run_analyzer as p

entry = lambda mp, tp: (
    f"\n=== RAW STREAM ENTRY (Tokens: 10) ===\n"
    f"Parameters: z=3, gravity=1.0, tier=0, grid=8x8\n"
    f"Grid: 8x8\n"
    f"InitialState: [0,0]\n"
    f"Actions: [[0,0],[1,0]]\n"
    f"TransitionRules: step=1\n"
    f"EnvironmentMetadata: gravity=1.0\n"
    f"[MODEL OUTPUT]\n{mp}\n"
    f"[GROUND TRUTH EXPECTED]\n{tp}\n"
)
block = (
    "\n" + "=" * 50 + "\n"
    "ARCUS-X RUN START\n"
    "Run ID: x\n"
    "Model: deepseek/deepseek-v4-pro\n"
    "Total Probes: 2\n"
    + "=" * 50 + "\n"
)
block += entry("[[0,0],[1,0],[2,0]]", "[[0,0],[1,0],[2,0]]")
block += entry("[[0,0],[1,0],[2,0]]", "[[0,0],[1,0],[2,0]]")
block += (
    "\n" + "=" * 50 + "\n"
    "ARCUS-X RUN END\n"
    "Status: COMPLETE\n"
    "Completed: 2/2\n"
    "Failures: 0\n"
    "End Time: t\n"
    + "=" * 50 + "\n"
)
runs = p._split_runs(block)
print("num runs:", len(runs))
for m, t in runs:
    print("meta:", m)
    parsed = p.parse_raw_log_text(t)
    print("  records:", len(parsed["records"]), "errors:", parsed["parse_errors"])
