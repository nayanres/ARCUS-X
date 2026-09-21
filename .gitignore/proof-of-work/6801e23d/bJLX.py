import post_run_analyzer as p

block = (
    "\n" + "=" * 50 + "\n"
    "ARCUS-X RUN START\n"
    "Run ID: x\n"
    "Model: m\n"
    "Total Probes: 2\n"
    + "=" * 50 + "\n"
    "=== RAW STREAM ENTRY (Tokens: 10) ===\n"
    "Parameters: z=3, gravity=1.0, tier=0, grid=8x8\n"
    "ARCUS-X RUN END\n"
    "Status: COMPLETE\n"
    "Completed: 2/2\n"
    "Failures: 0\n"
    "End Time: t\n"
    + "=" * 50 + "\n"
)
for line in block.splitlines(keepends=True):
    s = line.strip()
    if p.RUN_START_RE.match(s):
        print("START MATCH:", repr(s))
    elif p.RUN_END_RE.match(s):
        print("END MATCH:", repr(s))

