import post_run_analyzer as p

# Verify real log still works
r = p.analyze_raw_log('outputs/absolute_raw_stream_deepseekv4pro.txt')
with open('outputs/_dbg_final.txt', 'w', encoding='utf-8') as f:
    f.write(f"REAL LOG: status={r['status']} runs={len(r['runs'])} acc={r['metrics'].get('accuracy')} completed={r['metrics'].get('completed_probes')} total={r['metrics'].get('total_probes')} error={r.get('error')}\n")

# Verify multi-run on a synthetic 2-run log
def _entry(z, g, t, m, tr):
    return f"\n=== RAW STREAM ENTRY (Tokens: 10) ===\nParameters: z={z}, gravity={g}, tier={t}, grid=default\n------------------------------\n[MODEL OUTPUT]:\n{m}\n\n[GROUND TRUTH EXPECTED]:\n{tr}\n\n==============================\n"

SEP = "=" * 50
run1 = f"\n{SEP}\nARCUS-X RUN START\nRun ID: r1\nModel: m/x\nTotal Probes: 2\n{SEP}\n" + _entry(3,0,0,"[0,0]->[1,0]","['[0,0]', '[1,0]'") + _entry(3,0,0,"[0,0]->[1,0]","['[0,0]', '[1,0]'") + f"\n{SEP}\nARCUS-X RUN END\nStatus: COMPLETE\nCompleted: 2/2\nFailures: 0\nEnd Time: t\n{SEP}\n"
run2 = f"\n{SEP}\nARCUS-X RUN START\nRun ID: r2\nModel: m/x\nTotal Probes: 2\n{SEP}\n" + _entry(3,0,0,"[0,0]->[5,5]","['[0,0]', '[1,0]'") + _entry(3,0,0,"[0,0]->[5,5]","['[0,0]', '[1,0]'") + f"\n{SEP}\nARCUS-X RUN END\nStatus: COMPLETE\nCompleted: 2/2\nFailures: 0\nEnd Time: t\n{SEP}\n"

import tempfile, os
fd, path = tempfile.mkstemp(suffix=".txt")
with os.fdopen(fd, "w", encoding="utf-8") as fh:
    fh.write(run1 + run2)
try:
    r2 = p.analyze_raw_log(path)
    with open('outputs/_dbg_final.txt', 'a', encoding='utf-8') as f:
        f.write(f"MULTI-RUN: runs={len(r2['runs'])} agg_acc={r2['metrics']['accuracy']} num_runs={r2['metrics']['num_runs']}\n")
        for i, run in enumerate(r2['runs']):
            f.write(f"  run{i}: status={run['status']} completed={run['completed']} acc={run['metrics']['accuracy']}\n")
finally:
    os.remove(path)
