import post_run_analyzer as p
import json

r = p.analyze_raw_log('outputs/absolute_raw_stream_deepseekv4pro.txt')
with open('outputs/_dbg_real.txt', 'w', encoding='utf-8') as f:
    f.write(f"status: {r['status']}\n")
    f.write(f"num runs: {len(r['runs'])}\n")
    f.write(f"agg accuracy: {r['metrics'].get('accuracy')}\n")
    f.write(f"completed: {r['metrics'].get('completed_probes')}\n")
    f.write(f"total: {r['metrics'].get('total_probes')}\n")
    if r['runs']:
        f.write(f"run0 keys: {list(r['runs'][0].keys())}\n")
        f.write(f"run0 metrics: {json.dumps(r['runs'][0]['metrics'], indent=2)}\n")
    f.write(f"error: {r.get('error')}\n")
