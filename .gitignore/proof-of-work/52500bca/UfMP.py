import post_run_analyzer as p

r = p.analyze_raw_log('outputs/absolute_raw_stream_deepseekv4pro.txt')
with open('outputs/_dbg_real3.txt', 'w', encoding='utf-8') as f:
    f.write(f"status: {r['status']}\n")
    f.write(f"num runs: {len(r['runs'])}\n")
    f.write(f"agg accuracy: {r['metrics'].get('accuracy')}\n")
    f.write(f"completed: {r['metrics'].get('completed_probes')}\n")
    f.write(f"total: {r['metrics'].get('total_probes')}\n")
    f.write(f"error: {r.get('error')}\n")
