import post_run_analyzer as p

r = p.parse_raw_log('outputs/_test_newfmt_log.txt')
for i, rec in enumerate(r['records']):
    mode = p._classify_with_taxonomy(rec)
    print(f"entry {i}: mode={mode}")
