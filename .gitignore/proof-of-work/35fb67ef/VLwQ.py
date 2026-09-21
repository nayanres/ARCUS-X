import sys
sys.path.insert(0, 'c:/Users/games/python/arcus-x')
import post_run_analyzer as p

print("START", flush=True)
data = p.process_txt_file('outputs/absolute_raw_data_poolside_laguna_xs_2.1.txt')
print("PROBES PARSED:", len(data['results_matrix']), flush=True)
print("SAMPLE KEYS:", list(data['results_matrix'].keys())[:8], flush=True)
# Count how many entries have real (non-truncated) outputs
real = 0
for k, v in data['results_matrix'].items():
    if v.get('predicted_length', 0) > 0:
        real += 1
print("ENTRIES WITH PREDICTED PATH:", real, flush=True)
