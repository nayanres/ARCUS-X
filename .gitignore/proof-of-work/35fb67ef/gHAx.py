import sys
sys.path.insert(0, 'c:/Users/games/python/arcus-x')
import post_run_analyzer as p

data = p.process_txt_file('outputs/absolute_raw_data_poolside_laguna_xs_2.1.txt')
print("PROBES PARSED:", len(data['results_matrix']))
real = sum(1 for v in data['results_matrix'].values() if v.get('predicted_length',0) > 0)
print("ENTRIES WITH PREDICTED PATH:", real)
print("SAMPLE KEYS:", list(data['results_matrix'].keys())[:5])
