import sys, os, io, contextlib, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcus.experiments.benchmark_runner import BenchmarkRunner


class StubEval:
    def __init__(self, *a, **k):
        pass

    def evaluate_single_probe(self, probe, tier, depth=1):
        return {
            'step_accuracy': 0.9 if tier == 0 else 0.5,
            'exact_match': tier == 0,
            'continuity_score': 0.8,
            'predicted_length': depth,
            'ground_truth_length': depth,
            'horizon_compliance': 1.0,
            'generation_bloat_index': 0.1,
            'generation_efficiency': 0.9,
            'total_tokens': depth * 10,
            'output_tokens': depth * 10,
            'completion_tokens': depth * 10,
            'fol_depth': depth,
            'fracture_depth': 0,
            'error_mode': 'None',
            'raw_output': 'x',
            'finish_reason': 'stop',
        }


def main():
    r = BenchmarkRunner(
        model_name='deepseek/deepseek-v4-pro',
        evaluator_class=StubEval,
        n_probes=1,
        gravity_levels=[0.0, 1.0],
        seed=42,
    )
    r.results_matrix = {
        (1, 0.0, 0, 0, 'default'): {'step_accuracy': 0.9, 'exact_match': True, 'continuity_score': 0.8, 'fracture_depth': 0, 'fol_depth': 1, 'total_tokens': 10, 'output_tokens': 10, 'completion_tokens': 10, 'horizon_compliance': 1.0, 'generation_bloat_index': 0.1, 'generation_efficiency': 0.9, 'error_mode': 'None', 'finish_reason': 'stop'},
        (2, 0.0, 0, 0, 'default'): {'step_accuracy': 0.85, 'exact_match': False, 'continuity_score': 0.7, 'fracture_depth': 0, 'fol_depth': 2, 'total_tokens': 20, 'output_tokens': 20, 'completion_tokens': 20, 'horizon_compliance': 1.0, 'generation_bloat_index': 0.1, 'generation_efficiency': 0.9, 'error_mode': 'None', 'finish_reason': 'stop'},
        (1, 1.0, 1, 0, 'default'): {'step_accuracy': 0.5, 'exact_match': False, 'continuity_score': 0.4, 'fracture_depth': 1, 'fol_depth': 1, 'total_tokens': 10, 'output_tokens': 10, 'completion_tokens': 10, 'horizon_compliance': 1.0, 'generation_bloat_index': 0.1, 'generation_efficiency': 0.9, 'error_mode': 'None', 'finish_reason': 'stop'},
    }
    r.actual_probe_count = 3
    r.expected_probe_count = 3
    r.run_status = 'COMPLETE'
    agg = r._compute_aggregated_metrics()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r._print_summary_report(agg)
        r._export_metrics_json(agg)
    out = buf.getvalue()
    with open('outputs/_test_summary.txt', 'w') as f:
        f.write(out)

    print('SUMMARY_LEN', len(out))
    print('JSON_EXISTS', os.path.exists('outputs/metrics_deepseek-deepseek-v4-pro.json'))

    r.actual_probe_count = 1
    r.expected_probe_count = 3
    r.run_status = 'PARTIAL'
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        r._print_partial_run_report()
    with open('outputs/_test_partial.txt', 'w') as f:
        f.write(buf2.getvalue())
    print('PARTIAL_LEN', len(buf2.getvalue()))


if __name__ == '__main__':
    main()
