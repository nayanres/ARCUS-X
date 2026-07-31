class E:
    def __init__(self, *a, **k):
        pass

from arcus.experiments.benchmark_runner import BenchmarkRunner

r = BenchmarkRunner(model_name='x', evaluator_class=E, seeds=[42, 1337, 2026, 9001, 123456])
print('seeds stored:', r.seeds)
assert r.seeds == [42, 1337, 2026, 9001, 123456]
print('OK: seeds are consumed by the runner')
