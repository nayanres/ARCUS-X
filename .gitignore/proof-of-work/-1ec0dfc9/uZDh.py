import sys
sys.path.insert(0, 'tests')
import post_run_analyzer as p
print('FILE:', p.__file__)
print('has _split_runs:', hasattr(p, '_split_runs'))
import inspect
src = inspect.getsource(p._split_runs)
print('SEP_RE in _split_runs:', 'SEP_RE' in src)
print('ended flag in _split_runs:', 'ended' in src)
