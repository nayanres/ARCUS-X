import sys
sys.path.insert(0, 'tests')
# Simulate pytest's import of the test module
import importlib.util
spec = importlib.util.spec_from_file_location("test_multi_run_analyzer", "tests/test_multi_run_analyzer.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Run the first test manually
try:
    mod.test_split_runs_separates_segments()
    print("test_split_runs_separates_segments: PASSED")
except AssertionError as e:
    print("test_split_runs_separates_segments: FAILED", e)
except Exception as e:
    print("test_split_runs_separates_segments: ERROR", repr(e))
