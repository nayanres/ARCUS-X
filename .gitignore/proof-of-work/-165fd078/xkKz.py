import io

p = "arcus/experiments/benchmark_runner.py"
s = io.open(p, "r", encoding="utf-8").read()
orig = s

# exact_match * 100 (0-100 scale for report, matching original)
old = '"exact_match": exact_acc,'
new = '"exact_match": (exact_acc * 100.0),'
if old in s:
    s = s.replace(old, new, 1)
    print("exact_match patched")
else:
    print("WARN: exact_match pattern not found")

if s != orig:
    io.open(p, "w", encoding="utf-8").write(s)
    print("WRITTEN")
else:
    print("NO CHANGE")
