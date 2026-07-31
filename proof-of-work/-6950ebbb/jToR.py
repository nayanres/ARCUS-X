"""Verify the two post_run_analyzer bug fixes."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import post_run_analyzer as pra

OLD_LOG = "outputs/absolute_raw_stream_deepseekv4pro.txt"

print("=" * 60)
print("TEST 1: Old log (no metadata header) - backwards compat")
print("=" * 60)
res = pra.analyze_raw_log(OLD_LOG)
m = res["metrics"]
print(f"  status={res['status']}")
print(f"  run_status={m['run_status']}")
print(f"  completed/total={m['completed_probes']} / {m['total_probes']}")
print(f"  accuracy={m['accuracy']}  (should be a PERCENTAGE, e.g. ~80.0)")
print(f"  per_horizon sample={list(m['per_horizon_accuracy'].items())[:3]}")
# Bug 2 check: overall accuracy must now be a PERCENTAGE (0-100), not a
# 0-1 fraction. The per-horizon values are fractions displayed as *100, so a
# correct overall of 80.0% lines up with per-horizon displays like 80%/100%/90%.
assert m["accuracy"] > 1.0, f"Bug2: accuracy still a fraction? {m['accuracy']}"
assert m["accuracy"] <= 100.0, f"Bug2: accuracy out of range? {m['accuracy']}"
print("  [PASS] Bug 2 fixed: overall accuracy is a percentage consistent with per-horizon")

print()
print("=" * 60)
print("TEST 2: Simulated PARTIAL run (ExpectedProbes: 540, 213 entries)")
print("=" * 60)
# Build a partial log from the old log's entries + a metadata header.
with open(OLD_LOG, "r", encoding="utf-8") as f:
    content = f.read()
partial_path = "outputs/_partial_test_log.txt"
with open(partial_path, "w", encoding="utf-8") as f:
    f.write("=== RUN METADATA ===\n")
    f.write("Model: deepseekv4pro\n")
    f.write("ExpectedProbes: 540\n")
    f.write("=" * 30 + "\n")
    f.write(content)

res = pra.analyze_raw_log(partial_path)
m = res["metrics"]
print(f"  run_status={m['run_status']}")
print(f"  completed/total={m['completed_probes']} / {m['total_probes']}")
assert m["total_probes"] == 540, f"Bug1: total should be 540, got {m['total_probes']}"
assert m["completed_probes"] == 213, f"Bug1: completed should be 213, got {m['completed_probes']}"
assert m["run_status"] == "PARTIAL", f"Bug1: should be PARTIAL, got {m['run_status']}"
print("  [PASS] Bug 1 fixed: reports 213 / 540 PARTIAL")

print()
print("=" * 60)
print("TEST 3: --expected override")
print("=" * 60)
res = pra.analyze_raw_log(OLD_LOG, expected_override=540)
m = res["metrics"]
print(f"  run_status={m['run_status']}  completed/total={m['completed_probes']}/{m['total_probes']}")
assert m["total_probes"] == 540
assert m["run_status"] == "PARTIAL"
print("  [PASS] --expected override works")

os.remove(partial_path)
print("\nALL CHECKS PASSED")
