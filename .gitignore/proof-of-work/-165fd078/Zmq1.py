import io

p = "arcus/experiments/benchmark_runner.py"
s = io.open(p, "r", encoding="utf-8").read()

orig = s

# 1) Map ground_truth_trajectory -> ground_truth_path
old1 = '"ground_truth_path": getattr(task, "ground_truth_path", []),'
new1 = '"ground_truth_path": list(getattr(task, "ground_truth_trajectory", [])),'
if old1 in s:
    s = s.replace(old1, new1, 1)
else:
    print("WARN: old1 not found")

# 2) Add tier key after fol_depth
old2 = '"fol_depth": getattr(task, "horizon", 1),'
new2 = '"fol_depth": getattr(task, "horizon", 1),\n            "tier": getattr(task, "tier", 0),'
if old2 in s:
    s = s.replace(old2, new2, 1)
else:
    print("WARN: old2 not found")

# 3) run_baselines trajectory_accuracy * 100
old3 = '"trajectory_accuracy": mean(traj_accs) if traj_accs else 0.0,'
new3 = '"trajectory_accuracy": (mean(traj_accs) * 100.0) if traj_accs else 0.0,'
if old3 in s:
    s = s.replace(old3, new3, 1)
else:
    print("WARN: old3 not found")

if s != orig:
    io.open(p, "w", encoding="utf-8").write(s)
    print("PATCHED")
else:
    print("NO CHANGES (already patched or patterns missing)")

print("gt_traj:", "ground_truth_trajectory" in s)
print("tier_getattr:", '"tier": getattr' in s)
print("baselines_x100:", "mean(traj_accs) * 100.0" in s)
