import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import post_run_analyzer as p, json, traceback

content = open('outputs/absolute_raw_stream_deepseek_deepseek_v4_flash.txt', encoding='utf-8').read()
parts = p.ENTRY_HEADER_RE.split(content)
block = parts[2] if len(parts) > 2 else (parts[1] if len(parts) > 1 else "")

# Replicate the env-build with exception printing
import re
lines = block.splitlines()
grid_w = grid_h = None
rules_raw = actions_raw = env_meta_raw = None
for line in lines:
    stripped = line.strip()
    if stripped.startswith("Grid:"):
        m = p.GRID_RE.search(stripped)
        if m:
            grid_w = int(m.group(1)); grid_h = int(m.group(2))
    elif stripped.startswith("Actions:"):
        actions_raw = stripped[len("Actions:"):].strip()
    elif stripped.startswith("TransitionRules:"):
        rules_raw = stripped[len("TransitionRules:"):].strip()
    elif stripped.startswith("EnvironmentMetadata:"):
        env_meta_raw = stripped[len("EnvironmentMetadata:"):].strip()

print("grid_w/h:", grid_w, grid_h)
print("rules_raw:", repr(rules_raw)[:80])
print("env_meta_raw:", repr(env_meta_raw)[:80])
try:
    meta = {}
    for kv in env_meta_raw.split(","):
        kv = kv.strip()
        if "=" in kv:
            k, v = kv.split("=", 1)
            meta[k.strip()] = v.strip()
    print("parsed meta:", meta)
    env = {
        "transition_rules": json.loads(rules_raw),
        "actions": json.loads(actions_raw),
        "grid_width": grid_w,
        "grid_height": grid_h,
        "tier": int(meta.get("tier")),
        "gravity": float(meta.get("gravity")),
    }
    print("ENV OK:", env)
except Exception as e:
    traceback.print_exc()
