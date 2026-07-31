with open("post_run_analyzer.py", "r", encoding="utf-8") as f:
    content = f.read()

old_return = '''    return {
        "z": z,
        "gravity": gravity,
        "tier": tier,
        "grid": grid,
        "tokens": tokens,
        "empty": empty,
        "model_output": model_output,
        "ground_truth": ground_truth,
        "step_accuracy": step_accuracy,
        "_model_coords": model_path,
        "_truth_coords": truth_path,
        "env": env,
        "provider_metrics": provider_metrics,
        "_raw_output": model_output,
    }'''

new_return = '''    return {
        "z": z,
        "gravity": gravity,
        "tier": tier,
        "grid": grid,
        "tokens": tokens,
        "empty": empty,
        "model_output": model_output,
        "ground_truth": ground_truth,
        "step_accuracy": step_accuracy,
        "model_path": model_path,
        "truth_path": truth_path,
        "model_path_len": len(model_path),
        "truth_path_len": len(truth_path),
        "_model_coords": model_path,
        "_truth_coords": truth_path,
        "env": env,
        "provider_metrics": provider_metrics,
        "_raw_output": model_output,
    }'''

if old_return in content:
    content = content.replace(old_return, new_return, 1)
    with open("post_run_analyzer.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("FIXED _parse_entry_block return")
else:
    print("PATTERN NOT FOUND")
