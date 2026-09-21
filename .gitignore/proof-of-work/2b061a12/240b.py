with open("post_run_analyzer.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the line "def _analyze_segment" and insert _classify_with_taxonomy before it
insert_idx = None
for i, line in enumerate(lines):
    if line.startswith("def _analyze_segment"):
        insert_idx = i
        break

print(f"insert_idx={insert_idx}")

helper = '''def _classify_with_taxonomy(record: dict):
    """Classify a single parsed record with the evidence-based taxonomy.

    Returns a :class:`TaxonomyResult` on success, or ``None`` when the record
    lacks the environment metadata required for classification (e.g. legacy
    logs that only carry Parameters / MODEL OUTPUT / GROUND TRUTH).
    """
    env = record.get("env")
    if not env:
        return None
    try:
        # Reconstruct canonical [x,y] token lists from the parsed coord tuples.
        model_tokens = [f"[{r},{c}]" for r, c in record.get("_model_coords", [])]
        truth_tokens = [f"[{r},{c}]" for r, c in record.get("_truth_coords", [])]
        result = classify_from_result(
            model_tokens,
            truth_tokens,
            env["transition_rules"],
            env["actions"],
            int(record["tier"]),
            env["grid_width"],
            env["grid_height"],
            raw_output=record.get("_raw_output"),
        )
        if result.valid:
            return result
    except Exception:
        return None
    return None


'''

new_lines = lines[:insert_idx] + [helper] + lines[insert_idx:]

with open("post_run_analyzer.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("ADDED _classify_with_taxonomy")
