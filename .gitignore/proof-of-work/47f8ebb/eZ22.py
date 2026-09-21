with open("post_run_analyzer.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the corrupted region: from "def _parse_entry_block" to just before
# "def _analyze_segment" (line 250). We'll replace lines 186..249 (0-indexed 185..248)
# with the correct _parse_entry_block + _split_runs functions.

start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if line.startswith("def _parse_entry_block"):
        start_idx = i
    if line.startswith("def _analyze_segment"):
        end_idx = i
        break

print(f"start_idx={start_idx}, end_idx={end_idx}")

correct_functions = '''def _parse_entry_block(tokens_str: str, block: str):
    """Parse a single entry block (everything between two headers)."""
    tokens = int(tokens_str)
    lines = block.splitlines()

    z = gravity = tier = grid = None
    grid_w = grid_h = None
    initial_state_raw = None
    actions_raw = rules_raw = None
    env_meta_raw = None
    provider_metrics = None
    model_output = None
    ground_truth = None

    section = None  # "model" or "truth"
    model_lines = []
    truth_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Parameters:"):
            # Parameters: z=3, gravity=0.0, tier=1, grid=default
            for kv in stripped[len("Parameters:"):].split(","):
                kv = kv.strip()
                if kv.startswith("z="):
                    try:
                        z = int(kv[2:])
                    except ValueError:
                        z = None
                elif kv.startswith("gravity="):
                    try:
                        gravity = float(kv[len("gravity="):])
                    except ValueError:
                        gravity = None
                elif kv.startswith("tier="):
                    try:
                        tier = int(kv[len("tier="):])
                    except ValueError:
                        tier = None
                elif kv.startswith("grid="):
                    grid = kv[len("grid="):]
        elif stripped.startswith("[MODEL OUTPUT]:"):
            section = "model"
            continue
        elif stripped.startswith("[GROUND TRUTH EXPECTED]:"):
            section = "truth"
            continue
        elif stripped.startswith("Initial State:"):
            initial_state_raw = stripped[len("Initial State:"):].strip()
        elif stripped.startswith("Actions:"):
            actions_raw = stripped[len("Actions:"):].strip()
        elif stripped.startswith("Transition Rules:"):
            rules_raw = stripped[len("Transition Rules:"):].strip()
        elif stripped.startswith("Grid:"):
            env_meta_raw = stripped[len("Grid:"):].strip()
        elif stripped.startswith("Provider Metrics:"):
            try:
                import json
                provider_metrics = json.loads(stripped[len("Provider Metrics:"):].strip())
            except Exception:
                provider_metrics = None
        else:
            if section == "model":
                model_lines.append(stripped)
            elif section == "truth":
                truth_lines.append(stripped)

    model_output = "\\n".join(model_lines).strip()
    ground_truth = "\\n".join(truth_lines).strip()

    # Model path: support both arrow format "[0,1]->[0,0]" and list format.
    if "->" in model_output:
        model_path = _parse_coord_list(model_output)
    else:
        model_path = _parse_coord_list(model_output)

    truth_path = _parse_coord_list(ground_truth)

    empty = (not model_output) or model_output.upper().startswith("[EMPTY")

    step_accuracy = _step_accuracy(model_path, truth_path) if not empty else 0.0

    # Reconstruct a minimal env dict if grid metadata is present; otherwise None.
    env = None
    if env_meta_raw or rules_raw or actions_raw:
        try:
            import json
            if env_meta_raw:
                meta = json.loads(env_meta_raw)
                grid_w = meta.get("width")
                grid_h = meta.get("height")
            env = {
                "transition_rules": json.loads(rules_raw) if rules_raw else {},
                "actions": json.loads(actions_raw) if actions_raw else [],
                "grid_width": grid_w,
                "grid_height": grid_h,
            }
        except Exception:
            env = None

    return {
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
    }


def _split_runs(content: str) -> list:
    """Split a raw log into independent run segments using a state machine.

    The runner appends an ``ARCUS-X RUN START`` block before each run and an
    ``ARCUS-X RUN END`` block after it. We walk the lines and collect everything
    between a START and its matching END into one segment. This keeps multiple
    runs of the same model (or different models in one file) cleanly separated
    so old/new experiments never bleed into each other's metrics.

    Returns a list of ``(start_meta, segment_text)`` tuples, where ``start_meta``
    is a dict parsed from the RUN START block (run_id, model, total_probes) and
    ``segment_text`` is the raw text of that run (entries + RUN END block).
    """
    lines = content.splitlines(keepends=True)
    runs = []
    current = None  # dict: {"meta": {...}, "lines": [...], "ended": bool}
    SEP_RE = re.compile(r"^=+\\s*$")
    for line in lines:
        stripped = line.strip()
        if RUN_START_RE.match(stripped):
            # Close any previously-open run that lacked an END marker (treat the
            # stray START as the start of a new run; the prior one is kept as-is).
            if current is not None:
                runs.append((current["meta"], "".join(current["lines"])))
            current = {"meta": {}, "lines": [line], "ended": False}
        elif RUN_END_RE.match(stripped):
            if current is not None:
                current["lines"].append(line)
                current["ended"] = True
            else:
                # END without a preceding START: ignore (defensive).
                continue
        else:
            if current is not None:
                current["lines"].append(line)
                # Once the RUN END marker is seen, the run stays open only until
                # the closing separator line (or the next START) so the trailing
                # Status / Completed / Failures / End Time metadata is captured.
                if current["ended"] and SEP_RE.match(stripped):
                    runs.append((current["meta"], "".join(current["lines"])))
                    current = None
            # Lines outside any run block are ignored for per-run analysis.

    # A run that started but never ended (crash / truncated log) is still useful:
    # keep it so PARTIAL detection can flag it. Its END metadata will be absent.
    if current is not None:
        runs.append((current["meta"], "".join(current["lines"])))

    # Parse the RUN START metadata for each segment.
    parsed_runs = []
    for meta, text in runs:
        m_id = RUN_ID_RE.search(text)
        m_model = RUN_MODEL_RE.search(text)
        m_total = RUN_TOTAL_RE.search(text)
        start_meta = {
            "run_id": m_id.group(1).strip() if m_id else None,
            "model": m_model.group(1).strip() if m_model else None,
            "total_probes": int(m_total.group(1)) if m_total else None,
        }
        parsed_runs.append((start_meta, text))
    return parsed_runs


'''

new_lines = lines[:start_idx] + [correct_functions] + lines[end_idx:]

with open("post_run_analyzer.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("RECONSTRUCTED")
