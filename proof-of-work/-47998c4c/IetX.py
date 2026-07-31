import re

with open("post_run_analyzer.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the analyze_raw_log function start
start = None
for i, line in enumerate(lines):
    if line.startswith("def analyze_raw_log("):
        start = i
        break

# Find the line with "def _mode(" which marks the end of analyze_raw_log
end = None
for i, line in enumerate(lines):
    if line.startswith("def _mode("):
        end = i
        break

print("analyze_raw_log starts at line", start + 1)
print("_mode starts at line", end + 1)

# The correct body for analyze_raw_log (from the docstring end to before _mode)
correct_body = '''    try:
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:  # pragma: no cover - defensive
        return {"status": "FAILED", "error": str(e), "metrics": {}, "runs": []}

    run_segments = _split_runs(content)

    # No RUN START markers at all -> treat the whole file as a single legacy run.
    if not run_segments:
        seg_metrics = _analyze_segment(content, {}, expected_override=expected_override)
        return {
            "status": "COMPLETE",
            "error": None,
            "metrics": seg_metrics,
            "runs": [{
                "run_id": None,
                "model": seg_metrics["model"],
                "status": seg_metrics["run_status"],
                "completed": seg_metrics["completed_probes"],
                "total": seg_metrics["total_probes"],
                "metrics": seg_metrics,
            }],
        }

    runs = []
    for start_meta, seg_text in run_segments:
        seg_metrics = _analyze_segment(seg_text, start_meta, expected_override=expected_override)
        # Prefer the explicit RUN END status if present in the segment.
        m_status = RUN_STATUS_RE.search(seg_text)
        m_completed = RUN_COMPLETED_RE.search(seg_text)
        m_failures = RUN_FAILURES_RE.search(seg_text)
        m_reason = RUN_REASON_RE.search(seg_text)
        m_end = RUN_END_TIME_RE.search(seg_text)
        status = m_status.group(1).strip() if m_status else seg_metrics["run_status"]
        completed = int(m_completed.group(1)) if m_completed else seg_metrics["completed_probes"]
        total = int(m_completed.group(2)) if m_completed else seg_metrics["total_probes"]
        failures = int(m_failures.group(1)) if m_failures else seg_metrics["failure_count"]
        runs.append({
            "run_id": start_meta.get("run_id"),
            "model": start_meta.get("model") or seg_metrics["model"],
            "status": status,
            "completed": completed,
            "total": total,
            "failures": failures,
            "reason": m_reason.group(1).strip() if m_reason else "",
            "end_time": m_end.group(1).strip() if m_end else None,
            "metrics": seg_metrics,
        })

    # Aggregate view across all runs (weighted by completed probe count).
    total_completed = sum(r["completed"] for r in runs)
    total_intended = sum(r["total"] for r in runs)
    if total_completed > 0:
        agg_accuracy = round(
            sum(r["metrics"]["accuracy"] * r["completed"] for r in runs) / total_completed, 1
        )
    else:
        agg_accuracy = 0.0
    any_partial = any(r["status"] != "COMPLETE" for r in runs)
    agg_status = "PARTIAL" if any_partial else "COMPLETE"

    aggregate = {
        "model": runs[0]["model"] if runs else "unknown",
        "run_status": agg_status,
        "completed_probes": total_completed,
        "total_probes": total_intended,
        "accuracy": agg_accuracy,
        "num_runs": len(runs),
        "runs_summary": [
            {
                "run_id": r["run_id"],
                "model": r["model"],
                "status": r["status"],
                "completed": r["completed"],
                "total": r["total"],
                "failures": r["failures"],
                "accuracy": r["metrics"]["accuracy"],
            }
            for r in runs
        ],
    }

    return {"status": "COMPLETE", "error": None, "metrics": aggregate, "runs": runs}


'''

# Replace lines[start+1 : end] (i.e., after the def line, up to _mode)
new_lines = lines[:start + 1] + [correct_body] + lines[end:]

with open("post_run_analyzer.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Fixed. New line count:", len(new_lines))
