with open("post_run_analyzer.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add debug print after "stripped = line.strip()"
old = """    for line in lines:
        stripped = line.strip()
        if RUN_START_RE.match(stripped):"""
new = """    for line in lines:
        stripped = line.strip()
        import sys
        print(f"DBG line={stripped!r} start={bool(RUN_START_RE.match(stripped))} end={bool(RUN_END_RE.match(stripped))} sep={bool(SEP_RE.match(stripped))} cur={current is not None} ended={current['ended'] if current else None}", file=sys.stderr)
        if RUN_START_RE.match(stripped):"""

if old in content:
    content = content.replace(old, new, 1)
    with open("post_run_analyzer.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("patched")
else:
    print("PATTERN NOT FOUND")
