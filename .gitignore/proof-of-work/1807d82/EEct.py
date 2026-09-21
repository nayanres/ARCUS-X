with open("post_run_analyzer.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find the broken section and replace it with the correct version
broken = """    for line in lines:
        stripped = line.strip()
        import sys
        print(f"DBG line={stripped!r} start={bool(RUN_START_RE.match(stripped))} end={bool(RUN_END_RE.match(stripped))} sep={bool(SEP_RE.match(stripped))} cur={current is not None} ended={current['ended'] if current else None}", file=sys.stderr)
        if RUN_START_RE.match(stripped):"""

fixed = """    for line in lines:
        stripped = line.strip()
        if RUN_START_RE.match(stripped):"""

if broken in content:
    content = content.replace(broken, fixed, 1)
    with open("post_run_analyzer.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("REPAIRED")
else:
    print("BROKEN PATTERN NOT FOUND - showing context")
    idx = content.find("stripped = line.strip()")
    print(repr(content[idx-50:idx+300]))
