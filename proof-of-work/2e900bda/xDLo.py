with open("post_run_analyzer.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the _split_runs function and the corrupted loop
# We'll replace from "    for line in lines:" up to the duplicate block end
start = None
for i, line in enumerate(lines):
    if line.rstrip() == "    for line in lines:":
        start = i
        break

# Find the end of the duplicate mess: the line "    # Parse the RUN START metadata"
end = None
for i, line in enumerate(lines):
    if line.rstrip() == "    # Parse the RUN START metadata for each segment.":
        end = i
        break

print(f"start={start}, end={end}")

correct_loop = '''    for line in lines:
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

'''

new_lines = lines[:start] + [correct_loop] + lines[end:]

with open("post_run_analyzer.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("REPAIRED LOOP")
