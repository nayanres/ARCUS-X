import os, time, dis, marshal

py = os.path.getmtime('post_run_analyzer.py')
pyc = os.path.getmtime('__pycache__/post_run_analyzer.cpython-311.pyc')

with open('outputs/_dbg_pyc.txt', 'w') as f:
    f.write(f"py mtime: {time.ctime(py)}\n")
    f.write(f"pyc mtime: {time.ctime(pyc)}\n")
    f.write(f"pyc is newer: {pyc > py}\n")

    # Try to load the pyc and check if _split_runs and _parse_entry_block exist
    try:
        with open('__pycache__/post_run_analyzer.cpython-311.pyc', 'rb') as fh:
            fh.read(16)  # skip header
            code = marshal.load(fh)
        names = [c.co_name for c in code.co_consts if hasattr(c, 'co_name')]
        f.write(f"functions in pyc: {names}\n")
        f.write(f"has _split_runs: {'_split_runs' in names}\n")
        f.write(f"has _parse_entry_block: {'_parse_entry_block' in names}\n")
    except Exception as e:
        f.write(f"ERROR loading pyc: {e}\n")
