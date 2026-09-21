import re, sys, tempfile, pathlib
sys.path.insert(0, ".")
from tests.test_parallel_resume import _make_runner

def run(mode):
    tmp = pathlib.Path(tempfile.mkdtemp())
    r, raw = _make_runner(tmp, [42, 123, 456])
    r.run(output_filepath=str(tmp / "out.json"), parallel_seeds=mode)
    return raw.read_text(encoding="utf-8")

def entries(text):
    blocks = text.split("=== RAW STREAM ENTRY")
    norm = []
    for b in blocks:
        if "Parameters:" not in b:
            continue
        kept = [ln for ln in b.splitlines() if not ln.strip().startswith("End Time:")]
        norm.append("\n".join(kept))
    return sorted(norm)

serial = entries(run("1"))
parallel = entries(run("auto"))
print("serial:", len(serial), "parallel:", len(parallel))
if serial == parallel:
    print("FULL BLOCKS IDENTICAL (after End Time strip)")
else:
    for i, (s, p) in enumerate(zip(serial, parallel)):
        if s != p:
            print("FIRST DIFF AT INDEX", i)
            sl = s.splitlines(); pl = p.splitlines()
            diffs = 0
            for j in range(max(len(sl), len(pl))):
                a = sl[j] if j < len(sl) else "<none>"
                b = pl[j] if j < len(pl) else "<none>"
                if a != b:
                    diffs += 1
                    if diffs <= 12:
                        print(f"  line {j}: SERIAL={a!r}")
                        print(f"  line {j}: PARALL={b!r}")
            print("  total differing lines in this block:", diffs)
            break
