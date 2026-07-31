import re, sys
sys.path.insert(0, ".")
from tests.test_parallel_resume import _make_runner

def run(mode):
    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp())
    r, raw = _make_runner(tmp, [42, 123, 456])
    r.run(output_filepath=str(tmp / "out.json"), parallel_seeds=mode)
    return raw.read_text(encoding="utf-8")

def parse(text):
    out = {}
    blocks = text.split("=== RAW STREAM ENTRY")
    for b in blocks:
        if "Parameters:" not in b:
            continue
        params = re.search(r"Parameters: z=(\d+), gravity=([\d.]+), tier=(\d+), grid=(\S+)", b)
        meta = re.search(r"EnvironmentMetadata: seed=(\d+), tier=(\d+), gravity=([\d.]+), z=(\d+), grid_key=(\S+), tier_name=(\S+), experiment_hash=(\S+)", b)
        actions = re.search(r"Actions: (\[.*?\])\n", b)
        if not (params and meta):
            continue
        z = int(params.group(1)); gravity = float(params.group(2)); tier = int(params.group(3)); grid = params.group(4)
        seed = int(meta.group(1)); mz = int(meta.group(4)); mgrid = meta.group(5); h = meta.group(7)
        key = (seed, tier, gravity, z, grid, actions.group(1) if actions else "")
        out[key] = h
    return out

serial = parse(run("1"))
parallel = parse(run("auto"))

print("serial entries:", len(serial), "parallel entries:", len(parallel))
only_serial = set(serial) - set(parallel)
only_parallel = set(parallel) - set(serial)
print("only in serial:", len(only_serial), "only in parallel:", len(only_parallel))
diff = 0
for k in serial:
    if k in parallel and serial[k] != parallel[k]:
        diff += 1
        if diff <= 5:
            print("HASH DIFF:", k[:5], "serial=", serial[k][:12], "par=", parallel[k][:12])
print("hash diffs:", diff)
