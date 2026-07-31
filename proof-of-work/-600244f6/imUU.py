import sys, pathlib
sys.path.insert(0, ".")
from tests.test_parallel_resume import _make_runner

def run(mode, out_path):
    tmp = pathlib.Path(tempfile.mkdtemp())
    r, raw = _make_runner(tmp, [42, 123, 456])
    r.run(output_filepath=str(tmp / "out.json"), parallel_seeds=mode)
    data = raw.read_text(encoding="utf-8")
    pathlib.Path(out_path).write_text(data, encoding="utf-8")
    return data

import tempfile
serial = run("1", "outputs/_serial_raw.txt")
parallel = run("auto", "outputs/_parallel_raw.txt")
print("serial RUN END count:", serial.count("ARCUS-X RUN END"))
print("parallel RUN END count:", parallel.count("ARCUS-X RUN END"))
print("serial last 200 chars:\n", serial[-200:])
print("parallel last 200 chars:\n", parallel[-200:])
