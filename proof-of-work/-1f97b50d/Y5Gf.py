import io

p = "arcus/experiments/benchmark_runner.py"
s = io.open(p, "r", encoding="utf-8").read()
if "import hashlib" not in s:
    s = s.replace("import os\nimport json\n", "import os\nimport json\nimport hashlib\n", 1)
    io.open(p, "w", encoding="utf-8").write(s)
    print("hashlib import added")
else:
    print("hashlib already imported")
print("has hashlib:", "import hashlib" in s)
