import io

p = "arcus/experiments/benchmark_runner.py"
s = io.open(p, "r", encoding="utf-8").read()

# Fix _task_index_for to use hashlib.sha256 directly (original behavior).
old = '        h = hashlib_sha256(f"{probe_id}|{tier}|{z}|{self.seed}".encode("utf-8"))\n        return int(h[:8], 16) % 100000'
new = '        h = hashlib.sha256(f"{probe_id}|{tier}|{z}|{self.seed}".encode("utf-8")).hexdigest()[:8]\n        return int(h, 16) % 100000'
assert old in s, "task_index old not found"
s = s.replace(old, new, 1)

# Remove the broken helper function definition (keep file clean).
old_helper = '''def hashlib_sha256(text: str) -> str:
    """Deterministic hex digest helper (replaces non-deterministic builtin hash)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


'''
if old_helper in s:
    s = s.replace(old_helper, "", 1)
    print("helper removed")
else:
    print("helper not found (may already be gone)")

io.open(p, "w", encoding="utf-8").write(s)
print("fixed _task_index_for")
