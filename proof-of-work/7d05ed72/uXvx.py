with open('tests/test_multi_run_analyzer.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
with open('outputs/_dbg_testlines.txt', 'w', encoding='utf-8') as out:
    for i in range(60, 70):
        out.write(f"{i+1}: {lines[i]!r}\n")
