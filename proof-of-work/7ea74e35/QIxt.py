import post_run_analyzer as p, json

content = open('outputs/absolute_raw_stream_deepseek_deepseek_v4_flash.txt', encoding='utf-8').read()
parts = p.ENTRY_HEADER_RE.split(content)
print("num parts:", len(parts))
block = parts[2] if len(parts) > 2 else (parts[1] if len(parts) > 1 else "")
print("BLOCK (first 700 chars):")
print(block[:700])
print("=" * 40)
# Now manually run the parser on this block
rec = p._parse_entry_block(parts[1], block)
print("env:", rec.get("env"))
print("grid_w/grid_h via env:", rec.get("env"))
print("rules_raw present in rec? keys:", list(rec.keys())[:10])
