import sys
sys.path.insert(0, 'c:/Users/games/python/arcus-x')

# Count unique (z,gravity,tier) combos in the raw file
combos = set()
zgt = []
with open('outputs/absolute_raw_data_poolside_laguna_xs_2.1.txt', encoding='utf-8') as f:
    for line in f:
        s = line.strip()
        if s.startswith('Parameters:'):
            parts = [p.strip() for p in s.split(': ',1)[1].split(',')]
            d = {}
            for p in parts:
                if '=' in p:
                    k,v = p.split('=',1)
                    d[k.strip()] = v.strip()
            combos.add((d['z'], d['gravity'], d['tier'], d.get('grid','default')))
            zgt.append((d['z'], d['gravity'], d['tier']))

print("TOTAL PARAMETER LINES:", len(zgt))
print("UNIQUE (z,g,t,grid) COMBOS:", len(combos))
from collections import Counter
c = Counter(zgt)
dups = {k:v for k,v in c.items() if v > 1}
print("COMBOS APPEARING >1 TIME:", len(dups))
print("MAX REPEAT:", max(c.values()) if c else 0)
print("SAMPLE DUP COMBOS:", list(dups.items())[:5])
