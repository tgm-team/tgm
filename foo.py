import json
import pathlib
from collections import defaultdict

in_file, out_file = 'edgebank.log', 'edgebank.json'
in_file, out_file = 'gclstm.log', 'gclstm.json'
in_file = pathlib.Path(in_file)

groups = defaultdict(list)
for l in in_file.read_text().splitlines():
    msg_start_idx = l.find(']', l.find(']') + 1) + 2
    if l[msg_start_idx] == '{' and l.endswith('}'):
        d = json.loads(l[msg_start_idx:])
        keys = tuple(sorted(d.keys()))
        groups[keys].append(d)

out = {}
for k, v in groups.items():
    agg = defaultdict(list)
    if k == ('function', 'metric', 'value'):
        # Case: function/metric/value → group by (function, metric)
        for d in v:
            agg[(d['function'], d['metric'])].append(d['value'])
        out[k] = {fk: sum(vals) / len(vals) for fk, vals in agg.items()}
    else:
        for d in v:
            for kk, vv in d.items():
                agg[kk].append(vv)
        out[k] = dict(agg)

# Pretty print
for k, v in out.items():
    print('Key schema:', k)
    for kk, vv in v.items():
        print(' ', kk, '→', vv)
    input()
