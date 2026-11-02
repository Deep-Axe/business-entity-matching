"""Collect test pairs with p100k in [1e-4, 0.01) from a few partitions (below the CE filter) with text, for CE probing."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, ce_prep, dataset, owner_dir, policy_json, text_scorer_json, store, e5_small
import json, sqlite3, pickle, random
from pathlib import Path
import numpy as np
con = sqlite3.connect('file:artifacts/test_store_v2/records.sqlite3?mode=ro', uri=True)
def lookup(table, ids):
    ids = list(set(ids)); d = {}
    for i in range(0, len(ids), 900):
        ch = ids[i:i + 900]
        for e, n, a in con.execute(f"select entity_id,name,address from {table} where entity_id in ({','.join('?' * len(ch))})", ch): d[e] = (n, a)
    return d
q = json.load(open('artifacts/improvement_v2/inference/submission_controller_v1/queue.json'))['complete']
keys = sorted(q, key=int); random.seed(0); keys = random.sample(keys, 6)
rows = []; nref = {}
for key in keys:
    for shard in sorted((Path(q[key]['directory']) / 'shards').glob('[0-9]*')):
        z = dict(np.load(shard / 'pair_ids.npz')); p = np.load(shard / 'probabilities.npy')
        for qi in range(len(z['reference_ids'])): nref[str(z['reference_ids'][qi])] = str(z['countries'][qi])
        idx = np.flatnonzero((p >= 1e-4) & (p < 0.01))
        refs = [str(z['reference_ids'][z['query_ids'][i]]) for i in idx]; tg = [str(z['target_ids'][i]) for i in idx]
        an = lookup('anchors', refs); rc = lookup('records', tg)
        for i, r, t in zip(idx, refs, tg):
            rows.append((r, t, -1, str(z['countries'][z['query_ids'][i]]), float(p[i])) + an[r] + rc[t])
pickle.dump(rows, open(ce_data() + 'probe_lowp.pkl', 'wb')); pickle.dump(nref, open(ce_data() + 'probe_lowp_refs.pkl', 'wb'))
import collections; print('pairs', len(rows), 'S1s', len(nref), collections.Counter(nref.values()), collections.Counter(r[3] for r in rows))
