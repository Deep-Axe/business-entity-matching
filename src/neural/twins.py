"""Are decoys mutated copies of a true target? Measure each candidate's nearest same-S1 twin."""
import json, sqlite3, collections
import numpy as np
from rapidfuzz import fuzz

con = sqlite3.connect('file:artifacts/train_store_v2/records.sqlite3?mode=ro', uri=True)
p = np.load('artifacts/improvement_v2/cpu/fit_gate0_100k_v1/enhanced_dev_probabilities.npy')
al = json.load(open('artifacts/improvement_v2/cpu/fit_gate0_100k_v1/enhanced_dev_probabilities.alignment.json'))
rows = []
for s in al['shards']:
    z = np.load(f"artifacts/improvement_v2/cloud/fleet_v1/collected/v2_gate0_dev/shards/{s['shard']}/features.npz")
    ps = p[s['pair_start']:s['pair_end']]
    for i in np.where(ps >= 0.01)[0]:
        rows.append((str(z['reference_ids'][z['query_ids'][i]]), str(z['target_ids'][i]), float(ps[i]), int(z['y'][i])))
text, owned, s1 = {}, {}, {}
for ref, tgt, _, _ in rows:
    if tgt not in text:
        n, a = con.execute('select name,address from records where entity_id=?', (tgt,)).fetchone()
        text[tgt] = f'{n} | {a}'
        owned[tgt] = con.execute('select 1 from labels where target_id=?', (tgt,)).fetchone() is not None
    if ref not in s1:
        n, a = con.execute('select name,address from anchors where entity_id=?', (ref,)).fetchone()
        s1[ref] = f'{n} | {a}'
by_ref = collections.defaultdict(list)
for r in rows:
    by_ref[r[0]].append(r)

out = collections.defaultdict(list)
for ref, cands in by_ref.items():
    for _, t, pt, y in cands:
        if pt < 0.1:
            continue
        src = t[:2]
        sims = [(fuzz.ratio(text[t], text[u]), u, pu) for _, u, pu, _ in cands if u != t and u[:2] == src]
        twin, u, pu = max(sims) if sims else (0, None, 0)
        to_s1 = fuzz.ratio(text[t], s1[ref])
        cls = 'true match' if y else ('unowned decoy' if not owned[t] else 'owned by another S1')
        out[cls].append((twin, twin - to_s1, pu))

for cls, v in out.items():
    v = np.array(v)
    print(f'{cls:20s} n={len(v):5d}  twin ratio median {np.median(v[:, 0]):5.1f}  '
          f'twin>=90: {np.mean(v[:, 0] >= 90):.3f}  twin closer than S1 by >=10: {np.mean(v[:, 1] >= 10):.3f}  '
          f'twin>=90 & twin p>=0.76: {np.mean((v[:, 0] >= 90) & (v[:, 2] >= 0.76)):.3f}')
