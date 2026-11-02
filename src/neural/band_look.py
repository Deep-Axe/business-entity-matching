import json, sqlite3, random
import numpy as np
con = sqlite3.connect('file:artifacts/train_store_v2/records.sqlite3?mode=ro', uri=True)
p = np.load('artifacts/improvement_v2/cpu/fit_gate0_100k_v1/enhanced_full_tune_probabilities.npy')
al = json.load(open('artifacts/improvement_v2/cpu/fit_gate0_100k_v1/enhanced_full_tune_probabilities.alignment.json'))
rows = []
for s in al['shards']:
    z = np.load(f"artifacts/improvement_v2/cloud/fleet_v1/collected/v2_gate0_tune/shards/{s['shard']}/features.npz")
    ps = p[s['pair_start']:s['pair_end']]
    for i in np.where((ps >= 0.2) & (ps < 0.8))[0]:
        rows.append((str(z['reference_ids'][z['query_ids'][i]]), str(z['target_ids'][i]), float(ps[i]), int(z['y'][i])))
random.seed(7)
for ref, tgt, pp, y in random.sample(rows, 60):
    a = con.execute('select name,address from anchors where entity_id=?', (ref,)).fetchone()
    b = con.execute('select name,address from records where entity_id=?', (tgt,)).fetchone()
    o = con.execute('select reference_id from labels where target_id=?', (tgt,)).fetchone()
    tag = 'TRUE ' if y else ('decoy' if not o else 'OTHER')
    extra = ''
    if o and not y:
        oa = con.execute('select name,address from anchors where entity_id=?', (o[0],)).fetchone()
        extra = f'\n        owner: {oa[0]} | {oa[1]}'
    print(f'{tag} {pp:.2f} {a[0]} | {a[1]}\n        {tgt[:2]}: {b[0]} | {b[1]}{extra}')
