"""Signed house-number offset: target number minus S1 number, by label class."""
import json, re, sqlite3, collections
import numpy as np

con = sqlite3.connect('file:artifacts/train_store_v2/records.sqlite3?mode=ro', uri=True)

def nums(s):
    return [int(x) for x in re.findall(r'\d+', s or '') if len(x) <= 7]

def offset(a, b):
    A, B = nums(a), nums(b)
    if not A or not B:
        return 'missing'
    sa, sb = set(A), set(B)
    only_a = [x for x in A if x not in sb]; only_b = [y for y in B if y not in sa]
    if not only_a and not only_b:
        return 'same'
    if not only_a or not only_b:
        return 'extra/dropped number'
    for x in only_a:
        for y in only_b:
            xs, ys = str(x), str(y)
            if len(ys) == len(xs) - 1 and (xs[1:] == ys or xs[:-1] == ys):
                return 'digit dropped'
    d = min(((y - x) for x in only_a for y in only_b), key=abs)
    if d == 0:
        return 'same'
    if 1 <= d <= 25:
        return f'+{d:02d}' if d <= 25 else '+big'
    if -25 <= d <= -1:
        return f'-{-d:02d}'
    return 'large change'

stats = collections.defaultdict(collections.Counter)
for pf, fleet in [('enhanced_dev_probabilities', 'v2_gate0_dev'), ('enhanced_full_tune_probabilities', 'v2_gate0_tune')]:
    p = np.load(f'artifacts/improvement_v2/cpu/fit_gate0_100k_v1/{pf}.npy')
    al = json.load(open(f'artifacts/improvement_v2/cpu/fit_gate0_100k_v1/{pf}.alignment.json'))
    for s in al['shards']:
        z = np.load(f"artifacts/improvement_v2/cloud/fleet_v1/collected/{fleet}/shards/{s['shard']}/features.npz")
        ps = p[s['pair_start']:s['pair_end']]
        for i in np.where(ps >= 0.05)[0]:
            ref = str(z['reference_ids'][z['query_ids'][i]]); tgt = str(z['target_ids'][i])
            aa = con.execute('select address from anchors where entity_id=?', (ref,)).fetchone()[0]
            ba = con.execute('select address from records where entity_id=?', (tgt,)).fetchone()[0]
            y = int(z['y'][i])
            owned = y or con.execute('select 1 from labels where target_id=?', (tgt,)).fetchone() is not None
            cls = 'true' if y else ('decoy' if not owned else 'other S1')
            k = offset(aa, ba)
            if k.startswith('+'):
                stats['+1..+25 (any)'][cls] += 1
            if k.startswith('-'):
                stats['-1..-25 (any)'][cls] += 1
            stats[k][cls] += 1
order = sorted(stats, key=lambda k: -sum(stats[k].values()))
print(f"{'target minus S1 number':24s} {'true':>6s} {'decoy':>6s} {'otherS1':>7s} {'true rate':>9s}")
for k in order:
    c = stats[k]; n = sum(c.values())
    if n >= 15:
        print(f"{k:24s} {c['true']:6d} {c['decoy']:6d} {c['other S1']:7d} {c['true'] / n:9.3f}")
