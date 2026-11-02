"""Per-country probability bands and near-miss patterns from saved 100k test shards."""
import glob, sqlite3, sys, random, collections
import numpy as np
sys.path.insert(0, 'src/neural')
from fn_cats import cat

shards = sorted(glob.glob('artifacts/improvement_v2/inference/submission_controller_v1/collected/submission_100k_*/shards/*/'))
random.seed(0)
shards = random.sample(shards, 400)
test = sqlite3.connect('file:artifacts/test_store_v2/records.sqlite3?mode=ro', uri=True)

def rec(table, eid):
    r = test.execute(f'select name,address,country from {table} where entity_id=?', (eid,)).fetchone()
    return {'business_name': r[0], 'business_address': r[1], 'country': r[2]}

edges = [0, 0.01, 0.1, 0.3, 0.5, 0.76, 0.9, 1.01]
bands = collections.defaultdict(lambda: np.zeros(len(edges) - 1))
owners = collections.Counter()
near = collections.defaultdict(list)
for s in shards:
    z = np.load(s + 'pair_ids.npz', allow_pickle=True)
    p = np.load(s + 'probabilities.npy')
    refs, tg, q, cc = z['reference_ids'], z['target_ids'], z['query_ids'], z['countries']
    for i, c in enumerate(cc):
        owners[str(c)] += 1
    pc = np.array([str(cc[i]) for i in q])
    for c in set(pc):
        m = pc == c
        bands[c] += np.histogram(p[m], edges)[0]
    idx = np.where((p >= 0.3) & (p < 0.76))[0]
    for i in idx:
        if random.random() < 0.25:
            near[pc[i]].append((str(refs[q[i]]), str(tg[i]), float(p[i])))

print('pairs per S1 by probability band')
print(f"{'country':8s} {'S1s':>6s} " + ' '.join(f'{a:>5.2f}-{b:<4.2f}' for a, b in zip(edges, edges[1:])))
for c in sorted(bands):
    print(f'{c:8s} {owners[c]:6d} ' + ' '.join(f'{v / owners[c]:10.3f}' for v in bands[c]))

print('\nnear-miss (0.30-0.76) pattern mix per country')
mixes = {}
for c, pairs in near.items():
    mixes[c] = collections.Counter(cat({'reference': rec('anchors', o), 'target': rec('records', t)}) for o, t, _ in pairs)
keys = sorted({k for m in mixes.values() for k in m}, key=lambda k: -sum(m[k] for m in mixes.values()))
print(f"{'category':30s} " + ' '.join(f'{c:>14s}' for c in sorted(mixes)))
for k in keys:
    print(f'{k:30s} ' + ' '.join(f'{mixes[c][k] / owners[c] * 100:8.2f}/100S1' for c in sorted(mixes)))

random.seed(1)
for c in ['France']:
    print(f'\n{c} near-miss examples')
    for o, t, p in random.sample(near[c], min(40, len(near[c]))):
        a, b = rec('anchors', o), rec('records', t)
        print(f"{p:.2f} {a['business_name']} | {a['business_address']}\n      -> {t[:2]} {b['business_name']} | {b['business_address']}")
