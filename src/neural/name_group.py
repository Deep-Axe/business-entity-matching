"""Near-miss pairs: does name uniqueness / best-address-within-name-group separate true matches?"""
import glob, json, sqlite3, sys, random, collections
import numpy as np
from rapidfuzz import fuzz
sys.path.insert(0, 'src/neural')
from name_unique_lib import core

def groups(con):
    g = collections.defaultdict(list)
    for eid, country, name, addr in con.execute('select entity_id,country,name,address from anchors'):
        g[(country, core(name))].append((eid, addr))
    return g

def label(con, g, owner, target):
    o = con.execute('select country,name,address from anchors where entity_id=?', (owner,)).fetchone()
    t = con.execute('select address from records where entity_id=?', (target,)).fetchone()[0]
    members = g[(o[0], core(o[1]))]
    if len(members) <= 1:
        return 'unique_name'
    if not t.strip():
        return 'shared_name/target_address_empty'
    sims = {eid: fuzz.token_set_ratio(a.lower(), t.lower()) for eid, a in members}
    own = sims.get(owner, 0)
    best_other = max(v for k, v in sims.items() if k != owner)
    if own > best_other + 5:
        return 'shared_name/claimant_best_address'
    if best_other > own + 5:
        return 'shared_name/other_S1_better_address'
    return 'shared_name/address_tie'

lo, hi = float(sys.argv[1]), float(sys.argv[2])
train = sqlite3.connect('file:artifacts/train_store_v2/records.sqlite3?mode=ro', uri=True)
g = groups(train)
p = np.load('artifacts/improvement_v2/cpu/fit_gate0_100k_v1/enhanced_dev_probabilities.npy')
al = json.load(open('artifacts/improvement_v2/cpu/fit_gate0_100k_v1/enhanced_dev_probabilities.alignment.json'))
stats = collections.defaultdict(lambda: [0, 0])
for s in al['shards']:
    z = np.load(f"artifacts/improvement_v2/cloud/fleet_v1/collected/v2_gate0_dev/shards/{s['shard']}/features.npz")
    ps = p[s['pair_start']:s['pair_end']]
    for i in np.where((ps >= lo) & (ps < hi))[0]:
        k = label(train, g, str(z['reference_ids'][z['query_ids'][i]]), str(z['target_ids'][i]))
        stats[k][0] += int(z['y'][i]); stats[k][1] += 1
print(f'DEV pairs with {lo} <= p < {hi}')
for k, (tp, n) in sorted(stats.items(), key=lambda x: -x[1][1]):
    print(f'  {k:40s} pairs {n:5d}  true-match rate {tp / n:.3f}')
del g, train

test = sqlite3.connect('file:artifacts/test_store_v2/records.sqlite3?mode=ro', uri=True)
g = groups(test)
shards = sorted(glob.glob('artifacts/improvement_v2/inference/submission_controller_v1/collected/submission_100k_*/shards/*/'))
random.seed(0)
tstats = collections.defaultdict(collections.Counter)
for sd in random.sample(shards, int(sys.argv[3]) if len(sys.argv) > 3 else 300):
    z = np.load(sd + 'pair_ids.npz', allow_pickle=True); ps = np.load(sd + 'probabilities.npy')
    for i in np.where((ps >= lo) & (ps < hi))[0]:
        q = z['query_ids'][i]
        tstats[str(z['countries'][q])][label(test, g, str(z['reference_ids'][q]), str(z['target_ids'][i]))] += 1
print(f'TEST pairs with {lo} <= p < {hi} (sample of 300 shards)')
for c, m in sorted(tstats.items()):
    n = sum(m.values())
    print(f'  {c}: ' + ', '.join(f'{k} {v / n:.2f}' for k, v in m.most_common()) + f'  (n={n})')
