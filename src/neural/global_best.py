"""Target-centric test: is the claimant the best S1 for this target among ALL same-family S1s (not just retrieved)?"""
import json, re, sqlite3, sys, collections, unicodedata
import numpy as np
from rapidfuzz import fuzz
sys.path.insert(0, 'src/neural')
from name_unique_lib import core

SYN = {'incorporated': 'inc', 'corporation': 'corp', 'company': 'co', 'limited': 'ltd', 'private': 'pvt',
       'p c': 'pc', 'p a': 'pa', 'l l c': 'llc', 'and': '&'}
LEET = str.maketrans({'0': 'o', '1': 'l', '3': 'e', '8': 'b', '@': 'a'})

def full(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'(?<=[a-z])[0-9]|[0-9](?=[a-z])', lambda m: m.group(0).translate(LEET), s)
    s = ' '.join(re.sub(r'[^a-z0-9&]+', ' ', s).split())
    for k, v in SYN.items():
        s = re.sub(rf'\b{k}\b', v, s)
    return s

def addr_sim(a, b):
    return fuzz.token_set_ratio(a.lower(), b.lower()) if a.strip() and b.strip() else None

con = sqlite3.connect('file:artifacts/train_store_v2/records.sqlite3?mode=ro', uri=True)
fam = collections.defaultdict(list)
for eid, country, name, addr in con.execute('select entity_id,country,name,address from anchors'):
    fam[(country, core(name))].append((eid, name, addr))
print('families built', flush=True)

def classify(ref, tgt):
    country, an, aa = con.execute('select country,name,address from anchors where entity_id=?', (ref,)).fetchone()
    bn, ba = con.execute('select name,address from records where entity_id=?', (tgt,)).fetchone()
    members = {e: (n, a) for e, n, a in fam[(country, core(bn))] + fam[(country, core(an))]}
    members[ref] = (an, aa)
    empty = not ba.strip() or ba.strip().upper() in ('NULL', '<NULL>')
    def score(n, a):
        s = fuzz.ratio(full(n), full(bn))
        if not empty:
            s = 0.5 * s + 0.5 * (addr_sim(a, ba) or 0)
        return s
    sc = {e: score(n, a) for e, (n, a) in members.items()}
    own = sc[ref]; others = [v for e, v in sc.items() if e != ref]
    if not others:
        return empty, 'no other same-family S1'
    best = max(others)
    if own > best + 3:
        return empty, 'claimant is best'
    if best > own + 3:
        return empty, 'another S1 is better'
    return empty, 'tie'

stats = collections.defaultdict(lambda: [0, 0])
for pf, fleet in [('enhanced_dev_probabilities', 'v2_gate0_dev'), ('enhanced_full_tune_probabilities', 'v2_gate0_tune')]:
    p = np.load(f'artifacts/improvement_v2/cpu/fit_gate0_100k_v1/{pf}.npy')
    al = json.load(open(f'artifacts/improvement_v2/cpu/fit_gate0_100k_v1/{pf}.alignment.json'))
    for s in al['shards']:
        z = np.load(f"artifacts/improvement_v2/cloud/fleet_v1/collected/{fleet}/shards/{s['shard']}/features.npz")
        ps = p[s['pair_start']:s['pair_end']]
        for i in np.where(ps >= 0.05)[0]:
            empty, c = classify(str(z['reference_ids'][z['query_ids'][i]]), str(z['target_ids'][i]))
            band = 'accepted' if ps[i] >= 0.76 else 'uncertain'
            k = ('empty-address' if empty else 'with-address', band, c)
            stats[k][0] += int(z['y'][i]); stats[k][1] += 1
print(f"{'target':14s} {'band':10s} {'target-centric verdict':26s} {'pairs':>6s} {'true rate':>9s}")
for k in sorted(stats):
    tp, n = stats[k]
    print(f'{k[0]:14s} {k[1]:10s} {k[2]:26s} {n:6d} {tp / n:9.3f}')
