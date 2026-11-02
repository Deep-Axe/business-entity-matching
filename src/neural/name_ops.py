"""Name-side diff census: which added / substituted tokens mark decoys vs true matches?"""
import json, re, sqlite3, collections, unicodedata
import numpy as np
from rapidfuzz import fuzz

LEGAL = {'llc', 'inc', 'corp', 'corporation', 'co', 'company', 'ltd', 'limited', 'pvt', 'private', 'llp', 'lp', 'pllc',
         'pc', 'pa', 'plc', 'incorporated', 'sarl', 'sas', 'sasu', 'eurl', 'sa', 'sci'}

def toks(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
    return re.findall(r'[a-z]+', s)

con = sqlite3.connect('file:artifacts/train_store_v2/records.sqlite3?mode=ro', uri=True)
added = {c: collections.Counter() for c in ('true', 'decoy')}
subst = {c: collections.Counter() for c in ('true', 'decoy')}
legal = {c: collections.Counter() for c in ('true', 'decoy')}
n = collections.Counter()
for pf, fleet in [('enhanced_dev_probabilities', 'v2_gate0_dev'), ('enhanced_full_tune_probabilities', 'v2_gate0_tune')]:
    p = np.load(f'artifacts/improvement_v2/cpu/fit_gate0_100k_v1/{pf}.npy')
    al = json.load(open(f'artifacts/improvement_v2/cpu/fit_gate0_100k_v1/{pf}.alignment.json'))
    for s in al['shards']:
        z = np.load(f"artifacts/improvement_v2/cloud/fleet_v1/collected/{fleet}/shards/{s['shard']}/features.npz")
        ps = p[s['pair_start']:s['pair_end']]
        for i in np.where(ps >= 0.05)[0]:
            ref = str(z['reference_ids'][z['query_ids'][i]]); tgt = str(z['target_ids'][i]); y = int(z['y'][i])
            if not y and con.execute('select 1 from labels where target_id=?', (tgt,)).fetchone():
                continue
            cls = 'true' if y else 'decoy'
            an = con.execute('select name from anchors where entity_id=?', (ref,)).fetchone()[0]
            bn = con.execute('select name from records where entity_id=?', (tgt,)).fetchone()[0]
            if any(ord(c) > 0x900 for c in bn):
                continue
            A, B = toks(an), toks(bn)
            n[cls] += 1
            la = {t for t in A if t in LEGAL}; lb = {t for t in B if t in LEGAL}
            legal[cls]['legal same' if la == lb else ('legal changed' if la and lb else 'legal added/removed')] += 1
            A = [t for t in A if t not in LEGAL]; B = [t for t in B if t not in LEGAL]
            ua = [t for t in A if t not in B and not any(fuzz.ratio(t, u) >= 75 for u in B)]
            ub = [t for t in B if t not in A and not any(fuzz.ratio(t, u) >= 75 for u in A)]
            for u in set(ub):
                added[cls][u] += 1
            if ua and ub:
                subst[cls]['content word replaced'] += 1
            elif ub:
                subst[cls]['word added only'] += 1
            elif ua:
                subst[cls]['word dropped only'] += 1
            else:
                subst[cls]['no unmatched words'] += 1
print('pairs', dict(n))
for name, tab in (('word-level diff', subst), ('legal form', legal)):
    print(f'\n{name}:')
    for k in sorted(set(tab['true']) | set(tab['decoy'])):
        t, d = tab['true'][k] / n['true'], tab['decoy'][k] / n['decoy']
        print(f'  {k:24s} true {t:.3f}  decoy {d:.3f}')
print('\nadded tokens most over-represented in decoys (count >= 8):')
cand = [(added['decoy'][u] / n['decoy'] / ((added['true'][u] + 1) / n['true']), u) for u in added['decoy'] if added['decoy'][u] >= 8]
for r, u in sorted(cand, reverse=True)[:30]:
    print(f"  {u:14s} decoy {added['decoy'][u]:4d}  true {added['true'][u]:4d}  ratio {r:6.1f}")
