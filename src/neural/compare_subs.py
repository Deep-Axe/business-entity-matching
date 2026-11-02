"""Per-country comparison of two matching files, restricted to S1s whose partition has new scores."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, ce_prep, dataset, owner_dir, policy_json, text_scorer_json, store, e5_small
import sys, glob, pickle, csv, collections
old, new, suffix = sys.argv[1], sys.argv[2], sys.argv[3]
scored = set()
cty = {}
for f in sorted(glob.glob(ce_data() + 'test/part_*.pkl')):
    import os
    if os.path.exists(f'{f}.{suffix}.npy'):
        for r in pickle.load(open(f, 'rb')):
            scored.add(r[0]); cty[r[0]] = r[3]

def load(p):
    d = {}
    with open(p) as h:
        next(h)
        for line in h:
            s, x = line.rstrip('\n').split('\t'); d[s] = set(filter(None, x.split(',')))
    return d
A, B = load(old), load(new)
st = collections.defaultdict(collections.Counter)
for s in scored:
    c = cty[s]; a, b = A[s], B[s]
    st[c]['S1'] += 1; st[c]['old'] += len(a); st[c]['new'] += len(b); st[c]['added'] += len(b - a); st[c]['removed'] += len(a - b)
    st[c]['old_empty'] += not a; st[c]['new_empty'] += not b
for c, v in sorted(st.items()):
    n = v['S1']
    print(f"{c:7s} S1 {n:6d}  targets/S1 old {v['old'] / n:.3f} new {v['new'] / n:.3f}  added/S1 {v['added'] / n:.3f} removed/S1 {v['removed'] / n:.3f}  empty old {v['old_empty'] / n:.4f} new {v['new_empty'] / n:.4f}")
