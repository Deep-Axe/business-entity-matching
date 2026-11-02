"""Test key-channel rows: pairs from keych_test.pkl not already in ce/test/part_*.pkl or ce/test_dense_k10/*.pkl
-> ce/test_keych/kc_000.pkl (NaN p100k, label -1)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, ce_prep, dataset, owner_dir, policy_json, text_scorer_json, store, e5_small
import csv, glob, os, pickle, collections

C = ce_data(); DS = dataset() + 'test/'
out = C + 'test_keych/'; os.makedirs(out, exist_ok=True)
want = {(a, t) for a, t, _ in pickle.load(open(C + 'keych_test.pkl', 'rb'))}
print('key pairs', len(want), flush=True)
for f in sorted(glob.glob(C + 'test/part_*.pkl')) + sorted(glob.glob(C + 'test_dense_k10/*.pkl')):
    for r in pickle.load(open(f, 'rb')):
        want.discard((r[0], r[1]))
print('new pairs', len(want), flush=True)

def rows_of(path):
    with open(path, newline='') as h:
        r = csv.reader(h, delimiter='\t', quoting=csv.QUOTE_NONE); next(r)
        yield from r
qs = {a for a, _ in want}; need = {t for _, t in want}
s1 = {x[0]: (x[3], x[1], x[2]) for x in rows_of(DS + 'test_source1.tsv') if x[0] in qs}
rec = {}
for s in ('test_source2.tsv', 'test_source3.tsv'):
    for x in rows_of(DS + s):
        if x[0] in need:
            rec[x[0]] = (x[1], x[2])
d = [(a, t, -1, s1[a][0], float('nan'), s1[a][1], s1[a][2]) + rec[t] for a, t in sorted(want)]
fr = [r for r in d if r[3] == 'France']
kcx = [r for r in d if r[3] != 'France']
CH = 400_000
pickle.dump(d, open(out + 'kc_000.pkl', 'wb'))
for j in range(0, len(kcx), CH):
    pickle.dump(kcx[j:j + CH], open(out + f'kcx_{j // CH:03d}.pkl', 'wb'))
print('rows', len(d), 'kcx (no France)', len(kcx), 'france', len(fr), collections.Counter(r[3] for r in d), flush=True)
