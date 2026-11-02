"""New test pairs from the dense route: top-5 per view, kept if name_addr sim >= 0.895 or name sim >= 0.95, minus existing candidates.
Writes ce/test_dense/chunk_XXX.pkl rows (s1, target, -1, country, nan, s1 name, s1 addr, target name, target addr)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, ce_prep, dataset, owner_dir, policy_json, text_scorer_json, store, e5_small
import csv, glob, os, pickle, collections
import numpy as np

C = ce_data(); DS = dataset() + 'test/'
TOP, T_NA, T_N, CH = 5, 0.895, 0.95, 400_000
z = np.load(C + 'dense_test.npz'); qid = z['qid']
dec = lambda v: f'S3-{v - 1_000_000_000}' if v >= 1_000_000_000 else f'S2-{v}'
new = collections.defaultdict(set)
for k in z.files:
    if k.endswith('_ix'):
        v, c = k[:-3].rsplit('_', 1); thr = T_NA if v == 'name_addr' else T_N
        ids, ss = z[f'{v}_{c}_id'][:, :TOP], z[f'{v}_{c}_s'][:, :TOP]
        for qi, i_row, s_row in zip(z[k], ids, ss):
            for i, s in zip(i_row, s_row):
                if i >= 0 and s >= thr:
                    new[qid[qi]].add(dec(int(i)))
print('dense pairs passing the filter', sum(map(len, new.values())), flush=True)
for f in sorted(glob.glob(C + 'test/part_*.pkl')):
    for r in pickle.load(open(f, 'rb')):
        if r[0] in new:
            new[r[0]].discard(r[1])
need = set().union(*new.values()); print('new pairs', sum(map(len, new.values())), 'targets', len(need), flush=True)

def rows(path):
    with open(path, newline='') as h:
        r = csv.reader(h, delimiter='\t', quoting=csv.QUOTE_NONE); next(r)
        yield from r
s1 = {x[0]: (x[3], x[1], x[2]) for x in rows(DS + 'test_source1.tsv') if new.get(x[0])}
rec = {}
for s in ('test_source2.tsv', 'test_source3.tsv'):
    for x in rows(DS + s):
        if x[0] in need:
            rec[x[0]] = (x[1], x[2])
os.makedirs(C + 'test_dense', exist_ok=True)
out = [(a, t, -1, s1[a][0], float('nan'), s1[a][1], s1[a][2]) + rec[t] for a in sorted(new) for t in sorted(new[a])]
for j in range(0, len(out), CH):
    pickle.dump(out[j:j + CH], open(C + f'test_dense/chunk_{j // CH:03d}.pkl', 'wb'))
print('rows', len(out), 'chunks', (len(out) + CH - 1) // CH, 'by country', collections.Counter(r[3] for r in out), flush=True)
