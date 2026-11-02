"""Test dense rows for a new (TOP, T_NA, T_N) setting, reusing rows/features/scores already computed in ce/test_dense.
usage: dense_test_rows2.py OUTDIR TOP T_NA T_N
OUTDIR/kept_XXX.pkl = reused rows (with sliced .ops/.nf/.nf2/.e5_cont1300k arrays); OUTDIR/delta_XXX.pkl = rows still to featurize and score."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, ce_prep, dataset, owner_dir, policy_json, text_scorer_json, store, e5_small
import csv, glob, os, pickle, sys, collections
import numpy as np

C = ce_data(); DS = dataset() + 'test/'
out, TOP, T_NA, T_N = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]); CH = 400_000
os.makedirs(out, exist_ok=True)
z = np.load(C + 'dense_test.npz'); qid = z['qid']
dec = lambda v: f'S3-{v - 1_000_000_000}' if v >= 1_000_000_000 else f'S2-{v}'
want = collections.defaultdict(set)
for k in z.files:
    if k.endswith('_ix'):
        v, c = k[:-3].rsplit('_', 1); thr = T_NA if v == 'name_addr' else T_N
        for qi, i_row, s_row in zip(z[k], z[f'{v}_{c}_id'][:, :TOP], z[f'{v}_{c}_s'][:, :TOP]):
            for i, s in zip(i_row, s_row):
                if i >= 0 and s >= thr:
                    want[qid[qi]].add(dec(int(i)))
for f in sorted(glob.glob(C + 'test/part_*.pkl')):
    for r in pickle.load(open(f, 'rb')):
        if r[0] in want:
            want[r[0]].discard(r[1])
print('wanted new pairs', sum(map(len, want.values())), flush=True)
kept = 0
chunks = sorted(glob.glob(C + 'test_dense/chunk_*.pkl'))
if chunks:
    for j, f in enumerate(chunks):
        rows = pickle.load(open(f, 'rb')); keep = np.array([r[1] in want.get(r[0], ()) for r in rows])
        for r, k in zip(rows, keep):
            if k:
                want[r[0]].discard(r[1])
        pickle.dump([r for r, k in zip(rows, keep) if k], open(f'{out}/kept_{j:03d}.pkl', 'wb'))
        for s in ('ops', 'nf', 'nf2', 'e5_cont1300k'):
            feat = f'{f}.{s}.npy'
            if os.path.exists(feat):
                np.save(f'{out}/kept_{j:03d}.pkl.{s}.npy', np.load(feat)[keep])
        kept += int(keep.sum())
need = set().union(*want.values()) if want else set(); n_delta = sum(map(len, want.values()))
print('reused', kept, 'delta', n_delta, flush=True)

def rows_of(path):
    with open(path, newline='') as h:
        r = csv.reader(h, delimiter='\t', quoting=csv.QUOTE_NONE); next(r)
        yield from r
s1 = {x[0]: (x[3], x[1], x[2]) for x in rows_of(DS + 'test_source1.tsv') if want.get(x[0])}
rec = {}
for s in ('test_source2.tsv', 'test_source3.tsv'):
    for x in rows_of(DS + s):
        if x[0] in need:
            rec[x[0]] = (x[1], x[2])
d = [(a, t, -1, s1[a][0], float('nan'), s1[a][1], s1[a][2]) + rec[t] for a in sorted(want) for t in sorted(want[a])]
for j in range(0, len(d), CH):
    pickle.dump(d[j:j + CH], open(f'{out}/delta_{j // CH:03d}.pkl', 'wb'))
print('delta rows', len(d), collections.Counter(r[3] for r in d), flush=True)
