"""Dense-union candidate oracle curves on dev/tune, and (with --write KNA KN) union row pickles <sp>_u.pkl = existing rows + new dense rows.
Oracle per owner with t true and r retrieved true links: 1 if t == 0 else 5r / (t + 4r)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, ce_prep, dataset, owner_dir, policy_json, text_scorer_json, store, e5_small
import csv, pickle, sys, collections
import numpy as np

C = ce_data(); DS = dataset() + 'train/'
A = ce_prep()
z = np.load(C + 'dense_devtune.npz'); qid = z['qid']
dec = lambda v: (f'S3-{v - 1_000_000_000}' if v >= 1_000_000_000 else f'S2-{v}') if v >= 0 else None
dense = {v: collections.defaultdict(list) for v in ('name_addr', 'name')}
for k in z.files:
    if k.endswith('_ix'):
        v, c = k[:-3].rsplit('_', 1)
        for qi, ids, ss in zip(z[k], z[f'{v}_{c}_id'], z[f'{v}_{c}_s']):
            dense[v][qid[qi]] = [(dec(int(i)), float(s)) for i, s in zip(ids, ss) if i >= 0]
split = {}; truth = {}; cty = {}
for sp in ('dev', 'tune'):
    for r in np.load(A + f'prepared_gate0_full_{sp}_v1/reference_ids.npy'):
        split[r] = sp
with open(DS + 'train_ground_truth.tsv') as h:
    for a, b in csv.reader(h, delimiter='\t'):
        if a in split:
            truth[a] = set(b.split(',')) - {''}
for a in split:
    truth.setdefault(a, set())
cur = collections.defaultdict(set); rows = {}
for sp in ('dev', 'tune'):
    rows[sp] = pickle.load(open(C + f'{sp}.pkl', 'rb'))
    for r in rows[sp]:
        cur[r[0]].add(r[1]); cty[r[0]] = r[3]

def oracle(extra):
    res = collections.defaultdict(list); n_new = collections.Counter()
    for a, sp in split.items():
        cand = cur[a] | extra.get(a, set()); t = len(truth[a]); r = len(truth[a] & cand)
        res[sp].append(1.0 if t == 0 else 5 * r / (t + 4 * r)); n_new[sp] += len(extra.get(a, set()) - cur[a])
    return {sp: (np.mean(v), n_new[sp] / len(v)) for sp, v in res.items()}

def extra_of(kna, kn):
    return {a: {t for t, _ in dense['name_addr'].get(a, [])[:kna]} | {t for t, _ in dense['name'].get(a, [])[:kn]} for a in split}

base = oracle({})
print(f'current candidates: dev oracle {base["dev"][0]:.5f}  tune {base["tune"][0]:.5f}')
for kna, kn in ((5, 0), (10, 0), (20, 0), (30, 0), (0, 5), (0, 10), (0, 30), (5, 5), (10, 10), (20, 10), (30, 30)):
    o = oracle(extra_of(kna, kn))
    print(f'name_addr top{kna:>2} + name top{kn:>2}: dev {o["dev"][0]:.5f} (+{o["dev"][0] - base["dev"][0]:.5f}, {o["dev"][1]:.2f} new/S1)  '
          f'tune {o["tune"][0]:.5f} (+{o["tune"][0] - base["tune"][0]:.5f}, {o["tune"][1]:.2f} new/S1)')

def _pce_path(pkl, sp, usuf):
    for p in (C + f'e5_cont1300k/{sp}{usuf}_pce.npy', pkl + '.e5_cont1300k.npy'):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'no e5_cont1300k scores for {pkl}')

if len(sys.argv) > 1 and sys.argv[1] == '--k10':
    TOP, T_NA, T_N = int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    extra = {}
    for a in split:
        keep = set()
        for t, s in dense['name_addr'].get(a, [])[:TOP]:
            if t and s >= T_NA:
                keep.add(t)
        for t, s in dense['name'].get(a, [])[:TOP]:
            if t and s >= T_N:
                keep.add(t)
        extra[a] = keep
    need = {t for a in extra for t in extra[a] - cur[a]}
    s1 = {}; rec = {}
    with open(DS + 'train_source1.tsv') as h:
        for x in csv.reader(h, delimiter='\t', quoting=csv.QUOTE_NONE):
            if x[0] in split:
                s1[x[0]] = (x[1], x[2]); cty.setdefault(x[0], x[3])
    for s in ('train_source2.tsv', 'train_source3.tsv'):
        with open(DS + s) as h:
            for x in csv.reader(h, delimiter='\t', quoting=csv.QUOTE_NONE):
                if x[0] in need:
                    rec[x[0]] = (x[1], x[2])
    for sp in ('dev', 'tune'):
        new = [(a, t, int(t in truth[a]), cty[a], float('nan')) + s1[a] + rec[t]
               for a in split if split[a] == sp for t in sorted(extra[a] - cur[a])]
        pickle.dump(new, open(C + f'{sp}_k10new.pkl', 'wb'))
        print(f'{sp}_k10new.pkl: new {len(new)} (true {sum(r[2] for r in new)})')
    sys.exit()

if len(sys.argv) > 1 and sys.argv[1] == 'merge-k10':
    os.makedirs(C + 'e5_cont1300k', exist_ok=True)
    for sp in ('dev', 'tune'):
        a, b = pickle.load(open(C + f'{sp}.pkl', 'rb')), pickle.load(open(C + f'{sp}_k10new.pkl', 'rb'))
        pickle.dump(a + b, open(C + f'{sp}_k10.pkl', 'wb'))
        for s in ('ops', 'nf', 'nf2'):
            np.save(C + f'{sp}_k10.pkl.{s}.npy', np.concatenate([np.load(C + f'{sp}.pkl.{s}.npy'), np.load(C + f'{sp}_k10new.pkl.{s}.npy')]))
        pce = np.concatenate([np.load(_pce_path(C + f'{sp}.pkl', sp, '')), np.load(_pce_path(C + f'{sp}_k10new.pkl', sp, '_k10new'))])
        np.save(C + f'e5_cont1300k/{sp}_k10_pce.npy', pce)
        np.save(C + f'{sp}_k10.pkl.e5_cont1300k.npy', pce)
        print(sp, len(a), '+', len(b))
    sys.exit()

if '--write' in sys.argv:
    kna, kn = int(sys.argv[-2]), int(sys.argv[-1]); ex = extra_of(kna, kn)
    need = {t for a in ex for t in ex[a] - cur[a]}
    s1 = {}; rec = {}
    with open(DS + 'train_source1.tsv') as h:
        for x in csv.reader(h, delimiter='\t', quoting=csv.QUOTE_NONE):
            if x[0] in split:
                s1[x[0]] = (x[1], x[2]); cty.setdefault(x[0], x[3])
    for s in ('train_source2.tsv', 'train_source3.tsv'):
        with open(DS + s) as h:
            for x in csv.reader(h, delimiter='\t', quoting=csv.QUOTE_NONE):
                if x[0] in need:
                    rec[x[0]] = (x[1], x[2])
    for sp in ('dev', 'tune'):
        new = [(a, t, int(t in truth[a]), cty[a], float('nan')) + s1[a] + rec[t]
               for a in split if split[a] == sp for t in sorted(ex[a] - cur[a])]
        pickle.dump(rows[sp] + new, open(C + f'{sp}_u.pkl', 'wb'))
        print(f'{sp}_u.pkl: existing {len(rows[sp])} + new {len(new)} (true {sum(r[2] for r in new)})')
