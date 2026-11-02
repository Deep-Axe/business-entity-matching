"""New training-owner pairs for one more cross-encoder member: owners outside the 200k bundle and outside dev/tune.
Positives = all ground-truth links; negatives = same core-name records and same (house number, street word) records.
Writes ce/big_mix/big_XX.pkl rows (s1, t, y, country, 0.5, s1 name, s1 addr, t name, t addr)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, ce_prep, dataset, owner_dir, policy_json, text_scorer_json, store, e5_small
import csv, glob, os, pickle, re, sys, unicodedata, collections
import numpy as np
sys.path.insert(0, 'src/neural')
from name_unique_lib import core

DS = '6ab10eb3b23ba_student_resource/student_resource/dataset/train/'; A = 'artifacts/improvement_v2/cpu/'
C = ce_data(); OUT = C + 'big_mix/'; os.makedirs(OUT, exist_ok=True)
N_OWN, CAP_NAME, CAP_ADDR = int(os.environ.get('N_OWN', 250000)), 4, 3
rng = np.random.default_rng(41)

def rows(p):
    with open(p, newline='') as h:
        r = csv.reader(h, delimiter='\t', quoting=csv.QUOTE_NONE); next(r); yield from r

def akey(addr):
    s = unicodedata.normalize('NFKD', addr or '').encode('ascii', 'ignore').decode().lower()
    num = re.search(r'\d+', s); w = [t for t in re.findall(r'[a-z]{4,}', s)]
    return f'{num.group()}|{max(w, key=len)}' if num and w else None

used = set()
for sp in ('dev', 'tune'):
    used.update(np.load(A + f'prepared_gate0_full_{sp}_v1/reference_ids.npy').tolist())
for f in sorted(glob.glob(C + 'train_*.pkl')):
    used.update(r[0] for r in pickle.load(open(f, 'rb')))
print('excluded owners', len(used), flush=True)

s1 = {x[0]: (x[1], x[2], x[3]) for x in rows(DS + 'train_source1.tsv') if x[0] not in used}
pick = rng.choice(sorted(s1), min(N_OWN, len(s1)), replace=False)
s1 = {a: s1[a] for a in pick}
gt = {}
with open(DS + 'train_ground_truth.tsv') as h:
    r = csv.reader(h, delimiter='\t'); next(r)
    for a, b in r:
        if a in s1:
            gt[a] = set(b.split(',')) - {''}
nk = collections.defaultdict(list); ak = collections.defaultdict(list)
for a, (n, ad, c) in s1.items():
    nk[(c, core(n))].append(a)
    k = akey(ad)
    if k:
        ak[(c, k)].append(a)
need_pos = set().union(*gt.values())
print('owners', len(s1), 'positives', len(need_pos), flush=True)

txt = {}; cand = collections.defaultdict(set)
for s in ('train_source2.tsv', 'train_source3.tsv'):
    for x in rows(DS + s):
        hit = False
        for a in nk.get((x[3], core(x[1])), ()):
            if len(cand[a]) < CAP_NAME + CAP_ADDR and x[0] not in gt.get(a, ()):
                cand[a].add(x[0]); hit = True
        k = akey(x[2])
        for a in (ak.get((x[3], k), ()) if k else ()):
            if len(cand[a]) < CAP_NAME + CAP_ADDR and x[0] not in gt.get(a, ()):
                cand[a].add(x[0]); hit = True
        if hit or x[0] in need_pos:
            txt[x[0]] = (x[1], x[2])
print('negatives', sum(map(len, cand.values())), flush=True)

out = []
for a, (n, ad, c) in s1.items():
    for t in gt.get(a, ()):
        if t in txt:
            out.append((a, t, 1, c, 0.5, n, ad) + txt[t])
    for t in cand.get(a, ()):
        out.append((a, t, 0, c, 0.5, n, ad) + txt[t])
out = [out[i] for i in rng.permutation(len(out))]
CH = 450_000
for j in range(0, len(out), CH):
    pickle.dump(out[j:j + CH], open(OUT + f'big_{j // CH:02d}.pkl', 'wb'))
print('pairs', len(out), 'pos frac', round(np.mean([r[2] for r in out]), 3), collections.Counter(r[3] for r in out), flush=True)
