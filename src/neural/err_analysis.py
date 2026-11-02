"""Error taxonomy for the current best dev/tune policy: c8 stack on existing candidates (t=0.75) + dense k10 new pairs (t=0.9).
Writes ce/err_links.pkl: one record per true link and per false accepted link, with pattern tags."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, ce_prep, dataset, owner_dir, policy_json, text_scorer_json, store, e5_small
import csv, pickle, re, collections, sys
import numpy as np
sys.path.insert(0, 'src/neural')
from name_unique_lib import core

C = ce_data(); DS = dataset() + 'train/'; A = 'artifacts/improvement_v2/cpu/'
T_OLD, T_NEW = 0.75, 0.9
r0, o0 = pickle.load(open(C + 'stack_oof_c8.pkl', 'rb'))
r1, o1 = pickle.load(open(C + 'stack_oof_e5_cont1300k_k10.pkl', 'rb'))
spl = {}
for sp in ('dev', 'tune'):
    for x in np.load(A + f'prepared_gate0_full_{sp}_v1/reference_ids.npy'):
        spl[x] = sp
truth = {}
with open(DS + 'train_ground_truth.tsv') as h:
    for a, b in csv.reader(h, delimiter='\t'):
        if a in spl:
            truth[a] = set(b.split(',')) - {''}
cand = {}
for r, v in zip(r0, o0):
    cand[(r[1], r[2])] = ('old', float(v))
for r, v in zip(r1, o1):
    if np.isnan(r[5]):
        cand[(r[1], r[2])] = ('new', float(v))
acc = {}
for (a, t), (route, v) in cand.items():
    if v >= (T_OLD if route == 'old' else T_NEW):
        if t not in acc or v > acc[t][1]:
            acc[t] = (a, v)
pred = collections.defaultdict(set)
for t, (a, v) in acc.items():
    pred[a].add(t)

need_t = {t for a in spl for t in truth.get(a, ())} | {t for (a, t) in cand}
s1 = {}; rec = {}; cty = {}; freq = collections.Counter(); exact = collections.Counter()
with open(DS + 'train_source1.tsv') as h:
    r = csv.reader(h, delimiter='\t', quoting=csv.QUOTE_NONE); next(r)
    for x in r:
        freq[(x[3], ' '.join(core(x[1])))] += 1; exact[(x[3], x[1].strip().lower())] += 1
        if x[0] in spl:
            s1[x[0]] = (x[1], x[2]); cty[x[0]] = x[3]
for s in ('train_source2.tsv', 'train_source3.tsv'):
    with open(DS + s) as h:
        for x in csv.reader(h, delimiter='\t', quoting=csv.QUOTE_NONE):
            if x[0] in need_t:
                rec[x[0]] = (x[1], x[2])

tok = lambda s: set(re.findall(r'\w+', s.lower()))
def tags(a, t):
    sn, sa = s1[a]; tn, ta = rec[t]
    return dict(empty_addr=not ta.strip(), web=bool(re.search(r'\.(com|in|net|org|co|fr)\b|^@|www\.', tn.lower())),
                native=bool(re.search(r'[\u0900-\u0DFF]', tn)), name_j=len(tok(sn) & tok(tn)) / max(1, len(tok(sn) | tok(tn))),
                addr_j=len(tok(sa) & tok(ta)) / max(1, len(tok(sa) | tok(ta))),
                core_twins=freq[(cty[a], ' '.join(core(sn)))] - 1, exact_twins=exact[(cty[a], sn.strip().lower())] - 1)

out = []
for a, sp in spl.items():
    tr = truth.get(a, set())
    for t in tr:
        c = cand.get((a, t))
        if t in pred[a]:
            stage = 'accepted'
        elif c is None:
            stage = 'absent'
        elif c[1] >= (T_OLD if c[0] == 'old' else T_NEW):
            stage = 'lost_owner'
        else:
            stage = 'rejected'
        out.append(dict(sp=sp, s1=a, t=t, y=1, stage=stage, route=c[0] if c else None, score=c[1] if c else None, cty=cty[a],
                        n_true=len(tr), n_acc_true=len(tr & pred[a]), s1_text=s1[a], t_text=rec[t], **tags(a, t)))
    for t in pred[a] - tr:
        c = cand[(a, t)]
        out.append(dict(sp=sp, s1=a, t=t, y=0, stage='false_match', route=c[0], score=c[1], cty=cty[a],
                        n_true=len(tr), n_acc_true=len(tr & pred[a]), s1_text=s1[a], t_text=rec[t], owner=acc[t][0], **tags(a, t)))
pickle.dump(out, open(C + 'err_links.pkl', 'wb'))
for sp in ('dev', 'tune'):
    print(sp, collections.Counter(o['stage'] for o in out if o['sp'] == sp))
