"""Independent re-check of the key-channel dev/tune claims."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, ce_prep, dataset, owner_dir, policy_json, text_scorer_json, store, e5_small
import csv, pickle, collections
import numpy as np

C = ce_data(); A = ce_prep()
DS = '6ab10eb3b23ba_student_resource/student_resource/dataset/train/'
spl, tc = {}, {}
for sp in ('dev', 'tune'):
    P = A + f'prepared_gate0_full_{sp}_v1/'
    ids = np.load(P + 'reference_ids.npy'); spl.update({x: sp for x in ids}); tc.update(zip(ids, np.load(P + 'true_counts.npy')))
gt = collections.defaultdict(set)
with open(DS + 'train_ground_truth.tsv') as h:
    for a, b in csv.reader(h, delimiter='\t'):
        if a in spl:
            gt[a] = set(b.split(',')) - {''}
bad_tc = sum(len(gt[a]) != tc[a] for a in spl)
print(f'1. ground-truth sizes vs prepared true_counts: {bad_tc} mismatching S1s of {len(spl)}')

new_keys = set()
for sp in ('dev', 'tune'):
    kn = pickle.load(open(C + f'{sp}_kcnew.pkl', 'rb'))
    keys = [(r[0], r[1]) for r in kn]; new_keys |= set(keys)
    k10 = {(r[0], r[1]) for r in pickle.load(open(C + f'{sp}_k10.pkl', 'rb'))}
    orig = {(r[0], r[1]) for r in pickle.load(open(C + f'{sp}.pkl', 'rb'))}
    lab_ok = sum((r[1] in gt[r[0]]) == bool(r[2]) for r in kn)
    print(f'2. {sp}: new rows {len(kn)}, unique {len(set(keys))}, already in k10 {len(set(keys) & k10)}, in original {len(set(keys) & orig)}, '
          f'true (file label) {sum(r[2] for r in kn)}, true (re-derived from GT) {sum(r[1] in gt[r[0]] for r in kn)}, labels agree {lab_ok}/{len(kn)}, '
          f'S1 outside split {sum(spl.get(r[0]) != sp for r in kn)}, p100k all NaN {all(np.isnan(r[4]) for r in kn)}')
    kc = pickle.load(open(C + f'{sp}_kc.pkl', 'rb'))
    L = {s: len(np.load(C + f'{sp}_kc.pkl.{s}.npy')) for s in ('ops', 'nf', 'nf2')}
    pce = np.load(C + f'e5_cont1300k/{sp}_kc_pce.npy'); tail = np.load(C + f'{sp}_kcnew.pkl.e5_cont1300k.npy')
    print(f'3. {sp}: kc rows {len(kc)} = k10 {len(k10)} + new {len(kn)}; array lengths {L}, pce {len(pce)}; '
          f'tail matches new scores {np.allclose(pce[-len(kn):], tail, equal_nan=True)}; row order k10-first {[(r[0], r[1]) for r in kc[-len(kn):]] == keys}')

def load(tag):
    r, o = pickle.load(open(C + f'stack_oof_{tag}.pkl', 'rb'))
    return {(x[1], x[2]): (v, x[3], x[0]) for x, v in zip(r, o)}, r
base, _ = load('c8'); u10, _ = load('e5_cont1300k_k10'); ukc, rkc = load('e5_cont1300k_kc')
print(f'4. OOF rows: kc stack {len(rkc)}; key rows present in it {len(new_keys & set(ukc))}/{len(new_keys)}')
cty = {}
for sp in ('dev', 'tune'):
    for r in pickle.load(open(C + f'{sp}_kc.pkl', 'rb')):
        cty[r[0]] = r[3]
k10new = {k for k in u10 if k not in base}
print(f'   dense-new rows {len(k10new)}; key-new rows {len(new_keys)}; overlap {len(k10new & new_keys)}')

def f05(tp, k, t):
    return 1.0 if k + t == 0 else 1.25 * tp / (1.25 * tp + 0.25 * (t - tp) + (k - tp))

def run(edges, dedupe=True):
    best = {}
    for (a, t), v in edges.items():
        if not dedupe:
            best[(a, t)] = (a, v)
        elif t not in best or v > best[t][1]:
            best[t] = (a, v)
    tp, k = collections.Counter(), collections.Counter()
    for key, (a, v) in best.items():
        t = key[1] if isinstance(key, tuple) else key
        k[a] += 1; tp[a] += t in gt[a]
    out = {}
    for sp in ('dev', 'tune'):
        out[sp] = np.mean([f05(tp[o], k[o], tc[o]) for o in spl if spl[o] == sp])
        for c in ('US', 'India', 'France'):
            ss = [f05(tp[o], k[o], tc[o]) for o in spl if spl[o] == sp and cty.get(o) == c]
            out[f'{sp}_{c}'] = (np.mean(ss), len(ss))
    return out

T, TN = 0.75, 0.9
E0 = {k: v[0] for k, v in base.items() if v[0] >= T}
arms = {
    'C8 alone': E0,
    '+ dense (k10 stack)': {**E0, **{k: u10[k][0] for k in k10new if u10[k][0] >= TN}},
    '+ dense (kc stack)': {**E0, **{k: ukc[k][0] for k in k10new if ukc[k][0] >= TN}},
    '+ dense + key rows (kc stack)': {**E0, **{k: ukc[k][0] for k in k10new | new_keys if ukc[k][0] >= TN}},
}
res = {}
for n, e in arms.items():
    for dd in (True, False):
        o = run(e, dd); res[(n, dd)] = o
        print(f'5. {n:32s} owner-dedupe={dd!s:5s} tune {o["tune"]:.5f} dev {o["dev"]:.5f}  ' +
              '  '.join(f'{sp}-{c} {o[f"{sp}_{c}"][0]:.5f}(n={o[f"{sp}_{c}"][1]})' for sp in ('dev', 'tune') for c in ('US', 'India', 'France')))
acc = [(k, ukc[k][0], k[1] in gt[k[0]]) for k in new_keys if ukc[k][0] >= TN]
print(f'6. key rows accepted at {TN}: {len(acc)}, true {sum(a[2] for a in acc)}; by country',
      collections.Counter((cty[k[0]], y) for k, _, y in acc))
d_acc = {k for k in k10new if u10[k][0] >= TN}; d_acc2 = {k for k in k10new if ukc[k][0] >= TN}
print(f'7. dense rows accepted: k10 stack {len(d_acc)} (true {sum(k[1] in gt[k[0]] for k in d_acc)}), kc stack {len(d_acc2)} '
      f'(true {sum(k[1] in gt[k[0]] for k in d_acc2)}); only-k10 {len(d_acc - d_acc2)}, only-kc {len(d_acc2 - d_acc)}')
