"""Dev/tune F0.5 of the deployable hybrid: existing pairs scored by the c4 stack OOF, new dense pairs by the e5-small union stack OOF.
Same owner logic for both arms: a target accepted by several dev/tune S1s goes to the highest score (no rival file for new pairs)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, ce_prep, dataset, owner_dir, policy_json, text_scorer_json, store, e5_small
import pickle, collections, sys
import numpy as np

C = ce_data(); A = ce_prep()
base_tag, uni_tag = sys.argv[1], sys.argv[2]
r0, o0 = pickle.load(open(C + f'stack_oof_{base_tag}.pkl', 'rb'))
r1, o1 = pickle.load(open(C + f'stack_oof_{uni_tag}.pkl', 'rb'))
new = [(r, v) for r, v in zip(r1, o1) if np.isnan(r[5])]
old = list(zip(r0, o0))
tc = {}
for sp in ('dev', 'tune'):
    tc.update(zip(np.load(A + f'prepared_gate0_full_{sp}_v1/reference_ids.npy'), np.load(A + f'prepared_gate0_full_{sp}_v1/true_counts.npy')))
spl = {}
for sp in ('dev', 'tune'):
    for x in np.load(A + f'prepared_gate0_full_{sp}_v1/reference_ids.npy'):
        spl[x] = sp

def f05(tp, k, t):
    return 1.0 if k + t == 0 else 1.25 * tp / (1.25 * tp + 0.25 * (t - tp) + (k - tp))

def score(t, t_new, use_new):
    acc = [(r, v) for r, v in old if v >= t] + ([(r, v) for r, v in new if v >= t_new] if use_new else [])
    best = {}
    for r, v in acc:
        if r[2] not in best or v > best[r[2]][1]:
            best[r[2]] = (r, v)
    tp = collections.Counter(); k = collections.Counter()
    for r, v in best.values():
        k[r[1]] += 1; tp[r[1]] += r[3]
    out = {}
    for sp in ('dev', 'tune'):
        out[sp] = np.mean([f05(tp[o], k[o], tc[o]) for o in spl if spl[o] == sp])
    return out

b = score(0.75, 0, False)
print(f'{base_tag} alone @0.75: tune {b["tune"]:.5f} dev {b["dev"]:.5f}   (new rows available: {len(new)}, true {sum(r[3] for r, _ in new)})')
for tn in (0.5, 0.75, 0.85, 0.9, 0.95, 0.98):
    h = score(0.75, tn, True)
    print(f'  + new pairs @{tn}: tune {h["tune"]:.5f} ({h["tune"] - b["tune"]:+.5f})  dev {h["dev"]:.5f} ({h["dev"] - b["dev"]:+.5f})  '
          f'new accepted {sum(v >= tn for _, v in new)} true {sum(r[3] for r, v in new if v >= tn)}')
