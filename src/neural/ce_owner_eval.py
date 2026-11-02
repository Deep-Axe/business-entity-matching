"""Blend CE with p100k, apply the frozen owner-assignment rule, select (w, t) on tune only, report dev.
usage: ce_owner_eval.py <tag_dir_name>"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, ce_prep, dataset, owner_dir, policy_json, text_scorer_json, store, e5_small
import sys, pickle, collections, json
import numpy as np
sys.path.insert(0, 'tools')
from v3_owner_assignment import winner

C = ce_data(); tag = sys.argv[1]
policy = json.load(open('artifacts/improvement_v3/structural_v1/owner_eval/policy.json'))['selected']
riv = pickle.load(open(C + 'rivals.pkl', 'rb')); rce = np.load(C + f'rivals.pkl.{tag}.npy')

def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

def blend(pce, p100, w):
    return 1 / (1 + np.exp(-(w * lg(pce) + (1 - w) * lg(p100))))

def f05(tp, k, t):
    return 1.0 if k + t == 0 else 1.25 * tp / (1.25 * tp + 0.25 * (t - tp) + (k - tp))

def evaluate(sp, w, t, owner=True):
    rows = pickle.load(open(C + f'{sp}.pkl', 'rb')); pce = np.load(C + f'{tag}/{sp}_pce.npy')
    s = blend(pce, np.array([r[4] for r in rows]), w)
    rs = blend(rce, np.array([r[4] for r in riv]), w)
    claim = collections.defaultdict(list)
    for r, v in zip(riv, rs):
        if v >= t:
            claim[r[1]].append((r[0], v, r[9]))
    P = f'artifacts/improvement_v2/cpu/prepared_gate0_full_{sp}_v1/'
    refs = np.load(P + 'reference_ids.npy'); tc = dict(zip(refs, np.load(P + 'true_counts.npy')))
    tp = collections.Counter(); k = collections.Counter()
    for r, v in zip(rows, s):
        if v < t:
            continue
        cl = claim.get(r[1], [])
        others = [c for c in cl if c[0] != r[0]]
        if owner and others:
            focal_text = next((c[2] for c in cl if c[0] == r[0]), None)
            if focal_text is None:
                focal_text = 0.0
            cands = [(r[0], v, focal_text)] + others
            cands.sort(key=lambda c: c[0])
            ch = winner([c[1] for c in cands], [c[2] for c in cands], **policy)
            if ch is None or cands[ch][0] != r[0]:
                continue
        k[r[0]] += 1; tp[r[0]] += r[2]
    return float(np.mean([f05(tp[o], k[o], tc[o]) for o in refs]))

if __name__ == '__main__':
    grid = [(w, t) for w in (1.0, 0.8, 0.6, 0.5, 0.0) for t in (0.5, 0.6, 0.7, 0.76, 0.8, 0.85, 0.9, 0.95)]
    res = {(w, t): evaluate('tune', w, t) for w, t in grid}
    for w in (1.0, 0.8, 0.6, 0.5, 0.0):
        print(f'tune w={w}: ' + ' '.join(f'{t}:{res[(w, t)]:.5f}' for t in (0.5, 0.6, 0.7, 0.76, 0.8, 0.85, 0.9, 0.95)))
    w, t = max(res, key=res.get)
    print(f'baseline p100k@0.76 + owner: tune {evaluate("tune", 0.0, 0.76):.5f} dev {evaluate("dev", 0.0, 0.76):.5f}')
    print(f'tune-selected w={w} t={t}: tune {res[(w, t)]:.5f}  dev {evaluate("dev", w, t):.5f}  (dev without owner step {evaluate("dev", w, t, False):.5f})')
