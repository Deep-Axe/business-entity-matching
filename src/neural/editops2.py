"""Edit-operation + signed-offset features. Diagnostic block runs only as __main__."""
import json, re, sqlite3, sys, collections
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from editops import feats

DECOY_STEPS = {3, 4, 5, 7, 9, 11, 13, 21}

def nums(s):
    return [x for x in re.findall(r'\d+', s or '') if len(x) <= 7]

def offset_feats(a, b):
    A, B = nums(a), nums(b)
    if not A or not B:
        return [0, 0, 0, 0, 0, 99]
    sa, sb = set(A), set(B)
    oa = [x for x in A if x not in sb]; ob = [y for y in B if y not in sa]
    dropped = int(any(len(y) == len(x) - 1 and (x[1:] == y or x[:-1] == y) for x in oa for y in ob))
    if not oa or not ob:
        return [0, dropped, 0, 0, len(ob) - len(oa), 99]
    d = min(((int(y) - int(x)) for x in oa for y in ob), key=abs)
    return [1, dropped, int(d in DECOY_STEPS), int(1 <= d <= 25), len(ob) - len(oa), max(-60, min(60, d))]

if __name__ == '__main__':
    con = sqlite3.connect('file:artifacts/train_store_v2/records.sqlite3?mode=ro', uri=True)
    rows = []
    for split, pf, fleet in [('dev', 'enhanced_dev_probabilities', 'v2_gate0_dev'), ('tune', 'enhanced_full_tune_probabilities', 'v2_gate0_tune')]:
        p = np.load(f'artifacts/improvement_v2/cpu/fit_gate0_100k_v1/{pf}.npy')
        al = json.load(open(f'artifacts/improvement_v2/cpu/fit_gate0_100k_v1/{pf}.alignment.json'))
        for s in al['shards']:
            z = np.load(f"artifacts/improvement_v2/cloud/fleet_v1/collected/{fleet}/shards/{s['shard']}/features.npz")
            ps = p[s['pair_start']:s['pair_end']]
            for i in np.where(ps >= 0.02)[0]:
                ref = str(z['reference_ids'][z['query_ids'][i]]); tgt = str(z['target_ids'][i])
                an, aa = con.execute('select name,address from anchors where entity_id=?', (ref,)).fetchone()
                bn, ba = con.execute('select name,address from records where entity_id=?', (tgt,)).fetchone()
                rows.append((split, ref, float(ps[i]), int(z['y'][i]), feats(an, aa, bn, ba) + offset_feats(aa, ba)))
    split = np.array([r[0] for r in rows]); G = np.array([r[1] for r in rows]); P = np.array([r[2] for r in rows])
    Y = np.array([r[3] for r in rows]); X = np.array([r[4] for r in rows], float)
    L = np.log(np.clip(P, 1e-6, 1 - 1e-6) / (1 - np.clip(P, 1e-6, 1 - 1e-6)))
    XX = np.c_[X, L]
    oof = np.zeros(len(Y))
    for tr, te in GroupKFold(5).split(XX, Y, G):
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                           categorical_features=[14]).fit(XX[tr], Y[tr])
        oof[te] = m.predict_proba(XX[te])[:, 1]
    print('pairs', len(Y), 'AUC p100k', round(roc_auc_score(Y, P), 4), 'AUC new', round(roc_auc_score(Y, oof), 4))

    def macro(sp, score, t):
        D = f'artifacts/improvement_v2/cpu/prepared_gate0_full_{sp}_v1/'
        refs = np.load(D + 'reference_ids.npy'); tc = dict(zip(refs, np.load(D + 'true_counts.npy')))
        m = split == sp; tp = collections.Counter(); k = collections.Counter()
        for g, y, sc in zip(G[m], Y[m], score[m]):
            if sc >= t:
                k[g] += 1; tp[g] += y
        f = [1.0 if k[r] + tc[r] == 0 else 1.25 * tp[r] / (1.25 * tp[r] + 0.25 * (tc[r] - tp[r]) + (k[r] - tp[r])) for r in refs]
        return np.mean(f)

    for sp in ('tune', 'dev'):
        print(sp, 'p100k @0.76', round(macro(sp, P, 0.76), 5), ' new:',
              ' '.join(f'@{t}={macro(sp, oof, t):.5f}' for t in (0.5, 0.6, 0.7, 0.76, 0.8, 0.85, 0.9)))
