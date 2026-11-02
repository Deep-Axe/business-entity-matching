"""Diff-representation model: edit ops + signed offsets + bag of added/dropped tokens. CV owner-set F0.5 on dev/tune."""
import json, re, sqlite3, sys, collections, unicodedata, pickle, os
import numpy as np
from rapidfuzz import fuzz
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
for _f in ('editops.py', 'editops2.py'):
    exec(open('work/' + _f).read().split('con = sqlite3.connect')[0].replace('from editops import feats', ''))

def toks(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
    return re.findall(r'[a-z]+', s)

def diff_tokens(an, bn):
    A, B = toks(an), toks(bn)
    ua = [t for t in A if t not in B and not any(fuzz.ratio(t, u) >= 75 for u in B)]
    ub = [t for t in B if t not in A and not any(fuzz.ratio(t, u) >= 75 for u in A)]
    la = sorted({t for t in A if t in LEGAL}); lb = sorted({t for t in B if t in LEGAL})
    out = {'+' + t for t in ub} | {'-' + t for t in ua}
    if la != lb:
        out.add('legal:' + '_'.join(la) + '>' + '_'.join(lb))
    return out

CACHE = 'work/diff_rows.pkl'
if os.path.exists(CACHE):
    rows = pickle.load(open(CACHE, 'rb'))
else:
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
                rows.append((split, ref, tgt, float(ps[i]), int(z['y'][i]), feats(an, aa, bn, ba) + offset_feats(aa, ba),
                             diff_tokens(an, bn)))
    pickle.dump(rows, open(CACHE, 'wb'))

split = np.array([r[0] for r in rows]); G = np.array([r[1] for r in rows]); P = np.array([r[3] for r in rows])
Y = np.array([r[4] for r in rows]); X = np.array([r[5] for r in rows], float)
L = np.log(np.clip(P, 1e-6, 1 - 1e-6) / (1 - np.clip(P, 1e-6, 1 - 1e-6)))
df = collections.Counter(t for r in rows for t in r[6])
vocab = {t: i for i, t in enumerate(t for t, c in df.most_common() if c >= 5)}
ri, ci = [], []
for k, r in enumerate(rows):
    for t in r[6]:
        if t in vocab:
            ri.append(k); ci.append(vocab[t])
T = csr_matrix((np.ones(len(ri)), (ri, ci)), shape=(len(rows), len(vocab)))
print('pairs', len(rows), 'diff-token vocabulary', len(vocab))

def run(M, label):
    oof = np.zeros(len(Y))
    for tr, te in GroupKFold(5).split(M, Y, G):
        m = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=10,
                                           l2_regularization=1.0)
        m.fit(M[tr], Y[tr]); oof[te] = m.predict_proba(M[te])[:, 1]
    print(label, 'AUC', round(roc_auc_score(Y, oof), 4))
    return oof

def macro(sp, score, t):
    D = f'artifacts/improvement_v2/cpu/prepared_gate0_full_{sp}_v1/'
    refs = np.load(D + 'reference_ids.npy'); tc = dict(zip(refs, np.load(D + 'true_counts.npy')))
    m = split == sp; tp = collections.Counter(); k = collections.Counter()
    for g, y, sc in zip(G[m], Y[m], score[m]):
        if sc >= t:
            k[g] += 1; tp[g] += y
    return np.mean([1.0 if k[r] + tc[r] == 0 else 1.25 * tp[r] / (1.25 * tp[r] + 0.25 * (tc[r] - tp[r]) + (k[r] - tp[r])) for r in refs])

base = np.c_[X, L].astype(np.float32)
results = {'ops+offset': run(base, 'ops+offset'),
           'ops+offset+diff tokens': run(np.hstack([base, T.toarray().astype(np.float32)]), 'ops+offset+diff tokens')}
for sp in ('tune', 'dev'):
    print(sp, 'p100k @0.76', round(macro(sp, P, 0.76), 5))
    for k, v in results.items():
        print(f'   {k:26s}', ' '.join(f'@{t}={macro(sp, v, t):.5f}' for t in (0.6, 0.7, 0.76, 0.8, 0.85)))
np.save('work/diff_oof.npy', results['ops+offset+diff tokens'])
