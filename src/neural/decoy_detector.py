"""Can a target's own text tell whether it belongs to any S1 (owned) or is an unowned decoy?"""
import sqlite3, random
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack

con = sqlite3.connect('file:artifacts/train_store_v2/records.sqlite3?mode=ro', uri=True)
n = con.execute('select max(row_id) from records').fetchone()[0]
random.seed(0)
ids = random.sample(range(1, n + 1), 400000)
rows = []
for chunk in range(0, len(ids), 5000):
    part = ids[chunk:chunk + 5000]
    q = f"select r.source, r.country, r.name, r.address, l.target_id is not null from records r left join labels l on l.target_id = r.entity_id where r.row_id in ({','.join(map(str, part))})"
    rows.extend(con.execute(q).fetchall())
src = np.array([r[0] for r in rows]); y = np.array([r[4] for r in rows], dtype=int)
text_n = [f'{r[1]} {r[2]}' for r in rows]; text_a = [r[3] for r in rows]
hv = HashingVectorizer(analyzer='char_wb', ngram_range=(2, 5), n_features=2 ** 21, lowercase=False, alternate_sign=False, norm='l2')
hw = HashingVectorizer(analyzer='word', ngram_range=(1, 2), n_features=2 ** 20, lowercase=False, token_pattern=r'\S+', alternate_sign=False, norm='l2')
X = hstack([hv.transform(text_n), hv.transform(text_a), hw.transform(text_n), hw.transform(text_a),
            np.c_[src == 2, [not a.strip() for a in text_a]].astype(float)]).tocsr()
tr = np.arange(len(y)) < 300000
print('owned rate', y.mean())
m = LogisticRegression(C=4, max_iter=300, solver='liblinear').fit(X[tr], y[tr])
p = m.predict_proba(X[~tr])[:, 1]
print('target-only owned-vs-decoy AUC', round(roc_auc_score(y[~tr], p), 4))
for s in (2, 3):
    mm = src[~tr] == s
    print(f'  source {s} AUC', round(roc_auc_score(y[~tr][mm], p[mm]), 4))
for t in (0.05, 0.1, 0.2):
    lo = p < t
    print(f'  p<{t}: {lo.mean():.3f} of targets, owned among them {y[~tr][lo].mean():.3f}')
