"""Rival claim rows for owner-assignment eval.
Prefer owner_eval/rivals.sqlite3 (or BER_RIVALS). If that file is absent, build
from targets claimed by two or more Source-1s in dev.pkl+tune.pkl.
"""
import os, pickle, sqlite3, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, owner_dir, text_scorer_json, store
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
from v3_owner_assignment import load_text, text_logit

C = ce_data()
S = sqlite3.connect(f'file:{store("train")}?mode=ro', uri=True)
cands = [os.environ.get('BER_RIVALS', ''), str(owner_dir() / 'rivals.sqlite3'),
         'artifacts/improvement_v3/structural_v1/owner_eval/rivals.sqlite3']
db = next((p for p in cands if p and os.path.exists(p)), '')

def lookup(table, ids):
    ids = list(set(ids)); d = {}
    for i in range(0, len(ids), 900):
        ch = ids[i:i + 900]
        for e, n, a, c in S.execute(f"select entity_id,name,address,country from {table} where entity_id in ({','.join('?' * len(ch))})", ch):
            d[e] = (n, a, c)
    return d

if db:
    R = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    claims = list(R.execute('select target, owner, p, text from claims'))
    an = lookup('anchors', [c[1] for c in claims]); rc = lookup('records', [c[0] for c in claims])
    lab = dict(S.execute('select target_id, reference_id from labels'))
    rows = [(o, t, int(lab.get(t) == o), an[o][2], p) + an[o][:2] + rc[t][:2] + (tx,) for t, o, p, tx in claims]
    print('from', db, flush=True)
else:
    text_model = load_text(str(text_scorer_json()))
    src = []
    for sp in ('dev', 'tune'):
        src += pickle.load(open(C + f'{sp}.pkl', 'rb'))
    by_t = collections.defaultdict(list)
    for r in src:
        by_t[r[1]].append(r)
    rows = []
    for t, cs in by_t.items():
        if len({c[0] for c in cs}) < 2:
            continue
        for r in cs:
            tx = text_logit({'name': r[5], 'address': r[6]}, {'name': r[7], 'address': r[8]}, text_model)
            rows.append((r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], float(tx)))
    print('built from overlapping dev/tune targets', flush=True)
pickle.dump(rows, open(C + 'rivals.pkl', 'wb'))
print('rival claim rows', len(rows), 'true', sum(r[2] for r in rows))
