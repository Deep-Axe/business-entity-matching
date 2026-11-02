"""Stack edit-op/offset features + logit(p100k) + logit(CE) with HistGB (owner-grouped CV over dev+tune).
Owner assignment as in ce_owner_eval; (t) selected on tune OOF, reported on dev OOF.
usage: stack_eval.py TAG [--test]
  --test: fit on all dev+tune, write stacked scores.
  USUF=_k10|_kc trains on {dev,tune}{USUF}.pkl (must exist).
  TESTGLOB=glob[;glob] scores only those pickles (the dense or key head);
           without TESTGLOB, scores test/part_*.pkl.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, ce_prep, expand_route_globs, policy_json
import sys, pickle, collections, json, glob, os
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
from v3_owner_assignment import winner

C = ce_data(); tag = sys.argv[1]
USUF = os.environ.get('USUF', '')
TESTGLOB = os.environ.get('TESTGLOB', '')
policy = json.load(open(policy_json()))['selected']
policy.update(json.loads(os.environ.get('POLICY', '{}')))
CTY = {'US': 0, 'India': 1, 'France': 2}

def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

NOCTY = os.environ.get('NOCTY') == '1'

OWNF = os.environ.get('OWNF') == '1'

def owner_feats(rows, pce):
    """Per-S1 context over its own candidate list: ranks, counts of strong candidates, gap to best."""
    p = np.array([r[4] for r in rows]); g = collections.defaultdict(list)
    for i, r in enumerate(rows):
        g[r[0]].append(i)
    F = np.zeros((len(rows), 12), np.float32)
    for o, ix in g.items():
        ix = np.array(ix)
        for c, v in enumerate((p[ix], pce[ix])):
            order = np.argsort(-v); rank = np.empty(len(ix)); rank[order] = np.arange(len(ix))
            lv = lg(v); srt = np.sort(v)[::-1]
            F[ix, 6 * c + 0] = rank
            F[ix, 6 * c + 1] = (v >= 0.5).sum()
            F[ix, 6 * c + 2] = lv - lg(srt[0])
            F[ix, 6 * c + 3] = srt[3] if len(srt) > 3 else 0
            F[ix, 6 * c + 4] = np.array([((v >= 0.5) & (np.array([rows[j][1][:2] for j in ix]) == rows[i][1][:2])).sum() for i in ix])
            F[ix, 6 * c + 5] = len(ix)
    return F

DTOK = os.environ.get('DTOK') == '1'
NAMEF = os.environ.get('NAMEF') == '1'
NOP100 = os.environ.get('NOP100') == '1'
NAMEF2 = os.environ.get('NAMEF2') == '1'
ADDRF = os.environ.get('ADDRF') == '1'
NAMEF3 = os.environ.get('NAMEF3') == '1'
_dt = {}

def mat(rows, ops, pce, dt=None):
    base = [ops, lg(pce)] if NOP100 else [ops, lg(np.array([r[4] for r in rows])), lg(pce)]
    if DTOK:
        base.append(dt)
    if NAMEF:
        base.append(np.load(_nf_path[id(rows)]))
    if NAMEF2:
        base.append(np.load(_nf_path[id(rows)].replace('.nf.npy', '.nf2.npy')))
    if NAMEF3:
        base.append(np.load(_nf_path[id(rows)].replace('.nf.npy', '.nf3.npy')))
    if ADDRF:
        base.append(np.load(_nf_path[id(rows)].replace('.nf.npy', '.af.npy')))
    if OWNF:
        base.append(owner_feats(rows, pce))
    return np.c_[tuple(base if NOCTY else base + [[CTY[r[3]] for r in rows]])].astype(np.float32)

def model():
    hp = json.loads(os.environ.get('HP', '{}'))
    return HistGradientBoostingClassifier(max_iter=hp.get('it', 400), learning_rate=hp.get('lr', 0.04), max_leaf_nodes=hp.get('leaves', 31),
                                          min_samples_leaf=hp.get('msl', 20), l2_regularization=1.0, categorical_features=[14] if NOCTY else [14, 30 + 12 * OWNF], random_state=0)

_nf_path = {}

def _reg(r, p):
    _nf_path[id(r)] = p; return r

def _pkl(sp):
    if not USUF:
        return C + f'{sp}.pkl'
    p = C + f'{sp}{USUF}.pkl'
    if not os.path.exists(p):
        raise FileNotFoundError(f'USUF={USUF} missing {p} (build the union pickle before scoring this head)')
    return p

def _pce(sp, pkl):
    cands = []
    if USUF:
        cands.append(C + f'{tag}/{sp}{USUF}_pce.npy')
    cands += [C + f'{tag}/{sp}_pce.npy', pkl + f'.{tag}.npy']
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'no {tag} scores for {pkl}')

def _suffix():
    return f'{"_nocty" if NOCTY else ""}{"_ownf" if OWNF else ""}{"_dtok" if DTOK else ""}{"_nf" if NAMEF else ""}{"2" if NAMEF2 else ""}{"3" if NAMEF3 else ""}{"_nop" if NOP100 else ""}{"_af" if ADDRF else ""}'

rows, M = [], []
for sp in ('dev', 'tune'):
    pkl = _pkl(sp)
    r = pickle.load(open(pkl, 'rb')); _nf_path[id(r)] = pkl + '.nf.npy'
    M.append(mat(r, np.load(pkl + '.ops.npy'), np.load(_pce(sp, pkl)), np.load(pkl + '.dtok.npy') if DTOK else None)); rows += [(sp,) + x for x in r]
M = np.vstack(M); Y = np.array([r[3] for r in rows]); G = np.array([r[1] for r in rows])

def _score_pkl(f, require=False):
    out = f'{f}.stack_{tag}{_suffix()}.npy'
    tag_scores = f'{f}.{tag}.npy'
    if os.path.exists(out):
        return
    if not os.path.exists(tag_scores):
        if require:
            raise FileNotFoundError(f'missing {tag_scores}')
        return
    tr = pickle.load(open(f, 'rb')); _nf_path[id(tr)] = f + '.nf.npy'
    s = m.predict_proba(mat(tr, np.load(f + '.ops.npy'), np.load(tag_scores), np.load(f + '.dtok.npy') if DTOK else None))[:, 1].astype(np.float32)
    np.save(out, s)

if '--test' in sys.argv:
    m = model().fit(M, Y)
    if TESTGLOB:
        try:
            files = expand_route_globs(TESTGLOB, strict=True)
        except FileNotFoundError as exc:
            raise SystemExit(f'TESTGLOB: {exc}') from exc
    else:
        files = sorted(glob.glob(C + 'test/part_*.pkl'))
    for f in files:
        _score_pkl(f, require=bool(TESTGLOB))
    print('test stacked scores written'); sys.exit()

riv = pickle.load(open(C + 'rivals.pkl', 'rb')); _nf_path[id(riv)] = C + 'rivals.pkl.nf.npy'
_o = OWNF; OWNF = False
riv_pce = C + f'rivals.pkl.{tag}.npy'
RM = mat(riv, np.load(C + 'rivals.pkl.ops.npy'), np.load(riv_pce), np.load(C + 'rivals.pkl.dtok.npy') if DTOK else None)
M0 = np.vstack([mat(_reg(pickle.load(open(_pkl(sp), 'rb')), _pkl(sp) + '.nf.npy'), np.load(_pkl(sp) + '.ops.npy'), np.load(_pce(sp, _pkl(sp))), np.load(_pkl(sp) + '.dtok.npy') if DTOK else None) for sp in ('dev', 'tune')])
OWNF = _o

oof = np.zeros(len(Y)); fold_of = {}; models = []
for k, (tr, te) in enumerate(GroupKFold(5).split(M, Y, G)):
    m = model().fit(M[tr], Y[tr]); oof[te] = m.predict_proba(M[te])[:, 1]
    models.append(model().fit(M0[tr], Y[tr]) if OWNF else m)
    for i in te:
        fold_of[rows[i][2]] = k
rs = np.zeros(len(riv))
rf = np.array([fold_of.get(r[1], 0) for r in riv])
for k, m in enumerate(models):
    if (rf == k).any():
        rs[rf == k] = m.predict_proba(RM[rf == k])[:, 1]

def f05(tp, k, t):
    return 1.0 if k + t == 0 else 1.25 * tp / (1.25 * tp + 0.25 * (t - tp) + (k - tp))

def evaluate(sp, t, owner=True):
    claim = collections.defaultdict(list)
    for r, v in zip(riv, rs):
        if v >= t:
            claim[r[1]].append((r[0], v, r[9]))
    P = ce_prep() + f'prepared_gate0_full_{sp}_v1/'
    refs = np.load(P + 'reference_ids.npy'); tc = dict(zip(refs, np.load(P + 'true_counts.npy')))
    tp = collections.Counter(); kk = collections.Counter()
    for r, v in zip(rows, oof):
        if r[0] != sp or v < t:
            continue
        cl = claim.get(r[2], []); others = [c for c in cl if c[0] != r[1]]
        if owner and others:
            ft = next((c[2] for c in cl if c[0] == r[1]), 0.0)
            cands = sorted([(r[1], v, ft)] + others, key=lambda c: c[0])
            ch = winner([c[1] for c in cands], [c[2] for c in cands], **policy)
            if ch is None or cands[ch][0] != r[1]:
                continue
        kk[r[1]] += 1; tp[r[1]] += r[3]
    return float(np.mean([f05(tp[o], kk[o], tc[o]) for o in refs]))

pickle.dump((rows, oof), open(C + f'stack_oof_{tag}{"_dtok" if DTOK else ""}.pkl', 'wb'))
grid = (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
res = {t: evaluate('tune', t) for t in grid}
print('tune OOF + owner: ' + ' '.join(f'{t}:{v:.5f}' for t, v in res.items()))
t = max(res, key=res.get)
print(f'tune-selected t={t}: tune {res[t]:.5f}  dev {evaluate("dev", t):.5f}  (dev no owner {evaluate("dev", t, False):.5f})')
print('dev all t: ' + ' '.join(f'{u}:{evaluate("dev", u):.5f}' for u in grid))
