"""Build cross-encoder text datasets: train pickles plus dev/tune (p100k>=0.01 only).

Default layout (original 100k/200k collection):
  BER_TRAIN_SHARDS  glob of features.npz
  BER_GATE_DIR      directory with enhanced_{dev,full_tune}_probabilities.npy + alignment.json
  BER_DEV_FLEET / BER_TUNE_FLEET  collected feature-shard parents

Local labeled v2_run_pipeline outputs (pair_ids.npz + probabilities.npy):
  BER_TRAIN_INFER / BER_DEV_INFER / BER_TUNE_INFER  partition directories
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, store, v2_dir
import json, sqlite3, glob, pickle, time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

A = v2_dir()
OUT = ce_data()
os.makedirs(OUT, exist_ok=True)
con = sqlite3.connect(f'file:{store("train")}?mode=ro', uri=True)

def lookup(table, ids):
    ids = list(set(ids)); d = {}
    for i in range(0, len(ids), 900):
        ch = ids[i:i + 900]
        for e, n, a in con.execute(f"select entity_id,name,address from {table} where entity_id in ({','.join('?' * len(ch))})", ch):
            d[e] = (n, a)
    return d

def infer_shards(root):
    root = Path(root)
    shards = sorted(root.glob('shards/[0-9]*')) or sorted(root.glob('*/shards/[0-9]*'))
    if not shards:
        raise SystemExit(f'no shards under {root}')
    return shards

def from_infer(split, root, pmin=0.01, keep_pos=None, chunk=None):
    # Dev/tune must match test_pairs.py: keep p>=pmin only. Train may retain
    # true matches below the gate, as the original selector path did.
    if keep_pos is None:
        keep_pos = split == 'train'
    rows, n = [], 0
    parts, k = [], 0
    t0 = time.time()
    for shard in infer_shards(root):
        z = dict(np.load(shard / 'pair_ids.npz'))
        p = np.load(shard / 'probabilities.npy')
        y = z.get('y')
        keep = np.flatnonzero((p >= pmin) | ((y == 1) if (keep_pos and y is not None) else False))
        part = []
        refs = [str(z['reference_ids'][z['query_ids'][i]]) for i in keep]
        tg = [str(z['target_ids'][i]) for i in keep]
        an = lookup('anchors', refs); rc = lookup('records', tg)
        for i, r, t in zip(keep, refs, tg):
            lab = int(y[i]) if y is not None else -1
            part.append((r, t, lab, str(z['countries'][z['query_ids'][i]]), float(p[i])) + an[r] + rc[t])
        if chunk is None:
            rows += part
        else:
            parts += part
            if len(parts) >= chunk:
                pickle.dump(parts, open(OUT + f'train_{k:02d}.pkl', 'wb')); n += len(parts)
                print(f'train_{k:02d} {n} {time.time() - t0:.0f}s', flush=True)
                parts = []; k += 1
    if chunk is None:
        pickle.dump(rows, open(OUT + f'{split}.pkl', 'wb'))
        print(split, 'pairs', len(rows), 'pos', sum(r[2] for r in rows if r[2] == 1), flush=True)
        return
    if parts:
        pickle.dump(parts, open(OUT + f'train_{k:02d}.pkl', 'wb')); n += len(parts)
        print(f'train_{k:02d} {n} {time.time() - t0:.0f}s', flush=True)

dev_i, tune_i, train_i = os.environ.get('BER_DEV_INFER'), os.environ.get('BER_TUNE_INFER'), os.environ.get('BER_TRAIN_INFER')
if dev_i and tune_i:
    from_infer('dev', dev_i)
    from_infer('tune', tune_i)
    if train_i:
        from_infer('train', train_i, chunk=400_000)
    sys.exit()

def fx(z):
    return np.c_[z['X'], z['channel_scores'], z['channel_ranks'].astype(np.float32)]

train_glob = os.environ.get('BER_TRAIN_SHARDS', A + 'cloud/fleet_v1/collected/v2_gate0_train200k_p*/shards/*/features.npz')
gate = os.environ.get('BER_GATE_DIR', A + 'cpu/fit_gate0_100k_v1')
if not gate.endswith(('/', '\\')):
    gate += os.sep
dev_fleet = os.environ.get('BER_DEV_FLEET', A + 'cloud/fleet_v1/collected/v2_gate0_dev')
tune_fleet = os.environ.get('BER_TUNE_FLEET', A + 'cloud/fleet_v1/collected/v2_gate0_tune')
train_shards = sorted(glob.glob(train_glob))
print('train shards', len(train_shards), flush=True)
if not train_shards:
    raise SystemExit('no training feature shards; set BER_TRAIN_SHARDS or BER_TRAIN_INFER/BER_DEV_INFER/BER_TUNE_INFER')

Xs, ys = [], []
for f in train_shards[:40]:
    z = np.load(f); Xs.append(fx(z)); ys.append(z['y'])
sel = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1).fit(np.vstack(Xs), np.concatenate(ys))
del Xs, ys

def eval_split(split, pf, fleet):
    p = np.load(gate + f'{pf}.npy')
    al = json.load(open(gate + f'{pf}.alignment.json'))
    rows, ps_all = [], []
    for s in al['shards']:
        z = np.load(os.path.join(fleet, 'shards', s['shard'], 'features.npz'))
        ps = p[s['pair_start']:s['pair_end']]
        ps_all.append((ps, sel.predict_proba(fx(z))[:, 1]))
        idx = np.where(ps >= 0.01)[0]
        refs = [str(z['reference_ids'][z['query_ids'][i]]) for i in idx]; tg = [str(z['target_ids'][i]) for i in idx]
        an = lookup('anchors', refs); rc = lookup('records', tg)
        cty = [str(z['countries'][z['query_ids'][i]]) for i in idx]
        for i, r, t, c in zip(idx, refs, tg, cty):
            rows.append((r, t, int(z['y'][i]), c, float(ps[i])) + an[r] + rc[t])
    pickle.dump(rows, open(OUT + f'{split}.pkl', 'wb'))
    print(split, 'pairs', len(rows), 'pos', sum(r[2] for r in rows), flush=True)
    return ps_all

ps_all = eval_split('dev', 'enhanced_dev_probabilities', dev_fleet) + \
         eval_split('tune', 'enhanced_full_tune_probabilities', tune_fleet)
p100 = np.concatenate([a for a, b in ps_all]); psel = np.concatenate([b for a, b in ps_all])
hard = psel[p100 >= 0.01]
thr = float(np.quantile(hard, 0.003))
print('selector thr', thr, 'coverage of p100k>=0.01:', (hard >= thr).mean(), 'kept per dev pair', (psel >= thr).mean(), flush=True)

t0 = time.time(); part = []; n = 0; k = 0
for j, f in enumerate(train_shards):
    z = np.load(f)
    ps = sel.predict_proba(fx(z))[:, 1]
    idx = np.where((ps >= thr) | (z['y'] == 1))[0]
    refs = [str(z['reference_ids'][z['query_ids'][i]]) for i in idx]; tg = [str(z['target_ids'][i]) for i in idx]
    an = lookup('anchors', refs); rc = lookup('records', tg)
    for i, r, t in zip(idx, refs, tg):
        part.append((r, t, int(z['y'][i]), str(z['countries'][z['query_ids'][i]]), float(ps[i])) + an[r] + rc[t])
    if (j + 1) % 100 == 0 or j + 1 == len(train_shards):
        pickle.dump(part, open(OUT + f'train_{k:02d}.pkl', 'wb')); n += len(part)
        print(f'shard {j + 1} pairs {n} pos-frac {np.mean([r[2] for r in part]):.3f} {time.time() - t0:.0f}s', flush=True)
        part = []; k += 1
