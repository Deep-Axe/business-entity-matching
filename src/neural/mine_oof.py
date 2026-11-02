"""Out-of-fold error mix for kF/kG, same protocol as src/kaggle/kaggle_kernel_mine/run.py.
Writes CE_DATA/mine_mix/train_00.pkl plus symlinks of dev.pkl and tune.pkl.
Requires kA and kB weights and a GPU for OOF scoring.
usage: mine_oof.py
"""
import glob, os, pickle, shutil, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, weights_dir, e5_base

C = ce_data()
W = C + 'mine_work/'; os.makedirs(W, exist_ok=True)
MX = C + 'mine_mix/'; os.makedirs(MX, exist_ok=True)
files = sorted(glob.glob(C + 'train_*.pkl'))
if not files:
    raise SystemExit('no train_*.pkl in CE_DATA')
per = 1_900_000 // len(files)
rA = np.random.default_rng(7)
for k, f in enumerate(files):
    part = pickle.load(open(f, 'rb')); a = rA.choice(len(part), min(len(part), per), replace=False)
    inA = np.zeros(len(part), bool); inA[a] = True
    pickle.dump([part[i] for i in np.flatnonzero(~inA)], open(W + f'forA_{k:02d}.pkl', 'wb'))
    pickle.dump([part[i] for i in np.flatnonzero(inA)], open(W + f'forB_{k:02d}.pkl', 'wb'))
    print(os.path.basename(f), 'kA-trained', int(inA.sum()), flush=True)
    del part

def _wt(tag):
    for parent in (os.environ.get('CE_OUT', C), str(weights_dir())):
        p = os.path.join(parent.rstrip('/\\'), tag, 'model.pt')
        if os.path.exists(p):
            return p
    raise SystemExit(f'missing {tag}/model.pt (set CE_OUT or BER_WEIGHTS)')

score = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ce_score.py')
py = sys.executable
model = e5_base()
for tag, pat in (('kA', 'forA_'), ('kB', 'forB_')):
    cmd = [py, '-u', score, '--model', model, '--weights', _wt(tag), '--tag', 'oof', '--bs', '256'] + sorted(glob.glob(W + pat + '*.pkl'))
    print('score', tag, flush=True)
    r = subprocess.run(cmd)
    if r.returncode:
        raise SystemExit(f'ce_score {tag} failed')

rows, s = [], []
for f in sorted(glob.glob(W + '*.pkl')):
    rows += pickle.load(open(f, 'rb')); s.append(np.load(f + '.oof.npy'))
s = np.concatenate(s); y = np.array([r[2] for r in rows]); rng = np.random.default_rng(5)
groups = {'fn_hard': (y == 1) & (s < 0.5), 'fp_hard': (y == 0) & (s > 0.5),
          'fn_near': (y == 1) & (s >= 0.5) & (s < 0.9), 'fp_near': (y == 0) & (s > 0.1) & (s <= 0.5)}
cap = {'fn_hard': 10**9, 'fp_hard': 10**9, 'fn_near': 60000, 'fp_near': 60000}
pick = set()
for g, m in groups.items():
    ix = np.flatnonzero(m); ix = rng.choice(ix, min(len(ix), cap[g]), replace=False) if len(ix) else ix
    pick.update(ix.tolist()); print(f'{g}: available {m.sum()} taken {len(ix)}', flush=True)
n_hard = len(pick)
rest = np.setdiff1d(np.arange(len(rows)), np.fromiter(pick, int) if pick else np.array([], dtype=int))
if len(rest):
    pick.update(rng.choice(rest, min(len(rest), max(120000, n_hard)), replace=False).tolist())
mix = [rows[i] for i in rng.permutation(np.fromiter(pick, int))]
print('mix', len(mix), 'pos', np.mean([r[2] for r in mix]), flush=True)
pickle.dump(mix, open(MX + 'train_00.pkl', 'wb'))
for a in ('dev.pkl', 'tune.pkl'):
    src, dst = os.path.abspath(C + a), MX + a
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)
shutil.rmtree(W)
print('wrote', MX, flush=True)
