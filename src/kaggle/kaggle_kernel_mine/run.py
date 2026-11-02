"""Out-of-fold neural error mining: kA and kB were trained on disjoint training halves, so each scores the other's half.
Mined errors + near-errors + an ordinary slice train kF (from kA) and kG (from kB); then score rivals and the test band."""
import glob, os, subprocess, sys, time, shutil, pickle
import numpy as np

DATA = os.path.dirname(glob.glob('/kaggle/input/**/dev.pkl', recursive=True)[0]) + '/'
INIT = {k: glob.glob(f'/kaggle/input/**/{k}/model.pt', recursive=True)[0] for k in ('kA', 'kB')}
W = '/kaggle/working/'

def packaged(name):
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, '..', '_shared', name), os.path.join(here, name), DATA + name):
        if os.path.exists(p):
            return os.path.abspath(p)
    raise SystemExit(f'missing {name}; ship src/kaggle/_shared/{name} with this kernel or in the data input')

print('data', DATA, 'init', INIT, flush=True)
subprocess.run(['nvidia-smi'])

PREP = W + 'prep/'
for sp in ('dev', 'tune'):
    d = PREP + f'prepared_gate0_full_{sp}_v1/'; os.makedirs(d, exist_ok=True)
    for a in ('reference_ids', 'true_counts'):
        shutil.copy(DATA + f'prep_{sp}_{a}.npy', d + f'{a}.npy')

from huggingface_hub import snapshot_download
M = snapshot_download('intfloat/multilingual-e5-base', local_dir=W + 'e5_base',
                      revision=os.environ.get('BER_E5_BASE_REVISION', 'd128750597153bb5987e10b1c3493a34e5a4502a'),
                      allow_patterns=['config.json', 'model.safetensors', 'tokenizer*', 'sentencepiece*', 'special_tokens_map.json'])

def launch(cmd, gpu, log, env):
    return subprocess.Popen(cmd, env=dict(env, CUDA_VISIBLE_DEVICES=gpu), stdout=open(log, 'w'), stderr=subprocess.STDOUT)

def follow(procs, logs):
    pos = {l: 0 for l in logs}
    while any(p.poll() is None for p in procs):
        time.sleep(120)
        for l in logs:
            with open(l) as h:
                h.seek(pos[l]); new = h.read(); pos[l] = h.tell()
            for line in new.splitlines():
                if line.startswith('step ') and int(line.split()[1].split('/')[0]) % 1000:
                    continue
                print(f'[{os.path.basename(l)}] {line}', flush=True)
    for l in logs:
        with open(l) as h:
            h.seek(pos[l]); print(f'[{os.path.basename(l)}] {h.read()}', flush=True)
    return [p.returncode for p in procs]

env = dict(os.environ, TOKENIZERS_PARALLELISM='false')

# 1. reconstruct kA's per-file sample (seed 7, 1.9M/10 per file) exactly as ce_train.py drew it
files = sorted(glob.glob(DATA + 'train_*.pkl')); per = 1_900_000 // len(files)
rA = np.random.default_rng(7); MS = W + 'mine/'; os.makedirs(MS, exist_ok=True)
for k, f in enumerate(files):
    part = pickle.load(open(f, 'rb')); a = rA.choice(len(part), min(len(part), per), replace=False)
    inA = np.zeros(len(part), bool); inA[a] = True
    pickle.dump([part[i] for i in np.flatnonzero(~inA)], open(MS + f'forA_{k:02d}.pkl', 'wb'))
    pickle.dump([part[i] for i in np.flatnonzero(inA)], open(MS + f'forB_{k:02d}.pkl', 'wb'))
    print(f'{os.path.basename(f)}: {len(part)} rows, kA-trained {inA.sum()}', flush=True)
    del part

# 2. out-of-fold scoring: kA scores rows it never saw, kB scores kA's rows (kB excluded them)
procs, logs = [], []
for tag, gpu, pat in (('kA', '0', 'forA_'), ('kB', '1', 'forB_')):
    cmd = [sys.executable, '-u', packaged('ce_score.py'), '--model', M, '--weights', INIT[tag], '--tag', 'oof',
           '--bs', '256'] + sorted(glob.glob(MS + pat + '*.pkl'))
    procs.append(launch(cmd, gpu, W + f'oofscore_{tag}.log', env)); logs.append(W + f'oofscore_{tag}.log')
print('oof score exit codes', follow(procs, logs), flush=True)

# 3. mine
rows, s = [], []
for f in sorted(glob.glob(MS + '*.pkl')):
    r = pickle.load(open(f, 'rb')); rows += r; s.append(np.load(f + '.oof.npy'))
s = np.concatenate(s); y = np.array([r[2] for r in rows]); rng = np.random.default_rng(5)
np.save(W + 'oof_scores.npy', s); np.save(W + 'oof_labels.npy', y)
groups = {'fn_hard': (y == 1) & (s < 0.5), 'fp_hard': (y == 0) & (s > 0.5),
          'fn_near': (y == 1) & (s >= 0.5) & (s < 0.9), 'fp_near': (y == 0) & (s > 0.1) & (s <= 0.5)}
cap = {'fn_hard': 10**9, 'fp_hard': 10**9, 'fn_near': 60000, 'fp_near': 60000}
pick = set()
for g, m in groups.items():
    ix = np.flatnonzero(m); ix = rng.choice(ix, min(len(ix), cap[g]), replace=False) if len(ix) else ix
    pick.update(ix.tolist()); print(f'{g}: available {m.sum()} taken {len(ix)}', flush=True)
n_hard = len(pick)
rest = np.setdiff1d(np.arange(len(rows)), np.fromiter(pick, int)); pick.update(rng.choice(rest, min(len(rest), max(120000, n_hard)), replace=False).tolist())
mix = [rows[i] for i in rng.permutation(np.fromiter(pick, int))]
print(f'oof pairs {len(rows)}  oof AUC-free error rate {np.mean((s > 0.5) != (y == 1)):.4f}  mix {len(mix)} pos {np.mean([r[2] for r in mix]):.3f}', flush=True)
MX = W + 'mix/'; os.makedirs(MX, exist_ok=True)
pickle.dump(mix, open(MX + 'train_00.pkl', 'wb')); del rows, mix
for a in ('dev.pkl', 'tune.pkl'):
    os.symlink(DATA + a, MX + a)
shutil.rmtree(MS)

# 4. train kF (from kA) and kG (from kB) on the mined mix
tenv = dict(env, CE_DATA=MX, CE_OUT=W, CE_PREP=PREP)
jobs = {'kF': (['--seed', '31', '--init', INIT['kA']], '0'), 'kG': (['--seed', '32', '--init', INIT['kB']], '1')}
procs, logs, cmds = [], [], {}
for tag, (extra, gpu) in jobs.items():
    cmd = [sys.executable, '-u', packaged('ce_train.py'), '--model', M, '--tag', tag, '--max-train', '0', '--bs', '64',
           '--lr', '1e-5', '--min-avail-gb', '4', '--eval-bs', '128', '--ckpt-every', '1000'] + extra
    procs.append(launch(cmd, gpu, W + f'train_{tag}.log', tenv)); logs.append(W + f'train_{tag}.log'); cmds[tag] = (cmd, gpu)
codes = follow(procs, logs); print('train exit codes', codes, flush=True)
retry = [(t, c) for (t, c), code in zip(cmds.items(), codes) if code != 0]
if retry:
    procs = [launch(c + ['--resume'], g, W + f'train_{t}_resume.log', tenv) for t, (c, g) in retry]
    print('resume exit codes', follow(procs, [W + f'train_{t}_resume.log' for t, _ in retry]), flush=True)

# 5. score rivals + test band; keep model.pt, drop optimizer checkpoints
procs, logs = [], []
inputs = [DATA + 'rivals.pkl'] + sorted(glob.glob(DATA + 'band_test_*.pkl'))
for tag, (extra, gpu) in jobs.items():
    if not os.path.exists(W + f'{tag}/model.pt'):
        continue
    os.makedirs(W + f'scores_{tag}', exist_ok=True)
    cmd = [sys.executable, '-u', packaged('ce_score.py'), '--model', M, '--weights', W + f'{tag}/model.pt', '--tag', tag,
           '--outdir', W + f'scores_{tag}', '--bs', '256'] + inputs
    procs.append(launch(cmd, gpu, W + f'score_{tag}.log', env)); logs.append(W + f'score_{tag}.log')
print('score exit codes', follow(procs, logs), flush=True)
shutil.rmtree(W + 'e5_base', ignore_errors=True); shutil.rmtree(MX, ignore_errors=True)
for tag in jobs:
    if os.path.exists(W + f'{tag}/ckpt.pt'):
        os.remove(W + f'{tag}/ckpt.pt')
print(sorted(glob.glob(W + '**/*.npy', recursive=True)), flush=True)
