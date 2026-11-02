import glob, os, subprocess, sys, time, shutil

DATA = os.path.dirname(glob.glob('/kaggle/input/**/dev.pkl', recursive=True)[0]) + '/'
MIX = glob.glob('/kaggle/input/**/hard_mix.pkl', recursive=True)[0]
INIT = {k: glob.glob(f'/kaggle/input/**/{k}/model.pt', recursive=True)[0] for k in ('kA', 'kB')}
W = '/kaggle/working/'

def packaged(name):
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, '..', '_shared', name), os.path.join(here, name), DATA + name):
        if os.path.exists(p):
            return os.path.abspath(p)
    raise SystemExit(f'missing {name}; ship src/kaggle/_shared/{name} with this kernel or in the data input')

print('data', DATA, 'mix', MIX, 'init', INIT, flush=True)
subprocess.run(['nvidia-smi'])

PREP = W + 'prep/'
for sp in ('dev', 'tune'):
    d = PREP + f'prepared_gate0_full_{sp}_v1/'; os.makedirs(d, exist_ok=True)
    for a in ('reference_ids', 'true_counts'):
        shutil.copy(DATA + f'prep_{sp}_{a}.npy', d + f'{a}.npy')
HD = W + 'hd/'; os.makedirs(HD, exist_ok=True)
for a, b in ((DATA + 'dev.pkl', 'dev.pkl'), (DATA + 'tune.pkl', 'tune.pkl'), (MIX, 'train_00.pkl')):
    os.symlink(a, HD + b)

from huggingface_hub import snapshot_download
M = snapshot_download('intfloat/multilingual-e5-base', local_dir=W + 'e5_base',
                      revision=os.environ.get('BER_E5_BASE_REVISION', 'd128750597153bb5987e10b1c3493a34e5a4502a'),
                      allow_patterns=['config.json', 'model.safetensors', 'tokenizer*', 'sentencepiece*', 'special_tokens_map.json'])

env = dict(os.environ, CE_DATA=HD, CE_OUT=W, CE_PREP=PREP, TOKENIZERS_PARALLELISM='false')
jobs = {
    'kD': (['--seed', '21', '--init', INIT['kA']], '0'),
    'kE': (['--seed', '22', '--init', INIT['kB']], '1'),
}

def launch(cmd, gpu, log):
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

procs, logs, cmds = [], [], {}
for tag, (extra, gpu) in jobs.items():
    cmd = [sys.executable, '-u', packaged('ce_train.py'), '--model', M, '--tag', tag, '--max-train', '0', '--bs', '64',
           '--lr', '1e-5', '--min-avail-gb', '4', '--eval-bs', '128', '--ckpt-every', '1000'] + extra
    procs.append(launch(cmd, gpu, W + f'train_{tag}.log')); logs.append(W + f'train_{tag}.log'); cmds[tag] = (cmd, gpu)
codes = follow(procs, logs); print('train exit codes', codes, flush=True)
retry = [(t, c) for (t, c), code in zip(cmds.items(), codes) if code != 0]
if retry:
    procs = [launch(c + ['--resume'], g, W + f'train_{t}_resume.log') for t, (c, g) in retry]
    print('resume exit codes', follow(procs, [W + f'train_{t}_resume.log' for t, _ in retry]), flush=True)

procs, logs = [], []
inputs = [DATA + 'rivals.pkl'] + sorted(glob.glob(DATA + 'band_test_*.pkl'))
for tag, (extra, gpu) in jobs.items():
    if not os.path.exists(W + f'{tag}/model.pt'):
        continue
    os.makedirs(W + f'scores_{tag}', exist_ok=True)
    cmd = [sys.executable, '-u', packaged('ce_score.py'), '--model', M, '--weights', W + f'{tag}/model.pt', '--tag', tag,
           '--outdir', W + f'scores_{tag}', '--bs', '256'] + inputs
    procs.append(launch(cmd, gpu, W + f'score_{tag}.log')); logs.append(W + f'score_{tag}.log')
print('score exit codes', follow(procs, logs), flush=True)
shutil.rmtree(W + 'e5_base', ignore_errors=True)
for tag in jobs:
    if os.path.exists(W + f'{tag}/ckpt.pt'):
        os.remove(W + f'{tag}/ckpt.pt')
print(sorted(glob.glob(W + '**/*.npy', recursive=True)), flush=True)
