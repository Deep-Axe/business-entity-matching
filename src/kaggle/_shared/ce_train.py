"""Fine-tune a multilingual cross-encoder on (S1, target) text pairs; evaluate owner macro F0.5 on dev/tune."""
import argparse, glob, pickle, time, os, collections, math
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ap = argparse.ArgumentParser()
ap.add_argument('--model', default=os.environ.get('BER_E5', 'intfloat/multilingual-e5-small'))
ap.add_argument('--tag', default='e5_pilot')
ap.add_argument('--max-train', type=int, default=1_200_000)
ap.add_argument('--min-avail-gb', type=float, default=3.0)
ap.add_argument('--epochs', type=float, default=1.0)
ap.add_argument('--bs', type=int, default=64)
ap.add_argument('--lr', type=float, default=5e-5)
ap.add_argument('--maxlen', type=int, default=128)
ap.add_argument('--init', default='')
ap.add_argument('--exclude', default='', help='seed:max_train list of earlier runs whose samples are excluded')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--eval-bs', type=int, default=128)
ap.add_argument('--ckpt-every', type=int, default=4000)
ap.add_argument('--resume', action='store_true', help='continue from OUT/ckpt.pt (model, optimizer, scheduler, scaler, data position)')
args = ap.parse_args()

def _slash(p):
    p = os.fspath(p)
    return p if p.endswith(('/', '\\')) else p + os.sep

D = _slash(os.environ.get('CE_DATA', 'work/ce'))
OUT = _slash(os.environ.get('CE_OUT', D)) + args.tag + '/'
PREP = _slash(os.environ.get('CE_PREP', 'model/prep'))
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
_rev = {} if os.path.isdir(args.model) else {
    'revision': os.environ.get(
        'BER_E5_BASE_REVISION' if 'e5-base' in args.model else 'BER_E5_REVISION',
        'd128750597153bb5987e10b1c3493a34e5a4502a' if 'e5-base' in args.model
        else '614241f622f53c4eeff9890bdc4f31cfecc418b3')}
tok = AutoTokenizer.from_pretrained(args.model, **_rev)

def seg(n, a):
    return f"{n} | {a if a else '-'}"

def avail_gb():
    for l in open('/proc/meminfo'):
        if l.startswith('MemAvailable'):
            return int(l.split()[1]) / 2**20

class Tok:
    """Compact token store: one flat int32 array plus offsets."""
    def __init__(self):
        self.parts, self.lens = [], []
    def add(self, rows):
        for i in range(0, len(rows), 20000):
            ch = rows[i:i + 20000]
            enc = tok([seg(r[5], r[6]) for r in ch], [seg(r[7], r[8]) for r in ch], truncation=True, max_length=args.maxlen)['input_ids']
            self.parts.append(np.concatenate([np.asarray(x, np.int32) for x in enc])); self.lens += [len(x) for x in enc]
    def freeze(self):
        self.flat = np.concatenate(self.parts); del self.parts
        self.len = np.asarray(self.lens, np.int32); del self.lens
        self.off = np.concatenate([[0], np.cumsum(self.len, dtype=np.int64)])
        return self
    def __len__(self):
        return len(self.len)
    def get(self, j):
        return self.flat[self.off[j]:self.off[j + 1]]

def encode(rows):
    t = Tok(); t.add(rows); return t.freeze()

def batches(ids, order, bs):
    for i in range(0, len(order), bs):
        b = order[i:i + bs]; L = int(ids.len[b].max())
        x = np.full((len(b), L), tok.pad_token_id, np.int64); m = np.zeros((len(b), L), np.int64)
        for k, j in enumerate(b):
            v = ids.get(j); x[k, :len(v)] = v; m[k, :len(v)] = 1
        yield b, torch.from_numpy(x).cuda(), torch.from_numpy(m).cuda()

files = sorted(glob.glob(D + 'train_*.pkl'))
tr_ids, ylist = Tok(), []
t0 = time.time()
prior = [tuple(map(int, x.split(':'))) for x in args.exclude.split(',')] if args.exclude else []
prior_rngs = [(np.random.default_rng(s), m) for s, m in prior]
for f in files:
    if avail_gb() < args.min_avail_gb:
        raise SystemExit(f'MemAvailable {avail_gb():.1f} GB below {args.min_avail_gb} GB; stopping before {f}')
    part = pickle.load(open(f, 'rb'))
    seen = set()
    for r_e, m_e in prior_rngs:
        seen.update(r_e.choice(len(part), min(len(part), m_e // len(files)), replace=False).tolist())
    if seen:
        part = [x for i, x in enumerate(part) if i not in seen]
    if args.max_train:
        part = [part[i] for i in rng.choice(len(part), min(len(part), args.max_train // len(files)), replace=False)]
    tr_ids.add(part); ylist.append(np.array([r[2] for r in part], np.float32)); del part
tr_ids.freeze(); ytr = np.concatenate(ylist)
dev = pickle.load(open(D + 'dev.pkl', 'rb')); tune = pickle.load(open(D + 'tune.pkl', 'rb'))
ev = {'tune': (tune, encode(tune)), 'dev': (dev, encode(dev))}
print(f'train {len(tr_ids)} pos {ytr.mean():.3f} tokenized {time.time() - t0:.0f}s mean len {tr_ids.len.mean():.1f} '
      f'token store {tr_ids.flat.nbytes / 2**20:.0f} MiB, MemAvailable {avail_gb():.1f} GB', flush=True)

model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1, **_rev)
if args.init:
    model.load_state_dict(torch.load(args.init, map_location='cpu'))
model = model.cuda()
model.base_model.embeddings.word_embeddings.weight.requires_grad_(False)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
steps = int(math.ceil(len(tr_ids) / args.bs) * args.epochs); warm = int(0.03 * steps)
sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm)))
scaler = torch.amp.GradScaler(); lossf = torch.nn.BCEWithLogitsLoss()

def predict(ids):
    model.eval(); out = np.zeros(len(ids), np.float32)
    order = np.argsort(ids.len)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for b, x, m in batches(ids, order, args.eval_bs):
            out[b] = torch.sigmoid(model(input_ids=x, attention_mask=m).logits.float().squeeze(-1)).cpu().numpy()
    model.train(); return out

def macro(sp, rows, score, t):
    P = PREP + f'prepared_gate0_full_{sp}_v1/'
    refs = np.load(P + 'reference_ids.npy'); tc = dict(zip(refs, np.load(P + 'true_counts.npy')))
    tp = collections.Counter(); k = collections.Counter()
    for r, s in zip(rows, score):
        if s >= t:
            k[r[0]] += 1; tp[r[0]] += r[2]
    return float(np.mean([1.0 if k[r] + tc[r] == 0 else 1.25 * tp[r] / (1.25 * tp[r] + 0.25 * (tc[r] - tp[r]) + (k[r] - tp[r])) for r in refs]))

def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

def evaluate(final):
    """Blend weight and threshold are selected on tune only; dev is reported at the tune choice."""
    sc = {}
    for sp, (rows, ids) in ev.items():
        t1 = time.time(); pce = predict(ids); dt = time.time() - t1
        p100 = np.array([r[4] for r in rows]); np.save(OUT + f'{sp}_pce.npy', pce)
        sc[sp] = (rows, pce, p100)
        print(f'  {sp}: {len(ids) / dt:.0f} pairs/s  p100k@0.76 {macro(sp, rows, p100, 0.76):.5f}', flush=True)
    grid = []
    for w in ((1.0, 0.8, 0.6, 0.5) if final else (1.0,)):
        for t in (0.5, 0.6, 0.7, 0.76, 0.8, 0.85, 0.9, 0.95):
            rows, pce, p100 = sc['tune']
            grid.append((macro('tune', rows, 1 / (1 + np.exp(-(w * lg(pce) + (1 - w) * lg(p100)))), t), w, t))
    f_tune, w, t = max(grid)
    rows, pce, p100 = sc['dev']
    f_dev = macro('dev', rows, 1 / (1 + np.exp(-(w * lg(pce) + (1 - w) * lg(p100)))), t)
    print(f'  tune-selected w={w} t={t}: tune {f_tune:.5f}  dev {f_dev:.5f}', flush=True)
    return w, t

def save_ckpt(epoch_state, epoch_start):
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), scaler=scaler.state_dict(), step=step,
                    epoch_state=epoch_state, epoch_start=epoch_start), OUT + 'ckpt.tmp')
    os.replace(OUT + 'ckpt.tmp', OUT + 'ckpt.pt')

step = 0; skip_to = 0; base_step = 0; t0 = time.time(); run = 0.0
if args.resume and os.path.exists(OUT + 'ckpt.pt'):
    ck = torch.load(OUT + 'ckpt.pt', map_location='cuda', weights_only=False)
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sch.load_state_dict(ck['sch']); scaler.load_state_dict(ck['scaler'])
    rng.bit_generator.state = ck['epoch_state']; step = ck['epoch_start']; skip_to = ck['step']
    print(f'resumed at step {skip_to}/{steps}', flush=True); del ck
model.train()
while step < steps:
    epoch_state, epoch_start = rng.bit_generator.state, step
    order = rng.permutation(len(tr_ids))
    chunks = [order[i:i + args.bs * 50] for i in range(0, len(order), args.bs * 50)]
    order = np.concatenate([c[np.argsort(tr_ids.len[c])] for c in chunks])
    bl = [order[i:i + args.bs] for i in range(0, len(order), args.bs)]; rng.shuffle(bl)
    for b in bl:
        if step >= steps:
            break
        if step < skip_to:
            step += 1; continue
        if step == skip_to and skip_to:
            t0 = time.time() - 1e-9; skip_to = 0; base_step = step
        _, x, m = next(batches(tr_ids, b, args.bs))
        with torch.autocast('cuda', dtype=torch.float16):
            logit = model(input_ids=x, attention_mask=m).logits.float().squeeze(-1)
        loss = lossf(logit, torch.from_numpy(ytr[b]).cuda())
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
        step += 1; run = 0.98 * run + 0.02 * loss.item()
        if step % 500 == 0:
            print(f'step {step}/{steps} loss {run:.4f} {((step - base_step) * args.bs) / (time.time() - t0):.0f} pairs/s', flush=True)
        if step % args.ckpt_every == 0:
            torch.save(model.state_dict(), OUT + 'model.tmp'); os.replace(OUT + 'model.tmp', OUT + 'model.pt')
            save_ckpt(epoch_state, epoch_start)
        if step % 8000 == 0 and step < steps:
            evaluate(False)
torch.save(model.state_dict(), OUT + 'model.pt')
evaluate(True)
print('done', f'{time.time() - t0:.0f}s', flush=True)
