"""Score row pickles with a trained cross-encoder; writes <input>.<tag>.npy next to each input (skips existing)."""
import argparse, pickle, time, os, sys
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ap = argparse.ArgumentParser()
ap.add_argument('--model', default=os.environ.get('BER_E5', 'intfloat/multilingual-e5-small'))
ap.add_argument('--weights', required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--maxlen', type=int, default=128)
ap.add_argument('--bs', type=int, default=512)
ap.add_argument('--band', default='', help='SUFFIX:lo:hi  score only rows whose <input>.SUFFIX.npy is in [lo, hi); others NaN')
ap.add_argument('--outdir', default='', help='write scores here instead of next to the inputs')
ap.add_argument('inputs', nargs='+')
args = ap.parse_args()
if args.outdir:
    os.makedirs(args.outdir, exist_ok=True)
_rev = {} if os.path.isdir(args.model) else {
    'revision': os.environ.get(
        'BER_E5_BASE_REVISION' if 'e5-base' in args.model else 'BER_E5_REVISION',
        'd128750597153bb5987e10b1c3493a34e5a4502a' if 'e5-base' in args.model
        else '614241f622f53c4eeff9890bdc4f31cfecc418b3')}
tok = AutoTokenizer.from_pretrained(args.model, **_rev)
model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1, **_rev)
model.load_state_dict(torch.load(args.weights, map_location='cpu')); model = model.cuda().half().eval()

def seg(n, a):
    return f"{n} | {a if a else '-'}"

t0 = time.time(); n = 0
for f in args.inputs:
    out = os.path.join(args.outdir, os.path.basename(f)) + f'.{args.tag}.npy' if args.outdir else f'{f}.{args.tag}.npy'
    if os.path.exists(out):
        continue
    rows = pickle.load(open(f, 'rb')); n_all = len(rows); keep = None
    if args.band:
        suf, lo, hi = args.band.split(':'); s = np.load(f'{f}.{suf}.npy')
        keep = np.flatnonzero((s >= float(lo)) & (s < float(hi))); rows = [rows[i] for i in keep]
    enc = []
    for i in range(0, len(rows), 20000):
        ch = rows[i:i + 20000]
        enc += tok([seg(r[5], r[6]) for r in ch], [seg(r[7], r[8]) for r in ch], truncation=True, max_length=args.maxlen)['input_ids']
    del rows
    lens = np.array([len(x) for x in enc]); order = np.argsort(lens); res = np.zeros(len(enc), np.float32)
    with torch.no_grad():
        for i in range(0, len(order), args.bs):
            b = order[i:i + args.bs]; L = lens[b].max()
            x = np.full((len(b), L), tok.pad_token_id, np.int64); m = np.zeros((len(b), L), np.int64)
            for k, j in enumerate(b):
                x[k, :lens[j]] = enc[j]; m[k, :lens[j]] = 1
            lo = model(input_ids=torch.from_numpy(x).cuda(), attention_mask=torch.from_numpy(m).cuda()).logits.float().squeeze(-1)
            res[b] = torch.sigmoid(lo).cpu().numpy()
    if keep is not None:
        full = np.full(n_all, np.nan, np.float32); full[keep] = res; res = full
    np.save(out + '.tmp.npy', res); os.replace(out + '.tmp.npy', out)
    n += len(enc)
    print(f'{os.path.basename(f)} {len(enc)} pairs, cumulative {n} at {n / (time.time() - t0):.0f} pairs/s', flush=True)
