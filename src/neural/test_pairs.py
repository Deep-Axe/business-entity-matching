"""Gather all test candidate pairs with p100k>=0.01 plus text, one pickle per partition.

Inputs, in order:
  BER_QUEUE       controller queue.json with complete[id].directory / .references
  BER_TEST_PARTS  glob of v2_run_pipeline output directories (each has shards/ and manifest.json)
  default         $BER_V2/inference/submission_controller_v1/queue.json
                  or $BER_V2/inference/full_test_100k_p*_v1
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber_paths import ce_data, store, v2_dir
import json, sqlite3, pickle, glob, time
from pathlib import Path
import numpy as np

OUT = Path(ce_data() + 'test'); OUT.mkdir(parents=True, exist_ok=True)
con = sqlite3.connect(f'file:{store("test")}?mode=ro', uri=True)

def lookup(table, ids):
    ids = list(set(ids)); d = {}
    for i in range(0, len(ids), 900):
        ch = ids[i:i + 900]
        for e, n, a in con.execute(f"select entity_id,name,address from {table} where entity_id in ({','.join('?' * len(ch))})", ch):
            d[e] = (n, a)
    return d

def partitions():
    qf = os.environ.get('BER_QUEUE', '')
    default_q = v2_dir() + 'inference/submission_controller_v1/queue.json'
    if not qf and os.path.exists(default_q):
        qf = default_q
    if qf:
        return [(k, v) for k, v in sorted(json.load(open(qf))['complete'].items(), key=lambda x: int(x[0]))]
    globp = os.environ.get('BER_TEST_PARTS', v2_dir() + 'inference/full_test_100k_p*_v1')
    items = []
    for d in sorted(glob.glob(globp)):
        man = Path(d) / 'manifest.json'
        if not man.exists():
            continue
        m = json.loads(man.read_text())
        key = str(m.get('partition', len(items)))
        items.append((key, {'directory': d, 'references': m['references']}))
    if not items:
        raise SystemExit('no test partitions; set BER_QUEUE or BER_TEST_PARTS')
    return items

t0 = time.time(); total = refs_total = 0
for key, entry in partitions():
    out = OUT / f'part_{int(key):03d}.pkl'
    if out.exists():
        continue
    rows = []; nref = 0
    for shard in sorted((Path(entry['directory']) / 'shards').glob('[0-9]*')):
        z = dict(np.load(shard / 'pair_ids.npz')); p = np.load(shard / 'probabilities.npy')
        nref += len(z['reference_ids'])
        idx = np.flatnonzero(p >= 0.01)
        refs = [str(z['reference_ids'][z['query_ids'][i]]) for i in idx]; tg = [str(z['target_ids'][i]) for i in idx]
        an = lookup('anchors', refs); rc = lookup('records', tg)
        for i, r, t in zip(idx, refs, tg):
            rows.append((r, t, -1, str(z['countries'][z['query_ids'][i]]), float(p[i])) + an[r] + rc[t])
    assert nref == entry['references'], (key, nref, entry['references'])
    pickle.dump(rows, open(str(out) + '.tmp', 'wb')); os.replace(str(out) + '.tmp', out)
    total += len(rows); refs_total += nref
    print(f'part {key} refs {nref} pairs {len(rows)} total {total} {time.time() - t0:.0f}s', flush=True)
print('done', total, refs_total)
