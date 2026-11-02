"""Read-only input audit: does E5's 128-token limit explain v6 errors?

Loads only trusted local pickle files. No weights, CUDA, model scoring, or
changes to the peer session's artifacts. The output is a new diagnostic JSON.
"""
import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def describe(lengths):
    a = np.asarray(lengths, dtype=np.int32)
    return dict(n=len(a), over_128=int((a > 128).sum()),
                over_192=int((a > 192).sum()),
                maximum=int(a.max()) if len(a) else None,
                median=float(np.median(a)) if len(a) else None)


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path.cwd())
    ap.add_argument('--peer', type=Path, default=Path('work/ce'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    model = args.root / 'artifacts/improvement_v3/models/e5_base'
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    def lengths(rows):
        answer = []
        for i in range(0, len(rows), 256):
            chunk = rows[i:i + 256]
            encoded = tok([f'{r[0]} | {r[1] or "-"}' for r in chunk],
                          [f'{r[2]} | {r[3] or "-"}' for r in chunk],
                          truncation=False, padding=False)
            answer.extend(map(len, encoded['input_ids']))
        return answer
    audit = args.root / 'artifacts/improvement_v3/v6_band_error_audit_v1'
    report = {'complete': False, 'tokenizer': str(model), 'splits': {}}
    for split in ('dev', 'tune'):
        path = args.peer / f'{split}.pkl'
        with path.open('rb') as fh:
            rows = pickle.load(fh)
        lens = lengths([r[5:9] for r in rows])
        report['splits'][split] = dict(all=describe(lens),
            positive=describe([n for n, r in zip(lens, rows) if r[2]]),
            negative=describe([n for n, r in zip(lens, rows) if not r[2]]),
            input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    error_path = audit / 'development_errors.json'
    errors = json.loads(error_path.read_text())
    lens = lengths([(e['reference']['name'], e['reference']['address'],
                     e['target_record']['name'], e['target_record']['address'])
                    for e in errors])
    by_kind, by_stage = defaultdict(list), defaultdict(list)
    for e, n in zip(errors, lens):
        by_kind[e['kind']].append(n)
        by_stage[e['stage'] + '/' + e['kind']].append(n)
    report['development_errors'] = {k: describe(v) for k, v in by_kind.items()}
    report['development_stages'] = {k: describe(v) for k, v in by_stage.items()}
    report['error_input_sha256'] = hashlib.sha256(error_path.read_bytes()).hexdigest()
    report['complete'] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    run()
