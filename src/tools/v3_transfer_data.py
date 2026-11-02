"""Isolated transfer pilots: hash-bound data, owner metrics, no production writes."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/improvement_v2/cpu'
PILOT = ROOT / 'artifacts/improvement_v3/transfer_pilot_v1'
SEED = 20260925


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def load(split):
    return dict(np.load(PILOT / 'data' / f'{split}.npz', allow_pickle=False))


def owner_scores(a, p, threshold):
    accepted = np.asarray(p) >= threshold
    n = len(a['true_counts'])
    count = np.bincount(a['query_ids'][accepted], minlength=n)
    tp = np.bincount(a['query_ids'][accepted & (a['y'] == 1)], minlength=n)
    den = a['true_counts'] + 4 * count
    return np.divide(5 * tp, den, out=np.ones(n, dtype=float), where=den != 0)


def metric(a, p, threshold):
    s = owner_scores(a, p, threshold)
    accept = p >= threshold
    tp = int((accept & (a['y'] == 1)).sum())
    fp = int((accept & (a['y'] == 0)).sum())
    pred_by_owner = np.bincount(a['query_ids'][accept], minlength=len(s))
    masks = {str(c): a['countries'] == c for c in np.unique(a['countries'])}
    for field in ('cross_script', 'missing_address'):
        masks[field + '_retrieved_positive_owner'] = np.bincount(a['query_ids'], weights=a[field] & (a['y'] == 1), minlength=len(s)) > 0
    masks['singleton'] = a['true_counts'] == 0
    return dict(macro_f05=float(s.mean()), tp=tp, fp=fp,
                fn=int(a['true_counts'].sum()) - tp,
                singleton_false_merges=int((pred_by_owner[masks['singleton']] > 0).sum()),
                slices={k: dict(owners=int(m.sum()), macro_f05=float(s[m].mean()) if m.any() else None)
                        for k, m in masks.items()},
                retrieved_pair_slices={k: dict(candidates=int(m.sum()), positives=int((m & (a['y'] == 1)).sum()),
                    tp=int((m & accept & (a['y'] == 1)).sum()), fp=int((m & accept & (a['y'] == 0)).sum()))
                    for k, m in {'cross_script': a['cross_script'], 'missing_address': a['missing_address'],
                                 'source2': a['target_source'] == 2, 'source3': a['target_source'] == 3}.items()})


def tune(a, p):
    # Fixed grid, includes reject-all, deterministic ties toward conservatism.
    thresholds = np.r_[np.arange(0, 1.001, .01), np.nextafter(1., 2.)]
    return max(((float(owner_scores(a, p, t).mean()), float(t)) for t in thresholds))


def bootstrap(old, new, repetitions=3000):
    delta = np.asarray(new) - np.asarray(old)
    rng = np.random.default_rng(SEED)
    means = [float(delta[rng.integers(len(delta), size=len(delta))].mean()) for _ in range(repetitions)]
    return dict(delta=float(delta.mean()), ci95=np.quantile(means, [.025, .975]).tolist(),
                ci99=np.quantile(means, [.005, .995]).tolist(),
                improved_owners=int((delta > 0).sum()), worsened_owners=int((delta < 0).sum()),
                caveat='Exploratory reused development set; conditional on tuning selection. 99% interval is a conservative five-family screen, not correction for all adaptive work.')


def prepare(owners=10000):
    from ber.store import connect, lookup
    from catboost import CatBoostClassifier
    out = PILOT / 'data'
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(SEED)
    db = connect(ROOT / 'artifacts/train_store_v2')
    db.execute('PRAGMA cache_size=-2048')
    records, record_ids = [], {}
    seen_owners, manifest = {}, dict(complete=False, seed=SEED, splits={}, protocol={
        'train_sampling': 'Seeded 10k of 20k owners; all retrieved positives + 4 lexically hardest + 2 random remaining negatives per owner',
        'evaluation': 'All candidates from full-gallery retrieval, all 3000 tuning and 3000 development owners; full truth counts include retrieval misses',
        'lexicon': 'Existing train-fold-only supervised alias lexicon predates pilot and used more than 10k owners; this is a 10k classifier screen, NOT an end-to-end 10k-label or strict country-zero-shot experiment',
        'selection': 'Tune only; development exploratory; reserved audit untouched',
        'production': 'Frozen releases and queue tools are read-only',
    })
    models = {}
    for size in (100000, 200000):
        path = BASE / f'fit_gate0_{size // 1000}k_v1/enhanced.cbm'
        model = CatBoostClassifier()
        model.load_model(str(path))
        models[size] = model
        manifest[f'model_{size}_sha256'] = digest(path)

    def record(row, anchor=False):
        eid = str(row['entity_id'])
        if eid not in record_ids:
            record_ids[eid] = len(records)
            records.append(dict(id=eid, name=row['name'] or '', address=row['address'] or '',
                                country=row['country'], source=1 if anchor else int(row['source']),
                                fold=row['fold']))
        return record_ids[eid]

    def script(text):
        # Detect native Indic scripts; accents are not non-Latin script.
        return any('\u0900' <= c <= '\u0dff' for c in text)

    for split, suffix, fold in [('train', 'train_20k', 'train'), ('tune', 'full_tune', 'tune'), ('dev', 'full_dev', 'eval')]:
        start = time.monotonic()
        directory = BASE / f'prepared_gate0_{suffix}_v1'
        m = json.loads((directory / 'manifest.json').read_text())
        assert m['complete'] and m['source_config']['gate_terms'] == 0
        original = {k: np.load(directory / f'{k}.npy', mmap_mode='r')
                    for k in ('X', 'y', 'query_ids', 'reference_ids', 'countries', 'true_counts')}
        selected = np.sort(rng.choice(len(original['reference_ids']), owners, replace=False)) if split == 'train' else np.arange(len(original['reference_ids']))
        remap = np.full(len(original['reference_ids']), -1, dtype=np.int32)
        remap[selected] = np.arange(len(selected))
        refs = original['reference_ids'][selected]
        anchors = lookup(db, 'anchors', 'entity_id', refs.tolist())
        assert all(anchors[str(r)]['fold'] == fold for r in refs)
        seen_owners[split] = set(map(str, refs))
        assert all(not seen_owners[split] & old for s, old in seen_owners.items() if s != split)
        left_owner = np.array([record(anchors[str(r)], True) for r in refs], dtype=np.int32)
        picked, targets = [], []
        for shard in m['shards']:
            if not np.any(remap[shard['query_start']:shard['query_end']] >= 0):
                continue
            path = Path(m['source']) / 'shards' / shard['shard'] / 'features.npz'
            if digest(path) != shard['features_sha256']:
                raise ValueError('Changed feature cache')
            with np.load(path, allow_pickle=False) as z:
                q = z['query_ids'] + shard['query_start']
                assert np.array_equal(original['query_ids'][shard['pair_start']:shard['pair_end']], q)
                assert np.array_equal(original['y'][shard['pair_start']:shard['pair_end']], z['y'])
                if split == 'train':
                    chosen = []
                    for qi in np.unique(q):
                        if remap[qi] < 0:
                            continue
                        ix = np.flatnonzero(q == qi)
                        positives = ix[z['y'][ix] == 1]
                        negatives = ix[z['y'][ix] == 0]
                        strength = z['X'][negatives, 9] + z['X'][negatives, 12]
                        hard = negatives[np.argsort(-strength, kind='stable')[:4]]
                        rest = np.setdiff1d(negatives, hard)
                        random = rng.choice(rest, min(2, len(rest)), replace=False)
                        chosen.extend(np.r_[positives, hard, random].tolist())
                    ix = np.sort(chosen).astype(np.int64)
                else:
                    ix = np.arange(len(q))
                ids = z['target_ids'][ix].astype(str)
                rows = lookup(db, 'records', 'entity_id', np.unique(ids).tolist())
                if split == 'train':
                    assert all(row['fold'] == 'train' for row in rows.values())
                targets.extend(record(rows[eid]) for eid in ids)
                picked.extend((ix + shard['pair_start']).tolist())
        picked = np.array(picked, dtype=np.int64)
        q = remap[original['query_ids'][picked]]
        left = left_owner[q]
        right = np.asarray(targets, dtype=np.int32)
        a = dict(X=np.asarray(original['X'][picked]), y=np.asarray(original['y'][picked]),
                 query_ids=q, true_counts=original['true_counts'][selected], reference_ids=refs,
                 countries=original['countries'][selected], left=left, right=right, original_pair_index=picked,
                 target_source=np.asarray([records[i]['source'] for i in right], dtype=np.int8),
                 cross_script=np.asarray([script(records[l]['name']) != script(records[r]['name']) for l, r in zip(left, right)]),
                 missing_address=np.asarray([not records[l]['address'].strip() or not records[r]['address'].strip() for l, r in zip(left, right)]))
        if split != 'train':
            assert len(picked) == m['pairs']
            for size, model in models.items():
                probability = model.predict_proba(a['X'], thread_count=1)[:, 1]
                cached = np.load(BASE / f'fit_gate0_{size // 1000}k_v1' / f'enhanced_{"full_tune" if split == "tune" else "dev"}_probabilities.npy')
                if not np.allclose(probability, cached[picked], atol=1e-12):
                    raise ValueError('Production prediction alignment mismatch')
                a[f'baseline_{size}'] = probability
        np.savez_compressed(out / f'{split}.npz', **a)
        manifest['splits'][split] = dict(owners=len(refs), pairs=len(q), positives=int(a['y'].sum()),
            full_truth_pairs=int(a['true_counts'].sum()), matrix_manifest_sha256=digest(directory / 'manifest.json'),
            data_sha256=digest(out / f'{split}.npz'), seconds=time.monotonic() - start,
            countries={str(c): int((a['countries'] == c).sum()) for c in np.unique(a['countries'])})
        print(json.dumps(manifest['splits'][split]), flush=True)
        write(out / 'progress.json', manifest)
        del a, original
    with (out / 'records.jsonl').open('w') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    manifest.update(complete=True, records=len(records), record_sha256=digest(out / 'records.jsonl'),
                    feature_names=m['names'], runner_sha256=digest(__file__))
    write(out / 'manifest.json', manifest)
    db.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--owners', type=int, default=10000)
    args = p.parse_args()
    prepare(args.owners)
