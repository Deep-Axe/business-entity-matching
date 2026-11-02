"""Bounded train-owner-only pair sampling and untouched full-cohort evaluation caches."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from ber.context_features import model_matrix
from ber.features import FEATURE_NAMES
from ber.shards import shard_paths, file_hash
from ber.store import connect, lookup


def serialize(row):
    return f"name: {row['name']} ; address: {row['address']} ; country: {row['country']}"


def run(output, train_owners=6000, eval_owners=1000):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    db = connect('artifacts/train_store_v2')
    db.execute('PRAGMA cache_size=-8192')
    model_path = Path('artifacts/deadline_model_50k_v1/model.cbm')
    model = CatBoostClassifier(thread_count=2)
    model.load_model(str(model_path))
    rng = np.random.default_rng(20260925)
    splits = {'train': 'deadline_features_train_50k_v1',
              'tune': 'deadline_features_full_tune_v1',
              'dev': 'deadline_features_full_dev_v1'}
    manifest = {'seed': 20260925, 'model_sha256': file_hash(model_path),
                'audit_used': False, 'external_data_used': False, 'splits': {}}
    owner_sets = {}
    for split, directory in splits.items():
        started = time.perf_counter()
        remaining = train_owners if split == 'train' else eval_owners
        owner_sets[split] = set()
        if remaining == 0:
            manifest['splits'][split] = {'references': 0, 'pairs': 0, 'skipped': True}
            continue
        texts, refs, countries, allq, ally, allp, alltrue, target_ids = [], [], [], [], [], [], [], []
        offset = 0
        source = Path('artifacts') / directory
        for path in shard_paths(source):
            if not remaining:
                break
            with np.load(path, allow_pickle=False) as z:
                a = {k: z[k] for k in z.files}
            n = min(remaining, len(a['reference_ids']))
            ids = [str(i) for i in a['reference_ids'][:n]]
            owners = lookup(db, 'anchors', 'entity_id', ids)
            expected_fold = {'train': 'train', 'tune': 'tune', 'dev': 'eval'}[split]
            assert all(owners[x]['fold'] == expected_fold for x in ids)
            p = model.predict_proba(model_matrix(a, model.feature_names_), thread_count=2)[:, 1]
            choose = a['query_ids'] < n
            if split == 'train':
                # Positives + realistic lexical hard negatives, not in-sample error mining.
                choose[:] = False
                namecol = FEATURE_NAMES.index('ascii_name_ratio')
                addrcol = FEATURE_NAMES.index('address_ratio')
                hardness = a['X'][:, namecol] + a['X'][:, addrcol]
                for q in range(n):
                    pos = np.flatnonzero((a['query_ids'] == q) & (a['y'] == 1))
                    neg = np.flatnonzero((a['query_ids'] == q) & (a['y'] == 0))
                    if len(pos):
                        choose[rng.choice(pos, min(2, len(pos)), replace=False)] = True
                    if len(neg):
                        choose[neg[np.argsort(hardness[neg])[-2:]]] = True
                        choose[rng.choice(neg)] = True
            idx = np.flatnonzero(choose)
            selected_targets = [str(i) for i in a['target_ids'][idx]]
            targets = lookup(db, 'records', 'entity_id', selected_targets)
            if split == 'train':
                assert all(targets[x]['fold'] == 'train' for x in selected_targets)
            for i, tid in zip(idx, selected_targets):
                oid = str(a['reference_ids'][a['query_ids'][i]])
                texts.append([serialize(owners[oid]), serialize(targets[tid])])
            allq.append(a['query_ids'][idx] + offset)
            ally.append(a['y'][idx])
            allp.append(p[idx].astype(np.float32))
            alltrue.append(a['true_counts'][:n])
            refs.extend(ids)
            target_ids.extend(selected_targets)
            countries.extend(owners[x]['country'] for x in ids)
            owner_sets[split].update(ids)
            remaining -= n
            offset += n
        assert not remaining
        with (out / f'{split}.jsonl').open('x') as f:
            for pair in texts:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        np.savez_compressed(out / f'{split}.npz', query_ids=np.concatenate(allq),
                            y=np.concatenate(ally), tree=np.concatenate(allp),
                            true_counts=np.concatenate(alltrue),
                            reference_ids=np.array(refs), country=np.array(countries),
                            target_ids=np.array(target_ids))
        manifest['splits'][split] = {'references': len(refs), 'pairs': len(texts),
                                    'positives': int(np.concatenate(ally).sum()),
                                    'source': str(source), 'seconds': time.perf_counter()-started}
        print(split, manifest['splits'][split], flush=True)
    assert not owner_sets['train'] & owner_sets['tune']
    assert not owner_sets['train'] & owner_sets['dev']
    assert not owner_sets['tune'] & owner_sets['dev']
    manifest['owner_disjoint'] = True
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--train-owners', type=int, default=6000)
    p.add_argument('--eval-owners', type=int, default=1000)
    run(**vars(p.parse_args()))
