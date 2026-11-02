"""Read-only post-run diagnostics; does not select a new model on development."""
import argparse
import json
from pathlib import Path
import numpy as np
from v2_cascade_train import fuse


def run(data, run_dir):
    data, run_dir = Path(data), Path(run_dir)
    report = json.loads((run_dir/'report.json').read_text())
    selected = report['selected_on_tune']
    summary = {}
    for split in ('tune','dev'):
        a = dict(np.load(data/f'{split}.npz', allow_pickle=False))
        neural = np.load(run_dir/f'{split}_neural.npy')
        route = (a['tree'] >= selected['lower']) & (a['tree'] <= selected['upper'])
        p = fuse(a['tree'], neural, route, selected['weight'])
        masks = {'baseline': a['tree'] >= .74, 'mixture': p >= selected['threshold'],
                 'recalibrated_tree': a['tree'] >= report['tree_recalibrated_on_same_tune']['threshold']}
        summary[split] = {}
        for method, pred in masks.items():
            tp = int((pred & (a['y'] == 1)).sum())
            fp = int((pred & (a['y'] == 0)).sum())
            counts = np.bincount(a['query_ids'], weights=pred, minlength=len(a['true_counts']))
            singles = a['true_counts'] == 0
            summary[split][method] = {'tp': tp, 'fp': fp, 'fn_including_unretrieved': int(a['true_counts'].sum())-tp,
                'precision': tp/max(1,tp+fp), 'recall': tp/max(1,int(a['true_counts'].sum())),
                'singleton_false_merge_rate': float((counts[singles] > 0).mean()) if singles.any() else None}
        added = masks['mixture'] & ~masks['baseline']
        removed = masks['baseline'] & ~masks['mixture']
        summary[split]['changes_vs_baseline'] = {
            'added_true': int((added & (a['y'] == 1)).sum()),
            'added_false': int((added & (a['y'] == 0)).sum()),
            'removed_true': int((removed & (a['y'] == 1)).sum()),
            'removed_false': int((removed & (a['y'] == 0)).sum())}
    output = run_dir/'post_run_diagnostics.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--run-dir', required=True)
    run(**vars(parser.parse_args()))
