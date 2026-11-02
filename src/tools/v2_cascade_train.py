"""Small multilingual GPU cascade pilot, with tuning-only fusion selection."""
import argparse
import importlib.metadata
import json
import resource
import time
from pathlib import Path

import numpy as np

MODEL = 'microsoft/Multilingual-MiniLM-L12-H384'
REVISION = '6e8c1ec6b4ec4e3fc6eb7d2cd834fcd582b61daf'


def entity_scores(a, probabilities, threshold):
    pred = probabilities >= threshold
    q = a['query_ids']
    n = len(a['true_counts'])
    count = np.bincount(q, weights=pred, minlength=n)
    tp = np.bincount(q, weights=pred & (a['y'] == 1), minlength=n)
    true = a['true_counts']
    scores = np.divide(5*tp, true+4*count, out=np.zeros(n), where=true+4*count != 0)
    scores[(true == 0) & (count == 0)] = 1
    return scores


def score(a, p, threshold):
    s = entity_scores(a, p, threshold)
    return {'macro_f05': float(s.mean()),
            'countries': {c: float(s[a['country'] == c].mean()) for c in np.unique(a['country'])},
            'predicted_pairs': int((p >= threshold).sum())}


def fuse(tree, neural, routed, weight):
    result = tree.copy()
    result[routed] = (1-weight)*tree[routed] + weight*neural[routed]
    return result


def run(data, output, epochs=3, batch_size=8, max_length=128):
    import torch
    from transformers import BertForSequenceClassification, XLMRobertaTokenizer
    from huggingface_hub import snapshot_download
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260925)
    np.random.seed(20260925)
    if not torch.cuda.is_available():
        raise RuntimeError('Pilot requires the agreed local GPU; no CPU fallback')
    out, data = Path(output), Path(data)
    out.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    downloaded = snapshot_download(MODEL, revision=REVISION,
        allow_patterns=['config.json', 'pytorch_model.bin', 'sentencepiece.bpe.model',
                        'special_tokens_map.json', 'tokenizer_config.json', 'README.md'],
        local_dir=out.parent / 'pretrained_minilm', max_workers=2)
    tokenizer = XLMRobertaTokenizer.from_pretrained(downloaded)
    model = BertForSequenceClassification.from_pretrained(downloaded, num_labels=2, low_cpu_mem_usage=True)
    # This is a base encoder checkpoint, not a business-identity classifier.
    # Explicitly materialize a new head (also avoids missing meta tensors when
    # loading old encoder checkpoints with low_cpu_mem_usage).
    model.classifier = torch.nn.Linear(model.config.hidden_size, 2)
    model._init_weights(model.classifier)
    missing = [name for name, p in model.named_parameters() if p.is_meta]
    if missing:
        raise RuntimeError(f'Unmaterialized pretrained parameters: {missing}')
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (name.startswith('classifier.') or name.startswith('bert.pooler.')
            or any(name.startswith(f'bert.encoder.layer.{i}.') for i in (8, 9, 10, 11)))
    parameter_count = sum(p.numel() for p in model.parameters())
    assert parameter_count < 8_000_000_000
    model.cuda()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
    scaler = torch.amp.GradScaler('cuda')
    arrays = {split: dict(np.load(data/f'{split}.npz', allow_pickle=False)) for split in ('train','tune','dev')}
    texts = {split: [json.loads(x) for x in (data/f'{split}.jsonl').read_text().splitlines()]
             for split in arrays}
    assert all(len(texts[s]) == len(arrays[s]['y']) for s in arrays)

    def tokenize(split, indexes):
        pairs = [texts[split][int(i)] for i in indexes]
        enc = tokenizer([p[0] for p in pairs], [p[1] for p in pairs],
            truncation='longest_first', padding=True, max_length=max_length, return_tensors='pt')
        # MiniLM uses XLM-R vocabulary with BERT architecture. XLM-R has no segment IDs.
        return {k: v.cuda() for k, v in enc.items()}

    def predict(split, indexes):
        model.eval()
        pred = np.full(len(arrays[split]['y']), np.nan, dtype=np.float32)
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            for j in range(0, len(indexes), batch_size*2):
                ids = indexes[j:j+batch_size*2]
                pred[ids] = model(**tokenize(split, ids)).logits.softmax(-1)[:,1].float().cpu().numpy()
        torch.cuda.synchronize()
        sec = time.perf_counter()-start
        return pred, {'pairs': len(indexes), 'seconds': sec, 'pairs_per_second': len(indexes)/sec}

    start = time.perf_counter()
    history = []
    rng = np.random.default_rng(20260925)
    for epoch in range(epochs):
        model.train()
        ids = rng.permutation(len(texts['train']))
        running_loss, batches = 0., 0
        optimizer.zero_grad(set_to_none=True)
        for step, j in enumerate(range(0, len(ids), batch_size)):
            ix = ids[j:j+batch_size]
            labels = torch.tensor(arrays['train']['y'][ix], device='cuda', dtype=torch.long)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = model(**tokenize('train', ix), labels=labels).loss
            scaler.scale(loss/4).backward()
            running_loss += loss.item()
            batches += 1
            if (step+1) % 4 == 0 or j+batch_size >= len(ids):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if step % 200 == 0:
                progress = {'epoch': epoch+1, 'step': step, 'loss': running_loss/batches,
                    'elapsed_seconds': time.perf_counter()-start,
                    'rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                    'peak_gpu_mib': torch.cuda.max_memory_allocated()/1024**2}
                (out / 'progress.json').write_text(json.dumps(progress, indent=2))
                print(json.dumps(progress), flush=True)
        history.append({'epoch': epoch+1, 'mean_loss': running_loss/batches})
    training_seconds = time.perf_counter()-start
    model.save_pretrained(out / 'model')
    tokenizer.save_pretrained(out / 'model')
    perf, neural, routed = {}, {}, {}
    for split in ('tune','dev'):
        tree = arrays[split]['tree']
        routed[split] = (tree >= .01) & (tree <= .99)
        # First 100 owners get all candidates scored for a genuine neural-only diagnostic.
        evaluate = routed[split] | (arrays[split]['query_ids'] < 100)
        neural[split], perf[split] = predict(split, np.flatnonzero(evaluate))
        np.save(out / f'{split}_neural.npy', neural[split])
    rows = []
    for lower, upper in ((.01, .99), (.1, .9)):
        route = (arrays['tune']['tree'] >= lower) & (arrays['tune']['tree'] <= upper)
        for weight in (0., .1, .25, .5, .75, 1.):
            mixed = fuse(arrays['tune']['tree'], neural['tune'], route, weight)
            for threshold in np.unique(np.r_[np.arange(.4, .961, .02), .74]):
                rows.append({'weight': weight, 'threshold': float(threshold), 'lower': lower, 'upper': upper,
                             'macro_f05': float(entity_scores(arrays['tune'], mixed, threshold).mean())})
    best = max(rows, key=lambda r: (r['macro_f05'], -r['weight']))
    tree_best = max((r for r in rows if r['weight'] == 0), key=lambda r: r['macro_f05'])
    selected_routes = {s: (arrays[s]['tree'] >= best['lower']) & (arrays[s]['tree'] <= best['upper']) for s in ('tune','dev')}
    selected = {s: score(arrays[s], fuse(arrays[s]['tree'], neural[s], selected_routes[s], best['weight']), best['threshold'])
                for s in ('tune','dev')}
    dev_delta = entity_scores(arrays['dev'], fuse(arrays['dev']['tree'], neural['dev'],
                 selected_routes['dev'], best['weight']), best['threshold']) - entity_scores(
                 arrays['dev'], arrays['dev']['tree'], tree_best['threshold'])
    boot_rng = np.random.default_rng(20260926)
    bootstrap = np.array([dev_delta[boot_rng.integers(0, len(dev_delta), len(dev_delta))].mean()
                          for _ in range(2000)])
    # Neural-only threshold is tuned on first100 tuning owners, not dev.
    def sub(a):
        mask = a['query_ids'] < 100
        return {k: (v[:100] if k in ('country','true_counts','reference_ids') else v[mask]) for k,v in a.items()}
    ntune = sub(arrays['tune'])
    nth = max(np.arange(.1,.991,.01), key=lambda t: entity_scores(ntune,
              neural['tune'][arrays['tune']['query_ids']<100], t).mean())
    neural_only = {s: score(sub(arrays[s]), neural[s][arrays[s]['query_ids']<100], float(nth))
                   for s in ('tune','dev')}
    report = {'model': MODEL, 'revision': REVISION, 'license': 'MIT',
        'license_source': f'https://huggingface.co/{MODEL}/blob/{REVISION}/README.md',
        'parameters': parameter_count, 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'dependencies': {name: importlib.metadata.version(name) for name in
                         ('torch','transformers','accelerate','numpy','sentencepiece')},
        'frozen_embeddings_and_first_8_layers': True, 'training_epochs': epochs,
        'training_pairs': len(texts['train']), 'max_length': max_length,
        'batch_size': batch_size, 'gradient_accumulation_steps': 4, 'training_seconds': training_seconds,
        'training_history': history, 'baseline_threshold': .74,
        'baseline': {s: score(arrays[s], arrays[s]['tree'], .74) for s in ('tune','dev')},
        'tree_recalibrated_on_same_tune': {'threshold': tree_best['threshold'],
            'scores': {s: score(arrays[s], arrays[s]['tree'], tree_best['threshold']) for s in ('tune','dev')}},
        'routing': {s: {'pairs': int(routed[s].sum()), 'fraction': float(routed[s].mean())} for s in routed},
        'routing_rule': 'fixed tree probability between .01 and .99 inclusive',
        'selected_on_tune': best, 'selected_mixture': selected, 'tuning_grid': rows,
        'development_delta_vs_recalibrated_tree': {'mean': float(dev_delta.mean()),
            'paired_owner_bootstrap_95': np.quantile(bootstrap, [.025,.975]).tolist()},
        'neural_only_100_owner_diagnostic': {'threshold': float(nth), 'scores': neural_only},
        'throughput': perf, 'peak_gpu_mib': torch.cuda.max_memory_allocated()/1024**2,
        'max_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'final_audit_used': False, 'external_business_data_used': False,
        'limitations': ['Tiny pilot; 1000 tuning and 1000 development owners.',
                       'Partial transformer fine-tuning, not a definitive neural model test.',
                       'No French labels; no estimate of France hidden-test accuracy.',
                       'Routed scorer cannot recover absent candidates.']}
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='tuning_grid'}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-length', type=int, default=128)
    run(**vars(p.parse_args()))
