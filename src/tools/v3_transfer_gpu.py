"""Bounded local E5, contrastive E5, and XLM-R pilots; never alters releases."""
import argparse
import gc
import json
import resource
import time
from pathlib import Path

import numpy as np
from v3_transfer_data import PILOT, SEED, load, write, digest

PINS = {'e5': ('intfloat/multilingual-e5-small', '614241f622f53c4eeff9890bdc4f31cfecc418b3'),
        'xlmr': ('FacebookAI/xlm-roberta-base', 'e73636d4f797dec63c3081bb6ed5c7b0bb3f2089')}


def records():
    return [json.loads(line) for line in (PILOT / 'data/records.jsonl').open()]


def text(rec, flip=False, name_only=False):
    if name_only:
        return 'name: ' + rec['name']
    fields = [('name', rec['name']), ('address', rec['address'])]
    if flip:
        fields.reverse()
    return ' ; '.join(f'{k}: {v}' for k, v in fields)


def progress(out, **kwargs):
    import torch
    value = dict(**kwargs, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                 gpu_peak_mib=torch.cuda.max_memory_allocated()/1024**2)
    write(out / 'progress.json', value)
    print(json.dumps(value), flush=True)


def setup(tag, resume=False):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError('No local CUDA device; no automatic cloud provisioning')
    out = PILOT / tag
    out.mkdir(exist_ok=resume)
    torch.cuda.reset_peak_memory_stats()
    return out


def pool(model, batch):
    import torch
    import torch.nn.functional as F
    hidden = model(**batch).last_hidden_state
    mask = batch['attention_mask'].unsqueeze(-1).bool()
    hidden = hidden.masked_fill(~mask, 0)
    return F.normalize(hidden.sum(1).float() / mask.sum(1), p=2, dim=-1)


def encoding(tokenizer, texts, limit=96):
    return {k: v.cuda() for k, v in tokenizer(texts, padding=True, truncation=True,
            max_length=limit, return_tensors='pt').items()}


def e5(contrastive=False):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    tag = 'contrastive_features' if contrastive else 'e5_features'
    out = setup(tag)
    start = time.monotonic()
    path = PILOT / 'models/e5'
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModel.from_pretrained(path, local_files_only=True, low_cpu_mem_usage=True).cuda()
    rows = records()
    history = []
    if contrastive:
        a = load('train')
        for name, param in model.named_parameters():
            param.requires_grad = any(name.startswith(f'encoder.layer.{i}.') for i in (10, 11))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)
        scaler = torch.amp.GradScaler('cuda')
        positives = np.flatnonzero(a['y'] == 1)
        by_owner = {}
        for ix in positives:
            by_owner.setdefault(int(a['query_ids'][ix]), []).append(int(ix))
        # Only one positive per known owner per batch: no known sibling records
        # are pushed apart as in-batch negatives. Unknown duplicates remain possible.
        rng = np.random.default_rng(SEED)
        keys = np.array(sorted(by_owner))
        for epoch in range(2):
            model.train()
            shuffled = rng.permutation(keys)
            loss_sum = steps = 0
            for j in range(0, len(shuffled), 16):
                owners = shuffled[j:j+16]
                if len(owners) < 2:
                    continue
                ix = [rng.choice(by_owner[int(q)]) for q in owners]
                texts = []
                for side in ('left', 'right'):
                    texts.extend('query: ' + text(rows[int(a[side][i])], flip=bool(rng.integers(2)),
                        name_only=bool(rng.random() < .2)) for i in ix)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    emb = pool(model, encoding(tokenizer, texts))
                    l, r = emb[:len(ix)], emb[len(ix):]
                    scores = l @ r.T / .07
                    target = torch.arange(len(ix), device='cuda')
                    loss = (F.cross_entropy(scores, target) + F.cross_entropy(scores.T, target)) / 2
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.item()
                steps += 1
                if steps % 100 == 0:
                    progress(out, stage='contrastive_train', epoch=epoch+1, owners=j+len(ix),
                        total_owners=len(keys), mean_loss=loss_sum/steps, seconds=time.monotonic()-start)
            history.append(dict(epoch=epoch+1, mean_loss=loss_sum/steps, positive_owners=len(keys)))
        torch.save({k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}, out / 'trainable_weights.pt')
        del a, optimizer, scaler
        gc.collect()
        torch.cuda.empty_cache()
    training_seconds = time.monotonic()-start
    model.eval()
    # Deduplicate exact text, not entity identity. Empty fields are handled in
    # downstream features; do not mistake empty-text embedding cosine for evidence.
    unique, mapping = [], {}
    indexes = np.empty((len(rows), 2), dtype=np.int32)
    for i, row in enumerate(rows):
        for f, key in enumerate(('name', 'address')):
            value = row[key].strip()
            if value not in mapping:
                mapping[value] = len(unique)
                unique.append(value)
            indexes[i, f] = mapping[value]
    embeddings = np.lib.format.open_memmap(out / 'embeddings.npy', mode='w+', dtype=np.float16,
        shape=(len(unique), model.config.hidden_size))
    encode_start = time.monotonic()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        for j in range(0, len(unique), 64):
            texts = ['query: ' + s for s in unique[j:j+64]]
            embeddings[j:j+len(texts)] = pool(model, encoding(tokenizer, texts)).cpu().numpy()
            if j % 12800 == 0:
                embeddings.flush()
                if hasattr(embeddings._mmap, 'madvise'):
                    import mmap
                    embeddings._mmap.madvise(mmap.MADV_DONTNEED)
                progress(out, stage='encode', texts=j+len(texts), total_texts=len(unique),
                         seconds=time.monotonic()-encode_start)
    embeddings.flush()
    np.save(out / 'record_text_indexes.npy', indexes)
    for split in ('train', 'tune', 'dev'):
        a = load(split)
        features = np.empty((len(a['y']), 4), dtype=np.float32)
        for j in range(0, len(a['y']), 2048):
            l, r = a['left'][j:j+2048], a['right'][j:j+2048]
            name = np.einsum('ij,ij->i', embeddings[indexes[l, 0]].astype(np.float32), embeddings[indexes[r, 0]].astype(np.float32))
            addr = np.einsum('ij,ij->i', embeddings[indexes[l, 1]].astype(np.float32), embeddings[indexes[r, 1]].astype(np.float32))
            missing = a['missing_address'][j:j+2048]
            addr[missing] = 0
            features[j:j+len(l)] = np.column_stack([name, addr, name*addr, name-addr])
        np.save(out / f'{split}_features.npy', features)
        del a
    write(out / 'manifest.json', dict(complete=True, method=tag, model=PINS['e5'], license='MIT',
        data_manifest_sha256=digest(PILOT / 'data/manifest.json'),
        unique_texts=len(unique), fields=['name_cosine', 'address_cosine_empty_zero', 'product', 'difference'],
        history=history, training_seconds=training_seconds, encoding_seconds=time.monotonic()-encode_start,
        total_seconds=time.monotonic()-start, prefix='query: on both sides', max_length=96,
        note='Bounded supervised owner-contrastive adaptation with safe field order/address dropout; not a Sudowoodo reproduction',
        gpu_peak_mib=torch.cuda.max_memory_allocated()/1024**2))


def xlmr(resume=False):
    import torch
    from v3_transfer_checkpoint import atomic_save, read_checkpoint
    from transformers import AutoTokenizer, XLMRobertaForSequenceClassification
    out = setup('xlmr', resume=resume)
    if (out / 'manifest.json').exists():
        raise RuntimeError('Completed XLM-R pilot exists; leave it unchanged')
    if resume and (out / 'progress.json').exists():
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        (out / f'progress_before_recovery_{stamp}.json').write_bytes((out / 'progress.json').read_bytes())
    start = time.monotonic()
    path = PILOT / 'models/xlmr'
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = XLMRobertaForSequenceClassification.from_pretrained(path, local_files_only=True,
        num_labels=2, low_cpu_mem_usage=True)
    # Explicit new head initialization for base-encoder checkpoints.
    from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaClassificationHead
    model.classifier = XLMRobertaClassificationHead(model.config)
    model.classifier.apply(model._init_weights)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith('classifier.') or any(name.startswith(f'roberta.encoder.layer.{i}.') for i in (10, 11))
    model.cuda()
    rows = records()
    a = load('train')
    rng = np.random.default_rng(SEED)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-5)
    scaler = torch.amp.GradScaler('cuda')

    def batch(a, ix):
        left = [text(rows[int(i)]) for i in a['left'][ix]]
        right = [text(rows[int(i)]) for i in a['right'][ix]]
        return {k: v.cuda() for k, v in tokenizer(left, right, padding=True, truncation=True,
                    max_length=128, return_tensors='pt').items()}

    binding = dict(data=digest(PILOT / 'data/manifest.json'), model=list(PINS['xlmr']),
                   source=digest(Path(__file__)), seed=SEED, batch=8, accumulate=4)
    checkpoint = out / 'training_checkpoint.pt'
    history, initial_epoch, next_pair, saved_order = [], 0, 0, None
    previous_seconds = 0.
    saved_loss, saved_steps = 0., 0
    if resume and checkpoint.exists():
        state = read_checkpoint(checkpoint, binding)
        trainable_names = {k for k, v in model.named_parameters() if v.requires_grad}
        if set(state['weights']) != trainable_names:
            raise ValueError('Checkpoint trainable parameters do not match')
        model.load_state_dict(state['weights'], strict=False)
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        rng.bit_generator.state = state['numpy_rng']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        history, initial_epoch = state['history'], state['epoch']
        next_pair, saved_order = state['next_pair'], state['order']
        saved_loss, saved_steps = state['loss_sum'], state['steps']
        previous_seconds = state['elapsed_seconds']
        del state
        progress(out, stage='resumed_training', epoch=initial_epoch+1, pairs=next_pair)
    elif resume:
        progress(out, stage='restarting_uncheckpointed_training', saved_training_weights=False)

    def save_training(epoch, next_pair, order, loss_sum, steps):
        atomic_save(checkpoint, dict(binding=binding,
            weights={k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad},
            optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(), history=history,
            epoch=epoch, next_pair=next_pair, order=order, loss_sum=loss_sum, steps=steps,
            elapsed_seconds=previous_seconds+time.monotonic()-start))

    for epoch in range(initial_epoch, 2):
        model.train()
        continuing = epoch == initial_epoch and saved_order is not None
        order = saved_order if continuing else rng.permutation(len(a['y']))
        loss_sum, steps = (saved_loss, saved_steps) if continuing else (0., 0)
        optimizer.zero_grad(set_to_none=True)
        for j in range(next_pair if continuing else 0, len(order), 8):
            ix = order[j:j+8]
            with torch.autocast('cuda', dtype=torch.float16):
                loss = model(**batch(a, ix), labels=torch.tensor(a['y'][ix], dtype=torch.long, device='cuda')).loss
            scaler.scale(loss/4).backward()
            if (steps+1) % 4 == 0 or j+8 >= len(order):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += loss.item()
            steps += 1
            if steps % 400 == 0:
                # 400 batches is an optimizer boundary: no accumulated gradients
                # are omitted from this restartable state. Keep one bounded file.
                save_training(epoch, j+len(ix), order, loss_sum, steps)
                progress(out, stage='train', epoch=epoch+1, pairs=j+len(ix), total_pairs=len(order),
                    mean_loss=loss_sum/steps, seconds=time.monotonic()-start,
                    checkpoint_pairs=j+len(ix))
        history.append(dict(epoch=epoch+1, mean_loss=loss_sum/steps))
        save_training(epoch+1, 0, None, 0., 0)
    atomic_save(out / 'trainable_weights.pt', {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad})
    del a, optimizer, scaler
    gc.collect()
    training_seconds = time.monotonic()-start
    model.eval()
    for split in ('tune', 'dev'):
        a = load(split)
        p = np.empty(len(a['y']), dtype=np.float32)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            for j in range(0, len(p), 24):
                ix = np.arange(j, min(j+24, len(p)))
                p[ix] = model(**batch(a, ix)).logits.softmax(-1)[:, 1].float().cpu().numpy()
                if j % 12000 == 0:
                    progress(out, stage='predict', split=split, pairs=j+len(ix), total_pairs=len(p), seconds=time.monotonic()-start)
        np.save(out / f'{split}_probabilities.npy', p)
        del a
    write(out / 'manifest.json', dict(complete=True, model=PINS['xlmr'], license='MIT',
        data_manifest_sha256=digest(PILOT / 'data/manifest.json'), history=history,
        total_seconds=time.monotonic()-start, training_seconds=training_seconds,
        resumed_previous_seconds=previous_seconds, checkpoint_every_batches=400,
        max_length=128, trainable_layers=[10, 11],
        gpu_peak_mib=torch.cuda.max_memory_allocated()/1024**2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['e5', 'contrastive', 'xlmr'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.resume and args.mode != 'xlmr':
        parser.error('--resume currently applies only to XLM-R')
    xlmr(resume=args.resume) if args.mode == 'xlmr' else e5(args.mode == 'contrastive')
