"""Bounded, resumable name-only specialist training/scoring in the torch venv.

One epoch on naturally empty-address training pairs. No new labels, downloads,
evaluation-driven early stopping, or writes to the parent encoder checkpoint.
The caller holds the shared GPU lease; this worker additionally checks live
CUDA competitors. GPU allocator cap is 65% of device memory.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import time

import numpy as np

from v3_ce_score_partition import guard

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    path=Path(path);temp=path.with_suffix('.pending.json');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def missing(value):
    return not value.strip() or value.strip().upper() in ('NULL','<NULL>')


def check():
    guard()
    if shutil.disk_usage(ROOT).free<6*1024**3:raise RuntimeError('Name specialist stopped at 6GiB disk reserve')


def load_model(model_dir,weights):
    check()
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.65)
    if torch.cuda.mem_get_info()[0]<3.5*1024**3:raise RuntimeError('Insufficient free GPU memory')
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(model_dir,num_labels=1,local_files_only=True)
    state=torch.load(weights,map_location='cpu',weights_only=True);model.load_state_dict(state);del state
    return tokenizer,model.cuda()


def encode(tokenizer,left,right):
    pieces=[];lengths=[]
    for start in range(0,len(left),10000):
        check()
        ids=tokenizer(left[start:start+10000],right[start:start+10000],truncation=True,max_length=96)['input_ids']
        pieces.append(np.concatenate([np.asarray(x,np.int32) for x in ids]));lengths.extend(map(len,ids))
    flat=np.concatenate(pieces) if pieces else np.empty(0,np.int32)
    length=np.asarray(lengths,np.int32);offset=np.r_[0,np.cumsum(length,dtype=np.int64)]
    return flat,length,offset


def batch_arrays(tokens,indices,pad):
    flat,length,offset=tokens;size=int(length[indices].max())
    x=np.full((len(indices),size),pad,np.int64);mask=np.zeros_like(x)
    for j,i in enumerate(indices):
        n=length[i];x[j,:n]=flat[offset[i]:offset[i+1]];mask[j,:n]=1
    return x,mask


def train(data,model_dir,weights,output):
    import torch
    data,weights,output=map(Path,(data,weights,output));output.mkdir(parents=True,exist_ok=True)
    receipt=json.loads((data/'completion.json').read_text())
    if not receipt['complete'] or not receipt['training_only'] or digest(data/'pairs.pkl')!=receipt['pairs_sha256']:
        raise ValueError('Unverified name-specialist training pool')
    recipe=dict(data_completion_sha256=digest(data/'completion.json'),initial_checkpoint_sha256=digest(weights),
        code_sha256=digest(__file__),epochs=1,batch_size=64,max_length=96,learning_rate=1e-5,seed=202609268,
        word_embeddings_frozen=True,serialization='Raw left name as first sequence, raw target name as second; no addresses or IDs',
        cpu_threads=1,gpu_allocator_fraction=.65)
    if (output/'identity.json').exists():
        if json.loads((output/'identity.json').read_text())!=recipe:raise ValueError('Specialist training identity changed')
    else:write(output/'identity.json',recipe)
    if (output/'completion.json').exists():
        done=json.loads((output/'completion.json').read_text())
        if done['model_sha256']!=digest(output/'model.pt'):raise ValueError('Completed specialist checkpoint changed')
        return
    check()
    if shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('Need 8GiB disk headroom before specialist training')
    torch.manual_seed(recipe['seed']);tokenizer,model=load_model(model_dir,weights)
    rows=pickle.load((data/'pairs.pkl').open('rb'))
    if len(rows)!=receipt['pairs'] or len(rows)!=259298:raise ValueError('Unexpected specialist data size')
    y=np.asarray([r[2] for r in rows],np.float32);tokens=encode(tokenizer,[r[4] for r in rows],[r[5] for r in rows]);del rows
    rng=np.random.default_rng(recipe['seed']);order=rng.permutation(len(y))
    chunks=[order[s:s+3200] for s in range(0,len(order),3200)]
    order=np.concatenate([c[np.argsort(tokens[1][c])] for c in chunks])
    batches=[order[s:s+64] for s in range(0,len(order),64)];rng.shuffle(batches)
    model.base_model.embeddings.word_embeddings.weight.requires_grad_(False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=.01)
    steps=len(batches);warm=max(1,int(.03*steps))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min(1,(s+1)/warm)*max(0.,(steps-s)/max(1,steps-warm)))
    scaler=torch.amp.GradScaler();loss_function=torch.nn.BCEWithLogitsLoss();step=0;loss_ema=0.
    resume=output/'resume.pt'
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=True)
        if state['identity_sha256']!=digest(output/'identity.json'):raise ValueError('Resume checkpoint identity differs')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler']);step=state['step'];loss_ema=state['loss_ema']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng']);del state
    started=time.monotonic();end=started+3600;checked=0.;model.train()
    def save():
        state=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),step=step,loss_ema=loss_ema,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),identity_sha256=digest(output/'identity.json'))
        tmp=output/'resume.tmp.pt';torch.save(state,tmp);tmp.replace(resume)
    try:
        for i in range(step,steps):
            if time.monotonic()-checked>10:
                check();checked=time.monotonic()
                if checked>=end:raise TimeoutError('One-hour specialist training window reached')
            b=batches[i];x,mask=batch_arrays(tokens,b,tokenizer.pad_token_id)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(input_ids=torch.from_numpy(x).cuda(),attention_mask=torch.from_numpy(mask).cuda()).logits.float().squeeze(-1)
                loss=loss_function(logits,torch.from_numpy(y[b]).cuda())
            if not torch.isfinite(loss):raise ValueError('Nonfinite specialist training loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer);scaler.update();scheduler.step();step=i+1;loss_ema=.98*loss_ema+.02*float(loss.item())
            if step%250==0:
                save();write(output/'progress.json',dict(stage='training',step=step,steps=steps,loss_ema=loss_ema,seconds=time.monotonic()-started))
                print('step',step,steps,'loss',loss_ema,flush=True)
        save();temp=output/'model.tmp.pt';torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},temp);temp.replace(output/'model.pt')
        write(output/'completion.json',dict(complete=True,training_pairs=len(y),steps=step,identity_sha256=digest(output/'identity.json'),
            model_sha256=digest(output/'model.pt'),seconds_this_invocation=time.monotonic()-started,
            ready_for_submission=False,evaluation_used_for_training=False))
    except Exception as error:
        # Preserve optimizer/RNG progress before yielding to another session or
        # a low-resource guard. A CUDA OOM is reported, never silently retried.
        available=next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        if not isinstance(error,torch.cuda.OutOfMemoryError) and available>=4*1024**2 and shutil.disk_usage(ROOT).free>=7*1024**3:
            try:save()
            except Exception as checkpoint_error:print('Could not save a newer checkpoint:',str(checkpoint_error),flush=True)
        raise


def score(model_dir,weights,input,output):
    import torch
    input,output=Path(input),Path(output);check();tokenizer,model=load_model(model_dir,weights);model=model.half().eval()
    rows=pickle.load(input.open('rb'));mask=np.asarray([missing(r[8]) for r in rows]);ix=np.flatnonzero(mask)
    tokens=encode(tokenizer,[rows[i][5] for i in ix],[rows[i][7] for i in ix]);del rows
    values=np.full(len(mask),np.nan,np.float32);order=np.argsort(tokens[1]);checked=0.
    with torch.no_grad():
        for start in range(0,len(order),256):
            if time.monotonic()-checked>10:check();checked=time.monotonic()
            b=order[start:start+256];x,m=batch_arrays(tokens,b,tokenizer.pad_token_id)
            logits=model(input_ids=torch.from_numpy(x).cuda(),attention_mask=torch.from_numpy(m).cuda()).logits.float().squeeze(-1)
            values[ix[b]]=torch.sigmoid(logits).cpu().numpy()
    if not np.isfinite(values[mask]).all() or not np.isnan(values[~mask]).all():raise ValueError('Specialist score route invalid')
    tmp=output.with_suffix('.tmp.npy');np.save(tmp,values);tmp.replace(output)
    write(output.with_suffix('.receipt.json'),dict(complete=True,input_sha256=digest(input),checkpoint_sha256=digest(weights),
        scores_sha256=digest(output),worker_sha256=digest(__file__),scored_pairs=int(mask.sum()),total_pairs=len(mask),
        precision='FP16 weights, batch256, no autocast',route='Naturally missing target address only; NaN elsewhere, not a negative prediction'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    tr=sub.add_parser('train');sc=sub.add_parser('score')
    for parser in (tr,sc):
        for name in ('model-dir','weights','output'):parser.add_argument('--'+name,required=True)
    tr.add_argument('--data',required=True);sc.add_argument('--input',required=True)
    args=vars(p.parse_args());command=args.pop('command');(train if command=='train' else score)(**args)
