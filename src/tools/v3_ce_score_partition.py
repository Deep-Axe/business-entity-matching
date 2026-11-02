"""One bounded raw-text GPU scoring task in the existing isolated torch venv.

No training, downloads, nonlocal data, or persistent GPU allocations after exit.
Writes atomically; refuses to compete with another CUDA job.
"""
import argparse
import os
from pathlib import Path
import pickle
import subprocess
import time

import numpy as np


def others():
    r=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True,timeout=20)
    return {int(x) for x in r.stdout.splitlines() if x.strip().isdigit()}-{os.getpid()}


def guard():
    if others():raise RuntimeError('Another GPU job is active; yielding without touching it')
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:') and int(line.split()[1])<3*1024**2:
            raise RuntimeError('Available RAM below 3GiB; stopping this scorer')


def run(model_dir,weights,input,output):
    guard()
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    torch.set_num_threads(1)
    free,_=torch.cuda.mem_get_info()
    if free<3.5*1024**3:raise RuntimeError('Insufficient free GPU memory')
    tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(model_dir,num_labels=1,local_files_only=True)
    state=torch.load(weights,map_location='cpu',weights_only=True);model.load_state_dict(state);del state
    model=model.cuda().eval();rows=pickle.load(Path(input).open('rb'));ids=[]
    def seg(n,a):return f"{n} | {a if a else '-'}"
    for k in range(0,len(rows),10000):
        guard();ch=rows[k:k+10000]
        ids.extend(tok([seg(r[5],r[6]) for r in ch],[seg(r[7],r[8]) for r in ch],truncation=True,max_length=128)['input_ids'])
    del rows
    lengths=np.asarray([len(x) for x in ids]);order=np.argsort(lengths);scores=np.zeros(len(ids),dtype=np.float32)
    checked=0.
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        for k in range(0,len(order),256):
            if time.monotonic()-checked>20:guard();checked=time.monotonic()
            batch=order[k:k+256];length=int(lengths[batch].max())
            x=np.full((len(batch),length),tok.pad_token_id,dtype=np.int64);mask=np.zeros_like(x)
            for j,i in enumerate(batch):x[j,:lengths[i]]=ids[i];mask[j,:lengths[i]]=1
            logits=model(input_ids=torch.from_numpy(x).cuda(),attention_mask=torch.from_numpy(mask).cuda()).logits.float().squeeze(-1)
            scores[batch]=torch.sigmoid(logits).cpu().numpy()
    if not np.isfinite(scores).all():raise ValueError('Non-finite GPU predictions')
    output=Path(output);tmp=output.with_suffix('.tmp.npy');np.save(tmp,scores);tmp.replace(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir','weights','input','output'):p.add_argument('--'+name,required=True)
    run(**vars(p.parse_args()))
