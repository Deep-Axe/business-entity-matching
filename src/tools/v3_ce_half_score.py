"""Memory-bounded replay of the peer scorer's FP16-weight, batch-512 recipe.

Separate from the frozen FP32-weight/autocast release. Never mix precisions
without independent calibration and replay. No downloads or source mutations.
"""
import argparse
from pathlib import Path
import pickle
import time

import numpy as np

from v3_ce_score_partition import guard


def run(model_dir,weights,input,output):
    guard()
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    torch.set_num_threads(1)
    if torch.cuda.mem_get_info()[0]<3.5*1024**3:raise RuntimeError('Insufficient free GPU memory')
    tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(model_dir,num_labels=1,local_files_only=True)
    state=torch.load(weights,map_location='cpu',weights_only=True);model.load_state_dict(state);del state
    model=model.cuda().half().eval();rows=pickle.load(Path(input).open('rb'));enc=[]
    def seg(n,a):return f"{n} | {a if a else '-'}"
    for k in range(0,len(rows),20000):
        guard();part=rows[k:k+20000]
        enc.extend(tok([seg(r[5],r[6]) for r in part],[seg(r[7],r[8]) for r in part],truncation=True,max_length=128)['input_ids'])
    del rows
    lengths=np.asarray([len(x) for x in enc]);order=np.argsort(lengths);scores=np.zeros(len(enc),np.float32);checked=0.
    with torch.no_grad():
        for k in range(0,len(order),512):
            if time.monotonic()-checked>20:guard();checked=time.monotonic()
            batch=order[k:k+512];length=int(lengths[batch].max())
            x=np.full((len(batch),length),tok.pad_token_id,np.int64);mask=np.zeros_like(x)
            for j,i in enumerate(batch):x[j,:lengths[i]]=enc[i];mask[j,:lengths[i]]=1
            lo=model(input_ids=torch.from_numpy(x).cuda(),attention_mask=torch.from_numpy(mask).cuda()).logits.float().squeeze(-1)
            scores[batch]=torch.sigmoid(lo).cpu().numpy()
    if not np.isfinite(scores).all():raise ValueError('Invalid FP16-weight predictions')
    path=Path(output);temp=path.with_suffix('.tmp.npy');np.save(temp,scores);temp.replace(path)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir','weights','input','output'):p.add_argument('--'+name,required=True)
    run(**vars(p.parse_args()))
