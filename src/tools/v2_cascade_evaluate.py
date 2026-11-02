"""Evaluate a frozen neural cascade on larger cohorts without further calibration."""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
import numpy as np
from v2_cascade_train import entity_scores, fuse, score


def subset(a, start, stop):
    mask = (a['query_ids'] >= start) & (a['query_ids'] < stop)
    result = {k: (v[start:stop] if k in ('reference_ids','country','true_counts') else v[mask])
              for k,v in a.items()}
    result['query_ids'] -= start
    return result, mask


def compare(a, neural, choice):
    route = (a['tree'] >= choice['lower']) & (a['tree'] <= choice['upper'])
    mixture = fuse(a['tree'], neural, route, choice['weight'])
    base = entity_scores(a, a['tree'], .74)
    mixed = entity_scores(a, mixture, choice['threshold'])
    delta = mixed-base
    rng = np.random.default_rng(20260926)
    bootstrap = np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(2000)])
    result = {'references': len(delta), 'baseline': score(a,a['tree'],.74),
              'mixture': score(a,mixture,choice['threshold']), 'mean_delta': float(delta.mean()),
              'paired_owner_bootstrap_95': np.quantile(bootstrap,[.025,.975]).tolist(),
              'routed_pairs': int(route.sum()), 'routed_fraction': float(route.mean())}
    for method, mask in (('baseline',a['tree'] >= .74),('mixture',mixture >= choice['threshold'])):
        tp = int((mask & (a['y'] == 1)).sum())
        fp = int((mask & (a['y'] == 0)).sum())
        result[method].update(tp=tp,fp=fp,precision=tp/max(1,tp+fp),
                             recall=tp/max(1,int(a['true_counts'].sum())))
    return result


def run(data, checkpoint_run, output, batch_size=16):
    import torch
    from transformers import BertForSequenceClassification, XLMRobertaTokenizer
    from transformers.utils import logging
    logging.set_verbosity_error()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    data, checkpoint_run, out = map(Path,(data,checkpoint_run,output))
    out.mkdir(parents=True,exist_ok=False)
    prior = json.loads((checkpoint_run/'report.json').read_text())
    choice = prior['selected_on_tune']
    t = XLMRobertaTokenizer.from_pretrained(checkpoint_run/'model')
    m = BertForSequenceClassification.from_pretrained(checkpoint_run/'model',low_cpu_mem_usage=True).cuda().eval()
    h = hashlib.sha256()
    with (checkpoint_run/'model/model.safetensors').open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    report = {'frozen_policy':choice,'checkpoint_sha256':h.hexdigest(),
              'policy_selected_on_first_1000_tuning_owners_only':True,
              'no_retraining_or_retuning':True,'final_audit_used':False,'splits':{}}
    for split in ('tune','dev'):
        a = dict(np.load(data/f'{split}.npz',allow_pickle=False))
        route = (a['tree'] >= choice['lower']) & (a['tree'] <= choice['upper'])
        ids = np.flatnonzero(route)
        texts = []
        with (data/f'{split}.jsonl').open() as f:
            for i,line in enumerate(f):
                if route[i]:
                    texts.append(json.loads(line))
        assert len(texts) == len(ids)
        p = np.full(len(route),np.nan,dtype=np.float32)
        start = time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            for i in range(0,len(ids),batch_size):
                pairs = texts[i:i+batch_size]
                x = t([r[0] for r in pairs],[r[1] for r in pairs],padding=True,
                      truncation='longest_first',max_length=128,return_tensors='pt')
                p[ids[i:i+batch_size]] = m(**{k:v.cuda() for k,v in x.items()}).logits.softmax(-1)[:,1].float().cpu().numpy()
        torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        np.save(out/f'{split}_neural.npy',p)
        result = {'full':compare(a,p,choice),'inference_seconds':seconds,
                  'pairs_per_second':len(ids)/seconds}
        for label,first,last in (('original_pilot',0,1000),('additional_2000',1000,len(a['true_counts']))):
            small,mask = subset(a,first,last)
            result[label] = compare(small,p[mask],choice)
        report['splits'][split] = result
        print(json.dumps({'split':split,**result},indent=2),flush=True)
    report['max_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    report['peak_gpu_allocated_mib'] = torch.cuda.max_memory_allocated()/1024**2
    (out/'report.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--checkpoint-run',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--batch-size',type=int,default=16)
    run(**vars(p.parse_args()))
