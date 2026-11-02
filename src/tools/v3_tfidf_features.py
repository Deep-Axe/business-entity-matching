"""Exact TF-IDF cosine and bounded one-to-one fuzzy token cosine.

The fuzzy score is our symmetric greedy alignment variant, not a claim to
implement the original SoftTF-IDF paper exactly. No external data or labels.
"""
import argparse
from collections import Counter
from functools import lru_cache
import math
from pathlib import Path
import pickle

from anyascii import anyascii
import numpy as np
from rapidfuzz import fuzz

from ber.text import normalize
from v2_cpu_features import Weighting
from v3_transfer_data import ROOT,digest,write

SCRATCH=Path('work/ce')
NAMES=[f'{view}_{kind}' for view in ('name','ascii_name','address') for kind in ('tfidf_cosine','aligned_fuzzy_cosine')]


def similarities(left,right,idf,min_similarity=.8):
    lc,rc=Counter(left),Counter(right)
    lw={t:n*idf(t) for t,n in lc.items()};rw={t:n*idf(t) for t,n in rc.items()}
    norm=math.sqrt(sum(v*v for v in lw.values())*sum(v*v for v in rw.values()))
    if not norm:return [0.,0.]
    exact=sum(lw[t]*rw[t] for t in lw.keys()&rw.keys())/norm
    edges=[]
    for a,wa in lw.items():
        for b,wb in rw.items():
            sim=fuzz.ratio(a,b)/100
            if sim>=min_similarity:
                edges.append((wa*wb*sim,min(a,b),max(a,b),a,b))
    used_l,used_r=set(),set();total=0.
    for weight,_,__,a,b in sorted(edges,key=lambda e:(-e[0],e[1],e[2])):
        if a not in used_l and b not in used_r:
            total+=weight;used_l.add(a);used_r.add(b)
    return [min(1.,exact),min(1.,total/norm)]


def prepare(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    index=ROOT/'artifacts/train_index_v2';w=Weighting(index)
    @lru_cache(maxsize=40000)
    def words(name,address):
        return (normalize(name).split(),normalize(anyascii(name)).split(),normalize(address,True).split())
    inputs={}
    for split in ('tune','dev','rivals'):
        source=SCRATCH/f'{split}.pkl';inputs[split]=dict(path=str(source),sha256=digest(source))
        rows=pickle.load(source.open('rb'));out=np.zeros((len(rows),6),dtype=np.float32)
        for i,r in enumerate(rows):
            left,right=words(r[5],r[6]),words(r[7],r[8])
            for j,field in enumerate(('name_words','ascii_words','address_words')):
                out[i,2*j:2*j+2]=similarities(left[j],right[j],lambda t:w.idf(field,t))
        np.save(output/f'{split}.npy',out)
        inputs[split]['features_sha256']=digest(output/f'{split}.npy')
        print(split,len(rows),flush=True)
    write(output/'manifest.json',dict(complete=True,names=NAMES,inputs=inputs,
        feature_code_sha256=digest(Path(__file__)),index_config_sha256=digest(index/'config.json'),
        labels_used=False,external_data_used=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    prepare(**vars(p.parse_args()))
