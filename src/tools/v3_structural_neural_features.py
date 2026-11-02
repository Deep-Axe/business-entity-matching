"""Add independently reproducible address/name views to the frozen CE pilot.

Full S1 name counts use provided training-gallery text only; fragments use the
existing training-only inventory. No French labels or development label fitting.
"""
import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

from ber.store import connect
from v3_neural_test_pipeline import SCRATCH,resources
from v3_structural_features import View,core_name,city_proxy,pair_features,NAMES
from v3_transfer_data import ROOT,digest,write


def run(output,fresh_root=None,audit_tag=None,rival_scores=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    sources={s:SCRATCH/f'{s}.pkl' for s in ('tune','dev','rivals')}
    if fresh_root:
        fresh_root=Path(fresh_root);end=time.monotonic()+1800
        while not (fresh_root/'completion.json').exists():
            if time.monotonic()>end:raise TimeoutError('Fresh feature extraction not complete within 30 minutes')
            resources();time.sleep(20)
        fresh=json.loads((fresh_root/'completion.json').read_text())
        if not fresh['complete']:raise ValueError('Incomplete fresh stacker data')
        sources['train5k']=fresh_root/'train5k.pkl'
    bound={s:dict(sha256=digest(p),path=str(p)) for s,p in sources.items()}
    fragments_path=ROOT/'artifacts/improvement_v3/structural_v1/feature_pilot_v2/fragments.json'
    fragments=json.loads(fragments_path.read_text())['fragments']
    interested=set()
    for path in sources.values():
        rows=pickle.load(path.open('rb'))
        interested.update((r[3],core_name(r[5])) for r in rows)
    del rows
    counts=Counter();cities=Counter();db=connect(ROOT/'artifacts/train_store_v2');db.execute('PRAGMA cache_size=-8192')
    start=time.monotonic();n=0
    for row in db.execute('SELECT country,name,address FROM anchors'):
        key=(row['country'],core_name(row['name']))
        if key in interested:
            counts[key]+=1;city=city_proxy(row['address'],row['country'])
            if city:cities[(*key,city)]+=1
        n+=1
        if n%250000==0:
            resources();write(output/'progress.json',dict(stage='full_s1_counts',anchors=n,seconds=time.monotonic()-start))
            print('counts',n,flush=True)
    db.close()
    @lru_cache(maxsize=80000)
    def view(name,address,country):return View.make(name,address,country,fragments)
    for split,path in sources.items():
        rows=pickle.load(path.open('rb'));X=np.empty((len(rows),len(NAMES)),np.float32)
        for i,r in enumerate(rows):
            left,right=view(r[5],r[6],r[3]),view(r[7],r[8],r[3])
            key=(r[3],left.core)
            if counts[key]<1:raise ValueError('A known training S1 is missing from full-gallery name counts')
            X[i]=pair_features(left,right,counts[key],cities.get((*key,left.city),0) if left.city else 0)
        if not np.isfinite(X).all():raise ValueError('Invalid extra features')
        if digest(path)!=bound[split]['sha256']:raise ValueError('Raw pair inputs changed')
        np.save(output/f'{split}.npy',X)
        bound[split].update(features_sha256=digest(output/f'{split}.npy'),pairs=len(rows))
        print('features',split,len(rows),flush=True)
    manifest=dict(complete=True,names=NAMES,inputs=bound,anchors_scanned=n,code_sha256=digest(__file__),
        features_code_sha256=digest(ROOT/'tools/v3_structural_features.py'),fragments_sha256=digest(fragments_path),
        train_store_config_sha256=digest(ROOT/'artifacts/train_store_v2/config.json'),
        counts_source='Complete provided training S1 text, no ownership labels',
        note='Conservative house/city parsers; unknown is not mismatch; no French accuracy claim')
    write(output/'manifest.json',manifest)
    if audit_tag:
        args=[sys.executable,'tools/v3_neural_independent_audit.py','--tag',audit_tag,'--extra-root',str(output),
              '--output',str(output/'evaluation')]
        if rival_scores:args+=['--rival-scores',str(rival_scores)]
        with (output/'audit.log').open('a') as log:subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fresh-root',type=Path);p.add_argument('--audit-tag');p.add_argument('--rival-scores',type=Path)
    run(**vars(p.parse_args()))
