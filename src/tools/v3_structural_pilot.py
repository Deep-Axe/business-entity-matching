"""Resumable feature prototype and preregistered tune-only stacker selection.

No labels from development are used for features, fitting, threshold selection
or choosing the feature family. This remains an exploratory reused-dev pilot.
"""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
from catboost import CatBoostClassifier

from ber.store import connect
from ber.features import FEATURE_NAMES
from v2_run_pipeline import NAMES as BASE_NAMES
from v3_structural_features import View,core_name,city_proxy,learn_fragments,pair_features,NAMES,FEATURE_GROUPS
from v3_owner_assignment import OUT
from v3_owner_evaluate import inputs
from v3_transfer_data import ROOT,PILOT,digest,load,write,metric,owner_scores,bootstrap

# Saved test candidates retain probability/IDs, not original retrieval hits.
# Raw lexical features, IDF features and complete candidate-context features can
# be reconstructed without retrieval; channel/rank/BM25 columns cannot.
UNAVAILABLE_BASE_COLUMNS=set(FEATURE_NAMES[30:])
REPLAY_COLUMNS=[i for i,name in enumerate(BASE_NAMES) if name not in UNAVAILABLE_BASE_COLUMNS]
REPLAY_NAMES=[BASE_NAMES[i] for i in REPLAY_COLUMNS]


def prepare(output=OUT/'feature_pilot_v2'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    identity={'feature_code_sha256':digest(ROOT/'tools/v3_structural_features.py'),
              'data_manifest_sha256':digest(PILOT/'data/manifest.json')}
    if (output/'identity.json').exists():
        if json.loads((output/'identity.json').read_text())!=identity:raise ValueError('Feature recipe changed; use a new output')
    else:write(output/'identity.json',identity)
    rows,arrays=inputs()
    if (output/'fragments.json').exists(): fragments=json.loads((output/'fragments.json').read_text())
    else:
        train=load('train');fragments=learn_fragments(rows,train);del train
        write(output/'fragments.json',fragments)
    @lru_cache(maxsize=20000)
    def view(i):
        r=rows[int(i)];return View.make(r['name'],r['address'],r['country'],fragments['fragments'])
    if (output/'counts.json').exists():
        counts=json.loads((output/'counts.json').read_text())
    else:
        interested={(rows[int(i)]['country'],core_name(rows[int(i)]['name']))
                    for a in arrays.values() for i in np.unique(a['left'])}
        names,cities=Counter(),Counter();db=connect(ROOT/'artifacts/train_store_v2');db.execute('PRAGMA cache_size=-4096')
        started=time.monotonic();n=0
        for r in db.execute('SELECT country,name,address FROM anchors'):
            key=(r['country'],core_name(r['name']))
            if key in interested:
                names['\t'.join(key)]+=1
                city=city_proxy(r['address'],r['country'])
                if city:cities['\t'.join((*key,city))]+=1
            n+=1
            if n%250000==0:print(json.dumps({'stage':'full_s1_name_counts','anchors':n,'seconds':time.monotonic()-started}),flush=True)
        db.close();counts={'names':dict(names),'cities':dict(cities),'anchors_scanned':n,
            'description':'Full S1 text only; city is a conservative comma-component proxy, not a geocoder'}
        write(output/'counts.json',counts)
    for split,a in arrays.items():
        completed=output/f'{split}.complete.json'
        if completed.exists():
            done=json.loads(completed.read_text())
            if done['features_sha256']!=digest(output/f'{split}.npy'):raise ValueError('Completed features changed')
            continue
        progress=output/f'{split}.progress.json';path=output/f'{split}.npy'
        start=json.loads(progress.read_text())['pairs_done'] if progress.exists() else 0
        if path.exists() and not progress.exists():raise ValueError('Unbound partial matrix')
        if start:
            matrix=np.lib.format.open_memmap(path,mode='r+')
        else:
            matrix=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(a['y']),len(NAMES)))
            write(progress,{'pairs_done':0})
        started=time.monotonic()
        for low in range(start,len(a['y']),5000):
            high=min(low+5000,len(a['y']))
            for i in range(low,high):
                l,r=view(int(a['left'][i])),view(int(a['right'][i]))
                key='\t'.join((l.country,l.core));ckey=key+'\t'+l.city
                matrix[i]=pair_features(l,r,counts['names'].get(key,0),counts['cities'].get(ckey,0) if l.city else 0)
            matrix.flush();write(progress,{'pairs_done':high,'pairs_total':len(a['y']),'seconds':time.monotonic()-started})
            print(json.dumps({'stage':'features','split':split,'pairs':high,'total':len(a['y'])}),flush=True)
        if not np.isfinite(matrix).all():raise ValueError('Non-finite features')
        del matrix
        write(completed,{'complete':True,'pairs':len(a['y']),'names':NAMES,'features_sha256':digest(path),
                         'identity':identity,'probability_rival_features':'separate exact replay job; not fabricated here'})


def rival_features(a,rows,rivals):
    """Only observed, scored reverse neighborhoods get a probability margin.

    Low-score candidates outside the accepted-target evaluation route remain
    unknown with an explicit availability bit; they are NOT assigned zero rivals.
    """
    db=sqlite3.connect(f'{Path(rivals).resolve().as_uri()}?mode=ro',uri=True)
    if db.execute('SELECT count(*) FROM targets WHERE discovered=0').fetchone()[0]:raise ValueError('Rival discovery incomplete')
    table='chosen' if db.execute("SELECT value FROM meta WHERE key='shortlist'").fetchone() else 'rivals'
    if db.execute(f'SELECT count(DISTINCT r.owner) FROM {table} r LEFT JOIN scores s ON s.owner=r.owner WHERE s.owner IS NULL').fetchone()[0]:raise ValueError('Rival scores incomplete')
    rankings={}
    for target,owner,p in db.execute('SELECT target,owner,p FROM claims ORDER BY target,p DESC,owner'):
        rankings.setdefault(target,[]).append((owner,p))
    result=np.zeros((len(a['y']),4),np.float32)
    for i,(li,ri) in enumerate(zip(a['left'],a['right'])):
        target=rows[int(ri)]['id'];owner=rows[int(li)]['id']
        if target not in rankings:continue
        other=[p for r,p in rankings[target] if r!=owner]
        result[i]=[1.,max(other,default=0.),float(a['baseline_100000'][i])-max(other,default=0.),
                   np.log1p(sum(p>=.76 for p in other))]
    db.close();return result


def fit(output=OUT/'feature_pilot_v2',with_rivals=False):
    output=Path(output)
    if (output/'report.json').exists():raise FileExistsError('This pilot already reported development; use its existing report')
    identity={'with_rivals':with_rivals,'pilot_code_sha256':digest(__file__),
              'feature_code_sha256':digest(ROOT/'tools/v3_structural_features.py'),
              'feature_identity_sha256':digest(output/'identity.json')}
    if (output/'fit_identity.json').exists():
        if json.loads((output/'fit_identity.json').read_text())!=identity:raise ValueError('Fit recipe changed; use a new output')
    else:write(output/'fit_identity.json',identity)
    rows,arrays=inputs();blocks={}
    for split,a in arrays.items():
        done=json.loads((output/f'{split}.complete.json').read_text())
        if done['features_sha256']!=digest(output/f'{split}.npy'):raise ValueError('Feature matrix changed')
        blocks[split]=np.load(output/f'{split}.npy')
    names=list(NAMES);groups=dict(FEATURE_GROUPS)
    if with_rivals:
        rnames=['rival_context_observed','rival_best_probability','rival_probability_margin','rival_log_accepted_count']
        for split,a in arrays.items():blocks[split]=np.column_stack((blocks[split],rival_features(a,rows,OUT/'owner_eval/rivals.sqlite3')))
        names+=rnames;groups['rival']=rnames
    # Core-name groups, not pair rows, define the internal tune split.
    a=arrays['tune'];owner_bucket=[]
    for i,ref in enumerate(a['reference_ids']):
        pair=np.flatnonzero(a['query_ids']==i)
        if len(pair):r=rows[int(a['left'][pair[0]])];key=r['country']+'\t'+core_name(r['name'])
        else:key=str(ref)
        owner_bucket.append(int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%10)
    validation=np.array(owner_bucket)>=7;training=~validation
    def subset(a,owner_mask):
        ix=np.flatnonzero(owner_mask);mapping=np.full(len(owner_mask),-1,dtype=np.int32);mapping[ix]=np.arange(len(ix))
        pair=owner_mask[a['query_ids']]
        return {'query_ids':mapping[a['query_ids'][pair]],'y':a['y'][pair],
                'true_counts':a['true_counts'][owner_mask]},pair
    fit_a,fit_pair=subset(a,training);val_a,val_pair=subset(a,validation)
    configs=[('score_only',[]),('existing_features',[]),('all_new',names)]
    configs += [('without_'+g,[n for n in names if n not in cols]) for g,cols in groups.items()]
    thresholds=np.r_[np.arange(.05,1.,.01),np.nextafter(1.,2.)]
    hyper=dict(iterations=250,depth=5,learning_rate=.06,l2_leaf_reg=10.,loss_function='Logloss',
               thread_count=1,random_seed=20260926,allow_writing_files=False,verbose=False,used_ram_limit='900mb')
    results=[];models={}
    def matrix(split,kind,cols):
        ar=arrays[split];base=ar['baseline_100000'][:,None]
        if kind!='score_only':base=np.column_stack((base,ar['X'][:,REPLAY_COLUMNS]))
        if cols:base=np.column_stack((base,blocks[split][:,[names.index(n) for n in cols]]))
        return base.astype(np.float32,copy=False)
    # All choices and thresholds are fixed without looking at development labels.
    for kind,cols in configs:
        path=output/f'{kind}.selection.json'
        if path.exists():results.append(json.loads(path.read_text()));continue
        X=matrix('tune',kind,cols);started=time.monotonic()
        model=CatBoostClassifier(**hyper).fit(X[fit_pair],a['y'][fit_pair])
        p=model.predict_proba(X[val_pair],thread_count=1)[:,1]
        value,threshold=max((float(owner_scores(val_a,p,t).mean()),float(t)) for t in thresholds)
        result={'name':kind,'new_columns':cols,'validation_macro_f05':value,'threshold':threshold,
                'internal_fit_owners':int(training.sum()),'internal_validation_owners':int(validation.sum()),
                'seconds':time.monotonic()-started}
        results.append(result);write(path,result);print(json.dumps({'stage':'stacker_selection',**result}),flush=True)
    chosen=max(results,key=lambda r:r['validation_macro_f05'])
    protocol={'selected':chosen['name'],'trials':results,'hyperparameters':hyper,'rival_features_included':with_rivals,
              'reconstructible_base_columns':REPLAY_NAMES,'excluded_unsaved_retrieval_columns':sorted(UNAVAILABLE_BASE_COLUMNS),
              'feature_code_sha256':digest(ROOT/'tools/v3_structural_features.py'),
              'pilot_code_sha256':digest(__file__),'tune_fit_then_group_heldout_selection':True,
              'development_used_for_selection':False,'deployment_approved':False}
    frozen=output/'selection_before_dev.json'
    if frozen.exists() and json.loads(frozen.read_text())!=protocol:raise ValueError('Frozen selection changed')
    write(frozen,protocol)
    # Predeclared ablations are each refit on all tune owners and reported once.
    report={'protocol':protocol,'development':{},'baseline_100k':metric(arrays['dev'],arrays['dev']['baseline_100000'],.76),
            'baseline_200k':metric(arrays['dev'],arrays['dev']['baseline_200000'],.70),
            'limitations':['Exploratory reused US/India development, no French labels',
                          'Tuning owners become stacker training data; not eligible as a future independent audit',
                          'Native aliases and pseudo-fragments are optional views, not hard acceptance rules',
                          '200k deployment needs its own held-out or cross-fitted stacking data']}
    for result in results:
        kind=result['name'];saved=output/f'{kind}.dev.json'
        if saved.exists():report['development'][kind]=json.loads(saved.read_text());continue
        cols=result['new_columns'];model=CatBoostClassifier(**hyper)
        model.fit(matrix('tune',kind,cols),arrays['tune']['y'])
        model.save_model(str(output/f'{kind}.cbm'))
        p=model.predict_proba(matrix('dev',kind,cols),thread_count=1)[:,1]
        np.save(output/f'{kind}.dev_probabilities.npy',p)
        ar=arrays['dev'];score=metric(ar,p,result['threshold'])
        value={'threshold':result['threshold'],'metric':score,
               'comparison100k':bootstrap(owner_scores(ar,ar['baseline_100000'],.76),owner_scores(ar,p,result['threshold'])),
               'importance':sorted(zip(['base_probability']+([] if kind=='score_only' else REPLAY_NAMES)+cols,
                                       map(float,model.feature_importances_)),key=lambda x:-x[1])[:25]}
        report['development'][kind]=value;write(saved,value)
        print(json.dumps({'stage':'stacker_development_once','variant':kind,'macro_f05':score['macro_f05']}),flush=True)
    write(output/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','fit'])
    p.add_argument('--output',type=Path,default=OUT/'feature_pilot_v2');p.add_argument('--with-rivals',action='store_true');a=p.parse_args()
    if a.command=='prepare':prepare(a.output)
    else:fit(a.output,a.with_rivals)
