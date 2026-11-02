"""Owner-balanced training-loss pilot on the reproduced public-best recipe.

Three predeclared variants, same owner folds and coherent bounded rivals.
Threshold/model selection uses tune OOF only; reused dev is reported afterward.
No full-test inference or auto-promotion, and new 5k calibration stays untouched.
"""
from collections import defaultdict, Counter
import json
import pickle
import time

import numpy as np
from sklearn.model_selection import GroupKFold

from v3_neural_independent_audit import SCRATCH, STRUCT, lg, summary
from v3_neural_test_pipeline import resources
from v3_owner_assignment import load_text, text_logit, winner
from v3_public0976_error_audit import model, features
from v3_transfer_data import ROOT, PILOT, digest, write, bootstrap

WORK=ROOT/'artifacts/improvement_v3/public0976_weight_ablation_v1'
AUDIT=ROOT/'artifacts/improvement_v3/public0976_error_audit_v2'
EXTRA=ROOT/'artifacts/improvement_v3/p200_neural_pilot_v1/features'
GRID=(.5,.6,.65,.7,.75,.8,.85,.9)


def owner_weights(owners,truth):
    counts=Counter(owners)
    return dict(inverse_routed_count=np.asarray([1/counts[o] for o in owners]),
                inverse_true_count=np.asarray([1/max(1,truth[o]) for o in owners]))


def run():
    WORK.mkdir(parents=True,exist_ok=False);start=time.monotonic();bound={}
    def track(p):bound[str(p)]=digest(p);return p
    def read(p):return json.loads(track(p).read_text())
    def raw(p):return pickle.load(track(p).open('rb'))
    def arr(p):return np.load(track(p),allow_pickle=False)
    prior=read(AUDIT/'report.json')
    if not prior['complete'] or not prior['test_partition_recipe_reproduced']:
        raise ValueError('Submitted recipe must first reproduce')
    for p,h in prior['inputs_sha256'].items():
        # The other session is actively editing its script and reusing generic
        # OOF filenames. Neither is executed/read by this ablation. The local
        # reproduced model helper and every actual array remain hash-bound.
        if 'stack_oof_' not in p and p!='src/neural/stack_eval.py' and digest(p)!=h:
            raise ValueError('Submitted recipe input changed: '+p)
    track(ROOT/'tools/v3_public0976_error_audit.py')
    manifest=read(EXTRA/'manifest.json');rows=[];matrices=[];extra=[];data={};spans={};full=set()
    if not manifest['complete'] or manifest['names']!=['genuine_p200k_logit']:
        raise ValueError('Expected genuine forward-context 200k probabilities')
    ids=[json.loads(line)['id'] for line in track(PILOT/'data/records.jsonl').open()]
    for split in ('dev','tune'):
        p=SCRATCH/(split+'.pkl');r=raw(p);start_row=len(rows);rows.extend(r);spans[split]=slice(start_row,len(rows))
        matrices.append(features(r,arr(SCRATCH/(split+'.pkl.ops.npy')),arr(SCRATCH/'e5_cont1300k'/(split+'_pce.npy')),
                        np.column_stack((arr(SCRATCH/(split+'.pkl.nf.npy')),arr(SCRATCH/(split+'.pkl.nf2.npy'))))))
        if digest(p)!=manifest['inputs'][split]['sha256'] or digest(EXTRA/(split+'.npy'))!=manifest['inputs'][split]['features_sha256']:
            raise ValueError('Structural pair alignment differs')
        extra.append(arr(EXTRA/(split+'.npy')))
        with np.load(track(PILOT/'data'/(split+'.npz')),allow_pickle=False) as z:
            a={k:z[k] for k in ('reference_ids','query_ids','y','true_counts','countries','left','right','baseline_100000')}
        keys={(ids[int(l)],ids[int(t)]):i for i,(l,t) in enumerate(zip(a['left'],a['right']))};full.update(keys)
        ix=np.asarray([keys[v[:2]] for v in r]);data[split]=(a,ix)
        if not np.array_equal(np.sort(ix),np.flatnonzero(a['baseline_100000']>=.01)):raise ValueError('Routed coverage differs')
    n=len(rows);r=raw(SCRATCH/'rivals.pkl');ix=np.asarray([i for i,v in enumerate(r) if v[:2] not in full])
    rm=features(r,arr(SCRATCH/'rivals.pkl.ops.npy'),arr(SCRATCH/'rivals.pkl.e5_cont1300k.npy'),
                np.column_stack((arr(SCRATCH/'rivals.pkl.nf.npy'),arr(SCRATCH/'rivals.pkl.nf2.npy'))))
    if digest(SCRATCH/'rivals.pkl')!=manifest['inputs']['rivals']['sha256'] or digest(EXTRA/'rivals.npy')!=manifest['inputs']['rivals']['features_sha256']:
        raise ValueError('Rival structural alignment differs')
    rows.extend(r[i] for i in ix);matrices.append(rm[ix]);extra.append(arr(EXTRA/'rivals.npy')[ix])
    base=np.vstack(matrices);p200=np.vstack(extra)
    if p200.shape!=(len(rows),1) or not np.isfinite(p200).all():raise ValueError('Ablation feature shapes differ')
    variants={name:base for name in ('submitted33','inverse_routed_count','inverse_true_count')}
    write(WORK/'predeclared.json',dict(variants={k:v.shape[1] for k,v in variants.items()},thresholds=GRID,
        recipe='Same 33-feature HGB/owner folds, only sample weighting changes; normalize weights within each fitting fold',
        truth_used='Inverse true count uses fitting-owner labels only; never an inference feature',
        not_exact_macro_f05_optimizer=True,
        untouched_new_calibration=True,full_test_deployment=False,code_sha256=digest(__file__)))
    rule=read(STRUCT/'owner_eval/policy.json')['selected'];tm=load_text(track(STRUCT/'text_scorer.json'))
    text=np.asarray([text_logit({'name':r[5],'address':r[6]}, {'name':r[7],'address':r[8]},tm) for r in rows])
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r[1]].append(i)
    groups=[np.asarray(sorted(v,key=lambda i:rows[i][0])) for v in groups.values() if len(v)>1]
    eligible=np.asarray([r[4]>=.01 for r in rows]);y=np.asarray([r[2] for r in rows[:n]]);owners=np.asarray([r[0] for r in rows[:n]])
    truth={str(o):int(t) for a,_ in data.values() for o,t in zip(a['reference_ids'],a['true_counts'])}
    weights=owner_weights(owners,truth)
    def decisions(p,threshold):
        keep=(p>=threshold)&eligible
        for group in groups:
            ix=group[keep[group]]
            if len(ix)<2:continue
            k=winner(p[ix],text[ix],**rule);keep[ix]=False
            if k is not None:keep[ix[k]]=True
        return keep
    masks={name:{t:np.zeros(n,bool) for t in GRID} for name in variants};oof={name:np.zeros(n) for name in variants}
    for name,X in variants.items():
        for k,(fit,held) in enumerate(GroupKFold(5).split(base[:n],y,owners)):
            resources();write(WORK/'progress.json',dict(stage='fitting_predeclared_oof',variant=name,fold=k,seconds=time.monotonic()-start))
            fit_weights=None if name=='submitted33' else weights[name][fit]/weights[name][fit].mean()
            m=model().fit(X[fit],y[fit],sample_weight=fit_weights);p=m.predict_proba(X)[:,1];oof[name][held]=p[held]
            for threshold in GRID:masks[name][threshold][held]=decisions(p,threshold)[held]
    def metric(split,keep):
        a,ix=data[split];full=np.zeros(len(a['y']),bool);full[ix]=keep[spans[split]];return summary(a,full)
    choices={}
    for name,by_t in masks.items():
        grid=[dict(threshold=t,macro_f05=metric('tune',keep)[1]['macro_f05']) for t,keep in by_t.items()]
        choices[name]=dict(selected=max(grid,key=lambda r:(r['macro_f05'],r['threshold'])),grid=grid)
    selected=max(choices,key=lambda k:(choices[k]['selected']['macro_f05'],k=='submitted33'))
    write(WORK/'selection_before_development.json',dict(selected=selected,choices=choices,selection='Tune OOF only'))
    old,old_metric=metric('dev',masks['submitted33'][.75])
    if abs(old_metric['macro_f05']-prior['metrics']['dev']['coherent_crossfit_ownership']['macro_f05'])>1e-12:
        raise ValueError('Exact submitted development baseline did not replay')
    reports={}
    for name,choice in choices.items():
        t=choice['selected']['threshold'];keep=masks[name][t];s,metrics=metric('dev',keep)
        before=masks['submitted33'][.75][spans['dev']];after=keep[spans['dev']];labels=y[spans['dev']]
        reports[name]=dict(tune=choice['selected'],development=metrics,comparison=bootstrap(old,s),
            changes=dict(recovered_true=int((~before&after&(labels==1)).sum()),lost_true=int((before&~after&(labels==1)).sum()),
                         added_false=int((~before&after&(labels==0)).sum()),removed_false=int((before&~after&(labels==0)).sum())))
        np.savez_compressed(WORK/(name+'.npz'),probabilities=oof[name],accepted=keep)
    for path,h in bound.items():
        if digest(path)!=h:raise ValueError('Input changed while fitting: '+path)
    write(WORK/'report.json',dict(complete=True,selected=selected,variants=reports,baseline_at_submitted_threshold=old_metric,
        inputs_sha256=bound,seconds=time.monotonic()-start,ready_for_submission=False,
        limitations=['Exploratory reused owner-OOF data; no pristine confirmation or French labels.',
        'Bounded rivals, not exhaustive competition.', 'Original autocast validation features; deployment precision must be separately matched.',
        'No full-test scoring, fresh calibration use, or automatic promotion.']))
    write(WORK/'progress.json',dict(stage='complete',selected=selected))


if __name__=='__main__':run()
