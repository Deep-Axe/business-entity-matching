"""Diagnostic: separate old and Kaggle encoder logits, with same band cascade.

No public/test scoring. Threshold selected on tune only; reused dev reported
once at that choice. Uses the same bounded coherent owner competition.
"""
from collections import defaultdict
import pickle
import numpy as np
from sklearn.model_selection import GroupKFold
from v3_neural_test_pipeline import SCRATCH, STRUCT, resources, read
from v3_public0976_error_audit import features, model
from v3_neural_independent_audit import lg, summary
from v3_owner_assignment import load_text, text_logit, winner
from v3_audit_v6_band import blend_scores
from v3_transfer_data import ROOT, PILOT, digest, write, bootstrap

OUT=ROOT/'artifacts/improvement_v3/v6_separate_encoder_pilot_v1'
AUDIT=ROOT/'artifacts/improvement_v3/v6_band_error_audit_v1'


def run():
    OUT.mkdir(parents=True,exist_ok=False);bound={}
    def track(p):bound[str(p)]=digest(p);return p
    def raw(p):return pickle.load(track(p).open('rb'))
    def arr(p):return np.load(track(p),allow_pickle=False)
    data={};rows=[];mm=[];cc=[];fullkeys=set();ids=[__import__('json').loads(x)['id'] for x in track(PILOT/'data/records.jsonl').open()]
    for split in ('dev','tune'):
        rr=raw(SCRATCH/(split+'.pkl'));start=len(rows);rows+=rr
        mm.append(features(rr,arr(SCRATCH/(split+'.pkl.ops.npy')),arr(SCRATCH/'e5_cont1300k'/(split+'_pce.npy')),
            np.column_stack((arr(SCRATCH/(split+'.pkl.nf.npy')),arr(SCRATCH/(split+'.pkl.nf2.npy'))))))
        cc.extend(lg(arr(SCRATCH/'kC'/(split+'_pce.npy'))))
        with np.load(track(PILOT/'data'/(split+'.npz')),allow_pickle=False) as z:
            a={k:z[k] for k in ('reference_ids','query_ids','true_counts','countries','y','left','right')}
        keys={(ids[int(l)],ids[int(r)]):i for i,(l,r) in enumerate(zip(a['left'],a['right']))};fullkeys.update(keys)
        data[split]=(a,np.asarray([keys[r[:2]] for r in rr]),slice(start,len(rows)))
    focal=len(rows);M=np.vstack(mm);N=np.column_stack((M,np.asarray(cc))).astype(np.float32);Y=np.asarray([r[2] for r in rows]);owners=np.asarray([r[0] for r in rows])
    rivals=raw(SCRATCH/'rivals.pkl');RM=features(rivals,arr(SCRATCH/'rivals.pkl.ops.npy'),arr(SCRATCH/'rivals.pkl.e5_cont1300k.npy'),
        np.column_stack((arr(SCRATCH/'rivals.pkl.nf.npy'),arr(SCRATCH/'rivals.pkl.nf2.npy'))))
    RN=np.column_stack((RM,lg(arr(SCRATCH/'rivals.pkl.kC.npy')))).astype(np.float32)
    aux=np.asarray([i for i,r in enumerate(rivals) if r[:2] not in fullkeys]);allrows=rows+[rivals[i] for i in aux];AM=np.vstack((M,RM[aux]));AN=np.vstack((N,RN[aux]))
    groups=defaultdict(list)
    for i,r in enumerate(allrows):groups[r[1]].append(i)
    groups=[np.asarray(sorted(v,key=lambda i:allrows[i][0])) for v in groups.values() if len(v)>1]
    tm=load_text(track(STRUCT/'text_scorer.json'));rule=read(track(STRUCT/'owner_eval/policy.json'))['selected']
    text=np.asarray([text_logit({'name':r[5],'address':r[6]},{'name':r[7],'address':r[8]},tm) for r in allrows]);p100=np.asarray([r[4] for r in allrows])
    def decide(p,t):
        keep=(p>=t)&(p100>=.01)
        for g in groups:
            ix=g[keep[g]]
            if len(ix)<2:continue
            w=winner(p[ix],text[ix],**rule);keep[ix]=False
            if w is not None:keep[ix[w]]=True
        return keep
    grid=[.65,.70,.75,.80,.85];keeps={t:np.zeros(focal,bool) for t in grid};scores=np.zeros(focal)
    prior=arr(AUDIT/'decisions.npz')
    for fold,(fit,held) in enumerate(GroupKFold(5).split(M,Y,owners)):
        resources();write(OUT/'progress.json',dict(stage='crossfit',fold=fold))
        old=model().fit(M[fit],Y[fit]);new=model().fit(N[fit],Y[fit]);op=old.predict_proba(AM)[:,1];pp=new.predict_proba(AN)[:,1]
        if not np.allclose(op[held],prior['baseline_p'][held],atol=1e-12,rtol=0):raise ValueError('Baseline fold changed')
        pp=blend_scores(op,pp);scores[held]=pp[held]
        for t in grid:keeps[t][held]=decide(pp,t)[held]
    result={};vectors={};prior_vectors={}
    for split,(a,ix,sl) in data.items():
        result[split]={};vectors[split]={};prior_vectors[split]={}
        for name in ('baseline','band_cC'):
            keep=np.zeros(len(a['y']),bool);keep[ix]=prior[name+'_keep'][sl];prior_vectors[split][name],_=summary(a,keep)
        for t in grid:
            keep=np.zeros(len(a['y']),bool);keep[ix]=keeps[t][sl];vectors[split][t],result[split][t]=summary(a,keep)
    selected=max(grid,key=lambda t:(result['tune'][t]['macro_f05'],t))
    report=dict(complete=True,features=34,tune_selected_threshold=selected,tune_grid=result['tune'],dev_at_selected=result['dev'][selected],
        comparison={s:{n:bootstrap(vectors[s][selected],prior_vectors[s][n]) for n in []} for s in []},
        deployment_performed=False,note='One exploratory recipe; held-out owner OOF but reused data and bounded rivals, no French labels.')
    report['comparison']={s:{n:bootstrap(prior_vectors[s][n],vectors[s][selected]) for n in prior_vectors[s]} for s in data}
    for p,h in bound.items():
        if digest(p)!=h:raise ValueError('Pilot input changed: '+p)
    report['inputs_sha256']=bound;write(OUT/'report.json',report)
    np.savez_compressed(OUT/'decisions.npz',score=scores,keep=keeps[selected]);write(OUT/'completion.json',dict(complete=True,report_sha256=digest(OUT/'report.json')))


if __name__=='__main__':run()
