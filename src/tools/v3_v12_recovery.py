"""V12-compatible unfiltered top-10 rescue, with coherent bounded ownership.

Fits only project-local exploratory models; never rewrites peer artifacts.
Existing V12 base/dense/key scores stay fixed. Choose rescue threshold on tune.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import time

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

from ber.store import connect
from v3_owner_assignment import load_text, text_logit, winner
from v3_transfer_data import ROOT, PILOT, digest, write, bootstrap
from v3_v10_combined_audit import model, logit

C=Path('work/ce')
OUT=ROOT/'artifacts/improvement_v3/v12_top10_rescue_v1'


def run():
    OUT.mkdir(parents=True,exist_ok=False); started=time.monotonic(); bound={}
    def track(p):
        p=Path(p); h=digest(p)
        assert bound.get(str(p),h)==h
        bound[str(p)]=h
        return p
    def raw(p):
        with track(p).open('rb') as f:return pickle.load(f)
    def arr(p):return np.load(track(p),allow_pickle=False)
    def progress(stage,**kw):
        x=dict(stage=stage,seconds=time.monotonic()-started,**kw)
        write(OUT/'progress.json',x);print(json.dumps(x),flush=True)
    def matrix(r,p,ce):
        x=np.column_stack((arr(str(p)+'.ops.npy'),logit([v[4] for v in r]),
            logit(ce),arr(str(p)+'.nf.npy'),arr(str(p)+'.nf2.npy'))).astype(np.float32)
        assert x.shape==(len(r),33) and np.isfinite(np.delete(x,28,axis=1)).all()
        return x
    sub=C/'sub_v12_c8_dense10_keych'
    decl=json.loads(track(sub/'report.json').read_text())
    assert digest(track(sub/'matching_results.tsv'))==decl['matching_sha256']
    rows={k:[] for k in ('old','k10','kc','u10')}
    mats={k:[] for k in ('old','c8','k10','kc','u10')}
    arrays={}
    for sp in ('dev','tune'):
        for tag,suf in (('old',''),('k10','_k10'),('kc','_kc'),('u10','_u10')):
            r=raw(C/f'{sp}{suf}.pkl');rows[tag].extend(r)
            cp=C/f'{sp}_u10.pkl.e5_cont1300k.npy' if tag=='u10' else C/f'e5_cont1300k/{sp}{suf}_pce.npy'
            mats[tag].append(matrix(r,C/f'{sp}{suf}.pkl',arr(cp)))
            if tag=='old':mats['c8'].append(matrix(r,C/f'{sp}.pkl',arr(C/f'c8/{sp}_pce.npy')))
        with arr(PILOT/'data'/f'{sp}.npz') as z:
            arrays[sp]={k:z[k] for k in ('reference_ids','true_counts','countries','left','right')}
    mats={k:np.vstack(v) for k,v in mats.items()}
    keys={k:{r[:2] for r in rs} for k,rs in rows.items()}
    assert keys['old']<=keys['k10']<=keys['kc']
    ui=np.asarray([i for i,r in enumerate(rows['u10']) if r[:2] not in keys['kc']])
    extra=[rows['u10'][i] for i in ui]
    rows['union']=rows['kc']+extra
    mats['union']=np.vstack((mats['kc'],mats['u10'][ui]))
    assert len({r[:2] for r in rows['union']})==len(rows['union'])
    refs=[str(o) for a in arrays.values() for o in a['reference_ids']]
    db=connect(ROOT/'artifacts/train_store_v2');db.execute('PRAGMA cache_size=-4096')
    gold={o:{r[0] for r in db.execute('SELECT target_id FROM labels WHERE reference_id=?',(o,))} for o in refs};db.close()
    assert all(int(r[1] in gold[r[0]])==r[2] for r in rows['union'])
    for sp,a in arrays.items():assert [len(gold[str(o)]) for o in a['reference_ids']]==list(a['true_counts'])

    # Sample final-fit parity with the scores actually used in V12.
    samples={'old':(C/'test/part_000.pkl','e5_cont1300k'),
        'c8':(C/'test/part_000.pkl','c8'),
        'k10':(C/'test_dense_k10/delta_000.pkl','e5_cont1300k'),
        'kc':(C/'test_keych/kcx_007.pkl','e5_cont1300k')}
    parity={}
    for name,(path,ce_tag) in samples.items():
        rr=raw(path); ce=arr(str(path)+f'.{ce_tag}.npy')
        valid=np.flatnonzero(np.isfinite(ce))[:1024]
        xx=matrix(rr,path,np.nan_to_num(ce,nan=.5))[valid]
        fitrows=rows['old' if name=='c8' else name]
        m=model().fit(mats[name],[r[2] for r in fitrows])
        saved=arr(str(path)+f'.stack_{ce_tag}_nocty_nf2.npy')[valid]
        pp=m.predict_proba(xx)[:,1].astype(np.float32)
        parity[name]=dict(sample=len(valid),max_delta=float(np.max(np.abs(pp-saved))))
        assert parity[name]['max_delta']<1e-7,(name,parity[name])
        del rr,xx,ce,m
    progress('verified_v12_score_parity',parity=parity,new_pairs=len(extra),new_true=sum(r[2] for r in extra))

    ids=[json.loads(line)['id'] for line in track(PILOT/'data/records.jsonl').open()]
    rawkeys={(ids[int(l)],ids[int(r)]) for a in arrays.values() for l,r in zip(a['left'],a['right'])};del ids
    newrows=[r for r in rows['union'] if r[:2] not in keys['old']]
    newkeys={r[:2] for r in newrows}
    rival=raw(C/'rivals.pkl')
    aux=np.asarray([i for i,r in enumerate(rival) if r[:2] not in rawkeys and r[:2] not in newkeys])
    oldall=rows['old']+[rival[i] for i in aux]
    allrows=oldall+newrows;nold=len(rows['old']);nbase=len(oldall)
    am={}
    for name,tag in (('old','e5_cont1300k'),('c8','c8')):
        rm=matrix(rival,C/'rivals.pkl',arr(C/f'rivals.pkl.{tag}.npy'))
        am[name]=np.vstack((mats[name],rm[aux]))
    focalrows=rows['old']+newrows
    ix=np.r_[np.arange(nold),np.arange(nbase,len(allrows))]
    assert len({r[:2] for r in allrows})==len(allrows)
    rule=json.loads(track(ROOT/'artifacts/improvement_v3/structural_v1/owner_eval/policy.json').read_text())['selected']
    tm=load_text(track(ROOT/'artifacts/improvement_v3/structural_v1/text_scorer.json'))
    text=np.asarray([text_logit({'name':r[5],'address':r[6]},{'name':r[7],'address':r[8]},tm) for r in allrows])
    groups=defaultdict(list)
    for i,r in enumerate(allrows):groups[r[1]].append(i)
    groups=[np.asarray(sorted(g,key=lambda i:allrows[i][0])) for g in groups.values() if len(g)>1]
    dense=np.asarray([r[:2] in keys['k10'] for r in newrows])
    key=np.asarray([r[:2] in keys['kc'] and r[:2] not in keys['k10'] for r in newrows])
    rescue=~(dense|key)
    indices={tag:{r[:2]:i for i,r in enumerate(rows[tag])} for tag in ('k10','kc','union')}
    d10=np.asarray([indices['k10'][r[:2]] for r in newrows if r[:2] in keys['k10']])
    dkc=np.asarray([indices['kc'][r[:2]] for r in newrows if r[:2] in keys['kc'] and r[:2] not in keys['k10']])
    dr=np.asarray([indices['union'][r[:2]] for r in newrows if r[:2] not in keys['kc']])
    rr=mats['union'][dr]
    gate=np.asarray([r[4]>=.01 for r in oldall])
    def decide(p,threshold):
        keep=np.r_[(p[:nbase]>=.75)&gate,(p[nbase:]>=.9)&~rescue | ((p[nbase:]>=threshold)&rescue)]
        for g in groups:
            acc=g[keep[g]]
            if len(acc)>1:
                w=winner(p[acc],text[acc],**rule);keep[acc]=False
                if w is not None:keep[acc[w]]=True
        return keep[ix]
    own=np.asarray([r[0] for r in rows['old']]);y=np.asarray([r[2] for r in rows['old']])
    folds=list(GroupKFold(5).split(mats['old'],y,own))
    of={o:k for k,(_,held) in enumerate(folds) for o in own[held]}
    for o in refs:of.setdefault(o,int(hashlib.sha256(o.encode()).hexdigest()[:8],16)%5)
    ff={t:np.asarray([of[r[0]] for r in rows[t]]) for t in ('k10','kc','union')}
    rf=np.asarray([of[r[0]] for r in focalrows])
    thresholds=(.5,.75,.9,.95,.98,.99,.995,.999,1.1)
    arms={('baseline',1.1):np.zeros(len(focalrows),bool)}
    for head in ('kc','union'):
        for t in thresholds:arms[head,t]=np.zeros(len(focalrows),bool)
    foldbase=[]; foldrescue={'kc':[],'union':[]}
    for k,(fit,held) in enumerate(folds):
        progress('coherent_fold',fold=k)
        op=model().fit(mats['old'][fit],y[fit]).predict_proba(am['old'])[:,1]
        cp=model().fit(mats['c8'][fit],y[fit]).predict_proba(am['c8'])[:,1]
        bp=np.where((op>=.05)&(op<.98),cp,op)
        newp=np.zeros(len(newrows)); pp={}
        for name in ('k10','kc','union'):
            f=ff[name]!=k
            m=model().fit(mats[name][f],np.asarray([r[2] for r in rows[name]])[f])
            if name=='k10':newp[dense]=m.predict_proba(mats[name][d10])[:,1]
            if name=='kc':newp[key]=m.predict_proba(mats[name][dkc])[:,1]
            if name!='k10':pp[name]=m.predict_proba(rr)[:,1]
        p=np.r_[bp,newp];mask=rf==k
        arms['baseline',1.1][mask]=decide(p,1.1)[mask]
        foldbase.append(p.astype(np.float32))
        for head in pp:
            pnew=p.copy();pnew[nbase+np.flatnonzero(rescue)]=pp[head]
            foldrescue[head].append(pp[head].astype(np.float32))
            for t in thresholds:arms[head,t][mask]=decide(pnew,t)[mask]

    def metric(mask,sp):
        pred=defaultdict(set)
        for r,k in zip(focalrows,mask):
            if k:pred[r[0]].add(r[1])
        a=arrays[sp];os=list(map(str,a['reference_ids']));tt=a['true_counts']
        tp=np.asarray([len(pred[o]&gold[o]) for o in os]);nn=np.asarray([len(pred[o]) for o in os]);den=tt+4*nn
        ss=np.divide(5*tp,den,out=np.ones(len(os)),where=den!=0)
        return ss,dict(macro_f05=float(ss.mean()),tp=int(tp.sum()),fp=int((nn-tp).sum()),fn=int((tt-tp).sum()),
            singleton_false_merges=int(((tt==0)&(nn>0)).sum()),
            country={str(c):float(ss[a['countries']==c].mean()) for c in np.unique(a['countries'])})
    baseline={sp:metric(arms['baseline',1.1],sp) for sp in arrays}
    expected=json.loads(track(ROOT/'artifacts/improvement_v3/kc_union_audit_v1/report.json').read_text())
    for sp in arrays:assert abs(baseline[sp][1]['macro_f05']-expected['splits'][sp]['metrics']['k10_plus_key']['macro_f05'])<1e-12
    tune={f'{h}@{t}':metric(v,'tune')[1] for (h,t),v in arms.items()}
    selected=max(arms,key=lambda ht:(tune[f'{ht[0]}@{ht[1]}']['macro_f05'],ht[1]))
    result={}
    for sp,a in arrays.items():
        ss,met=metric(arms[selected],sp);rrset=set(map(str,a['reference_ids']));tr=Counter()
        for i,r in enumerate(focalrows):
            if r[0] not in rrset:continue
            if arms[selected][i] and not arms['baseline',1.1][i]:tr['added_true' if r[2] else 'added_false']+=1
            if arms['baseline',1.1][i] and not arms[selected][i]:tr['removed_true' if r[2] else 'removed_false']+=1
        result[sp]=dict(baseline=baseline[sp][1],selected=met,delta=bootstrap(baseline[sp][0],ss),transitions=dict(tr))
    for p,h in bound.items():assert digest(p)==h,'Input changed: '+p
    with (OUT/'context.pkl').open('wb') as f:pickle.dump(dict(rows=focalrows,allrows=allrows,ix=ix,fold=rf,arrays=arrays,gold=gold,
        groups=groups,text=text,nbase=nbase,rescue=rescue,gate=gate,rule=rule),f)
    np.savez_compressed(OUT/'fold_scores.npz',baseline=np.asarray(foldbase),kc=np.asarray(foldrescue['kc']),union=np.asarray(foldrescue['union']))
    np.savez_compressed(OUT/'decisions.npz',baseline=arms['baseline',1.1],selected=arms[selected])
    report=dict(complete=True,public_v12_score=.980214,submission_sha256=decl['matching_sha256'],parity=parity,
        new_pairs=len(extra),new_true=sum(r[2] for r in extra),tune_grid=tune,selected=dict(head=selected[0],threshold=selected[1]),splits=result,
        inputs_sha256=bound,seconds=time.monotonic()-started,
        caveats=['Exploratory repeatedly used US/India owners, not French truth or fresh reporting.',
            'Same held-owner folds and bounded original reverse rivals; not exhaustive owner competition.',
            'Only new recovery edges can enter at a changed threshold; V12 input scores remain fixed.',
            'Threshold/head selected on tune only; dev is reported at that choice.'])
    write(OUT/'report.json',report);progress('complete',selected=report['selected'],splits=result)


if __name__=='__main__':run()
