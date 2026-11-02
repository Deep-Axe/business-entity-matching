"""Reproduce cC stacker and evaluate the actual band cascade, not full replacement.

Grouped owner OOF, full ground truth including retrieval misses, coherent
same-fold rivals, fixed deployed threshold first. No peer/test output changes.
"""
from collections import Counter, defaultdict
import pickle
import time
import numpy as np
from sklearn.model_selection import GroupKFold
from rapidfuzz import fuzz
from ber.store import connect, lookup
from v3_neural_test_pipeline import SCRATCH, STRUCT, resources, read
from v3_public0976_error_audit import features, model
from v3_neural_independent_audit import lg, summary
from v3_owner_assignment import load_text, text_logit, winner
from v3_owner_error_analysis import attribution, score, missing
from v3_structural_features import folded, house_number, street_component, single_digit_change
from v3_transfer_data import ROOT, PILOT, digest, write, bootstrap

WORK=ROOT/'artifacts/improvement_v3/v6_band_error_audit_v1'
BASE=ROOT/'artifacts/improvement_v3/public0976_error_audit_v2'


def blend_scores(old,new):
    return np.where((old>=.05)&(old<.98),new,old)


def run():
    WORK.mkdir(parents=True,exist_ok=False);started=time.monotonic();bound={}
    def track(p):bound[str(p)]=digest(p);return p
    def raw(p):return pickle.load(track(p).open('rb'))
    def arr(p):return np.load(track(p),allow_pickle=False)
    def progress(stage,**kw):write(WORK/'progress.json',dict(stage=stage,seconds=time.monotonic()-started,**kw))
    resources();ids=[__import__('json').loads(s)['id'] for s in track(PILOT/'data/records.jsonl').open()]
    rows=[];matrices={k:[] for k in ('old','new')};data={};all_keys=set()
    for split in ('dev','tune'):
        r=raw(SCRATCH/(split+'.pkl'));start=len(rows);rows+=r
        ce=arr(SCRATCH/'e5_cont1300k'/(split+'_pce.npy'));kc=arr(SCRATCH/'kC'/(split+'_pce.npy'));cc=arr(SCRATCH/'cC'/(split+'_pce.npy'))
        if not np.array_equal(cc,1/(1+np.exp(-(.5*lg(ce)+.5*lg(kc))))):raise ValueError('cC logit ensemble identity differs')
        ops=arr(SCRATCH/(split+'.pkl.ops.npy'));freq=np.column_stack((arr(SCRATCH/(split+'.pkl.nf.npy')),arr(SCRATCH/(split+'.pkl.nf2.npy'))))
        matrices['old'].append(features(r,ops,ce,freq));matrices['new'].append(features(r,ops,cc,freq))
        with np.load(track(PILOT/'data'/(split+'.npz')),allow_pickle=False) as z:
            a={k:z[k] for k in ('reference_ids','query_ids','true_counts','countries','y','left','right','baseline_100000')}
        keys={(ids[int(l)],ids[int(t)]):i for i,(l,t) in enumerate(zip(a['left'],a['right']))}
        ix=np.asarray([keys[x[:2]] for x in r]);all_keys.update(keys)
        if not np.array_equal(a['y'][ix],[x[2] for x in r]):raise ValueError('Label alignment mismatch')
        data[split]=(a,ix,slice(start,len(rows)),keys)
    focal=len(rows);Y=np.asarray([r[2] for r in rows]);owners=np.asarray([r[0] for r in rows]);M={k:np.vstack(v) for k,v in matrices.items()}
    rivals=raw(SCRATCH/'rivals.pkl');ro=arr(SCRATCH/'rivals.pkl.e5_cont1300k.npy');rn=arr(SCRATCH/'rivals.pkl.cC.npy')
    ops=arr(SCRATCH/'rivals.pkl.ops.npy');freq=np.column_stack((arr(SCRATCH/'rivals.pkl.nf.npy'),arr(SCRATCH/'rivals.pkl.nf2.npy')))
    RM={'old':features(rivals,ops,ro,freq),'new':features(rivals,ops,rn,freq)}
    aux=np.asarray([i for i,r in enumerate(rivals) if r[:2] not in all_keys]);all_rows=rows+[rivals[i] for i in aux]
    AM={k:np.vstack((M[k],RM[k][aux])) for k in M}
    progress('final_fit_parity_check')
    final=model().fit(M['new'],Y);f=SCRATCH/'test/part_000.pkl';test=raw(f)
    # Test cC has NaN off-band. The deployed generic stacker handles those, but
    # the cascade discards those outputs. Validate only genuinely scored rows.
    ce=arr(__import__('pathlib').Path(str(f)+'.cC.npy'));mask=np.isfinite(ce)
    ops=arr(__import__('pathlib').Path(str(f)+'.ops.npy'))
    freq=np.column_stack((arr(__import__('pathlib').Path(str(f)+'.nf.npy')),arr(__import__('pathlib').Path(str(f)+'.nf2.npy'))))
    ix=np.flatnonzero(mask);fm=features([test[i] for i in ix],ops[ix],ce[ix],freq[ix])
    expected=arr(__import__('pathlib').Path(str(f)+'.stack_cC_nocty_nf2.npy'))[ix]
    generated=final.predict_proba(fm)[:,1].astype(np.float32);delta=float(np.max(np.abs(generated-expected)))
    write(WORK/'parity.json',dict(test_partition=0,in_band_rows=len(ix),max_delta=delta,exact=np.array_equal(generated,expected)))
    if delta>1e-7:raise ValueError('cC deployed stacker not reproduced')
    del final,test,fm,ops,freq,ce,expected,generated
    rule=read(track(STRUCT/'owner_eval/policy.json'))['selected'];tm=load_text(track(STRUCT/'text_scorer.json'))
    text=np.asarray([text_logit({'name':r[5],'address':r[6]},{'name':r[7],'address':r[8]},tm) for r in all_rows]);base=np.asarray([r[4] for r in all_rows])
    groups=defaultdict(list)
    for i,r in enumerate(all_rows):groups[r[1]].append(i)
    groups=[np.asarray(sorted(v,key=lambda i:all_rows[i][0])) for v in groups.values() if len(v)>1]
    def decide(p,threshold=.75):
        keep=(p>=threshold)&(base>=.01)
        for g in groups:
            ix=g[keep[g]]
            if len(ix)<2:continue
            ch=winner(p[ix],text[ix],**rule);keep[ix]=False
            if ch is not None:keep[ix[ch]]=True
        return keep
    names=('baseline','full_cC','band_cC');probs={k:np.zeros(focal) for k in names};keeps={k:np.zeros(focal,bool) for k in names}
    grid=[.65,.70,.75,.80,.85];threshold_keeps={t:np.zeros(focal,bool) for t in grid}
    old_saved=arr(BASE/'decisions.npz');saved_old_p=old_saved['oof'];saved_old_keep=old_saved['coherent']
    for fold,(fit,held) in enumerate(GroupKFold(5).split(M['old'],Y,owners)):
        resources();progress('crossfit_actual_cascade',fold=fold)
        old=model().fit(M['old'][fit],Y[fit]);new=model().fit(M['new'][fit],Y[fit])
        op=old.predict_proba(AM['old'])[:,1];np_=new.predict_proba(AM['new'])[:,1]
        if not np.allclose(op[held],saved_old_p[held],atol=1e-12,rtol=0):raise ValueError('Baseline OOF recipe changed')
        p={'baseline':op,'full_cC':np_,'band_cC':blend_scores(op,np_)}
        for name in names:probs[name][held]=p[name][held];keeps[name][held]=decide(p[name])[held]
        for t in grid:threshold_keeps[t][held]=decide(p['band_cC'],t)[held]
    if not np.array_equal(keeps['baseline'],saved_old_keep):raise ValueError('Baseline ownership replay differs')
    peer_rows,peer_oof=raw(SCRATCH/'stack_oof_cC.pkl');ordered=[(s,)+r for s in ('dev','tune') for r in rows[data[s][2]]]
    peer_parity=bool(peer_rows==ordered and np.allclose(peer_oof,probs['full_cC'],rtol=0,atol=1e-12))
    metrics={};scores={};comparison={};threshold_metrics={}
    for split,(a,ix,sl,keys) in data.items():
        metrics[split]={};scores[split]={};threshold_metrics[split]={}
        for name in names:
            keep=np.zeros(len(a['y']),bool);keep[ix]=keeps[name][sl];scores[split][name],metrics[split][name]=summary(a,keep)
        comparison[split]=bootstrap(scores[split]['baseline'],scores[split]['band_cC'])
        for t in grid:
            keep=np.zeros(len(a['y']),bool);keep[ix]=threshold_keeps[t][sl];_,threshold_metrics[split][t]=summary(a,keep)
    selected=max(grid,key=lambda t:(threshold_metrics['tune'][t]['macro_f05'],t))
    write(WORK/'metrics.json',dict(metrics=metrics,paired_comparison=comparison,full_cC_peer_oof_reproduced=peer_parity,
        tune_only_threshold_probe=dict(grid=grid,selected=selected,tune=threshold_metrics['tune'],development_at_selected=threshold_metrics['dev'][selected])))
    progress('attributing_remaining_errors_and_regressions')
    a,ix,sl,keys=data['dev'];refs=list(map(str,a['reference_ids']));qi={o:i for i,o in enumerate(refs)};rr=rows[sl]
    db=connect(ROOT/'artifacts/train_store_v2');db.execute('PRAGMA cache_size=-8192')
    gold={o:{r[0] for r in db.execute('SELECT target_id FROM labels WHERE reference_id=?',(o,))} for o in refs}
    if not np.array_equal(a['true_counts'],[len(gold[o]) for o in refs]):raise ValueError('Full truth differs')
    preds={name:{o:set() for o in refs} for name in names}
    for name in names:
        for r,k in zip(rr,keeps[name][sl]):
            if k:preds[name][r[0]].add(r[1])
    newpred=preds['band_cC'];oldpred=preds['baseline'];pairs=[];transitions=Counter()
    for o in refs:
        for t in newpred[o]-oldpred[o]:transitions['added_true' if t in gold[o] else 'added_false']+=1
        for t in oldpred[o]-newpred[o]:transitions['removed_true' if t in gold[o] else 'removed_false']+=1
        pairs += [(o,t,'FN') for t in sorted(gold[o]-newpred[o])]+[(o,t,'FP') for t in sorted(newpred[o]-gold[o])]
    left=lookup(db,'anchors','entity_id',list({o for o,_,_ in pairs}));right=lookup(db,'records','entity_id',list({t for _,t,_ in pairs}))
    tp=np.asarray([len(newpred[o]&gold[o]) for o in refs]);fp=np.asarray([len(newpred[o]-gold[o]) for o in refs]);current,fn_loss,fp_loss=attribution(tp,fp,a['true_counts'])
    routed={r[:2]:i for i,r in enumerate(rr)};records=[];counts=defaultdict(Counter);repairs=defaultdict(lambda:[np.zeros(len(refs),int),np.zeros(len(refs),int)])
    for o,t,kind in pairs:
        l,r=left[o],right[t];j=routed.get((o,t));p=None if j is None else float(probs['band_cC'][sl][j]);op=None if j is None else float(probs['baseline'][sl][j])
        stage='retrieval_missing' if (o,t) not in keys else 'neural_route_excluded' if j is None else 'owner_assignment_removed' if kind=='FN' and p>=.75 else 'stacker_classification'
        lh,rh=house_number(l['address'],l['country']),house_number(r['address'],r['country']);ls,rs=street_component(l['address'],l['country']),street_component(r['address'],r['country']);low=fuzz.token_set_ratio(folded(l['name']),folded(r['name']))<50
        patterns=dict(empty_target_address=missing(r['address']),indic_target_name=any('\u0900'<=c<='\u0dff' for c in r['name']),
            low_name_similarity=low,low_name_same_house_street=bool(low and lh and lh==rh and ls and rs and fuzz.token_sort_ratio(ls,rs)>=90),
            house_digit_deleted=bool(lh and rh and len(lh)==len(rh)+1 and single_digit_change(lh,rh)),small_house_shift=bool(lh and rh and 0<abs(int(lh)-int(rh))<=25),
            website_style_target=any(s in r['name'].lower() for s in ('.com','.net','.org','.in')))
        old_error=(t not in oldpred[o]) if kind=='FN' else (t in oldpred[o]);band='not_routed' if op is None else 'inside' if .05<=op<.98 else 'outside'
        cats=['stage/'+stage,'country/'+l['country'],'band/'+band,'status/'+('persistent' if old_error else 'new_regression')]+['pattern/'+k for k,v in patterns.items() if v]
        for key in cats:counts[key][kind]+=1;repairs[key][kind=='FP'][qi[o]]+=1
        records.append(dict(owner=o,target=t,kind=kind,stage=stage,country=l['country'],patterns=[k for k,v in patterns.items() if v],band=band,
            new_regression=not old_error,baseline_score=op,v6_score=p,reference=dict(name=l['name'],address=l['address']),target_record=dict(name=r['name'],address=r['address'])))
    db.close();groups={k:dict(counts[k],oracle_repair_gain=float((score(tp+fn,fp-f,a['true_counts'])-current).mean())) for k,(fn,f) in repairs.items()}
    write(WORK/'development_errors.json',records)
    write(WORK/'report.json',dict(complete=True,metrics=metrics,paired_comparison=comparison,development_transitions=transitions,development_remaining_groups=groups,
        fn_share_of_loss=float(fn_loss.sum()/(fn_loss+fp_loss).sum()),full_cC_peer_oof_reproduced=peer_parity,
        threshold_probe_selected_on_tune=selected,threshold_probe_development=threshold_metrics['dev'][selected],
        inputs_sha256=bound,seconds=time.monotonic()-started,limitations=['Reused US/India owner-grouped OOF, no French labels.',
        'Cascade replay uses held-out baseline routing; underlying encoders fixed. Rival set bounded.',
        'Evaluation encoder autocast versus deployment FP16 weights remains a precision caveat.',
        'No leaderboard forecast, new submission, threshold deployment or peer-file changes.']))
    for path,h in bound.items():
        if digest(path)!=h:raise ValueError('Input changed during audit: '+path)
    np.savez_compressed(WORK/'decisions.npz',**{k+'_keep':v for k,v in keeps.items()},**{k+'_p':v for k,v in probs.items()})
    write(WORK/'completion.json',dict(complete=True,report_sha256=digest(WORK/'report.json'),errors_sha256=digest(WORK/'development_errors.json')))
    progress('complete')


if __name__=='__main__':run()
