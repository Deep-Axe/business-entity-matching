"""Reconstruct the submitted 0.976 recipe and audit out-of-fold errors.

No external labels, new policy selection, peer-file mutation or test rescore.
Verify reconstructed final-fit probabilities on a retained test partition,
then reproduce the original OOF protocol and also report coherent per-fold
ownership using actual text scores and the deployed neural-route constraint.
"""
from collections import Counter, defaultdict
import json
import pickle
import time

import joblib
import numpy as np
from rapidfuzz import fuzz
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from ber.store import connect, lookup
from v3_neural_independent_audit import SCRATCH, STRUCT, lg, summary
from v3_neural_test_pipeline import resources
from v3_owner_assignment import load_text, text_logit, winner
from v3_owner_error_analysis import attribution, missing, score
from v3_structural_features import folded, house_number, street_component, single_digit_change, leet_view
from v3_transfer_data import ROOT, PILOT, digest, write

WORK = ROOT / 'artifacts/improvement_v3/public0976_error_audit_v2'
TAG = 'e5_cont1300k'
THRESHOLD = .75


def model():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=.04,
        max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.,
        categorical_features=[14], random_state=0)


def features(rows, ops, ce, frequency):
    matrix = np.column_stack((ops, lg([r[4] for r in rows]), lg(ce), frequency)).astype(np.float32)
    if matrix.shape != (len(rows), 33) or not np.isfinite(matrix).all():
        raise ValueError('Submitted 33-feature recipe changed')
    return matrix


def run():
    WORK.mkdir(parents=True, exist_ok=False); started = time.monotonic(); bound = {}
    def track(path):
        bound[str(path)] = digest(path); return path
    def raw(path):
        return pickle.load(track(path).open('rb'))
    def arr(path):
        return np.load(track(path), allow_pickle=False)
    def progress(stage, **kw):
        write(WORK/'progress.json', dict(stage=stage, seconds=time.monotonic()-started, **kw))
    resources(); submitted = SCRATCH/'sub_v4_stack_all/matching_results.tsv'
    expected_hash = '5a0be554e60f88e763de95a5b4cc8f30f7cee352067ba24c16947e6319752118'
    if digest(track(submitted)) != expected_hash:
        raise ValueError('Public-best TSV changed')
    declaration = json.loads(track(SCRATCH/'sub_v4_stack_all/report.json').read_text())
    if declaration['score'] != 'stack_e5_cont1300k_nocty_nf2' or declaration['t'] != THRESHOLD:
        raise ValueError('Public-best model recipe changed')
    track(ROOT/'tools/v3_owner_assignment.py'); track(ROOT/'tools/v3_structural_features.py')
    track('src/neural/stack_eval.py')
    ids = [json.loads(line)['id'] for line in track(PILOT/'data/records.jsonl').open()]
    rows, matrices, spans, arrays, full_keys = [], [], {}, {}, set()
    for split in ('dev', 'tune'):  # Original fold order matters for reproducibility.
        r = raw(SCRATCH/(split+'.pkl')); lo = len(rows); rows.extend(r); spans[split] = slice(lo, len(rows))
        ce = arr(SCRATCH/TAG/(split+'_pce.npy'))
        matrices.append(features(r, arr(SCRATCH/(split+'.pkl.ops.npy')), ce,
            np.column_stack((arr(SCRATCH/(split+'.pkl.nf.npy')),arr(SCRATCH/(split+'.pkl.nf2.npy'))))))
        with np.load(track(PILOT/'data'/(split+'.npz')), allow_pickle=False) as z:
            a = {k:z[k] for k in ('reference_ids','query_ids','true_counts','countries','y','left','right','baseline_100000')}
        keys = {(ids[int(l)],ids[int(t)]):i for i,(l,t) in enumerate(zip(a['left'],a['right']))}
        ix = np.asarray([keys[v[:2]] for v in r])
        if not np.array_equal(np.sort(ix), np.flatnonzero(a['baseline_100000']>=.01)):
            raise ValueError('OOF neural route coverage mismatch')
        if not np.array_equal(a['y'][ix], [v[2] for v in r]):
            raise ValueError('OOF labels misaligned')
        arrays[split] = (a, ix, ce, keys); full_keys.update(keys)
    M = np.vstack(matrices); Y = np.asarray([r[2] for r in rows]); owners = np.asarray([r[0] for r in rows])
    rivals = raw(SCRATCH/'rivals.pkl')
    RM = features(rivals, arr(SCRATCH/'rivals.pkl.ops.npy'), arr(SCRATCH/('rivals.pkl.'+TAG+'.npy')),
                  np.column_stack((arr(SCRATCH/'rivals.pkl.nf.npy'),arr(SCRATCH/'rivals.pkl.nf2.npy'))))
    rule = json.loads(track(STRUCT/'owner_eval/policy.json').read_text())['selected']
    text_model = load_text(track(STRUCT/'text_scorer.json'))
    progress('reconstructing_final_fit_and_test_probability_parity')
    final = model().fit(M, Y)
    test = SCRATCH/'test/part_000.pkl'; tr = raw(test)
    test_m = features(tr, arr(str(test)+'.ops.npy'), arr(str(test)+'.'+TAG+'.npy'),
        np.column_stack((arr(str(test)+'.nf.npy'),arr(str(test)+'.nf2.npy'))))
    saved = arr(str(test)+'.stack_'+TAG+'_nocty_nf2.npy')
    generated = final.predict_proba(test_m)[:,1].astype(np.float32)
    delta = float(np.max(np.abs(generated-saved)))
    write(WORK/'test_recipe_parity.json', dict(partition=0, pairs=len(tr), maximum_probability_delta=delta,
        exact=np.array_equal(generated,saved), passes=delta<=1e-7, input_precision='Original peer training autocast, test FP16 weights'))
    if delta > 1e-7:
        raise ValueError('Cannot attribute this model to the submitted recipe: final-fit probability replay differs')
    joblib.dump(final,WORK/'reconstructed_final.joblib'); del final,tr,test_m,saved,generated
    focal = len(rows); aux = np.asarray([i for i,r in enumerate(rivals) if r[:2] not in full_keys])
    all_rows = rows+[rivals[i] for i in aux]; all_M = np.vstack((M, RM[aux]))
    base = np.asarray([r[4] for r in all_rows])
    text = np.asarray([text_logit({'name':r[5],'address':r[6]}, {'name':r[7],'address':r[8]},text_model) for r in all_rows])
    groups = defaultdict(list)
    for i,r in enumerate(all_rows): groups[r[1]].append(i)
    groups = [np.asarray(sorted(v,key=lambda i:all_rows[i][0])) for v in groups.values() if len(v)>1]
    def decide(p):
        before = (p>=THRESHOLD)&(base>=.01); keep=before.copy()
        for group in groups:
            ix=group[keep[group]]
            if len(ix)<2:continue
            k=winner(p[ix],text[ix],**rule);keep[ix]=False
            if k is not None:keep[ix[k]]=True
        return before,keep
    oof=np.zeros(focal); coherent=np.zeros(focal,bool); before=np.zeros(focal,bool)
    folds=[]; target_fold={}; rival_by_fold=[]
    for k,(fit,held) in enumerate(GroupKFold(5).split(M,Y,owners)):
        resources();progress('cross_fit_and_consistent_owner_decisions',fold=k)
        if set(owners[fit])&set(owners[held]):raise ValueError('OOF left-owner leakage')
        m=model().fit(M[fit],Y[fit]);p=m.predict_proba(all_M)[:,1];oof[held]=p[held]
        b,a=decide(p);before[held]=b[held];coherent[held]=a[held]
        rival_by_fold.append(m.predict_proba(RM)[:,1])
        for i in held:target_fold[rows[i][1]]=k
        folds.append(dict(fold=k,fit_owners=len(set(owners[fit])),heldout_owners=len(set(owners[held]))))
    original_rival=np.asarray([rival_by_fold[target_fold.get(r[1],0)][i] for i,r in enumerate(rivals)])
    claims=defaultdict(list)
    for r,p in zip(rivals,original_rival):
        if p>=THRESHOLD:claims[r[1]].append((r[0],p,r[9]))
    legacy=oof>=THRESHOLD
    for i,r in enumerate(rows):
        if not legacy[i]:continue
        cl=claims.get(r[1],[]);other=[c for c in cl if c[0]!=r[0]]
        if not other:continue
        ft=next((c[2] for c in cl if c[0]==r[0]),0.)
        candidates=sorted([(r[0],oof[i],ft)]+other,key=lambda c:c[0])
        ch=winner([c[1] for c in candidates],[c[2] for c in candidates],**rule)
        if ch is None or candidates[ch][0]!=r[0]:legacy[i]=False
    saved_rows,saved_oof=raw(SCRATCH/('stack_oof_'+TAG+'.pkl'))
    ordered=[(s,)+r for s in ('dev','tune') for r in rows[spans[s]]]
    retained_match=(saved_rows==ordered and np.array_equal(saved_oof,oof))
    reports={}
    for split,(a,ix,ce,keys) in arrays.items():
        result={}
        for name,mask in [('legacy_peer_protocol',legacy),('coherent_crossfit_ownership',coherent),('without_owner',before)]:
            keep=np.zeros(len(a['y']),bool);keep[ix]=mask[spans[split]];_,result[name]=summary(a,keep)
        reports[split]=result
    # Error attribution includes original retrieval and route misses, not only
    # the 15k routed development pairs passed to the neural model.
    a,ix,ce,keys=arrays['dev']; refs=list(map(str,a['reference_ids']));qi={o:i for i,o in enumerate(refs)}
    db=connect(ROOT/'artifacts/train_store_v2');db.execute('PRAGMA cache_size=-8192')
    gold={o:{r[0] for r in db.execute('SELECT target_id FROM labels WHERE reference_id=?',(o,))} for o in refs}
    if not np.array_equal(a['true_counts'],[len(gold[o]) for o in refs]):raise ValueError('Truth coverage differs')
    raw_dev=rows[spans['dev']];routed={r[:2]:i for i,r in enumerate(raw_dev)}
    error_reports={}
    for protocol,mask in [('legacy_peer_protocol',legacy),('coherent_crossfit_ownership',coherent)]:
        pred={o:set() for o in refs}
        for r,k in zip(raw_dev,mask[spans['dev']]):
            if k:pred[r[0]].add(r[1])
        tp=np.asarray([len(pred[o]&gold[o]) for o in refs]);fp=np.asarray([len(pred[o]-gold[o]) for o in refs])
        current,fn_loss,fp_loss=attribution(tp,fp,a['true_counts']);errors=[]
        for o in refs:
            errors.extend((o,t,'FN') for t in sorted(gold[o]-pred[o]));errors.extend((o,t,'FP') for t in sorted(pred[o]-gold[o]))
        left=lookup(db,'anchors','entity_id',list({o for o,_,_ in errors}));right=lookup(db,'records','entity_id',list({t for _,t,_ in errors}))
        counters=defaultdict(Counter);repairs=defaultdict(lambda:[np.zeros(len(refs),int),np.zeros(len(refs),int)]);records=[]
        for o,t,kind in errors:
            l,r=left[o],right[t];j=routed.get((o,t));p=None if j is None else float(oof[spans['dev']][j])
            stage=('retrieval_missing' if (o,t) not in keys else 'neural_route_excluded' if j is None
                   else 'owner_assignment_removed' if kind=='FN' and p>=THRESHOLD else 'stacker_classification')
            lh,rh=house_number(l['address'],l['country']),house_number(r['address'],r['country'])
            ls,rs=street_component(l['address'],l['country']),street_component(r['address'],r['country'])
            low=fuzz.token_set_ratio(folded(l['name']),folded(r['name']))<50
            patterns=dict(empty_target_address=missing(r['address']),
                indic_target_name=any('\u0900'<=c<='\u0dff' for c in r['name']),
                low_name_similarity=low, low_name_same_house_street=bool(low and lh and lh==rh and ls and rs and fuzz.token_sort_ratio(ls,rs)>=90),
                house_digit_deleted=bool(lh and rh and len(lh)==len(rh)+1 and single_digit_change(lh,rh)),
                small_house_shift=bool(lh and rh and 0<abs(int(lh)-int(rh))<=25),
                website_style_target=any(s in r['name'].lower() for s in ('.com','.net','.org','.in')),
                strong_neural_rejected=bool(j is not None and ce[j]>=.9 and p<THRESHOLD),
                weak_neural_rejected=bool(j is not None and ce[j]<.5 and p<THRESHOLD))
            categories=['stage/'+stage,'country/'+l['country']]+['pattern/'+k for k,v in patterns.items() if v]
            for key in categories:counters[key][kind]+=1;repairs[key][kind=='FP'][qi[o]]+=1
            records.append(dict(owner=o,target=t,kind=kind,stage=stage,patterns=[k for k,v in patterns.items() if v],
                country=l['country'],oof_score=p,ce_score=None if j is None else float(ce[j]),
                reference=dict(name=l['name'],address=l['address']),target_record=dict(name=r['name'],address=r['address'])))
        groups_report={k:dict(counters[k],oracle_repair_macro_gain=float((score(tp+fn,fp-f,a['true_counts'])-current).mean()))
                       for k,(fn,f) in repairs.items()}
        retrieved=np.bincount(a['query_ids'],weights=a['y'],minlength=len(refs));routed_tp=np.zeros(len(refs),int)
        for r in raw_dev:routed_tp[qi[r[0]]]+=r[2]
        error_reports[protocol]=dict(macro_f05=float(current.mean()),tp=int(tp.sum()),fp=int(fp.sum()),fn=int(a['true_counts'].sum()-tp.sum()),
            fn_loss=float(fn_loss.mean()),fp_loss=float(fp_loss.mean()),fn_share_of_loss=float(fn_loss.sum()/(fn_loss+fp_loss).sum()),
            retrieval_oracle_macro=float(score(retrieved,np.zeros(len(refs)),a['true_counts']).mean()),
            routed_oracle_macro=float(score(routed_tp,np.zeros(len(refs)),a['true_counts']).mean()),groups=groups_report)
        write(WORK/(protocol+'_errors.json'),records)
    db.close()
    for name,h in bound.items():
        if digest(name)!=h:raise ValueError('An input changed during the audit: '+name)
    write(WORK/'report.json',dict(complete=True,submitted_tsv_sha256=expected_hash,reported_public_score=.976,
        reported_local_score=.98571,test_partition_recipe_reproduced=True,retained_oof_exact=retained_match,
        regenerated_oof_max_delta=float(np.max(np.abs(saved_oof-oof))) if saved_rows==ordered else None,
        threshold=THRESHOLD,threshold_retuned=False,folds=folds,metrics=reports,development_error_attribution=error_reports,
        inputs_sha256=bound,seconds=time.monotonic()-started,ready_for_submission=False,
        limitations=['Reused US/India OOF data; not a pristine confirmation or French validation.',
        'Final submitted stacker fitted on all tune/dev owners, so its in-sample score is not reported as validation.',
        'Coherent diagnostic scores all rivals with each focal heldout fold model and real text; bounded rival coverage remains.',
        'Pattern repair gains are overlapping oracle ceilings, not additive expected gains.',
        'Partition 0 verifies model recipe, not a claim that every saved test probability has been regenerated.']))
    np.savez_compressed(WORK/'decisions.npz',oof=oof,legacy=legacy,coherent=coherent,before=before)
    progress('complete')


if __name__=='__main__':run()
