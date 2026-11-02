"""Replay an already frozen neural stacker and attribute its remaining errors.

No training, threshold search, or test-label inference. Development is reused and
all 'oracle gains' are diagnostic ceilings, never promised improvements.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import pickle

import joblib
import numpy as np
from rapidfuzz import fuzz

from ber.store import connect, lookup
from v3_neural_test_pipeline import SCRATCH, lg
from v3_owner_assignment import load_text, text_logit, winner
from v3_owner_error_analysis import attribution, score, missing
from v3_structural_features import folded, house_number, street_component, single_digit_change
from v3_transfer_data import ROOT, PILOT, digest, write


def run(audit, rival_scores, output, ce_root=None, extra_root=None):
    audit,output=Path(audit),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    report=json.loads((audit/'report.json').read_text());tag=report['tag'];selected=report['selected']
    for path,value in report['inputs_sha256'].items():
        if digest(path)!=value:raise ValueError('Frozen audit input changed')
    extra_manifest=None
    if extra_root:
        extra_root=Path(extra_root)
        extra_manifest=json.loads((extra_root/'manifest.json').read_text())
        if (not extra_manifest['complete'] or
                report['inputs_sha256'].get(str(extra_root/'manifest.json'))!=digest(extra_root/'manifest.json')):
            raise ValueError('Extra features do not belong to this frozen audit')
    dimensions=30+(len(extra_manifest['names']) if extra_manifest else 0)
    if report['feature_columns'][selected['name']]!=list(range(dimensions)):
        raise ValueError('Diagnostic feature columns differ from the frozen model')
    def extend(matrix,split,ix=None):
        if not extra_manifest:return matrix
        raw=extra_root/f'{split}.npy';info=extra_manifest['inputs'][split]
        if digest(raw)!=info['features_sha256']:
            raise ValueError('Structural features changed')
        values=np.load(raw,allow_pickle=False)
        if ix is not None:values=values[ix]
        if len(values)!=len(matrix):raise ValueError('Structural row alignment differs')
        return np.column_stack((matrix,values))
    if report['inputs_sha256'].get(str(rival_scores))!=digest(rival_scores):
        raise ValueError('Rival scoring does not belong to audit')
    model=joblib.load(audit/(selected['name']+'.joblib'))
    ids=[json.loads(line)['id'] for line in (PILOT/'data/records.jsonl').open()]
    arrays={};full=set();rows=[];matrices=[];spans={}
    for split in ('tune','dev'):
        with np.load(PILOT/'data'/f'{split}.npz',allow_pickle=False) as z:
            a={k:z[k] for k in ('reference_ids','query_ids','y','true_counts','countries','left','right','baseline_100000')}
        arrays[split]=a
        full.update((ids[int(l)],ids[int(r)]) for l,r in zip(a['left'],a['right']))
        r=pickle.load((SCRATCH/f'{split}.pkl').open('rb'));start=len(rows);rows.extend(r);spans[split]=slice(start,len(rows))
        ce_path=Path(ce_root or SCRATCH/tag)/f'{split}_pce.npy'
        if report['inputs_sha256'].get(str(ce_path))!=digest(ce_path):
            raise ValueError('CE precision/checkpoint scores differ from frozen audit')
        ce=np.load(ce_path,allow_pickle=False)
        matrices.append(extend(np.column_stack((np.load(SCRATCH/f'{split}.pkl.ops.npy'),lg([x[4] for x in r]),lg(ce))),split))
    rr=pickle.load((SCRATCH/'rivals.pkl').open('rb'))
    ix=np.asarray([i for i,r in enumerate(rr) if (r[0],r[1]) not in full])
    rows.extend(rr[i] for i in ix)
    matrices.append(extend(np.column_stack((np.load(SCRATCH/'rivals.pkl.ops.npy')[ix],lg([rr[i][4] for i in ix]),lg(np.load(rival_scores)[ix]))),'rivals',ix))
    p=model.predict_proba(np.vstack(matrices).astype(np.float32))[:,1]
    before=(p>=selected['threshold'])&(np.asarray([r[4] for r in rows])>=.01);after=before.copy()
    structure=ROOT/'artifacts/improvement_v3/structural_v1'
    text_model=load_text(structure/'text_scorer.json')
    rule=json.loads((structure/'owner_eval/policy.json').read_text())['selected']
    groups=defaultdict(list)
    for i in np.flatnonzero(before):groups[rows[i][1]].append(int(i))
    for group in groups.values():
        if len(group)<2:continue
        group=sorted(group,key=lambda i:rows[i][0])
        text=[text_logit({'name':rows[i][5],'address':rows[i][6]},
                         {'name':rows[i][7],'address':rows[i][8]},text_model) for i in group]
        best=winner(p[group],text,**rule);after[group]=False
        if best is not None:after[group[best]]=True
    a=arrays['dev'];refs=list(map(str,a['reference_ids']));qi={r:i for i,r in enumerate(refs)}
    prediction={r:set() for r in refs};routed={};raw_accepted=set()
    for i in range(spans['dev'].start,spans['dev'].stop):
        r=rows[i];routed[r[:2]]=i
        if before[i]:raw_accepted.add(r[:2])
        if after[i]:prediction[r[0]].add(r[1])
    full_dev={(ids[int(l)],ids[int(r)]) for l,r in zip(a['left'],a['right'])}
    db=connect(ROOT/'artifacts/train_store_v2');db.execute('PRAGMA cache_size=-8192')
    gold={o:{r[0] for r in db.execute('SELECT target_id FROM labels WHERE reference_id=?',(o,))} for o in refs}
    if not np.array_equal(a['true_counts'],[len(gold[o]) for o in refs]):raise ValueError('Gold coverage mismatch')
    tp=np.asarray([len(prediction[o]&gold[o]) for o in refs]);fp=np.asarray([len(prediction[o]-gold[o]) for o in refs])
    scores,fn_loss,fp_loss=attribution(tp,fp,a['true_counts'])
    expected=report['models'][selected['name']]['development']
    if (abs(scores.mean()-expected['macro_f05'])>1e-12 or tp.sum()!=expected['tp'] or fp.sum()!=expected['fp']):
        raise ValueError('Neural owner replay failed')
    errors=[(o,t,'FN') for o in refs for t in sorted(gold[o]-prediction[o])]
    errors += [(o,t,'FP') for o in refs for t in sorted(prediction[o]-gold[o])]
    anchors=lookup(db,'anchors','entity_id',list({o for o,_,_ in errors}))
    targets=lookup(db,'records','entity_id',list({t for _,t,_ in errors}));db.close()
    counts=defaultdict(Counter);repairs=defaultdict(lambda:[np.zeros(len(refs),int),np.zeros(len(refs),int)])
    error_rows=[]
    for o,t,kind in errors:
        l,r=anchors[o],targets[t];key=(o,t)
        stage=('retrieval_missing' if key not in full_dev else 'neural_route_excluded' if key not in routed
               else 'owner_assignment_removed' if key in raw_accepted and kind=='FN' else 'classifier')
        lh,rh=house_number(l['address'],l['country']),house_number(r['address'],r['country'])
        ls,rs=street_component(l['address'],l['country']),street_component(r['address'],r['country'])
        low=fuzz.token_set_ratio(folded(l['name']),folded(r['name']))<50
        masks=dict(empty_target_address=missing(r['address']),
            indic_target_name=any('\u0900'<=c<='\u0dff' for c in r['name']),
            low_name_similarity=low,
            low_name_same_house_street=bool(low and lh and lh==rh and ls and rs and fuzz.token_sort_ratio(ls,rs)>=90),
            house_digit_deleted=bool(lh and rh and len(lh)==len(rh)+1 and single_digit_change(lh,rh)),
            small_house_shift=bool(lh and rh and 0<abs(int(lh)-int(rh))<=25))
        keys=['stage/'+stage,'country/'+l['country']]
        keys += ['pattern/'+k for k,v in masks.items() if v]
        for k in keys:
            counts[k][kind]+=1;repairs[k][kind=='FP'][qi[o]]+=1
        i=routed.get(key)
        error_rows.append(dict(owner=o,target=t,kind=kind,stage=stage,patterns=[k for k,v in masks.items() if v],
            country=l['country'],score=None if i is None else float(p[i]),
            base_score=None if i is None else float(rows[i][4]),
            reference=dict(name=l['name'],address=l['address']),target_record=dict(name=r['name'],address=r['address'])))
    groups_report={}
    for key,(fn,fs) in repairs.items():
        groups_report[key]=dict(counts[key],oracle_repair_macro_gain=float((score(tp+fn,fp-fs,a['true_counts'])-scores).mean()))
    retrieved=np.bincount(a['query_ids'],weights=a['y'],minlength=len(refs))
    routed_true=np.zeros(len(refs),int)
    for key,i in routed.items():routed_true[qi[key[0]]]+=int(rows[i][2])
    result=dict(complete=True,scope='Previously reused development only; frozen model/policy; no tuning or deployment',
        audit_sha256=digest(audit/'report.json'),selected=selected,development=expected,
        fn_loss=float(fn_loss.mean()),fp_loss=float(fp_loss.mean()),fn_share_of_loss=float(fn_loss.sum()/(fn_loss+fp_loss).sum()),
        retrieval_oracle_macro=float(score(retrieved,np.zeros(len(refs)),a['true_counts']).mean()),
        routed_oracle_macro=float(score(routed_true,np.zeros(len(refs)),a['true_counts']).mean()),
        groups=groups_report,note='Stage groups are exclusive; pattern groups overlap. Oracle gains are separate hypothetical repairs, not additive forecasts. Rival neighborhood is bounded, not exhaustive. France has no labels.')
    write(output/'report.json',result);write(output/'errors.json',error_rows)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('audit','rival-scores','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--ce-root',type=Path);p.add_argument('--extra-root',type=Path)
    run(**vars(p.parse_args()))
