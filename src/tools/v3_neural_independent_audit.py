"""Independent fitting audit of local CE artifacts, with complete-owner metrics.

No GPU, test writes, or changes to the other session's scripts. The existing dev
set has been explored before: this is not a pristine final audit. Stacker fitting
uses only two thirds of tune owners; selection uses its remaining third, never
dev labels. Rival coverage is the original bounded 100k neighborhood.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from v3_owner_assignment import load_text,text_logit,winner
from v3_transfer_data import ROOT,PILOT,digest,write,bootstrap

SCRATCH=Path('work/ce')
STRUCT=ROOT/'artifacts/improvement_v3/structural_v1'


def lg(p):
    p=np.clip(np.asarray(p),1e-6,1-1e-6)
    return np.log(p/(1-p))


def blend(p,ce,w):return 1/(1+np.exp(-((1-w)*lg(p)+w*lg(ce))))


def summary(a,accepted):
    n=len(a['reference_ids'])
    k=np.bincount(a['query_ids'][accepted],minlength=n)
    tp=np.bincount(a['query_ids'][accepted&(a['y']==1)],minlength=n)
    den=a['true_counts']+4*k
    scores=np.divide(5*tp,den,out=np.ones(n,dtype=float),where=den!=0)
    return scores,dict(macro_f05=float(scores.mean()),tp=int(tp.sum()),fp=int(k.sum()-tp.sum()),
        fn=int(a['true_counts'].sum()-tp.sum()),
        countries={str(c):float(scores[a['countries']==c].mean()) for c in np.unique(a['countries'])})


def run(tag,output,ablations=False,extra_root=None,rival_scores=None,training_root=None,ce_root=None,rival_rows=None,rival_ops=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();inputs={};data={}
    def track(path):
        path=Path(path);inputs[str(path)]=digest(path);return path
    extra_manifest=None
    if extra_root:
        extra_root=Path(extra_root);extra_manifest=json.loads(track(extra_root/'manifest.json').read_text())
        if not extra_manifest['complete']:raise ValueError('Extra features incomplete')
    def extra(split):
        info=extra_manifest['inputs'][split];path=extra_root/f'{split}.npy'
        raw=Path(info.get('path',SCRATCH/f'{split}.pkl'))
        if digest(raw)!=info['sha256'] or digest(path)!=info['features_sha256']:
            raise ValueError('Extra-feature alignment changed')
        return np.load(track(path),allow_pickle=False)
    text_model=load_text(track(STRUCT/'text_scorer.json'))
    rule=json.loads(track(STRUCT/'owner_eval/policy.json').read_text())['selected']
    # Read IDs only: don't retain the large JSON record dictionaries.
    ids=[json.loads(line)['id'] for line in track(PILOT/'data/records.jsonl').open()]
    full_keys=set();all_rows=[];mats=[];pair_slices={};array_slices={}
    for split in ('tune','dev'):
        with np.load(track(PILOT/'data'/f'{split}.npz'),allow_pickle=False) as z:
            a={k:z[k] for k in ('reference_ids','query_ids','y','true_counts','countries','baseline_100000','baseline_200000','left','right')}
        rows=pickle.load(track(SCRATCH/f'{split}.pkl').open('rb'))
        ce=np.load(track(Path(ce_root or SCRATCH/tag)/f'{split}_pce.npy'),allow_pickle=False)
        ops=np.load(track(SCRATCH/f'{split}.pkl.ops.npy'),allow_pickle=False)
        if len(ce)!=len(rows) or ops.shape!=(len(rows),28):raise ValueError('CE/feature alignment mismatch')
        if not np.isfinite(ce).all() or ((ce<0)|(ce>1)).any():raise ValueError('Invalid CE probabilities')
        keys={(ids[int(l)],ids[int(r)]):i for i,(l,r) in enumerate(zip(a['left'],a['right']))}
        if len(keys)!=len(a['y']):raise ValueError('Duplicate original candidate')
        ix=np.asarray([keys[(r[0],r[1])] for r in rows],dtype=np.int64)
        if len(set(ix))!=len(ix):raise ValueError('Duplicate routed candidate')
        expected=np.flatnonzero(a['baseline_100000']>=.01)
        if not np.array_equal(np.sort(ix),expected):raise ValueError('Route is not complete p100k>=.01')
        if not np.array_equal(a['y'][ix],np.asarray([r[2] for r in rows])):raise ValueError('Labels misaligned')
        if not np.allclose(a['baseline_100000'][ix],[r[4] for r in rows],atol=1e-8,rtol=0):
            raise ValueError('Base probabilities misaligned')
        start=len(all_rows);all_rows.extend(rows);pair_slices[split]=slice(start,len(all_rows))
        data[split]=dict(a=a,ix=ix,ce=ce,ops=ops,rows=rows)
        full_keys.update(keys)
        matrix=np.column_stack((ops,lg(a['baseline_100000'][ix]),lg(ce))).astype(np.float32)
        if extra_manifest:matrix=np.column_stack((matrix,extra(split)))
        mats.append(matrix)
    custom_rivals=rival_rows is not None
    original_rivals=pickle.load(track(SCRATCH/'rivals.pkl').open('rb'))
    old_rival_keys={(r[0],r[1]) for r in original_rivals}
    rival_rows=pickle.load(track(rival_rows).open('rb')) if custom_rivals else original_rivals
    rival_ce=np.load(track(rival_scores or SCRATCH/f'rivals.pkl.{tag}.npy'),allow_pickle=False)
    rival_ops=np.load(track(rival_ops or SCRATCH/'rivals.pkl.ops.npy'),allow_pickle=False)
    if len(rival_ce)!=len(rival_rows) or rival_ops.shape!=(len(rival_rows),28):raise ValueError('Rival scores misaligned')
    if not np.isfinite(rival_ce).all() or ((rival_ce<0)|(rival_ce>1)).any():raise ValueError('Invalid rival CE probabilities')
    aux=np.asarray([i for i,r in enumerate(rival_rows) if (r[0],r[1]) not in full_keys])
    rs=len(all_rows);all_rows.extend(rival_rows[i] for i in aux)
    matrix=np.column_stack((rival_ops[aux],lg([rival_rows[i][4] for i in aux]),lg(rival_ce[aux]))).astype(np.float32)
    if extra_manifest:matrix=np.column_stack((matrix,extra('rivals')[aux]))
    mats.append(matrix)
    M=np.vstack(mats);base=np.asarray([r[4] for r in all_rows]);ce=np.r_[data['tune']['ce'],data['dev']['ce'],rival_ce[aux]]
    text=np.asarray([text_logit({'name':r[5],'address':r[6]}, {'name':r[7],'address':r[8]},text_model) for r in all_rows])
    groups=defaultdict(list)
    for i,r in enumerate(all_rows):groups[r[1]].append(i)
    groups=[np.asarray(sorted(v,key=lambda i:all_rows[i][0])) for v in groups.values() if len(v)>1]
    # Auxiliary labels are never read for fitting, selection or the metric.
    old_eligible=np.asarray([i<rs or (r[0],r[1]) in old_rival_keys for i,r in enumerate(all_rows)])
    def accepted(scores,threshold,assign=True,original_coverage=False):
        # The test GPU route is p100k>=.01 for every S1, including rival owners.
        # Lower-base auxiliary scores may exist for diagnostics, but cannot be
        # admitted as claimants when their test equivalents are never scored.
        keep=(scores>=threshold)&(base>=.01)
        if original_coverage:keep&=old_eligible
        if assign:
            for group in groups:
                ix=group[keep[group]]
                if len(ix)<2:continue
                chosen=winner(scores[ix],text[ix],**rule)
                keep[ix]=False
                if chosen is not None:keep[ix[chosen]]=True
        return keep
    def result(split,keep):
        d=data[split];full=np.zeros(len(d['a']['y']),dtype=bool)
        full[d['ix']]=keep[pair_slices[split]]
        return summary(d['a'],full)
    # First prove exact replay of the existing public-best policy.
    baseline_keep=accepted(base,.76,original_coverage=True)
    old,old_report=result('dev',baseline_keep)
    expected=json.loads(track(STRUCT/'owner_eval/report.json').read_text())['splits']['dev']['macro_f05']
    if abs(old_report['macro_f05']-expected)>1e-12:
        raise ValueError(f'100k owner replay mismatch: {old_report["macro_f05"]} vs {expected}')
    refs=data['tune']['a']['reference_ids']
    calibration=np.asarray([int(hashlib.sha256(str(o).encode()).hexdigest()[:8],16)%3==0 for o in refs])
    train_mask=~calibration[data['tune']['a']['query_ids'][data['tune']['ix']]]
    if not train_mask.any() or not calibration.any():raise ValueError('Empty train/calibration split')
    Y=np.asarray([r[2] for r in data['tune']['rows']])
    fit_X=M[pair_slices['tune']][train_mask];fit_y=Y[train_mask]
    fit_owners=int((~calibration).sum());fit_source='two thirds of tune owners'
    if training_root:
        from v3_neural_test_pipeline import feature_functions,edit_matrix
        training_root=Path(training_root);fresh=json.loads(track(training_root/'completion.json').read_text())
        if not fresh['complete'] or not fresh['training_only'] or not fresh['cpu_models_out_of_sample_by_left_owner']:
            raise ValueError('Fresh stack training data incomplete')
        for name,value in fresh['output_sha256'].items():
            if digest(training_root/name)!=value:raise ValueError('Fresh training data changed')
        fr=pickle.load(track(training_root/'train5k.pkl').open('rb'))
        cp=np.load(track(training_root/f'train5k.{tag}.npy'),allow_pickle=False)
        ce_receipt=json.loads(track(training_root/f'train5k.{tag}.receipt.json').read_text())
        if (not ce_receipt['complete'] or ce_receipt['input_sha256']!=digest(training_root/'train5k.pkl')
                or ce_receipt['probabilities_sha256']!=digest(training_root/f'train5k.{tag}.npy')
                or ce_receipt['checkpoint_sha256']!=digest(SCRATCH/tag/'model.pt')):
            raise ValueError('Fresh training CE provenance changed')
        if cp.shape!=(len(fr),) or not np.isfinite(cp).all() or ((cp<0)|(cp>1)).any():
            raise ValueError('Invalid fresh training CE probabilities')
        known=set(data['tune']['a']['reference_ids'])|set(data['dev']['a']['reference_ids'])
        if {r[0] for r in fr}&known:raise ValueError('Fresh training overlaps tune/development')
        fit_X=np.column_stack((edit_matrix(fr,feature_functions()),lg([r[4] for r in fr]),lg(cp))).astype(np.float32)
        if extra_manifest:
            if extra_manifest['inputs']['train5k']['sha256']!=digest(training_root/'train5k.pkl'):
                raise ValueError('Extra fresh features belong to different raw training rows')
            fresh_extra=extra('train5k')
            if fresh_extra.shape!=(len(fr),len(extra_manifest['names'])):
                raise ValueError('Fresh extra-feature dimensions changed')
            fit_X=np.column_stack((fit_X,fresh_extra))
        fit_y=np.asarray([r[2] for r in fr]);fit_owners=fresh['owners']
        fit_source='new train-fold owners outside the 200k first-stage/CE left-owner training selection'
        # Keep the SAME held-out calibration owners for comparisons with the
        # earlier stacker, which was fitted on the other two thirds of tune.
    predictions={f'blend_{w}':blend(base,ce,w) for w in (0.,.5,.8,1.)}
    specs=[(f'ops_ce_leaves{leaves}',leaves,list(range(30)),[14]) for leaves in (15,31)]
    if ablations:
        specs.extend([('ops_base_no_ce',15,list(range(29)),[14]),
                      ('ops_ce_no_base',15,list(range(28))+[29],[14]),
                      ('base_ce_no_ops',15,[28,29],None)])
    if extra_manifest:
        specs.append(('ops_ce_extra',15,list(range(M.shape[1])),[14]))
        if extra_manifest['names']==[f'{view}_{kind}' for view in ('name','ascii_name','address') for kind in ('tfidf_cosine','aligned_fuzzy_cosine')]:
            specs.extend([('ops_ce_exact_cosine',15,list(range(30))+[30,32,34],[14]),
                          ('ops_ce_soft_cosine',15,list(range(30))+[31,33,35],[14])])
    for key,leaves,columns,categorical in specs:
        m=HistGradientBoostingClassifier(max_iter=350,learning_rate=.04,max_leaf_nodes=leaves,
            min_samples_leaf=25,l2_regularization=1.,categorical_features=categorical,random_state=20260926,
            early_stopping=False)
        m.fit(fit_X[:,columns],fit_y)
        predictions[key]=m.predict_proba(M[:,columns])[:,1]
        joblib.dump(m,output/f'{key}.joblib')
    grid=[]
    for name,scores in predictions.items():
        for threshold in (.5,.6,.65,.7,.76,.8,.85,.9,.95):
            keep=accepted(scores,threshold)
            s,_=result('tune',keep)
            grid.append(dict(name=name,threshold=threshold,calibration_macro_f05=float(s[calibration].mean())))
    # Deterministic tie preference is conservative threshold, then simple blend.
    selected=max(grid,key=lambda r:(r['calibration_macro_f05'],r['threshold'],r['name'].startswith('blend')))
    write(output/'selection_before_development.json',dict(selected=selected,grid=grid,
        stack_fit_owners=fit_owners,stack_fit_source=fit_source,calibration_owners=int(calibration.sum()),
        dev_labels_used_for_fitting=False,dev_labels_used_for_selection=False))
    reports={}
    for name,scores in predictions.items():
        # Per-model threshold selected on calibration, not development.
        choice=max((r for r in grid if r['name']==name),key=lambda r:(r['calibration_macro_f05'],r['threshold']))
        keep=accepted(scores,choice['threshold']);new,metrics=result('dev',keep)
        raw,raw_metrics=result('dev',accepted(scores,choice['threshold'],False))
        d=data['dev'];oldf=baseline_keep[pair_slices['dev']];newf=keep[pair_slices['dev']]
        route_y=np.asarray([r[2] for r in d['rows']])
        slices={
            'empty_target_address':np.asarray([not r[8].strip() for r in d['rows']]),
            'indic_target_name':np.asarray([any('\u0900'<=c<='\u0dff' for c in r[7]) for r in d['rows']]),
            'house_digit_drop':d['ops'][:,23]>0,
        }
        reports[name]=dict(selection=choice,development=metrics,without_owner=raw_metrics,
            comparison_to_100k_owner=bootstrap(old,new),
            changes=dict(recovered_true=int((~oldf&newf&(route_y==1)).sum()),
                lost_true=int((oldf&~newf&(route_y==1)).sum()),added_false=int((~oldf&newf&(route_y==0)).sum()),
                removed_false=int((oldf&~newf&(route_y==0)).sum())),
            overlapping_error_slices={k:dict(recovered_true=int((m&~oldf&newf&(route_y==1)).sum()),
                lost_true=int((m&oldf&~newf&(route_y==1)).sum()),
                remaining_missed=int((m&~newf&(route_y==1)).sum()),
                false_positive=int((m&newf&(route_y==0)).sum())) for k,m in slices.items()})
    for path,value in inputs.items():
        if digest(path)!=value:raise ValueError('An input changed during audit: '+path)
    _,widened_baseline=result('dev',accepted(base,.76))
    report=dict(complete=True,tag=tag,ce_root=str(ce_root) if ce_root else None,baseline_replay=old_report,
        widened_rival_coverage=custom_rivals,baseline_with_current_rival_coverage=widened_baseline,selected=selected,ablations=ablations,
        feature_columns={key:columns for key,_,columns,_ in specs},
        extra_features=extra_manifest['names'] if extra_manifest else [],
        models=reports,inputs_sha256=inputs,seconds=time.monotonic()-started,
        limitations=['Existing development was previously inspected; not a pristine final audit.',
            ('Rivals use the supplied widened bounded cache; not exhaustive S1 competition.' if custom_rivals else
             'Rivals come from old 100k accepted-target neighborhoods. Newly recovered targets have incomplete competition.'),
            'Every accepted rival also satisfies the deployed p100k>=.01 neural route.',
            'Text score is computed for every focal pair, never replaced by a missing zero.',
            'No French labels; no country feature; no promised leaderboard score.',
            'Ops features were proposed after earlier development inspection.',
            'All owners, zero-match owners and retrieval misses remain in denominators.'])
    write(output/'report.json',report)
    print(json.dumps(dict(selected=selected,baseline=old_report,development=reports[selected['name']]['development'],
        comparison=reports[selected['name']]['comparison_to_100k_owner']),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ablations',action='store_true')
    p.add_argument('--extra-root',type=Path)
    p.add_argument('--rival-scores',type=Path)
    p.add_argument('--training-root',type=Path)
    p.add_argument('--ce-root',type=Path)
    p.add_argument('--rival-rows',type=Path);p.add_argument('--rival-ops',type=Path)
    run(**vars(p.parse_args()))
