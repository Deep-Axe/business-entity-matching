"""Recalibrate only owner tie handling on a frozen neural model.

Wait for precision-matched scores and wider rivals. Prove original-policy replay,
keep classifier/threshold fixed, select owner rules on calibration owners only,
then report reused development. No automatic test rescoring or submission edits.
"""
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import pickle
import time

import joblib
import numpy as np

from v3_frozen_widened_audit import decisions
from v3_neural_independent_audit import blend, lg, summary
from v3_neural_test_pipeline import SCRATCH, STRUCT, read, resources
from v3_owner_assignment import load_text, text_logit
from v3_transfer_data import ROOT, PILOT, digest, write, bootstrap


def choose_strict(candidates):
    best = candidates[0]
    for candidate in candidates[1:]:
        if candidate['calibration_macro_f05'] > best['calibration_macro_f05'] + 1e-12:
            best = candidate
    return best


def run(audit, widened, output):
    audit, widened, output = map(Path, (audit, widened, output))
    output.mkdir(parents=True, exist_ok=True)
    end = time.monotonic() + 10*3600
    while not all(p.exists() for p in (audit/'report.json', widened/'union.json')):
        if time.monotonic() >= end:
            raise TimeoutError('Finite owner-margin pilot waiting window ended')
        write(output/'progress.json', dict(stage='waiting_for_precision_audit_and_wider_rivals', ready_for_submission=False))
        time.sleep(30)
    if (output/'report.json').exists():
        raise FileExistsError('This owner-margin pilot already reported development')
    resources(); report = read(audit/'report.json'); selected = report['selected']; name = selected['name']
    for path, expected in report['inputs_sha256'].items():
        if digest(path) != expected:
            raise ValueError('Frozen neural audit input changed: '+path)
    if not name.startswith('blend_') and report['feature_columns'][name] != list(range(30)):
        raise ValueError('Owner-margin screen requires the frozen 30-feature neural stacker')
    ce_root = Path(report.get('ce_root') or SCRATCH/report['tag'])
    union = read(widened/'union.json')
    if not union['complete']:
        raise ValueError('Wider rivals incomplete')
    for name_, expected in union['outputs_sha256'].items():
        if digest(widened/name_) != expected:
            raise ValueError('Wider-rival artifact changed')
    import json
    ids = [json.loads(line)['id'] for line in (PILOT/'data/records.jsonl').open()]
    rows = []; matrices = []; data = {}; full_keys = set()
    for split in ('tune', 'dev'):
        with np.load(PILOT/'data'/f'{split}.npz', allow_pickle=False) as z:
            a = {k:z[k] for k in ('reference_ids','query_ids','y','true_counts','countries','left','right','baseline_100000')}
        keys = {(ids[int(l)],ids[int(r)]):i for i,(l,r) in enumerate(zip(a['left'],a['right']))}
        raw = pickle.load((SCRATCH/f'{split}.pkl').open('rb')); ix = np.asarray([keys[r[:2]] for r in raw])
        if len(set(ix)) != len(ix) or not np.array_equal(np.sort(ix),np.flatnonzero(a['baseline_100000'] >= .01)):
            raise ValueError('Focal route lost or duplicated candidates')
        if not np.array_equal(a['y'][ix],[r[2] for r in raw]):
            raise ValueError('Focal labels misaligned')
        base = np.asarray([r[4] for r in raw])
        if not np.allclose(base,a['baseline_100000'][ix],rtol=0,atol=1e-8):
            raise ValueError('Focal CPU context changed')
        ce = np.load(ce_root/f'{split}_pce.npy',allow_pickle=False)
        ops = np.load(SCRATCH/f'{split}.pkl.ops.npy',allow_pickle=False)
        start=len(rows); rows.extend(raw); data[split]=(a,ix,slice(start,len(rows))); full_keys.update(keys)
        matrices.append(np.column_stack((ops,lg(base),lg(ce))))
    focal_count = len(rows)
    old_keys = {r[:2] for r in pickle.load((SCRATCH/'rivals.pkl').open('rb'))}
    rivals = pickle.load((widened/'rivals.pkl').open('rb'))
    if not old_keys.issubset({r[:2] for r in rivals}):
        raise ValueError('Wider cache lost original claimants')
    ri = np.asarray([i for i,r in enumerate(rivals) if r[:2] not in full_keys])
    rows.extend(rivals[i] for i in ri)
    matrices.append(np.column_stack((np.load(widened/'rivals_ops.npy')[ri],lg([rivals[i][4] for i in ri]),
                                     lg(np.load(widened/'rivals_pce.npy')[ri]))))
    matrix = np.vstack(matrices).astype(np.float32)
    if len({r[:2] for r in rows}) != len(rows) or not np.isfinite(matrix).all():
        raise ValueError('Duplicate or invalid neural rows')
    base = np.asarray([r[4] for r in rows])
    if name.startswith('blend_'):
        ce = 1/(1+np.exp(-matrix[:,29])); scores = blend(base,ce,float(name.split('_')[1])); model_sha = None
    else:
        model_path = audit/(name+'.joblib'); model = joblib.load(model_path)
        scores = model.predict_proba(matrix)[:,1]; model_sha=digest(model_path)
    if scores.shape != (len(rows),) or not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
        raise ValueError('Invalid frozen scores')
    text_model=load_text(STRUCT/'text_scorer.json')
    text=np.asarray([text_logit({'name':r[5],'address':r[6]},{'name':r[7],'address':r[8]},text_model) for r in rows])
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        groups[r[1]].append(i)
    groups=[np.asarray(sorted(g,key=lambda i:rows[i][0])) for g in groups.values() if len(g)>1]
    old_eligible=np.asarray([i<focal_count or r[:2] in old_keys for i,r in enumerate(rows)])
    all_eligible=np.ones(len(rows),bool)
    calibration=np.asarray([int(hashlib.sha256(str(o).encode()).hexdigest()[:8],16)%3==0 for o in data['tune'][0]['reference_ids']])
    rule=read(STRUCT/'owner_eval/policy.json')['selected']
    def evaluate(candidate,split,eligible=all_eligible):
        keep=decisions(scores,base,text,groups,selected['threshold'],candidate,eligible)
        a,ix,sl=data[split]; accepted=np.zeros(len(a['y']),bool); accepted[ix]=keep[sl]
        return summary(a,accepted)
    original_dev, original_metrics=evaluate(rule,'dev',old_eligible)
    original_tune,_=evaluate(rule,'tune',old_eligible)
    if abs(original_metrics['macro_f05']-report['models'][name]['development']['macro_f05'])>1e-12:
        raise ValueError('Original neural owner policy did not replay exactly')
    if abs(original_tune[calibration].mean()-selected['calibration_macro_f05'])>1e-12:
        raise ValueError('Original calibration policy did not replay exactly')
    base_tune,_=evaluate(rule,'tune')
    grid=[dict(rule=rule,calibration_macro_f05=float(base_tune[calibration].mean()))]
    for alpha in (0.,.1,.25,.5,1.):
        for floor in (0.,.5,.8):
            for margin in (0.,.1,.25,.5,1.,1.5,2.,3.):
                candidate=dict(alpha=alpha,floor=floor,margin=margin)
                values,_=evaluate(candidate,'tune')
                grid.append(dict(rule=candidate,calibration_macro_f05=float(values[calibration].mean())))
    choice=choose_strict(grid)
    identity=dict(audit_sha256=digest(audit/'report.json'),model_sha256=model_sha,
                  union_sha256=digest(widened/'union.json'),code_sha256=digest(__file__),
                  old_policy_sha256=digest(STRUCT/'owner_eval/policy.json'),threshold=selected['threshold'])
    write(output/'selection_before_development.json',dict(identity=identity,selected=choice,grid=grid,
        no_classifier_refitting=True,selection_owners='Original held-out calibration third of tune only'))
    old,old_metrics=evaluate(rule,'dev'); new,new_metrics=evaluate(choice['rule'],'dev')
    write(output/'report.json',dict(complete=True,identity=identity,selected=choice,
        calibration_gain=choice['calibration_macro_f05']-grid[0]['calibration_macro_f05'],
        original_coverage_replay=original_metrics,widened_baseline=old_metrics,widened_selected=new_metrics,
        comparison=bootstrap(old,new),country_delta={c:new_metrics['countries'][c]-v for c,v in old_metrics['countries'].items()},
        ready_for_submission=False,full_test_scored=False,
        note='Reused development; fixed classifier/threshold; no French labels. Do not promote from this diagnostic alone.'))
    write(output/'progress.json',dict(stage='complete',ready_for_submission=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,default=ROOT/'artifacts/duf-fel_full_output_v4_neural1300k_peerhalf_v1/evaluation')
    p.add_argument('--widened',type=Path,default=ROOT/'artifacts/improvement_v3/widened_rival_neural_v1')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/improvement_v3/neural_owner_margin_peerhalf_v1')
    run(**vars(p.parse_args()))
