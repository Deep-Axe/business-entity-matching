"""Bounded CPU pilot: expose below-threshold rival evidence to the stacker.

No ownership labels are used in these features. All eligible claims obey the
same p100k>=.01 route as test. Reverse neighborhoods are bounded, not exhaustive;
newly recovered nonempty-address targets can still have missing competition.
The fitting audit selects on tune owners and reports reused development only.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np

from v3_neural_test_pipeline import SCRATCH,STRUCT,read,lg,feature_functions,edit_matrix
from v3_owner_assignment import load_text,text_logit
from v3_transfer_data import ROOT,digest,write

NAMES=['reverse_log_routed_owners','rival_best_cpu','rival_cpu_logit_margin',
       'rival_best_ce','rival_ce_logit_margin','reverse_cpu_log_rank','reverse_ce_log_rank',
       'rival_best_text','rival_text_margin','rival_log_ce_mass']


def neighborhood(rows,ce,text):
    if len(rows)!=len(ce) or len(rows)!=len(text):raise ValueError('Neighborhood scores misaligned')
    if len({r[:2] for r in rows})!=len(rows):raise ValueError('Duplicate neighborhood edges')
    groups=defaultdict(list);out=np.empty((len(rows),len(NAMES)),np.float32)
    base=np.asarray([r[4] for r in rows]);ce=np.asarray(ce);text=np.asarray(text)
    if not np.isfinite(ce).all() or ((ce<0)|(ce>1)).any():raise ValueError('Invalid neighborhood probabilities')
    for i,r in enumerate(rows):groups[r[1]].append(i)
    for group in groups.values():
        eligible=[i for i in group if base[i]>=.01]
        for i in group:
            others=[j for j in eligible if j!=i]
            pc=float(base[others].max()) if others else 0.
            pn=float(ce[others].max()) if others else 0.
            pt=float(text[others].max()) if others else -20.
            out[i]=[np.log1p(len(others)),pc,float(lg(base[i])-lg(pc)),pn,float(lg(ce[i])-lg(pn)),
                np.log1p(sum(base[j]>base[i] for j in others)),np.log1p(sum(ce[j]>ce[i] for j in others)),
                pt,float(text[i]-pt),np.log1p(float(ce[others].sum()))]
    if not np.isfinite(out).all():raise ValueError('Invalid neighborhood features')
    return out


def run(output,tag='e5_cont1300k',ce_root=None,rival_rows=None,rival_scores=None,run_audit=False,audit_reference=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    ce_root=Path(ce_root or SCRATCH/tag);rival_rows=Path(rival_rows or SCRATCH/'rivals.pkl')
    paths={s:SCRATCH/f'{s}.pkl' for s in ('tune','dev')};paths['rivals']=rival_rows
    cp={s:ce_root/f'{s}_pce.npy' for s in ('tune','dev')}
    cp['rivals']=Path(rival_scores or SCRATCH/f'rivals.pkl.{tag}.npy')
    reference=Path(audit_reference or ROOT/'artifacts/improvement_v3/neural_followup1300k_v1/evaluation/report.json')
    audit=read(reference)
    if not audit['complete'] or audit['tag']!=tag:raise ValueError('Wrong source audit')
    verified={Path(p).resolve():h for p,h in audit['inputs_sha256'].items()}
    for path in [*paths.values(),*cp.values()]:
        if verified.get(path.resolve())!=digest(path):raise ValueError('Neighborhood input lacks audited provenance: '+str(path))
    rows_by={s:pickle.load(p.open('rb')) for s,p in paths.items()}
    scores_by={s:np.load(p,allow_pickle=False) for s,p in cp.items()}
    all_rows=[];probs=[];keys={};indices={};duplicates=0;max_duplicate_delta=0.
    for split,rows in rows_by.items():
        if scores_by[split].shape!=(len(rows),):raise ValueError('Neural/raw alignment differs')
        ix=[]
        for row,p in zip(rows,scores_by[split]):
            key=row[:2]
            if key in keys:
                i=keys[key];before=all_rows[i]
                if before[3]!=row[3] or before[5:9]!=row[5:9] or abs(before[4]-row[4])>1e-8:
                    raise ValueError('Duplicate raw/context evidence differs')
                duplicates+=1;max_duplicate_delta=max(max_duplicate_delta,abs(probs[i]-float(p)))
                # Length-sorted batches can have different padding shapes.
                # Identical-checkpoint provenance was checked above; use the
                # focal score canonically, exactly as the fitting audit does.
                # Report numerical variation instead of counting an edge twice.
                if max_duplicate_delta>.01:raise ValueError('Repeated CE scores differ unexpectedly despite audited provenance')
            else:
                i=len(all_rows);keys[key]=i;all_rows.append(row);probs.append(float(p))
            ix.append(i)
        indices[split]=np.asarray(ix,dtype=np.int64)
    text_model=load_text(STRUCT/'text_scorer.json')
    text=np.asarray([text_logit({'name':r[5],'address':r[6]},{'name':r[7],'address':r[8]},text_model) for r in all_rows])
    x=neighborhood(all_rows,np.asarray(probs),text);bound={}
    for split,path in paths.items():
        np.save(output/f'{split}.npy',x[indices[split]])
        bound[split]=dict(path=str(path),sha256=digest(path),pairs=len(indices[split]),features_sha256=digest(output/f'{split}.npy'))
    write(output/'manifest.json',dict(complete=True,names=NAMES,inputs=bound,code_sha256=digest(__file__),
        probabilities_sha256={str(p):digest(p) for p in cp.values()},duplicate_edges=duplicates,duplicate_ce_max_delta=max_duplicate_delta,
        source_audit_sha256=digest(reference),duplicate_policy='Canonical focal score; duplicate auxiliary edge omitted, as in audit',
        note='All routed below-threshold rival scores; no labels. Bounded reverse competition is not full test coverage. No source one-to-one cap.'))
    if run_audit:
        args=[sys.executable,'tools/v3_neural_independent_audit.py','--tag',tag,'--extra-root',str(output),
            '--ce-root',str(ce_root),'--rival-scores',str(cp['rivals']),'--output',str(output/'evaluation')]
        if rival_rows.resolve()!=(SCRATCH/'rivals.pkl').resolve():
            ops=output/'rivals_ops.npy';np.save(ops,edit_matrix(rows_by['rivals'],feature_functions()))
            args+=['--rival-rows',str(rival_rows),'--rival-ops',str(ops)]
        with (output/'audit.log').open('a') as log:
            subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tag',default='e5_cont1300k');p.add_argument('--ce-root',type=Path)
    p.add_argument('--rival-rows',type=Path);p.add_argument('--rival-scores',type=Path)
    p.add_argument('--audit-reference',type=Path)
    p.add_argument('--run-audit',action='store_true');run(**vars(p.parse_args()))
