"""Label-free sibling-name signals on all routed tune/dev pairs; diagnostics only.

Seeds must have a nonempty address, CE >= .99 and CPU >= .95. A candidate cannot
seed itself. No retrieval, hard acceptance rule, fitting or submission mutation.
Counts over all candidates avoid reporting precision from errors alone. Auxiliary
owner seed coverage remains unbuilt, so this does NOT evaluate global assignment.
"""
from collections import defaultdict,Counter
from functools import lru_cache
import json
import pickle

import numpy as np
from rapidfuzz import fuzz

from v3_neural_test_pipeline import SCRATCH
from v3_structural_features import core_name,folded
from v3_transfer_data import ROOT,digest,write


def missing(value):return not value.strip() or value.strip().upper() in ('NULL','<NULL>')


def signals(rows,ce):
    if len(rows)!=len(ce):raise ValueError('CE and raw candidates misaligned')
    seeds=defaultdict(list)
    for i,r in enumerate(rows):
        if r[4]>=.95 and ce[i]>=.99 and not missing(r[8]):seeds[r[0]].append(i)
    cores=[core_name(r[7]) for r in rows];left=[core_name(r[5]) for r in rows]
    # Empty or non-ASCII-collapsed names are not exact-match evidence.
    out=[]
    for i,r in enumerate(rows):
        own=fuzz.ratio(left[i],cores[i])/100 if left[i] and cores[i] else 0.
        vals=[fuzz.ratio(cores[j],cores[i])/100 for j in seeds[r[0]] if j!=i and cores[j] and cores[i]]
        best=max(vals,default=0.)
        out.append(dict(seeds=len(vals),best=best,gain=best-own,empty=missing(r[8])))
    return out


def run():
    out=ROOT/'artifacts/improvement_v3/sibling_alias_diagnostic_v1'
    if (out/'report.json').exists():raise FileExistsError('No blind repeat')
    peer=ROOT/'artifacts/duf-fel_full_output_v4_neural1300k_peerhalf_v1'
    audit=json.loads((peer/'evaluation/report.json').read_text())
    bound={str((ROOT/p).resolve()) if not p.startswith('/') else p:h for p,h in audit['inputs_sha256'].items()}
    error_path=ROOT/'artifacts/improvement_v3/neural_structural_error_peerhalf_v1/errors.json'
    errors=json.loads(error_path.read_text());missed={(e['owner'],e['target']) for e in errors if e['kind']=='FN' and e['stage']=='classifier'}
    report={}
    for split in ('tune','dev'):
        raw=SCRATCH/f'{split}.pkl';cp=peer/'half_calibration'/f'{split}_pce.npy'
        if bound.get(str(raw))!=digest(raw) or bound.get(str(cp))!=digest(cp):raise ValueError('Raw/CE provenance mismatch')
        rows=pickle.load(raw.open('rb'));ce=np.load(cp,allow_pickle=False);views=signals(rows,ce)
        groups=defaultdict(Counter)
        for r,v in zip(rows,views):
            masks={'all':True,'empty':v['empty'],
                'empty_seed_name_95':v['empty'] and v['best']>=.95,
                'empty_seed_name_95_gain_10':v['empty'] and v['best']>=.95 and v['gain']>=.10,
                'empty_seed_exact_gain_10':v['empty'] and v['best']==1 and v['gain']>=.10,
                'all_seed_name_95_gain_10':v['best']>=.95 and v['gain']>=.10}
            for key,flag in masks.items():
                if not flag:continue
                for group in (key,r[3]+'/'+key):
                    groups[group]['pairs']+=1;groups[group]['positive' if r[2] else 'negative']+=1
                    if split=='dev' and r[:2] in missed:groups[group]['selected_structural_classifier_FN']+=1
        report[split]={k:dict(v,positive_fraction=v['positive']/v['pairs']) for k,v in groups.items()}
    write(out/'report.json',dict(complete=True,groups=report,seed_rule='nonmissing target address AND pCE>=.99 AND p100>=.95; exclude self',
        feature_labels_used=False,development_used_for_reporting_only=True,full_owner_gain_evaluated=False,
        reference_error_sha256=digest(error_path),code_sha256=digest(__file__),ready_for_submission=False,
        note='Fixed diagnostic groups, not a tuned policy. Sibling candidates are already retrieved; a seed can still be wrong. Full rival-owner seed coverage would be required for an honest stacker/assignment comparison. No French labels.'))


if __name__=='__main__':run()
