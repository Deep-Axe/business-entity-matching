"""Profile submitted neural decisions on 400 unlabeled test owners per country.

Uses retained original stacker probabilities, not a different reconstructed
model. Flags describe observable text, never asserted false positives/negatives.
No output mutation, threshold search, or external API.
"""
from collections import Counter, defaultdict
import csv
from pathlib import Path
import pickle
import time

import numpy as np
from rapidfuzz import fuzz

from v3_neural_test_pipeline import SCRATCH, read, resources
from v3_owner_error_analysis import missing
from v3_structural_features import folded, house_number, street_component, single_digit_change
from v3_transfer_data import ROOT, digest, write

WORK=ROOT/'artifacts/improvement_v3/public0976_test_profile_v2'
SAMPLE=ROOT/'artifacts/improvement_v3/recovery_test_volume_v1'
PUBLIC=SCRATCH/'sub_v4_stack_all/matching_results.tsv'
OLD=ROOT/'artifacts/duf-fel_full_output_v3_100k_owner_v1/matching_results.tsv'


def selected_matches(path, owners):
    result={}
    with path.open(newline='') as handle:
        for r in csv.DictReader(handle,delimiter='\t'):
            owner=r['source1_entity_id']
            if owner in owners:
                if owner in result:raise ValueError('Duplicate sampled submission owner')
                result[owner]=set(filter(None,r['matched_entity_ids'].split(',')))
    if set(result)!=owners:raise ValueError('Sampled submission coverage differs')
    return result


def run():
    WORK.mkdir(parents=True,exist_ok=False);resources();start=time.monotonic()
    report=read(SAMPLE/'report.json');source=read(SAMPLE/'selection.json')
    if not report['complete'] or digest(SAMPLE/'selection.json')!=report['outputs_sha256']['selection.json']:
        raise ValueError('Frozen unbiased text sample changed')
    if digest(PUBLIC)!='5a0be554e60f88e763de95a5b4cc8f30f7cee352067ba24c16947e6319752118':
        raise ValueError('Submitted neural file changed')
    refs={r['entity_id']:r for r in source['selected']};selected=set(refs)
    new,old=selected_matches(PUBLIC,selected),selected_matches(OLD,selected)
    stats=defaultdict(Counter);patterns=defaultdict(lambda:defaultdict(Counter));seen=defaultdict(set);examples=[];hashes={}
    for owner,r in refs.items():
        s=stats[r['country']];s.update(owners=1,public_matches=len(new[owner]),old_matches=len(old[owner]),
            public_empty=int(not new[owner]),old_empty=int(not old[owner]),added=len(new[owner]-old[owner]),removed=len(old[owner]-new[owner]))
    for part in range(128):
        resources();raw=SCRATCH/'test'/f'part_{part:03d}.pkl'
        ce_path=Path(str(raw)+'.e5_cont1300k.npy');stack_path=Path(str(raw)+'.stack_e5_cont1300k_nocty_nf2.npy')
        frequency_path=Path(str(raw)+'.nf.npy')
        rows=pickle.load(raw.open('rb'));ce=np.load(ce_path);p=np.load(stack_path);nf=np.load(frequency_path)
        if len(rows)!=len(ce) or len(rows)!=len(p) or nf.shape!=(len(rows),2):raise ValueError('Sample score arrays misaligned')
        if not np.isfinite(p).all() or not np.isfinite(ce).all():raise ValueError('Invalid saved probability')
        for path in (raw,ce_path,stack_path,frequency_path):hashes[str(path)]=digest(path)
        for i,r in enumerate(rows):
            owner,target=r[:2]
            if owner not in selected:continue
            if target in seen[owner]:raise ValueError('Duplicate sampled candidate')
            seen[owner].add(target);accepted=target in new[owner];country=r[3]
            if accepted and p[i]<.75:raise ValueError('Submitted edge does not match saved stacker threshold')
            s=stats[country];s['routed_candidates']+=1;s['raw_above_threshold']+=int(p[i]>=.75)
            s['owner_removed_candidates']+=int(p[i]>=.75 and not accepted)
            s['uncertain_candidates']+=int(.1<=p[i]<.75)
            lh,rh=house_number(r[6],country),house_number(r[8],country)
            ls,rs=street_component(r[6],country),street_component(r[8],country)
            shared=nf[i,0]>np.log1p(1)+1e-5
            flags=dict(empty_target_address=missing(r[8]),native_target_name=any('\u0900'<=c<='\u0dff' for c in r[7]),
                shared_s1_ascii_core=shared,shared_target_core_in_s1=nf[i,1]>np.log1p(1)+1e-5,
                shared_name_and_missing_address=shared and missing(r[8]),
                shared_name_and_house_difference=bool(shared and lh and rh and lh!=rh),
                shared_name_and_street_difference=bool(shared and ls and rs and fuzz.token_sort_ratio(ls,rs)<70),
                house_digit_deleted=bool(lh and rh and len(lh)==len(rh)+1 and single_digit_change(lh,rh)),
                small_house_shift=bool(lh and rh and 0<abs(int(lh)-int(rh))<=25),
                low_name_similarity=fuzz.token_set_ratio(folded(r[5]),folded(r[7]))<50,
                very_high_ce_rejected=bool(ce[i]>=.99 and not accepted))
            for name,value in flags.items():
                if value:patterns[country][name].update(candidates=1,accepted=int(accepted),rejected=int(not accepted),
                    added_over_old=int(accepted and target not in old[owner]),removed_from_old=int(not accepted and target in old[owner]))
            if (country=='France' and accepted and (flags['shared_name_and_house_difference'] or flags['shared_name_and_street_difference'])) or flags['very_high_ce_rejected']:
                examples.append(dict(owner=owner,target=target,country=country,accepted=accepted,stack_probability=float(p[i]),
                    ce_probability=float(ce[i]),s1_name=r[5],s1_address=r[6],target_name=r[7],target_address=r[8],
                    flags=[k for k,v in flags.items() if v],truth_unknown=True))
        if part%16==15:write(WORK/'progress.json',dict(stage='profiling_saved_test_scores',partitions=part+1,seconds=time.monotonic()-start))
    if any(not new[o]<=seen[o] for o in selected):raise ValueError('Submitted edge absent from retained neural route')
    write(WORK/'examples.json',examples)
    write(WORK/'report.json',dict(complete=True,owners=len(selected),countries=stats,overlapping_patterns=patterns,
        sample_sha256=digest(SAMPLE/'selection.json'),public_sha256=digest(PUBLIC),previous_sha256=digest(OLD),
        score_inputs_sha256=hashes,seconds=time.monotonic()-start,labels_used=False,accuracy_evaluated=False,
        note='Unbiased stratified owner sample. Observable risks and decisions only: no French or other test error labels. ASCII core frequencies can collapse native-only names.'))


if __name__=='__main__':run()
