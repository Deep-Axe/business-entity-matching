"""Build honest rival evidence for tune-selected candidate-recovery proposals.

No change to a submission or cutoff. Proposed pairs keep p100=None: they were
not in their owner's original retrieval. Existing claims are instead computed
through genuine full forward retrieval and the frozen 100k model. No labels are
used for reverse retrieval, shortlist selection, or rival feature generation.
"""
from collections import defaultdict
import json
from pathlib import Path
import pickle
import time

import numpy as np
from catboost import CatBoostClassifier

from ber.deadline_retrieval import retrievers
from ber.direct_features import make_arrays
from ber.disk_index import TargetStore
from ber.store import connect,lookup,record_row
from v2_cpu_features import Weighting
from v2_owner_competition import Competition
from v2_run_pipeline import enhanced_matrix,NAMES
from v3_neural_test_pipeline import resources
from v3_owner_assignment import load_text,text_logit
from v3_transfer_data import ROOT,PILOT,digest,write

WORK=ROOT/'artifacts/improvement_v3/recovery_rivals_v1'
BASE=ROOT/'artifacts/improvement_v3'
ROUTES=(('native','native_alias_recovery_3k_v1','native_alias_ce_diagnostic_v1'),
        ('website','website_alias_recovery_3k_v1','website_alias_ce_diagnostic_v1'),
        ('deep','retrieval_error_slices_v1','retrieval_ce_diagnostic_v1'))


def indices(scores,threshold):
    p=np.asarray(scores)
    if p.ndim!=1 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid recovery scores')
    return np.flatnonzero(p>=threshold)


def run():
    WORK.mkdir(parents=True,exist_ok=True)
    if (WORK/'completion.json').exists():raise FileExistsError('Completed rival build retained')
    proposals={};bound={};provenance=defaultdict(list)
    for name,raw_name,scored_name in ROUTES:
        source=BASE/raw_name;scores=BASE/scored_name
        report=json.loads((scores/'report.json').read_text());identity=json.loads((scores/'identity.json').read_text())
        if not report['complete'] or identity['source_report_sha256']!=digest(source/'report.json'):
            raise ValueError('Recovery diagnostic provenance differs')
        source_report=json.loads((source/'report.json').read_text());threshold=report['tune_only_screen']['threshold']
        bound[name]=dict(report_sha256=digest(scores/'report.json'),threshold=threshold)
        for split in ('tune','dev'):
            raw=source/f'{split}_extra.pkl';cp=scores/f'{split}_pce.npy'
            receipt=json.loads((scores/f'{split}_scores.json').read_text())
            if digest(raw)!=source_report['outputs_sha256'][raw.name] or digest(cp)!=receipt['scores_sha256']:
                raise ValueError('Scored recovery candidates changed')
            rows=pickle.load(raw.open('rb'));p=np.load(cp,allow_pickle=False)
            if len(rows)!=len(p):raise ValueError('Recovery row alignment differs')
            for i in indices(p,threshold):
                r=rows[i]
                if r[4] is not None:raise ValueError('New pair has invented original-query probability')
                key=(split,r[0],r[1])
                if key in proposals and proposals[key]['row']!=r:raise ValueError('Duplicate recovery text/labels differ')
                proposals[key]=dict(split=split,row=r)
                provenance[key].append(dict(route=name,ce=float(p[i]),threshold=threshold))
    selected=[dict(v,provenance=provenance[k]) for k,v in sorted(proposals.items())]
    watched={r['row'][1] for r in selected};chosen=defaultdict(set)
    for row in selected:chosen[row['row'][1]].add(row['row'][0])
    identity=dict(routes=bound,code_sha256=digest(__file__),targets=len(watched),top_text_rivals=5,
        proposals=len(selected),labels_used_for_selection=False)
    write(WORK/'identity.json',identity)
    with (WORK/'proposals.pkl').open('wb') as f:pickle.dump(selected,f)
    source=connect(ROOT/'artifacts/train_store_v2');source.execute('PRAGMA cache_size=-8192')
    text=load_text(BASE/'structural_v1/text_scorer.json')
    reverse=Competition(ROOT/'artifacts/train_store_v2',ROOT/'artifacts/improvement_v2/owner/train_s1_index_v1',top_k=10)
    targets_raw=lookup(source,'records','entity_id',list(watched))
    for target in sorted(watched):
        resources();_,owners=reverse.rivals(target)
        raw=lookup(source,'anchors','entity_id',list(owners))
        best=sorted(((text_logit(raw[o],targets_raw[target],text),o) for o in raw),reverse=True)[:5]
        chosen[target].update(o for _,o in best)
    reverse.db.close();del reverse
    owners=set().union(*chosen.values());claims={};focal=set()
    ids=[json.loads(line)['id'] for line in (PILOT/'data/records.jsonl').open()]
    for split in ('tune','dev'):
        with np.load(PILOT/'data'/f'{split}.npz',allow_pickle=False) as z:
            a={k:z[k] for k in ('left','right','reference_ids','baseline_100000')}
        focal.update(map(str,a['reference_ids']))
        for l,r,p in zip(a['left'],a['right'],a['baseline_100000']):
            target=ids[int(r)];owner=ids[int(l)]
            if target in watched:claims[(owner,target)]=float(p)
    del a,ids
    model_path=ROOT/'artifacts/improvement_v2/cpu/fit_gate0_100k_v1/enhanced.cbm'
    model=CatBoostClassifier(thread_count=1);model.load_model(str(model_path))
    if model.feature_names_!=NAMES:raise ValueError('Wrong frozen CPU model')
    base,extra=retrievers(ROOT/'artifacts/train_index_v2',ROOT/'artifacts/training_lexicon_v1',0,None,store=ROOT/'artifacts/train_store_v2')
    targets=TargetStore(ROOT/'artifacts/train_store_v2');weight=Weighting(ROOT/'artifacts/train_index_v2')
    profiles=WORK/'profiles';profiles.mkdir(exist_ok=True);start=time.monotonic();cache=BASE/'empty_target_rivals_v1/profiles'
    built=reused=0
    for owner in sorted(owners-focal):
        resources();path=cache/f'{owner}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as z:tids=z['target_ids'];p=z['p100']
            reused+=1
        else:
            raw=lookup(source,'anchors','entity_id',[owner])[owner]
            a,_=make_arrays([record_row(raw)],base,extra,targets,None,20)
            right=lookup(source,'records','entity_id',a['target_ids'].tolist())
            X=enhanced_matrix(a,[record_row(raw)],right,weight)
            p=model.predict_proba(X,thread_count=1)[:,1] if len(X) else np.empty(0);tids=a['target_ids']
            np.savez_compressed(profiles/f'{owner}.npz',reference_ids=a['reference_ids'],target_ids=tids,p100=p,X=X)
            built+=1
        for target,value in zip(tids,p):
            if str(target) in watched:claims[(owner,str(target))]=float(value)
        write(WORK/'progress.json',dict(stage='full_forward_rival_profiles',built=built,reused=reused,
            total=len(owners-focal),seconds=time.monotonic()-start))
    # Only genuinely routed existing claims need neural scoring. No fake score
    # is attached to any proposed pair that was absent from original retrieval.
    routed={k:p for k,p in claims.items() if p>=.01};left=lookup(source,'anchors','entity_id',list({o for o,t in routed}))
    rows=[]
    for (o,t),p in sorted(routed.items()):
        l,r=left[o],targets_raw[t]
        rows.append((o,t,-1,l['country'],p,l['name'],l['address'],r['name'],r['address']))
    with (WORK/'rivals.pkl').open('wb') as f:pickle.dump(rows,f)
    source.close();targets.close()
    write(WORK/'completion.json',dict(complete=True,targets=len(watched),proposals=len(selected),routed_existing_claims=len(rows),
        full_forward_profiles_built=built,full_forward_profiles_reused=reused,identity_sha256=digest(WORK/'identity.json'),
        output_sha256={n:digest(WORK/n) for n in ('rivals.pkl','proposals.pkl')},model_sha256=digest(model_path),
        ready_for_submission=False,labels_used_for_rival_discovery=False,
        note='Bounded reverse top10/channel and text top5; complete forward contexts. Absent claims remain unknown, not negatives. Global owner evaluation and neural scoring still required.'))


if __name__=='__main__':run()
