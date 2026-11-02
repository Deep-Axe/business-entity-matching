"""Full owner-set metric with bounded reverse-index rival discovery.

Rival probabilities are computed only with genuine forward retrieval and full
candidate context, never by fabricating rank/context features. Reverse top-k is
not exhaustive over S1; discovered rivals missing the target in their forward
list are explicitly not claimants. Focal tune/dev owners are scored separately.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
from catboost import CatBoostClassifier

from ber.direct_features import make_arrays
from ber.deadline_retrieval import retrievers
from ber.disk_index import TargetStore
from ber.metrics import score_arrays
from ber.store import connect,lookup,record_row
from ber.text import Record
from v2_owner_competition import Competition
from v2_run_pipeline import enhanced_matrix,NAMES
from v2_cpu_features import Weighting
from v3_owner_assignment import OUT,MODEL,load_text,text_logit,winner
from v3_transfer_data import ROOT,PILOT,digest,load,write,owner_scores,bootstrap


def database(output):
    Path(output).mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(Path(output)/'rivals.sqlite3');db.row_factory=sqlite3.Row
    db.execute('PRAGMA cache_size=-8192');db.execute('PRAGMA temp_store=FILE')
    db.executescript('''
      CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS targets(target TEXT PRIMARY KEY,discovered INTEGER NOT NULL DEFAULT 0);
      CREATE TABLE IF NOT EXISTS seeds(split TEXT,owner TEXT,target TEXT,p REAL,y INTEGER,text REAL,
                                     PRIMARY KEY(split,owner,target));
      CREATE INDEX IF NOT EXISTS seeds_target ON seeds(target);
      CREATE TABLE IF NOT EXISTS rivals(target TEXT,owner TEXT,text_strength REAL,PRIMARY KEY(target,owner));
      CREATE INDEX IF NOT EXISTS rivals_owner ON rivals(owner);
      CREATE TABLE IF NOT EXISTS chosen(target TEXT,owner TEXT,PRIMARY KEY(target,owner));
      CREATE INDEX IF NOT EXISTS chosen_owner ON chosen(owner);
      CREATE TABLE IF NOT EXISTS scores(owner TEXT PRIMARY KEY,seconds REAL,pairs INTEGER);
      CREATE TABLE IF NOT EXISTS claims(target TEXT,owner TEXT,p REAL,text REAL,PRIMARY KEY(target,owner));
    ''')
    return db


def inputs(model=MODEL,baseline_key='baseline_100000'):
    model=Path(model)
    m=json.loads((PILOT/'data/manifest.json').read_text())
    if digest(PILOT/'data/records.jsonl')!=m['record_sha256']: raise ValueError('Record cache changed')
    rows=[json.loads(x) for x in (PILOT/'data/records.jsonl').open()]
    a={}
    for split in ('tune','dev'):
        if digest(PILOT/'data'/f'{split}.npz')!=m['splits'][split]['data_sha256']: raise ValueError('Pair cache changed')
        a[split]=load(split)
        filename='enhanced_full_tune_probabilities.npy' if split=='tune' else 'enhanced_dev_probabilities.npy'
        if not np.array_equal(a[split][baseline_key],np.load(model.parent/filename)):
            raise ValueError('Base probability alignment changed')
    if set(a['tune']['reference_ids'])&set(a['dev']['reference_ids']): raise ValueError('Owner split overlap')
    return rows,a


def setup(output=OUT/'owner_eval',text_scorer=OUT/'text_scorer.json',model=MODEL,threshold=.76,baseline_key='baseline_100000'):
    model=Path(model)
    if not 0 < threshold < 1: raise ValueError('Invalid acceptance threshold')
    db=database(output);text_model=load_text(text_scorer)
    recipe={'model':digest(model),'text':digest(text_scorer),'threshold':threshold,
        'pilot_data':digest(PILOT/'data/manifest.json'),'reverse_k_per_channel':10}
    if model.resolve()!=MODEL.resolve():
        recipe.update(model_path=str(model.resolve()),baseline_key=baseline_key)
    identity=json.dumps(recipe,sort_keys=True)
    old=db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
    if old:
        if old[0]!=identity: raise ValueError('Owner-evaluation identity changed')
        db.close();return
    rows,arrays=inputs(model,baseline_key)
    with db:
        for split,a in arrays.items():
            for i in np.flatnonzero(a[baseline_key]>=threshold):
                l,r=rows[int(a['left'][i])],rows[int(a['right'][i])]
                val=(split,l['id'],r['id'],float(a[baseline_key][i]),int(a['y'][i]),text_logit(l,r,text_model))
                db.execute('INSERT INTO seeds VALUES(?,?,?,?,?,?)',val)
                db.execute('INSERT OR IGNORE INTO targets(target) VALUES(?)',(r['id'],))
                db.execute('INSERT OR IGNORE INTO claims VALUES(?,?,?,?)',(r['id'],l['id'],val[3],val[5]))
        db.execute("INSERT INTO meta VALUES('identity',?)",(identity,))
    db.close()


def recipe_for(db):
    recipe=json.loads(db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0])
    model=Path(recipe.get('model_path',MODEL))
    if digest(model)!=recipe['model']:raise ValueError('Bound model changed')
    return recipe,model,recipe.get('baseline_key','baseline_100000'),recipe['threshold']


def reuse_discovery(output,previous):
    """Copy only label-free reverse retrieval, never old-model claim scores."""
    db=database(output)
    old=sqlite3.connect((Path(previous)/'rivals.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
    current=json.loads(db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0])
    prior=json.loads(old.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0])
    for key in ('pilot_data','reverse_k_per_channel','text'):
        if current[key]!=prior[key]:raise ValueError('Reverse-retrieval recipe changed: '+key)
    copied=0
    with db:
        for (target,) in db.execute('SELECT target FROM targets WHERE discovered=0').fetchall():
            row=old.execute('SELECT discovered FROM targets WHERE target=?',(target,)).fetchone()
            if row is None or not row[0]:continue
            db.executemany('INSERT OR IGNORE INTO rivals VALUES(?,?,?)',
                old.execute('SELECT target,owner,text_strength FROM rivals WHERE target=?',(target,)))
            db.execute('UPDATE targets SET discovered=1 WHERE target=?',(target,));copied+=1
    old.close();db.close()
    return dict(reused_targets=copied,model_scores_reused=False)


def seed_cached_profiles(output,text_scorer=OUT/'text_scorer.json'):
    """Focal owners already have complete genuine forward lists; replay those."""
    db=database(output);_,model,baseline_key,_=recipe_for(db)
    rows,arrays=inputs(model,baseline_key);text_model=load_text(text_scorer)
    targets={r[0] for r in db.execute('SELECT target FROM targets')}
    done={r[0] for r in db.execute('SELECT owner FROM scores')}
    with db:
        for a in arrays.values():
            counts=np.bincount(a['query_ids'],minlength=len(a['reference_ids']))
            for i in range(len(a['y'])):
                left,right=rows[int(a['left'][i])],rows[int(a['right'][i])]
                if left['id'] in done or right['id'] not in targets:continue
                db.execute('INSERT OR REPLACE INTO claims VALUES(?,?,?,?)',
                    (right['id'],left['id'],float(a[baseline_key][i]),text_logit(left,right,text_model)))
            for owner,count in zip(a['reference_ids'],counts):
                db.execute('INSERT OR IGNORE INTO scores VALUES(?,?,?)',(str(owner),0.,int(count)))
    db.close()


def discover(output=OUT/'owner_eval',limit=0):
    db=database(output);competition=Competition(ROOT/'artifacts/train_store_v2',ROOT/'artifacts/improvement_v2/owner/train_s1_index_v1',top_k=10)
    started=time.monotonic();n=0
    while not limit or n<limit:
        pending=db.execute('SELECT target FROM targets WHERE discovered=0 ORDER BY target LIMIT 100').fetchall()
        if not pending: break
        for row in pending:
            target=row['target'];_,rivals=competition.rivals(target)
            with db:
                db.executemany('INSERT OR IGNORE INTO rivals VALUES(?,?,?)',[(target,owner,float(strength)) for owner,strength in rivals.items()])
                db.execute('UPDATE targets SET discovered=1 WHERE target=?',(target,))
            n+=1
            if limit and n>=limit:break
        result=dict(stage='discover',targets_total=db.execute('SELECT count(*) FROM targets').fetchone()[0],
                    targets_done=db.execute('SELECT count(*) FROM targets WHERE discovered=1').fetchone()[0],
                    unique_rival_owners=db.execute('SELECT count(DISTINCT owner) FROM rivals').fetchone()[0],
                    rival_pairs=db.execute('SELECT count(*) FROM rivals').fetchone()[0],
                    this_invocation_targets=n,seconds=time.monotonic()-started,
                    caveat='Full S1 reverse gallery; top10 per channel is bounded retrieval, not exhaustive competition')
        write(Path(output)/'discovery.json',result);print(json.dumps(result),flush=True)
    competition.db.close();db.close()


def shortlist(output=OUT/'owner_eval',text_scorer=OUT/'text_scorer.json',top=3):
    db=database(output);model=load_text(text_scorer)
    if db.execute('SELECT count(*) FROM targets WHERE discovered=0').fetchone()[0]:raise ValueError('Finish discovery first')
    old=db.execute("SELECT value FROM meta WHERE key='shortlist'").fetchone()
    recipe=json.dumps({'top':top,'text_scorer_sha256':digest(text_scorer)},sort_keys=True)
    if old and old[0]!=recipe:raise ValueError('Shortlist recipe changed')
    if old:db.close();return
    source=connect(ROOT/'artifacts/train_store_v2');source.execute('PRAGMA cache_size=-8192');n=0
    with db:
        for target, in db.execute('SELECT target FROM targets ORDER BY target').fetchall():
            ids=[r[0] for r in db.execute('SELECT owner FROM rivals WHERE target=?',(target,))]
            owners=lookup(source,'anchors','entity_id',ids);r=lookup(source,'records','entity_id',[target])[target]
            ranked=sorted(((text_logit(owners[e],r,model),e) for e in ids),reverse=True)
            picked={e for _,e in ranked[:top]}
            picked.update(r[0] for r in db.execute('SELECT owner FROM seeds WHERE target=?',(target,)))
            db.executemany('INSERT OR IGNORE INTO chosen VALUES(?,?)',[(target,e) for e in picked])
            n+=1
            if n%1000==0:print(json.dumps({'stage':'shortlist','targets':n}),flush=True)
        db.execute("INSERT INTO meta VALUES('shortlist',?)",(recipe,))
    result=dict(top_text_claimants=top,targets=n,chosen_owner_profiles=db.execute('SELECT count(DISTINCT owner) FROM chosen').fetchone()[0],
                discovered_owner_profiles=db.execute('SELECT count(DISTINCT owner) FROM rivals').fetchone()[0],
                selection_uses_labels=False,all_seed_owners_retained=True,
                caveat='Cheap text ranking shortlist; exact classifier context only for selected owners. Unscored rivals are unknown, not proven negatives.')
    write(Path(output)/'shortlist.json',result);print(json.dumps(result),flush=True);source.close();db.close()


def score_rivals(output=OUT/'owner_eval',text_scorer=OUT/'text_scorer.json',limit=0):
    db=database(output);model_text=load_text(text_scorer)
    expected,model_path,_,_=recipe_for(db)
    if expected['text']!=digest(text_scorer):raise ValueError('Scoring identity changed')
    store=ROOT/'artifacts/train_store_v2';index=ROOT/'artifacts/train_index_v2'
    base,extra=retrievers(index,ROOT/'artifacts/training_lexicon_v1',0,None,store=store)
    targets=TargetStore(store);weights=Weighting(index)
    model=CatBoostClassifier(thread_count=1);model.load_model(str(model_path));assert model.feature_names_==NAMES
    source=connect(store);source.execute('PRAGMA cache_size=-8192')
    table='chosen' if db.execute("SELECT value FROM meta WHERE key='shortlist'").fetchone() else 'rivals'
    n=0;started=time.monotonic();max_delta=0.
    while not limit or n<limit:
        pending=db.execute(f'SELECT DISTINCT r.owner FROM {table} r LEFT JOIN scores s ON r.owner=s.owner WHERE s.owner IS NULL ORDER BY r.owner LIMIT 10').fetchall()
        if not pending:break
        for row in pending:
            owner=row['owner'];stamp=time.monotonic();raw=lookup(source,'anchors','entity_id',[owner])[owner]
            a,timing=make_arrays([record_row(raw)],base,extra,targets,None,20)
            target_rows=lookup(source,'records','entity_id',a['target_ids'].tolist())
            X=enhanced_matrix(a,[record_row(raw)],target_rows,weights)
            p=model.predict_proba(X,thread_count=1)[:,1] if len(X) else np.empty(0)
            # Score genuine complete candidate lists, retain every score for later
            # feature access. Seed parity checks catch retrieval/context drift.
            seeds={r['target']:r['p'] for r in db.execute('SELECT target,p FROM seeds WHERE owner=?',(owner,))}
            found=set()
            with db:
                for target,prob in zip(a['target_ids'],p):
                    target=str(target);found.add(target)
                    if target in seeds:
                        delta=abs(float(prob)-seeds[target]);max_delta=max(max_delta,delta)
                        if delta>1e-9:raise ValueError(f'Forward replay differs from seed probability: {delta}')
                    if db.execute('SELECT 1 FROM targets WHERE target=?',(target,)).fetchone():
                        db.execute('INSERT OR REPLACE INTO claims VALUES(?,?,?,?)',
                                   (target,owner,float(prob),text_logit(raw,target_rows[target],model_text)))
                if set(seeds)-found:raise ValueError('Forward replay lost an accepted seed candidate')
                db.execute('INSERT INTO scores VALUES(?,?,?)',(owner,time.monotonic()-stamp,len(p)))
            n+=1
            if limit and n>=limit:break
        done,total=db.execute('SELECT count(*) FROM scores').fetchone()[0],db.execute(f'SELECT count(DISTINCT owner) FROM {table}').fetchone()[0]
        result=dict(stage='score',owners_done=done,owners_discovered=total,this_invocation_owners=n,
                    seconds=time.monotonic()-started,seed_replay_max_abs_delta=max_delta,
                    shortlist_table=table,semantics='Genuine full-gallery forward retrieval and76feature context; no injected rival pairs')
        write(Path(output)/'scoring.json',result);print(json.dumps(result),flush=True)
    source.close();db.close()


def evaluate(output=OUT/'owner_eval',text_scorer=OUT/'text_scorer.json'):
    output=Path(output)
    if (output/'policy.json').exists():raise FileExistsError('Development has already been reported for this experiment')
    db=database(output)
    if db.execute('SELECT count(*) FROM targets WHERE discovered=0').fetchone()[0]:raise ValueError('Discovery incomplete')
    table='chosen' if db.execute("SELECT value FROM meta WHERE key='shortlist'").fetchone() else 'rivals'
    if db.execute(f'SELECT count(DISTINCT r.owner) FROM {table} r LEFT JOIN scores s ON s.owner=r.owner WHERE s.owner IS NULL').fetchone()[0]:raise ValueError('Rival scoring incomplete')
    _,model_path,baseline_key,threshold=recipe_for(db)
    rows,arrays=inputs(model_path,baseline_key);groups={}
    for row in db.execute('SELECT target,owner,p,text FROM claims WHERE p>=? ORDER BY target,owner',(threshold,)):
        groups.setdefault(row['target'],[]).append((row['owner'],row['p'],row['text']))
    groups={k:v for k,v in groups.items() if len(v)>1}
    maps={s:{(rows[int(l)]['id'],rows[int(r)]['id']):i for i,(l,r) in enumerate(zip(a['left'],a['right']))} for s,a in arrays.items()}
    def predicted(split,rule):
        a=arrays[split];p=a[baseline_key].copy()
        if rule is None:return p
        for target,edges in groups.items():
            chosen=winner([e[1] for e in edges],[e[2] for e in edges],**rule)
            for j,(owner,_,_) in enumerate(edges):
                i=maps[split].get((owner,target))
                if i is not None and j!=chosen:p[i]=-1.
        return p
    choices=[(None,float(owner_scores(arrays['tune'],predicted('tune',None),threshold).mean()))]
    for alpha in (0.,.1,.25,.5,1.):
        for floor in (0.,.5,.8):
            for margin in (0.,.1,.25,.5,1.,1.5,2.):
                rule=dict(alpha=alpha,floor=floor,margin=margin)
                choices.append((rule,float(owner_scores(arrays['tune'],predicted('tune',rule),threshold).mean())))
    # Strict improvement required; ties preserve unchanged baseline.
    selected,best=choices[0]
    for rule,value in choices[1:]:
        if value>best+1e-12:selected,best=rule,value
    policy=dict(frozen_on_tune=True,development_reported=False,enabled=selected is not None,
                selected=selected or dict(alpha=0.,floor=0.,margin=0.),model_sha256=digest(model_path),threshold=threshold,
                text_scorer_sha256=digest(text_scorer),tune_macro_f05=best,
                selection_metric='Full owner-set macroF0.5 on all3000 focal tune owners, including zero-match owners',
                rival_scope=table,
                caveat='Bounded reverse top10/channel then text top3+focal-owner shortlist; not exhaustive. Some rivals occur in base-model training.')
    write(output/'policy_before_dev.json',policy)
    report={'tune_grid':[{'rule':r,'macro_f05':s} for r,s in choices],'splits':{}}
    for split,a in arrays.items():
        p=predicted(split,selected);base=a[baseline_key];old=owner_scores(a,base,threshold);new=owner_scores(a,p,threshold)
        metrics=score_arrays(a['query_ids'],a['y'],p,a['true_counts'],threshold);metrics.pop('entity_scores')
        report['splits'][split]=dict(**metrics,baseline_macro_f05=float(old.mean()),comparison=bootstrap(old,new),
            removed_true=int(((base>=threshold)&(p<threshold)&(a['y']==1)).sum()),removed_false=int(((base>=threshold)&(p<threshold)&(a['y']==0)).sum()),
            country={str(c):{'baseline':float(old[a['countries']==c].mean()),'new':float(new[a['countries']==c].mean())} for c in np.unique(a['countries'])})
    # Labels below diagnose neighborhood coverage only, AFTER the tune policy is
    # frozen. Never insert the labeled owner into reverse retrieval/shortlisting.
    source=connect(ROOT/'artifacts/train_store_v2');coverage={}
    for split in ('tune','dev'):
        counts=dict(targets=0,labeled_targets=0,reverse_true_owner_present=0,shortlist_true_owner_present=0,
                    scored_true_pair_present=0,scored_true_owner_accepted=0,other_labeled_owner=0)
        for row in db.execute('SELECT DISTINCT target FROM seeds WHERE split=?',(split,)):
            target=row[0];counts['targets']+=1
            truth=source.execute('SELECT reference_id FROM labels WHERE target_id=?',(target,)).fetchone()
            if truth is None:continue
            owner=truth[0];counts['labeled_targets']+=1
            counts['reverse_true_owner_present']+=db.execute('SELECT 1 FROM rivals WHERE target=? AND owner=?',(target,owner)).fetchone() is not None
            counts['shortlist_true_owner_present']+=db.execute(f'SELECT 1 FROM {table} WHERE target=? AND owner=?',(target,owner)).fetchone() is not None
            score=db.execute('SELECT p FROM claims WHERE target=? AND owner=?',(target,owner)).fetchone()
            counts['scored_true_pair_present']+=score is not None
            counts['scored_true_owner_accepted']+=score is not None and score[0]>=threshold
            counts['other_labeled_owner']+=db.execute('SELECT 1 FROM seeds WHERE split=? AND target=? AND owner=?',(split,target,owner)).fetchone() is None
        coverage[split]=counts
    source.close();report['neighborhood_coverage_not_used_for_selection']=coverage
    policy['margin_scale']='logit(base probability) + alpha * training-only text-ranking logit; NOT a probability margin'
    report['interpretation']='Exploratory existing US/India development, not French accuracy. Rival owners affect assignment but focal owner sets define evaluation; no per-edge utility approximation.'
    write(output/'report.json',report);policy['development_reported']=True;write(output/'policy.json',policy)
    db.close();return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['setup','reuse','cached-profiles','discover','shortlist','score','evaluate'])
    p.add_argument('--output',type=Path,default=OUT/'owner_eval');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--model-size',type=int,choices=(100000,200000),default=100000)
    p.add_argument('--previous',type=Path)
    args=p.parse_args()
    if args.command=='setup':
        model=ROOT/f'artifacts/improvement_v2/cpu/fit_gate0_{args.model_size//1000}k_v1/enhanced.cbm'
        threshold=json.loads((model.parent/'enhanced_report.json').read_text())['threshold']
        setup(args.output,model=model,threshold=threshold,baseline_key=f'baseline_{args.model_size}')
    elif args.command=='reuse':
        if args.previous is None:p.error('--previous is required for reuse')
        print(json.dumps(reuse_discovery(args.output,args.previous)),flush=True)
    elif args.command=='cached-profiles':seed_cached_profiles(args.output)
    elif args.command=='discover':discover(args.output,args.limit)
    elif args.command=='shortlist':shortlist(args.output)
    elif args.command=='score':score_rivals(args.output,limit=args.limit)
    else:print(json.dumps(evaluate(args.output),indent=2))
