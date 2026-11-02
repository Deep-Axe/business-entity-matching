"""Measure residual KC retrieval misses using cached, label-blind channels.

No retrieval jobs, peer-file changes, model fitting, or test predictions. Gold
labels are used only for diagnostic coverage/oracle reports on all 6k owners.
"""
from collections import Counter, defaultdict
import json
import pickle
from pathlib import Path
import sys
import time

import numpy as np

from ber.store import connect
from v3_transfer_data import ROOT, PILOT, digest, write

sys.path.insert(0, 'src/neural')
from name_unique_lib import core

C = Path('work/ce')
OUT = ROOT/'artifacts/improvement_v3/kc_residual_retrieval_audit_v1'


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    start = time.monotonic(); hashes = {}

    def track(p):
        p=Path(p); h=digest(p)
        assert hashes.get(str(p),h)==h
        hashes[str(p)]=h
        return p

    def raw(p):
        with track(p).open('rb') as f:return pickle.load(f)

    arrays={}; base=defaultdict(set); split={}; info={}
    for sp in ('dev','tune'):
        for r in raw(C/f'{sp}_kc.pkl'):
            base[r[0]].add(r[1])
        with np.load(track(PILOT/'data'/f'{sp}.npz')) as z:
            arrays[sp]={k:z[k] for k in ('reference_ids','true_counts','left','right')}
        split.update({str(o):sp for o in arrays[sp]['reference_ids']})
    db=connect(ROOT/'artifacts/train_store_v2');db.execute('PRAGMA cache_size=-4096')
    gold={o:{x[0] for x in db.execute('SELECT target_id FROM labels WHERE reference_id=?',(o,))} for o in split}
    db.close()
    for r in raw(C/'err_links.pkl'):
        if r['y']:
            assert r['t'] in gold[r['s1']]
            info[(r['s1'],r['t'])]=r
    assert len(info)==sum(map(len,gold.values()))
    misses={(o,t) for o in split for t in gold[o]-base[o]}

    routes={}
    for cap in (10,20):
        p=C/f'keych_devtune_k{cap}.pkl'
        if p.exists():
            route=defaultdict(set)
            for o,t,ch in raw(p):route[o].add(t)
            routes[f'key_cap{cap}']=route
    with np.load(track(C/'dense_devtune.npz')) as z:
        dense=defaultdict(dict)
        qid=z['qid']
        for k in z.files:
            if not k.endswith('_ix'):continue
            view,country=k[:-3].rsplit('_',1)
            for qi,ids,ss in zip(z[k],z[f'{view}_{country}_id'],z[f'{view}_{country}_s']):
                dense[str(qid[qi])][view]=[
                    (f'S3-{int(t)-1000000000}' if t>=1000000000 else f'S2-{int(t)}',float(s))
                    for t,s in zip(ids,ss) if t>=0]
    for top in (10,20,30):
        for filtered in (True,False):
            route=defaultdict(set)
            for o,views in dense.items():
                for view,items in views.items():
                    for t,s in items[:top]:
                        if not filtered or s >= (.9 if view=='name_addr' else .96):route[o].add(t)
            routes[f'dense_top{top}_'+('current_cutoffs' if filtered else 'no_cutoffs')]=route
    for view in ('name_addr','name'):
        routes[f'dense_{view}_top30_no_cutoffs']={o:{t for t,s in views.get(view,[])} for o,views in dense.items()}
    ids=[json.loads(line)['id'] for line in track(PILOT/'data/records.jsonl').open()]
    rawroute=defaultdict(set)
    for a in arrays.values():
        for l,r in zip(a['left'],a['right']):rawroute[ids[int(l)]].add(ids[int(r)])
    routes['original_pre_neural_gate']=rawroute
    native=defaultdict(set)
    for sp in arrays:
        p=ROOT/f'artifacts/improvement_v3/native_alias_recovery_3k_v1/{sp}_extra.pkl'
        if p.exists():
            for r in raw(p):native[r[0]].add(r[1])
    routes['previous_native_alias_pilot']=native
    routes['cached_dense30_plus_key10']={o:routes['dense_top30_no_cutoffs'].get(o,set())|routes.get('key_cap10',{}).get(o,set()) for o in split}
    route_names=list(routes)
    routes['all_available_cached_routes']={o:set().union(*(routes[k].get(o,set()) for k in route_names)) for o in split}

    def measure(sp,extra):
        owners=[o for o in split if split[o]==sp]; scores=[]; newcounts=[]; recovered=0; remaining=0
        for o in owners:
            add=extra.get(o,set())-base[o]; cand=base[o]|add
            hit=len(cand&gold[o]); t=len(gold[o])
            scores.append(5*hit/(t+4*hit) if t else 1.)
            newcounts.append(len(add)); recovered+=len(add&gold[o]);remaining+=t-hit
        return dict(oracle=float(np.mean(scores)),new_pairs=sum(newcounts),
            new_true=recovered,remaining_missed=remaining,
            additional_per_owner=float(np.mean(newcounts)),additional_p95=float(np.quantile(newcounts,.95)),
            additional_p99=float(np.quantile(newcounts,.99)))

    reports={}; cases=[]
    for sp in arrays:
        current=measure(sp,{})
        channelstats={k:measure(sp,v) for k,v in routes.items()}
        for v in channelstats.values():v['oracle_gain']=v['oracle']-current['oracle']
        slices=defaultdict(list); country=Counter()
        for o,t in sorted(misses):
            if split[o]!=sp:continue
            r=info[(o,t)];country[r['cty']]+=1
            same_name=r['s1_text'][0].strip().casefold()==r['t_text'][0].strip().casefold()
            same_core=core(r['s1_text'][0])==core(r['t_text'][0])
            tags=dict(empty_address=r['empty_addr'],native_script=r['native'],website=r['web'],
                empty_address_owner_has_raw_name_twins=r['empty_addr'] and r['exact_twins']>0,
                empty_address_same_raw_name_with_twins=r['empty_addr'] and same_name and r['exact_twins']>0,
                empty_address_same_core_with_twins=r['empty_addr'] and same_core and r['core_twins']>0,
                low_name_overlap_with_address=not r['empty_addr'] and r['name_j']<.3,
                original_gate_dropped=t in rawroute[o])
            ranks={v:next((dict(rank=i+1,similarity=s) for i,(tid,s) in enumerate(items) if tid==t),None) for v,items in dense[o].items()}
            available=[k for k,v in routes.items() if t in v.get(o,set())]
            for k,on in tags.items():
                if on:slices[k].append((o,t))
            hit=len(base[o]&gold[o]); n=len(gold[o]); now=5*hit/(n+4*hit) if n else 1.
            one=(5*(hit+1)/(n+4*(hit+1))-now)/3000
            cases.append(dict(split=sp,owner=o,target=t,country=r['cty'],owner_text=r['s1_text'],
                target_text=r['t_text'],true_matches=n,retrieved_true=hit,
                marginal_oracle_gain_if_this_link_alone_recovered=one,tags=tags,
                core_twins=r['core_twins'],exact_twins=r['exact_twins'],dense_positions=ranks,
                cached_channels_containing_link=available))
        slicestats={}
        for k,links in slices.items():
            extra=defaultdict(set)
            for o,t in links:extra[o].add(t)
            m=measure(sp,extra)
            slicestats[k]=dict(missed_links=len(links),oracle_gain_if_all_recovered=m['oracle']-current['oracle'])
        residual=[x for x in cases if x['split']==sp]
        reports[sp]=dict(baseline=current,countries=dict(country),overlapping_patterns=slicestats,
            cached_channel_expansions=channelstats,
            dense_residual_positions=dict(
                inside_top10_but_cut_off=sum(any(v and v['rank']<=10 for v in r['dense_positions'].values()) for r in residual),
                only_rank11_to30=sum(any(r['dense_positions'].values()) and not any(v and v['rank']<=10 for v in r['dense_positions'].values()) for r in residual),
                absent_from_both_top30=sum(not any(r['dense_positions'].values()) for r in residual)))
    cases.sort(key=lambda r:(r['split'],-r['marginal_oracle_gain_if_this_link_alone_recovered'],r['owner'],r['target']))
    for p,h in hashes.items():assert digest(p)==h,'Input changed: '+p
    write(OUT/'cases.json',cases)
    report=dict(complete=True,splits=reports,inputs_sha256=hashes,seconds=time.monotonic()-start,
        caveats=['Retrieval oracle only, not end-to-end acceptance or submission improvement.',
            'All channels are cached label-blind proposals evaluated on all 6000 owners.',
            'Error slices overlap; oracle gains cannot be added.',
            'No labels from these diagnostic IDs may enter training or candidate rules.',
            'Current dense name-only gallery includes empty-address targets only; name+address includes all targets.',
            'No French labels or French retrieval ceiling is measured.'])
    write(OUT/'report.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':run()
