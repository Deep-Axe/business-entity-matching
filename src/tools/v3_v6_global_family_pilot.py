"""Label-free global same-name rival discovery; tune-only veto pilot.

No external lookup and no test modifications. All provided train S1 text is
searched for relevant families, not only the old reverse-retrieval shortlist.
No rival is invented from labels, and no unscored match is added.
"""
from collections import defaultdict,Counter
from functools import lru_cache
import pickle
import numpy as np
from rapidfuzz import fuzz
from ber.store import connect
from v3_neural_test_pipeline import SCRATCH,resources,read
from v3_structural_features import core_name,folded,address_view,house_number
from v3_transfer_data import ROOT,PILOT,write,digest,bootstrap
from v3_neural_independent_audit import summary

OUT=ROOT/'artifacts/improvement_v3/v6_global_family_pilot_v1'
AUDIT=ROOT/'artifacts/improvement_v3/v6_band_error_audit_v1'


def run():
    OUT.mkdir(parents=True,exist_ok=False);rows=[];data={};bound={}
    def raw(p):bound[str(p)]=digest(p);return pickle.load(p.open('rb'))
    decisions=np.load(AUDIT/'decisions.npz');accepted=decisions['band_cC_keep'];prob=decisions['band_cC_p']
    ids=[__import__('json').loads(line)['id'] for line in (PILOT/'data/records.jsonl').open()]
    for split in ('dev','tune'):
        rr=raw(SCRATCH/(split+'.pkl'));start=len(rows);rows+=rr
        with np.load(PILOT/'data'/(split+'.npz'),allow_pickle=False) as z:
            a={k:z[k] for k in ('reference_ids','query_ids','true_counts','countries','y','left','right')}
        keys={(ids[int(l)],ids[int(r)]):i for i,(l,r) in enumerate(zip(a['left'],a['right']))}
        data[split]=(a,np.asarray([keys[r[:2]] for r in rr]),slice(start,len(rows)))
    core=lru_cache(maxsize=50000)(core_name);wanted=set()
    for r,k in zip(rows,accepted):
        if k:
            for s in (r[5],r[7]):
                c=core(s)
                if c:wanted.add((r[3],c))
    db=connect(ROOT/'artifacts/train_store_v2');db.execute('PRAGMA cache_size=-8192');families=defaultdict(list)
    for scanned,r in enumerate(db.execute('SELECT entity_id,country,name,address FROM anchors'),1):
        key=(r['country'],core(r['name']))
        if key in wanted:families[key].append(tuple(r))
        if scanned%250000==0:resources();write(OUT/'progress.json',dict(stage='global_text_family_scan',scanned=scanned))
    db.close()
    if scanned!=2206821:raise ValueError('Full S1 text coverage differs')
    flags=np.zeros((len(rows),3),bool);feature_rows=[]
    @lru_cache(maxsize=50000)
    def view(name,address,country):return folded(name),address_view(address,country),house_number(address,country)
    for i,(r,keep) in enumerate(zip(rows,accepted)):
        if not keep:continue
        ln,la,lh=view(r[5],r[6],r[3]);tn,ta,th=view(r[7],r[8],r[3]);own_name=fuzz.ratio(ln,tn);own_addr=fuzz.token_set_ratio(la,ta) if ta and la else 0.
        rivals={x[0]:x for c in (core(r[5]),core(r[7])) if c for x in families.get((r[3],c),[]) if x[0]!=r[0]}
        best_margin=-100.;ambiguous=False;qualifying=[]
        for eid,country,name,address in rivals.values():
            nn,na,nh=view(name,address,country);ns=fuzz.ratio(nn,tn)
            if not ta and ns>=95 and ns>=own_name-.5:ambiguous=True
            if ta and th and nh==th and lh and lh!=th and ns>=90 and ns>=own_name-3:
                addr=fuzz.token_set_ratio(na,ta)
                if addr>=95:
                    best_margin=max(best_margin,addr-own_addr);qualifying.append(eid)
        flags[i]=[best_margin>=10,best_margin>=20,ambiguous]
        if any(flags[i]):feature_rows.append(dict(owner=r[0],target=r[1],global_rivals=len(rivals),address_margin=best_margin,empty_name_ambiguous=ambiguous,qualified_address_rivals=qualifying))
    # Fixed, country-agnostic rule family, including no change. Select on tune.
    zero=np.zeros(len(rows),bool)
    veto={'none':zero,'address10':flags[:,0],'address20':flags[:,1],
          'empty80':flags[:,2]&(prob<.8),'empty90':flags[:,2]&(prob<.9),
          'address10_empty80':flags[:,0]|(flags[:,2]&(prob<.8)),
          'address20_empty80':flags[:,1]|(flags[:,2]&(prob<.8))}
    metrics={};vectors={}
    for split,(a,ix,sl) in data.items():
        metrics[split]={};vectors[split]={}
        for name,v in veto.items():
            k=np.zeros(len(a['y']),bool);k[ix]=(accepted&~v)[sl];vectors[split][name],metrics[split][name]=summary(a,k)
    selected=max(veto,key=lambda n:(metrics['tune'][n]['macro_f05'],n=='none'))
    report=dict(complete=True,selected_on_tune=selected,tune_grid=metrics['tune'],dev_at_selected=metrics['dev'][selected],
        comparisons={s:bootstrap(vectors[s]['none'],vectors[s][selected]) for s in data},
        label_free_families=len(families),selected_family_records=sum(map(len,families.values())),global_reference_records=scanned,
        affected_pair_counts={k:int((v&accepted).sum()) for k,v in veto.items()},input_sha256=bound,
        no_matches_added=True,deployed=False,limitations=['Reused owner OOF; no French labels.',
        'Only exact normalized name families, not exhaustive fuzzy/address rival search.',
        'Alternative existence does not prove its ownership. Abstention tested, never label-directed reassignment.'])
    write(OUT/'flagged_pairs.json',feature_rows);write(OUT/'report.json',report);np.savez_compressed(OUT/'features.npz',flags=flags)
    write(OUT/'completion.json',dict(complete=True,report_sha256=digest(OUT/'report.json')))


if __name__=='__main__':run()
