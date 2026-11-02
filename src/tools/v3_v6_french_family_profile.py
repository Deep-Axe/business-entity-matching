"""Unlabeled v6 global-family risk profile on frozen 400-owner/country sample."""
from collections import Counter,defaultdict
from functools import lru_cache
import pickle
import numpy as np
from rapidfuzz import fuzz
from ber.store import connect,lookup
from v3_neural_test_pipeline import ROOT,SCRATCH,resources,read
from v3_public0976_test_profile import selected_matches
from v3_structural_features import core_name,folded,address_view,house_number
from v3_transfer_data import digest,write

OUT=ROOT/'artifacts/improvement_v3/v6_french_family_profile_v1'


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    selection=ROOT/'artifacts/improvement_v3/recovery_test_volume_v1/selection.json'
    selected={r['entity_id'] for r in read(selection)['selected']}
    source=SCRATCH/'sub_v6_cC_band/matching_results.tsv';accepted=selected_matches(source,selected)
    db=connect(ROOT/'artifacts/test_store_v2');db.execute('PRAGMA cache_size=-8192')
    left=lookup(db,'anchors','entity_id',list(selected));targets={t for ts in accepted.values() for t in ts};right=lookup(db,'records','entity_id',list(targets))
    scores={}
    for part in range(128):
        resources();p=SCRATCH/f'test/part_{part:03}.pkl';rows=pickle.load(p.open('rb'));v=np.load(str(p)+'.stack_cC_band.npy')
        for r,s in zip(rows,v):
            if r[0] in selected and r[1] in accepted[r[0]]:scores[r[:2]]=float(s)
        if part%16==15:write(OUT/'progress.json',dict(stage='sample_scores',partitions=part+1))
    if len(scores)!=sum(map(len,accepted.values())):raise ValueError('Sample scores missing')
    core=lru_cache(maxsize=50000)(core_name);wanted=set()
    for o,ts in accepted.items():
        for t in ts:
            for name in (left[o]['name'],right[t]['name']):
                c=core(name)
                if c:wanted.add((left[o]['country'],c))
    families=defaultdict(list)
    for n,r in enumerate(db.execute('SELECT entity_id,country,name,address FROM anchors'),1):
        key=(r['country'],core(r['name']))
        if key in wanted:families[key].append(tuple(r))
    if n!=1732544:raise ValueError('Full test reference coverage differs')
    stats=defaultdict(Counter);examples=[]
    for o,ts in accepted.items():
        l=left[o];country=l['country'];stats[country]['owners']+=1
        for t in ts:
            r=right[t];stats[country]['accepted']+=1
            tn=folded(r['name']);ta=address_view(r['address'],country);th=house_number(r['address'],country)
            ln=folded(l['name']);la=address_view(l['address'],country);lh=house_number(l['address'],country)
            own_name=fuzz.ratio(ln,tn);own_addr=fuzz.token_set_ratio(la,ta) if la and ta else 0.
            rivals={x[0]:x for c in (core(l['name']),core(r['name'])) if c for x in families.get((country,c),[]) if x[0]!=o}
            margin=-100.;empty_tie=False;best=None
            for eid,cc,name,address in rivals.values():
                ns=fuzz.ratio(folded(name),tn)
                if not ta and ns>=95 and ns>=own_name-.5:empty_tie=True
                if ta and th and lh and lh!=th and house_number(address,cc)==th and ns>=90 and ns>=own_name-3:
                    addr=fuzz.token_set_ratio(address_view(address,cc),ta)
                    if addr>=95 and addr-own_addr>margin:margin=addr-own_addr;best=(eid,name,address)
            flags=dict(shared_family=bool(rivals),stronger_address_rival=margin>=10,
                       empty_name_ambiguous=empty_tie,tune_selected_empty80=empty_tie and scores[o,t]<.8)
            for k,v in flags.items():stats[country][k]+=int(v)
            if flags['stronger_address_rival'] or flags['tune_selected_empty80']:
                examples.append(dict(owner=o,target=t,country=country,probability=scores[o,t],reference=dict(l),target_record=dict(r),
                    alternative=best,address_margin=margin,flags=flags,truth_unknown=True))
    db.close();write(OUT/'examples.json',examples)
    write(OUT/'report.json',dict(complete=True,statistics=stats,sample_sha256=digest(selection),matching_sha256=digest(source),
        accuracy_measured=False,labels_used=False,submitted_or_modified=False,
        note='400 independently selected owners per country; all text-selected same-core rivals. Flags are not errors or expected score gains.'))


if __name__=='__main__':run()
