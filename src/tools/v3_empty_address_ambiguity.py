"""Train-fold-only audit of indistinguishable empty-address target text.

No model fitting, new submission, or reserved-holdout labels. Counts indicate
observable ambiguity; they are not an overall F0.5 impossibility bound.
"""
from collections import Counter,defaultdict
import argparse
import time

from ber.store import connect
from v3_transfer_data import ROOT,write


def run(view='exact'):
    out=ROOT/('artifacts/improvement_v3/empty_address_ambiguity_v1' if view=='exact' else f'artifacts/improvement_v3/empty_address_ambiguity_{view}_v1')
    out.mkdir(parents=True,exist_ok=False);db=connect(ROOT/'artifacts/train_store_v2')
    db.execute('PRAGMA cache_size=-8192');start=time.monotonic()
    totals=defaultdict(Counter);examples=[]
    if view=='normalized':
        from ber.text import normalize
        key=normalize
    elif view=='core':
        from v3_structural_features import core_name
        key=core_name
    else:key=lambda x:x
    db.create_function('name_key',1,key,deterministic=True)
    sql="""SELECT r.country,r.source,name_key(r.name) AS name,count(*) AS n,
        count(l.target_id) AS positive,count(DISTINCT l.reference_id) AS owners
        FROM records r LEFT JOIN labels l ON l.target_id=r.entity_id AND l.fold='train'
        WHERE r.fold='train' AND trim(r.address)=''
        GROUP BY r.country,r.source,name_key(r.name)"""
    for row in db.execute(sql):
        c=totals[row['country']];n,pos,owners=row['n'],row['positive'],row['owners'];negative=n-pos
        c.update(exact_text_groups=1,targets=n,positive_targets=pos,unmatched_targets=negative)
        if n>1:c.update(duplicate_text_groups=1,targets_in_duplicate_groups=n)
        ambiguous=owners>1 or (pos>0 and negative>0)
        if ambiguous:
            c.update(ambiguous_groups=1,ambiguous_targets=n,ambiguous_positive_targets=pos,
                     ambiguous_unmatched_targets=negative)
            if len(examples)<15:
                examples.append(dict(country=row['country'],source=row['source'],name=row['name'],
                    targets=n,positive_targets=pos,distinct_true_owners=owners,unmatched_targets=negative))
    db.close()
    result=dict(complete=True,scope='Only train-fold targets and train-fold ownership labels',
        grouping=f'{view} target name, empty address, country and source; IDs deliberately excluded',
        countries=dict(totals),examples=examples,seconds=time.monotonic()-start,
        interpretation=('Ambiguous name-view groups are observed; this alone gives no macro-score ceiling and may reflect information lost by normalization.'
            if any(c['ambiguous_groups'] for c in totals.values()) else
            'No cross-owner or positive-versus-unmatched ambiguity was observed in these exact name-view groups. This does not prove all missing-address pairs are identifiable or that names are globally unique.'),
        reserved_audit_scored=False,model_or_submission_changed=False)
    write(out/'report.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--view',choices=['exact','normalized','core'],default='exact')
    run(**vars(p.parse_args()))
