"""Compare an additional 200k-derived alias channel on fixed tune/dev owners.

Existing candidates are retained. Larger-dictionary retrieval precedes label
inspection, uses the full gallery, and cannot generate a submission by itself.
"""
from collections import Counter
from pathlib import Path
import time

from ber.expansion import ExpansionRetriever
from ber.store import connect,lookup,record_row
from ber.text import Record
from v3_neural_test_pipeline import read,resources
from v3_transfer_data import ROOT,digest,write


def run():
    lexicon=ROOT/'artifacts/improvement_v3/lexicon200k_v1'
    origin=ROOT/'artifacts/improvement_v3/retrieval_recovery_400_v1'
    output=ROOT/'artifacts/improvement_v3/lexicon_recovery_400_v1';output.mkdir(parents=True,exist_ok=True)
    end=time.monotonic()+3*3600
    while not (lexicon/'manifest.json').exists():
        if time.monotonic()>=end:raise TimeoutError('Dictionary not ready in finite pilot window')
        time.sleep(20)
    manifest=read(lexicon/'manifest.json')
    if not manifest['complete'] or digest(lexicon/'lexicon.json')!=manifest['lexicon_sha256']:
        raise ValueError('Incomplete new dictionary')
    identity=dict(lexicon_sha256=digest(lexicon/'lexicon.json'),source_report_sha256=digest(origin/'report.json'),
        code_sha256=digest(__file__),full_gallery=True,top_k=20,label_based_ranking=False)
    if (output/'identity.json').exists():
        if read(output/'identity.json')!=identity:raise ValueError('Lexicon pilot identity changed')
    else:write(output/'identity.json',identity)
    retriever=ExpansionRetriever(ROOT/'artifacts/train_index_v2',lexicon,fold=None,max_df_ratio=.01,frequency_mode='gate')
    db=connect(ROOT/'artifacts/train_store_v2');db.execute('PRAGMA cache_size=-8192')
    start=time.monotonic();splits={}
    for split,fold in (('tune','tune'),('dev','eval')):
        totals=Counter();patterns=Counter();bycountry={};newcorrect=[]
        for path in sorted((origin/split).glob('[0-9]*.json')):
            resources()
            if time.monotonic()>=end:raise TimeoutError('Finite lexicon evaluation window expired')
            saved=read(path);owner=saved['owner'];left=lookup(db,'anchors','entity_id',[owner])[owner]
            if left['fold']!=fold:raise ValueError('Evaluation owner overlaps dictionary training fold')
            dest=output/split/path.name
            if dest.exists():row=read(dest)
            else:
                stamp=time.monotonic();found=retriever.candidates(Record.from_row(record_row(left)),20)
                targets=lookup(db,'records','row_id',[int(rid) for rid,_ in found]);retrieved={r['entity_id'] for r in targets.values()}
                if any(r['country']!=left['country'] for r in targets.values()):raise ValueError('Cross-country expansion')
                # Consult saved labels only after label-free retrieval.
                base=set(saved['baseline']);truth=set(saved['truth']);extra=retrieved-base;correct=extra&truth
                deeper=set().union(*(set(v) for v in saved['channels'].values()))
                positive=lookup(db,'records','entity_id',list(correct))
                flags=Counter()
                for r in positive.values():
                    flags['indic_target_name']+=any('\u0900'<=c<='\u0dff' for c in r['name'])
                    flags['empty_target_address']+=not r['address'].strip()
                row=dict(owner=owner,country=left['country'],additional_candidates=len(extra),additional_correct=len(correct),
                    recovered_beyond_deeper_union=len(correct-deeper),additional_ids=sorted(extra),correct_ids=sorted(correct),
                    patterns=dict(flags),seconds=time.monotonic()-stamp)
                write(dest,row)
            if row['owner']!=owner:raise ValueError('Resumed owner changed')
            totals.update(owners=1,additional_candidates=row['additional_candidates'],additional_correct=row['additional_correct'],
                recovered_beyond_deeper_union=row['recovered_beyond_deeper_union'])
            patterns.update(row['patterns']);newcorrect.extend((owner,t) for t in row['correct_ids'])
            bycountry.setdefault(row['country'],Counter()).update(owners=1,extra=row['additional_candidates'],correct=row['additional_correct'])
            if totals['owners']%25==0:write(output/'progress.json',dict(stage='retrieval_only',split=split,**totals,seconds=time.monotonic()-start))
        if totals['owners']!=400:raise ValueError('Pilot cohort changed')
        splits[split]=dict(totals=totals,overlapping_positive_patterns=patterns,countries=bycountry,correct_pairs=newcorrect)
    db.close();write(output/'report.json',dict(complete=True,identity_sha256=digest(output/'identity.json'),splits=splits,
        seconds=time.monotonic()-start,ready_for_submission=False,
        note='Full-gallery retrieval-only union on fixed 400 tune and 400 dev owners. No matching gain measured; new pairs need calibrated classification and owner handling. Production 12k dictionary unchanged.'))


if __name__=='__main__':run()
