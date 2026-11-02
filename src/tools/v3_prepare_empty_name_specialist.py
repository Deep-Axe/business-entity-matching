"""Retain naturally empty-address training pairs for a bounded name specialist.

No address is artificially dropped, no label is invented, and no tune/dev pair
enters training. The data is drawn from the original supervised CE training pool,
so it is not claimed to be unseen by the existing encoder. No GPU work here.
"""
from collections import Counter,defaultdict
from pathlib import Path
import pickle

from ber.data import read_tsv
from v3_neural_test_pipeline import read,resources
from v3_transfer_data import ROOT,digest,write


def missing(value):
    return not value.strip() or value.strip().upper() in ('NULL','<NULL>')


def run():
    out=ROOT/'artifacts/improvement_v3/empty_name_specialist_data_v1';out.mkdir(parents=True,exist_ok=True)
    census=ROOT/'artifacts/improvement_v3/neural_training_pool_profile_v1/report.json';report=read(census)
    source=ROOT/'artifacts/improvement_v2/selection/inputs_v1/train_200k/references.tsv'
    if not report['complete'] or digest(source)!=report['training_owner_selection_sha256']:
        raise ValueError('Training-pool provenance incomplete')
    identity=dict(census_sha256=digest(census),code_sha256=digest(__file__),
        selection='All naturally empty/NULL target addresses from original 200k-owner CE candidate pool',
        serialization='Raw left business name paired with raw target business name; no addresses or ID values',
        addresses_artificially_dropped=False,evaluation_labels_read=False)
    if (out/'identity.json').exists():
        if read(out/'identity.json')!=identity:raise ValueError('Empty-name training recipe changed')
    else:write(out/'identity.json',identity)
    if (out/'completion.json').exists():
        done=read(out/'completion.json')
        if done['identity_sha256']!=digest(out/'identity.json') or done['pairs_sha256']!=digest(out/'pairs.pkl'):
            raise ValueError('Retained name-specialist data changed')
        return
    owners={r['entity_id'] for r in read_tsv(source)};rows=[];seen=set();counts=defaultdict(Counter);ambiguity=defaultdict(Counter)
    for index,(name,meta) in enumerate(sorted(report['files'].items()),1):
        resources();path=Path(name)
        if digest(path)!=meta['sha256']:raise ValueError('Original training partition changed')
        part=pickle.load(path.open('rb'))
        for r in part:
            if r[0] not in owners:raise ValueError('Pair is not from known training owners')
            if not missing(r[8]):continue
            if r[:2] in seen:raise ValueError('Duplicate specialist training edge')
            seen.add(r[:2]);rows.append((r[0],r[1],r[2],r[3],r[5],r[7]))
            label='positive' if r[2] else 'negative';counts['all'][label]+=1;counts[r[3]][label]+=1
            ambiguity[(r[5],r[7])][label]+=1
        del part
        if digest(path)!=meta['sha256']:raise ValueError('Training partition changed during read')
        write(out/'progress.json',dict(stage='retaining_naturally_empty_training_rows',files_done=index,files_total=len(report['files']),pairs=len(rows)))
    expected=report['counts']['empty_target_address']
    if dict(counts['all'])!=expected:raise ValueError('Specialist counts do not replay training census')
    ambiguous=[v for v in ambiguity.values() if v['positive'] and v['negative']]
    with (out/'pairs.pkl').open('wb') as handle:pickle.dump(rows,handle)
    write(out/'completion.json',dict(complete=True,training_only=True,pairs=len(rows),counts=dict(counts),
        identity_sha256=digest(out/'identity.json'),pairs_sha256=digest(out/'pairs.pkl'),
        unique_training_owners=len({r[0] for r in rows}),exact_name_pair_groups=len(ambiguity),
        exact_input_groups_with_mixed_labels=len(ambiguous),
        positive_rows_in_mixed_groups=sum(v['positive'] for v in ambiguous),
        negative_rows_in_mixed_groups=sum(v['negative'] for v in ambiguous),new_training_started=False,
        note='Name-only duplicates may be intrinsically ambiguous; retain their genuine labels, no forced pseudo-matches. Earlier encoder may have seen sampled rows.'))


if __name__=='__main__':run()
