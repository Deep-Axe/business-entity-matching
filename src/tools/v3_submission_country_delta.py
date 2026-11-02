"""Stream country-level changes between two validated full submissions.

This measures behavior, not accuracy: there are no test labels, including France.
No submission modifications and no country policy selection from public scores.
"""
import argparse
from collections import Counter,defaultdict
import csv
from itertools import zip_longest
import json
from pathlib import Path

from v3_transfer_data import ROOT,digest,write

SOURCE=ROOT/'6ab10eb3b23ba_student_resource/student_resource/dataset/test/test_source1.tsv'


def target_set(value):
    parts=value.split(',') if value else []
    if len(set(parts))!=len(parts):raise ValueError('Duplicate target in an owner list')
    return set(parts)


def verify(folder):
    folder=Path(folder);m=folder/'matching_results.tsv';receipt=json.loads((folder/'completion.json').read_text())
    stream=json.loads((folder/'stream_validation.json').read_text())
    official=json.loads((folder/'official_matching_validation.json').read_text())
    if (not receipt['ready_for_submission'] or not stream['valid'] or stream['references']!=1732544
            or not official['valid'] or not official['check_ids'] or official['errors'] or official['warnings']):
        raise ValueError('Require complete validated submissions')
    h=digest(m)
    if any(v!=h for v in (receipt['output_sha256']['matching_results.tsv'],
                          stream['sha256']['matching_results.tsv'],official['sha256']['matching_results.tsv'])):
        raise ValueError('Validated TSV changed')
    return m,h


def verify_peer(folder):
    folder=Path(folder).resolve();root=ROOT/'artifacts/improvement_v3/peer_submission_validation_v1'
    report=json.loads((root/'report.json').read_text())
    matches=[v for v in report['variants'] if Path(v['source']).resolve()==folder]
    if not report['complete'] or len(matches)!=1:raise ValueError('Peer source has no exact validation receipt')
    item=matches[0];validated=root/folder.name
    stream=validated/'stream.json';official=validated/'official.json'
    if digest(stream)!=item['stream_sha256'] or digest(official)!=item['official_sha256']:
        raise ValueError('Peer validation receipts changed')
    s=json.loads(stream.read_text());o=json.loads(official.read_text());path=folder/'matching_results.tsv';h=digest(path)
    if (not item['structural_validation_passed'] or item['references']!=1732544
            or not s['valid'] or s['references']!=1732544 or not o['valid'] or not o['check_ids']
            or o['errors'] or o['warnings'] or any(v!=h for v in
                (item['matching_sha256'],s['sha256']['matching_results.tsv'],o['sha256']['matching_results.tsv']))):
        raise ValueError('Peer TSV is not the exact fully validated file')
    return path,h


def run(baseline,candidate,output,peer_baseline=False):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    old,oh=(verify_peer(baseline) if peer_baseline else verify(baseline));new,nh=verify(candidate)
    stats=defaultdict(Counter);hist=defaultdict(lambda:dict(old=Counter(),new=Counter()))
    with SOURCE.open() as source,old.open() as left,new.open() as right:
        readers=[csv.DictReader(f,delimiter='\t') for f in (source,left,right)]
        for ref,a,b in zip_longest(*readers):
            if ref is None or a is None or b is None:raise ValueError('TSV coverage length differs')
            if ref['entity_id']!=a['source1_entity_id'] or ref['entity_id']!=b['source1_entity_id']:
                raise ValueError('TSV order/owner coverage differs from source')
            x,y=target_set(a['matched_entity_ids']),target_set(b['matched_entity_ids']);country=ref['country']
            c=stats[country];c['owners']+=1;c['old_matches']+=len(x);c['new_matches']+=len(y)
            c['old_empty']+=not x;c['new_empty']+=not y;c['changed_owners']+=x!=y
            c['added_edges']+=len(y-x);c['removed_edges']+=len(x-y);c['shared_edges']+=len(x&y)
            c['became_empty']+=bool(x and not y);c['became_matched']+=bool(y and not x)
            hist[country]['old'][str(len(x))]+=1;hist[country]['new'][str(len(y))]+=1
    if sum(c['owners'] for c in stats.values())!=1732544:raise ValueError('Unexpected complete owner count')
    if digest(old)!=oh or digest(new)!=nh:raise ValueError('Submission changed during comparison')
    countries={k:dict(v,old_matches_per_owner=v['old_matches']/v['owners'],
        new_matches_per_owner=v['new_matches']/v['owners'],match_count_histogram=hist[k]) for k,v in stats.items()}
    write(output,dict(complete=True,baseline=str(old),candidate=str(new),
        baseline_sha256=oh,candidate_sha256=nh,references=1732544,countries=countries,
        labels_used=False,accuracy_evaluated=False,
        note='Behavioral change only; added matches may be right or wrong, and removed matches may be right or wrong. France accuracy remains unknown.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('baseline','candidate','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--peer-baseline',action='store_true',help='Require separately retained full validation receipts for other-session baseline')
    run(**vars(p.parse_args()))
