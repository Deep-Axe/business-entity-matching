"""Read-only full-file validation and exact v6 test band/fallback audit."""
from collections import Counter, defaultdict
import csv
from itertools import zip_longest
import sys
import subprocess
import numpy as np
from v3_neural_test_pipeline import DATA, SCRATCH, resources, read
from v3_transfer_data import ROOT, digest, write

OUT=ROOT/'artifacts/improvement_v3/v6_band_validation_v1'
SOURCE=SCRATCH/'sub_v6_cC_band'
BASE=ROOT/'artifacts/public_best_0976_20260926/matching_results.tsv'
EXPECTED='20d58a44a56c28145abaa9d5598ae71580d3318c8cbbf3dd87cefdec19a20aa1'


def run():
    OUT.mkdir(parents=True,exist_ok=False);resources()
    identity=dict(code_sha256=digest(__file__),matching_sha256=digest(SOURCE/'matching_results.tsv'),
        baseline_sha256=digest(BASE),report_sha256=digest(SOURCE/'report.json'),band_index_sha256=digest(SCRATCH/'band_index.npy'))
    if identity['matching_sha256']!=EXPECTED or identity['baseline_sha256']!='5a0be554e60f88e763de95a5b4cc8f30f7cee352067ba24c16947e6319752118':raise ValueError('Frozen TSV changed')
    write(OUT/'identity.json',identity)
    for stage,args in (
        ('stream',['tools/validate_full_submission.py','--test-dir',DATA,'--output',SOURCE,'--report',OUT/'stream.json']),
        ('official',['tools/package_v2_200k_submission.py','official-validate','--validator',DATA.parent.parent/'utils/validate_submission.py',
                    '--test-dir',DATA,'--output',SOURCE,'--report',OUT/'official.json'])):
        write(OUT/'progress.json',dict(stage=stage))
        with (OUT/(stage+'.log')).open('w') as log:
            subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
    stream,official=read(OUT/'stream.json'),read(OUT/'official.json')
    if not(stream['valid'] and stream['references']==1732544 and official['valid'] and official['check_ids']
           and not official['errors'] and not official['warnings']):raise ValueError('Full validator failed')
    idx=np.load(SCRATCH/'band_index.npy');total=changed=band=0;hashes={}
    if idx.shape!=(922443,2) or set(idx[:,0])!=set(range(128)):raise ValueError('Band partition coverage differs')
    for part in range(128):
        resources();write(OUT/'progress.json',dict(stage='checking_all_saved_band_merges',partition=part))
        f=SCRATCH/f'test/part_{part:03}.pkl'
        paths=[str(f)+'.'+tag+'.npy' for tag in ('stack_e5_cont1300k_nocty_nf2','stack_cC_nocty_nf2','stack_cC_band')]
        arrays=[]
        for p in paths:hashes[p]=digest(p);arrays.append(np.load(p))
        old,new,merged=arrays;ix=idx[idx[:,0]==part,1];mask=np.zeros(len(old),bool);mask[ix]=True
        if len(np.unique(ix))!=len(ix) or not np.array_equal(mask,(old>=.05)&(old<.98)):raise ValueError('Band does not match declared scores')
        expected=old.copy();expected[mask]=new[mask]
        if not np.array_equal(expected,merged) or not np.isfinite(merged).all():raise ValueError('Band merge or fallback differs')
        total+=len(old);band+=int(mask.sum());changed+=int(np.count_nonzero(merged!=old))
    counts=defaultdict(Counter)
    with BASE.open(newline='') as left,(SOURCE/'matching_results.tsv').open(newline='') as right,(DATA/'test_source1.tsv').open(newline='') as source:
        for a,b,s in zip_longest(csv.DictReader(left,delimiter='\t'),csv.DictReader(right,delimiter='\t'),csv.DictReader(source,delimiter='\t')):
            if a is None or b is None or s is None or a['source1_entity_id']!=b['source1_entity_id'] or a['source1_entity_id']!=s['entity_id']:raise ValueError('Full source/TSV row alignment differs')
            aa=set(filter(None,a['matched_entity_ids'].split(',')));bb=set(filter(None,b['matched_entity_ids'].split(',')))
            counts[s['country']].update(owners=1,changed_owners=int(aa!=bb),added=len(bb-aa),removed=len(aa-bb),before=len(aa),after=len(bb),empty_before=int(not aa),empty_after=int(not bb))
    if digest(SOURCE/'matching_results.tsv')!=EXPECTED or digest(SCRATCH/'band_index.npy')!=identity['band_index_sha256']:raise ValueError('Inputs changed during validation')
    write(OUT/'completion.json',dict(complete=True,ready_as_experimental_submission=True,accuracy_improvement_established=False,
        matching=str(SOURCE/'matching_results.tsv'),matching_sha256=EXPECTED,references=sum(v['owners'] for v in counts.values()),
        full_coverage_and_official_validation=True,test_partitions=128,routed_pairs=total,band_pairs=band,changed_scores=changed,
        all_outside_band_scores_preserved=True,country_changes=counts,score_input_sha256=hashes,
        uploaded_to_portal=False))


if __name__=='__main__':run()
