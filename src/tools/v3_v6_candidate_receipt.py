"""Export exact retained neural-matcher inputs, never post-match pruning.

Record upstream CPU scoring separately. The exported set includes every pair
fed to the parent neural matcher, not only the Kaggle uncertainty band. This
does not change any predictions or replace the peer's shared candidate file.
"""
import csv
import gc
import pickle
from pathlib import Path
import runpy
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from v3_neural_test_pipeline import ROOT, DATA, SCRATCH, resources, read
from v3_transfer_data import digest, write

OUT=ROOT/'artifacts/improvement_v3/v6_model_input_candidates_v1'
SOURCE=SCRATCH/'sub_v6_cC_band/matching_results.tsv'
EXPECTED='20d58a44a56c28145abaa9d5598ae71580d3318c8cbbf3dd87cefdec19a20aa1'


def official():
    validator=DATA.parent.parent/'utils/validate_submission.py'
    paths=[OUT/'matching_results.tsv',OUT/'candidate_pairs.tsv',validator]+[DATA/f'test_source{i}.tsv' for i in (1,2,3)]
    hashes={str(p):digest(p) for p in paths}
    errors,warnings=runpy.run_path(str(validator))['validate'](str(paths[0]),str(paths[1]),str(DATA),check_ids=True)
    if any(digest(p)!=h for p,h in hashes.items()):raise ValueError('Validator input changed')
    write(OUT/'official_both.json',dict(valid=not errors and not warnings,errors=errors,warnings=warnings,check_ids=True,input_sha256=hashes))
    if errors or warnings:raise ValueError('Both-file validation did not pass cleanly')


def run():
    OUT.mkdir(parents=True,exist_ok=False);resources()
    if digest(SOURCE)!=EXPECTED:raise ValueError('Matching file changed')
    db=sqlite3.connect(OUT/'assembly.sqlite3');db.execute('PRAGMA cache_size=-8192');db.execute('PRAGMA temp_store=FILE')
    db.execute('CREATE TABLE candidates(owner TEXT PRIMARY KEY,targets TEXT,country TEXT) WITHOUT ROWID')
    hashes={};pairs=0;country_pairs=Counter()
    for part in range(128):
        resources();f=SCRATCH/f'test/part_{part:03}.pkl';hashes[str(f)]=digest(f)
        rows=pickle.load(f.open('rb'));groups=defaultdict(list);countries={}
        for r in rows:
            if r[4]<.01 or not r[1].startswith(('S2-','S3-')):raise ValueError('Unexpected neural input route')
            groups[r[0]].append(r[1]);countries[r[0]]=r[3];country_pairs[r[3]]+=1
        if any(len(ts)!=len(set(ts)) for ts in groups.values()):raise ValueError('Repeated candidate pair')
        with db:db.executemany('INSERT INTO candidates VALUES(?,?,?)',[(o,','.join(sorted(ts)),countries[o]) for o,ts in groups.items()])
        pairs+=len(rows);write(OUT/'progress.json',dict(stage='exporting_actual_neural_inputs',partitions=part+1,pairs=pairs))
        del rows,groups,countries
    if pairs!=10258586:raise ValueError('Unexpected full neural candidate count')
    counts=defaultdict(Counter);hist=Counter();refs=matched=0
    with (DATA/'test_source1.tsv').open(newline='') as source,SOURCE.open(newline='') as matching,(OUT/'candidate_pairs.tsv').open('w') as out:
        src=csv.DictReader(source,delimiter='\t');mat=iter(csv.DictReader(matching,delimiter='\t'));out.write('source1_entity_id\tcandidate_entity_ids\n')
        for r in src:
            m=next(mat,None)
            if m is None or m['source1_entity_id']!=r['entity_id']:raise ValueError('Matching/source alignment differs')
            c=db.execute('SELECT targets,country FROM candidates WHERE owner=?',(r['entity_id'],)).fetchone()
            if c and c[1]!=r['country']:raise ValueError('Country mismatch')
            ts=c[0].split(',') if c else [];accepted=set(filter(None,m['matched_entity_ids'].split(',')))
            if not accepted.issubset(ts):raise ValueError('Match not in actual neural input set')
            out.write(r['entity_id']+'\t'+','.join(ts)+'\n');refs+=1;matched+=len(accepted);hist[len(ts)]+=1
            counts[r['country']].update(owners=1,candidates=len(ts),matches=len(accepted),zero_candidates=int(not ts))
        if next(mat,None) is not None:raise ValueError('Extra matching rows')
    if refs!=1732544 or sum(k*v for k,v in hist.items())!=pairs:raise ValueError('Candidate/source coverage differs')
    db.close();gc.collect();shutil.copyfile(SOURCE,OUT/'matching_results.tsv')
    if digest(OUT/'matching_results.tsv')!=EXPECTED:raise ValueError('Prediction copy changed')
    write(OUT/'progress.json',dict(stage='official_validation_both_files'))
    with (OUT/'official.log').open('w') as log:
        subprocess.run([sys.executable,__file__,'official'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1500)
    for p,h in hashes.items():
        if digest(p)!=h:raise ValueError('Neural inputs changed during export')
    assembly=read(ROOT/'artifacts/duf-fel_full_output_v2_200k/assembly.json')
    write(OUT/'completion.json',dict(complete=True,matching_sha256=EXPECTED,candidate_sha256=digest(OUT/'candidate_pairs.tsv'),
        references=refs,final_matching_input_pairs=pairs,mean_candidates_per_owner=pairs/refs,candidate_count_histogram=hist,countries=counts,
        matched_pairs=matched,all_final_matches_in_candidates=True,official_both_files_with_ids_passed=True,
        earlier_cpu_scored_pairs=assembly['candidate_pairs'],kaggle_rescored_band_pairs=922443,input_sha256=hashes,
        candidate_definition='All p100>=0.01 pairs fed to the parent neural matcher/final stacker, BEFORE final matching threshold and ownership. Includes unchanged fallback outside Kaggle band.',
        transparency='CPU classifier scores 101510800 early pairs before this learned filter. Disclose that inference cost and retain the upstream candidate artifact; do not present this export as the only pairwise ML scoring.',
        no_postmatching_pruning=True,peer_files_changed=False,uploaded_to_portal=False,final_package_complete=False))
    write(OUT/'progress.json',dict(stage='complete'))


if __name__=='__main__':
    official() if len(sys.argv)>1 and sys.argv[1]=='official' else run()
