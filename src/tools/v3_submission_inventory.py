"""Publish a small, continuously verified list of ready TSVs; never upload.

Receipt presence alone is insufficient: require complete coverage, clean official
ID validation, and agreement of the current matching file's SHA256 in all three
receipts. Hashes are cached only while size/mtime/ctime are unchanged.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from v3_transfer_data import ROOT, digest, write


SPECS = [
    ('100k + owner assignment', 'duf-fel_full_output_v3_100k_owner_v1',
     None, None, .958, .971091326095),
    ('200k raw', 'duf-fel_full_output_v2_200k',
     'artifacts/duf-fel_bundle_v2_200k/completion.json',
     'artifacts/improvement_v2/inference/completion_200k_v1', None, .9715227041),
    ('200k + owner assignment', 'duf-fel_full_output_v3_200k_owner_v1',
     None, None, None, None),
    ('1.3M-pair E5 + edit/CPU stacker + owner assignment (validated peer precision)',
     'duf-fel_full_output_v4_neural1300k_peerhalf_v1', None, None, None, None),
    ('Neural CPU-only follow-on (only if selected and quality-gated)',
     'duf-fel_full_output_v4_neural1300k_cpu_upgrade_v1', None, None, None, None),
]


def inspect(spec, hash_file=digest):
    name, directory, receipt, validation, public, development = spec
    out = ROOT/'artifacts'/directory
    result = dict(name=name, matching=str(out/'matching_results.tsv'),
                  ready=False, public_score_reported_by_user=public,
                  development_macro_f05=development)
    completion = ROOT/receipt if receipt else out/'completion.json'
    checks = ROOT/validation if validation else out
    paths = [completion, checks/'stream_validation.json', checks/'official_matching_validation.json']
    if not all(p.exists() for p in paths):
        result['status'] = 'Waiting for completion and both validation receipts'
        return result
    c, s, o = [json.loads(p.read_text()) for p in paths]
    if not (c.get('ready_for_submission') is True and s.get('valid') is True
            and s.get('references') == 1732544 and o.get('valid') is True
            and o.get('check_ids') is True and o.get('errors') == [] and o.get('warnings') == []):
        result['status'] = 'Not ready: incomplete coverage or validation'
        return result
    path = out/'matching_results.tsv'
    if not path.is_file():
        result['status'] = 'Not ready: matching TSV missing'
        return result
    value = hash_file(path)
    if not all(d.get(k, {}).get('matching_results.tsv') == value
               for d, k in [(c, 'output_sha256'), (s, 'sha256'), (o, 'sha256')]):
        result['status'] = 'Not ready: current TSV differs from validated receipt'
        return result
    result.update(ready=True, status='Ready; full coverage and official matching/ID validation passed',
                  references=1732544, sha256=value)
    if 'development' in c:
        result['development_macro_f05'] = c['development'].get('macro_f05', development)
    return result


def run(minutes=720, interval=30):
    end=time.monotonic()+minutes*60;cache={};previous=None
    def cached(path):
        s=path.stat();key=(s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if path not in cache or cache[path][0]!=key:
            value=digest(path);after=path.stat()
            if key!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
                raise ValueError('Matching file changed during integrity check')
            cache[path]=(key,value)
        return cache[path][1]
    while True:
        rows=[]
        for spec in SPECS:
            try:rows.append(inspect(spec,cached))
            except (OSError,ValueError,KeyError) as error:
                rows.append(dict(name=spec[0],ready=False,status='Check failed: '+str(error)))
        signature=json.dumps(rows,sort_keys=True)
        if signature!=previous:
            stamp=datetime.now(timezone.utc).isoformat()
            write(ROOT/'artifacts/improvement_v3/submission_inventory.json',
                  dict(utc=stamp,submissions=rows,uploaded_to_portal=False))
            lines=['# Submission files', '', 'Last change (UTC): '+stamp, '',
                   'Only **Ready** entries below are eligible to upload. No upload is automated.',
                   'Development scores are not public-score predictions; French accuracy is unmeasured.', '']
            for row in rows:
                lines += ['## '+row['name'], '', row['status']+'.', '']
                if row['ready']:
                    lines += ['TSV: ['+Path(row['matching']).name+']('+row['matching']+')', '',
                              'SHA256: `'+row['sha256']+'`', '']
                if row.get('public_score_reported_by_user') is not None:
                    lines += ['Public score reported by you: '+str(row['public_score_reported_by_user'])+'.', '']
                elif row.get('development_macro_f05') is not None:
                    lines += ['Development only: '+str(row['development_macro_f05'])+'. Public score unknown.', '']
            tmp=ROOT/'SUBMISSIONS_READY.md.tmp';tmp.write_text('\n'.join(lines));tmp.replace(ROOT/'SUBMISSIONS_READY.md')
            print(json.dumps(dict(utc=stamp,ready=[r['name'] for r in rows if r['ready']])),flush=True)
            previous=signature
        if time.monotonic()>=end:return
        time.sleep(min(interval,max(0,end-time.monotonic())))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--minutes',type=float,default=720);p.add_argument('--interval',type=float,default=30)
    a=p.parse_args()
    if a.minutes<0 or not 1<=a.interval<=60:p.error('Finite nonnegative minutes and interval 1–60 required')
    run(**vars(a))
