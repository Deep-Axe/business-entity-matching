"""Read-only structural validation of two other-session submission files.

Receipts live in this project. No peer file is altered, no model-quality score
is endorsed, and nothing is uploaded. Process variants serially within a cgroup.
"""
from pathlib import Path
import subprocess
import sys

from v3_neural_test_pipeline import DATA, read, resources
from v3_transfer_data import ROOT, digest, write


def run():
    output=ROOT/'artifacts/improvement_v3/peer_submission_validation_v1'; output.mkdir(parents=True,exist_ok=True)
    results=[]
    for tag in ('sub_v4_stack_all','sub_v4_stack_usin_frold'):
        source=Path('work/ce')/tag; work=output/tag; work.mkdir(exist_ok=True)
        resources(); original=read(source/'report.json'); before=digest(source/'matching_results.tsv')
        if before!=original['matching_sha256'] or original['references']!=1732544:
            raise ValueError('Peer assembly receipt does not bind the full TSV')
        identity=dict(source=str(source),matching_sha256=before,assembly_sha256=digest(source/'report.json'))
        if (work/'identity.json').exists() and read(work/'identity.json')!=identity:
            raise ValueError('Peer submission changed during validation')
        write(work/'identity.json',identity)
        for stage,args in (
            ('stream',['tools/validate_full_submission.py','--test-dir',DATA,'--output',source,'--report',work/'stream.json']),
            ('official',['tools/package_v2_200k_submission.py','official-validate',
                '--validator',DATA.parent.parent/'utils/validate_submission.py','--test-dir',DATA,
                '--output',source,'--report',work/'official.json'])):
            write(output/'progress.json',dict(stage=stage,variant=tag,model_quality_endorsed=False))
            resources()
            with (work/(stage+'.log')).open('a') as log:
                subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                               timeout=900,check=True)
        stream,official=read(work/'stream.json'),read(work/'official.json')
        valid=(stream['valid'] and stream['references']==1732544 and official['valid'] and
               not official['errors'] and not official['warnings'] and official['check_ids'])
        if not valid or any(r['sha256']['matching_results.tsv']!=before for r in (stream,official)):
            raise ValueError('Peer TSV failed full structural or ID validation')
        if digest(source/'matching_results.tsv')!=before:
            raise ValueError('Peer TSV changed after validation')
        result=dict(**identity,structural_validation_passed=True,references=1732544,
            stream_sha256=digest(work/'stream.json'),official_sha256=digest(work/'official.json'),
            model_quality_independently_reproduced=False,uploaded_to_portal=False,
            caveat='Different other-session stacker/fitting protocol. Structural validation is not a held-out quality measurement or recommendation.')
        write(work/'completion.json',result); results.append(result)
    write(output/'report.json',dict(complete=True,variants=results,peer_files_modified=False,uploaded_to_portal=False))
    write(output/'progress.json',dict(stage='complete',variants=len(results),model_quality_endorsed=False))


if __name__=='__main__':run()
