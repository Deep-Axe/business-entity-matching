"""Portable CPU-only inference from the bundled immutable 200k release."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from ber.shards import file_hash
from ber.store import build_store
from ber.disk_index import build_index
from v2_200k_integrity import read_200k_release

ARTIFACTS = {'model': 'model/model.cbm', 'model_report': 'model/model_report.json',
    'training_manifest': 'model/training_manifest.json', 'release': 'model/release.json',
    'lexicon': 'model/lexicon/lexicon.json'}


def verify_bundle(package):
    package = Path(package)
    manifest = json.loads((package / 'bundle_manifest.json').read_text())
    if (manifest['schema_version'] != 1 or not 0 < manifest['threshold'] <= 1
            or manifest['training_references'] != 200000 or manifest['features'] != 76
            or manifest['gate_terms'] != 0 or manifest['owner_enabled']
            or manifest['artifacts'] != ARTIFACTS
            or not set(ARTIFACTS.values()).issubset(manifest['sha256'])):
        raise ValueError('Unexpected packaged inference recipe')
    source_files = {p.relative_to(package).as_posix() for p in (package / 'src').rglob('*.py')}
    if source_files != {name for name in manifest['sha256'] if name.startswith('src/') and name.endswith('.py')}:
        raise ValueError('Runtime source inventory differs from checksum manifest')
    for name, digest in manifest['sha256'].items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or (package / rel).is_symlink():
            raise ValueError('Unsafe bundle member')
        if file_hash(package / rel) != digest:
            raise ValueError(f'Packaged artifact changed: {name}')
    release_path = package / manifest['artifacts']['release']
    release = read_200k_release(release_path)
    if (release['threshold'] != manifest['threshold']
            or file_hash(release_path) != manifest['release_sha256']):
        raise ValueError('Portable threshold/release binding differs')
    for portable_key, release_key in (('model', 'model'), ('model_report', 'model_report'),
                                      ('training_manifest', 'prepared_train_manifest')):
        if file_hash(package / ARTIFACTS[portable_key]) != release['artifacts'][release_key]['sha256']:
            raise ValueError('Portable artifact differs from frozen release: ' + portable_key)
    return manifest


def run(args):
    package = Path(__file__).resolve().parents[1]
    manifest = verify_bundle(package)
    artifacts = {key: package / name for key, name in manifest['artifacts'].items()}
    data, work, output = map(lambda p: Path(p).resolve(), (args.data_dir, args.work_dir, args.output))
    if args.workers < 1:
        raise ValueError('workers must be positive')
    for i in (1, 2, 3):
        if not (data / 'test' / f'test_source{i}.tsv').is_file():
            raise FileNotFoundError(f'Missing original test_source{i}.tsv')
    if (work.exists() or output.exists()) and not args.resume:
        raise FileExistsError('Use new work/output directories, or --resume with unchanged inputs')
    work.mkdir(parents=True, exist_ok=True)
    store, index, raw = (work / name for name in ('test_store', 'test_index', 'predictions'))
    build_store(data, store, split='test', resume=args.resume, minimum_free_gib=8)
    build_index(store, index, threads=min(4, args.workers), heap_mb=256,
                commit_every=250000, resume=args.resume, minimum_free_gib=8)
    command = [sys.executable, str(package / 'src/v2_run_pipeline.py')]
    options = {'index': index, 'store': store, 'references': data / 'test/test_source1.tsv',
        'lexicon': artifacts['lexicon'].parent, 'model': artifacts['model'],
        'model-report': artifacts['model_report'], 'training-manifest': artifacts['training_manifest'],
        'output': raw, 'threshold': str(manifest['threshold']), 'gate-terms': '0', 'workers': args.workers,
        'weighting-policy': 'forward_gallery_index_idf_v1', 'minimum-free-gib': '8'}
    for key, value in options.items():
        command.extend(['--' + key, str(value)])
    if args.resume:
        command.append('--resume')
    subprocess.run(command, check=True)
    if not output.exists():
        subprocess.run([sys.executable, str(package / 'src/v2_200k_assemble.py'), 'assemble',
            '--sources', str(raw), '--references', str(data / 'test/test_source1.tsv'),
            '--release', str(artifacts['release']), '--output', str(output)], check=True)
    else:
        assembled = json.loads((output / 'assembly.json').read_text())
        if (not assembled.get('complete_submission') or assembled['release_sha256'] != file_hash(artifacts['release'])
                or assembled['references_sha256'] != file_hash(data / 'test/test_source1.tsv')):
            raise ValueError('Existing assembled output has a different recipe/input')
        for name, digest in assembled['output_sha256'].items():
            if name not in ('matching_results.tsv', 'candidate_pairs.tsv') or file_hash(output / name) != digest:
                raise ValueError('Existing assembled output changed')
    subprocess.run([sys.executable, str(package / 'src/validate_full_submission.py'),
        '--test-dir', str(data / 'test'), '--output', str(output),
        '--report', str(work / 'validation.json')], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('data-dir', 'work-dir', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    run(parser.parse_args())
