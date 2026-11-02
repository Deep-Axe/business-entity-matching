"""Verify one collected 50k partition before the full 200k assembly is ready."""
import argparse
import itertools
import json
from pathlib import Path
import time

from v2_assemble_features import FIELDS, file_hash, read_tsv, verify_shard, write_json


def run(source, references, output):
    source, references, output = map(Path, (source, references, output))
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    config_hash, manifest_hash = file_hash(source / 'config.json'), file_hash(source / 'manifest.json')
    config = json.loads((source / 'config.json').read_text())
    manifest = json.loads((source / 'manifest.json').read_text())
    node = config['partition']
    if (config['mode'] != 'features' or config['gallery_fold'] != 'train'
            or config['gate_terms'] != 0 or not config['labels_available']
            or config['partitions'] != 4 or config['reference_count'] != 200000
            or config['query_batch_size'] != 200 or not 0 <= node < 4
            or config.get('assigned_partitions', [node]) != [node]
            or not manifest['complete'] or not manifest['partition_complete']
            or manifest['config'] != config or manifest['references'] != 50000):
        raise ValueError('Not a complete expected 50k gate-0 training partition')
    if file_hash(references) != config['references_sha256']:
        raise ValueError('Original reference checksum differs')
    expected = set(range(node, 1000, 4))
    visible = [p for p in (source / 'shards').iterdir() if not p.name.startswith('.')]
    if ({p.name for p in visible} != {f'{i:06d}' for i in expected}
            or any(p.is_symlink() or not p.is_dir() for p in visible)):
        raise ValueError('Partition shard coverage differs')
    rows = iter(read_tsv(references, FIELDS))
    count = pairs = 0
    inventory = []
    for shard_id in range(1000):
        batch = list(itertools.islice(rows, 200))
        if len(batch) != 200:
            raise ValueError('Truncated original references')
        if shard_id in expected:
            directory = source / 'shards' / f'{shard_id:06d}'
            done = verify_shard(directory, shard_id, batch, config)
            count += done['references']
            pairs += done['candidate_pairs']
            inventory.append(dict(shard_id=shard_id, done_sha256=file_hash(directory / 'done.json'),
                input_sha256=done['input_sha256'], output_sha256=done['output_sha256']))
    if next(rows, None) is not None or count != 50000 or pairs != manifest['candidate_pairs']:
        raise ValueError('Partition/reference totals differ')
    if (file_hash(source / 'config.json') != config_hash
            or file_hash(source / 'manifest.json') != manifest_hash
            or file_hash(references) != config['references_sha256']):
        raise ValueError('Inputs changed during verification')
    result = dict(complete=True, partition=node, references=count, candidate_pairs=pairs,
        shards=250, config_sha256=config_hash, manifest_sha256=manifest_hash,
        original_references_sha256=config['references_sha256'], source=str(source.resolve()),
        inventory=inventory, seconds=time.perf_counter() - started,
        checks=['all committed file hashes', 'exact original references and global shard IDs',
                'feature dimensions and finiteness', 'candidate uniqueness', 'provided training-label consistency'])
    write_json(output, result)
    print(json.dumps({k:v for k,v in result.items() if k != 'inventory'}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'references', 'output'):
        parser.add_argument('--' + name, required=True)
    run(**vars(parser.parse_args()))
