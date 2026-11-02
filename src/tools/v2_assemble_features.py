"""Assemble verified immutable feature-shard prefixes without waiting for later work.

Source caches/configs are never rewritten. The requested reference TSV must be an
exact ordered prefix of the original source-reference TSV identified by the
workers' config hash. Only committed shards are eligible. No AWS calls.
"""
import argparse
import csv
import errno
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np

FIELDS = ['entity_id','business_name','business_address','country']
GOLD_FIELDS = ['source1_entity_id','matched_entity_ids']
SHARED_KEYS = ('mode','store','index','lexicon','model_sha256','gate_terms','gallery_fold',
    'labels_available','top_k_per_channel','channels','channel_order','feature_names',
    'query_batch_size','partitions','reference_count','references_sha256',
    'lexicon_sha256','index_config_sha256','feature_code_sha256','retrieval_code_sha256',
    'direct_feature_code_sha256','runner_code_sha256','wand_code_sha256',
    'max_df_ratio','frequency_mode','country_block','external_data_used')


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def read_tsv(path,fields):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader = csv.DictReader(f,delimiter='\t')
        if reader.fieldnames != fields:
            raise ValueError(f'Unexpected TSV columns: {path}')
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'Malformed TSV row: {path}')
            yield row


def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2)


def row_signature(rows):
    # Exactly matches run_direct_pipeline.process, including default JSON spacing.
    return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()


def reference_prefix(references,source_references,config):
    if file_hash(source_references) != config['references_sha256']:
        raise ValueError('Original source-reference file checksum mismatch')
    seen_source = set()
    requested_count = source_count = 0
    for requested,source in itertools.zip_longest(read_tsv(references,FIELDS),read_tsv(source_references,FIELDS)):
        if source is None:
            raise ValueError('Requested references exceed original source reference list')
        if not source['entity_id'].startswith('S1-') or source['entity_id'] in seen_source:
            raise ValueError('Duplicate/invalid source owner ID')
        seen_source.add(source['entity_id'])
        source_count += 1
        if requested is not None:
            if requested != source:
                raise ValueError('Requested references are not an exact ordered prefix')
            requested_count += 1
    if source_count != config['reference_count'] or requested_count == 0:
        raise ValueError('Source reference count mismatch or empty requested prefix')
    if requested_count < source_count and requested_count % config['query_batch_size']:
        raise ValueError('A shorter prefix must end on a shard boundary; immutable shards cannot be truncated')
    return requested_count


def source_configs(sources):
    if len(set(sources)) != len(sources):
        raise ValueError('Duplicate source directory')
    configs = [json.loads((source/'config.json').read_text()) for source in sources]
    if not configs:
        raise ValueError('No source partitions')
    for config in configs:
        missing = set(SHARED_KEYS)-set(config)
        if missing:
            raise ValueError(f'Missing source config keys: {sorted(missing)}')
        if config['mode'] != 'features' or config['model_sha256'] is not None:
            raise ValueError('Only feature-mode caches may be assembled')
        if type(config['labels_available']) is not bool:
            raise ValueError('Invalid labels_available metadata')
        if config['partitions'] < 1 or config['query_batch_size'] < 1:
            raise ValueError('Invalid partition/batch size')
        for key in SHARED_KEYS:
            if config[key] != configs[0][key]:
                raise ValueError(f'Source partition config mismatch: {key}')
    assigned = set()
    assignments = []
    total = configs[0]['partitions']
    for config in configs:
        values = config.get('assigned_partitions',[config['partition']])
        if not values or len(values) != len(set(values)) or any(type(i) is not int or not 0 <= i < total for i in values):
            raise ValueError('Invalid assigned partitions')
        if assigned & set(values):
            raise ValueError('Duplicate/overlapping source partition assignment')
        assigned.update(values)
        assignments.append(set(values))
    if assigned != set(range(total)):
        raise ValueError('Missing source partitions')
    return configs,assignments


def verify_shard(directory,shard_id,rows,config):
    if (directory/'done.json').is_symlink():
        raise ValueError('Committed metadata may not be a symlink')
    done = json.loads((directory/'done.json').read_text())
    ids = [row['entity_id'] for row in rows]
    if done['shard_id'] != shard_id or done['references'] != len(rows) or done['input_sha256'] != row_signature(rows):
        raise ValueError(f'Shard {shard_id} input hash/count/ID mismatch')
    id_hash = hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    if done['reference_ids_sha256'] != id_hash:
        raise ValueError(f'Shard {shard_id} reference-ID checksum mismatch')
    expected_files = {'features.npz','references.tsv'}
    if config['labels_available']:
        expected_files.add('ground_truth.tsv')
    if set(done['output_sha256']) != expected_files:
        raise ValueError(f'Shard {shard_id} unexpected/missing committed files')
    for name in expected_files:
        path = directory/name
        if path.is_symlink() or not path.is_file() or file_hash(path) != done['output_sha256'][name]:
            raise ValueError(f'Shard {shard_id} committed file checksum mismatch: {name}')
    if list(read_tsv(directory/'references.tsv',FIELDS)) != rows:
        raise ValueError(f'Shard {shard_id} reference TSV differs from requested rows')
    with np.load(directory/'features.npz',allow_pickle=False) as z:
        a = {key:z[key] for key in z.files}
    required = {'X','y','query_ids','target_ids','reference_ids','countries','true_counts',
                'channel_scores','channel_ranks','channel_order'}
    if not required <= set(a):
        raise ValueError(f'Shard {shard_id} missing array fields')
    n = done['candidate_pairs']
    owners = len(ids)
    if list(a['reference_ids']) != ids or list(a['countries']) != [r['country'] for r in rows]:
        raise ValueError(f'Shard {shard_id} array reference identity/order mismatch')
    if a['X'].shape != (n,len(config['feature_names'])) or not np.isfinite(a['X']).all():
        raise ValueError(f'Shard {shard_id} feature dimensions/values differ')
    if a['channel_scores'].shape != (n,len(config['channel_order'])) or a['channel_ranks'].shape != a['channel_scores'].shape:
        raise ValueError(f'Shard {shard_id} channel dimensions differ')
    if list(a['channel_order']) != config['channel_order'] or not np.isfinite(a['channel_scores']).all():
        raise ValueError(f'Shard {shard_id} channel order/values differ')
    if a['channel_ranks'].dtype.kind not in 'iu' or (a['channel_ranks']<0).any() or (a['channel_ranks']>config['top_k_per_channel']).any():
        raise ValueError(f'Shard {shard_id} invalid channel ranks')
    if a['query_ids'].shape != (n,) or a['query_ids'].dtype.kind not in 'iu':
        raise ValueError(f'Shard {shard_id} invalid query mapping')
    if (a['query_ids']<0).any() or (a['query_ids']>=owners).any() or (np.diff(a['query_ids'].astype(np.int64))<0).any():
        raise ValueError(f'Shard {shard_id} out-of-range/unordered query indices')
    if a['target_ids'].shape != (n,) or a['y'].shape != (n,) or a['true_counts'].shape != (owners,):
        raise ValueError(f'Shard {shard_id} label/target/count dimensions differ')
    pair_ids = set()
    for q,target in zip(a['query_ids'],a['target_ids']):
        if not str(target).startswith(('S2-','S3-')) or (int(q),str(target)) in pair_ids:
            raise ValueError(f'Shard {shard_id} invalid/duplicate candidate pair')
        pair_ids.add((int(q),str(target)))
    if config['labels_available']:
        truth_rows = list(read_tsv(directory/'ground_truth.tsv',GOLD_FIELDS))
        if [r['source1_entity_id'] for r in truth_rows] != ids:
            raise ValueError(f'Shard {shard_id} ground-truth owner order differs')
        gold = []
        for row in truth_rows:
            values = row['matched_entity_ids'].split(',') if row['matched_entity_ids'] else []
            if len(values) != len(set(values)) or any(not value.startswith(('S2-','S3-')) for value in values):
                raise ValueError(f'Shard {shard_id} malformed ground truth')
            gold.append(set(values))
        if list(a['true_counts']) != [len(values) for values in gold]:
            raise ValueError(f'Shard {shard_id} true-label counts differ')
        expected_y = np.array([int(target in gold[q]) for q,target in zip(a['query_ids'],a['target_ids'])],dtype=np.int8)
        if not np.array_equal(a['y'],expected_y):
            raise ValueError(f'Shard {shard_id} pair labels disagree with ground truth')
    elif not (a['y'] == -1).all() or not (a['true_counts'] == -1).all():
        raise ValueError(f'Shard {shard_id} unlabeled sentinel mismatch')
    return done


def link_or_copy(source,destination):
    try:
        os.link(source,destination,follow_symlinks=False)
        return 'hardlink'
    except OSError as error:
        allowed = {errno.EXDEV,errno.EPERM,errno.EACCES,errno.ENOTSUP,errno.EOPNOTSUPP}
        if error.errno not in allowed:
            raise
        # Exclusive destination protects against accidental replacement on fallback.
        with Path(source).open('rb') as src,Path(destination).open('xb') as dst:
            shutil.copyfileobj(src,dst,length=1024*1024)
        shutil.copystat(source,destination,follow_symlinks=False)
        return 'copy'


def run(sources,references,output,source_references=None,ignore_extra_shards=False):
    start = time.perf_counter()
    sources = [Path(p).resolve() for p in sources]
    references,output = Path(references).resolve(),Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if any(source in output.parents for source in sources):
        raise ValueError('Output must not be nested inside an original source cache')
    source_config_hashes = [file_hash(source/'config.json') for source in sources]
    requested_hash = file_hash(references)
    configs,assignments = source_configs(sources)
    common = configs[0]
    original = Path(source_references or common['references']).resolve()
    n = reference_prefix(references,original,common)
    size = common['query_batch_size']
    expected = set(range(math.ceil(n/size)))
    full_count = math.ceil(common['reference_count']/size)
    located,extras,uncommitted,hidden = {},[],[],[]
    for source,assigned in zip(sources,assignments):
        for directory in sorted((source/'shards').iterdir()):
            if directory.name.startswith('.'):
                hidden.append(str(directory))
                continue
            if not directory.is_dir() or not directory.name.isdecimal() or directory.is_symlink():
                raise ValueError(f'Unexpected visible shard path: {directory}')
            shard_id = int(directory.name)
            if directory.name != f'{shard_id:06d}' or shard_id >= full_count or shard_id % common['partitions'] not in assigned:
                raise ValueError(f'Shard outside configured source/partition assignment: {directory}')
            if not (directory/'done.json').is_file():
                uncommitted.append(str(directory))
                continue
            if shard_id in located:
                raise ValueError(f'Duplicate global shard {shard_id}')
            located[shard_id] = directory
            if shard_id not in expected:
                extras.append(shard_id)
    if extras and not ignore_extra_shards:
        raise ValueError('Committed out-of-prefix shards require --ignore-extra-shards')
    missing = expected-set(located)
    if missing:
        raise ValueError(f'Missing required prefix shards: {sorted(missing)}')
    verified = []
    iterator = iter(read_tsv(references,FIELDS))
    count = pairs = 0
    for shard_id in sorted(expected):
        rows = list(itertools.islice(iterator,size))
        done = verify_shard(located[shard_id],shard_id,rows,common)
        count += len(rows)
        pairs += done['candidate_pairs']
        verified.append((shard_id,located[shard_id],done,file_hash(located[shard_id]/'done.json')))
    if count != n or next(iterator,None) is not None:
        raise ValueError('Requested prefix reference coverage changed during validation')
    if file_hash(references) != requested_hash or file_hash(original) != common['references_sha256']:
        raise ValueError('Reference input files changed during validation')
    if any(file_hash(source/'config.json') != digest for source,digest in zip(sources,source_config_hashes)):
        raise ValueError('Source configuration changed during validation')
    # Reserve destination exclusively. A failed copy leaves no complete manifest.
    output.mkdir(parents=True,exist_ok=False)
    (output/'shards').mkdir()
    config = dict(common,mode='features',model=None,model_sha256=None,from_features=None,
        partition=0,partitions=1,assigned_partitions=[0],selected_shards=None,
        references=str(references),references_sha256=requested_hash,reference_count=n,output=str(output))
    write_json(output/'config.json',config)
    inventory = []
    linked = copied = 0
    for shard_id,source,done,done_hash in verified:
        destination = output/'shards'/f'{shard_id:06d}'
        destination.mkdir()
        for name,expected_hash in {**done['output_sha256'],'done.json':done_hash}.items():
            mode = link_or_copy(source/name,destination/name)
            linked += mode == 'hardlink'
            copied += mode == 'copy'
            if file_hash(destination/name) != expected_hash:
                raise ValueError(f'Shard {shard_id} changed during publication; output remains incomplete')
        inventory.append({'shard_id':shard_id,'source':str(source),'references':done['references'],
            'candidate_pairs':done['candidate_pairs'],'input_sha256':done['input_sha256'],
            'done_sha256':done_hash,'output_sha256':done['output_sha256']})
    report = {'complete':True,'feature_prefix_only':True,'references':n,'candidate_pairs':pairs,
        'shards':len(expected),'requested_references_sha256':config['references_sha256'],
        'original_source_references':str(original),'original_source_references_sha256':common['references_sha256'],
        'source_reference_count':common['reference_count'],'source_partitions':[
            {'path':str(source),'config_sha256':digest,'assigned_partitions':sorted(assigned)}
            for source,assigned,digest in zip(sources,assignments,source_config_hashes)],
        'ignored_extra_shards':sorted(extras),'ignored_uncommitted_paths':uncommitted,'ignored_hidden_paths':hidden,
        'hardlinked_files':linked,'copied_files':copied,'source_files_modified':False,
        'remote_store_and_index_metadata_preserved':True,'inventory':inventory,
        'assembler_sha256':file_hash(__file__),'wall_seconds':time.perf_counter()-start}
    write_json(output/'assembly.json',report)
    write_json(output/'manifest.json',{'complete':True,'partition_complete':True,'complete_submission':False,
        'references':n,'candidate_pairs':pairs,'partition':0,'partitions':1,'config':config,
        'assembled_feature_prefix':True,'assembly_sha256':file_hash(output/'assembly.json'),
        'wall_seconds':time.perf_counter()-start})
    print(json.dumps({k:v for k,v in report.items() if k not in ('inventory','ignored_hidden_paths','ignored_uncommitted_paths')},indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',nargs='+',required=True)
    parser.add_argument('--references',required=True)
    parser.add_argument('--source-references',help='Local unchanged copy of the workers\' full reference list')
    parser.add_argument('--output',required=True)
    parser.add_argument('--ignore-extra-shards',action='store_true',
                        help='Explicitly ignore committed shards after the requested prefix')
    run(**vars(parser.parse_args()))
