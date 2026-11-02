"""Checksum-bound cascade input preparation and independent GPU scoring.

Producer contract: call write_probability_binding immediately after prediction
serialization, while the model and matrix provenance are known. A naked array's
length cannot establish row identity. No thresholds/fusion are selected here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_new_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2)


def write_probability_binding(probability_path, matrix_manifest_path, model_path):
    """Call at the producer, not as an assertion that arbitrary old arrays align."""
    probability_path, matrix_manifest_path, model_path = map(Path,
        (probability_path, matrix_manifest_path, model_path))
    matrix = json.loads(matrix_manifest_path.read_text())
    if not matrix.get('complete'):
        raise ValueError('Incomplete matrix manifest')
    p = np.load(probability_path, mmap_mode='r', allow_pickle=False)
    if p.shape != (matrix['pairs'],) or p.dtype.kind != 'f':
        raise ValueError('Probability dimensions/type differ from producer matrix')
    binding = {'version': 1, 'probabilities_sha256': digest(probability_path),
        'matrix_manifest_sha256': digest(matrix_manifest_path),
        'source_config_sha256': matrix['source_config_sha256'],
        'model_sha256': digest(model_path), 'model_path_at_production': str(model_path.resolve()),
        'matrix_manifest_at_production': str(matrix_manifest_path.resolve()),
        'pairs': matrix['pairs'], 'references': matrix['references'],
        'shards': matrix['shards'], 'producer_asserts_exact_matrix_row_order': True}
    path = probability_path.with_suffix('.alignment.json')
    write_new_json(path, binding)
    return binding


def validate_inputs(features, probabilities, matrix_manifest, store=None, index=None, binding=None):
    features, probabilities, matrix_manifest = map(Path,(features, probabilities, matrix_manifest))
    config = json.loads((features/'config.json').read_text())
    manifest = json.loads((features/'manifest.json').read_text())
    matrix = json.loads(matrix_manifest.read_text())
    binding_path = Path(binding) if binding else probabilities.with_suffix('.alignment.json')
    provenance = json.loads(binding_path.read_text())
    if not manifest.get('complete') or not matrix.get('complete'):
        raise ValueError('Input cache/matrix is not complete')
    if not provenance.get('producer_asserts_exact_matrix_row_order'):
        raise ValueError('Missing producer row-order assertion')
    checks = ((digest(features/'config.json'), matrix['source_config_sha256']),
              (digest(features/'config.json'), provenance['source_config_sha256']),
              (digest(matrix_manifest), provenance['matrix_manifest_sha256']),
              (digest(probabilities), provenance['probabilities_sha256']))
    if any(a != b for a,b in checks) or matrix['shards'] != provenance['shards']:
        raise ValueError('Probability/cache/matrix provenance checksum mismatch')
    resolved_store = Path(store or config['store'])
    resolved_index = Path(index or config['index'])
    if digest(resolved_index/'config.json') != config['index_config_sha256']:
        raise ValueError('Index override does not match original cache index checksum')
    index_config = json.loads((resolved_index/'config.json').read_text())
    if digest(resolved_store/'config.json') != index_config['store_config_sha256']:
        raise ValueError('Store override does not match indexed store checksum')
    p = np.load(probabilities, mmap_mode='r', allow_pickle=False)
    if p.shape != (matrix['pairs'],) or p.dtype.kind != 'f':
        raise ValueError('Unexpected probability array shape/type')
    if provenance['pairs'] != matrix['pairs'] or provenance['references'] != matrix['references']:
        raise ValueError('Probability binding dimensions differ')
    actual_shards = {path.name for path in (features/'shards').glob('[0-9]*')}
    listed = [item['shard'] for item in matrix['shards']]
    if len(listed) != len(set(listed)) or actual_shards != set(listed):
        raise ValueError('Source shard coverage/uniqueness mismatch')
    pair_offset = owner_offset = 0
    for item in matrix['shards']:
        shard = features/'shards'/item['shard']
        done = json.loads((shard/'done.json').read_text())
        got = digest(shard/'features.npz')
        if got != item['features_sha256'] or got != done['output_sha256']['features.npz']:
            raise ValueError('Source shard changed since probability production')
        if done['input_sha256'] != item['input_sha256']:
            raise ValueError('Source reference input checksum differs from matrix')
        if item['pair_start'] != pair_offset or item['query_start'] != owner_offset:
            raise ValueError('Noncontiguous probability/query slices')
        if item['pair_end']-pair_offset != done['candidate_pairs'] or item['query_end']-owner_offset != done['references']:
            raise ValueError('Shard dimension mismatch')
        pair_offset, owner_offset = item['pair_end'],item['query_end']
    if pair_offset != matrix['pairs'] or owner_offset != matrix['references']:
        raise ValueError('Incomplete probability/matrix coverage')
    if config['reference_count'] != owner_offset:
        raise ValueError('Source reference coverage differs from matrix')
    return config,matrix,provenance,resolved_store,resolved_index,p


def lookup(db, table, values):
    result = {}
    values = list(set(values))
    for i in range(0,len(values),800):
        group = values[i:i+800]
        result.update((r['entity_id'],r) for r in db.execute(
            f"SELECT * FROM {table} WHERE entity_id IN ({','.join('?' for _ in group)})",group))
    if set(result) != set(values):
        raise ValueError(f'Missing {table} IDs in verified store')
    return result


def serialize(row):
    return f"name: {row['name']} ; address: {row['address']} ; country: {row['country']}"


def prepare(features, probabilities, matrix_manifest, output, store=None, index=None,
            binding=None, lower=.01, upper=.99):
    if not 0 <= lower <= upper <= 1:
        raise ValueError('Invalid probability routing bounds')
    config,matrix,provenance,resolved_store,resolved_index,p = validate_inputs(
        features,probabilities,matrix_manifest,store,index,binding)
    features,out = Path(features),Path(output)
    out.mkdir(parents=True,exist_ok=False)
    db = sqlite3.connect(f'{(resolved_store/"records.sqlite3").resolve().as_uri()}?mode=ro',uri=True)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA query_only=ON')
    db.execute('PRAGMA cache_size=-8192')
    inventory = []
    started = time.perf_counter()
    for item in matrix['shards']:
        source = features/'shards'/item['shard']/'features.npz'
        with np.load(source,allow_pickle=False) as z:
            a = {key:z[key] for key in ('query_ids','reference_ids','target_ids','y','true_counts','countries')}
        tree = np.asarray(p[item['pair_start']:item['pair_end']])
        if not np.isfinite(tree).all() or ((tree<0)|(tree>1)).any():
            raise ValueError('Non-finite/out-of-range probabilities')
        if len(tree) != len(a['target_ids']) or len(a['reference_ids']) != item['query_end']-item['query_start']:
            raise ValueError('Probability/feature alignment dimensions differ')
        if len(a['query_ids']) != len(tree) or (a['query_ids']<0).any() or (a['query_ids']>=len(a['reference_ids'])).any():
            raise ValueError('Invalid query index mapping')
        routed = np.flatnonzero((tree>=lower)&(tree<=upper))
        owners = lookup(db,'anchors',[str(a['reference_ids'][a['query_ids'][i]]) for i in routed])
        targets = lookup(db,'records',[str(a['target_ids'][i]) for i in routed])
        target = out/'shards'/item['shard']
        target.mkdir(parents=True)
        np.savez_compressed(target/'arrays.npz',**a,tree=tree,routed_indices=routed)
        with (target/'pairs.jsonl').open('x',encoding='utf-8') as f:
            for i in routed:
                oid = str(a['reference_ids'][a['query_ids'][i]])
                tid = str(a['target_ids'][i])
                f.write(json.dumps({'pair_index':int(i),'reference_id':oid,'target_id':tid,
                                   'left':serialize(owners[oid]),'right':serialize(targets[tid])},ensure_ascii=False)+'\n')
        inventory.append({'shard':item['shard'],'pairs':len(tree),'references':len(a['reference_ids']),
            'routed_pairs':len(routed),'arrays_sha256':digest(target/'arrays.npz'),
            'text_sha256':digest(target/'pairs.jsonl'),'source_features_sha256':item['features_sha256'],
            'global_pair_start':item['pair_start'],'global_pair_end':item['pair_end'],
            'global_query_start':item['query_start'],'global_query_end':item['query_end']})
    db.close()
    result = {'complete':True,'source_features':str(features.resolve()),
        'serialization_format':'name_address_country_v1','runner_sha256':digest(__file__),
        'source_config_sha256':digest(features/'config.json'),'probability_binding':provenance,
        'resolved_store':str(resolved_store.resolve()),'resolved_index':str(resolved_index.resolve()),
        'original_metadata_modified':False,'routing':{'lower':lower,'upper':upper},
        'shards':inventory,'references':matrix['references'],'pairs':matrix['pairs'],
        'routed_pairs':sum(s['routed_pairs'] for s in inventory),'seconds':time.perf_counter()-started}
    write_new_json(out/'manifest.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('shards','probability_binding')},indent=2))


def score_cache(prepared, checkpoint, expected_checkpoint_sha256, output, batch_size=16):
    import torch
    from transformers import BertForSequenceClassification,XLMRobertaTokenizer
    from transformers.utils import logging
    logging.set_verbosity_error()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('No GPU available; no automatic CPU/cloud fallback')
    prepared,checkpoint,out = map(Path,(prepared,checkpoint,output))
    manifest = json.loads((prepared/'manifest.json').read_text())
    if not manifest.get('complete'):
        raise ValueError('Prepared inputs incomplete')
    if manifest.get('serialization_format') != 'name_address_country_v1':
        raise ValueError('Unexpected record serialization format')
    if digest(checkpoint/'model.safetensors') != expected_checkpoint_sha256:
        raise ValueError('Checkpoint differs from explicitly frozen model')
    out.mkdir(parents=True,exist_ok=False)
    tokenizer = XLMRobertaTokenizer.from_pretrained(checkpoint)
    model = BertForSequenceClassification.from_pretrained(checkpoint,low_cpu_mem_usage=True).cuda().eval()
    completed = []
    start = time.perf_counter()
    for item in manifest['shards']:
        source = prepared/'shards'/item['shard']
        if digest(source/'arrays.npz') != item['arrays_sha256'] or digest(source/'pairs.jsonl') != item['text_sha256']:
            raise ValueError('Prepared input changed')
        with np.load(source/'arrays.npz',allow_pickle=False) as z:
            a = {key:z[key] for key in z.files}
        texts = [json.loads(line) for line in (source/'pairs.jsonl').read_text().splitlines()]
        route = a['routed_indices']
        if len(texts) != len(route):
            raise ValueError('Routed text count mismatch')
        for i,row in zip(route,texts):
            if row['pair_index'] != int(i) or row['target_id'] != str(a['target_ids'][i]) or row['reference_id'] != str(a['reference_ids'][a['query_ids'][i]]):
                raise ValueError('Routed text/array ID alignment changed')
        neural = np.full(len(a['tree']),np.nan,dtype=np.float32)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            for offset in range(0,len(route),batch_size):
                batch = texts[offset:offset+batch_size]
                encoded = tokenizer([x['left'] for x in batch],[x['right'] for x in batch],
                    padding=True,truncation='longest_first',max_length=128,return_tensors='pt')
                neural[route[offset:offset+batch_size]] = model(**{k:v.cuda() for k,v in encoded.items()}).logits.softmax(-1)[:,1].float().cpu().numpy()
        if not np.isfinite(neural[route]).all():
            raise ValueError('Non-finite neural probabilities')
        target = out/'shards'/item['shard']
        target.mkdir(parents=True)
        np.save(target/'neural.npy',neural)
        done = {**item,'neural_sha256':digest(target/'neural.npy')}
        write_new_json(target/'done.json',done)
        completed.append(done)
    torch.cuda.synchronize()
    result = {'complete':True,'prepared_manifest_sha256':digest(prepared/'manifest.json'),
        'checkpoint_sha256':expected_checkpoint_sha256,'checkpoint':str(checkpoint.resolve()),
        'checkpoint_assets_sha256':{path.name:digest(path) for path in checkpoint.iterdir()
            if path.is_file() and path.suffix in ('.json','.model')},
        'runner_sha256':digest(__file__),'serialization_format':'name_address_country_v1',
        'fixed_checkpoint_no_training':True,'fusion_or_threshold_selected':False,
        'routing':manifest['routing'],'shards':completed,'seconds':time.perf_counter()-start,
        'routed_pairs':manifest['routed_pairs'],'pairs':manifest['pairs'],'references':manifest['references'],
        'unrouted_probabilities':'NaN: not scored; never interpret as zero'}
    write_new_json(out/'manifest.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='shards'},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    prep = sub.add_parser('prepare')
    for name in ('features','probabilities','matrix-manifest','output'):
        prep.add_argument('--'+name,required=True)
    for name in ('store','index','binding'):
        prep.add_argument('--'+name)
    prep.add_argument('--lower',type=float,default=.01)
    prep.add_argument('--upper',type=float,default=.99)
    scoring = sub.add_parser('score')
    for name in ('prepared','checkpoint','expected-checkpoint-sha256','output'):
        scoring.add_argument('--'+name,required=True)
    scoring.add_argument('--batch-size',type=int,default=16)
    args = vars(parser.parse_args())
    command = args.pop('command')
    (prepare if command=='prepare' else score_cache)(**args)
