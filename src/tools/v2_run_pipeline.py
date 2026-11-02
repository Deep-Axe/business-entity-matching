"""Isolated, provenance-checked CPU inference for the enhanced 76-feature model.

No frozen driver/module is modified. Forward-gallery IDF statistics are an
explicit policy: the same index used for candidate retrieval supplies IDF, both
on training-gallery evaluation and on the actual test gallery. Neural scoring
remains a separately validated second phase.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import hashlib
import importlib
import json
import multiprocessing
from pathlib import Path
import resource
import tempfile
import time
import numpy as np
from catboost import CatBoostClassifier

from ber.context_features import CONTEXT_NAMES,context_matrix
from ber.data import FIELDS,read_tsv,write_json,write_tsv
from ber.deadline_retrieval import retrievers
from ber.direct_features import CHANNELS,make_arrays
from ber.disk_index import TargetStore
from ber.features import FEATURE_NAMES
from ber.shards import file_hash,gold_for
from ber.store import batches,ensure_disk,lookup,record_row
from v2_cpu_features import EXTRA_NAMES,View,Weighting,extra_features

NAMES = FEATURE_NAMES+CONTEXT_NAMES+EXTRA_NAMES
WEIGHTING_POLICY = 'forward_gallery_index_idf_v1'
SAFE_CACHED_FIELDS = ('X','query_ids','target_ids','reference_ids','countries',
                      'channel_scores','channel_ranks','channel_order')
STATE = None


def signature(rows):
    return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()


def code_hashes():
    modules = {'base_features':'ber.features','context':'ber.context_features',
        'retrieval':'ber.deadline_retrieval','direct_features':'ber.direct_features',
        'text':'ber.text','v2_features':'v2_cpu_features'}
    return {key:file_hash(importlib.import_module(value).__file__) for key,value in modules.items()}


def validate_recipe(index,store,lexicon,model,model_report,training_manifest,gate_terms,
                    top_k,weighting_policy,allow_retrieval_shift=False):
    if weighting_policy != WEIGHTING_POLICY:
        raise ValueError('Only explicit forward-gallery IDF weighting is implemented')
    idx = json.loads((Path(index)/'config.json').read_text())
    source = json.loads((Path(store)/'config.json').read_text())
    training = json.loads(Path(training_manifest).read_text())
    report = json.loads(Path(model_report).read_text())
    hashes = code_hashes()
    if idx['store_config_sha256'] != file_hash(Path(store)/'config.json'):
        raise ValueError('Retrieval/weighting index does not match the provided store')
    if idx['normalization_sha256'] != hashes['text']:
        raise ValueError('Current text normalization differs from indexed text')
    if not training.get('complete') or training['names'] != NAMES:
        raise ValueError('Training matrix/schema is incomplete or not the 76-feature schema')
    if training['feature_code_sha256'] != hashes['v2_features']:
        raise ValueError('Enhanced feature implementation changed since model training')
    if report['model_sha256'] != file_hash(model) or report['features'] != NAMES:
        raise ValueError('Model/report/schema identity mismatch')
    original = training['source_config']
    for key,actual in (('feature_code_sha256',hashes['base_features']),
                       ('direct_feature_code_sha256',hashes['direct_features']),
                       ('retrieval_code_sha256',hashes['retrieval'])):
        if original[key] != actual:
            raise ValueError(f'Base feature implementation changed: {key}')
    requested = {'gate_terms':gate_terms,'top_k_per_channel':top_k,'channels':list(CHANNELS),
                 'lexicon_sha256':file_hash(Path(lexicon)/'lexicon.json')}
    shifts = {key:{'training':original[key],'requested':value} for key,value in requested.items()
              if original[key] != value}
    if shifts and not allow_retrieval_shift:
        raise ValueError(f'Retrieval recipe changed; explicit diagnostic override required: {shifts}')
    return {'store_split':source['split'],'index_config_sha256':file_hash(Path(index)/'config.json'),
        'store_config_sha256':file_hash(Path(store)/'config.json'),'lexicon_sha256':requested['lexicon_sha256'],
        'model_sha256':report['model_sha256'],'model_report_sha256':file_hash(model_report),
        'training_manifest_sha256':file_hash(training_manifest),'component_sha256':hashes,
        'model_report_threshold':report['threshold'],'retrieval_shift':shifts,
        'weighting_policy':weighting_policy,'weighting_index':str(Path(index).resolve()),
        'weighting_index_config_sha256':file_hash(Path(index)/'config.json'),
        'weighting_target_records':idx['target_records'],
        'weighting_formula':'log((N+1)/(df+1))+1; rarity normalized by log1p(N)+1',
        'training_weighting_index_config_sha256':original['index_config_sha256']}


def cached_arrays(path,rows,config):
    done = json.loads((path/'done.json').read_text())
    if done['input_sha256'] != signature(rows):
        raise ValueError('Cached reference rows differ')
    if file_hash(path/'features.npz') != done['output_sha256']['features.npz']:
        raise ValueError('Cached base features changed')
    with np.load(path/'features.npz',allow_pickle=False) as z:
        # Do not read cached label members even if the source cache was labeled.
        a = {key:z[key] for key in SAFE_CACHED_FIELDS}
    if list(a['reference_ids']) != [r['entity_id'] for r in rows] or len(a['target_ids']) != done['candidate_pairs']:
        raise ValueError('Cached pair/reference mapping mismatch')
    a['y'] = np.full(len(a['target_ids']),-1,dtype=np.int8)
    a['true_counts'] = np.full(len(rows),-1,dtype=np.int32)
    return a


def enhanced_matrix(arrays,rows,target_rows,weights):
    anchors = [View.from_row(r) for r in rows]
    views = {eid:View.from_row(record_row(row)) for eid,row in target_rows.items()}
    extras = np.asarray([extra_features(anchors[int(q)],views[str(eid)],weights)
                         for q,eid in zip(arrays['query_ids'],arrays['target_ids'])],dtype=np.float32).reshape(-1,len(EXTRA_NAMES))
    return np.column_stack((context_matrix(arrays),extras))


def initialize(config):
    global STATE
    base = extra = None
    if not config['from_features']:
        base,extra = retrievers(config['index'],config['lexicon'],config['gate_terms'],config['gallery_fold'],store=config['store'])
    targets = TargetStore(config['store'])
    weights = Weighting(Path(config['weighting_index']))
    if weights.documents != config['weighting_target_records']:
        raise ValueError('Live weighting index document count changed')
    model = CatBoostClassifier(thread_count=1)
    model.load_model(config['model'])
    if list(model.feature_names_) != NAMES:
        raise ValueError('Loaded model has an unexpected feature schema')
    STATE = config,base,extra,targets,weights,model


def verify_committed(directory,rows,config_sha,emit_matrix=False):
    done = json.loads((directory/'done.json').read_text())
    ids = [r['entity_id'] for r in rows]
    if (done['input_sha256'] != signature(rows) or done['config_sha256'] != config_sha
            or done['references'] != len(rows) or done['reference_ids_sha256'] != hashlib.sha256('\n'.join(ids).encode()).hexdigest()):
        raise ValueError('Resume input/config changed')
    expected = {'candidate_pairs.tsv','matching_results.tsv','probabilities.npy','pair_ids.npz'}
    if emit_matrix:
        expected.add('X76.npy')
    if set(done['output_sha256']) != expected:
        raise ValueError('Committed prediction file set differs')
    for name,digest in done['output_sha256'].items():
        if Path(name).name != name or file_hash(directory/name) != digest:
            raise ValueError('Committed prediction output changed')
    return done


def process(task):
    shard_id,rows = task
    config,base,extra,targets,weights,model = STATE
    output = Path(config['output'])
    destination = output/'shards'/f'{shard_id:06d}'
    config_sha = file_hash(output/'config.json')
    if destination.exists():
        return dict(verify_committed(destination,rows,config_sha,config['emit_matrix']),resumed=True)
    ensure_disk(output,config['minimum_free_gib'])
    started = time.perf_counter()
    ids = [r['entity_id'] for r in rows]
    anchors = lookup(targets.db,'anchors','entity_id',ids)
    if len(anchors) != len(ids) or any(record_row(anchors[row['entity_id']]) != row for row in rows):
        raise ValueError('Reference rows differ from verified original store')
    if config['gallery_fold'] and any(anchors[eid]['fold'] != config['gallery_fold'] for eid in ids):
        raise ValueError('Reference owner fold differs from gallery fold')
    # The only ownership-label read in this driver is behind this explicit gate.
    gold = gold_for(targets.db,ids) if config['labels_available'] else None
    if config['from_features']:
        a = cached_arrays(Path(config['from_features'])/'shards'/f'{shard_id:06d}',rows,config)
        timing = {'query_seconds':0.,'base_features_reused':True}
        if gold is not None:
            a['y'] = np.asarray([int(eid in gold[ids[q]]) for q,eid in zip(a['query_ids'],a['target_ids'])],dtype=np.int8)
            a['true_counts'] = np.asarray([len(gold[eid]) for eid in ids],dtype=np.int32)
    else:
        a,timing = make_arrays(rows,base,extra,targets,gold,config['top_k_per_channel'])
    raw = lookup(targets.db,'records','entity_id',list(set(map(str,a['target_ids']))))
    if len(raw) != len(set(a['target_ids'])):
        raise ValueError('Candidate target missing from provided store')
    if any(raw[str(eid)]['country'] != rows[int(q)]['country'] for q,eid in zip(a['query_ids'],a['target_ids'])):
        raise ValueError('Country blocking violated')
    stamp = time.perf_counter()
    matrix = enhanced_matrix(a,rows,raw,weights)
    if matrix.shape != (len(a['target_ids']),len(NAMES)) or not np.isfinite(matrix).all():
        raise ValueError('Invalid enhanced feature matrix')
    p = model.predict_proba(matrix,thread_count=1)[:,1] if len(matrix) else np.empty(0)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid classifier probabilities')
    timing['enhanced_features_and_prediction_seconds'] = time.perf_counter()-stamp
    staged = Path(tempfile.mkdtemp(prefix=f'.{shard_id:06d}.',dir=output/'shards'))
    candidate,matched = [[] for _ in ids],[[] for _ in ids]
    for q,eid,probability in zip(a['query_ids'],a['target_ids'],p):
        candidate[q].append(str(eid))
        if probability >= config['threshold']:
            matched[q].append(str(eid))
    for filename,column,groups in (('candidate_pairs.tsv','candidate_entity_ids',candidate),
                                   ('matching_results.tsv','matched_entity_ids',matched)):
        write_tsv(staged/filename,({'source1_entity_id':eid,column:','.join(sorted(group))} for eid,group in zip(ids,groups)),
                  ['source1_entity_id',column])
    np.save(staged/'probabilities.npy',p)
    alignment = {key:a[key] for key in ('reference_ids','target_ids','query_ids','countries')}
    if config['labels_available']:
        alignment.update(y=a['y'],true_counts=a['true_counts'])
    np.savez_compressed(staged/'pair_ids.npz',**alignment)
    names = ['candidate_pairs.tsv','matching_results.tsv','probabilities.npy','pair_ids.npz']
    if config['emit_matrix']:
        np.save(staged/'X76.npy',matrix)
        names.append('X76.npy')
    done = {'shard_id':shard_id,'input_sha256':signature(rows),
        'reference_ids_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
        'config_sha256':config_sha,'references':len(ids),'candidate_pairs':len(p),
        'wall_seconds':time.perf_counter()-started,'peak_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'output_sha256':{name:file_hash(staged/name) for name in names},**timing}
    write_json(staged/'done.json',done)
    if destination.exists():
        raise FileExistsError('A concurrent writer committed this shard')
    staged.rename(destination)
    return done


def run(index,store,references,lexicon,model,model_report,training_manifest,output,
        threshold,weighting_policy,gate_terms=1,top_k=20,query_batch_size=200,workers=1,
        partition=0,partitions=1,gallery_fold=None,labeled=False,from_features=None,
        resume=False,emit_matrix=False,minimum_free_gib=6,allow_retrieval_shift=False):
    if not 0 < threshold <= 1 or gate_terms < -1 or min(workers,query_batch_size,top_k,partitions)<1 or not 0 <= partition < partitions:
        raise ValueError('Invalid inference/concurrency/partition options')
    provenance = validate_recipe(index,store,lexicon,model,model_report,training_manifest,
                                 gate_terms,top_k,weighting_policy,allow_retrieval_shift)
    if labeled and provenance['store_split'] != 'train':
        raise ValueError('Labels are available only in the supplied training store')
    if gallery_fold and not labeled:
        raise ValueError('Unlabeled inference must use the complete gallery')
    count,seen = 0,set()
    for row in read_tsv(references):
        if row['entity_id'] in seen or not row['entity_id'].startswith('S1-'):
            raise ValueError('Duplicate/invalid requested reference ID')
        seen.add(row['entity_id'])
        count += 1
    del seen
    if not count:
        raise ValueError('Empty reference list')
    output = Path(output).resolve()
    config = dict(provenance,index=str(Path(index).resolve()),store=str(Path(store).resolve()),
        references=str(Path(references).resolve()),references_sha256=file_hash(references),reference_count=count,
        lexicon=str(Path(lexicon).resolve()),model=str(Path(model).resolve()),
        model_report=str(Path(model_report).resolve()),training_manifest=str(Path(training_manifest).resolve()),
        output=str(output),mode='predict',threshold=float(threshold),
        threshold_differs_from_training_report=float(threshold)!=float(provenance['model_report_threshold']),
        gate_terms=gate_terms,top_k_per_channel=top_k,channels=list(CHANNELS),channel_order=list(CHANNELS),
        feature_names=NAMES,query_batch_size=query_batch_size,partition=partition,partitions=partitions,
        assigned_partitions=[partition],gallery_fold=gallery_fold,labels_available=labeled,
        from_features=str(Path(from_features).resolve()) if from_features else None,emit_matrix=emit_matrix,
        minimum_free_gib=minimum_free_gib,country_block=True,max_df_ratio=.01,frequency_mode='gate',
        external_data_used=False,feature_code_sha256=provenance['component_sha256']['base_features'],
        direct_feature_code_sha256=provenance['component_sha256']['direct_features'],
        retrieval_code_sha256=provenance['component_sha256']['retrieval'],
        v2_feature_code_sha256=provenance['component_sha256']['v2_features'],
        runner_code_sha256=file_hash(__file__),
        wand_code_sha256=file_hash(importlib.import_module('ber.wand_retrieval').__file__) if gate_terms==-1 else None)
    if from_features:
        cached = json.loads((Path(from_features)/'config.json').read_text())
        for key in ('index_config_sha256','gate_terms','gallery_fold','top_k_per_channel','channels','channel_order',
                    'query_batch_size','lexicon_sha256','feature_code_sha256','direct_feature_code_sha256','retrieval_code_sha256'):
            if cached[key] != config[key]:
                raise ValueError(f'Cached base-feature recipe differs: {key}')
        config['source_cache_config_sha256'] = file_hash(Path(from_features)/'config.json')
    if output.exists():
        if not resume or json.loads((output/'config.json').read_text()) != config:
            raise ValueError('Existing output requires identical provenance/config and --resume')
    else:
        (output/'shards').mkdir(parents=True)
        write_json(output/'config.json',config)
    expected = {i for i in range((count+query_batch_size-1)//query_batch_size) if i%partitions==partition}
    existing = {int(p.name) for p in (output/'shards').iterdir() if p.name.isdecimal()}
    if not existing <= expected:
        raise ValueError('Output contains out-of-coverage committed/stale shard paths')
    def tasks():
        for shard,rows in enumerate(batches(read_tsv(references),query_batch_size)):
            if shard in expected:
                yield shard,rows
    started = time.perf_counter()
    done_ids = set()
    references_done = pairs = 0
    source = iter(tasks())
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(config,),
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = {}
        for _ in range(workers*2):
            item = next(source,None)
            if item is not None:
                pending[pool.submit(process,item)] = item[0]
        while pending:
            ready,_ = wait(pending,timeout=30,return_when=FIRST_COMPLETED)
            if not ready:
                print(json.dumps({'status':'working','references':references_done,'seconds':time.perf_counter()-started}),flush=True)
            for future in ready:
                shard = pending.pop(future)
                result = future.result()
                if result['shard_id'] != shard or shard in done_ids:
                    raise ValueError('Duplicate/wrong completed global shard')
                done_ids.add(shard)
                references_done += result['references']
                pairs += result['candidate_pairs']
                progress = {'completed_shards':len(done_ids),'expected_shards':len(expected),
                    'references':references_done,'candidate_pairs':pairs,'seconds':time.perf_counter()-started}
                write_json(output/'progress.json',progress)
                print(json.dumps(progress),flush=True)
                item = next(source,None)
                if item is not None:
                    pending[pool.submit(process,item)] = item[0]
    expected_references = sum(min(query_batch_size,count-i*query_batch_size) for i in expected)
    if done_ids != expected or references_done!=expected_references or file_hash(references)!=config['references_sha256']:
        raise ValueError('Exact requested reference/shard coverage not established')
    report = {'complete':True,'partition_complete':True,'complete_submission':False,
        'references':references_done,'candidate_pairs':pairs,'partition':partition,'partitions':partitions,
        'wall_seconds':time.perf_counter()-started,'workers':workers,'config':config,
        'scope':'All requested references assigned to this partition; submission-wide validation remains separate.'}
    write_json(output/'manifest.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='config'},indent=2),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('index','store','references','lexicon','model','model-report','training-manifest','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--threshold',type=float,required=True)
    p.add_argument('--weighting-policy',choices=[WEIGHTING_POLICY],required=True)
    p.add_argument('--gate-terms',type=int,default=1)
    p.add_argument('--top-k',type=int,default=20)
    p.add_argument('--query-batch-size',type=int,default=200)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--partition',type=int,default=0)
    p.add_argument('--partitions',type=int,default=1)
    p.add_argument('--gallery-fold',choices=['train','tune','eval'])
    p.add_argument('--labeled',action='store_true')
    p.add_argument('--from-features')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--emit-matrix',action='store_true')
    p.add_argument('--minimum-free-gib',type=float,default=6)
    p.add_argument('--allow-retrieval-shift',action='store_true',help='Explicit diagnostic override; requires separate calibration.')
    run(**vars(p.parse_args()))
