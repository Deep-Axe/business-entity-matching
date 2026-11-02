"""Bounded reverse-S1 retrieval and label-free competing-owner veto experiments.

The original stores/indexes/models are read-only. Reverse evidence never uses
ownership labels. This evaluates a text-competition veto, not probabilities from
incomplete forward candidate neighborhoods.
"""

import argparse
import json
import math
import resource
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import tantivy
from catboost import CatBoostClassifier
from rapidfuzz import fuzz

from ber.context_features import model_matrix
from ber.data import write_json
from ber.disk_index import DiskRetriever, _analyzer
from ber.metrics import score_arrays
from ber.retrieval import INDEX_FIELDS, field_terms
from ber.shards import file_hash
from ber.store import connect, lookup, record_row
from ber.text import Record


def build(store, output, limit=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    builder = tantivy.SchemaBuilder()
    builder.add_integer_field("row_id", fast=True, indexed=True)
    for field in ("country", "fold"):
        builder.add_text_field(field, tokenizer_name="raw", index_option="basic")
    for field in INDEX_FIELDS:
        builder.add_text_field(field, tokenizer_name="split", index_option="freq")
    (output / "tantivy").mkdir()
    index = tantivy.Index(builder.build(), path=str(output / "tantivy"), reuse=False)
    _analyzer(index)
    writer = index.writer(heap_size=64 * 1024**2, num_threads=1)
    db = connect(store)
    expected = db.execute("SELECT COUNT(*) FROM anchors").fetchone()[0]
    started = time.perf_counter()
    config = dict(store=str(Path(store).resolve()), store_sha256=file_hash(Path(store)/"config.json"),
                  purpose="Reverse retrieval over Source-1 TEXT only; no ownership labels", limit=limit)
    write_json(output / "config.json", config)
    count = 0
    for row in db.execute("SELECT rowid AS anchor_row_id,* FROM anchors ORDER BY rowid"):
        if limit is not None and count >= limit:
            break
        record = Record.from_row(record_row(row))
        fields = {field: [" ".join(sorted(values))] for field, values in field_terms(record).items()}
        fields.update(row_id=[row["anchor_row_id"]], country=[row["country"]], fold=[row["fold"]])
        writer.add_document(tantivy.Document.from_dict(fields,index.schema))
        count += 1
        if count % 100000 == 0:
            writer.commit()
            print(json.dumps(dict(indexed=count, total=expected, seconds=time.perf_counter()-started)), flush=True)
    writer.commit()
    writer.wait_merging_threads()
    index.reload()
    assert index.searcher().num_docs == count
    report = dict(complete=True, full_source1_coverage=count==expected, indexed_records=count,
                  expected_records=expected, seconds=time.perf_counter()-started,
                  peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(output/"manifest.json",report)
    db.close()
    print(json.dumps(report),flush=True)


def number_set(record):
    return {str(int(token)) for token in record.numbers}


def text_strength(left, right):
    """Ranking heuristic only; tuned veto margin, not calibrated probability."""
    name = max(fuzz.token_sort_ratio(left.name,right.name),
               fuzz.token_sort_ratio(left.ascii_name,right.ascii_name))/100
    if left.address and right.address:
        address = max(fuzz.token_sort_ratio(left.address,right.address),
                      fuzz.token_sort_ratio(left.ascii_address,right.ascii_address))/100
        numbers_l, numbers_r = number_set(left), number_set(right)
        common = len(numbers_l & numbers_r)/max(1,len(numbers_l | numbers_r))
        conflict = float(bool(numbers_l) and bool(numbers_r) and not numbers_l & numbers_r)
        return .55*name+.4*address+.05*common-.15*conflict
    return .75*name  # Missing address cannot be treated as strong agreeing evidence.


class Competition:
    def __init__(self, store, index, top_k=10):
        manifest=json.loads((Path(index)/"manifest.json").read_text())
        config=json.loads((Path(index)/"config.json").read_text())
        if not manifest.get("full_source1_coverage"):
            raise ValueError("Competition evaluation requires full Source-1 coverage")
        if config["store_sha256"]!=file_hash(Path(store)/"config.json"):
            raise ValueError("Reverse index and store differ")
        self.db=connect(store)
        self.retriever=DiskRetriever(index,channels=["name_words","ascii_words","combined"],
                                     country_block=True,max_df_ratio=.01,frequency_mode="gate")
        self.top_k=top_k
        self.rivals=lru_cache(maxsize=4096)(self._rivals)

    def _rivals(self,target_id):
        row=lookup(self.db,"records","entity_id",[target_id])[target_id]
        target=Record.from_row(record_row(row))
        candidates=self.retriever.candidates(target,self.top_k)
        records={}
        # sqlite.Row only includes explicitly selected rowid, so use a bound query.
        ids=[rid for rid,_ in candidates]
        if ids:
            sql="SELECT rowid AS rid,* FROM anchors WHERE rowid IN ("+",".join("?" for _ in ids)+")"
            for anchor in self.db.execute(sql,ids):
                record=Record.from_row(record_row(anchor))
                records[record.entity_id]=text_strength(record,target)
        return target,records

    def feature(self, reference, target_id):
        target,rivals=self.rivals(target_id)
        own=text_strength(reference,target)
        others=[score for eid,score in rivals.items() if eid!=reference.entity_id]
        best=max(others,default=-1.)
        return own,best,len(others),int(reference.entity_id in rivals)


def veto(probabilities, own, other, threshold, allowance):
    accepted=probabilities>=threshold
    # Positive allowance tolerates a slightly stronger rival. Negative allowance
    # requires a winning margin and can abstain on ties; zero preserves ties.
    # No arbitrary winner by entity ID or file order.
    rejected=accepted & (other > own + allowance)
    result=np.array(probabilities,copy=True)
    result[rejected]=-1.
    return result


def evaluate(store, index, features, model, output, count=3000,
             probabilities=None,matrix_manifest=None,forward_index=None):
    bound=None; inventory={}; binding=None
    if bool(probabilities)!=bool(matrix_manifest):
        raise ValueError('External probabilities require their producer matrix manifest')
    if probabilities:
        from v2_cascade_cache import validate_inputs
        _,matrix,binding,_,_,bound=validate_inputs(features,probabilities,matrix_manifest,
                                                  store=store,index=forward_index)
        if binding['model_sha256']!=file_hash(model):
            raise ValueError('Probability producer model differs from supplied model')
        inventory={item['shard']:item for item in matrix['shards']}
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    competition=Competition(store,index)
    classifier=None
    if bound is None:
        classifier=CatBoostClassifier(thread_count=1)
        classifier.load_model(model)
    started=time.perf_counter(); chunks={}; offset=0
    for path in sorted((Path(features)/"shards").glob("[0-9]*/features.npz")):
        if offset>=count: break
        with np.load(path,allow_pickle=False) as z: a={k:z[k] for k in z.files}
        take=min(len(a["reference_ids"]),count-offset)
        rows=lookup(competition.db,"anchors","entity_id",a["reference_ids"][:take].tolist())
        refs={eid:Record.from_row(record_row(row)) for eid,row in rows.items()}
        if bound is None:
            probability=classifier.predict_proba(model_matrix(a,classifier.feature_names_),thread_count=1)[:,1]
        else:
            item=inventory[path.parent.name]
            probability=bound[item['pair_start']:item['pair_end']]
            if len(probability)!=len(a['y']) or not np.isfinite(probability).all():
                raise ValueError('Invalid probability slice')
        keep=a["query_ids"]<take
        p=probability[keep]; q=a["query_ids"][keep]
        target_ids=a["target_ids"][keep]
        own=np.zeros(len(p),dtype=np.float32); other=np.full(len(p),-1.,dtype=np.float32)
        rivals=np.zeros(len(p),dtype=np.int16); found=np.zeros(len(p),dtype=np.int8)
        # Evaluate at a range of thresholds without redoing retrieval; all below
        # .40 are permanently rejected in this experiment and get no veto work.
        for i in np.flatnonzero(p>=.40):
            own[i],other[i],rivals[i],found[i]=competition.feature(refs[str(a['reference_ids'][q[i]])],str(target_ids[i]))
        values=dict(query_ids=q+offset,y=a["y"][keep],probabilities=p,own=own,other=other,
                    rivals=rivals,reverse_found=found,target_ids=target_ids,
                    reference_ids=a["reference_ids"][:take],countries=a["countries"][:take],true_counts=a["true_counts"][:take])
        for k,v in values.items(): chunks.setdefault(k,[]).append(v)
        offset+=take
        print(json.dumps(dict(references=offset,seconds=time.perf_counter()-started)),flush=True)
    a={k:np.concatenate(v) for k,v in chunks.items()}
    np.savez_compressed(output/"evidence.npz",**a)
    report=dict(references=offset,candidate_pairs=len(a['y']),seconds=time.perf_counter()-started,
                model_sha256=file_hash(model),feature_config_sha256=file_hash(Path(features)/"config.json"),
                reverse_index_config_sha256=file_hash(Path(index)/"config.json"),
                threshold_floor=.40,reverse_top_k_per_channel=10,
                probability_binding=binding,
                evidence="Full Source-1 text gallery. Heuristic text ranking veto; no ownership labels used for rivals.",
                caveat="Reverse candidate search can miss competitors; this is not exhaustive best-owner assignment.")
    write_json(output/"manifest.json",report)
    competition.db.close()
    print(json.dumps(report),flush=True)


def compare(tune,dev,output,baseline_threshold=.74):
    if not .55<=baseline_threshold<=1:raise ValueError('Baseline threshold outside evidence grid')
    output=Path(output)
    arrays={}; manifests={}
    for split,path in [('tune',tune),('dev',dev)]:
        manifests[split]=json.loads((Path(path)/'manifest.json').read_text())
        with np.load(Path(path)/"evidence.npz",allow_pickle=False) as z: arrays[split]={k:z[k] for k in z.files}
        a=arrays[split]; n=len(a['reference_ids']); pairs=len(a['probabilities'])
        if n!=manifests[split]['references'] or pairs!=manifests[split]['candidate_pairs']:
            raise ValueError('Evidence sizes differ from evaluation manifest')
        if len(set(a['reference_ids']))!=n or not n:
            raise ValueError('Empty or duplicate evaluation reference IDs')
        if len(a['countries'])!=n or len(a['true_counts'])!=n:
            raise ValueError('Reference-level evidence sizes differ')
        for key in ['query_ids','y','own','other','target_ids']:
            if len(a[key])!=pairs: raise ValueError('Pair-level evidence sizes differ: '+key)
        if pairs and (a['query_ids'].min()<0 or a['query_ids'].max()>=n):
            raise ValueError('Query index outside evaluated references')
        if not np.isin(a['y'],[0,1]).all():
            raise ValueError('Comparison requires provided-label evaluation evidence')
        if not all(np.isfinite(a[key]).all() for key in ['probabilities','own','other']):
            raise ValueError('Non-finite probability/text evidence')
    if set(arrays['tune']['reference_ids']) & set(arrays['dev']['reference_ids']):
        raise ValueError("Tuning/development overlap")
    # The reverse-index config hash transitively fingerprints the source store
    # config; both evaluations must use the identical full-S1 index/model and
    # retrieval budget. Forward feature-config hashes legitimately differ by split.
    compatibility=['model_sha256','reverse_index_config_sha256','threshold_floor','reverse_top_k_per_channel']
    for key in compatibility:
        if key not in manifests['tune'] or key not in manifests['dev'] or manifests['tune'][key]!=manifests['dev'][key]:
            raise ValueError('Incompatible tuning/development evidence: '+key)
    if not 0<=manifests['tune']['threshold_floor']<=.55:
        raise ValueError('Evaluation evidence floor does not cover comparison threshold grid')
    output.mkdir(parents=True,exist_ok=False)
    def score(a,threshold,allowance):
        p=a['probabilities'] if allowance is None else veto(a['probabilities'],a['own'],a['other'],threshold,allowance)
        return score_arrays(a['query_ids'],a['y'],p,a['true_counts'],threshold)
    choices=[]
    for allowance in [None,-.05,-.025,-.01,0.,.01,.025,.05,.10]:
        for threshold in sorted(set([.55,.60,.65,.70,.74,.78,.82,.86,.90,baseline_threshold])):
            s=score(arrays['tune'],threshold,allowance)
            choices.append(dict(allowance=allowance,threshold=threshold,tune_macro_f05=s['macro_f05']))
    best=max(choices,key=lambda x:x['tune_macro_f05'])
    threshold_control=max((c for c in choices if c['allowance'] is None),key=lambda x:x['tune_macro_f05'])
    fixed_veto=max((c for c in choices if c['threshold']==baseline_threshold),key=lambda x:x['tune_macro_f05'])
    best_veto=max((c for c in choices if c['allowance'] is not None),key=lambda x:x['tune_macro_f05'])
    policies=dict(fixed_baseline=dict(threshold=baseline_threshold,allowance=None),
                  threshold_only=threshold_control,fixed_threshold_veto=fixed_veto,
                  selected=best,selected_veto=best_veto)
    def paired(left,right,countries):
        delta=left-right
        def summary(values):
            rng=np.random.default_rng(20260925)
            draws=[rng.choice(values,len(values),replace=True).mean() for _ in range(1000)]
            return dict(references=len(values),delta=float(values.mean()),
                        paired_bootstrap_95=np.quantile(draws,[.025,.975]).tolist())
        result=summary(delta)
        result['countries']={str(c):summary(delta[countries==c]) for c in np.unique(countries)}
        return result
    results={}
    for split,a in arrays.items():
        summaries={}; scores={}
        for name,policy in policies.items():
            threshold,allowance=policy['threshold'],policy['allowance']
            result=score(a,threshold,allowance)
            scores[name]=result.pop('entity_scores')
            before=a['probabilities']>=threshold
            after=(a['probabilities'] if allowance is None else veto(a['probabilities'],a['own'],a['other'],threshold,allowance))>=threshold
            removed=before & ~after
            summaries[name]=dict(policy=policy,metrics=result,
                accepted_pairs=int(after.sum()),before_veto_accepted_pairs=int(before.sum()),
                vetoed_pairs=int(removed.sum()),vetoed_true_positives=int((removed & (a['y']==1)).sum()),
                vetoed_false_positives=int((removed & (a['y']==0)).sum()),
                references_with_veto=int(len(np.unique(a['query_ids'][removed]))),
                countries={str(c):dict(references=int((a['countries']==c).sum()),
                    macro_f05=float(scores[name][a['countries']==c].mean()),
                    accepted_pairs=int((after & (a['countries'][a['query_ids']]==c)).sum()),
                    vetoed_true_positives=int((removed & (a['y']==1) & (a['countries'][a['query_ids']]==c)).sum()),
                    vetoed_false_positives=int((removed & (a['y']==0) & (a['countries'][a['query_ids']]==c)).sum()))
                    for c in np.unique(a['countries'])})
        comparisons={}
        for left,right in [('selected','fixed_baseline'),('selected','threshold_only'),
                           ('selected_veto','fixed_baseline'),('selected_veto','threshold_only'),
                           ('fixed_threshold_veto','fixed_baseline')]:
            comparisons[left+'_vs_'+right]=paired(scores[left],scores[right],a['countries'])
        main=comparisons['selected_vs_fixed_baseline']
        results[split]=dict(baseline=summaries['fixed_baseline']['metrics'],
            selected=summaries['selected']['metrics'],delta=main['delta'],
            paired_bootstrap_95=main['paired_bootstrap_95'],policies=summaries,comparisons=comparisons)
    report=dict(selected_on_tune_only=best,threshold_only_selected_on_tune=threshold_control,
                fixed_threshold_veto_selected_on_tune=fixed_veto,selected_veto_on_tune_only=best_veto,
                results=results,grid=choices,compatibility={k:manifests['tune'][k] for k in compatibility},
                source_manifests={split:file_hash(Path(path)/'manifest.json') for split,path in [('tune',tune),('dev',dev)]},
                note="All policies selected on tuning only; threshold-only control separates recalibration from veto gain. Veto is conservative rival-text evidence, not complete calibrated global assignment. Paired CIs condition on selected policy; no submission changed.")
    write_json(output/'report.json',report); print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('build'); p.add_argument('--store',required=True);p.add_argument('--output',required=True)
    p.add_argument('--limit',type=int)
    p=sub.add_parser('evaluate')
    for k in ('store','index','features','model','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--count',type=int,default=3000)
    for k in ('probabilities','matrix-manifest','forward-index'):p.add_argument('--'+k)
    p=sub.add_parser('compare')
    for k in ('tune','dev','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--baseline-threshold',type=float,default=.74)
    args=vars(parser.parse_args());action=args.pop('action')
    {'build':build,'evaluate':evaluate,'compare':compare}[action](**args)
