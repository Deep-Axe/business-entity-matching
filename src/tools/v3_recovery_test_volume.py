"""Unlabeled full-gallery test-volume pilot for frozen recovery queries.

Stratified reservoir sample, selected without model outputs. Retain original
candidates, record only new query pairs, and estimate volume by country. No
submission changes or test-accuracy claims. GPU scoring is optional and separate.
"""
from collections import Counter, defaultdict
import csv
import fcntl
import pickle
import random
import time

from ber.data import read_tsv
from ber.disk_index import DiskRetriever
from ber.store import connect, lookup
from ber.text import Record
from v3_native_alias_retrieval import queries as native_queries
from v3_website_alias_retrieval import queries as website_queries
from v3_neural_test_pipeline import DATA, read, resources
from v3_transfer_data import ROOT, digest, write

WORK = ROOT / 'artifacts/improvement_v3/recovery_test_volume_v1'
BASE = ROOT / 'artifacts/duf-fel_full_output_v2_100k/candidate_pairs.tsv'
DICTIONARY = ROOT / 'artifacts/improvement_v3/native_alias_recovery_400_v1/dictionary.json'


def reservoir(rows, per_country=400):
    rng = random.Random(20260926); totals = Counter(); selected = defaultdict(list)
    for row in rows:
        country = row['country']; totals[country] += 1
        values = selected[country]
        if len(values) < per_country:
            values.append(row)
        else:
            position = rng.randrange(totals[country])
            if position < per_country:
                values[position] = row
    return totals, sorted((r for rs in selected.values() for r in rs), key=lambda r: r['entity_id'])


def run():
    WORK.mkdir(parents=True, exist_ok=True)
    lock = (WORK / 'lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (WORK / 'report.json').exists():
        return
    resources(); source = DATA / 'test_source1.tsv'
    identity = dict(code_sha256=digest(__file__), source_sha256=digest(source),
        candidates_sha256=digest(BASE), dictionary_sha256=digest(DICTIONARY),
        native_query_sha256=digest(ROOT / 'tools/v3_native_alias_retrieval.py'),
        website_query_sha256=digest(ROOT / 'tools/v3_website_alias_retrieval.py'),
        index_config_sha256=digest(ROOT / 'artifacts/test_index_v2/config.json'),
        owners_per_country=400, top_k_per_query=20, test_labels_used=False)
    if (WORK / 'identity.json').exists():
        if read(WORK / 'identity.json') != identity:
            raise ValueError('Frozen unlabeled probe identity changed')
    else:
        write(WORK / 'identity.json', identity)
    totals, selected = reservoir(read_tsv(source))
    if sum(totals.values()) != 1732544 or set(totals) != {'US', 'India', 'France'}:
        raise ValueError('Unexpected full-test reference coverage')
    write(WORK / 'selection.json', dict(total_countries=totals, selected=selected,
        identity_sha256=digest(WORK / 'identity.json'), labels_or_predictions_used=False))
    keep = {r['entity_id'] for r in selected}; original = {}
    with BASE.open(newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            owner = row['source1_entity_id']
            if owner in keep:
                if owner in original:
                    raise ValueError('Duplicate selected original candidate owner')
                original[owner] = set(filter(None, row['candidate_entity_ids'].split(',')))
    if set(original) != keep:
        raise ValueError('Original candidate sample coverage differs')
    db = connect(ROOT / 'artifacts/test_store_v2'); db.execute('PRAGMA cache_size=-8192')
    index = DiskRetriever(ROOT / 'artifacts/test_index_v2', transliteration=True, max_df_ratio=None)
    mapping = read(DICTIONARY)['mapping']; stats = defaultdict(Counter); examples = []; route_keys = defaultdict(list)
    start = time.monotonic(); query_seconds = 0.
    for i, row in enumerate(selected):
        resources()
        if time.monotonic() - start > 3600:
            raise TimeoutError('Finite unlabeled recovery probe window expired')
        record = Record.from_row(row); path = WORK / 'owners' / (record.entity_id + '.json')
        if path.exists():
            saved = read(path)
        else:
            channels = {}; stamp = time.monotonic()
            routes = dict(native=native_queries(index, record, mapping), website=website_queries(index, record))
            for route, queries in routes.items():
                found = set()
                for query in queries.values():
                    hits = index.searcher.search(query, limit=20, count=False).hits
                    records = lookup(db, 'records', 'row_id', index.searcher.fast_field_values('row_id', [doc for _, doc in hits]))
                    if any(r['country'] != record.country for r in records.values()):
                        raise ValueError('Cross-country recovery query')
                    found.update(r['entity_id'] for r in records.values())
                channels[route] = sorted(found - original[record.entity_id])
            saved = dict(owner=record.entity_id, country=record.country, channels=channels, seconds=time.monotonic()-stamp)
            write(path, saved)
        if saved['owner'] != record.entity_id or saved['country'] != record.country:
            raise ValueError('Resumed test owner identity differs')
        query_seconds += saved['seconds']; union = set().union(*(set(v) for v in saved['channels'].values()))
        targets = lookup(db, 'records', 'entity_id', sorted(union))
        for t in sorted(union):
            r = targets[t]; j = len(examples)
            # Neither fake negatives nor fake first-stage probabilities.
            examples.append((record.entity_id, t, None, record.country, None,
                             row['business_name'], row['business_address'], r['name'], r['address']))
            for route, values in saved['channels'].items():
                if t in values:
                    route_keys[route].append(j)
        stats[record.country].update(owners=1, union_pairs=len(union))
        for route, values in saved['channels'].items():
            stats[record.country][route + '_pairs'] += len(values)
        if (i+1) % 100 == 0:
            write(WORK / 'progress.json', dict(stage='unlabeled_full_gallery_queries', owners=i+1,
                expected=len(selected), new_pairs=len(examples), seconds=time.monotonic()-start))
    db.close()
    with (WORK / 'extra.pkl').open('wb') as handle:
        pickle.dump(examples, handle)
    write(WORK / 'routes.json', route_keys)
    extrapolated = {route: sum(totals[c] * counts[route+'_pairs'] / counts['owners'] for c, counts in stats.items())
                    for route in ('native', 'website', 'union')}
    write(WORK / 'report.json', dict(complete=True, owners=len(selected), countries=stats, full_countries=totals,
        full_test_pair_volume_estimate=extrapolated, original_query_seconds=query_seconds,
        wall_seconds=time.monotonic()-start, outputs_sha256={n:digest(WORK/n) for n in ('extra.pkl','routes.json','selection.json')},
        identity_sha256=digest(WORK/'identity.json'), ready_for_submission=False,
        notes=['Unlabeled sample; no recall, precision or French accuracy estimated.',
               'Stratified sample estimates are uncertain; local cached-query speed is not a cloud runtime guarantee.',
               'Deeper top100 recovery is excluded from this cost pilot; no rule or model was retuned.']))


if __name__ == '__main__':
    run()
