"""Streaming/reopenable lexical retrieval without a corpus-sized Python heap."""

import hashlib
import json
import resource
import time
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import tantivy

from . import text as text_module
from .data import write_json
from .retrieval import ASCII_CHANNELS, BASE_CHANNELS, INDEX_FIELDS, field_terms
from .store import batches, connect, ensure_disk, lookup, record_row
from .text import Record


def _analyzer(index):
    index.register_tokenizer("split", tantivy.TextAnalyzerBuilder(
        tantivy.Tokenizer.whitespace()).build())


def build_index(store, output, threads=4, heap_mb=256, commit_every=100000,
                resume=False, stop_after_commits=None, minimum_free_gib=10):
    if threads < 1 or heap_mb < 32 * threads or commit_every < 1:
        raise ValueError("Need positive threads/commit size and >=32 MiB heap per thread")
    store, output = Path(store), Path(output)
    manifest = json.loads((store / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Complete the record store before indexing")
    config = {
        "index_version": 2, "store": str(store.resolve()),
        "store_config_sha256": hashlib.sha256((store / "config.json").read_bytes()).hexdigest(),
        "target_records": manifest["targets"], "fields": list(INDEX_FIELDS),
        "normalization_sha256": hashlib.sha256(Path(text_module.__file__).read_bytes()).hexdigest(),
        "external_data_used": False,
    }
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Index configuration changed; use a new output")
    else:
        output.mkdir(parents=True)
        write_json(output / "config.json", config)
    ensure_disk(output, minimum_free_gib)
    directory = output / "tantivy"
    if directory.exists():
        index = tantivy.Index.open(str(directory))
    else:
        directory.mkdir()
        builder = tantivy.SchemaBuilder()
        builder.add_integer_field("row_id", fast=True, indexed=True)
        for field in ("country", "fold"):
            builder.add_text_field(field, tokenizer_name="raw", index_option="basic")
        for field in INDEX_FIELDS:
            builder.add_text_field(field, tokenizer_name="split", index_option="freq")
        index = tantivy.Index(builder.build(), path=str(directory), reuse=False)
    _analyzer(index)
    index.config_reader("Manual")
    index.reload()
    done = index.searcher().num_docs
    db = connect(store)
    n, maximum = db.execute("SELECT COUNT(*),MAX(row_id) FROM records").fetchone()
    if n != maximum or done > n:
        raise ValueError("Index resume requires a contiguous, unchanged record store")
    started, start_count, commits = time.perf_counter(), done, 0
    writer = index.writer(heap_size=heap_mb * 1024**2, num_threads=threads)
    try:
        iterator = db.execute("SELECT * FROM records WHERE row_id>? ORDER BY row_id", (done,))
        for group in batches(iterator, min(commit_every, 1000)):
            for row in group:
                record = Record.from_row(record_row(row))
                fields = {field: [" ".join(sorted(values))]
                          for field, values in field_terms(record).items()}
                fields.update(row_id=[row["row_id"]], country=[row["country"]],
                              fold=[row["fold"]])
                writer.add_document(tantivy.Document.from_dict(fields, index.schema))
                done += 1
            if done // commit_every > (done - len(group)) // commit_every or done == n:
                writer.commit()
                index.reload()
                if index.searcher().num_docs != done:
                    raise RuntimeError("Committed index count differs from streamed row count")
                elapsed = time.perf_counter() - started
                progress = {
                    "complete": False, "all_records_committed": done == n,
                    "indexed_records": done, "target_records": n,
                    "last_invocation_seconds": round(elapsed, 3),
                    "rows_per_second_this_invocation": (done - start_count) / max(elapsed, .001),
                    "peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                }
                write_json(output / "manifest.json", progress)
                print(f"index: {done:,}/{n:,} committed, "
                      f"{progress['rows_per_second_this_invocation']:.0f} rows/s, "
                      f"peak RSS {progress['peak_rss_MiB']:.0f} MiB.", flush=True)
                commits += 1
                ensure_disk(output, minimum_free_gib)
                if stop_after_commits and commits >= stop_after_commits:
                    break
        writer.wait_merging_threads()
        index.reload()
        progress = {
            "complete": done == n, "indexed_records": index.searcher().num_docs,
            "target_records": n, "last_invocation_seconds": round(time.perf_counter() - started, 3),
            "peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }
        write_json(output / "manifest.json", progress)
        return progress
    finally:
        db.close()


class TargetStore:
    def __init__(self, store, cache_size=2048):
        self.db = connect(store)
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def get_many(self, row_ids):
        missing = [r for r in row_ids if r not in self.cache]
        found = lookup(self.db, "records", "row_id", missing)
        result = {}
        for row_id in row_ids:
            if row_id in self.cache:
                record = self.cache.pop(row_id)
            else:
                record = Record.from_row(record_row(found[row_id]))
            self.cache[row_id] = record
            result[row_id] = record
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return result

    def close(self):
        self.db.close()
        self.cache.clear()


class DiskRetriever:
    def __init__(self, directory, transliteration=False, country_block=True,
                 fold=None, channels=None, max_row_id=None, max_df_ratio=None,
                 frequency_mode="drop"):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if not manifest["complete"]:
            raise ValueError("Index is incomplete; resume its build first")
        self.index = tantivy.Index.open(str(directory / "tantivy"))
        _analyzer(self.index)
        self.schema, self.searcher = self.index.schema, self.index.searcher()
        if self.searcher.num_docs != manifest["indexed_records"]:
            raise ValueError("Index count differs from completed manifest")
        self.channels = tuple(channels) if channels is not None else (
            BASE_CHANNELS + (ASCII_CHANNELS if transliteration else ()))
        if not set(self.channels) <= set((*BASE_CHANNELS, *ASCII_CHANNELS)):
            raise ValueError("Unknown retrieval channel")
        self.country_block, self.fold, self.max_row_id = country_block, fold, max_row_id
        if max_df_ratio is not None and not 0 < max_df_ratio <= 1:
            raise ValueError("max_df_ratio must be in (0, 1]")
        self.max_df = self.searcher.num_docs * max_df_ratio if max_df_ratio is not None else None
        if frequency_mode not in ("drop", "gate"):
            raise ValueError("frequency_mode must be drop or gate")
        self.frequency_mode = frequency_mode
        # Per-instance bounded cache; no corpus-sized Python document-frequency map.
        self.frequency = lru_cache(maxsize=100000)(self.searcher.doc_freq)

    def _field_parts(self, field, values):
        maximum = 24 if "grams" in field else 12
        weighted = [(self.frequency(field, value), value) for value in values]
        # OOV terms cannot retrieve anything; do not let them consume the cap.
        present = [(frequency, value) for frequency, value in sorted(weighted) if frequency]
        eligible = [value for frequency, value in present
                    if self.max_df is None or frequency <= self.max_df]
        # Retain a few rarest available terms for generic names/addresses with
        # no term below the frequency cutoff. Coverage is measured empirically.
        make = lambda selected: tantivy.Query.boolean_query([
            (tantivy.Occur.Should, tantivy.Query.term_query(self.schema, field, value, "freq"))
            for value in selected], minimum_number_should_match=1) if selected else None
        if self.frequency_mode == "drop":
            return make((eligible or [value for _, value in present[:3]])[:maximum]), None
        full = make([value for _, value in present[:maximum]])
        gate = make(eligible[:maximum]) if self.max_df is not None else None
        return full, gate

    def candidates(self, record, top_k):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        terms = field_terms(record)
        needed = set(self.channels) - {"combined"}
        if "combined" in self.channels:
            needed.update(("name_words", "address_words"))
        parts = {field: self._field_parts(field, terms[field]) for field in needed}
        full = {field: pair[0] for field, pair in parts.items()}
        gates = {field: pair[1] for field, pair in parts.items()}
        combined = [(tantivy.Occur.Should, tantivy.Query.boost_query(full[f], boost))
                    for f, boost in (("name_words", 2.), ("address_words", 1.))
                    if full.get(f) is not None]
        full["combined"] = tantivy.Query.boolean_query(
            combined, minimum_number_should_match=1) if combined else None
        combined_gates = [(tantivy.Occur.Should, gates[f]) for f in ("name_words", "address_words")
                          if gates.get(f) is not None]
        gates["combined"] = tantivy.Query.boolean_query(
            combined_gates, minimum_number_should_match=1) if combined_gates else None
        queries = {field: tantivy.Query.boolean_query([
            (tantivy.Occur.Must, query),
            (tantivy.Occur.Must, tantivy.Query.const_score_query(gates[field], 0.))])
            if query is not None and gates.get(field) is not None else query
            for field, query in full.items()}
        filters = []
        for field, value in (("country", record.country if self.country_block else None),
                             ("fold", self.fold)):
            if value is not None:
                filters.append((tantivy.Occur.Must, tantivy.Query.const_score_query(
                    tantivy.Query.term_query(self.schema, field, value, "basic"), 0.)))
        if self.max_row_id is not None:
            filters.append((tantivy.Occur.Must, tantivy.Query.range_query(
                self.schema, "row_id", tantivy.FieldType.Integer, upper_bound=self.max_row_id)))
        results = {}
        for channel in self.channels:
            query = queries[channel]
            if query is None:
                continue
            if filters:
                query = tantivy.Query.boolean_query([(tantivy.Occur.Must, query), *filters])
            hits = self.searcher.search(query, limit=top_k, count=False).hits
            row_ids = self.searcher.fast_field_values("row_id", [address for _, address in hits])
            best_score = hits[0][0] if hits else 1.
            for rank, ((score, _), row_id) in enumerate(zip(hits, row_ids), 1):
                results.setdefault(row_id, {})[channel] = (
                    float(score / max(best_score, 1e-12)), 1. / rank)
        return sorted(results.items(),
                      key=lambda item: (-sum(v[1] for v in item[1].values()), item[0]))
