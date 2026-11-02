"""Flat BM25 term unions with post-retrieval country/fold eligibility.

Tantivy's Block-WAND optimization applies to term unions, but not the old nested
MUST filters. Adaptive over-retrieval preserves eligible top-k until a bounded
cap; difficult underfilled queries fall back to the explicitly measured gate-1
retriever. This is a new retrieval recipe and must be validated/recalibrated.
"""

import fcntl
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import tantivy

from .disk_index import DiskRetriever
from .retrieval import field_terms
from .shards import file_hash
from .store import connect, batches


def eligibility(store):
    store = Path(store)
    directory = store.parent / (store.name+"_eligibility_v1")
    directory.mkdir(exist_ok=True)
    signature = file_hash(store / "config.json")
    with (directory / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (directory / "complete.json").exists():
            db = connect(store)
            keys = sorted((row["country"], row["fold"]) for row in db.execute("SELECT DISTINCT country,fold FROM records"))
            if len(keys) > 254:
                raise ValueError("Too many eligibility groups for uint8")
            mapping = {key: i+1 for i, key in enumerate(keys)}
            count, maximum = db.execute("SELECT COUNT(*),MAX(row_id) FROM records").fetchone()
            if count != maximum:
                raise ValueError("Eligibility requires contiguous immutable row IDs")
            fd, temporary = tempfile.mkstemp(prefix="eligibility.", suffix=".npy", dir=directory)
            os.close(fd)
            data = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=(count+1,))
            data[0] = 0
            for rows in batches(db.execute("SELECT row_id,country,fold FROM records ORDER BY row_id"), 100000):
                start, end = rows[0]["row_id"], rows[-1]["row_id"]+1
                data[start:end] = [mapping[(row["country"], row["fold"])] for row in rows]
            data.flush()
            del data
            Path(temporary).replace(directory / "codes.npy")
            (directory / "complete.json").write_text(json.dumps(dict(store_signature=signature, records=count, keys=keys)))
            db.close()
        metadata = json.loads((directory / "complete.json").read_text())
        if metadata["store_signature"] != signature:
            raise ValueError("Eligibility/store mismatch")
    return np.load(directory / "codes.npy", mmap_mode="r"), metadata["keys"]


class WandRetriever(DiskRetriever):
    def __init__(self, index, store, fold=None):
        super().__init__(index, transliteration=True, fold=fold, max_df_ratio=.01, frequency_mode="gate")
        self.codes, self.eligibility_keys = eligibility(store)
        self.allowed = {}
        from .hybrid_retrieval import GatedRetriever
        self.fallback = GatedRetriever(index, transliteration=True, fold=fold,
                                          max_df_ratio=.01, frequency_mode="gate", gate_terms=1)

    def term_clauses(self, field, terms, boost=1.):
        present = sorted((self.frequency(field, value), value) for value in terms
                         if self.frequency(field, value))
        limit = 24 if "grams" in field else 12
        clauses = []
        for _, value in present[:limit]:
            query = tantivy.Query.term_query(self.schema, field, value, "freq")
            if boost != 1:
                query = tantivy.Query.boost_query(query, boost)
            clauses.append((tantivy.Occur.Should, query))
        return clauses

    def candidates(self, record, top_k):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if record.country not in self.allowed:
            self.allowed[record.country] = {i+1 for i, (country, fold) in enumerate(self.eligibility_keys)
                if country == record.country and (self.fold is None or fold == self.fold)}
        allowed = self.allowed[record.country]
        if not allowed:
            return []
        terms, result = field_terms(record), {}
        for channel in self.channels:
            if channel == "combined":
                clauses = self.term_clauses("name_words", terms["name_words"], 2.)
                clauses += self.term_clauses("address_words", terms["address_words"])
            else:
                clauses = self.term_clauses(channel, terms[channel])
            if not clauses:
                continue
            query = tantivy.Query.boolean_query(clauses, minimum_number_should_match=1)
            limit, cap = max(40, top_k*2), max(1280, top_k*2)
            while True:
                hits = self.searcher.search(query, limit=limit, count=False).hits
                ids = self.searcher.fast_field_values("row_id", [address for _, address in hits])
                selected = [(float(score), row_id) for (score, _), row_id in zip(hits, ids)
                            if int(self.codes[row_id]) in allowed][:top_k]
                if len(selected) >= top_k or len(hits) < limit or limit >= cap:
                    break
                limit = min(limit*2, cap)
            if len(selected) < top_k and len(hits) == limit and limit >= cap:
                self.fallback.channels = (channel,)
                for row_id, channel_hits in self.fallback.candidates(record, top_k):
                    result.setdefault(row_id, {}).update(channel_hits)
                continue
            best = selected[0][0] if selected else 1.
            for rank, (score, row_id) in enumerate(selected, 1):
                result.setdefault(row_id, {})[channel] = (score/max(best, 1e-12), 1./rank)
        return sorted(result.items(), key=lambda item: (-sum(v[1] for v in item[1].values()), item[0]))
