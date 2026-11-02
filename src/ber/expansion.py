"""Optional training-learned, word-family plus address retrieval channel."""

import json
import math
from pathlib import Path

import tantivy

from .disk_index import DiskRetriever

CHANNEL = "learned_combined"


class ExpansionRetriever:
    def __init__(self, index, lexicon, country_block=True, fold=None,
                 max_df_ratio=.01, frequency_mode="gate"):
        self.mapping = json.loads((Path(lexicon) / "lexicon.json").read_text())
        self.base = DiskRetriever(index, channels=["ascii_words"], country_block=country_block,
            fold=fold, max_df_ratio=max_df_ratio, frequency_mode=frequency_mode)

    def candidates(self, record, top_k=20):
        base = self.base
        original = set(record.ascii_name.split())
        expanded = set(original)
        if record.country != "India":
            return []
        for word in original:
            expanded.update(variant for variant, count in self.mapping.get(word, []))
        if expanded == original:
            return []
        _, gate = base._field_parts("ascii_words", expanded)
        families = []
        words = sorted(original, key=lambda word: (base.frequency("ascii_words", word), word))[:12]
        for word in words:
            variants = {word, *(variant for variant, count in self.mapping.get(word, []))}
            terms = [(tantivy.Occur.Should, tantivy.Query.term_query(base.schema, "ascii_words", variant, "freq"))
                     for variant in sorted(variants) if base.frequency("ascii_words", variant)]
            if terms:
                group = tantivy.Query.boolean_query(terms, minimum_number_should_match=1)
                weight = math.log1p(base.searcher.num_docs / (base.frequency("ascii_words", word) + 1))
                families.append((tantivy.Occur.Should, tantivy.Query.const_score_query(group, weight)))
        if not families:
            return []
        full = tantivy.Query.boolean_query(families, minimum_number_should_match=1)
        address, address_gate = base._field_parts("address_words", record.address_tokens)
        if address is not None:
            full = tantivy.Query.boolean_query([
                (tantivy.Occur.Should, tantivy.Query.boost_query(full, 2.)),
                (tantivy.Occur.Should, address)], minimum_number_should_match=1)
            if gate is not None and address_gate is not None:
                gate = tantivy.Query.boolean_query([(tantivy.Occur.Should, gate),
                    (tantivy.Occur.Should, address_gate)], minimum_number_should_match=1)
        clauses = [(tantivy.Occur.Must, full)]
        if gate is not None:
            clauses.append((tantivy.Occur.Must, tantivy.Query.const_score_query(gate, 0.)))
        for field, value in (("country", record.country if base.country_block else None), ("fold", base.fold)):
            if value is not None:
                clauses.append((tantivy.Occur.Must, tantivy.Query.const_score_query(
                    tantivy.Query.term_query(base.schema, field, value, "basic"), 0.)))
        hits = base.searcher.search(tantivy.Query.boolean_query(clauses), limit=top_k, count=False).hits
        row_ids = base.searcher.fast_field_values("row_id", [address for _, address in hits])
        best = hits[0][0] if hits else 1.
        return [(row_id, (float(score / max(best, 1e-12)), 1. / rank))
                for rank, (row_id, (score, _)) in enumerate(zip(row_ids, hits), 1)]
