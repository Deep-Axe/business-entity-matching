"""Overlapping lexical retrieval using a disk-backed Tantivy index."""

from collections import Counter
from pathlib import Path
import tantivy

BASE_CHANNELS = ("name_words", "name_grams", "address_words", "combined")
ASCII_CHANNELS = ("ascii_words", "ascii_grams")
INDEX_FIELDS = ("name_words", "name_grams", "address_words",
                "ascii_words", "ascii_grams")


def field_terms(record):
    return {
        "name_words": record.name_tokens,
        "name_grams": record.name_grams,
        "address_words": record.address_tokens,
        "ascii_words": frozenset(record.ascii_name.split()),
        "ascii_grams": record.ascii_grams,
    }


class Retriever:
    def __init__(self, records, directory, transliteration=False,
                 country_block=True, threads=4):
        Path(directory).mkdir(parents=True, exist_ok=False)
        builder = tantivy.SchemaBuilder()
        builder.add_integer_field("row_id", stored=True)
        builder.add_text_field("country", tokenizer_name="raw", index_option="basic")
        for field in INDEX_FIELDS:
            builder.add_text_field(field, tokenizer_name="split", index_option="freq")
        self.schema = builder.build()
        self.index = tantivy.Index(self.schema, path=str(directory), reuse=False)
        analyzer = tantivy.TextAnalyzerBuilder(tantivy.Tokenizer.whitespace()).build()
        self.index.register_tokenizer("split", analyzer)
        self.frequency = {f: Counter() for f in INDEX_FIELDS}
        writer = self.index.writer(heap_size=128_000_000, num_threads=threads)
        for row_id, record in enumerate(records):
            terms = field_terms(record)
            fields = {field: [" ".join(sorted(values))]
                      for field, values in terms.items()}
            fields.update(row_id=[row_id], country=[record.country])
            writer.add_document(tantivy.Document.from_dict(fields, self.schema))
            for field, values in terms.items():
                self.frequency[field].update(values)
        writer.commit()
        writer.wait_merging_threads()
        self.index.reload()
        self.searcher = self.index.searcher()
        self.channels = BASE_CHANNELS + (ASCII_CHANNELS if transliteration else ())
        self.country_block = country_block
        self.doc_ids = {}

    def _field_query(self, field, values):
        # Rare terms bound query size; selection is deterministic. Their effect
        # on blocking recall is measured, never assumed lossless.
        maximum = 24 if "grams" in field else 12
        values = sorted(values, key=lambda x: (self.frequency[field][x], x))[:maximum]
        if not values:
            return None
        return tantivy.Query.boolean_query([
            (tantivy.Occur.Should,
             tantivy.Query.term_query(self.schema, field, value, "freq"))
            for value in values
        ], minimum_number_should_match=1)

    def candidates(self, record, top_k):
        terms = field_terms(record)
        queries = {field: self._field_query(field, terms[field])
                   for field in INDEX_FIELDS}
        combined_parts = []
        for field, boost in (("name_words", 2.0), ("address_words", 1.0)):
            if queries[field] is not None:
                combined_parts.append((tantivy.Occur.Should,
                                       tantivy.Query.boost_query(queries[field], boost)))
        queries["combined"] = tantivy.Query.boolean_query(
            combined_parts, minimum_number_should_match=1) if combined_parts else None
        results = {}
        for channel in self.channels:
            query = queries[channel]
            if query is None:
                continue
            if self.country_block:
                query = tantivy.Query.boolean_query([
                    (tantivy.Occur.Must, query),
                    (tantivy.Occur.Must, tantivy.Query.term_query(
                        self.schema, "country", record.country, "basic")),
                ])
            hits = self.searcher.search(query, limit=top_k, count=False).hits
            best_score = hits[0][0] if hits else 1.0
            for rank, (score, doc_address) in enumerate(hits, 1):
                key = (doc_address.segment_ord, doc_address.doc)
                if key not in self.doc_ids:
                    self.doc_ids[key] = self.searcher.doc(doc_address)["row_id"][0]
                row_id = self.doc_ids[key]
                results.setdefault(row_id, {})[channel] = (
                    float(score / max(best_score, 1e-12)), 1.0 / rank)
        return sorted(results.items(),
                      key=lambda item: (-sum(v[1] for v in item[1].values()), item[0]))
