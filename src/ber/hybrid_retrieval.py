"""BM25 retrieval with an optional rare-term gate and learned query expansion.

`gate_terms=0` keeps the default gate. A positive value restricts eligibility to
that many rare terms. Scoring terms are never truncated further.
"""

import tantivy

from .disk_index import DiskRetriever
from .expansion import ExpansionRetriever


class GatedRetriever(DiskRetriever):
    def __init__(self, *args, gate_terms=0, **kwargs):
        if gate_terms < 0:
            raise ValueError("gate_terms must be nonnegative")
        self.gate_terms = gate_terms
        super().__init__(*args, **kwargs)

    def _field_parts(self, field, values):
        full, gate = super()._field_parts(field, values)
        if not self.gate_terms or full is None:
            return full, gate
        present = sorted((self.frequency(field, value), value) for value in values
                         if self.frequency(field, value))
        selected = [value for _, value in present[:self.gate_terms]]
        gate = tantivy.Query.boolean_query([
            (tantivy.Occur.Should, tantivy.Query.term_query(self.schema, field, value, "freq"))
            for value in selected], minimum_number_should_match=1)
        return full, gate


def retrievers(index, lexicon, gate_terms=0, fold=None, store=None):
    if gate_terms == -1:
        if store is None:
            raise ValueError("WAND eligibility requires the record store")
        from .wand_retrieval import WandRetriever
        base = WandRetriever(index, store, fold)
    else:
        base = GatedRetriever(index, transliteration=True, fold=fold,
                              max_df_ratio=.01, frequency_mode="gate", gate_terms=gate_terms)
    extra = ExpansionRetriever(index, lexicon, fold=fold,
                               max_df_ratio=.01, frequency_mode="gate")
    extra.base = GatedRetriever(index, channels=["ascii_words"], fold=fold,
                                max_df_ratio=.01, frequency_mode="gate",
                                gate_terms=max(1, gate_terms) if gate_terms == -1 else gate_terms)
    return base, extra
