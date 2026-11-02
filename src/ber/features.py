"""Pair comparisons; no entity ID values or external observations are features."""

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

FEATURE_NAMES = [
    "name_ratio", "name_partial", "name_token_sort", "name_token_set",
    "name_wratio", "name_jaro_winkler", "name_jaccard", "name_overlap",
    "name_trigram_jaccard", "ascii_name_ratio", "bare_name_ratio", "name_exact",
    "address_ratio", "address_token_sort", "address_token_set",
    "address_jaccard", "address_overlap", "address_trigram_jaccard",
    "ascii_address_ratio", "address_exact", "number_jaccard",
    "both_have_numbers", "disjoint_numbers", "reference_address_missing",
    "target_address_missing", "name_length_ratio", "address_length_ratio",
    "target_is_s2", "reference_name_nonascii", "target_name_nonascii",
    "retrieval_channels", "best_relative_score", "best_reciprocal_rank",
    "name_word_retrieval", "name_gram_retrieval", "address_retrieval",
    "combined_retrieval", "ascii_retrieval",
]


def jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def overlap(left, right):
    denominator = min(len(left), len(right))
    return len(left & right) / denominator if denominator else 0.0


def length_ratio(left, right):
    return min(len(left), len(right)) / max(len(left), len(right), 1)


def safe_ratio(function, left, right):
    return function(left, right) / 100 if left and right else 0.0


def pair_features(left, right, retrieval):
    name, other_name = left.name, right.name
    address, other_address = left.address, right.address
    scores = [value[0] for value in retrieval.values()]
    ranks = [value[1] for value in retrieval.values()]
    relative = lambda field: retrieval.get(field, (0.0, 0.0))[0]
    result = [
        safe_ratio(fuzz.ratio, name, other_name),
        safe_ratio(fuzz.partial_ratio, name, other_name),
        safe_ratio(fuzz.token_sort_ratio, name, other_name),
        safe_ratio(fuzz.token_set_ratio, name, other_name),
        safe_ratio(fuzz.WRatio, name, other_name),
        JaroWinkler.normalized_similarity(name, other_name) if name and other_name else 0.,
        jaccard(left.name_tokens, right.name_tokens),
        overlap(left.name_tokens, right.name_tokens),
        jaccard(left.name_grams, right.name_grams),
        safe_ratio(fuzz.ratio, left.ascii_name, right.ascii_name),
        safe_ratio(fuzz.ratio, left.bare_name, right.bare_name),
        float(bool(name) and name == other_name),
        safe_ratio(fuzz.ratio, address, other_address),
        safe_ratio(fuzz.token_sort_ratio, address, other_address),
        safe_ratio(fuzz.token_set_ratio, address, other_address),
        jaccard(left.address_tokens, right.address_tokens),
        overlap(left.address_tokens, right.address_tokens),
        jaccard(left.address_grams, right.address_grams),
        safe_ratio(fuzz.ratio, left.ascii_address, right.ascii_address),
        float(bool(address) and address == other_address),
        jaccard(left.numbers, right.numbers),
        float(bool(left.numbers) and bool(right.numbers)),
        float(bool(left.numbers) and bool(right.numbers)
              and not (left.numbers & right.numbers)),
        float(not address), float(not other_address),
        length_ratio(name, other_name), length_ratio(address, other_address),
        float(right.entity_id.startswith("S2-")),
        float(not left.original_name.isascii()),
        float(not right.original_name.isascii()),
        float(len(retrieval)), max(scores, default=0.0), max(ranks, default=0.0),
        relative("name_words"), relative("name_grams"), relative("address_words"),
        relative("combined"), max(relative("ascii_words"), relative("ascii_grams")),
    ]
    assert len(result) == len(FEATURE_NAMES)
    return result
