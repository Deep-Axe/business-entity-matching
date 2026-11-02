"""Text views preserve Unicode and add optional deterministic romanization."""

import re
import unicodedata
from dataclasses import dataclass
from anyascii import anyascii

LEGAL = {"pvt": "private", "ltd": "limited", "corp": "corporation",
         "inc": "incorporated", "co": "company"}
ADDRESS = {"rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
           "apt": "apartment", "ste": "suite", "ln": "lane", "dr": "drive"}
SUFFIXES = frozenset((*LEGAL.values(), "llc", "llp", "plc"))


def normalize(value, address=False):
    value = unicodedata.normalize("NFKC", value).casefold().replace("&", " and ")
    value = "".join(c if c.isalnum() or unicodedata.category(c).startswith("M")
                    else " " for c in value)
    mapping = ADDRESS if address else LEGAL
    return " ".join(mapping.get(token, token) for token in value.split())


def grams(value):
    result = set()
    for token in value.split():
        padded = "^" + token + "$"
        result.update(padded[i:i + 3] for i in range(max(0, len(padded) - 2)))
    return frozenset(result)


@dataclass(slots=True)
class Record:
    entity_id: str
    country: str
    original_name: str
    original_address: str
    name: str
    address: str
    ascii_name: str
    ascii_address: str
    bare_name: str
    name_tokens: frozenset
    address_tokens: frozenset
    name_grams: frozenset
    ascii_grams: frozenset
    address_grams: frozenset
    numbers: frozenset

    @classmethod
    def from_row(cls, row):
        original_name, original_address = row["business_name"], row["business_address"]
        name, address = normalize(original_name), normalize(original_address, True)
        ascii_name = normalize(anyascii(original_name))
        ascii_address = normalize(anyascii(original_address), True)
        return cls(row["entity_id"], row["country"], original_name, original_address,
                   name, address, ascii_name, ascii_address,
                   " ".join(t for t in name.split() if t not in SUFFIXES),
                   frozenset(name.split()), frozenset(address.split()),
                   grams(name), grams(ascii_name), grams(address),
                   frozenset(re.findall(r"\d+", ascii_address)))
