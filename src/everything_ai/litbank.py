# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Mentions and chains for the lesson of domain taal about coreference: the coreference layer of LitBank (Bamman,
Lewke and Mansoor, LREC 2020; CC-BY-4.0), about 2,000 words from each of 100 works of English fiction
(data/litbank-coref.json.gz, made by data/tools/make_litbank_coref.py).

A mention is a name, a description or a personal pronoun that refers to a person, facility, location, geo-political
entity, organization or vehicle; the mentions of one entity form a chain. The kind of a mention is what a pronoun has
to match in the lesson: a pronoun takes its kind from its word (it, its and itself a thing, the other personal pronouns
a person), a name or a description from its entity type (PER a person, the other types a thing)."""
import functools
import gzip
import json
from typing import NamedTuple

from everything_ai import paths

FILE = paths.ROOT / "data" / "litbank-coref.json.gz"
KINDS = ("person", "thing")
THING_PRONOUNS = frozenset({"it", "its", "itself"})


class Mention(NamedTuple):
    sentence: int       # the sentence of the work, from 0
    first: int          # its first token in that sentence, from 0
    last: int           # its last token (included)
    type: str           # PER, FAC, LOC, GPE, ORG or VEH
    category: str       # PROP (a name), NOM (a description) or PRON (a pronoun)
    chain: int          # the chain of the work it belongs to, numbered in the order of the first mention
    text: str           # the tokens of the mention, with spaces between, such as "the old house"
    kind: str           # person or thing (see above)


class Work(NamedTuple):
    gutenberg: int      # the number of the work at Project Gutenberg
    year: int
    author: str         # as LitBank writes it: "Austen, Jane"
    title: str
    sentences: tuple    # each a tuple of tokens
    shown: tuple        # per sentence, False when a lesson does not show it as an example (see the maker)
    mentions: tuple     # Mention, in the order of the text; a mention inside a longer one comes after it


def kind(text: str, category: str, type_: str) -> str:
    """The kind of a mention: person or thing."""
    if category == "PRON":
        return "thing" if text.lower() in THING_PRONOUNS else "person"
    return "person" if type_ == "PER" else "thing"


@functools.cache
def works() -> tuple:
    """The 100 works, from the oldest to the newest."""
    data = json.loads(gzip.decompress(FILE.read_bytes()).decode("utf-8"))
    out = []
    for d in data["docs"]:
        sentences = tuple(tuple(s.split(" ")) for s in d["sentences"])
        mentions = []
        for s, first, last, type_, category, chain in d["mentions"]:
            text = " ".join(sentences[s][first:last + 1])
            mentions.append(Mention(s, first, last, type_, category, chain, text, kind(text, category, type_)))
        out.append(Work(d["gutenberg"], d["year"], d["author"], d["title"], sentences,
                        tuple(bool(x) for x in d["shown"]), tuple(mentions)))
    return tuple(out)


def before(work: Work, i: int) -> list:
    """The indices of the mentions of a work that end before mention i starts: the ones it can refer back to."""
    m = work.mentions[i]
    return [j for j, x in enumerate(work.mentions) if (x.sentence, x.last) < (m.sentence, m.first)]
