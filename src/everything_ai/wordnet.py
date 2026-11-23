# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""WordNet for the lessons of domain taal about the meanings of words: the senses of a word, its synonyms, and the
relations between senses, in English with the Dutch words beside them.

English: the synsets of Open English WordNet 2025 (McCrae and others; CC BY 4.0 and the WordNet License), which the
course ships (data/wordnet-en.tsv.gz, made by data/tools/make_wordnet_en.py, without slurs and crude words). A synset
is a set of words with one meaning, such as car, auto and automobile; its name is the one NLTK gives it, such as
car.n.01: its first word, its part of speech (n noun, v verb, a adjective, s adjective in a cluster, r adverb) and the
sense number of that word. Dutch: the words of Open Dutch WordNet (Postma and others, 2016; CC BY-SA 4.0) on the same
synsets, by way of Princeton WordNet 3.0 (data/wordnet-nl.tsv, made by data/tools/make_wordnet_nl.py)."""
import functools
import gzip
import types
from typing import NamedTuple

from everything_ai import paths

EN = paths.ROOT / "data" / "wordnet-en.tsv.gz"
NL = paths.ROOT / "data" / "wordnet-nl.tsv"
POS = {"n": 0, "v": 1, "a": 2, "s": 2, "r": 3}   # the order in which senses lists the parts of speech: a and s together


class Synset(NamedTuple):
    name: str               # such as dog.n.01
    pos: str                # n, v, a, s or r
    words: tuple            # (word, sense number) pairs, the word with _ for a space, such as ("dog", 1)
    definition: str
    hypernyms: tuple        # names of the synsets one step more general: dog.n.01 has canine.n.02 and domestic_animal.n.01
    parts: tuple            # names of its parts: car.n.01 has accelerator.n.01, air_bag.n.01 and more
    members: tuple          # names of its members: forest.n.01 has underbrush.n.01 and tree.n.01
    substances: tuple       # names of what it is made of: water.n.01 has hydrogen.n.01 and oxygen.n.01
    antonyms: tuple         # (word, sense number, name) triples: hot in hot.a.01 has cold in cold.a.01
    dutch: tuple            # the Dutch words of Open Dutch WordNet on this synset, such as ("hond", "joekel")
    pwn30: str              # its offset in Princeton WordNet 3.0, such as 02084071-n, or "" where there is none


@functools.cache
def _synsets() -> types.MappingProxyType:
    dutch = {}
    with open(NL, encoding="utf-8") as f:
        next(f)
        for line in f:
            offset, word = line.rstrip("\n").split("\t")
            dutch.setdefault(offset, []).append(word)
    out = {}
    lines = gzip.decompress(EN.read_bytes()).decode("utf-8").split("\n")
    assert lines[0].split("\t") == ["name", "pwn30", "pos", "words", "definition", "hypernym", "part", "member",
                                    "substance", "antonym"], lines[0]
    for line in lines[1:]:
        if not line:
            continue
        name, pwn30, pos, words, definition, hyper, part, member, substance, antonym = line.split("\t")
        pairs = tuple((w, int(n)) for w, n in (x.rsplit(":", 1) for x in words.split()))
        anti = []
        for x in antonym.split():
            word_n, other = x.split(">")
            w, n = word_n.rsplit(":", 1)
            anti.append((w, int(n), other))
        out[name] = Synset(name, pos, pairs, definition, tuple(hyper.split()), tuple(part.split()),
                           tuple(member.split()), tuple(substance.split()), tuple(anti),
                           tuple(dutch.get(pwn30, ())) if pwn30 else (), pwn30)
    return types.MappingProxyType(out)


def synsets() -> types.MappingProxyType:
    """Every synset by its name, read-only: 107,425 of them."""
    return _synsets()


def synset(name: str) -> Synset:
    """The synset with this name, such as synset("dog.n.01")."""
    return _synsets()[name]


@functools.cache
def _senses() -> dict:
    out = {}
    for s in _synsets().values():
        for word, n in s.words:
            out.setdefault(word, []).append((POS[s.pos], n, s.name))
    return {w: tuple(name for _, _, name in sorted(v)) for w, v in out.items()}


def senses(word: str) -> list:
    """The names of the synsets of an English word, nouns first, then verbs, adjectives and adverbs, each part of
    speech from its most to its least common sense. A space in the word may also be written as _."""
    return list(_senses().get(word.replace(" ", "_"), ()))


@functools.cache
def _dutch_senses() -> dict:
    out = {}
    for s in _synsets().values():
        for word in s.dutch:
            out.setdefault(word, []).append(s.name)
    return {w: tuple(v) for w, v in out.items()}


def dutch_senses(word: str) -> list:
    """The names of the synsets that have this Dutch word, in the order of the file."""
    return list(_dutch_senses().get(word, ()))


def sense_pairs() -> list:
    """A (word, name) pair for every word of every synset, in the order of the file: 184,845 pairs."""
    return [(word, s.name) for s in _synsets().values() for word, _ in s.words]


def parents() -> dict:
    """A new dictionary from the name of every synset that has a hypernym to the name of its first hypernym."""
    return {s.name: s.hypernyms[0] for s in _synsets().values() if s.hypernyms}


def words(name: str) -> list:
    """The words of a synset, with spaces for _."""
    return [w.replace("_", " ") for w, _ in _synsets()[name].words]
