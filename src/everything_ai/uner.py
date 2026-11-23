# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Names in English sentences, for the lessons of domain taal about sequence labeling, names and span F1: the
sentences of UNER_English-EWT of Universal NER v1 (Mayhew and others, NAACL 2024; CC-BY-SA-4.0), which the course
ships without the emails (data/uner-en-ewt.tsv.gz, made by data/tools/make_uner.py).

Each sentence is a tuple of (word, tag) pairs, as the sentences of the treebanks in the lesson about parts of speech.
A tag is O, or B- or I- followed by a type: PER for a person, LOC for a place, ORG for an organization (IOB2). A span
is a tuple (start, end, kind): the positions of its first word and of the word just after its last, as in a slice."""
import functools
import gzip
from typing import NamedTuple

from everything_ai import paths

FILE = paths.ROOT / "data" / "uner-en-ewt.tsv.gz"
KINDS = ("PER", "LOC", "ORG")
SPLITS = ("train", "dev", "test")


class Sentence(NamedTuple):
    sent_id: str        # as in Universal Dependencies, such as answers-20090717131757AAVh6rG_ans-0007
    split: str          # train, dev or test
    shown: bool         # False for a sentence that a lesson does not show as an example (see make_uner.py)
    pairs: tuple        # (word, tag) pairs, such as ("Miramar", "B-LOC")


@functools.cache
def sentences() -> tuple:
    """Every sentence of the file, in its order: 8,773 train, 1,478 dev and 1,471 test, 11,722 in all."""
    lines = gzip.decompress(FILE.read_bytes()).decode("utf-8").split("\n")
    assert lines[0].split("\t") == ["sent_id", "split", "shown", "words", "tags"], lines[0]
    out = []
    for line in lines[1:]:
        if not line:
            continue
        sent_id, split, shown, words, tags = line.split("\t")
        words, tags = words.split(" "), tags.split(" ")
        assert len(words) == len(tags), sent_id
        out.append(Sentence(sent_id, split, shown == "1", tuple(zip(words, tags))))
    return tuple(out)


@functools.cache
def _by_id() -> dict:
    return {s.sent_id: s for s in sentences()}


def sentence(sent_id: str) -> Sentence:
    """The sentence with this sent_id."""
    return _by_id()[sent_id]


def split(name: str) -> tuple:
    """The sentences of one part, train, dev or test, each as a tuple of (word, tag) pairs."""
    assert name in SPLITS, name
    return tuple(s.pairs for s in sentences() if s.split == name)


def spans(tags) -> list:
    """The spans of a list of tags, from left to right, as (start, end, kind) tuples. A B- starts a span; an I- makes
    the span before it one word longer when that span ends right before it and has the same type, and starts a span
    of its own otherwise, as the evaluation script of the CoNLL shared tasks reads it."""
    out = []
    for i, tag in enumerate(tags):
        if tag == "O":
            continue
        kind = tag[2:]
        if tag[:2] == "I-" and out and out[-1][1] == i and out[-1][2] == kind:
            out[-1] = (out[-1][0], i + 1, kind)
        else:
            out.append((i, i + 1, kind))
    return out


def tags(n: int, spans_) -> list:
    """The IOB2 tags of n words with these spans: B- on the first word of each span, I- on the words after it, O on
    the others."""
    out = ["O"] * n
    for start, end, kind in spans_:
        out[start] = "B-" + kind
        for i in range(start + 1, end):
            out[i] = "I-" + kind
    return out
